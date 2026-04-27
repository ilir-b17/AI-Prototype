"""OutcomeJudge — lightweight post-completion feedback scorer.

Runs after a Task is marked *completed* in the objective_backlog.
Compares ``acceptance_criteria`` against ``result_json`` using fast
pattern-matching (System 1) without requiring an LLM call.

Triviality bypass: simple read-only / observational tasks receive a
neutral score of 3 without deeper analysis.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.ledger_db import LedgerMemory

logger = logging.getLogger(__name__)

# Keywords that indicate a task is read-only / observational.
# These tasks produce output but have no meaningful pass/fail criteria,
# so we skip grading and assign the neutral score (3).
_READ_ONLY_KEYWORDS = frozenset(
    {
        "scan",
        "read",
        "list",
        "fetch",
        "check",
        "report",
        "summarize",
        "summarise",
        "analyze",
        "analyse",
        "view",
        "get",
        "query",
        "review",
        "inspect",
        "monitor",
        "collect",
        "describe",
        "log",
        "show",
    }
)

# Minimum word length for token overlap scoring (removes stop words)
_MIN_TOKEN_LEN = 3

_TRIVIAL_SCORE = 3  # Neutral for read-only tasks
_SCORE_FLOOR = 1
_SCORE_CEILING = 5


class OutcomeJudge:
    """Post-completion outcome scorer (System 1, pure Python)."""

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_trivial(task_row: Dict[str, Any]) -> bool:
        """Return True when the task is a simple read-only / observational task.

        Triviality is detected by matching the first meaningful word (verb) of
        the task title against ``_READ_ONLY_KEYWORDS``.  The bypass prevents
        false negatives for tasks that have no deterministic acceptance criteria.
        """
        title = str(task_row.get("title") or "").strip().lower()
        if not title:
            return True
        # Split on whitespace, take first non-empty word as the action verb
        words = title.split()
        for word in words:
            clean = re.sub(r"[^a-z]", "", word)
            if clean:
                return clean in _READ_ONLY_KEYWORDS
        return False

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Return a set of lower-case tokens (length ≥ _MIN_TOKEN_LEN)."""
        return {
            w.lower()
            for w in re.split(r"[\s,.;:!?()\[\]{}'\"]+", text)
            if len(w) >= _MIN_TOKEN_LEN
        }

    @classmethod
    def score_system1(cls, task_row: Dict[str, Any]) -> int:
        """Compute a 1–5 outcome score by matching acceptance criteria against result.

        Algorithm (pure Python, no LLM):
        1. If no acceptance criteria → neutral (3).
        2. If no result → failure (1).
        3. Count how many criteria tokens appear in result text.
        4. Map coverage ratio to 1–5.

        Score mapping:
        - ≥ 90% coverage → 5 (excellent)
        - ≥ 70% coverage → 4 (good)
        - ≥ 40% coverage → 3 (partial)
        - ≥ 20% coverage → 2 (poor)
        - < 20% coverage → 1 (failed)
        """
        criteria = str(task_row.get("acceptance_criteria") or "").strip()
        if not criteria:
            return _TRIVIAL_SCORE  # No criteria defined → neutral

        raw_result = task_row.get("result_json")
        if not raw_result:
            return _SCORE_FLOOR  # No result at all → failure

        # Decode result_json into a string for token matching
        if isinstance(raw_result, dict):
            result_text = json.dumps(raw_result)
        else:
            try:
                parsed = json.loads(str(raw_result))
                result_text = json.dumps(parsed)
            except (TypeError, ValueError):
                result_text = str(raw_result)

        criteria_tokens = cls._tokenize(criteria)
        if not criteria_tokens:
            return _TRIVIAL_SCORE

        result_lower = result_text.lower()
        matched = sum(1 for token in criteria_tokens if token in result_lower)
        ratio = matched / len(criteria_tokens)

        if ratio >= 0.9:
            return 5
        if ratio >= 0.7:
            return 4
        if ratio >= 0.4:
            return 3
        if ratio >= 0.2:
            return 2
        return 1

    @classmethod
    async def record_outcome(
        cls,
        ledger: "LedgerMemory",
        task_id: int,
        *,
        task_row: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Judge a completed task and persist the outcome record.

        Fetches the task row from the ledger if not provided, applies
        triviality bypass or System 1 scoring, and writes the result.

        Args:
            ledger: The LedgerMemory instance to read/write.
            task_id: Objective backlog Task id.
            task_row: Optional pre-fetched task row (skips DB read if supplied).
        """
        try:
            row = task_row
            if row is None:
                row = await ledger.get_task_row(task_id)

            if row is None:
                logger.warning("OutcomeJudge: Task #%s not found — skipping.", task_id)
                return

            if row.get("outcome_score") is not None:
                logger.debug("OutcomeJudge: Task #%s already has outcome — skipping.", task_id)
                return

            if cls.is_trivial(row):
                score = _TRIVIAL_SCORE
                notes = "triviality_bypass: read-only/observational task"
            else:
                score = cls.score_system1(row)
                notes = f"system1_pattern_match: coverage_based score {score}/5"

            await ledger.write_task_outcome(
                task_id,
                score=score,
                outcome_notes=notes,
            )
            logger.info(
                "OutcomeJudge: Task #%s scored %s/5 (%s)",
                task_id,
                score,
                "trivial" if cls.is_trivial(row) else "system1",
            )
        except Exception as exc:
            # Best-effort; never raise from a fire-and-forget context
            logger.warning("OutcomeJudge: failed for Task #%s: %s", task_id, exc)
