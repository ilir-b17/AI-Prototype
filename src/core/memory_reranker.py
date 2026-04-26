"""
Two-stage memory retrieval reranker for AIDEN.

Stage 1 (recall):  VectorMemory cosine similarity — returns
                   n_results * RECALL_MULTIPLIER candidates.
Stage 2 (rerank):  System 1 batch scoring — scores each candidate's
                   relevance to the query in one call, returns top N
                   sorted by blended LLM+cosine score.

Design principles:
  - Callable injection: route_to_system_1 is passed at rerank() call
    time so the class is testable without a live Ollama instance.
  - Fail-safe: any failure in Stage 2 falls back to cosine ordering.
  - Compact prompt: documents are truncated to RERANK_DOCUMENT_CHARS
    for scoring; full documents are returned in results.
  - Single LLM call: all candidates scored in one batch, not N calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankConfig:
    """Configuration for the MemoryReranker, read from environment."""
    enabled: bool = True
    recall_multiplier: int = 4
    max_candidates: int = 16
    document_chars: int = 300
    alpha: float = 0.70          # LLM score weight in blended score
    timeout_seconds: float = 20.0
    min_candidates: int = 3      # don't rerank if fewer candidates


def load_rerank_config() -> RerankConfig:
    """Load RerankConfig from environment variables."""

    def _bool(name: str, default: bool) -> bool:
        raw = os.getenv(name, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    def _int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        try:
            return max(1, int(raw)) if raw else default
        except ValueError:
            logger.warning("Invalid %s=%r; using %d", name, raw, default)
            return default

    def _float(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            logger.warning("Invalid %s=%r; using %.2f", name, raw, default)
            return default

    return RerankConfig(
        enabled=_bool("ENABLE_MEMORY_RERANKING", True),
        recall_multiplier=_int("MEMORY_RECALL_MULTIPLIER", 4),
        max_candidates=_int("MEMORY_RERANK_MAX_CANDIDATES", 16),
        document_chars=_int("MEMORY_RERANK_DOCUMENT_CHARS", 300),
        alpha=max(0.0, min(1.0, _float("MEMORY_RERANK_ALPHA", 0.70))),
        timeout_seconds=max(5.0, _float("MEMORY_RERANK_TIMEOUT_SECONDS", 20.0)),
        min_candidates=_int("MEMORY_RERANK_MIN_CANDIDATES", 3),
    )


_RERANKER_SYSTEM_PROMPT = (
    "You are a memory relevance scorer. "
    "Score each memory's relevance to the query. "
    "Return ONLY strict JSON matching this schema: "
    '{"scores": {"0": <int>, "1": <int>, ...}} '
    "Scores are integers 0-10. 10=perfectly relevant. 0=completely irrelevant. "
    "Include every id. No other text. No markdown."
)


class MemoryReranker:
    """Two-stage memory retrieval reranker.

    Usage:
        reranker = MemoryReranker(config)
        reranked = await reranker.rerank(
            query="energy budget decisions",
            candidates=[...],   # from VectorMemory.query_memory
            n_results=3,
            route_to_system_1=some_async_callable,
        )
    """

    def __init__(self, config: Optional[RerankConfig] = None) -> None:
        self._config = config or load_rerank_config()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def config(self) -> RerankConfig:
        return self._config

    # ── Public API ────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        n_results: int,
        route_to_system_1: Callable[..., Awaitable[Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank candidates by LLM+cosine blended score.

        Args:
            query:             The original user/archival query.
            candidates:        Raw VectorMemory results dicts containing
                               'id', 'document', 'metadata', 'distance'.
            n_results:         Final number of results to return.
            route_to_system_1: Async callable matching the orchestrator's
                               _route_to_system_1 signature. May be a mock
                               in tests.

        Returns:
            List of result dicts sorted by blended score, length <= n_results.
            Each dict has extra keys: rerank_score, cosine_score, combined_score.
            Falls back to cosine-ordered results on any failure.
        """
        if not candidates:
            return []

        capped_n = min(n_results, len(candidates))

        # Skip reranking if too few candidates to be worth it
        if len(candidates) <= self._config.min_candidates:
            logger.debug(
                "Reranking skipped: %d candidates <= min_candidates %d",
                len(candidates),
                self._config.min_candidates,
            )
            return self._cosine_order(candidates)[:capped_n]

        start = time.monotonic()
        try:
            scores = await self._score_candidates(query, candidates, route_to_system_1)
        except Exception as exc:
            logger.warning(
                "Reranker scoring failed (falling back to cosine): %s", exc
            )
            return self._cosine_order(candidates)[:capped_n]

        blended = self._blend_scores(candidates, scores)
        blended.sort(key=lambda r: r["combined_score"], reverse=True)

        elapsed = time.monotonic() - start
        logger.info(
            "Memory reranker: %d → %d results in %.2fs "
            "(top combined_score=%.3f)",
            len(candidates),
            capped_n,
            elapsed,
            blended[0]["combined_score"] if blended else 0.0,
        )
        return blended[:capped_n]

    def broad_n_results(self, n_results: int) -> int:
        """Return the broad recall count for Stage 1.

        Caps at MEMORY_RERANK_MAX_CANDIDATES to prevent huge ChromaDB
        queries and oversized scoring prompts.
        """
        broad = n_results * self._config.recall_multiplier
        return min(broad, self._config.max_candidates)

    # ── Internal ──────────────────────────────────────────────────────

    def _truncate_doc(self, document: str) -> str:
        """Truncate document to RERANK_DOCUMENT_CHARS for scoring."""
        text = str(document or "").strip()
        limit = self._config.document_chars
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "…"

    def _build_scoring_messages(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Build the System 1 messages for batch scoring."""
        memory_entries = {
            str(idx): self._truncate_doc(c.get("document", ""))
            for idx, c in enumerate(candidates)
        }
        user_payload = json.dumps(
            {"query": str(query or "")[:300], "memories": memory_entries},
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": _RERANKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]

    @staticmethod
    def _extract_json_text(raw: str) -> str:
        text = str(raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        return text

    def _parse_scores(
        self,
        raw_content: str,
        n_candidates: int,
    ) -> Optional[Dict[int, float]]:
        """Parse LLM score response. Returns {index: score} or None on failure."""
        text = self._extract_json_text(raw_content)
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.debug(
                "Reranker: could not parse JSON scores from: %r", text[:200]
            )
            return None

        # Accept both {"scores": {...}} and {"0": ...} flat formats
        if isinstance(payload, dict) and "scores" in payload:
            raw_scores = payload["scores"]
        elif isinstance(payload, dict):
            raw_scores = payload
        else:
            return None

        if not isinstance(raw_scores, dict):
            return None

        result: Dict[int, float] = {}
        for key, value in raw_scores.items():
            try:
                idx = int(str(key).strip())
                score = float(value)
                if 0 <= idx < n_candidates and 0.0 <= score <= 10.0:
                    result[idx] = score
            except (TypeError, ValueError):
                continue

        # Accept if at least half the candidates were scored
        if len(result) >= max(1, n_candidates // 2):
            return result
        return None

    async def _score_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        route_to_system_1: Callable[..., Awaitable[Any]],
    ) -> Dict[int, float]:
        """Call System 1 to score candidates. Returns {index: score}.

        Falls back to empty dict (triggering cosine fallback) on any error.
        """
        messages = self._build_scoring_messages(query, candidates)

        try:
            import asyncio
            route_result = await asyncio.wait_for(
                route_to_system_1(
                    messages,
                    allowed_tools=[],
                    max_output_tokens=150,
                ),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Reranker: System 1 timed out after %.1fs",
                self._config.timeout_seconds,
            )
            return {}
        except Exception as exc:
            logger.warning("Reranker: System 1 call failed: %s", exc)
            return {}

        # Handle RouterResult or plain string
        if hasattr(route_result, "status"):
            if getattr(route_result, "status", "") != "ok":
                logger.debug(
                    "Reranker: System 1 returned non-ok status: %s",
                    getattr(route_result, "status", "unknown"),
                )
                return {}
            content = str(getattr(route_result, "content", "") or "")
        else:
            content = str(route_result or "")

        parsed = self._parse_scores(content, len(candidates))
        if parsed is None:
            logger.debug(
                "Reranker: score parse failed; using cosine fallback. "
                "Raw content: %r",
                content[:300],
            )
            return {}
        return parsed

    def _cosine_score(self, distance: float) -> float:
        """Convert cosine distance to 0-10 score (lower distance = higher score)."""
        # ChromaDB cosine distances are in [0, 2].
        # distance=0 means identical, distance=2 means opposite.
        # Map to [0, 10] where 10 = identical.
        raw_distance = 1.0 if distance is None else float(distance)
        clamped = max(0.0, min(2.0, raw_distance))
        return (1.0 - clamped / 2.0) * 10.0

    def _combine(self, cosine_score_10: float, llm_score: float) -> float:
        """Blend cosine and LLM scores using alpha weighting."""
        alpha = self._config.alpha
        return alpha * llm_score + (1.0 - alpha) * cosine_score_10

    def _blend_scores(
        self,
        candidates: List[Dict[str, Any]],
        llm_scores: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        """Add rerank_score, cosine_score, combined_score to each candidate."""
        enriched: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            raw_distance = candidate.get("distance")
            distance = float(raw_distance if raw_distance is not None else 1.0)
            cosine_score = self._cosine_score(distance)
            llm_score = float(llm_scores.get(idx, cosine_score))
            combined = self._combine(cosine_score, llm_score)
            enriched_candidate = dict(candidate)
            enriched_candidate["rerank_score"] = round(llm_score, 3)
            enriched_candidate["cosine_score"] = round(cosine_score, 3)
            enriched_candidate["combined_score"] = round(combined, 3)
            enriched_candidate["reranked"] = bool(llm_scores)
            enriched.append(enriched_candidate)
        return enriched

    def _cosine_order(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return candidates sorted by cosine distance (cosine-only fallback)."""
        ordered = sorted(
            candidates,
            key=lambda c: float(c.get("distance") if c.get("distance") is not None else 1.0),
        )
        result: List[Dict[str, Any]] = []
        for c in ordered:
            enriched = dict(c)
            raw_distance = c.get("distance")
            distance = float(raw_distance if raw_distance is not None else 1.0)
            cosine_score = self._cosine_score(distance)
            enriched["rerank_score"] = cosine_score   # no LLM score
            enriched["cosine_score"] = round(cosine_score, 3)
            enriched["combined_score"] = round(cosine_score, 3)
            enriched["reranked"] = False
            result.append(enriched)
        return result
