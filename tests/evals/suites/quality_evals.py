"""
Response quality eval suite - Tier 3, requires Ollama.

Runs full process_message() calls and grades responses using
the local model as rubric judge.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple

from tests.evals.fixtures.quality_cases import QUALITY_CASES
from tests.evals.framework.base import EvalCase, EvalSuite, EvalTier
from tests.evals.framework.grader import grade_response

logger = logging.getLogger(__name__)

EVAL_USER_ID = "eval_system"


class ResponseQualityEvalSuite(EvalSuite):
    """End-to-end response quality evaluation using local LLM grading."""

    name = "response_quality"
    tier = EvalTier.LLM_GRADED

    def __init__(self, orchestrator, local_model: str = "gemma4:e4b"):
        """
        orchestrator: a fully initialized Orchestrator instance.
        local_model: local Ollama model to use for grading.
        """
        self._orchestrator = orchestrator
        self._local_model = local_model
        self._loop = getattr(orchestrator, "_eval_loop", None)

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c["id"],
                description=c["prompt"][:80],
                tier=EvalTier.LLM_GRADED,
                tags=c["tags"],
                input={
                    "prompt": c["prompt"],
                    "rubric": c["rubric"],
                    "expected_tools": c["expected_tools"],
                    "must_not_contain": c["must_not_contain"],
                },
                expected={"passed": True},
            )
            for c in QUALITY_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        """Synchronous wrapper that runs the async eval coroutine."""
        if self._loop is not None and not self._loop.is_closed():
            return self._loop.run_until_complete(self._run_case_async(case))
        return asyncio.run(self._run_case_async(case))

    async def _run_case_async(
        self,
        case: EvalCase,
    ) -> Tuple[Dict[str, Any], List[str]]:
        prompt = case.input["prompt"]
        rubric = case.input["rubric"]
        must_not_contain = case.input.get("must_not_contain", [])

        # Get AIDEN's actual response.
        try:
            response = await asyncio.wait_for(
                self._orchestrator.process_message(prompt, EVAL_USER_ID),
                timeout=120.0,
            )
        except Exception as exc:
            return {"error": str(exc)}, [f"process_message raised: {exc}"]

        response_text = str(response or "")
        failures: List[str] = []

        for forbidden in must_not_contain:
            if forbidden in response_text:
                failures.append(
                    f"Response contains forbidden string: {forbidden!r}"
                )

        grade = await grade_response(
            user_prompt=prompt,
            ai_response=response_text,
            rubric=rubric,
            local_model=self._local_model,
            timeout_seconds=60.0,
        )

        if grade.get("grader_error"):
            failures.append(f"Grader error: {grade['grader_error']}")
        elif not grade.get("passed"):
            low_scores = {
                criterion: score
                for criterion, score in grade.get("scores", {}).items()
                if isinstance(score, (int, float)) and score < 3
            }
            if low_scores:
                failures.append(
                    f"Low rubric scores: {low_scores}. "
                    f"Reasoning: {grade.get('reasoning', '')[:200]}"
                )
            else:
                failures.append(
                    "Grader marked failed. "
                    f"Reasoning: {grade.get('reasoning', '')[:200]}"
                )

        actual = {
            "response_preview": response_text[:300],
            "response_length": len(response_text),
            "grade": grade,
            "expected_tools": list(case.input.get("expected_tools", [])),
        }
        return actual, failures
