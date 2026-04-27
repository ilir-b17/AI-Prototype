from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENERGY_JUDGE_SYSTEM_PROMPT = (
    "You are EnergyJudge, a local pre-execution evaluator. "
    "Return ONLY strict JSON with exactly two keys: estimated_effort and expected_value. "
    "Both values MUST be integers from 1 to 10. "
    "Do not include markdown, prose, explanations, or additional fields."
)


@dataclass
class EnergyEvaluation:
    estimated_effort: int
    expected_value: int
    used_fallback: bool = False
    fallback_reason: str = ""


class EnergyJudge:
    """Local (System 1) effort/value scorer for predictive energy gating."""

    def __init__(self, *, fallback_effort: int = 5, fallback_value: int = 5) -> None:
        self.fallback_effort = int(fallback_effort)
        self.fallback_value = int(fallback_value)

    @staticmethod
    def _extract_json_text(raw_response: str) -> str:
        text = str(raw_response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        return text

    @staticmethod
    def _build_parent_summary(parent: Optional[Dict[str, Any]], *, label: str) -> str:
        if not parent:
            return f"{label}: <none provided>"
        return (
            f"{label}: {str(parent.get('title') or '').strip()}\n"
            f"{label} acceptance_criteria: {str(parent.get('acceptance_criteria') or '').strip()}\n"
            f"{label} status: {str(parent.get('status') or '').strip()}"
        )

    @staticmethod
    def _parse_bound_score(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 1 or parsed > 10:
            return None
        return parsed

    def _fallback(self, reason: str) -> EnergyEvaluation:
        logger.warning("EnergyJudge fallback used: %s", reason)
        return EnergyEvaluation(
            estimated_effort=self.fallback_effort,
            expected_value=self.fallback_value,
            used_fallback=True,
            fallback_reason=reason,
        )

    def build_evaluation_payload(
        self,
        *,
        task: Dict[str, Any],
        story: Optional[Dict[str, Any]],
        epic: Optional[Dict[str, Any]],
        additional_context: str = "",
    ) -> Dict[str, Any]:
        task_title = str(task.get("title") or "").strip()
        task_acceptance = str(task.get("acceptance_criteria") or "").strip()
        task_energy = int(task.get("estimated_energy") or 0)

        return {
            "task": "score_energy_roi",
            "instructions": {
                "output_schema": {
                    "estimated_effort": "integer 1-10",
                    "expected_value": "integer 1-10",
                },
                "hard_rules": [
                    "Return JSON only.",
                    "Do not return markdown.",
                    "Do not include extra keys.",
                    "estimated_effort must be an integer from 1 to 10.",
                    "expected_value must be an integer from 1 to 10.",
                    "Score expected_value by how directly and significantly this task contributes to completing the parent Story and parent Epic.",
                ],
            },
            "task_context": {
                "title": task_title,
                "acceptance_criteria": task_acceptance,
                "estimated_energy": task_energy,
                "status": str(task.get("status") or ""),
                "depends_on_ids": list(task.get("depends_on_ids") or []),
            },
            "parent_story": story or {},
            "parent_epic": epic or {},
            "additional_context": str(additional_context or "").strip(),
        }

    def build_system1_messages(
        self,
        *,
        task: Dict[str, Any],
        story: Optional[Dict[str, Any]],
        epic: Optional[Dict[str, Any]],
        additional_context: str = "",
    ) -> List[Dict[str, str]]:
        payload = self.build_evaluation_payload(
            task=task,
            story=story,
            epic=epic,
            additional_context=additional_context,
        )

        compact_context = (
            f"Task: {str(task.get('title') or '').strip()}\n"
            f"Task acceptance_criteria: {str(task.get('acceptance_criteria') or '').strip()}\n"
            f"Task estimated_energy: {int(task.get('estimated_energy') or 0)}\n"
            f"{self._build_parent_summary(story, label='Parent Story')}\n"
            f"{self._build_parent_summary(epic, label='Parent Epic')}\n"
            "Expected_value must reflect direct and significant contribution to the parent Story and Epic."
        )

        return [
            {"role": "system", "content": _ENERGY_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    compact_context
                    + "\n\n"
                    + json.dumps(payload, indent=2)
                ),
            },
        ]

    def parse_system1_response(self, response_content: Any) -> EnergyEvaluation:
        if isinstance(response_content, dict):
            decoded = response_content
        else:
            text = self._extract_json_text(str(response_content or ""))
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError):
                return self._fallback("invalid_json")

        if not isinstance(decoded, dict):
            return self._fallback("non_object_response")

        estimated_effort = self._parse_bound_score(decoded.get("estimated_effort"))
        expected_value = self._parse_bound_score(decoded.get("expected_value"))
        if estimated_effort is None or expected_value is None:
            return self._fallback("missing_or_out_of_bounds_scores")

        return EnergyEvaluation(
            estimated_effort=estimated_effort,
            expected_value=expected_value,
            used_fallback=False,
            fallback_reason="",
        )

    async def evaluate_with_system1(
        self,
        *,
        task: Dict[str, Any],
        story: Optional[Dict[str, Any]],
        epic: Optional[Dict[str, Any]],
        route_to_system_1: Callable[[List[Dict[str, str]]], Awaitable[Any]],
        additional_context: str = "",
    ) -> EnergyEvaluation:
        messages = self.build_system1_messages(
            task=task,
            story=story,
            epic=epic,
            additional_context=additional_context,
        )

        try:
            route_result = await route_to_system_1(messages)
        except Exception as e:
            return self._fallback(f"system1_exception:{e}")

        if hasattr(route_result, "status"):
            if getattr(route_result, "status", "") != "ok":
                return self._fallback(f"system1_status:{getattr(route_result, 'status', 'unknown')}")
            response_content = getattr(route_result, "content", "")
        else:
            response_content = route_result

        return self.parse_system1_response(response_content)

    @staticmethod
    def blend_with_historical_scores(
        evaluation: "EnergyEvaluation",
        historical_scores: List[int],
    ) -> "EnergyEvaluation":
        """Blend the LLM-derived evaluation with historical outcome scores.

        Blending rule:
        - N ≥ 5 outcomes: 70% historical, 30% heuristic (LLM).
        - N < 5 outcomes: 100% heuristic (unchanged).

        Historical outcome_scores are on a 1–5 scale; they are scaled to
        the 1–10 expected_value range before blending.

        Args:
            evaluation: The base EnergyEvaluation from the LLM (or fallback).
            historical_scores: List of 1–5 outcome scores for similar tasks.

        Returns:
            A new EnergyEvaluation with a potentially adjusted expected_value.
            estimated_effort is never modified by historical data.
        """
        if len(historical_scores) < 5:
            return evaluation

        # Scale 1-5 → 1-10 (preserve range boundaries)
        scaled = [max(1, min(10, round((s - 1) / 4 * 9 + 1))) for s in historical_scores]
        historical_avg = sum(scaled) / len(scaled)

        heuristic_value = evaluation.expected_value
        blended = round(0.7 * historical_avg + 0.3 * heuristic_value)
        blended_clamped = max(1, min(10, blended))

        if blended_clamped == heuristic_value:
            return evaluation

        return EnergyEvaluation(
            estimated_effort=evaluation.estimated_effort,
            expected_value=blended_clamped,
            used_fallback=evaluation.used_fallback,
            fallback_reason=evaluation.fallback_reason,
        )
