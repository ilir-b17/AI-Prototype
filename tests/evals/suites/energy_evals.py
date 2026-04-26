"""
Energy system eval suite - tests EnergyROIEngine.evaluate decisions.
Tier 1: pure deterministic, no IO.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tests.evals.fixtures.energy_cases import ENERGY_CASES
from tests.evals.framework.base import EvalCase, EvalSuite, EvalTier


class EnergyROIEvalSuite(EvalSuite):
    """Evaluates EnergyROIEngine.evaluate() gate decisions."""

    name = "energy_roi"
    tier = EvalTier.DETERMINISTIC

    def __init__(self):
        from src.core.energy_roi_engine import EnergyPolicy, EnergyROIEngine

        # Use default policy (reads from env, which matches test expectations)
        self._engine = EnergyROIEngine(EnergyPolicy())

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c["id"],
                description=c["description"],
                tier=EvalTier.DETERMINISTIC,
                tags=c["tags"],
                input={
                    "effort": c["effort"],
                    "value": c["value"],
                    "available": c["available"],
                    "defer_count": c["defer_count"],
                },
                expected={
                    "should_execute": c["expected_execute"],
                    "reason_contains": c["expected_reason_contains"],
                },
            )
            for c in ENERGY_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        decision = self._engine.evaluate(
            estimated_effort=case.input["effort"],
            expected_value=case.input["value"],
            available_energy=case.input["available"],
            defer_count=case.input["defer_count"],
        )

        failures = []
        if decision.should_execute != case.expected["should_execute"]:
            failures.append(
                f"should_execute: expected {case.expected['should_execute']}, "
                f"got {decision.should_execute} (reason={decision.reason!r}, "
                f"roi={decision.effective_roi:.3f})"
            )

        reason_fragment = case.expected.get("reason_contains", "")
        if reason_fragment and reason_fragment.lower() not in decision.reason.lower():
            failures.append(
                f"reason: expected to contain {reason_fragment!r}, "
                f"got {decision.reason!r}"
            )

        actual = {
            "should_execute": decision.should_execute,
            "reason": decision.reason,
            "base_roi": round(decision.base_roi, 4),
            "effective_roi": round(decision.effective_roi, 4),
            "predicted_cost": decision.predicted_cost,
            "reserve_after": decision.reserve_after_execution,
        }
        return actual, failures


class EnergyJudgeParsingEvalSuite(EvalSuite):
    """Evaluates EnergyJudge.parse_system1_response JSON parsing robustness."""

    name = "energy_judge_parsing"
    tier = EvalTier.DETERMINISTIC

    _PARSE_CASES = [
        # (id, raw_response, expected_effort, expected_value, expect_fallback)
        ("ej_001", '{"estimated_effort": 4, "expected_value": 7}', 4, 7, False),
        ("ej_002", '```json\n{"estimated_effort": 6, "expected_value": 8}\n```', 6, 8, False),
        ("ej_003", '{"estimated_effort": 0, "expected_value": 5}', 5, 5, True),
        ("ej_004", '{"estimated_effort": 11, "expected_value": 5}', 5, 5, True),
        ("ej_005", 'not json at all', 5, 5, True),
        ("ej_006", '{"estimated_effort": 3}', 5, 5, True),
        ("ej_007", '{"estimated_effort": 3, "expected_value": 8, "extra": "ignored"}', 3, 8, False),
        ("ej_008", '{"estimated_effort": 1, "expected_value": 10}', 1, 10, False),
        ("ej_009", '{"estimated_effort": 10, "expected_value": 1}', 10, 1, False),
        ("ej_010", '', 5, 5, True),
    ]

    def __init__(self):
        from src.core.energy_judge import EnergyJudge

        self._judge = EnergyJudge(fallback_effort=5, fallback_value=5)

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c[0],
                description=f"Parse: {c[1][:40]!r}",
                tier=EvalTier.DETERMINISTIC,
                tags=["energy_judge", "parsing"],
                input={"raw_response": c[1]},
                expected={
                    "effort": c[2],
                    "value": c[3],
                    "used_fallback": c[4],
                },
            )
            for c in self._PARSE_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        result = self._judge.parse_system1_response(case.input["raw_response"])
        failures = []

        effort_fail = self._check_exact(
            "estimated_effort", result.estimated_effort, case.expected["effort"]
        )
        if effort_fail:
            failures.append(effort_fail)

        value_fail = self._check_exact(
            "expected_value", result.expected_value, case.expected["value"]
        )
        if value_fail:
            failures.append(value_fail)

        fallback_fail = self._check_exact(
            "used_fallback", result.used_fallback, case.expected["used_fallback"]
        )
        if fallback_fail:
            failures.append(fallback_fail)

        return {
            "effort": result.estimated_effort,
            "value": result.expected_value,
            "used_fallback": result.used_fallback,
            "fallback_reason": result.fallback_reason,
        }, failures
