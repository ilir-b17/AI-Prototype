"""
Moral audit eval suite.
Tier 1: payload validation and triviality bypass (deterministic).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tests.evals.fixtures.moral_cases import ALL_MORAL_CASES, TRIVIALITY_CASES
from tests.evals.framework.base import EvalCase, EvalSuite, EvalTier


class MoralPayloadEvalSuite(EvalSuite):
    """Evaluates parse_moral_decision_response payload parsing and validation."""

    name = "moral_payload"
    tier = EvalTier.DETERMINISTIC

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c["id"],
                description=c["description"],
                tier=EvalTier.DETERMINISTIC,
                tags=c["tags"],
                input={"payload": c["payload"]},
                expected={
                    "valid": c["expect_valid"],
                    "approved": c.get("expect_approved"),
                    "decision_mode": c.get("expect_decision_mode"),
                },
            )
            for c in ALL_MORAL_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        from src.core.moral_ledger import parse_moral_decision_response

        decision = parse_moral_decision_response(case.input["payload"])
        failures: List[str] = []

        is_valid = decision.validation_error == ""
        expected_valid = case.expected["valid"]
        if is_valid != expected_valid:
            failures.append(
                f"validity: expected valid={expected_valid}, "
                f"got valid={is_valid} "
                f"(validation_error={decision.validation_error!r})"
            )

        expected_approved = case.expected.get("approved")
        if expected_approved is not None and is_valid:
            if decision.is_approved != expected_approved:
                failures.append(
                    f"is_approved: expected {expected_approved}, "
                    f"got {decision.is_approved}"
                )

        expected_mode = case.expected.get("decision_mode")
        if expected_mode is not None and is_valid:
            if str(decision.decision_mode) != str(expected_mode):
                failures.append(
                    f"decision_mode: expected {expected_mode!r}, "
                    f"got {decision.decision_mode!r}"
                )

        return {
            "is_valid": is_valid,
            "is_approved": decision.is_approved,
            "validation_error": decision.validation_error,
            "decision_mode": decision.decision_mode,
        }, failures


class TrivialityBypassEvalSuite(EvalSuite):
    """Evaluates _try_triviality_bypass_decision behavior."""

    name = "triviality_bypass"
    tier = EvalTier.HEURISTIC

    def __init__(self, orchestrator_stub):
        self._orch = orchestrator_stub

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c["id"],
                description=c["description"],
                tier=EvalTier.HEURISTIC,
                tags=c["tags"],
                input={
                    "user_input": c["user_input"],
                    "output": c["output"],
                    "current_plan": c["current_plan"],
                    "iteration_count": c["iteration_count"],
                    "critic_instructions": c["critic_instructions"],
                },
                expected={"should_bypass": c["expected_bypass"]},
            )
            for c in TRIVIALITY_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        state = {
            "user_input": case.input["user_input"],
            "current_plan": case.input["current_plan"],
            "iteration_count": case.input["iteration_count"],
            "critic_instructions": case.input["critic_instructions"],
        }
        output = case.input["output"]

        decision = self._orch._try_triviality_bypass_decision(state, output)
        actual_bypass = decision is not None
        expected_bypass = case.expected["should_bypass"]

        failures: List[str] = []
        if actual_bypass != expected_bypass:
            bypass_reason = decision.bypass_reason if decision else "no bypass"
            failures.append(
                f"bypass: expected {expected_bypass}, got {actual_bypass} "
                f"(reason={bypass_reason!r})"
            )

        return {
            "bypassed": actual_bypass,
            "bypass_reason": decision.bypass_reason if decision else None,
        }, failures


class MoralSeverityEvalSuite(EvalSuite):
    """Evaluates _classify_moral_rejection severe/moderate/standard outcomes."""

    name = "moral_severity"
    tier = EvalTier.DETERMINISTIC

    _SEVERITY_CASES = [
        (
            "sev_001",
            {
                "security_conflict": True,
                "violated_tiers": ("tier_1",),
                "remediation_constraints": (),
                "reasoning": "Security conflict detected",
            },
            "severe",
        ),
        (
            "sev_002",
            {
                "security_conflict": False,
                "violated_tiers": ("tier_1",),
                "remediation_constraints": (),
                "reasoning": "Tier 1 harm reduction violated",
            },
            "severe",
        ),
        (
            "sev_003",
            {
                "security_conflict": False,
                "violated_tiers": ("tier_2",),
                "remediation_constraints": ("Add disclaimer",),
                "reasoning": "Overconfident claim",
            },
            "moderate",
        ),
        (
            "sev_004",
            {
                "security_conflict": False,
                "violated_tiers": ("tier_3",),
                "remediation_constraints": ("Clean output format",),
                "reasoning": "Output has markdown headers",
            },
            "moderate",
        ),
        (
            "sev_005",
            {
                "security_conflict": False,
                "violated_tiers": (),
                "remediation_constraints": (),
                "reasoning": "Minor quality issue",
            },
            "standard",
        ),
        (
            "sev_006",
            {
                "security_conflict": False,
                "violated_tiers": (),
                "remediation_constraints": ("Improve phrasing",),
                "reasoning": "Phrasing could be clearer",
            },
            "moderate",
        ),
    ]

    def __init__(self, orchestrator_stub):
        self._orch = orchestrator_stub

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=case_id,
                description=f"severity: {expected_severity}",
                tier=EvalTier.DETERMINISTIC,
                tags=["moral_severity", expected_severity],
                input=decision_kwargs,
                expected={"severity": expected_severity},
            )
            for case_id, decision_kwargs, expected_severity in self._SEVERITY_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        from src.core.moral_ledger import MORAL_DIMENSIONS, MORAL_RUBRIC_VERSION, MoralDecision

        decision = MoralDecision(
            rubric_version=MORAL_RUBRIC_VERSION,
            scores=dict.fromkeys(MORAL_DIMENSIONS, 3),
            reasoning=case.input["reasoning"],
            is_approved=False,
            security_conflict=case.input["security_conflict"],
            violated_tiers=tuple(case.input["violated_tiers"]),
            remediation_constraints=tuple(case.input["remediation_constraints"]),
        )

        severity, summary = self._orch._classify_moral_rejection(decision)

        failures: List[str] = []
        if severity != case.expected["severity"]:
            failures.append(
                f"severity: expected {case.expected['severity']!r}, got {severity!r}"
            )

        return {"severity": severity, "summary": summary[:120]}, failures
