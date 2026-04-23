from unittest.mock import MagicMock

import pytest

from src.core.moral_ledger import (
    MORAL_DIMENSIONS,
    MoralDecision,
    validate_moral_decision_payload,
)
from src.core.orchestrator import Orchestrator
from src.core.state_model import AgentState


def _valid_payload():
    return {
        "rubric_version": "moral_ledger_v1",
        "scores": dict.fromkeys(MORAL_DIMENSIONS, 4),
        "reasoning": "The plan is aligned with charter constraints.",
        "is_approved": True,
    }


def test_moral_decision_validation_accepts_strict_payload():
    decision = validate_moral_decision_payload(_valid_payload())

    assert isinstance(decision, MoralDecision)
    assert decision.is_approved is True
    assert decision.decision_mode == "system2_audit"
    assert decision.validation_error == ""
    assert set(decision.scores.keys()) == set(MORAL_DIMENSIONS)


def test_moral_decision_validation_fails_closed_when_required_field_missing():
    payload = _valid_payload()
    payload.pop("is_approved")

    decision = validate_moral_decision_payload(payload)

    assert decision.is_approved is False
    assert decision.decision_mode == "validation_failure"
    assert "missing_required_fields" in decision.validation_error


def test_moral_decision_validation_rejects_missing_score_dimension():
    payload = _valid_payload()
    payload["scores"].pop("data_privacy")

    decision = validate_moral_decision_payload(payload)

    assert decision.is_approved is False
    assert decision.decision_mode == "validation_failure"
    assert "invalid_score_dimensions" in decision.validation_error


def test_moral_decision_validation_rejects_non_integer_score_values():
    payload = _valid_payload()
    payload["scores"]["harm_reduction"] = True

    decision = validate_moral_decision_payload(payload)

    assert decision.is_approved is False
    assert decision.decision_mode == "validation_failure"
    assert "invalid_score_value" in decision.validation_error


@pytest.mark.asyncio
async def test_triviality_gate_sets_deterministic_bypass_trace_for_benign_read_only_task():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.registry.get_schemas.return_value = [
        {
            "name": "get_system_info",
            "description": "Returns current date, time, timezone, and system platform details.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    ]

    state = AgentState.new("user-1", "Can you tell me the current time?").to_dict()
    state["final_response"] = "The current time is 09:15."

    result = await Orchestrator.critic_node(orchestrator, state)

    assert result["critic_feedback"] == "PASS"
    assert result["moral_audit_mode"] == "triviality_bypass"
    assert result["moral_audit_bypassed"] is True
    assert result["moral_decision"]["is_approved"] is True
    assert result["moral_decision"]["decision_mode"] == "triviality_bypass"


def test_triviality_gate_rejects_mutation_or_escalation_tasks():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.registry.get_schemas.return_value = []

    state = AgentState.new("user-1", "Please update the system and deploy the change").to_dict()
    state["current_plan"] = [
        {
            "agent": "coder_agent",
            "task": "Modify core runtime and deploy update",
            "reason": "Need to write and deploy the change",
        }
    ]

    decision = Orchestrator._try_triviality_bypass_decision(
        orchestrator,
        state,
        output_to_eval="I will modify files and deploy changes.",
    )

    assert decision is None
