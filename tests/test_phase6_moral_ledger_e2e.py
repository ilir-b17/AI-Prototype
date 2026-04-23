import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import RequiresHITLError, RouterResult
from src.core.moral_ledger import MORAL_DIMENSIONS, MORAL_RUBRIC_VERSION
from src.core.orchestrator import Orchestrator
from src.core.state_model import AgentState
from src.memory.ledger_db import LedgerMemory


def _build_moral_payload(
    *,
    is_approved: bool,
    reasoning: str,
    remediation_constraints=None,
    violated_tiers=None,
    security_conflict: bool = False,
    scores=None,
) -> str:
    payload_scores = dict.fromkeys(MORAL_DIMENSIONS, 4)
    if scores:
        payload_scores.update(scores)
    return json.dumps(
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": payload_scores,
            "reasoning": reasoning,
            "is_approved": bool(is_approved),
            "decision_mode": "system2_audit",
            "bypass_reason": "",
            "remediation_constraints": list(remediation_constraints or []),
            "violated_tiers": list(violated_tiers or []),
            "security_conflict": bool(security_conflict),
        }
    )


def _build_minimal_orchestrator(ledger: LedgerMemory) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = ledger
    orchestrator.charter_text = Path("charter.md").read_text(encoding="utf-8")
    orchestrator.outbound_queue = asyncio.Queue()
    orchestrator._compiled_graph = None
    return orchestrator


@pytest.mark.asyncio
async def test_phase6_scenario_a_benign_trivial_bypass_logs_bypass_mode(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase6_scenario_a.db"))
    await ledger.initialize()
    try:
        orchestrator = _build_minimal_orchestrator(ledger)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.get_system_2_available.return_value = True
        orchestrator.cognitive_router.route_to_system_2 = AsyncMock(
            return_value=RouterResult(status="ok", content="unused")
        )
        orchestrator.cognitive_router.registry.get_schemas.return_value = [
            {
                "name": "get_system_info",
                "description": "Returns current date and time details.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]

        state = AgentState.new("phase6-user-a", "What time is it now?").to_dict()
        state["final_response"] = "The current time is 09:15."

        result = await Orchestrator.critic_node(orchestrator, state)

        assert result["critic_feedback"] == "PASS"
        assert result["moral_audit_mode"] == "triviality_bypass"
        assert result["moral_audit_bypassed"] is True
        orchestrator.cognitive_router.route_to_system_2.assert_not_awaited()

        logs = await ledger.get_moral_audit_logs(user_id="phase6-user-a", limit=5)
        assert len(logs) == 1
        assert logs[0]["audit_mode"] == "triviality_bypass"
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_phase6_scenario_b_moderate_risk_retries_with_constraints_then_passes(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase6_scenario_b.db"))
    await ledger.initialize()
    try:
        orchestrator = _build_minimal_orchestrator(ledger)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.get_system_2_available.return_value = True
        orchestrator.cognitive_router.route_to_system_2 = AsyncMock(
            side_effect=[
                RouterResult(
                    status="ok",
                    content=_build_moral_payload(
                        is_approved=False,
                        reasoning="Output is risky but remediable.",
                        remediation_constraints=[
                            "Remove write/delete operations.",
                            "Keep the next response read-only and scoped.",
                        ],
                        violated_tiers=["tier_2"],
                        security_conflict=False,
                        scores={"epistemic_humility": 2, "alignment_with_user_intent": 2},
                    ),
                ),
                RouterResult(
                    status="ok",
                    content=_build_moral_payload(
                        is_approved=True,
                        reasoning="Remediated output satisfies charter constraints.",
                        remediation_constraints=[],
                        violated_tiers=[],
                        security_conflict=False,
                    ),
                ),
            ]
        )
        orchestrator._route_to_system_1 = AsyncMock(
            return_value=RouterResult(status="ok", content="unused")
        )
        orchestrator._finalize_user_response = AsyncMock(
            side_effect=lambda _uid, _msg, response: response
        )

        seen_critic_instructions = []

        async def supervisor_stub(state):
            await asyncio.sleep(0)
            seen_critic_instructions.append(str(state.get("critic_instructions") or ""))
            if state.get("critic_instructions"):
                state["worker_outputs"] = {
                    "coder_agent": (
                        "Reworked response with constrained, read-only actions only. "
                        "No file writes, no deletes, no security-sensitive actions. "
                    )
                    * 4
                }
            else:
                state["worker_outputs"] = {
                    "coder_agent": (
                        "Initial risky response that proposes write and delete operations without safeguards. "
                    )
                    * 4
                }
            state["current_plan"] = [
                {
                    "agent": "coder_agent",
                    "task": "Draft a response",
                    "reason": "Handle user request",
                    "depends_on": [],
                    "preferred_model": "",
                }
            ]
            return state

        async def execute_stub(state):
            await asyncio.sleep(0)
            return state

        orchestrator.supervisor_node = supervisor_stub
        orchestrator.execute_workers_node = execute_stub

        state = AgentState.new("phase6-user-b", "Generate a deployment answer.").to_dict()
        final_response = await Orchestrator._run_graph_loop(
            orchestrator,
            state,
            user_id="phase6-user-b",
            user_message="Generate a deployment answer.",
        )

        assert "Reworked response" in final_response
        assert orchestrator.cognitive_router.route_to_system_2.await_count == 2
        assert any("Remediation constraints" in text for text in seen_critic_instructions if text)

        logs = await ledger.get_moral_audit_logs(user_id="phase6-user-b", limit=10)
        assert len(logs) >= 2
        assert any(log["moral_decision"].get("is_approved") is False for log in logs)
        assert any(log["moral_decision"].get("is_approved") is True for log in logs)
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_phase6_scenario_c_severe_risk_triggers_immediate_hitl_halt(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase6_scenario_c.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Phase6 Epic")
        story_id = await ledger.add_objective(tier="Story", title="Phase6 Story", parent_id=epic_id)
        task_id = await ledger.add_objective(
            tier="Task",
            title="High-risk task",
            parent_id=story_id,
            priority=1,
        )

        orchestrator = _build_minimal_orchestrator(ledger)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.get_system_2_available.return_value = True
        orchestrator.cognitive_router.route_to_system_2 = AsyncMock(
            return_value=RouterResult(
                status="ok",
                content=_build_moral_payload(
                    is_approved=False,
                    reasoning="Tier 1 violation: unauthorized write/delete against protected system assets.",
                    remediation_constraints=[],
                    violated_tiers=["tier_1"],
                    security_conflict=True,
                    scores={
                        "harm_reduction": 1,
                        "data_privacy": 1,
                        "admin_authority_security": 1,
                        "data_sovereignty_local_priority": 1,
                    },
                ),
            )
        )
        orchestrator._route_to_system_1 = AsyncMock(
            return_value=RouterResult(status="ok", content="unused")
        )

        state = AgentState.new(
            "heartbeat",
            f"[HEARTBEAT TASK #{task_id}]: Execute malicious overwrite of core runtime.",
        ).to_dict()
        state["critic_instructions"] = "force_system2_review"
        state["worker_outputs"] = {
            "coder_agent": (
                "I will modify core execution files and delete archival memory immediately without MFA safeguards. "
            )
            * 4
        }
        state["current_plan"] = [
            {
                "agent": "coder_agent",
                "task": "Modify and delete protected assets",
                "reason": "malicious rewrite",
                "depends_on": [],
                "preferred_model": "",
            }
        ]

        with pytest.raises(RequiresHITLError, match="Moral governor halt"):
            await Orchestrator.critic_node(orchestrator, state)

        queued_notification = await orchestrator.outbound_queue.get()
        assert "Moral governor halt" in queued_notification

        task_row = await ledger.get_task_with_parent_context(task_id)
        assert task_row is not None
        assert str(task_row["task"]["status"]) == "suspended"

        logs = await ledger.get_moral_audit_logs(user_id="heartbeat", limit=5)
        assert len(logs) == 1
        assert logs[0]["moral_decision"].get("is_approved") is False
        assert logs[0]["moral_decision"].get("security_conflict") is True
    finally:
        await ledger.close()