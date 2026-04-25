import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import RequiresHITLError
from src.core.orchestrator import Orchestrator


def _build_heartbeat_hitl_orchestrator(admin_user_id: str) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._admin_user_id = admin_user_id
    orchestrator._ready = asyncio.Event()
    orchestrator._ready.set()

    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}

    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()

    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = 100

    orchestrator._synthesis_in_progress = set()
    orchestrator._consolidation_turn_counts = {}
    orchestrator._background_tasks = set()
    orchestrator._heartbeat_failure_counts = {}

    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_resume_tool_approval = AsyncMock(return_value=None)
    orchestrator._remember_user_profile = AsyncMock(return_value=False)
    orchestrator._remember_assistant_identity = AsyncMock(return_value=False)
    orchestrator._try_goal_planning_response = AsyncMock(return_value=None)
    orchestrator._try_fast_path_response = AsyncMock(return_value=None)
    orchestrator._apply_text_memory_hooks = AsyncMock(return_value=None)

    orchestrator._get_predictive_energy_budget_remaining = AsyncMock(return_value=100)
    orchestrator._clear_heartbeat_failure_count = AsyncMock()
    orchestrator._increment_heartbeat_failure_count = AsyncMock(return_value=1)

    async def _finalize(_user_id: str, _user_message: str, response: str, state=None) -> str:
        return response

    orchestrator._finalize_user_response = AsyncMock(side_effect=_finalize)

    orchestrator._select_executable_heartbeat_tasks = AsyncMock(
        return_value=[
            {
                "task": {
                    "id": 42,
                    "title": "Investigate issue",
                    "priority": 1,
                },
                "story": None,
                "epic": None,
            }
        ]
    )
    orchestrator._evaluate_energy_for_context = AsyncMock(
        return_value=(
            SimpleNamespace(estimated_effort=1, expected_value=8),
            SimpleNamespace(
                should_execute=True,
                predicted_cost=1,
                min_reserve=0,
                defer_count=0,
                reason="ok",
                roi=8.0,
            ),
            100,
        )
    )
    orchestrator._build_energy_evaluation_record = MagicMock(return_value={"ok": True})
    orchestrator._try_reserve_predictive_energy_budget = AsyncMock(return_value=True)

    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)

    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.update_objective_status = AsyncMock()
    orchestrator.ledger_memory.record_task_energy_evaluation = AsyncMock()
    orchestrator.ledger_memory.clear_hitl_state = AsyncMock()
    orchestrator.ledger_memory.save_hitl_state = AsyncMock()
    orchestrator.ledger_memory.get_chat_history = AsyncMock(return_value=[])
    orchestrator.ledger_memory.save_chat_turn = AsyncMock()
    orchestrator.ledger_memory.set_system_state = AsyncMock()

    orchestrator.outbound_queue = asyncio.Queue()
    return orchestrator


@pytest.mark.asyncio
async def test_heartbeat_hitl_routes_to_admin_and_resumes_task_completion() -> None:
    admin_user_id = "123456"
    orchestrator = _build_heartbeat_hitl_orchestrator(admin_user_id)

    async def _run_graph_loop(state, _user_id, _user_message):
        if "[ADMIN GUIDANCE:" in str(state.get("user_input") or ""):
            return "Heartbeat task completed after admin guidance."
        raise RequiresHITLError("Need guidance on X")

    orchestrator._run_graph_loop = AsyncMock(side_effect=_run_graph_loop)

    await Orchestrator._run_heartbeat_cycle(orchestrator)

    assert admin_user_id in orchestrator.pending_hitl_state
    assert "heartbeat" not in orchestrator.pending_hitl_state
    assert orchestrator.pending_hitl_state[admin_user_id].get("_heartbeat_origin_task_id") == 42

    outbound_message = await orchestrator.outbound_queue.get()
    assert "Need guidance on X" in outbound_message

    resume_response = await Orchestrator.process_message(orchestrator, "here is my guidance", admin_user_id)

    assert "completed after admin guidance" in resume_response.lower()
    assert any(call.args == (42, "completed") for call in orchestrator.ledger_memory.update_objective_status.await_args_list)
    assert admin_user_id not in orchestrator.pending_hitl_state
