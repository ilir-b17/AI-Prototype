import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.orchestrator import Orchestrator


def _minimal_state(user_id: str, user_input: str, user_prompt: dict) -> dict:
    return {
        "user_id": user_id,
        "user_input": user_input,
        "chat_history": [],
        "current_plan": [],
        "worker_outputs": {},
        "final_response": "",
        "iteration_count": 0,
        "admin_guidance": "",
        "energy_remaining": 100,
        "hitl_count": 0,
        "user_prompt": dict(user_prompt),
    }


def _build_tool_approval_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._ready = asyncio.Event()
    orchestrator._ready.set()

    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}

    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()

    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = 100

    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_goal_planning_response = AsyncMock(return_value=None)
    orchestrator._try_fast_path_response = AsyncMock(return_value="fast-path-reply")
    orchestrator._run_graph_loop = AsyncMock(return_value="graph")

    orchestrator._remember_user_profile = AsyncMock(return_value=False)
    orchestrator._remember_assistant_identity = AsyncMock(return_value=False)

    orchestrator._load_state = AsyncMock(
        side_effect=lambda user_id, user_message, **kwargs: _minimal_state(
            user_id,
            user_message,
            kwargs.get("user_prompt", {}),
        )
    )

    async def _finalize(_user_id: str, _user_message: str, response: str, state=None) -> str:
        return response

    orchestrator._finalize_user_response = AsyncMock(side_effect=_finalize)
    orchestrator._update_synthesis_run_status_if_supported = AsyncMock()

    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.register_dynamic_tool = AsyncMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)

    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.clear_pending_approval = AsyncMock()
    orchestrator.ledger_memory.register_tool = AsyncMock()
    orchestrator.ledger_memory.approve_tool = AsyncMock()

    orchestrator.core_memory = MagicMock()
    orchestrator.core_memory.get_all = AsyncMock(return_value={"known_capabilities": ""})
    orchestrator.core_memory.update = AsyncMock()

    return orchestrator


@pytest.mark.asyncio
async def test_yes_tool_approval_does_not_deadlock_on_user_lock_reentry() -> None:
    orchestrator = _build_tool_approval_orchestrator()
    user_id = "user-1"

    orchestrator.pending_tool_approval[user_id] = {
        "_created_at": time.time(),
        "synthesis": {
            "tool_name": "hello_tool",
            "description": "Simple test tool",
            "code": "async def hello_tool() -> str:\n    return 'ok'\n",
            "schema_json": {"name": "hello_tool", "description": "Simple test tool"},
        },
        "original_state": {
            "user_id": user_id,
            "user_input": "Hi",
        },
    }

    response = await asyncio.wait_for(
        Orchestrator.process_message(orchestrator, "YES", user_id),
        timeout=5.0,
    )

    assert "deployed" in response.lower()
    assert "fast-path-reply" in response
