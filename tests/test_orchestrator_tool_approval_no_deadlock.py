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


def _build_orchestrator_for_approval_retry() -> Orchestrator:
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
    orchestrator._predictive_energy_budget_last_replenished_at = time.time()

    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_goal_planning_response = AsyncMock(return_value=None)
    orchestrator._try_fast_path_response = AsyncMock(return_value="fast-path-retry")
    orchestrator._run_graph_loop = AsyncMock(return_value="graph")

    orchestrator._remember_user_profile = AsyncMock(return_value=False)
    orchestrator._remember_assistant_identity = AsyncMock(return_value=False)
    orchestrator._apply_text_memory_hooks = AsyncMock(return_value=None)

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


def _seed_pending_approval(orchestrator: Orchestrator, user_id: str) -> None:
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


@pytest.mark.asyncio
async def test_tool_approval_yes_returns_deploy_and_retry_within_five_seconds() -> None:
    orchestrator = _build_orchestrator_for_approval_retry()
    user_id = "admin-user"
    _seed_pending_approval(orchestrator, user_id)
    orchestrator._get_user_lock = AsyncMock(side_effect=[asyncio.Lock(), asyncio.Lock()])

    response = await asyncio.wait_for(
        Orchestrator.process_message(orchestrator, "YES", user_id),
        timeout=5.0,
    )

    assert "deployed" in response.lower()
    assert "fast-path-retry" in response


@pytest.mark.asyncio
async def test_tool_approval_retry_acquires_user_lock_twice_not_nested() -> None:
    orchestrator = _build_orchestrator_for_approval_retry()
    user_id = "admin-user"
    _seed_pending_approval(orchestrator, user_id)
    orchestrator._get_user_lock = AsyncMock(side_effect=[asyncio.Lock(), asyncio.Lock()])

    await asyncio.wait_for(
        Orchestrator.process_message(orchestrator, "YES", user_id),
        timeout=5.0,
    )

    assert orchestrator._get_user_lock.await_count == 2
