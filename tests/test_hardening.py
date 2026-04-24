import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.llm_router import CognitiveRouter, RouterResult
from src.core.orchestrator import (
    Orchestrator,
    _BLOCKED_ENV_PREFIXES,
    _build_safe_subprocess_env,
)
from src.core.runtime_context import get_vector_memory, set_runtime_context


class _ToolCall:
    def __init__(self, tool_call_id: str):
        self.id = tool_call_id


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


def test_fix1_runtime_context_returns_same_vector_instance():
    sentinel_vector = object()
    set_runtime_context(None, None, sentinel_vector)  # type: ignore[arg-type]
    try:
        assert get_vector_memory() is sentinel_vector
    finally:
        set_runtime_context(None, None, None)


def test_fix2_asyncio_sleep_allowed_but_asyncio_subprocess_blocked():
    safe_code = (
        "import asyncio\n"
        "async def safe_tool() -> str:\n"
        "    await asyncio.sleep(0)\n"
        "    return 'ok'\n"
    )
    CognitiveRouter._validate_tool_code_ast(safe_code, "safe_tool")

    blocked_code = (
        "import asyncio\n"
        "async def blocked_tool() -> str:\n"
        "    await asyncio.create_subprocess_shell('python -V')\n"
        "    return 'nope'\n"
    )
    with pytest.raises(ValueError, match="create_subprocess_shell"):
        CognitiveRouter._validate_tool_code_ast(blocked_code, "blocked_tool")


def test_fix3_safe_subprocess_env_excludes_blocked_prefixes(monkeypatch):
    blocked_keys = []
    for prefix in _BLOCKED_ENV_PREFIXES:
        key = f"{prefix}HARDENING_TEST"
        blocked_keys.append(key)
        monkeypatch.setenv(key, "secret")

    safe_env = _build_safe_subprocess_env()

    for key in blocked_keys:
        assert key not in safe_env


@pytest.mark.asyncio
async def test_fix5_replenish_applies_across_two_turns(monkeypatch):
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "200")
    monkeypatch.setenv("ENERGY_REPLENISH_PER_TURN", "5")

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._ready = asyncio.Event()
    orchestrator._ready.set()
    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}
    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = 40

    orchestrator._get_user_lock = AsyncMock(return_value=asyncio.Lock())
    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_resume_tool_approval = AsyncMock(return_value=None)
    orchestrator._remember_user_profile = AsyncMock(return_value=False)
    orchestrator._remember_assistant_identity = AsyncMock(return_value=False)
    orchestrator._try_goal_planning_response = AsyncMock(return_value=None)
    orchestrator._run_graph_loop = AsyncMock(return_value="graph")
    orchestrator._finalize_user_response = AsyncMock(side_effect=lambda _uid, _msg, response: response)

    def fake_load_state(user_id: str, user_message: str, *, user_prompt=None):
        return _minimal_state(user_id, user_message, user_prompt or {})

    async def fake_fast_path(state):
        async with orchestrator._predictive_energy_budget_lock:
            orchestrator._predictive_energy_budget_remaining -= 3
        state["_energy_gate_cleared"] = True
        return "ok"

    orchestrator._load_state = AsyncMock(side_effect=fake_load_state)
    orchestrator._try_fast_path_response = AsyncMock(side_effect=fake_fast_path)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)

    initial_budget = orchestrator._predictive_energy_budget_remaining

    resp1 = await Orchestrator.process_message(orchestrator, "first turn", "user-1")
    resp2 = await Orchestrator.process_message(orchestrator, "second turn", "user-1")

    assert resp1 == "ok"
    assert resp2 == "ok"

    projected_without_replenish = initial_budget - 6
    assert orchestrator._predictive_energy_budget_remaining == projected_without_replenish + 10


@pytest.mark.asyncio
async def test_fix6_energy_gate_flag_set_after_fast_path_single_tool():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._assess_request_route = MagicMock(
        return_value={
            "mode": "single_tool",
            "tool_name": "web_search",
            "arguments": {"query": "Vienna weather"},
        }
    )
    orchestrator._try_ad_hoc_dispatch_energy_gate = AsyncMock(return_value=None)
    orchestrator._get_capabilities_string = MagicMock(return_value="")
    orchestrator._route_to_system_1 = AsyncMock(return_value=RouterResult(status="ok", content="summary"))
    orchestrator.core_memory = MagicMock()
    orchestrator.core_memory.get_context_string = AsyncMock(return_value="")
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router._execute_tool = AsyncMock(
        return_value=RouterResult(status="ok", content="tool output")
    )

    state = {
        "user_id": "user-1",
        "user_input": "What's the weather in Vienna?",
        "chat_history": [],
        "user_prompt": {},
    }
    orchestrator._coerce_fast_path_state = MagicMock(return_value=state)

    result = await Orchestrator._try_fast_path_response(orchestrator, state)

    assert result == "summary"
    assert state["_energy_gate_cleared"] is True


def test_fix7_groq_followup_arguments_are_json_strings_for_dict_inputs():
    messages = [{"role": "user", "content": "weather"}]
    executed_calls = [
        {
            "tool_call": _ToolCall("call-1"),
            "raw_tool_name": "web_search",
            "raw_arguments": {"query": "Vienna weather"},
            "tool_output": "sunny",
        }
    ]

    followup = CognitiveRouter._build_groq_followup_messages(messages, executed_calls)
    arguments = followup[-2]["tool_calls"][0]["function"]["arguments"]

    assert isinstance(arguments, str)


@pytest.mark.asyncio
async def test_fix13_fire_and_forget_logs_background_exceptions(caplog):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._background_tasks = set()

    async def boom():
        raise RuntimeError("boom")

    with caplog.at_level("ERROR", logger="src.core.orchestrator"):
        task = Orchestrator._fire_and_forget(orchestrator, boom())
        while not task.done():
            await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert any(
        "Background task" in record.message and "boom" in record.message
        for record in caplog.records
    )
