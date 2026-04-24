import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import CognitiveRouter, RouterResult


def _make_router(*, enable_parallel_tools: bool) -> CognitiveRouter:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._ollama_options = {}
    router._ollama_timeout = 5.0
    router._enable_parallel_tools = enable_parallel_tools
    router._parallel_read_only_tool_concurrency = 4
    router.registry = MagicMock()
    router.registry.get_schemas.return_value = []
    return router


@pytest.mark.asyncio
async def test_system1_read_only_batch_runs_concurrently_when_flag_enabled():
    router = _make_router(enable_parallel_tools=True)

    active_calls = 0
    peak_calls = 0

    async def fake_execute(tool_name: str, arguments: dict) -> RouterResult:
        nonlocal active_calls, peak_calls
        _ = (tool_name, arguments)
        active_calls += 1
        peak_calls = max(peak_calls, active_calls)
        await asyncio.sleep(0.05)
        active_calls -= 1
        return RouterResult(status="ok", content=f"ok:{tool_name}")

    router._execute_tool = AsyncMock(side_effect=fake_execute)

    first_message = {
        "tool_calls": [
            {"function": {"name": "web_search", "arguments": {"query": "alpha"}}},
            {"function": {"name": "get_system_info", "arguments": {}}},
            {"function": {"name": "search_archival_memory", "arguments": {"query": "beta"}}},
        ]
    }

    client = MagicMock()
    client.chat = AsyncMock(return_value={"message": {"content": "batch complete"}})

    result = await router._run_tool_loop(
        client,
        "gemma4:e4b",
        first_message,
        [{"role": "user", "content": "run tools"}],
        active_tools=[{"type": "function"}],
    )

    assert result.status == "ok"
    assert result.content == "batch complete"
    assert router._execute_tool.await_count == 3
    assert peak_calls >= 2


@pytest.mark.asyncio
async def test_system1_read_only_batch_runs_sequentially_when_flag_disabled():
    router = _make_router(enable_parallel_tools=False)

    active_calls = 0
    peak_calls = 0

    async def fake_execute(tool_name: str, arguments: dict) -> RouterResult:
        nonlocal active_calls, peak_calls
        _ = (tool_name, arguments)
        active_calls += 1
        peak_calls = max(peak_calls, active_calls)
        await asyncio.sleep(0.05)
        active_calls -= 1
        return RouterResult(status="ok", content=f"ok:{tool_name}")

    router._execute_tool = AsyncMock(side_effect=fake_execute)

    first_message = {
        "tool_calls": [
            {"function": {"name": "web_search", "arguments": {"query": "alpha"}}},
            {"function": {"name": "get_system_info", "arguments": {}}},
            {"function": {"name": "search_archival_memory", "arguments": {"query": "beta"}}},
        ]
    }

    client = MagicMock()
    client.chat = AsyncMock(return_value={"message": {"content": "batch complete"}})

    result = await router._run_tool_loop(
        client,
        "gemma4:e4b",
        first_message,
        [{"role": "user", "content": "run tools"}],
        active_tools=[{"type": "function"}],
    )

    assert result.status == "ok"
    assert result.content == "batch complete"
    assert router._execute_tool.await_count == 3
    assert peak_calls == 1


@pytest.mark.asyncio
async def test_groq_cloud_path_executes_full_tool_call_batch():
    router = _make_router(enable_parallel_tools=True)
    router.groq_model = "llama-3.3-70b-versatile"

    async def fake_execute(tool_name: str, arguments: dict) -> RouterResult:
        _ = arguments
        await asyncio.sleep(0)
        return RouterResult(status="ok", content=f"result:{tool_name}")

    router._execute_tool = AsyncMock(side_effect=fake_execute)

    tool_calls = [
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="web_search", arguments='{"query":"alpha"}'),
        ),
        SimpleNamespace(
            id="call_2",
            function=SimpleNamespace(name="get_system_info", arguments='{}'),
        ),
        SimpleNamespace(
            id="call_3",
            function=SimpleNamespace(name="search_archival_memory", arguments='{"query":"beta"}'),
        ),
    ]

    choice = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(tool_calls=tool_calls, content=None),
    )
    router._create_groq_completion = AsyncMock(return_value=SimpleNamespace(choices=[choice]))
    router._create_groq_text_completion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="groq final"))]
        )
    )

    result = await router._route_to_groq(
        [{"role": "user", "content": "run batch"}],
        allowed_tools=None,
    )

    assert result.status == "ok"
    assert result.content == "groq final"
    assert router._execute_tool.await_count == 3

    followup_messages = router._create_groq_text_completion.await_args.args[0]
    tool_messages = [msg for msg in followup_messages if msg.get("role") == "tool"]
    assert len(tool_messages) == 3


@pytest.mark.asyncio
async def test_ollama_cloud_path_executes_full_tool_call_batch():
    router = _make_router(enable_parallel_tools=True)
    router.system_2_model = "deepseek-v3.2"
    router.groq_client = None

    async def fake_execute(tool_name: str, arguments: dict) -> RouterResult:
        _ = arguments
        await asyncio.sleep(0)
        return RouterResult(status="ok", content=f"result:{tool_name}")

    router._execute_tool = AsyncMock(side_effect=fake_execute)

    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name="web_search", arguments={"query": "alpha"})),
        SimpleNamespace(function=SimpleNamespace(name="get_system_info", arguments={})),
        SimpleNamespace(function=SimpleNamespace(name="search_archival_memory", arguments={"query": "beta"})),
    ]

    first_response = SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls, content=""))
    second_response = SimpleNamespace(message=SimpleNamespace(tool_calls=[], content="ollama final"))

    router.ollama_cloud_client = MagicMock()
    router.ollama_cloud_client.chat = AsyncMock(side_effect=[first_response, second_response])

    result = await router._route_to_ollama_cloud(
        [{"role": "user", "content": "run batch"}],
        allowed_tools=None,
    )

    assert result.status == "ok"
    assert result.content == "ollama final"
    assert router._execute_tool.await_count == 3
    assert router.ollama_cloud_client.chat.await_count == 2

    followup_messages = router.ollama_cloud_client.chat.await_args_list[1].kwargs["messages"]
    tool_messages = [msg for msg in followup_messages if msg.get("role") == "tool"]
    assert len(tool_messages) == 3
