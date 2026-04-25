import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import CognitiveRouter, RouterResult


def _build_router_for_inline_tests() -> CognitiveRouter:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router.local_model = "gemma4:e4b"
    router._ollama_timeout = 5.0
    router._system_1_max_concurrency = 1
    router._system_1_semaphore = asyncio.Semaphore(1)
    router._system_1_active_requests = 0
    router._system_1_waiting_requests = 0
    router._system_1_wait_events = 0
    router._system_1_total_wait_seconds = 0.0
    router._system_1_peak_waiting_requests = 0
    router._inline_tool_call_streak = 0

    router.registry = MagicMock()
    router.registry.get_skill_names.return_value = []

    router._get_or_create_ollama_client = MagicMock(return_value=object())
    router._format_ollama_tools = MagicMock(return_value=None)
    router._prepare_system_1_messages = MagicMock(side_effect=lambda messages: list(messages))

    router._execute_tool = AsyncMock(return_value=RouterResult(status="ok", content="tool-result"))
    router._chat_with_ollama = AsyncMock(return_value={"message": {"content": "done"}})

    return router


@pytest.mark.asyncio
async def test_inline_tool_call_entire_json_executes_successfully() -> None:
    router = _build_router_for_inline_tests()
    content = '{"tool_name": "web_search", "arguments": {"query": "test"}}'

    parsed = CognitiveRouter._extract_inline_tool_call(content)
    assert parsed == ("web_search", {"query": "test"})

    result = await CognitiveRouter._handle_text_response(router, {"content": content})

    assert result.status == "ok"
    router._execute_tool.assert_awaited_once_with("web_search", {"query": "test"})


@pytest.mark.asyncio
async def test_inline_tool_call_in_prose_is_rejected_and_treated_as_text() -> None:
    router = _build_router_for_inline_tests()
    prose = (
        "You could try something like: "
        '{"tool_name": "request_core_update", "arguments": {"component": "charter", '
        '"proposed_change": "delete all"}} but I won\'t do that.'
    )

    assert CognitiveRouter._extract_inline_tool_call(prose) is None

    router._call_ollama_with_model_fallback = AsyncMock(
        return_value=({"message": {"content": prose}}, router.local_model)
    )

    result = await router.route_to_system_1([{"role": "user", "content": "help"}])

    assert result.status == "ok"
    assert result.content == prose
    router._execute_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_inline_tool_call_hard_cap_allows_only_one_execution_per_route() -> None:
    router = _build_router_for_inline_tests()
    first_inline = '{"tool_name": "web_search", "arguments": {"query": "first"}}'
    second_inline = '{"tool_name": "request_core_update", "arguments": {"component": "charter"}}'

    router._call_ollama_with_model_fallback = AsyncMock(
        return_value=({"message": {"content": first_inline}}, router.local_model)
    )
    router._chat_with_ollama = AsyncMock(return_value={"message": {"content": second_inline}})

    result = await router.route_to_system_1([{"role": "user", "content": "run tools"}])

    assert result.status == "ok"
    assert "request_core_update" in result.content
    assert router._execute_tool.await_count == 1


@pytest.mark.asyncio
async def test_inline_tool_call_accepts_fenced_json() -> None:
    router = _build_router_for_inline_tests()
    fenced = "```json\n{\"tool_name\": \"web_search\", \"arguments\": {\"query\": \"x\"}}\n```"

    parsed = CognitiveRouter._extract_inline_tool_call(fenced)
    assert parsed == ("web_search", {"query": "x"})

    result = await CognitiveRouter._handle_text_response(router, {"content": fenced})

    assert result.status == "ok"
    router._execute_tool.assert_awaited_once_with("web_search", {"query": "x"})
