from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator
from src.core.prompt_config import build_supervisor_prompt, build_supervisor_turn_context


def _stock_schema():
    return {
        "name": "get_stock_price",
        "description": "Fetch stock price, quote, ticker, shares, and market data.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Ticker symbol"}},
            "required": ["ticker"],
        },
    }


def _build_route_orchestrator(schemas=None):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.registry.get_schemas.return_value = list(schemas or [])
    return orchestrator


def test_multi_ticker_detection_requires_two_symbols_and_finance_intent():
    assert Orchestrator._extract_multiple_tickers("TSLA price") == []
    assert Orchestrator._extract_multiple_tickers("Check MSFT and AAPL prices") == ["MSFT", "AAPL"]
    assert Orchestrator._extract_multiple_tickers("AI and ML are hot") == []
    assert Orchestrator._extract_multiple_tickers("I saw MSFT in the news today") == []


def test_single_ticker_finance_request_still_routes_to_stock_tool():
    orchestrator = _build_route_orchestrator([_stock_schema()])

    assessment = Orchestrator._assess_request_route(orchestrator, "TSLA price")

    assert assessment["mode"] == "single_tool"
    assert assessment["tool_name"] == "get_stock_price"
    assert assessment["arguments"] == {"ticker": "TSLA"}


def test_multi_ticker_false_positive_phrases_fall_back_to_graph():
    orchestrator = _build_route_orchestrator([_stock_schema()])

    ai_assessment = Orchestrator._assess_request_route(orchestrator, "AI and ML are hot")
    news_assessment = Orchestrator._assess_request_route(orchestrator, "I saw MSFT in the news today")

    assert ai_assessment["mode"] == "graph"
    assert news_assessment["mode"] == "graph"


@pytest.mark.asyncio
async def test_intent_classifier_classifies_capability_query_and_caches():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._route_to_system_1 = AsyncMock(
        return_value=RouterResult(status="ok", content='{"intent":"capability_query"}')
    )
    message = "do you have the capability to find papers about transformers"

    first = await Orchestrator._classify_user_intent(orchestrator, message)
    second = await Orchestrator._classify_user_intent(orchestrator, message)

    assert first == "capability_query"
    assert second == "capability_query"
    orchestrator._route_to_system_1.assert_awaited_once()
    _, kwargs = orchestrator._route_to_system_1.await_args
    assert kwargs["allowed_tools"] == []
    assert kwargs["deadline_seconds"] == pytest.approx(10.0)
    assert kwargs["max_output_tokens"] == 60


@pytest.mark.asyncio
async def test_intent_classifier_classifies_explicit_web_search_as_task():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._route_to_system_1 = AsyncMock(
        return_value=RouterResult(status="ok", content='{"intent":"task"}')
    )

    intent = await Orchestrator._classify_user_intent(
        orchestrator,
        "search the web for papers about transformers",
    )

    assert intent == "task"


@pytest.mark.asyncio
async def test_capability_classifier_drives_meta_fast_path_even_when_scoped_skill_context_disabled():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._route_to_system_1 = AsyncMock(
        return_value=RouterResult(status="ok", content='{"intent":"capability_query"}')
    )
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router._enable_scoped_skill_context = False
    orchestrator.cognitive_router.registry.get_skill_names.return_value = ["web_search", "extract_pdf_text"]

    result = await Orchestrator._try_meta_fast_path_response(
        orchestrator,
        {
            "user_id": "user-1",
            "user_input": "do you have the capability to find papers about transformers",
            "chat_history": [],
        },
    )

    assert "web_search" in result
    orchestrator._route_to_system_1.assert_awaited_once()


def test_supervisor_system_prompt_is_stable_when_turn_context_changes():
    prompt_one = build_supervisor_prompt(
        charter_text="Charter",
        core_mem_str="Core memory",
        archival_block="Archival hit for turn one",
        capabilities_str="Capabilities",
        agent_descriptions="Agents",
        sensory_context="2026-04-24 10:01 CPU 5%",
        os_name="Windows",
        downloads_dir="downloads",
    )
    prompt_two = build_supervisor_prompt(
        charter_text="Charter",
        core_mem_str="Core memory",
        archival_block="Different archival hit for turn two",
        capabilities_str="Capabilities",
        agent_descriptions="Agents",
        sensory_context="2026-04-24 10:01 CPU 42%",
        os_name="Windows",
        downloads_dir="downloads",
    )
    turn_context = build_supervisor_turn_context(
        sensory_context="2026-04-24 10:01 CPU 42%",
        archival_block="Different archival hit for turn two",
    )

    assert prompt_one.encode("utf-8") == prompt_two.encode("utf-8")
    assert "CPU 42%" not in prompt_two
    assert "Different archival hit" not in prompt_two
    assert "CPU 42%" in turn_context
    assert "Different archival hit" in turn_context


def test_age_extraction_requires_first_person_subject():
    assert Orchestrator._extract_user_profile_updates("I am 37 years old") == {"age": 37}
    assert Orchestrator._extract_user_profile_updates("I'm 38 years old") == {"age": 38}
    assert "age" not in Orchestrator._extract_user_profile_updates("I ran a 5 years old marathon.")


# ---------------------------------------------------------------------------
# Issue 8 — compact skill-list fast path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "Can you tell me the list of skills you have?",
    "list your skills",
    "list your tools",
    "show me your tools",
    "what tools do you have?",
    "what skills do you have available?",
    "what are your capabilities?",
    "all your tools",
    "display your capabilities",
    "enumerate your skills",
    "give me a list of your tools",
])
def test_is_skill_list_request_detects_inventory_queries(msg):
    assert Orchestrator._is_skill_list_request(msg) is True, f"Expected True for: {msg!r}"


@pytest.mark.parametrize("msg", [
    "can you search the web for me?",
    "search for a list of tools for woodworking",
    "find me a list of Python tools for data science",
    "What is the weather like today?",
    "do you have access to the internet?",
    "hello, how are you?",
    "list the files in my downloads folder",
])
def test_is_skill_list_request_rejects_non_inventory_queries(msg):
    assert Orchestrator._is_skill_list_request(msg) is False, f"Expected False for: {msg!r}"


def _build_skill_list_orchestrator(skill_names_and_descs):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    catalog = [{"name": n, "description": d} for n, d in skill_names_and_descs]
    orchestrator.cognitive_router.registry.get_skill_catalog.return_value = catalog
    return orchestrator


def test_build_compact_skill_list_response_format():
    orch = _build_skill_list_orchestrator([
        ("web_search", "Performs an internet search to retrieve the most up-to-date information including news and data."),
        ("get_stock_price", "Fetches current stock price and quote data for a given ticker."),
        ("extract_pdf_text", "Reads and extracts plain text content from a PDF file."),
    ])
    response = orch._build_compact_skill_list_response()

    assert "3 registered skills" in response
    assert "• web_search" in response
    assert "• get_stock_price" in response
    assert "• extract_pdf_text" in response
    # Each description is truncated to ≤ 73 chars (70 + "…")
    for line in response.splitlines():
        if line.startswith("•"):
            parts = line.split(" — ", 1)
            if len(parts) == 2:
                assert len(parts[1]) <= 73, f"Description too long: {parts[1]!r}"


def test_build_compact_skill_list_response_empty_registry():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.registry.get_skill_catalog.return_value = []
    orchestrator.cognitive_router.registry.get_schemas.return_value = []
    assert "don't have any registered skills" in orchestrator._build_compact_skill_list_response()


@pytest.mark.asyncio
async def test_skill_list_fast_path_returns_without_llm():
    orch = _build_skill_list_orchestrator([
        ("web_search", "Searches the internet for current information."),
        ("consolidate_memory", "Consolidates recent interactions into long-term memory."),
    ])
    orch._route_to_system_1 = AsyncMock()
    orch.core_memory = MagicMock()
    orch.core_memory.get_context_string = AsyncMock(return_value="")

    result = await Orchestrator._try_meta_fast_path_response(
        orch,
        {"user_input": "Can you tell me the list of skills you have?", "chat_history": []},
    )

    assert result is not None
    assert "• web_search" in result
    assert "• consolidate_memory" in result
    # Must not have called the LLM
    orch._route_to_system_1.assert_not_awaited()