import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator
from src.core.prompt_config import build_supervisor_prompt, build_supervisor_turn_context
from src.memory.ledger_db import LedgerMemory


_SYSTEM_PROMPT = build_supervisor_prompt(
    charter_text="Core Directive: Do no harm.",
    core_mem_str="<Core_Working_Memory>core secret user preference</Core_Working_Memory>",
    archival_block="<Archival_Memory>archival secret memory</Archival_Memory>",
    capabilities_str="- web_search: Search current web results.\n- get_stock_price: Fetch live prices.",
    agent_descriptions="- research_agent: Gather relevant facts.\n- coder_agent: Implement code changes.",
    sensory_context="[Machine Context - OS: Windows 11 | CPU: 6% | CWD: C:\\Users\\Alice\\Project]",
    os_name="Windows 11",
    downloads_dir="downloads",
)
_TURN_CONTEXT = build_supervisor_turn_context(
    archival_block="<Archival_Memory>archival secret memory</Archival_Memory>",
    sensory_context="[Machine Context - OS: Windows 11 | CPU: 6% | CWD: C:\\Users\\Alice\\Project]",
)


def test_redaction_preserves_catalogs_but_removes_sensitive_context() -> None:
    redacted = Orchestrator._redact_text_for_cloud(_SYSTEM_PROMPT)
    redacted_turn_context = Orchestrator._redact_text_for_cloud(_TURN_CONTEXT)

    assert "<available_capabilities>" in redacted
    assert "web_search" in redacted
    assert "get_stock_price" in redacted
    assert "<available_agents>" in redacted
    assert "research_agent" in redacted
    assert "coder_agent" in redacted

    assert "<context_and_memory>" in redacted
    assert "core secret user preference" not in redacted
    assert "[REDACTED_CONTEXT_BLOCK]" in redacted
    assert "archival secret memory" not in redacted_turn_context
    assert "C:\\Users\\Alice" not in redacted_turn_context
    assert "[REDACTED_CONTEXT_BLOCK]" in redacted_turn_context
    assert "[REDACTED_SENSORY_STATE]" in redacted_turn_context


@pytest.mark.asyncio
async def test_supervisor_fallback_receives_tool_catalog_and_writes_audit(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "cloud_payload_audit.db"))
    await ledger.initialize()
    try:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.get_system_2_available.return_value = True
        orchestrator.cognitive_router.route_to_system_2 = AsyncMock(
            return_value=RouterResult(status="ok", content="WORKERS: []")
        )
        orchestrator._route_to_system_1 = AsyncMock(
            return_value=RouterResult(status="ok", content="[System 1 - Error] local failure")
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "Please plan this with the available tools."},
        ]

        result = await Orchestrator._route_supervisor_request(orchestrator, messages)

        assert result is not None
        assert result.status == "ok"

        args, kwargs = orchestrator.cognitive_router.route_to_system_2.await_args
        forwarded_messages = args[0]
        forwarded_text = "\n".join(message["content"] for message in forwarded_messages)

        assert "web_search" in forwarded_text
        assert "get_stock_price" in forwarded_text
        assert "research_agent" in forwarded_text
        assert "coder_agent" in forwarded_text
        assert "core secret user preference" not in forwarded_text
        assert "archival secret memory" not in forwarded_text
        assert kwargs["allowed_tools"] is None

        entries = await ledger.get_cloud_payload_audit_entries(limit=5)
        assert len(entries) == 1
        assert entries[0]["purpose"] == "supervisor_fallback"
        assert entries[0]["message_count_before"] == 2
        assert entries[0]["message_count_after"] == 2
        assert entries[0]["allow_sensitive_context"] is False
        assert entries[0]["payload_sha256"] == Orchestrator._cloud_payload_audit_sha256(
            forwarded_messages,
            None,
        )
    finally:
        await ledger.close()


def test_system2_redaction_call_sites_use_literal_sensitivity_decisions() -> None:
    source = Path("src/core/orchestrator.py").read_text(encoding="utf-8")

    assert "_SENSITIVE_CONTEXT_HINT_RE" not in source
    assert "_requires_sensitive_cloud_context" not in source

    tree = ast.parse(source)
    route_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "_route_to_system_2_redacted":
            route_calls.append(node)

    assert route_calls
    for call in route_calls:
        sensitivity_keywords = [kw for kw in call.keywords if kw.arg == "allow_sensitive_context"]
        if not sensitivity_keywords:
            continue
        keyword = sensitivity_keywords[0]
        assert isinstance(keyword.value, ast.Constant)
        assert keyword.value.value is False
