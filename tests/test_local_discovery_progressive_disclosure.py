import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import CognitiveRouter, RouterResult
from src.core.orchestrator import Orchestrator
from src.core.skill_manager import SkillRegistry


def _write_skill(skill_dir: Path, skill_name: str, skill_markdown: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill_markdown, encoding="utf-8")
    (skill_dir / "__init__.py").write_text(
        (
            f"async def {skill_name}(**kwargs):\n"
            "    return \"ok\"\n"
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_capability_gap_is_resolved_locally_and_bypasses_synthesis() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda value: value)
    orchestrator.cognitive_router.registry = MagicMock()
    orchestrator.cognitive_router.registry.get_skill_catalog.return_value = [
        {
            "name": "web_search",
            "description": "Search current web sources for news and weather updates.",
        }
    ]
    orchestrator.cognitive_router.registry.get_schemas.return_value = []

    orchestrator._route_to_system_1 = AsyncMock(
        return_value=RouterResult(status="ok", content="Recovered locally via web_search.")
    )
    orchestrator._fire_and_forget = MagicMock()
    orchestrator._async_tool_synthesis = AsyncMock()
    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.ledger_memory = MagicMock()

    blocked = RouterResult(
        status="capability_gap",
        gap_description="Need live internet weather data for Vienna.",
        suggested_tool_name="internet_weather_lookup",
    )

    response = await Orchestrator._handle_blocked_result(
        orchestrator,
        blocked,
        "user-1",
        {"user_input": "What is the weather in Vienna right now?"},
    )

    assert response == "Recovered locally via web_search."
    orchestrator._route_to_system_1.assert_awaited_once()
    orchestrator._fire_and_forget.assert_not_called()


@pytest.mark.asyncio
async def test_skill_body_is_scoped_to_runtime_tool_turn_only(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(
        skills_dir / "web_search",
        "web_search",
        """---
name: web_search
description: Lightweight web retrieval for routing decisions.
---

Detailed runtime instructions for web_search skill.
Use this body only when this skill is invoked.

## Schema
```json
{
  "name": "web_search",
  "description": "Schema description for web_search.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"}
    },
    "required": ["query"]
  }
}
```
""",
    )

    registry = SkillRegistry(skills_dir=str(skills_dir))

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.registry = registry

    global_capabilities = Orchestrator._get_capabilities_string(orchestrator)
    assert "web_search: Lightweight web retrieval for routing decisions." in global_capabilities
    assert "Detailed runtime instructions for web_search skill." not in global_capabilities

    router = CognitiveRouter.__new__(CognitiveRouter)
    router._enable_parallel_tools = False
    router._parallel_read_only_tool_concurrency = 1
    router._skill_context_resolver = (
        lambda tool_name: Orchestrator._build_scoped_skill_runtime_context(orchestrator, tool_name)
    )
    router._execute_tool = AsyncMock(
        side_effect=[
            RouterResult(status="ok", content="result-alpha"),
            RouterResult(status="ok", content="result-beta"),
        ]
    )
    router._chat_with_ollama = AsyncMock(return_value={"message": {"content": "done"}})

    base_messages = [
        {"role": "system", "content": global_capabilities},
        {"role": "user", "content": "Find current weather in Vienna."},
    ]
    assert "Detailed runtime instructions for web_search skill." not in json.dumps(base_messages)

    first_message = {
        "tool_calls": [
            {"function": {"name": "web_search", "arguments": {"query": "Vienna weather alpha"}}},
            {"function": {"name": "web_search", "arguments": {"query": "Vienna weather beta"}}},
        ]
    }

    result = await CognitiveRouter._run_tool_loop(
        router,
        client=MagicMock(),
        model="gemma4:e4b",
        first_message=first_message,
        messages=base_messages,
        active_tools=[{"type": "function"}],
    )

    assert result.status == "ok"
    assert result.content == "done"

    followup_messages = router._chat_with_ollama.await_args.args[2]
    scoped_contexts = [
        msg.get("content", "")
        for msg in followup_messages
        if msg.get("role") == "system" and "<scoped_skill_context>" in str(msg.get("content", ""))
    ]
    assert len(scoped_contexts) == 1
    assert "Detailed runtime instructions for web_search skill." in scoped_contexts[0]
    assert "Scope: immediate execution turn only" in scoped_contexts[0]
