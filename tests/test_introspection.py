"""
Tests for Direction 4 — Observation and Introspection.
All tests are Tier 1 or Tier 2 (no Ollama required).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.ledger_db import LedgerMemory


# ── LedgerMemory introspection tests ─────────────────────────────────

@pytest.fixture
async def fresh_ledger():
    """Temporary LedgerMemory for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = LedgerMemory(db_path)
    await db.initialize()
    yield db
    await db.close()
    os.unlink(db_path)


async def test_log_supervisor_decision_basic(fresh_ledger):
    """log_supervisor_decision returns a positive id."""
    row_id = await fresh_ledger.log_supervisor_decision(
        user_id="test_user",
        user_input="What time is it?",
        plan_json="[]",
        is_direct=True,
        reasoning="Simple system info",
        energy_before=95,
    )
    assert row_id > 0


async def test_log_supervisor_decision_never_raises_on_bad_input(fresh_ledger):
    """log_supervisor_decision must not raise even with invalid input."""
    # None values, oversized strings — must return 0 or positive id, never raise
    row_id = await fresh_ledger.log_supervisor_decision(
        user_id=None,
        user_input="x" * 10000,  # way over the 500 char limit
        plan_json="invalid json",
        is_direct=False,
        reasoning="",
        energy_before=-1,
    )
    assert isinstance(row_id, int)


async def test_get_recent_supervisor_decisions_empty(fresh_ledger):
    """Returns empty list when no decisions logged."""
    decisions = await fresh_ledger.get_recent_supervisor_decisions()
    assert decisions == []


async def test_get_recent_supervisor_decisions_order(fresh_ledger):
    """Most recent decision returned first."""
    await fresh_ledger.log_supervisor_decision(
        user_id="u1", user_input="First",
        plan_json="[]", is_direct=True, reasoning="", energy_before=100,
    )
    await fresh_ledger.log_supervisor_decision(
        user_id="u1", user_input="Second",
        plan_json="[]", is_direct=True, reasoning="", energy_before=90,
    )
    decisions = await fresh_ledger.get_recent_supervisor_decisions("u1", limit=5)
    assert len(decisions) == 2
    assert decisions[0]["user_input"] == "Second"  # most recent first


async def test_get_recent_supervisor_decisions_user_filter(fresh_ledger):
    """User filter returns only that user's decisions."""
    await fresh_ledger.log_supervisor_decision(
        user_id="user_a", user_input="A question",
        plan_json="[]", is_direct=True, reasoning="", energy_before=100,
    )
    await fresh_ledger.log_supervisor_decision(
        user_id="user_b", user_input="B question",
        plan_json="[]", is_direct=True, reasoning="", energy_before=100,
    )
    a_results = await fresh_ledger.get_recent_supervisor_decisions("user_a")
    b_results = await fresh_ledger.get_recent_supervisor_decisions("user_b")
    assert all(d["user_id"] == "user_a" for d in a_results)
    assert all(d["user_id"] == "user_b" for d in b_results)


async def test_get_last_supervisor_decision(fresh_ledger):
    """get_last_supervisor_decision returns None when empty."""
    result = await fresh_ledger.get_last_supervisor_decision()
    assert result is None


async def test_get_last_supervisor_decision_returns_most_recent(fresh_ledger):
    """get_last_supervisor_decision returns the most recent entry."""
    await fresh_ledger.log_supervisor_decision(
        user_id="u1", user_input="Older",
        plan_json="[]", is_direct=True, reasoning="", energy_before=100,
    )
    await fresh_ledger.log_supervisor_decision(
        user_id="u1", user_input="Newer",
        plan_json="[]", is_direct=False, reasoning="Agent plan", energy_before=85,
    )
    last = await fresh_ledger.get_last_supervisor_decision("u1")
    assert last is not None
    assert last["user_input"] == "Newer"
    assert last["is_direct"] == False


async def test_get_deferred_tasks_empty(fresh_ledger):
    """Returns empty list when no deferred tasks exist."""
    result = await fresh_ledger.get_deferred_tasks_with_energy_context()
    assert result == []


async def test_get_system_error_log_empty(fresh_ledger):
    """Returns empty list when no errors logged."""
    result = await fresh_ledger.get_system_error_log(hours=24)
    assert result == []


async def test_get_system_error_log_filters_info(fresh_ledger):
    """INFO level events are not returned by get_system_error_log."""
    from src.memory.ledger_db import LogLevel
    await fresh_ledger.log_event(LogLevel.INFO, "Normal operation")
    await fresh_ledger.log_event(LogLevel.WARNING, "Something odd happened")
    await fresh_ledger.log_event(LogLevel.ERROR, "Something failed")
    errors = await fresh_ledger.get_system_error_log(hours=24)
    levels = {e["log_level"] for e in errors}
    assert "INFO" not in levels
    assert "WARNING" in levels
    assert "ERROR" in levels


async def test_get_moral_audit_summary_empty(fresh_ledger):
    """Returns empty list when no audit records exist."""
    result = await fresh_ledger.get_moral_audit_summary()
    assert result == []


async def test_get_objective_counts_by_status_empty(fresh_ledger):
    """Returns dict (possibly empty) — never raises."""
    result = await fresh_ledger.get_objective_counts_by_status()
    assert isinstance(result, dict)


async def test_count_supervisor_decisions(fresh_ledger):
    """count_supervisor_decisions returns accurate count."""
    assert await fresh_ledger.count_supervisor_decisions() == 0
    await fresh_ledger.log_supervisor_decision(
        user_id="u1", user_input="Test",
        plan_json="[]", is_direct=True, reasoning="", energy_before=100,
    )
    assert await fresh_ledger.count_supervisor_decisions() == 1
    assert await fresh_ledger.count_supervisor_decisions("u1") == 1
    assert await fresh_ledger.count_supervisor_decisions("u2") == 0


# ── Introspection detection tests ─────────────────────────────────────

def test_is_introspection_query_positive_cases():
    """Known introspection queries are correctly detected."""
    from src.core.orchestrator import Orchestrator

    positive_cases = [
        "Why did you choose research_agent?",
        "Why did you select the coder agent for that task?",
        "What tasks are currently deferred?",
        "Show me blocked tasks",
        "What is my energy budget?",
        "Any recent errors?",
        "Were there any system errors in the last 24 hours?",
        "What happened in the last turn?",
        "Was my response rejected by the moral audit?",
        "Show me the task backlog",
        "Why was that task deferred?",
        "System health status",
        "What went wrong with the synthesis?",
    ]
    for msg in positive_cases:
        assert Orchestrator._is_introspection_query(msg), (
            f"Expected introspection detection for: {msg!r}"
        )


def test_is_introspection_query_negative_cases():
    """Non-introspection queries are correctly rejected."""
    from src.core.orchestrator import Orchestrator

    negative_cases = [
        "What is machine learning?",
        "What time is it?",
        "Hi",
        "Search my memory for AIDEN notes",
        "What is the AAPL stock price?",
        "Weather in Vienna",
        "Convert 100 km to miles",
        "Write a Python script to sort a list",
    ]
    for msg in negative_cases:
        assert not Orchestrator._is_introspection_query(msg), (
            f"False positive introspection detection for: {msg!r}"
        )


def test_classify_introspection_type():
    """Introspection type classification returns expected categories."""
    from src.core.orchestrator import Orchestrator

    type_cases = [
        ("Why did you choose research_agent?", "decision"),
        ("What happened last turn?", "decision"),
        ("What tasks are deferred?", "energy"),
        ("Energy budget status", "energy"),
        ("Show me blocked tasks", "backlog"),
        ("What objectives are pending?", "backlog"),
        ("Any system errors?", "health"),
        ("Synthesis history", "health"),
    ]
    for msg, expected_type in type_cases:
        actual = Orchestrator._classify_introspection_type(msg)
        assert actual == expected_type, (
            f"Expected type {expected_type!r} for {msg!r}, got {actual!r}"
        )


# ── Skill return value tests (no LLM, no network) ────────────────────

async def test_query_decision_log_returns_valid_json():
    """query_decision_log returns valid JSON in all scopes."""
    from src.skills.query_decision_log import query_decision_log

    for scope in ("last_turn", "recent", "rejections", "all"):
        result = await query_decision_log(scope=scope, limit=5)
        parsed = json.loads(result)
        assert "status" in parsed, f"Missing status for scope={scope}"


async def test_query_objective_status_returns_valid_json():
    """query_objective_status returns valid JSON for all status filters."""
    from src.skills.query_objective_status import query_objective_status

    for status in ("all", "pending", "deferred", "blocked", "active"):
        result = await query_objective_status(status_filter=status, limit=5)
        parsed = json.loads(result)
        assert "status" in parsed, f"Missing status for filter={status}"
        assert "backlog_summary" in parsed


async def test_query_energy_state_returns_valid_json():
    """query_energy_state returns valid JSON with expected structure."""
    from src.skills.query_energy_state import query_energy_state

    result = await query_energy_state()
    parsed = json.loads(result)
    assert parsed.get("status") == "success"
    assert "energy" in parsed
    assert "current_predictive_budget" in parsed["energy"]


async def test_query_system_health_returns_valid_json():
    """query_system_health returns valid JSON with expected structure."""
    from src.skills.query_system_health import query_system_health

    result = await query_system_health(hours=1, error_limit=5)
    parsed = json.loads(result)
    assert parsed.get("status") == "success"
    assert "error_log" in parsed
    assert "synthesis_history" in parsed


async def test_all_skills_handle_missing_runtime_context():
    """Skills must not crash when ledger is not in runtime context."""
    # Runtime context returns None by default in test environment.
    # Each skill must create its own ledger and return gracefully.
    from src.skills.query_decision_log import query_decision_log
    from src.skills.query_energy_state import query_energy_state
    from src.skills.query_system_health import query_system_health

    # These should all return valid JSON, not raise
    r1 = await query_decision_log()
    r2 = await query_energy_state()
    r3 = await query_system_health()

    for label, result in [("decision_log", r1), ("energy_state", r2), ("health", r3)]:
        parsed = json.loads(result)
        assert "status" in parsed, f"{label}: missing status key"


# ── SKILL.md contract verification ───────────────────────────────────

def test_introspection_skills_load_in_registry():
    """All 4 introspection skills must be discoverable by SkillRegistry."""
    from src.core.skill_manager import SkillRegistry
    registry = SkillRegistry()
    skill_names = registry.get_skill_names()
    required = [
        "query_decision_log",
        "query_objective_status",
        "query_energy_state",
        "query_system_health",
    ]
    for name in required:
        assert name in skill_names, (
            f"Skill {name!r} not found in registry. "
            f"Available: {sorted(skill_names)}"
        )


def test_introspection_skills_have_schemas():
    """All 4 introspection skills must have valid JSON schemas."""
    from src.core.skill_manager import SkillRegistry
    registry = SkillRegistry()
    schemas = {s["name"]: s for s in registry.get_schemas()}
    required = [
        "query_decision_log",
        "query_objective_status",
        "query_energy_state",
        "query_system_health",
    ]
    for name in required:
        assert name in schemas, f"Schema missing for {name!r}"
        schema = schemas[name]
        assert "description" in schema, f"No description in schema for {name!r}"
        assert "parameters" in schema, f"No parameters in schema for {name!r}"


def test_research_agent_allowed_tools_updated():
    """research_agent AGENT.md must list all 4 introspection tools."""
    from src.core.agent_registry import AgentRegistry
    registry = AgentRegistry()
    agent = registry.get("research_agent")
    assert agent is not None
    required_tools = [
        "query_decision_log",
        "query_objective_status",
        "query_energy_state",
        "query_system_health",
    ]
    for tool in required_tools:
        assert tool in agent.allowed_tools, (
            f"research_agent missing allowed_tool: {tool!r}. "
            f"Current tools: {agent.allowed_tools}"
        )
