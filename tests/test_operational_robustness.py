import asyncio
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.core.orchestrator as orchestrator_module
from src.core.llm_router import CognitiveRouter
from src.core.orchestrator import HEARTBEAT_TASK_PREFIX_FMT, Orchestrator
from src.core.skill_manager import SkillRegistry
from src.core.state_model import AgentState
from src.memory.ledger_db import LedgerMemory


@pytest.mark.asyncio
async def test_groq_cooldown_restore_failure_notifies_admin() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.get_system_state = AsyncMock(side_effect=RuntimeError("db unavailable"))
    orchestrator.outbound_queue = asyncio.Queue()

    await Orchestrator._restore_persisted_groq_cooldown(orchestrator)

    notification = await orchestrator.outbound_queue.get()
    assert "WARNING: Groq cooldown restoration failed" in notification
    assert "db unavailable" in notification


def test_groq_rate_limit_invokes_persistence_callback_without_router_task() -> None:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._system2_cooldown_until = 0.0
    router._persist_cooldown_cb = MagicMock()
    router._format_cooldown_message = MagicMock(return_value="Rate limited. Retry in 1m0s.")

    result = CognitiveRouter._handle_groq_rate_limit(router, "rate limit: try again in 1m0s")

    assert result is not None
    assert result.status == "ok"
    router._persist_cooldown_cb.assert_called_once()


def test_skill_registry_records_load_failures_and_logs_summary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good_dir = tmp_path / "good_skill"
    good_dir.mkdir()
    (good_dir / "SKILL.md").write_text(
        "```json\n"
        '{"name":"good_skill","description":"Good skill","parameters":{"type":"object","properties":{},"required":[]}}'
        "\n```\n",
        encoding="utf-8",
    )
    (good_dir / "__init__.py").write_text("def good_skill():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "broken_skill").mkdir()

    with caplog.at_level(logging.INFO):
        registry = SkillRegistry(skills_dir=str(tmp_path))

    assert registry.get_skill_names() == ["good_skill"]
    assert registry.get_load_errors()[0][0] == "broken_skill"
    assert "1 skills loaded, 1 failed: [broken_skill]." in caplog.text


@pytest.mark.asyncio
async def test_deferred_heartbeat_task_with_unmet_dependency_is_not_selected(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "heartbeat_deps.db"))
    await ledger.initialize()
    try:
        dependency_id = await ledger.add_objective(tier="Task", title="Dependency")
        blocked_id = await ledger.add_objective(
            tier="Task",
            title="Blocked deferred task",
            depends_on_ids=[dependency_id],
            estimated_energy=1,
            priority=1,
        )
        ready_id = await ledger.add_objective(
            tier="Task",
            title="Ready task",
            estimated_energy=1,
            priority=2,
        )
        await ledger.update_objective_status(blocked_id, "deferred_due_to_energy")
        async with ledger._lock:
            await ledger._db.execute(
                "UPDATE objective_backlog SET next_eligible_at = datetime('now', '-1 minute') WHERE id = ?",
                (blocked_id,),
            )
            await ledger._db.commit()

        unresolved = await ledger.get_tasks_with_unresolved_dependencies()
        assert {row["id"] for row in unresolved} == {blocked_id}

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger

        candidates = await Orchestrator._select_executable_heartbeat_tasks(orchestrator)
        candidate_ids = {entry["task"]["id"] for entry in candidates}

        assert blocked_id not in candidate_ids
        assert ready_id in candidate_ids
    finally:
        await ledger.close()


def test_charter_tier_extraction_reports_malformed_xml() -> None:
    malformed = "<Identity_Charter><Tier_1_Axioms><Directive>broken</Tier_1_Axioms>"

    result = Orchestrator._extract_charter_tier_block_from_text(malformed, "Tier_1_Axioms")

    assert result.startswith("[MALFORMED_CHARTER_XML:")


def test_charter_tier_extraction_uses_cached_parse_after_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubVectorMemory:
        def __init__(self, persist_dir: str):
            self.persist_dir = persist_dir

        def close(self) -> None:
            return None

    monkeypatch.setattr(orchestrator_module, "VectorMemory", _StubVectorMemory)

    orchestrator = Orchestrator(
        vector_db_path=str(tmp_path / "chroma"),
        ledger_db_path=str(tmp_path / "ledger.db"),
        core_memory_path=str(tmp_path / "core_memory.json"),
    )
    try:
        baseline_tier = orchestrator._extract_charter_tier_block("Tier_1_Axioms")
        assert "Do No Harm" in baseline_tier

        def _unexpected_parse(*args, **kwargs):
            raise AssertionError("ET.fromstring must not be called after Orchestrator.__init__")

        monkeypatch.setattr(orchestrator_module.ET, "fromstring", _unexpected_parse)

        cached_tier = orchestrator._extract_charter_tier_block("Tier_1_Axioms")
        assert cached_tier == baseline_tier
    finally:
        orchestrator.close()


def test_malformed_charter_fails_closed_without_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MFA_PASSPHRASE", "correct horse battery staple")
    monkeypatch.delenv("MFA_TOTP_SECRET", raising=False)
    monkeypatch.delenv("ALLOW_MISSING_CHARTER", raising=False)
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.charter_text = "<Identity_Charter><Tier_1_Axioms><Directive>broken</Tier_1_Axioms>"

    with pytest.raises(RuntimeError, match="malformed"):
        Orchestrator._enforce_charter_policy(orchestrator)


def test_malformed_charter_allowed_with_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MFA_PASSPHRASE", "correct horse battery staple")
    monkeypatch.delenv("MFA_TOTP_SECRET", raising=False)
    monkeypatch.setenv("ALLOW_MISSING_CHARTER", "true")
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.charter_text = "<Identity_Charter><Tier_1_Axioms><Directive>broken</Tier_1_Axioms>"

    Orchestrator._enforce_charter_policy(orchestrator)

    assert orchestrator.charter_text == Orchestrator._CHARTER_FALLBACK


def test_heartbeat_prefix_round_trips_task_id() -> None:
    prompt = Orchestrator._build_heartbeat_execution_prompt({"id": 42, "title": "Check status"})

    assert HEARTBEAT_TASK_PREFIX_FMT.format(task_id=42) in prompt
    assert Orchestrator._extract_heartbeat_task_id(prompt) == 42


@pytest.mark.asyncio
async def test_fallback_charter_disables_critic_and_moral_audit(caplog: pytest.LogCaptureFixture) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.charter_text = Orchestrator._CHARTER_FALLBACK
    orchestrator.cognitive_router = MagicMock()
    orchestrator._route_to_system_1 = AsyncMock()
    orchestrator._route_to_system_2_redacted = AsyncMock()
    orchestrator._persist_moral_audit_log = AsyncMock()

    state = AgentState.new("user-1", "Review this").to_dict()
    state["current_plan"] = [{"agent": "coder_agent", "task": "Draft", "reason": "Test"}]
    state["worker_outputs"] = {"coder_agent": "Long enough output. " * 40}

    with caplog.at_level(logging.WARNING):
        result = await Orchestrator.critic_node(orchestrator, state)

    assert result["critic_feedback"] == "PASS"
    assert result["moral_audit_mode"] == "disabled_fallback_charter"
    assert result["final_response"].startswith("Long enough output")
    assert "Critic disabled because fallback charter is active" in caplog.text
    orchestrator._route_to_system_1.assert_not_awaited()
    orchestrator._route_to_system_2_redacted.assert_not_awaited()
    orchestrator._persist_moral_audit_log.assert_not_awaited()


@pytest.mark.asyncio
async def test_long_tool_synthesis_hitl_payload_writes_code_attachment(tmp_path: Path) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.pending_tool_approval = {}
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.save_pending_approval = AsyncMock()
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.get_system_2_available.return_value = True
    orchestrator._update_synthesis_run_status_if_supported = AsyncMock()
    long_code = "def synthesized_tool():\n    return 'ok'\n" + "# filler\n" * 600
    proof = Orchestrator._compute_synthesis_proof_sha256(long_code, "def test_ok():\n    assert True\n")
    orchestrator._execute_synthesis_repair_loop = AsyncMock(
        return_value={
            "status": "passed",
            "synthesis": {
                "tool_name": "long_tool",
                "description": "A deliberately long tool.",
                "code": long_code,
                "pytest_code": "def test_ok():\n    assert True\n",
                "schema_json": {"name": "long_tool", "description": "A deliberately long tool."},
            },
            "proof_sha256": proof,
            "attempts_used": 1,
            "max_retries": 3,
            "test_summary": "1 passed",
            "run_id": 99,
        }
    )
    state = {"user_id": "user-1", "user_input": "Need a long helper"}
    router_result = MagicMock(gap_description="Need long code", suggested_tool_name="long_tool")

    payload = await Orchestrator.tool_synthesis_node(orchestrator, state, router_result)

    try:
        assert isinstance(payload, dict)
        assert len(payload["text"]) <= 3500
        assert proof in payload["text"]
        assert "Code attached as Telegram document" in payload["text"]
        attachment_path = Path(payload["document_path"])
        assert attachment_path.exists()
        assert attachment_path.read_text(encoding="utf-8").startswith("def synthesized_tool")
    finally:
        document_path = payload.get("document_path") if isinstance(payload, dict) else None
        if document_path and os.path.exists(document_path):
            os.remove(document_path)
