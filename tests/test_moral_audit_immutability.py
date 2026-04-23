import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import RouterResult
from src.core.moral_ledger import MORAL_DIMENSIONS, MORAL_RUBRIC_VERSION
from src.core.orchestrator import Orchestrator
from src.core.state_model import AgentState
from src.memory.ledger_db import LedgerMemory


def _valid_system2_moral_decision_json() -> str:
    return json.dumps(
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": dict.fromkeys(MORAL_DIMENSIONS, 4),
            "reasoning": "Tier 1-3 checks pass with no material policy violations.",
            "is_approved": True,
            "decision_mode": "system2_audit",
            "bypass_reason": "",
        }
    )


@pytest.mark.asyncio
async def test_system2_moral_evaluation_is_logged_and_sent_through_redaction_boundary(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "moral_audit_integration.db"))
    await ledger.initialize()
    try:
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger
        orchestrator.charter_text = Path("charter.md").read_text(encoding="utf-8")
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.get_system_2_available.return_value = True
        orchestrator.cognitive_router.route_to_system_2 = AsyncMock(
            return_value=RouterResult(status="ok", content=_valid_system2_moral_decision_json())
        )
        orchestrator._route_to_system_1 = AsyncMock(
            return_value=RouterResult(status="ok", content="unused")
        )

        state = AgentState.new(
            "audit-user",
            "Please review this response with my email sensitive@example.com and path C:\\Users\\Secret\\notes.txt",
        ).to_dict()
        state["critic_instructions"] = "force_system2_review"
        worker_output = (
            "This is a deliberately long worker output for critic evaluation. " * 8
            + "Contains telemetry: sensitive@example.com and local path C:\\Users\\Secret\\notes.txt"
        )
        state["worker_outputs"] = {"research_agent": worker_output}
        state["final_response"] = worker_output

        result = await Orchestrator.critic_node(orchestrator, state)

        assert result["critic_feedback"] == "PASS"
        assert result["moral_audit_mode"] == "system2_audit"
        assert result["moral_decision"]["rubric_version"] == MORAL_RUBRIC_VERSION

        logs = await ledger.get_moral_audit_logs(user_id="audit-user", limit=5)
        assert len(logs) == 1
        assert logs[0]["audit_mode"] == "system2_audit"
        assert logs[0]["moral_decision"]["is_approved"] is True

        args, kwargs = orchestrator.cognitive_router.route_to_system_2.await_args
        forwarded_messages = args[0]
        forwarded_text = "\n".join(str(message.get("content") or "") for message in forwarded_messages)
        assert "sensitive@example.com" not in forwarded_text
        assert "C:\\Users\\Secret\\notes.txt" not in forwarded_text
        assert kwargs["allowed_tools"] == []
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_moral_audit_log_sql_triggers_block_raw_update_and_delete(tmp_path: Path):
    db_path = tmp_path / "moral_audit_immutable.db"
    ledger = LedgerMemory(db_path=str(db_path))
    await ledger.initialize()
    try:
        entry_id = await ledger.append_moral_audit_log(
            user_id="security-user",
            audit_mode="system2_audit",
            audit_trace="initial_insert",
            critic_feedback="PASS",
            moral_decision={
                "rubric_version": MORAL_RUBRIC_VERSION,
                "scores": dict.fromkeys(MORAL_DIMENSIONS, 4),
                "reasoning": "Stored for immutability test",
                "is_approved": True,
                "decision_mode": "system2_audit",
                "bypass_reason": "",
                "validation_error": "",
            },
            request_redacted="request",
            output_redacted="output",
        )
    finally:
        await ledger.close()

    with sqlite3.connect(db_path) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="Update not allowed on moral_audit_log"):
            conn.execute(
                "UPDATE moral_audit_log SET audit_mode = ? WHERE id = ?",
                ("tampered", int(entry_id)),
            )
            conn.commit()

        with pytest.raises(sqlite3.DatabaseError, match="Delete not allowed on moral_audit_log"):
            conn.execute(
                "DELETE FROM moral_audit_log WHERE id = ?",
                (int(entry_id),),
            )
            conn.commit()

        row = conn.execute(
            "SELECT audit_mode FROM moral_audit_log WHERE id = ?",
            (int(entry_id),),
        ).fetchone()
        assert row is not None
        assert str(row[0]) == "system2_audit"