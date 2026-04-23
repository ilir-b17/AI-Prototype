import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator
from src.memory.ledger_db import LedgerMemory


def _candidate_payload(tool_name: str, code: str, pytest_code: str) -> dict:
    return {
        "tool_name": tool_name,
        "description": "Simple deterministic helper.",
        "code": code,
        "schema_json": (
            '{"name":"%s","description":"Simple deterministic helper.",'
            '"parameters":{"type":"object","properties":{},"required":[]}}' % tool_name
        ),
        "pytest_code": pytest_code,
        "test_manifest_json": '{"version":"synthesis_contract_v2","test_targets":["%s"],"cases":["happy_path"]}'
        % tool_name,
    }


@pytest.mark.asyncio
async def test_phase7_e2e_retry_repair_persistence_and_augmented_hitl(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase7_synthesis_e2e.db"))
    await ledger.initialize()
    try:
        initial_tool = _candidate_payload(
            "demo_tool",
            (
                "async def demo_tool() -> str:\n"
                "    return undefined_name\n"
            ),
            (
                "import pytest\n"
                "from demo_tool import demo_tool\n\n"
                "@pytest.mark.asyncio\n"
                "async def test_demo_tool():\n"
                "    value = await demo_tool()\n"
                "    assert value == 'ok'\n"
            ),
        )
        repaired_tool = _candidate_payload(
            "demo_tool",
            (
                "async def demo_tool() -> str:\n"
                "    return 'ok'\n"
            ),
            (
                "import pytest\n"
                "from demo_tool import demo_tool\n\n"
                "@pytest.mark.asyncio\n"
                "async def test_demo_tool():\n"
                "    value = await demo_tool()\n"
                "    assert value == 'ok'\n"
            ),
        )

        failed_self_test = {
            "status": "failed",
            "timed_out": False,
            "timeout_seconds": 12.0,
            "exit_code": 1,
            "duration_ms": 101,
            "passed": 0,
            "failed": 1,
            "errors": 0,
            "skipped": 0,
            "stdout": "F                                                                        [100%]",
            "stderr": "Traceback (most recent call last): NameError: name 'undefined_name' is not defined",
            "error": "Sandboxed self-test reported pytest failures.",
        }
        passed_self_test = {
            "status": "passed",
            "timed_out": False,
            "timeout_seconds": 12.0,
            "exit_code": 0,
            "duration_ms": 96,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "stdout": ".                                                                        [100%]",
            "stderr": "",
            "error": "",
        }

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger
        orchestrator.pending_tool_approval = {}

        router = MagicMock()
        router.get_system_2_available.return_value = True
        router.synthesize_tool = AsyncMock(return_value=initial_tool)
        router.repair_synthesized_tool = AsyncMock(return_value=repaired_tool)
        orchestrator.cognitive_router = router

        orchestrator._run_synthesis_self_test = AsyncMock(
            side_effect=[failed_self_test, passed_self_test]
        )

        state = {"user_id": "phase7-user", "user_input": "Please add a deterministic helper."}
        router_result = RouterResult(
            status="capability_gap",
            gap_description="Need a deterministic helper tool.",
            suggested_tool_name="demo_tool",
        )

        hitl_message = await Orchestrator.tool_synthesis_node(orchestrator, state, router_result)

        assert router.synthesize_tool.await_count == 1
        assert router.repair_synthesized_tool.await_count == 1
        assert "Cryptographic proof (SHA-256 tool+tests):" in hitl_message
        assert "attempt 2/3" in hitl_message.lower()

        pending = orchestrator.pending_tool_approval["phase7-user"]["synthesis"]
        run_id = pending.get("synthesis_run_id")
        assert run_id is not None
        assert int(pending.get("synthesis_attempts_used")) == 2
        assert int(pending.get("synthesis_max_retries")) == 3

        expected_digest = Orchestrator._compute_synthesis_proof_sha256(
            repaired_tool["code"],
            repaired_tool["pytest_code"],
        )
        assert pending.get("synthesis_proof_sha256") == expected_digest
        assert expected_digest in hitl_message

        run_row = await ledger.get_synthesis_run(int(run_id))
        assert run_row is not None
        assert run_row["status"] == "pending_approval"
        assert int(run_row["total_attempts"]) == 2
        assert int(run_row["successful_attempt"]) == 2
        assert run_row["code_sha256"] == expected_digest

        attempts = await ledger.get_synthesis_attempts(int(run_id))
        assert len(attempts) == 2
        assert attempts[0]["attempt_number"] == 1
        assert attempts[0]["status"] == "failed"
        assert "Traceback" in attempts[0]["stderr_text"]
        assert attempts[1]["attempt_number"] == 2
        assert attempts[1]["status"] == "passed"
        assert attempts[1]["code_sha256"] == expected_digest
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_phase7_synthesis_schema_migration_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "phase7_migration.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE synthesis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                gap_description TEXT NOT NULL,
                suggested_tool_name TEXT NOT NULL,
                original_input TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE synthesis_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                attempt_number INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(run_id, attempt_number)
            )
            """
        )
        conn.commit()

    ledger = LedgerMemory(db_path=str(db_path))
    await ledger.initialize()
    try:
        # Re-run migrations to prove idempotency.
        await ledger._initialize_tables()

        cursor_runs = await ledger._db.execute("PRAGMA table_info(synthesis_runs)")
        run_columns = {row["name"] for row in await cursor_runs.fetchall()}
        assert "status" in run_columns
        assert "total_attempts" in run_columns
        assert "successful_attempt" in run_columns
        assert "max_retries" in run_columns
        assert "code_sha256" in run_columns
        assert "test_summary" in run_columns
        assert "blocked_reason" in run_columns
        assert "synthesis_json" in run_columns

        cursor_attempts = await ledger._db.execute("PRAGMA table_info(synthesis_attempts)")
        attempt_columns = {row["name"] for row in await cursor_attempts.fetchall()}
        assert "phase" in attempt_columns
        assert "tool_name" in attempt_columns
        assert "status" in attempt_columns
        assert "timed_out" in attempt_columns
        assert "stdout_text" in attempt_columns
        assert "stderr_text" in attempt_columns
        assert "code_sha256" in attempt_columns
        assert "synthesis_json" in attempt_columns
    finally:
        await ledger.close()
