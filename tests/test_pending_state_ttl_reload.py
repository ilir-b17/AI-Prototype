import json
from pathlib import Path

import pytest

from src.memory.ledger_db import LedgerMemory


async def _create_ledger(tmp_path: Path, db_name: str) -> LedgerMemory:
    ledger = LedgerMemory(db_path=str(tmp_path / db_name))
    await ledger.initialize()
    return ledger


async def _fetch_user_ids(ledger: LedgerMemory, table_name: str) -> list[str]:
    async with ledger._lock:
        cursor = await ledger._db.execute(
            f"SELECT user_id FROM {table_name} ORDER BY user_id"
        )
        rows = await cursor.fetchall()
    return [str(row["user_id"]) for row in rows]


@pytest.mark.asyncio
async def test_load_mfa_states_filters_and_prunes_stale_rows(tmp_path: Path) -> None:
    ledger = await _create_ledger(tmp_path, "pending_mfa_ttl.db")
    try:
        async with ledger._lock:
            await ledger._db.execute(
                "INSERT INTO pending_mfa_states (user_id, tool_name, arguments_json, created_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                ("fresh-mfa", "request_core_update", json.dumps({"k": "v"})),
            )
            await ledger._db.execute(
                "INSERT INTO pending_mfa_states (user_id, tool_name, arguments_json, created_at) "
                "VALUES (?, ?, ?, datetime('now', '-2 days'))",
                ("stale-mfa", "request_core_update", json.dumps({"k": "v"})),
            )
            await ledger._db.commit()

        loaded = await ledger.load_mfa_states()

        assert set(loaded.keys()) == {"fresh-mfa"}
        assert await _fetch_user_ids(ledger, "pending_mfa_states") == ["fresh-mfa"]
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_load_hitl_states_filters_and_prunes_stale_rows(tmp_path: Path) -> None:
    ledger = await _create_ledger(tmp_path, "pending_hitl_ttl.db")
    try:
        async with ledger._lock:
            await ledger._db.execute(
                "INSERT INTO pending_hitl_states (user_id, state_json, created_at) "
                "VALUES (?, ?, datetime('now'))",
                ("fresh-hitl", json.dumps({"user_id": "fresh-hitl", "workflow": "x"})),
            )
            await ledger._db.execute(
                "INSERT INTO pending_hitl_states (user_id, state_json, created_at) "
                "VALUES (?, ?, datetime('now', '-2 days'))",
                ("stale-hitl", json.dumps({"user_id": "stale-hitl", "workflow": "x"})),
            )
            await ledger._db.commit()

        loaded = await ledger.load_hitl_states()

        assert set(loaded.keys()) == {"fresh-hitl"}
        assert await _fetch_user_ids(ledger, "pending_hitl_states") == ["fresh-hitl"]
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_load_pending_approvals_filters_and_prunes_stale_rows(tmp_path: Path) -> None:
    ledger = await _create_ledger(tmp_path, "pending_approvals_ttl.db")
    try:
        async with ledger._lock:
            await ledger._db.execute(
                "INSERT INTO pending_tool_approvals (user_id, synthesis_json, original_input, created_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                ("fresh-approval", json.dumps({"tool_name": "new_tool"}), "build tool"),
            )
            await ledger._db.execute(
                "INSERT INTO pending_tool_approvals (user_id, synthesis_json, original_input, created_at) "
                "VALUES (?, ?, ?, datetime('now', '-2 days'))",
                ("stale-approval", json.dumps({"tool_name": "old_tool"}), "stale tool"),
            )
            await ledger._db.commit()

        loaded = await ledger.load_pending_approvals()

        assert set(loaded.keys()) == {"fresh-approval"}
        assert await _fetch_user_ids(ledger, "pending_tool_approvals") == ["fresh-approval"]
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_load_mfa_states_honors_env_ttl_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PENDING_STATE_TTL_SECONDS", "3600")
    ledger = await _create_ledger(tmp_path, "pending_mfa_ttl_override.db")
    try:
        async with ledger._lock:
            await ledger._db.execute(
                "INSERT INTO pending_mfa_states (user_id, tool_name, arguments_json, created_at) "
                "VALUES (?, ?, ?, datetime('now', '-2 hours'))",
                ("mfa-two-hours-old", "request_core_update", json.dumps({"k": "v"})),
            )
            await ledger._db.commit()

        loaded = await ledger.load_mfa_states()

        assert loaded == {}
        assert await _fetch_user_ids(ledger, "pending_mfa_states") == []
    finally:
        await ledger.close()
