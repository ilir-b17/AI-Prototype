"""
Integration tests for the session context system.
Tests are async and use real LedgerMemory with a temp DB.
"""
import os

import pytest

from src.memory.ledger_db import LedgerMemory


@pytest.fixture
async def ledger(tmp_path):
    db_path = os.path.join(str(tmp_path), "session_context.db")
    db = LedgerMemory(db_path)
    await db.initialize()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_session_lifecycle(ledger):
    """Create, activate, use, and deactivate a session."""
    sid = await ledger.create_session("Test Project", "A test")
    assert sid > 0

    row = await ledger.activate_session(sid)
    assert row is not None
    assert row["name"] == "Test Project"

    active = await ledger.get_active_session()
    assert active is not None
    assert active["id"] == sid

    await ledger.deactivate_all_sessions()
    active_after = await ledger.get_active_session()
    assert active_after is None


@pytest.mark.asyncio
async def test_session_scoped_chat_history(ledger):
    """chat_history rows are scoped by session_id."""
    sid1 = await ledger.create_session("Project A")
    sid2 = await ledger.create_session("Project B")

    await ledger.save_chat_turn_with_session("u1", "user", "Hello A", session_id=sid1)
    await ledger.save_chat_turn_with_session("u1", "assistant", "Hi A", session_id=sid1)
    await ledger.save_chat_turn_with_session("u1", "user", "Hello B", session_id=sid2)
    await ledger.save_chat_turn_with_session("u1", "assistant", "Hi B", session_id=sid2)

    history_a = await ledger.get_session_chat_history("u1", sid1)
    assert len(history_a) == 2
    assert history_a[0]["content"] == "Hello A"

    history_b = await ledger.get_session_chat_history("u1", sid2)
    assert len(history_b) == 2
    assert history_b[0]["content"] == "Hello B"


@pytest.mark.asyncio
async def test_session_turn_count(ledger):
    """Turn count increments correctly."""
    sid = await ledger.create_session("Counter Test")
    await ledger.increment_session_turn_count(sid)
    await ledger.increment_session_turn_count(sid)
    row = await ledger.get_session(sid)
    assert row["turn_count"] == 2


@pytest.mark.asyncio
async def test_session_list_ordering(ledger):
    """Active session appears first in list."""
    sid1 = await ledger.create_session("First")
    sid2 = await ledger.create_session("Second")
    _ = sid1
    await ledger.activate_session(sid2)
    sessions = await ledger.list_sessions()
    assert sessions[0]["id"] == sid2
    assert sessions[0]["is_active"] == 1


@pytest.mark.asyncio
async def test_no_session_fallback_chat_history(ledger):
    """Global get_chat_history still works when no session is set."""
    await ledger.save_chat_turn_with_session("u2", "user", "Global message", session_id=None)
    history = await ledger.get_chat_history("u2", limit=10)
    assert len(history) == 1
    assert history[0]["content"] == "Global message"


def test_where_clause_builder():
    """_build_where_clause produces correct ChromaDB filter syntax."""
    from src.skills.search_archival_memory import _build_where_clause

    assert _build_where_clause(None, None) is None

    single = _build_where_clause(3, None)
    assert single == {"session_id": {"$eq": 3}}

    both = _build_where_clause(3, 7)
    assert both == {"$and": [
        {"session_id": {"$eq": 3}},
        {"epic_id": {"$eq": 7}},
    ]}
