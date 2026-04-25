import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.orchestrator import Orchestrator


def test_intent_classifier_cache_evicts_least_recently_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENT_CLASSIFIER_CACHE_MAX_SIZE", "3")
    orchestrator = Orchestrator.__new__(Orchestrator)

    Orchestrator._cache_user_intent(orchestrator, "intent-a", "task")
    Orchestrator._cache_user_intent(orchestrator, "intent-b", "task")
    Orchestrator._cache_user_intent(orchestrator, "intent-c", "task")

    # Touch A so B becomes LRU before we insert D.
    assert Orchestrator._get_cached_user_intent(orchestrator, "intent-a") == "task"
    Orchestrator._cache_user_intent(orchestrator, "intent-d", "task")

    cache = getattr(orchestrator, "_intent_classification_cache", {})
    assert "intent-a" in cache
    assert "intent-c" in cache
    assert "intent-d" in cache
    assert "intent-b" not in cache
    assert len(cache) == 3


@pytest.mark.asyncio
async def test_consolidation_turn_counts_are_bounded_and_lru_evicted() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.save_chat_turn = AsyncMock()
    orchestrator._consolidate_memory = AsyncMock()

    def _noop_fire_and_forget(coro):
        coro.close()

    orchestrator._fire_and_forget = _noop_fire_and_forget

    for user_index in range(100):
        await Orchestrator._persist_chat_turns(
            orchestrator,
            f"user_{user_index}",
            "hello",
            "world",
        )

    # Touch user_0 so user_1 becomes the least recently touched key.
    await Orchestrator._persist_chat_turns(orchestrator, "user_0", "hello", "world")
    await Orchestrator._persist_chat_turns(orchestrator, "user_100", "hello", "world")

    counts = getattr(orchestrator, "_consolidation_turn_counts", {})
    assert len(counts) == 100
    assert "user_0" in counts
    assert "user_100" in counts
    assert "user_1" not in counts


@pytest.mark.asyncio
async def test_notify_admin_drops_oldest_when_outbound_queue_is_full(caplog: pytest.LogCaptureFixture) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.outbound_queue = asyncio.Queue(maxsize=2)

    await Orchestrator._notify_admin(orchestrator, "first")
    await Orchestrator._notify_admin(orchestrator, "second")

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(Orchestrator._notify_admin(orchestrator, "third"), timeout=0.5)

    first_out = await orchestrator.outbound_queue.get()
    second_out = await orchestrator.outbound_queue.get()

    assert first_out == "second"
    assert second_out == "third"
    assert "Dropped oldest outbound message" in caplog.text