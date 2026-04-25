import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.orchestrator import Orchestrator
from src.interfaces.telegram_bot import _drain_background_tasks


@pytest.mark.asyncio
async def test_drain_background_tasks_waits_for_inflight_writes() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._background_tasks = set()

    completed_writes: list[str] = []

    async def delayed_save_chat_turn(user_id: str, role: str, content: str) -> None:
        await asyncio.sleep(1.0)
        completed_writes.append(f"{user_id}:{role}:{content}")

    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.save_chat_turn = AsyncMock(side_effect=delayed_save_chat_turn)

    for index in range(3):
        Orchestrator._fire_and_forget(
            orchestrator,
            orchestrator.ledger_memory.save_chat_turn("user-1", "assistant", f"msg-{index}"),
        )

    await _drain_background_tasks(orchestrator, timeout=5.0)

    assert len(completed_writes) == 3


@pytest.mark.asyncio
async def test_drain_background_tasks_times_out_and_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._background_tasks = set()

    async def very_slow_save_chat_turn(_user_id: str, _role: str, _content: str) -> None:
        await asyncio.sleep(10.0)

    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.save_chat_turn = AsyncMock(side_effect=very_slow_save_chat_turn)

    Orchestrator._fire_and_forget(
        orchestrator,
        orchestrator.ledger_memory.save_chat_turn("user-1", "assistant", "slow-write"),
    )

    start = time.perf_counter()
    with caplog.at_level("WARNING", logger="src.interfaces.telegram_bot"):
        await _drain_background_tasks(orchestrator, timeout=5.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 6.5
    assert any("background task" in record.message.lower() and "did not drain" in record.message.lower() for record in caplog.records)

    remaining = list(orchestrator._background_tasks)
    for task in remaining:
        task.cancel()
    if remaining:
        await asyncio.gather(*remaining, return_exceptions=True)
