import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator


def _close_fire_and_forget(coro) -> None:
    coro.close()


def _build_lockout_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}
    orchestrator._synthesis_in_progress = {}
    orchestrator._try_resolve_capability_gap_locally = AsyncMock(return_value=None)
    orchestrator._fire_and_forget = _close_fire_and_forget
    orchestrator.outbound_queue = asyncio.Queue()
    return orchestrator


@pytest.mark.asyncio
async def test_stale_lockout_entry_allows_new_synthesis_run(monkeypatch: pytest.MonkeyPatch) -> None:
    user_id = "user-1"
    orchestrator = _build_lockout_orchestrator()
    orchestrator._synthesis_in_progress[user_id] = 1_300.0
    monkeypatch.setattr("src.core.orchestrator._SYNTHESIS_LOCKOUT_TTL_SECONDS", 600)
    monkeypatch.setattr("src.core.orchestrator.time.time", lambda: 2_000.0)

    response = await Orchestrator._handle_blocked_result(
        orchestrator,
        RouterResult(
            status="capability_gap",
            gap_description="Need new capability",
            suggested_tool_name="new_tool",
        ),
        user_id,
        {"user_input": "Please do something new"},
    )

    assert "already in progress" not in response.lower()
    assert orchestrator._synthesis_in_progress[user_id] == pytest.approx(2_000.0)


@pytest.mark.asyncio
async def test_fresh_lockout_entry_blocks_new_synthesis_run(monkeypatch: pytest.MonkeyPatch) -> None:
    user_id = "user-2"
    orchestrator = _build_lockout_orchestrator()
    orchestrator._fire_and_forget = MagicMock()
    monkeypatch.setattr("src.core.orchestrator._SYNTHESIS_LOCKOUT_TTL_SECONDS", 600)
    monkeypatch.setattr("src.core.orchestrator.time.time", lambda: 2_500.0)
    orchestrator._synthesis_in_progress[user_id] = 2_500.0

    response = await Orchestrator._handle_blocked_result(
        orchestrator,
        RouterResult(
            status="capability_gap",
            gap_description="Need new capability",
            suggested_tool_name="new_tool",
        ),
        user_id,
        {"user_input": "Please do something new"},
    )

    assert "already in progress" in response.lower()
    assert orchestrator._synthesis_in_progress[user_id] == pytest.approx(2_500.0)
    orchestrator._fire_and_forget.assert_not_called()


@pytest.mark.asyncio
async def test_hung_synthesis_is_aborted_by_outer_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    user_id = "user-3"
    orchestrator = _build_lockout_orchestrator()
    orchestrator._synthesis_in_progress[user_id] = time.time()
    monkeypatch.setenv("SYNTHESIS_LOCKOUT_TTL_SECONDS", "1")
    monkeypatch.setattr("src.core.orchestrator._SYNTHESIS_LOCKOUT_TTL_SECONDS", 1)

    async def _hung_tool_synthesis_node(_state, _result):
        await asyncio.sleep(60)
        return "never reached"

    orchestrator.tool_synthesis_node = AsyncMock(side_effect=_hung_tool_synthesis_node)

    started = time.perf_counter()
    await Orchestrator._async_tool_synthesis(
        orchestrator,
        user_id,
        RouterResult(
            status="capability_gap",
            gap_description="Need new capability",
            suggested_tool_name="new_tool",
        ),
        {"user_id": user_id, "user_input": "Please do something new"},
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 2.5
    assert user_id not in orchestrator._synthesis_in_progress

    critical_message = await asyncio.wait_for(orchestrator.outbound_queue.get(), timeout=1.0)
    assert "critical" in critical_message.lower()
    assert "timed out" in critical_message.lower()
