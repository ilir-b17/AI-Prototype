"""
Unit tests for the ProgressEvent and ProgressEmitter system.
No Ollama, no Telegram, no database required.
"""

import asyncio
import pytest

from src.core.progress import (
    ProgressEmitter,
    ProgressEvent,
    ProgressStage,
    get_current_emitter,
    reset_emitter,
    set_current_emitter,
)


# -- ProgressEvent factory tests --------------------------------------


def test_progress_event_supervisor_planning():
    e = ProgressEvent.supervisor_planning()
    assert e.stage == ProgressStage.SUPERVISOR
    assert "Planning" in e.message
    assert e.agent_name is None
    assert e.plan is None


def test_progress_event_supervisor_done_single():
    e = ProgressEvent.supervisor_done(["research_agent"])
    assert e.stage == ProgressStage.SUPERVISOR
    assert "1 agent" in e.message
    assert e.plan == ["research_agent"]


def test_progress_event_supervisor_done_plural():
    e = ProgressEvent.supervisor_done(["a", "b", "c"])
    assert "3 agents" in e.message
    assert e.plan == ["a", "b", "c"]


def test_progress_event_agent_start():
    e = ProgressEvent.agent_start("research_agent")
    assert e.stage == ProgressStage.AGENT_START
    assert e.agent_name == "research_agent"


def test_progress_event_agent_done():
    e = ProgressEvent.agent_done("coder_agent", 3.7)
    assert e.stage == ProgressStage.AGENT_DONE
    assert e.agent_name == "coder_agent"
    assert e.duration_seconds == pytest.approx(3.7, abs=0.001)


def test_progress_event_all_factories():
    """All factory methods must return ProgressEvent without raising."""
    factories = [
        ProgressEvent.supervisor_planning,
        ProgressEvent.supervisor_direct,
        ProgressEvent.critic_start,
        ProgressEvent.synthesis_start,
        ProgressEvent.goal_planning,
        ProgressEvent.hitl_raised,
        ProgressEvent.mfa_required,
        ProgressEvent.capability_gap,
    ]
    for factory in factories:
        event = factory()
        assert isinstance(event, ProgressEvent)
        assert event.message


# -- ProgressEmitter rate limiting tests -------------------------------


@pytest.mark.asyncio
async def test_emitter_delivers_immediately_when_budget_available():
    received = []

    async def cb(event):
        await asyncio.sleep(0)
        received.append(event)

    emitter = ProgressEmitter(cb, min_interval_seconds=0.0)
    await emitter.emit(ProgressEvent.supervisor_planning())
    assert len(received) == 1


@pytest.mark.asyncio
async def test_emitter_deduplicates_identical_messages():
    received = []

    async def cb(event):
        await asyncio.sleep(0)
        received.append(event)

    emitter = ProgressEmitter(cb, min_interval_seconds=0.0)
    event = ProgressEvent.supervisor_planning()
    await emitter.emit(event)
    await emitter.emit(event)  # same message - should be skipped
    assert len(received) == 1


@pytest.mark.asyncio
async def test_emitter_rate_limits_rapid_events():
    received = []

    async def cb(event):
        await asyncio.sleep(0)
        received.append(event)

    emitter = ProgressEmitter(cb, min_interval_seconds=10.0)
    await emitter.emit(ProgressEvent.supervisor_planning())
    # Second emit within interval - stored as pending
    await emitter.emit(ProgressEvent.agent_start("research_agent"))
    # Only the first should have been delivered
    assert len(received) == 1


@pytest.mark.asyncio
async def test_emitter_flush_delivers_pending():
    received = []

    async def cb(event):
        await asyncio.sleep(0)
        received.append(event)

    emitter = ProgressEmitter(cb, min_interval_seconds=10.0)
    await emitter.emit(ProgressEvent.supervisor_planning())
    await emitter.emit(ProgressEvent.agent_start("research_agent"))  # pending
    assert len(received) == 1  # only first delivered
    await emitter.flush()  # pending should now be delivered
    assert len(received) == 2


@pytest.mark.asyncio
async def test_emitter_emit_immediate_bypasses_rate_limit():
    received = []

    async def cb(event):
        await asyncio.sleep(0)
        received.append(event)

    emitter = ProgressEmitter(cb, min_interval_seconds=10.0)
    await emitter.emit(ProgressEvent.supervisor_planning())
    await emitter.emit_immediate(ProgressEvent.hitl_raised())
    assert len(received) == 2


@pytest.mark.asyncio
async def test_emitter_callback_error_does_not_propagate():
    call_count = 0

    async def bad_cb(event):
        nonlocal call_count
        await asyncio.sleep(0)
        call_count += 1
        raise RuntimeError("Telegram connection broken")

    emitter = ProgressEmitter(bad_cb, min_interval_seconds=0.0)
    # Should not raise
    await emitter.emit(ProgressEvent.supervisor_planning())
    await emitter.emit(ProgressEvent.critic_start())
    assert call_count >= 1  # callback was called


@pytest.mark.asyncio
async def test_emitter_agent_duration_tracking():
    async def _noop(_event):
        await asyncio.sleep(0)

    emitter = ProgressEmitter(_noop, min_interval_seconds=0.0)
    emitter.record_agent_start("test_agent")
    await asyncio.sleep(0.05)
    duration = emitter.get_agent_duration("test_agent")
    assert duration >= 0.04, f"Duration too short: {duration}"
    assert duration < 2.0, f"Duration too long: {duration}"


# -- ContextVar isolation tests ---------------------------------------


@pytest.mark.asyncio
async def test_contextvar_default_is_none():
    assert get_current_emitter() is None


@pytest.mark.asyncio
async def test_contextvar_set_and_reset():
    async def _noop(_event):
        await asyncio.sleep(0)

    emitter = ProgressEmitter(_noop)
    token = set_current_emitter(emitter)
    assert get_current_emitter() is emitter
    reset_emitter(token)
    assert get_current_emitter() is None


@pytest.mark.asyncio
async def test_contextvar_task_isolation():
    """Two concurrent tasks must each see their own emitter."""

    async def _noop(_event):
        await asyncio.sleep(0)

    emitter_a = ProgressEmitter(_noop)
    emitter_b = ProgressEmitter(_noop)
    results = {}

    async def task_a():
        token = set_current_emitter(emitter_a)
        await asyncio.sleep(0.05)
        results["a"] = get_current_emitter()
        reset_emitter(token)

    async def task_b():
        token = set_current_emitter(emitter_b)
        await asyncio.sleep(0.05)
        results["b"] = get_current_emitter()
        reset_emitter(token)

    await asyncio.gather(task_a(), task_b())
    assert results["a"] is emitter_a
    assert results["b"] is emitter_b
    assert results["a"] is not results["b"]


# -- process_message integration (no network required) ----------------


@pytest.mark.asyncio
async def test_process_message_accepts_progress_callback():
    """process_message must accept progress_callback without raising."""
    from unittest.mock import AsyncMock, MagicMock, patch
    import conftest

    events = []

    async def collect(event):
        await asyncio.sleep(0)
        events.append(event)

    from src.core.orchestrator import Orchestrator

    class _DummyLock:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return None

    orch = object.__new__(Orchestrator)
    orch._ready = asyncio.Event()
    orch._ready.set()
    orch.pending_mfa = {}
    orch.pending_hitl_state = {}
    orch.pending_tool_approval = {}
    orch._energy_budget_lock = asyncio.Lock()
    orch._energy_budget = 100
    orch.synthesis_pipeline = MagicMock()
    orch.synthesis_pipeline.try_resume_tool_approval = AsyncMock(return_value=None)

    with patch.object(
        Orchestrator, "_try_resume_mfa", new=AsyncMock(return_value=None)
    ), patch.object(
        Orchestrator,
        "_get_user_lock",
        new=AsyncMock(return_value=_DummyLock()),
    ), patch.object(
        Orchestrator,
        "_run_user_turn_locked",
        new=AsyncMock(return_value="Fast answer"),
    ), patch.object(
        Orchestrator,
        "process_message",
        new=conftest._orig_process_message,
    ):
        result = await orch.process_message(
            "hi", "test_user", progress_callback=collect
        )

    assert result == "Fast answer"
    assert get_current_emitter() is None
    # Fast-path responses do not need to emit events - just verify no crash
    print("process_message progress_callback: OK")
