"""
Unit tests for the Telegram streaming helpers.
No Telegram API calls - all dependencies are mocked.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.interfaces.streaming import DeferredStatusMessage, StatusFormatter
from src.core.progress import ProgressEvent, ProgressStage


# -- StatusFormatter tests --------------------------------------------


def test_formatter_initial_render():
    f = StatusFormatter()
    t = f.render()
    assert "AIDEN" in t
    assert "Processing" in t


def test_formatter_supervisor_planning():
    f = StatusFormatter()
    f.apply(ProgressEvent.supervisor_planning())
    t = f.render()
    assert "Planning" in t


def test_formatter_supervisor_done_shows_agents():
    f = StatusFormatter()
    f.apply(ProgressEvent.supervisor_done(["research_agent", "coder_agent"]))
    t = f.render()
    assert "research_agent" in t
    assert "coder_agent" in t


def test_formatter_agent_state_transitions():
    f = StatusFormatter()
    f.apply(ProgressEvent.supervisor_done(["research_agent"]))
    f.apply(ProgressEvent.agent_start("research_agent"))
    t_running = f.render()
    assert "Running" in t_running

    f.apply(ProgressEvent.agent_done("research_agent", 2.5))
    t_done = f.render()
    assert "Done" in t_done
    assert "2.5s" in t_done


def test_formatter_critic_event():
    f = StatusFormatter()
    f.apply(ProgressEvent.critic_start())
    t = f.render()
    assert "Evaluating" in t


def test_formatter_hitl_event():
    f = StatusFormatter()
    f.apply(ProgressEvent.hitl_raised())
    t = f.render()
    assert "guidance" in t.lower()


def test_formatter_output_under_telegram_limit():
    f = StatusFormatter()
    # Simulate many agents with long names
    agents = [f"very_long_agent_name_number_{i}" for i in range(20)]
    f.apply(ProgressEvent.supervisor_done(agents))
    for agent in agents:
        f.apply(ProgressEvent.agent_start(agent))
    t = f.render()
    assert len(t) <= 4096, f"Exceeded Telegram limit: {len(t)} chars"


def test_formatter_unknown_agent_added_mid_execution():
    """Agent not in original plan should still appear if it emits events."""
    f = StatusFormatter()
    f.apply(ProgressEvent.supervisor_done(["research_agent"]))
    f.apply(ProgressEvent.agent_start("dynamic_agent"))  # not in plan
    t = f.render()
    assert "dynamic_agent" in t


# -- DeferredStatusMessage tests --------------------------------------


def _make_mock_message():
    """Build a mock Telegram Message with reply_text and edit_text."""
    sent_msg = MagicMock()
    sent_msg.edit_text = AsyncMock()

    msg = MagicMock()
    msg.reply_text = AsyncMock(return_value=sent_msg)
    return msg, sent_msg


@pytest.mark.asyncio
async def test_deferred_message_does_not_send_if_fast():
    """If finalize() is called before delay, no status message is sent."""
    mock_msg, _ = _make_mock_message()
    dsm = DeferredStatusMessage(mock_msg, delay_seconds=10.0)
    await dsm.start()
    delivered = await dsm.finalize("Final answer")
    assert not delivered, "Should return False - status message never sent"
    mock_msg.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_deferred_message_sends_after_delay():
    """Status message is sent after delay_seconds elapses."""
    mock_msg, _ = _make_mock_message()
    dsm = DeferredStatusMessage(mock_msg, delay_seconds=0.05)
    await dsm.start()
    await asyncio.sleep(0.15)  # wait for delay to fire
    assert dsm._status_msg is not None
    mock_msg.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_deferred_message_finalize_edits_status():
    """finalize() edits the status message with the final response."""
    mock_msg, sent_msg = _make_mock_message()
    dsm = DeferredStatusMessage(mock_msg, delay_seconds=0.05)
    await dsm.start()
    await asyncio.sleep(0.15)

    delivered = await dsm.finalize("Here is your answer!")
    assert delivered is True
    sent_msg.edit_text.assert_called_with("Here is your answer!")


@pytest.mark.asyncio
async def test_deferred_message_update_no_op_before_send():
    """update() before status message is sent does nothing."""
    mock_msg, sent_msg = _make_mock_message()
    dsm = DeferredStatusMessage(mock_msg, delay_seconds=10.0)
    await dsm.start()
    await dsm.update("Some update text")
    sent_msg.edit_text.assert_not_called()


@pytest.mark.asyncio
async def test_deferred_message_update_skips_identical_text():
    """update() with unchanged text does not call edit_text again."""
    mock_msg, sent_msg = _make_mock_message()
    dsm = DeferredStatusMessage(mock_msg, delay_seconds=0.05)
    await dsm.start()
    await asyncio.sleep(0.15)

    await dsm.update("Same text")
    await dsm.update("Same text")  # duplicate - should be skipped
    # edit_text called once for first update, skipped for duplicate
    assert sent_msg.edit_text.call_count == 1


@pytest.mark.asyncio
async def test_deferred_message_edit_failure_is_non_fatal():
    """edit_text failure during update() must not propagate."""
    mock_msg, sent_msg = _make_mock_message()
    sent_msg.edit_text = AsyncMock(side_effect=Exception("Telegram error"))

    dsm = DeferredStatusMessage(mock_msg, delay_seconds=0.05)
    await dsm.start()
    await asyncio.sleep(0.15)

    # Should not raise
    await dsm.update("Some text")
    # finalize should attempt edit even if update failed
    result = await dsm.finalize("Final response")
    # result may be False since edit fails, but no exception raised
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_streaming_end_to_end_simulation():
    """Simulate a complete streaming turn: plan -> agents -> critic -> finalize."""
    mock_msg, sent_msg = _make_mock_message()
    dsm = DeferredStatusMessage(mock_msg, delay_seconds=0.05)
    formatter = StatusFormatter()

    await dsm.start()
    await asyncio.sleep(0.1)  # let status message appear

    # Simulate pipeline events
    events = [
        ProgressEvent.supervisor_planning(),
        ProgressEvent.supervisor_done(["research_agent", "coder_agent"]),
        ProgressEvent.agent_start("research_agent"),
        ProgressEvent.agent_done("research_agent", 1.8),
        ProgressEvent.agent_start("coder_agent"),
        ProgressEvent.agent_done("coder_agent", 3.2),
        ProgressEvent.critic_start(),
    ]
    for event in events:
        formatter.apply(event)
        await dsm.update(formatter.render())

    delivered = await dsm.finalize("The answer to your question is 42.")
    assert delivered is True

    # Final edit should contain the answer
    last_call_args = sent_msg.edit_text.call_args_list[-1]
    assert "42" in str(last_call_args)
