"""
Telegram streaming helpers for AIDEN.

DeferredStatusMessage — sends a status message only if the response
takes longer than delay_seconds. Edits it with the final response.
If the response arrives before the delay, no message is sent.

StatusFormatter — builds human-readable progressive status text from
ProgressEvents. Tracks per-agent state across multiple events.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from src.core.progress import ProgressEvent, ProgressStage

logger = logging.getLogger(__name__)

# Emoji indicators for agent states
_EMOJI_WAITING = "⏳"
_EMOJI_RUNNING = "🔄"
_EMOJI_DONE = "✅"
_EMOJI_TOOL = "🔧"
_EMOJI_ERROR = "❌"

# Telegram message length limit
_TELEGRAM_MAX_CHARS = 4096


class StatusFormatter:
    """Builds progressive status message text from a stream of ProgressEvents.

    Maintains per-agent state so the rendered message always shows the
    current status of every agent in the plan simultaneously.

    Example output:
        ⏳ AIDEN — Processing

        📋 Plan: research_agent → coder_agent

        • research_agent: ✅ Done (2.1s)
        • coder_agent:    🔄 Running...

        ⏱ 8s
    """

    def __init__(self) -> None:
        self._start_time: float = time.monotonic()
        self._supervisor_line: str = ""
        self._plan: List[str] = []
        self._agent_lines: Dict[str, str] = {}
        self._footer_line: str = ""

    def apply(self, event: ProgressEvent) -> None:
        """Update internal state from an event."""
        stage = event.stage

        if stage == ProgressStage.SUPERVISOR:
            self._supervisor_line = event.message
            if event.plan:
                self._plan = list(event.plan)
                for agent in self._plan:
                    if agent not in self._agent_lines:
                        self._agent_lines[agent] = (
                            f"{_EMOJI_WAITING} Waiting..."
                        )
            elif not event.plan and "Plan ready" not in event.message:
                # Direct response — clear any stale agent lines
                self._agent_lines.clear()

        elif stage == ProgressStage.AGENT_START:
            name = event.agent_name or ""
            self._agent_lines[name] = f"{_EMOJI_RUNNING} Running..."
            if name not in self._plan:
                self._plan.append(name)

        elif stage == ProgressStage.AGENT_TOOL:
            name = event.agent_name or ""
            tool = event.tool_name or "tool"
            self._agent_lines[name] = f"{_EMOJI_TOOL} {tool}..."

        elif stage == ProgressStage.AGENT_DONE:
            name = event.agent_name or ""
            dur = event.duration_seconds
            dur_str = f"{dur:.1f}s" if dur is not None else ""
            self._agent_lines[name] = (
                f"{_EMOJI_DONE} Done ({dur_str})" if dur_str
                else f"{_EMOJI_DONE} Done"
            )

        elif stage == ProgressStage.CRITIC:
            self._footer_line = "⚖️  Evaluating output..."

        elif stage == ProgressStage.SYNTHESIS:
            self._footer_line = "✍️  Synthesising response..."

        elif stage == ProgressStage.GOAL_PLANNING:
            self._supervisor_line = "🗓  Decomposing goal into tasks..."

        elif stage == ProgressStage.HITL:
            self._footer_line = "⚠️  Awaiting admin guidance..."

        elif stage == ProgressStage.MFA:
            self._footer_line = "🔐  MFA verification required..."

        elif stage == ProgressStage.CAPABILITY_GAP:
            self._footer_line = "⚙️  Synthesising new capability..."

    def render(self) -> str:
        """Return the current status as a Telegram-safe plain-text string."""
        parts: List[str] = ["⏳ AIDEN — Processing", ""]

        if self._supervisor_line:
            parts.append(f"📋 {self._supervisor_line}")

        # Show agents in plan order, then any agents not in original plan
        display_order = list(self._plan)
        for name in self._agent_lines:
            if name not in display_order:
                display_order.append(name)

        if display_order:
            for name in display_order:
                status = self._agent_lines.get(name, f"{_EMOJI_WAITING} Waiting...")
                # Truncate long agent names so lines stay readable
                display_name = name[:24] + "…" if len(name) > 24 else name
                parts.append(f"  • {display_name}: {status}")

        if self._footer_line:
            parts.append("")
            parts.append(self._footer_line)

        elapsed = int(time.monotonic() - self._start_time)
        if elapsed >= 1:
            parts.append(f"\n⏱ {elapsed}s")

        text = "\n".join(parts)
        # Safety truncation (status text should always be well under limit)
        return text[:_TELEGRAM_MAX_CHARS]


class DeferredStatusMessage:
    """Sends a Telegram status message only if the response takes too long.

    If process_message() returns within delay_seconds, no status message
    is sent and finalize() returns False — the caller sends the response
    normally via reply_text().

    If process_message() takes longer than delay_seconds, an initial
    status message is sent, then updated via edit_text() as events arrive.
    finalize() edits the status message with the final response and
    returns True.

    Usage:
        dsm = DeferredStatusMessage(update.message)
        formatter = StatusFormatter()
        await dsm.start()

        async def on_progress(event):
            formatter.apply(event)
            await dsm.update(formatter.render())

        response = await orchestrator.process_message(
            message, user_id, progress_callback=on_progress
        )

        delivered = await dsm.finalize(response)
        if not delivered:
            await update.message.reply_text(response[:4096])
    """

    def __init__(
        self,
        telegram_message,
        delay_seconds: float = 2.5,
        initial_text: str = "⏳ AIDEN — Processing...",
    ) -> None:
        self._message = telegram_message
        self._delay = delay_seconds
        self._initial_text = initial_text
        self._status_msg = None
        self._cancelled: bool = False
        self._send_task: Optional[asyncio.Task] = None
        self._last_text: str = ""
        self._edit_count: int = 0

    async def start(self) -> None:
        """Begin the deferred send timer."""
        self._send_task = asyncio.create_task(
            self._deferred_send(),
            name="aiden_deferred_status",
        )

    async def update(self, text: str) -> None:
        """Edit the status message with new text.

        No-op if the status message has not been sent yet or text is unchanged.
        Telegram returns a 400 if the text is identical to the current content
        — we prevent that with the _last_text check.
        """
        if self._cancelled or self._status_msg is None:
            return
        if not text or text == self._last_text:
            return
        truncated = text[:_TELEGRAM_MAX_CHARS]
        try:
            await self._status_msg.edit_text(truncated)
            self._last_text = truncated
            self._edit_count += 1
        except Exception as exc:
            # edit_text can fail if the message was deleted or if Telegram
            # returns "message is not modified" (400). Both are non-fatal.
            logger.debug("Status message edit failed (non-fatal): %s", exc)

    async def finalize(self, final_response: str) -> bool:
        """Deliver the final response and cancel the pending timer.

        Returns True  — response delivered by editing the status message.
        Returns False — timer had not fired; caller must reply_text() normally.
        """
        self._cancelled = True
        if self._send_task is not None and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass

        if self._status_msg is None:
            # Timer never fired — caller sends response normally
            return False

        truncated = final_response[:_TELEGRAM_MAX_CHARS]
        if truncated == self._last_text:
            # Already showing this text (unlikely but possible)
            return True
        try:
            await self._status_msg.edit_text(truncated)
            return True
        except Exception as exc:
            logger.warning(
                "Could not edit status message with final response "
                "(will send as new message): %s",
                exc,
            )
            return False

    async def cancel(self) -> None:
        """Cancel the deferred timer and delete the status bubble if shown.

        After calling cancel(), the caller is responsible for sending the
        final response via reply_text() so it appears as a proper new message.
        """
        self._cancelled = True
        if self._send_task is not None and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        if self._status_msg is not None:
            try:
                await self._status_msg.delete()
            except Exception as exc:
                logger.debug("Status message delete failed (non-fatal): %s", exc)
            finally:
                self._status_msg = None

    async def _deferred_send(self) -> None:
        """Wait delay_seconds then send the initial status message."""
        await asyncio.sleep(self._delay)
        if self._cancelled:
            return
        try:
            self._status_msg = await self._message.reply_text(self._initial_text)
            self._last_text = self._initial_text
            logger.debug("Deferred status message sent after %.1fs", self._delay)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Deferred status message send failed: %s", exc)