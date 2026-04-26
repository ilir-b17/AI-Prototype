"""
Progress event system for AIDEN streaming responses.

The ProgressEmitter is stored in a ContextVar so that orchestrator
nodes (supervisor_node, _run_agent, critic_node) can emit events
without receiving the emitter as a parameter.

Design:
    ProgressCallback   — async callable the Telegram handler provides
    ProgressEvent      — lightweight dataclass emitted at each stage
    ProgressEmitter    — rate-limited wrapper; stores pending events
    get_current_emitter() — reads ContextVar for current asyncio task
    set_current_emitter() — sets ContextVar; returns token for reset

Zero-cost when inactive: get_current_emitter() returns None when
no callback is registered. Every caller checks for None before calling
emit(). The ContextVar lookup is O(1) and allocation-free.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


ProgressCallback = Callable[["ProgressEvent"], Awaitable[None]]

_PROGRESS_EMITTER_VAR: contextvars.ContextVar[Optional["ProgressEmitter"]] = (
    contextvars.ContextVar("aiden_progress_emitter", default=None)
)


class ProgressStage(Enum):
    GOAL_PLANNING = "goal_planning"
    SUPERVISOR = "supervisor"
    AGENT_START = "agent_start"
    AGENT_TOOL = "agent_tool"
    AGENT_DONE = "agent_done"
    CRITIC = "critic"
    SYNTHESIS = "synthesis"
    HITL = "hitl"
    MFA = "mfa"
    CAPABILITY_GAP = "capability_gap"


@dataclass
class ProgressEvent:
    """Lightweight event emitted at each pipeline stage."""

    stage: ProgressStage
    message: str
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    duration_seconds: Optional[float] = None
    plan: Optional[List[str]] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # ── Factory methods ──────────────────────────────────────────────

    @classmethod
    def supervisor_planning(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.SUPERVISOR,
            message="Planning your request...",
        )

    @classmethod
    def supervisor_done(cls, agent_names: List[str]) -> "ProgressEvent":
        count = len(agent_names)
        label = f"{count} agent" + ("s" if count != 1 else "")
        return cls(
            stage=ProgressStage.SUPERVISOR,
            message=f"Plan ready — {label}",
            plan=list(agent_names),
        )

    @classmethod
    def supervisor_direct(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.SUPERVISOR,
            message="Composing response...",
        )

    @classmethod
    def agent_start(cls, agent_name: str) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.AGENT_START,
            message=f"{agent_name}: starting",
            agent_name=agent_name,
        )

    @classmethod
    def agent_tool(cls, agent_name: str, tool_name: str) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.AGENT_TOOL,
            message=f"{agent_name}: calling {tool_name}",
            agent_name=agent_name,
            tool_name=tool_name,
        )

    @classmethod
    def agent_done(cls, agent_name: str, duration: float) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.AGENT_DONE,
            message=f"{agent_name}: done ({duration:.1f}s)",
            agent_name=agent_name,
            duration_seconds=duration,
        )

    @classmethod
    def critic_start(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.CRITIC,
            message="Evaluating output...",
        )

    @classmethod
    def synthesis_start(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.SYNTHESIS,
            message="Synthesising response...",
        )

    @classmethod
    def goal_planning(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.GOAL_PLANNING,
            message="Decomposing goal into tasks...",
        )

    @classmethod
    def hitl_raised(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.HITL,
            message="Awaiting admin guidance...",
        )

    @classmethod
    def mfa_required(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.MFA,
            message="MFA verification required...",
        )

    @classmethod
    def capability_gap(cls) -> "ProgressEvent":
        return cls(
            stage=ProgressStage.CAPABILITY_GAP,
            message="Synthesising new capability...",
        )


class ProgressEmitter:
    """Rate-limited async emitter with deduplication and pending flush.

    Rate limit: Telegram allows ~1 edit/second per chat. We enforce
    a configurable minimum interval (default 1.5s) between callbacks.
    Events arriving too soon are stored as pending and delivered on
    the next eligible emit or on flush().

    All callback errors are caught and logged at DEBUG level so a
    broken Telegram connection never breaks pipeline execution.
    """

    DEFAULT_MIN_INTERVAL: float = 1.5

    def __init__(
        self,
        callback: ProgressCallback,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL,
    ) -> None:
        self._callback = callback
        self._min_interval = min_interval_seconds
        self._last_emit_at: float = 0.0
        self._last_message: str = ""
        self._pending: Optional[ProgressEvent] = None
        self._agent_start_times: Dict[str, float] = {}

    # ── Public API ───────────────────────────────────────────────────

    def record_agent_start(self, agent_name: str) -> None:
        """Record the start time for an agent (used to compute duration)."""
        self._agent_start_times[agent_name] = time.monotonic()

    def get_agent_duration(self, agent_name: str) -> float:
        """Return elapsed seconds since record_agent_start was called."""
        start = self._agent_start_times.get(agent_name)
        return (time.monotonic() - start) if start is not None else 0.0

    async def emit(self, event: ProgressEvent) -> None:
        """Emit an event, respecting the minimum interval.

        If called too soon after the previous emit, the event is stored
        as pending and replaces any previous pending event. It will be
        delivered on the next eligible emit or on flush().
        """
        if event.message == self._last_message:
            return
        now = time.monotonic()
        elapsed = now - self._last_emit_at
        if self._last_emit_at > 0 and elapsed < self._min_interval:
            self._pending = event
            return
        await self._deliver(event)

    async def emit_immediate(self, event: ProgressEvent) -> None:
        """Emit bypassing the rate limit. Use for terminal events (HITL, MFA)."""
        self._pending = None
        await self._deliver(event)

    async def flush(self) -> None:
        """Deliver any pending event. Call before returning the final response."""
        if self._pending is not None:
            pending = self._pending
            self._pending = None
            await self._deliver(pending)

    async def flush_pending(self) -> None:
        """Backward-compatible alias used by orchestrator finalization."""
        await self.flush()

    # ── Internal ─────────────────────────────────────────────────────

    async def _deliver(self, event: ProgressEvent) -> None:
        try:
            self._last_emit_at = time.monotonic()
            self._last_message = event.message
            await self._callback(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Progress callback error (non-fatal): %s", exc)


# ── ContextVar accessors ──────────────────────────────────────────────

def get_current_emitter() -> Optional[ProgressEmitter]:
    """Return the ProgressEmitter for the current asyncio task, or None.

    O(1) lookup. Returns None when no emitter is registered (heartbeat
    tasks, fast-path responses, callers without a progress_callback).
    """
    return _PROGRESS_EMITTER_VAR.get()


def set_current_emitter(
    emitter: Optional[ProgressEmitter],
) -> contextvars.Token:
    """Set the emitter for the current asyncio task.

    Returns a Token that must be passed to reset_emitter() in a
    finally block to restore the previous value.
    """
    return _PROGRESS_EMITTER_VAR.set(emitter)


def reset_emitter(token: contextvars.Token) -> None:
    """Reset the ContextVar to its value before set_current_emitter()."""
    _PROGRESS_EMITTER_VAR.reset(token)
