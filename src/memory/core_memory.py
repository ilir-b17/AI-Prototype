"""
Core Memory Module - Implements Working Memory (OS-Level RAM)

This module handles short-term context like current focus and user preferences,
stored in a JSON file to act as temporary working memory (RAM) for the Orchestrator
and Worker Nodes.
"""

import json
import os
import asyncio
import logging
from typing import Dict, Any

import aiofiles

logger = logging.getLogger(__name__)

class CoreMemory:
    """
    Manages short-term working memory stored in a local JSON file.

    All runtime I/O is non-blocking via aiofiles.  The constructor only creates
    the directory; call ``initialize()`` (or let the Orchestrator's async_init
    do it) before the first read/write in an async context.
    """

    def __init__(self, memory_file_path: str = "data/core_memory.json"):
        self.memory_file_path = memory_file_path
        self._lock = asyncio.Lock()
        os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
        # Synchronous bootstrap: create the file if it does not yet exist so
        # the Orchestrator's sync __init__ path still works.
        default_state = {"current_focus": "", "user_preferences": ""}
        try:
            # Atomic create: open with 'x' fails if the file already exists,
            # avoiding a TOCTOU race between existence check and creation.
            with open(self.memory_file_path, 'x') as f:
                json.dump(default_state, f, indent=4)
            logger.info(f"Initialized new Core Working Memory at {self.memory_file_path}")
        except FileExistsError:
            # File already exists — validate it is not corrupt.
            try:
                with open(self.memory_file_path, 'r') as f:
                    json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    f"Core Working Memory at {self.memory_file_path} corrupted. Re-initializing."
                )
                try:
                    with open(self.memory_file_path, 'w') as f:
                        json.dump(default_state, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed to re-initialize core memory: {e}")
        except Exception as e:
            logger.error(f"Failed to bootstrap core memory: {e}")

    # ──────────────────────────────────────────────────────────────
    # Async I/O  (primary API — use these in the async event loop)
    # ──────────────────────────────────────────────────────────────

    async def _save_memory_async(self, state: Dict[str, Any]) -> None:
        """Write state dict to disk without blocking the event loop."""
        try:
            async with aiofiles.open(self.memory_file_path, 'w') as f:
                await f.write(json.dumps(state, indent=4))
        except Exception as e:
            logger.error(f"Failed to save core memory: {e}")

    async def get_all(self) -> Dict[str, Any]:
        """Retrieve entire memory state (non-blocking)."""
        try:
            async with aiofiles.open(self.memory_file_path, 'r') as f:
                content = await f.read()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load core memory: {e}")
            return {"current_focus": "", "user_preferences": ""}

    async def get_context_string(self) -> str:
        """Return the core memory formatted for prompt injection (non-blocking)."""
        state = await self.get_all()
        host_os = state.get('host_os', '')
        os_line = f"\n  <Host_OS>{host_os}</Host_OS>" if host_os else ""
        summary = state.get('conversation_summary', '')
        summary_line = f"\n  <Conversation_Summary>{summary}</Conversation_Summary>" if summary else ""
        insights = state.get('consolidated_insights', '')
        insights_line = f"\n  <Consolidated_Insights>{insights}</Consolidated_Insights>" if insights else ""
        return (
            f"<Core_Working_Memory>\n"
            f"  <Current_Focus>{state.get('current_focus', '')}</Current_Focus>\n"
            f"  <User_Preferences>{state.get('user_preferences', '')}</User_Preferences>"
            f"{os_line}"
            f"{summary_line}"
            f"{insights_line}\n"
            f"</Core_Working_Memory>"
        )

    async def update(self, key: str, value: Any) -> bool:
        """Update a specific key in core memory (non-blocking, concurrency-safe)."""
        async with self._lock:
            state = await self.get_all()
            state[key] = value
            await self._save_memory_async(state)
        logger.info(f"Updated Core Memory: {key} = {value}")
        return True
