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
from typing import Dict, Any, List

import aiofiles

logger = logging.getLogger(__name__)

_DEFAULT_NOCTURNAL_FACTS_CHARS = 1500

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
        async with self._lock:
            try:
                async with aiofiles.open(self.memory_file_path, 'r') as f:
                    content = await f.read()
                return json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load core memory: {e}")
                return {"current_focus": "", "user_preferences": ""}

    @staticmethod
    def _dedupe_fact_lines(raw_facts: Any) -> List[str]:
        if isinstance(raw_facts, str):
            candidates = [raw_facts]
        elif isinstance(raw_facts, list):
            candidates = raw_facts
        else:
            candidates = []

        deduped: List[str] = []
        seen = set()
        for item in candidates:
            fact = " ".join(str(item or "").split())
            if not fact:
                continue
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    @staticmethod
    def _nocturnal_facts_char_limit() -> int:
        try:
            return max(0, int(os.getenv("MAX_NOCTURNAL_FACTS_CHARS", str(_DEFAULT_NOCTURNAL_FACTS_CHARS))))
        except ValueError:
            return _DEFAULT_NOCTURNAL_FACTS_CHARS

    def _format_nocturnal_core_facts(self, state: Dict[str, Any]) -> str:
        facts = self._dedupe_fact_lines(state.get("nocturnal_core_facts", []))
        if not facts:
            return ""

        rendered = "\n".join(f"    <Fact>{fact}</Fact>" for fact in facts)
        char_limit = self._nocturnal_facts_char_limit()
        if char_limit == 0:
            return ""
        if len(rendered) > char_limit:
            rendered = rendered[:char_limit].rstrip()

        return f"\n  <Nocturnal_Core_Facts>\n{rendered}\n  </Nocturnal_Core_Facts>"

    async def get_context_string(self, include_summary: bool = False) -> str:
        """Return the core memory formatted for prompt injection (non-blocking).

        ``conversation_summary`` is excluded by default because it is a lossy,
        model-generated recap that can easily inject stale or low-quality
        behavioral commentary back into the live prompt.
        """
        state = await self.get_all()
        host_os = state.get('host_os', '')
        os_line = f"\n  <Host_OS>{host_os}</Host_OS>" if host_os else ""
        summary = state.get('conversation_summary', '')
        summary_line = (
            f"\n  <Conversation_Summary>{summary}</Conversation_Summary>"
            if include_summary and summary else ""
        )
        insights = state.get('consolidated_insights', '')
        insights_line = f"\n  <Consolidated_Insights>{insights}</Consolidated_Insights>" if insights else ""
        nocturnal_facts = self._format_nocturnal_core_facts(state)
        return (
            f"<Core_Working_Memory>\n"
            f"  <Current_Focus>{state.get('current_focus', '')}</Current_Focus>\n"
            f"  <User_Preferences>{state.get('user_preferences', '')}</User_Preferences>"
            f"{os_line}"
            f"{summary_line}"
            f"{insights_line}"
            f"{nocturnal_facts}\n"
            f"</Core_Working_Memory>"
        )

    async def update(self, key: str, value: Any) -> bool:
        """Update a specific key in core memory (non-blocking, concurrency-safe)."""
        async with self._lock:
            try:
                async with aiofiles.open(self.memory_file_path, 'r') as f:
                    content = await f.read()
                state = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load core memory for update: {e}")
                state = {"current_focus": "", "user_preferences": ""}
            state[key] = value
            await self._save_memory_async(state)
        logger.info(f"Updated Core Memory: {key} = {value}")
        return True
