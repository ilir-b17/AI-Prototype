import asyncio
import logging

logger = logging.getLogger(__name__)


def _sync_update_core_memory(key: str, value: str) -> str:
    try:
        from src.memory.core_memory import CoreMemory
        mem = CoreMemory()
        mem.update(key, value)
        return f"Success: Core memory key '{key}' updated."
    except Exception as exc:
        return f"Error: Could not update core memory due to [{exc}]."


async def update_core_memory(key: str, value: str) -> str:
    logger.info(f"update_core_memory: {key}={value}")
    return await asyncio.to_thread(_sync_update_core_memory, key, value)
