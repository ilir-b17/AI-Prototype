import logging

logger = logging.getLogger(__name__)


async def update_core_memory(key: str, value: str) -> str:
    logger.info(f"update_core_memory: {key}={value}")
    mem = None
    try:
        from src.core.runtime_context import get_core_memory
        from src.memory.core_memory import CoreMemory
        mem = get_core_memory() or CoreMemory()
        await mem.update(key, value)
        return f"Success: Core memory key '{key}' updated."
    except Exception as exc:
        return f"Error: Could not update core memory due to [{exc}]."
