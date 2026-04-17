import logging

logger = logging.getLogger(__name__)


async def update_core_memory(key: str, value: str) -> str:
    logger.info(f"update_core_memory: {key}={value}")
    try:
        from src.memory.core_memory import CoreMemory
        mem = CoreMemory()
        await mem.update(key, value)
        return f"Success: Core memory key '{key}' updated."
    except Exception as exc:
        return f"Error: Could not update core memory due to [{exc}]."
