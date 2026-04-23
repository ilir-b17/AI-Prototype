import logging

logger = logging.getLogger(__name__)


import json

async def update_core_memory(key: str, value: str) -> str:
    logger.info(f"update_core_memory: {key}={value}")

    if not isinstance(key, str) or not key.strip():
         return json.dumps({
             "status": "error",
             "message": "Invalid key",
             "details": "The key must be a non-empty string."
         })

    mem = None
    try:
        from src.core.runtime_context import get_core_memory
        from src.memory.core_memory import CoreMemory
        mem = get_core_memory() or CoreMemory()
        await mem.update(key, value)
        return json.dumps({
            "status": "success",
            "message": f"Core memory key '{key}' updated successfully."
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Could not update core memory",
            "details": str(exc),
            "suggestion": "Check file permissions for the core_memory.json file."
        })
