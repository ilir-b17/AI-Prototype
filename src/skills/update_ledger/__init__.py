import logging

logger = logging.getLogger(__name__)


import json

async def update_ledger(task_description: str, priority: int = 5) -> str:
    logger.info(f"update_ledger: {task_description}")

    if not isinstance(task_description, str) or not task_description.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid task_description",
            "details": "The task_description must be a non-empty string."
        })

    try:
        priority = int(priority)
    except ValueError:
        return json.dumps({
            "status": "error",
            "message": "Invalid priority",
            "details": "The priority must be an integer."
        })

    ledger = None
    owns_connection = False
    try:
        from src.core.runtime_context import get_ledger
        from src.memory.ledger_db import LedgerMemory
        ledger = get_ledger()
        if ledger is None:
            ledger = LedgerMemory()
            await ledger.initialize()
            owns_connection = True
        task_id = await ledger.add_task(task_description=task_description, priority=priority)
        return json.dumps({
            "status": "success",
            "message": "Task added to ledger successfully.",
            "data": {
                "id": task_id,
                "priority": priority
            }
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Could not write to ledger database",
            "details": str(exc)
        })
    finally:
        if ledger and owns_connection:
            await ledger.close()
