import logging

logger = logging.getLogger(__name__)


import json

async def update_objective_status(task_id: int, new_status: str) -> str:
    logger.info(f"update_objective_status: id={task_id} -> {new_status}")

    try:
        task_id = int(task_id)
    except ValueError:
        return json.dumps({
            "status": "error",
            "message": "Invalid task_id",
            "details": "The task_id must be an integer."
        })

    if not isinstance(new_status, str) or not new_status.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid new_status",
            "details": "The new_status must be a non-empty string."
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
        await ledger.update_objective_status(task_id, new_status)
        return json.dumps({
            "status": "success",
            "message": f"Objective {task_id} marked as '{new_status}'."
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Could not update objective status",
            "details": str(exc)
        })
    finally:
        if ledger and owns_connection:
            await ledger.close()
