import logging

logger = logging.getLogger(__name__)


import json

async def query_highest_priority_task() -> str:
    logger.info("query_highest_priority_task called")
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
        task = await ledger.get_highest_priority_task()
        if not task:
            return json.dumps({
                "status": "success",
                "message": "No pending tasks found in the backlog.",
                "task": None
            })

        return json.dumps({
            "status": "success",
            "task": {
                "id": task['id'],
                "title": task['title'],
                "estimated_energy": task['estimated_energy'],
                "priority": task['priority'],
                "origin": task['origin']
            }
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Could not query the backlog",
            "details": str(exc),
            "suggestion": "Check the SQLite ledger database connection."
        })
    finally:
        if ledger and owns_connection:
            await ledger.close()
