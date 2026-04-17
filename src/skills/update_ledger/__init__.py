import logging

logger = logging.getLogger(__name__)


async def update_ledger(task_description: str, priority: int = 5) -> str:
    logger.info(f"update_ledger: {task_description}")
    ledger = None
    try:
        from src.memory.ledger_db import LedgerMemory
        ledger = LedgerMemory()
        await ledger.initialize()
        await ledger.add_task(task_description=task_description, priority=priority)
        return "Success: Task added to ledger."
    except Exception as exc:
        return f"Error: Could not write to database due to [{exc}]."
    finally:
        if ledger:
            await ledger.close()
