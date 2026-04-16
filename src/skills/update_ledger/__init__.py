import asyncio
import logging

logger = logging.getLogger(__name__)


def _sync_update_ledger(task_description: str, priority: int = 5) -> str:
    ledger = None
    try:
        from src.memory.ledger_db import LedgerMemory
        ledger = LedgerMemory()
        ledger.add_task(task_description=task_description, priority=priority)
        return "Success: Task added to ledger."
    except Exception as exc:
        return f"Error: Could not write to database due to [{exc}]."
    finally:
        if ledger:
            ledger.close()


async def update_ledger(task_description: str, priority: int = 5) -> str:
    logger.info(f"update_ledger: {task_description}")
    return await asyncio.to_thread(_sync_update_ledger, task_description, priority)
