import logging

logger = logging.getLogger(__name__)


async def update_ledger(task_description: str, priority: int = 5) -> str:
    logger.info(f"update_ledger: {task_description}")
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
        await ledger.add_task(task_description=task_description, priority=priority)
        return "Success: Task added to ledger."
    except Exception as exc:
        return f"Error: Could not write to database due to [{exc}]."
    finally:
        if ledger and owns_connection:
            await ledger.close()
