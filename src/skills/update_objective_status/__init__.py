import logging

logger = logging.getLogger(__name__)


async def update_objective_status(task_id: int, new_status: str) -> str:
    logger.info(f"update_objective_status: id={task_id} -> {new_status}")
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
        return f"Success: Objective {task_id} marked as {new_status!r}."
    except Exception as exc:
        return f"Error: Could not update objective status due to [{exc}]."
    finally:
        if ledger and owns_connection:
            await ledger.close()
