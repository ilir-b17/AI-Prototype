import logging

logger = logging.getLogger(__name__)


async def update_objective_status(task_id: int, new_status: str) -> str:
    logger.info(f"update_objective_status: id={task_id} -> {new_status}")
    ledger = None
    try:
        from src.memory.ledger_db import LedgerMemory
        ledger = LedgerMemory()
        await ledger.initialize()
        await ledger.update_objective_status(task_id, new_status)
        return f"Success: Objective {task_id} marked as {new_status!r}."
    except Exception as exc:
        return f"Error: Could not update objective status due to [{exc}]."
    finally:
        if ledger:
            await ledger.close()
