import asyncio
import logging

logger = logging.getLogger(__name__)


async def request_core_update(component: str, proposed_change: str) -> str:
    """
    Called only after MFA is validated by the orchestrator.
    Logs the approved change request to the ledger for admin review.
    """
    logger.info(f"request_core_update (post-MFA): {component}")
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
        task_desc = f"Approved Core Update Request for {component}: {proposed_change}"
        await ledger.add_task(task_description=task_desc, priority=1)
        return "Success: MFA authorized. Core update request logged to ledger for admin review."
    except Exception as exc:
        return f"Error: Failed to log core update — {exc}"
    finally:
        if ledger and owns_connection:
            await ledger.close()
