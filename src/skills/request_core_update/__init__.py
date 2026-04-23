import asyncio
import logging

logger = logging.getLogger(__name__)


import json

async def request_core_update(component: str, proposed_change: str) -> str:
    """
    Called only after MFA is validated by the orchestrator.
    Logs the approved change request to the ledger for admin review.
    """
    logger.info(f"request_core_update (post-MFA): {component}")

    if not isinstance(component, str) or not component.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid component",
            "details": "The component name must be a non-empty string."
        })

    if not isinstance(proposed_change, str) or not proposed_change.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid proposed_change",
            "details": "The proposed_change must be a non-empty string."
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
        task_desc = f"Approved Core Update Request for {component}: {proposed_change}"
        await ledger.add_task(task_description=task_desc, priority=1)
        return json.dumps({
            "status": "success",
            "message": "MFA authorized. Core update request logged to ledger for admin review."
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Failed to log core update to ledger",
            "details": str(exc)
        })
    finally:
        if ledger and owns_connection:
            await ledger.close()
