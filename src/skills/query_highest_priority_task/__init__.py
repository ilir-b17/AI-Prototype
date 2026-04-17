import logging

logger = logging.getLogger(__name__)


async def query_highest_priority_task() -> str:
    logger.info("query_highest_priority_task called")
    ledger = None
    try:
        from src.memory.ledger_db import LedgerMemory
        ledger = LedgerMemory()
        await ledger.initialize()
        task = await ledger.get_highest_priority_task()
        if not task:
            return "BACKLOG: No pending Tasks found."
        return (
            f"BACKLOG TASK\n"
            f"  ID: {task['id']}\n"
            f"  Title: {task['title']}\n"
            f"  Estimated Energy: {task['estimated_energy']}\n"
            f"  Priority: {task['priority']}\n"
            f"  Origin: {task['origin']}"
        )
    except Exception as exc:
        return f"Error: Could not query backlog due to [{exc}]."
    finally:
        if ledger:
            await ledger.close()
