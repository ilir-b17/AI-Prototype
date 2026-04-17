import logging

logger = logging.getLogger(__name__)


async def spawn_new_objective(tier: str, title: str, estimated_energy: int) -> str:
    logger.info(f"spawn_new_objective: [{tier}] {title}")
    ledger = None
    try:
        from src.memory.ledger_db import LedgerMemory
        ledger = LedgerMemory()
        await ledger.initialize()
        obj_id = await ledger.add_objective(
            tier=tier, title=title,
            estimated_energy=estimated_energy, origin="System"
        )
        return f"Success: New {tier} added to backlog (id={obj_id}): {title!r}"
    except Exception as exc:
        return f"Error: Could not spawn objective due to [{exc}]."
    finally:
        if ledger:
            await ledger.close()
