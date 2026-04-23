import logging

logger = logging.getLogger(__name__)


import json

async def spawn_new_objective(tier: str, title: str, estimated_energy: int) -> str:
    logger.info(f"spawn_new_objective: [{tier}] {title}")

    if not isinstance(title, str) or not title.strip():
         return json.dumps({
             "status": "error",
             "message": "Invalid title",
             "details": "The title must be a non-empty string."
         })

    try:
        estimated_energy = int(estimated_energy)
    except ValueError:
        return json.dumps({
             "status": "error",
             "message": "Invalid estimated_energy",
             "details": "estimated_energy must be an integer."
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
        obj_id = await ledger.add_objective(
            tier=tier, title=title,
            estimated_energy=estimated_energy, origin="System"
        )
        return json.dumps({
            "status": "success",
            "message": f"New {tier} added to backlog.",
            "data": {
                "id": obj_id,
                "tier": tier,
                "title": title
            }
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Could not spawn objective",
            "details": str(exc)
        })
    finally:
        if ledger and owns_connection:
            await ledger.close()
