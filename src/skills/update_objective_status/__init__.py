import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


import json


def _completion_refund_amount(estimated_energy: int) -> int:
    return max(0, int(max(0, int(estimated_energy)) * 0.2))


async def _refund_predictive_budget_for_completion(task_id: int, ledger: Any) -> int:
    """Refund predictive budget for completed Task updates made through this skill."""
    try:
        context: Dict[str, Any] = await ledger.get_task_with_parent_context(task_id)
    except Exception:
        return 0
    if not context:
        return 0

    task = dict(context.get("task") or {})
    if str(task.get("status") or "").strip().lower() != "completed":
        return 0

    refund = _completion_refund_amount(int(task.get("estimated_energy") or 0))
    if refund <= 0:
        return 0

    try:
        from src.core.runtime_context import get_orchestrator

        orchestrator = get_orchestrator()
        refund_fn = getattr(orchestrator, "_refund_predictive_energy_budget", None)
        if not callable(refund_fn):
            return 0
        await refund_fn(refund, "task_completed_via_update_objective_status")
        logger.info(
            "Applied predictive budget completion refund for task %s: +%s",
            task_id,
            refund,
        )
        return refund
    except Exception:
        return 0

async def update_objective_status(task_id: int, new_status: str) -> str:
    logger.info(f"update_objective_status: id={task_id} -> {new_status}")

    try:
        task_id = int(task_id)
    except ValueError:
        return json.dumps({
            "status": "error",
            "message": "Invalid task_id",
            "details": "The task_id must be an integer."
        })

    if not isinstance(new_status, str) or not new_status.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid new_status",
            "details": "The new_status must be a non-empty string."
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
        await ledger.update_objective_status(task_id, new_status)
        refund_amount = 0
        if str(new_status).strip().lower() == "completed":
            refund_amount = await _refund_predictive_budget_for_completion(task_id, ledger)

        message = f"Objective {task_id} marked as '{new_status}'."
        if refund_amount > 0:
            message += f" Predictive budget refund: +{refund_amount}."

        return json.dumps({
            "status": "success",
            "message": message,
        }, indent=2)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Could not update objective status",
            "details": str(exc)
        })
    finally:
        if ledger and owns_connection:
            await ledger.close()
