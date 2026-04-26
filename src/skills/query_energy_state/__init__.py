"""
query_energy_state skill - introspect AIDEN's energy budget and deferrals.
"""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

_INITIAL_BUDGET = int(os.getenv("INITIAL_ENERGY_BUDGET", "100"))
_ROI_THRESHOLD = float(os.getenv("ENERGY_ROI_THRESHOLD", "1.25"))
_MIN_RESERVE = int(os.getenv("ENERGY_MIN_RESERVE", "10"))
_MAX_DEFER = int(os.getenv("MAX_DEFER_COUNT", "5"))


async def query_energy_state(
    include_deferred_tasks: bool = True,
    include_blocked_tasks: bool = True,
) -> str:
    """Return AIDEN's current energy budget and deferred task analysis."""
    timeout = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "15.0"))

    try:
        from src.core.runtime_context import get_ledger, get_orchestrator
        from src.memory.ledger_db import LedgerMemory

        ledger = get_ledger()
        owns_ledger = False
        if ledger is None:
            ledger = LedgerMemory()
            await ledger.initialize()
            owns_ledger = True

        current_budget: int = _INITIAL_BUDGET
        orchestrator = get_orchestrator()
        if orchestrator is not None:
            budget_getter = getattr(
                orchestrator, "_get_predictive_energy_budget_remaining", None
            )
            if callable(budget_getter):
                try:
                    current_budget = int(
                        await asyncio.wait_for(budget_getter(), timeout=5.0)
                    )
                except Exception as budget_exc:
                    logger.debug("Could not read live energy budget: %s", budget_exc)

        try:
            result = {
                "status": "success",
                "energy": {
                    "current_predictive_budget": current_budget,
                    "initial_budget": _INITIAL_BUDGET,
                    "budget_percent": round(
                        (current_budget / _INITIAL_BUDGET) * 100, 1
                    ) if _INITIAL_BUDGET > 0 else 0,
                    "policy": {
                        "roi_threshold": _ROI_THRESHOLD,
                        "min_reserve": _MIN_RESERVE,
                        "max_defer_count": _MAX_DEFER,
                    },
                },
            }

            if include_deferred_tasks:
                deferred = await asyncio.wait_for(
                    ledger.get_deferred_tasks_with_energy_context(limit=15),
                    timeout=timeout,
                )
                result["deferred_tasks"] = [
                    {
                        "id": t.get("id"),
                        "title": t.get("title"),
                        "priority": t.get("priority"),
                        "estimated_energy": t.get("estimated_energy"),
                        "defer_count": t.get("defer_count", 0),
                        "next_eligible_at": t.get("next_eligible_at"),
                        "deferral_reason": (
                            t.get("energy_eval", {}).get("reason", "unknown")
                        ),
                        "roi_at_deferral": round(
                            float(t.get("energy_eval", {}).get("roi", 0)), 3
                        ),
                        "force_execute_at_defer_count": _MAX_DEFER,
                        "will_force_execute": (
                            int(t.get("defer_count", 0)) >= _MAX_DEFER
                        ),
                    }
                    for t in deferred
                ]
                result["deferred_count"] = len(deferred)

            if include_blocked_tasks:
                blocked = await asyncio.wait_for(
                    ledger.get_blocked_tasks(limit=10),
                    timeout=timeout,
                )
                result["blocked_tasks"] = [
                    {
                        "id": t.get("id"),
                        "title": t.get("title"),
                        "priority": t.get("priority"),
                        "parent_title": t.get("parent_title"),
                    }
                    for t in blocked
                ]
                result["blocked_count"] = len(blocked)

            counts = await ledger.get_objective_counts_by_status()
            result["backlog_snapshot"] = {
                "pending": counts.get("pending", 0),
                "active": counts.get("active", 0),
                "deferred": counts.get("deferred_due_to_energy", 0),
                "blocked": counts.get("blocked", 0),
            }

            return json.dumps(result, indent=2, default=str)

        finally:
            if owns_ledger:
                await ledger.close()

    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"query_energy_state timed out after {timeout}s.",
        })
    except Exception as exc:
        logger.error("query_energy_state error: %s", exc, exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Could not query energy state",
            "details": str(exc),
        })
