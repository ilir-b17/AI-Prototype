"""
query_objective_status skill - introspect the objective backlog.
"""

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_STATUSES = {
    "pending", "active", "deferred", "blocked", "completed", "all",
    "deferred_due_to_energy",
}
_VALID_TIERS = {"Epic", "Story", "Task"}

_STATUS_MAP = {
    "deferred": "deferred_due_to_energy",
}


def _format_energy_eval(eval_dict: dict) -> dict:
    if not eval_dict:
        return {}
    return {
        "reason": eval_dict.get("reason", ""),
        "roi": round(float(eval_dict.get("roi", 0)), 3),
        "estimated_effort": eval_dict.get("estimated_effort"),
        "expected_value": eval_dict.get("expected_value"),
        "predicted_cost": eval_dict.get("predicted_cost"),
        "available_energy_at_eval": eval_dict.get("available_energy"),
        "defer_count": eval_dict.get("defer_count", 0),
        "evaluated_at": eval_dict.get("evaluated_at", ""),
    }


def _normalize_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        parsed = 20
    if parsed <= 0:
        parsed = 20
    return min(parsed, 50)


def _normalize_status_filter(status_filter: str) -> tuple[str, str]:
    normalized = str(status_filter or "all").lower()
    if normalized not in _VALID_STATUSES:
        normalized = "all"
    return normalized, _STATUS_MAP.get(normalized, normalized)


def _normalize_tier(tier: Optional[str]) -> Optional[str]:
    return tier if tier in _VALID_TIERS else None


def _format_deferred_tasks(raw_tasks: list, include_energy_context: bool) -> list[dict]:
    tasks_formatted = []
    for t in raw_tasks:
        item = {
            "id": t.get("id"),
            "tier": t.get("tier"),
            "title": t.get("title"),
            "priority": t.get("priority"),
            "estimated_energy": t.get("estimated_energy"),
            "defer_count": t.get("defer_count", 0),
            "next_eligible_at": t.get("next_eligible_at"),
            "status": "deferred_due_to_energy",
        }
        if include_energy_context:
            item["energy_deferral_context"] = _format_energy_eval(
                t.get("energy_eval", {})
            )
        tasks_formatted.append(item)
    return tasks_formatted


def _format_blocked_tasks(raw_tasks: list) -> list[dict]:
    return [
        {
            "id": t.get("id"),
            "tier": t.get("tier"),
            "title": t.get("title"),
            "priority": t.get("priority"),
            "estimated_energy": t.get("estimated_energy"),
            "parent_title": t.get("parent_title"),
            "status": "blocked",
            "updated_at": t.get("updated_at"),
        }
        for t in raw_tasks
    ]


def _format_generic_tasks(
    raw_all: list,
    *,
    db_status: str,
    normalized_tier: Optional[str],
    limit: int,
) -> list[dict]:
    filtered = raw_all
    if db_status != "all":
        filtered = [
            t for t in filtered
            if str(t.get("status", "")).lower() == db_status
        ]
    if normalized_tier:
        filtered = [t for t in filtered if t.get("tier") == normalized_tier]

    filtered = filtered[:limit]
    return [
        {
            "id": t.get("id"),
            "tier": t.get("tier"),
            "title": t.get("title"),
            "status": t.get("status"),
            "priority": t.get("priority"),
            "estimated_energy": t.get("estimated_energy"),
            "acceptance_criteria_preview": (
                str(t.get("acceptance_criteria", ""))[:100]
            ),
            "defer_count": t.get("defer_count", 0),
        }
        for t in filtered
    ]


async def _fetch_tasks_for_filter(
    ledger,
    *,
    db_status: str,
    normalized_tier: Optional[str],
    include_energy_context: bool,
    limit: int,
) -> list[dict]:
    if db_status == "deferred_due_to_energy" and include_energy_context:
        raw_tasks = await ledger.get_deferred_tasks_with_energy_context(limit=limit)
        return _format_deferred_tasks(raw_tasks, include_energy_context)

    if db_status == "blocked":
        raw_tasks = await ledger.get_blocked_tasks(limit=limit)
        return _format_blocked_tasks(raw_tasks)

    raw_all = await ledger.get_all_active_goals()
    return _format_generic_tasks(
        raw_all,
        db_status=db_status,
        normalized_tier=normalized_tier,
        limit=limit,
    )


async def query_objective_status(
    status_filter: str = "all",
    tier: Optional[str] = None,
    include_energy_context: bool = True,
    limit: int = 20,
) -> str:
    """Query the objective backlog with filters and energy context."""
    limit = _normalize_limit(limit)
    status_filter, db_status = _normalize_status_filter(status_filter)
    normalized_tier = _normalize_tier(tier)

    timeout = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "15.0"))

    try:
        from src.core.runtime_context import get_ledger
        from src.memory.ledger_db import LedgerMemory

        ledger = get_ledger()
        owns = False
        if ledger is None:
            ledger = LedgerMemory()
            await ledger.initialize()
            owns = True

        try:
            async with asyncio.timeout(timeout):
                counts = await ledger.get_objective_counts_by_status()
                tasks_formatted = await _fetch_tasks_for_filter(
                    ledger,
                    db_status=db_status,
                    normalized_tier=normalized_tier,
                    include_energy_context=include_energy_context,
                    limit=limit,
                )

            return json.dumps({
                "status": "success",
                "filter_applied": {
                    "status": status_filter,
                    "tier": normalized_tier or "all",
                },
                "backlog_summary": {
                    "pending": counts.get("pending", 0),
                    "active": counts.get("active", 0),
                    "deferred_due_to_energy": counts.get("deferred_due_to_energy", 0),
                    "blocked": counts.get("blocked", 0),
                    "completed": counts.get("completed", 0),
                },
                "total_returned": len(tasks_formatted),
                "tasks": tasks_formatted,
            }, indent=2, default=str)

        finally:
            if owns:
                await ledger.close()

    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"query_objective_status timed out after {timeout}s.",
        })
    except Exception as exc:
        logger.error("query_objective_status error: %s", exc, exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Could not query objective status",
            "details": str(exc),
        })
