"""
query_system_health skill - introspect system errors and synthesis history.
"""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

_STATUS_ICONS = {
    "approved": "\u2705",
    "rejected": "\u274c",
    "blocked": "\U0001F6AB",
    "in_progress": "\U0001F504",
    "pending_approval": "\u23f3",
}


def _bounded_int(value, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _count_by_key(records: list[dict], key: str, default: str) -> dict:
    counts: dict = {}
    for record in records:
        value = str(record.get(key, default))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _synthesis_summary(run: dict) -> dict:
    status = str(run.get("status", "unknown"))
    return {
        "id": run.get("id"),
        "tool_name": run.get("suggested_tool_name", ""),
        "final_name": run.get("final_tool_name", ""),
        "status": status,
        "status_icon": _STATUS_ICONS.get(status, "?"),
        "attempts": run.get("total_attempts", 0),
        "max_retries": run.get("max_retries", 3),
        "test_summary_preview": str(run.get("test_summary", ""))[:120],
        "blocked_reason_preview": str(run.get("blocked_reason", ""))[:120],
        "created_at": run.get("created_at", ""),
    }


def _build_error_log_section(errors: list[dict], hours: int) -> dict:
    section = {
        "summary": _count_by_key(errors, "log_level", "UNKNOWN"),
        "total_entries": len(errors),
        "entries": [
            {
                "id": e.get("id"),
                "level": e.get("log_level"),
                "message_preview": str(e.get("message", ""))[:200],
                "timestamp": e.get("timestamp"),
            }
            for e in errors
        ],
    }
    if not errors:
        section["note"] = (
            f"No WARNING/ERROR/CRITICAL entries in the last {hours}h. "
            "System appears healthy."
        )
    return section


def _build_synthesis_section(synthesis_runs: list[dict]) -> dict:
    section = {
        "summary": _count_by_key(synthesis_runs, "status", "unknown"),
        "total_runs": len(synthesis_runs),
        "runs": [_synthesis_summary(r) for r in synthesis_runs],
    }
    if not synthesis_runs:
        section["note"] = (
            "No tool synthesis runs recorded yet. "
            "Synthesis is triggered when System 1 identifies a capability gap."
        )
    return section


async def query_system_health(
    hours: int = 24,
    include_synthesis_history: bool = True,
    include_error_log: bool = True,
    error_limit: int = 20,
) -> str:
    """Return system health report with errors and synthesis history."""
    hours = _bounded_int(hours, default=24, min_value=1, max_value=168)
    error_limit = _bounded_int(error_limit, default=20, min_value=1, max_value=50)
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
            result: dict = {
                "status": "success",
                "report_window_hours": hours,
            }

            if include_error_log:
                async with asyncio.timeout(timeout):
                    errors = await ledger.get_system_error_log(
                        hours=hours,
                        limit=error_limit,
                    )
                result["error_log"] = _build_error_log_section(errors, hours)

            if include_synthesis_history:
                async with asyncio.timeout(timeout):
                    synthesis_runs = await ledger.get_recent_synthesis_runs(limit=10)
                result["synthesis_history"] = _build_synthesis_section(synthesis_runs)

            return json.dumps(result, indent=2, default=str)

        finally:
            if owns:
                await ledger.close()

    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"query_system_health timed out after {timeout}s.",
        })
    except Exception as exc:
        logger.error("query_system_health error: %s", exc, exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Could not query system health",
            "details": str(exc),
        })
