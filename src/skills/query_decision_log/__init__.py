"""
query_decision_log skill - introspect supervisor decisions and moral audit.
"""

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_SCOPES = {"last_turn", "recent", "rejections", "all"}


def _format_plan(plan: list) -> str:
    if not plan:
        return "Direct response (no agents)"
    agents = [str(step.get("agent", "?")) for step in plan if isinstance(step, dict)]
    return " -> ".join(agents) if agents else "Unknown plan"


def _format_decision(d: dict) -> dict:
    return {
        "id": d.get("id"),
        "timestamp": d.get("created_at", ""),
        "user_input_preview": str(d.get("user_input", ""))[:120],
        "plan": _format_plan(d.get("plan", [])),
        "is_direct": d.get("is_direct", False),
        "reasoning_preview": str(d.get("reasoning", ""))[:200],
        "energy_before": d.get("energy_before", 0),
        "worker_count": d.get("worker_count", 0),
    }


def _format_moral_record(r: dict) -> dict:
    return {
        "id": r.get("id"),
        "timestamp": r.get("created_at", ""),
        "audit_mode": r.get("audit_mode", ""),
        "is_approved": r.get("is_approved", True),
        "critic_feedback_preview": str(r.get("critic_feedback", ""))[:150],
        "reasoning_preview": str(r.get("audit_trace", ""))[:200],
        "violated_tiers": r.get("violated_tiers", []),
        "remediation_constraints": r.get("remediation_constraints", []),
        "request_preview": str(r.get("request_redacted", ""))[:100],
    }


def _normalize_scope(scope: str) -> str:
    candidate = str(scope or "").strip()
    return candidate if candidate in _VALID_SCOPES else "last_turn"


def _normalize_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        parsed = 5
    if parsed <= 0:
        parsed = 5
    return min(parsed, 20)


async def _fetch_decisions(
    ledger,
    *,
    scope: str,
    user_id: Optional[str],
    limit: int,
) -> list[dict]:
    if scope not in ("last_turn", "recent", "all"):
        return []
    raw_decisions = await ledger.get_recent_supervisor_decisions(
        user_id=user_id,
        limit=(1 if scope == "last_turn" else limit),
    )
    return [_format_decision(d) for d in raw_decisions]


async def _fetch_moral_records(
    ledger,
    *,
    scope: str,
    user_id: Optional[str],
    limit: int,
) -> list[dict]:
    if scope not in ("rejections", "all"):
        return []

    raw_moral = await ledger.get_moral_audit_summary(
        user_id=user_id,
        limit=limit,
    )
    if scope == "rejections":
        raw_moral = [r for r in raw_moral if not r.get("is_approved", True)]
    return [_format_moral_record(r) for r in raw_moral]


def _build_success_result(
    *,
    scope: str,
    total_decisions: int,
    decisions: list[dict],
    moral_records: list[dict],
) -> dict:
    result: dict = {
        "status": "success",
        "scope": scope,
        "total_decisions_on_record": total_decisions,
    }

    include_decisions = scope in ("last_turn", "recent", "all")
    include_moral = scope in ("rejections", "all")

    if include_decisions:
        result["decisions"] = decisions
    if include_moral:
        result["moral_audit_records"] = moral_records

    if include_decisions and not decisions:
        result["note"] = (
            "No supervisor decisions logged yet. Decisions are "
            "recorded after the first user turn completes."
        )
    if scope == "rejections" and not moral_records:
        result["note"] = (
            "No moral audit rejections found. All recent outputs "
            "have passed moral review."
        )
    return result


async def query_decision_log(
    scope: str = "last_turn",
    limit: int = 5,
    user_id: Optional[str] = None,
) -> str:
    """Query AIDEN's supervisor decision log and moral audit history."""
    scope = _normalize_scope(scope)
    limit = _normalize_limit(limit)

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
                decisions = await _fetch_decisions(
                    ledger,
                    scope=scope,
                    user_id=user_id,
                    limit=limit,
                )
                moral_records = await _fetch_moral_records(
                    ledger,
                    scope=scope,
                    user_id=user_id,
                    limit=limit,
                )
                total_decisions = await ledger.count_supervisor_decisions(user_id)
            result = _build_success_result(
                scope=scope,
                total_decisions=total_decisions,
                decisions=decisions,
                moral_records=moral_records,
            )
            return json.dumps(result, indent=2, default=str)

        finally:
            if owns:
                await ledger.close()

    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"query_decision_log timed out after {timeout}s.",
        })
    except Exception as exc:
        logger.error("query_decision_log error: %s", exc, exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Could not query decision log",
            "details": str(exc),
        })
