"""LLM gateway helpers for the Orchestrator facade."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

from src.core import cloud_redaction
from src.core.llm_router import RouterResult

logger = logging.getLogger(__name__)


def is_system_1_error(result: Optional[RouterResult], error_prefix: str) -> bool:
    return bool(
        result
        and result.status == "ok"
        and isinstance(result.content, str)
        and result.content.startswith(error_prefix)
    )


def is_system_2_error(result: Optional[RouterResult], error_prefixes: Sequence[str]) -> bool:
    return bool(
        result
        and result.status == "ok"
        and isinstance(result.content, str)
        and result.content.startswith(tuple(error_prefixes))
    )


def redact_text_for_cloud(
    text: str,
    *,
    allow_sensitive_context: bool = False,
    max_chars: int = 12000,
) -> str:
    return cloud_redaction.redact_text_for_cloud(
        text,
        allow_sensitive_context=allow_sensitive_context,
        max_chars=max_chars,
    )


def redact_messages_for_cloud(
    messages: List[Dict[str, str]],
    *,
    allow_sensitive_context: bool = False,
) -> List[Dict[str, str]]:
    return cloud_redaction.redact_messages_for_cloud(
        messages,
        allow_sensitive_context=allow_sensitive_context,
    )


def cloud_payload_audit_sha256(
    messages: List[Dict[str, str]],
    allowed_tools: Optional[List[str]],
) -> str:
    return cloud_redaction.compute_payload_sha256(messages, allowed_tools)


async def append_cloud_payload_audit_if_supported(
    ledger: Any,
    *,
    purpose: str,
    message_count_before: int,
    message_count_after: int,
    allow_sensitive_context: bool,
    payload_sha256: str,
    log: Optional[logging.Logger] = None,
) -> None:
    append_fn = getattr(ledger, "append_cloud_payload_audit", None)
    if not callable(append_fn):
        return

    active_logger = log or logger
    try:
        await append_fn(
            purpose=purpose,
            message_count_before=message_count_before,
            message_count_after=message_count_after,
            allow_sensitive_context=allow_sensitive_context,
            payload_sha256=payload_sha256,
        )
    except Exception as exc:
        active_logger.warning("Failed to append cloud payload audit entry for %s: %s", purpose, exc)


async def route_to_system_2_redacted(
    cognitive_router: Any,
    ledger_memory: Any,
    messages: List[Dict[str, str]],
    *,
    allowed_tools: Optional[List[str]] = None,
    purpose: str = "system_2",
    allow_sensitive_context: bool = False,
    log: Optional[logging.Logger] = None,
) -> RouterResult:
    active_logger = log or logger
    minimized_messages = redact_messages_for_cloud(
        messages,
        allow_sensitive_context=allow_sensitive_context,
    )
    active_logger.info(
        "System 2 payload redacted for %s: %d -> %d message(s)",
        purpose,
        len(messages),
        len(minimized_messages),
    )
    await append_cloud_payload_audit_if_supported(
        ledger_memory,
        purpose=purpose,
        message_count_before=len(messages),
        message_count_after=len(minimized_messages),
        allow_sensitive_context=allow_sensitive_context,
        payload_sha256=cloud_payload_audit_sha256(minimized_messages, allowed_tools),
        log=active_logger,
    )
    return await cognitive_router.route_to_system_2(
        minimized_messages,
        allowed_tools=allowed_tools,
    )


def get_system_1_gate_metrics(
    cognitive_router: Any,
    *,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    getter = getattr(cognitive_router, "get_system_1_gate_metrics", None)
    if not callable(getter):
        return {}
    active_logger = log or logger
    try:
        metrics = getter()
    except Exception as exc:
        active_logger.debug("Unable to read System 1 gate metrics: %s", exc)
        return {}
    return metrics if isinstance(metrics, dict) else {}


def log_system_1_gate_pressure(
    context: str,
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    if not after_metrics:
        return

    before_wait_events = int(before_metrics.get("wait_events", 0) or 0)
    after_wait_events = int(after_metrics.get("wait_events", 0) or 0)
    waiting_requests = int(after_metrics.get("waiting_requests", 0) or 0)
    new_wait_events = max(0, after_wait_events - before_wait_events)

    if new_wait_events == 0 and waiting_requests == 0:
        return

    active_logger = log or logger
    active_logger.info(
        "System 1 gate pressure after %s: active=%s waiting=%s max=%s new_wait_events=%s total_wait=%.3fs avg_wait=%.3fs peak_waiting=%s",
        context,
        int(after_metrics.get("active_requests", 0) or 0),
        waiting_requests,
        int(after_metrics.get("max_concurrency", 0) or 0),
        new_wait_events,
        float(after_metrics.get("total_wait_seconds", 0.0) or 0.0),
        float(after_metrics.get("average_wait_seconds", 0.0) or 0.0),
        int(after_metrics.get("peak_waiting_requests", 0) or 0),
    )


async def route_to_system_1(
    cognitive_router: Any,
    messages: List[Dict[str, str]],
    allowed_tools: Optional[List[str]] = None,
    *,
    deadline_seconds: Optional[float] = None,
    context: str = "orchestrator",
    log: Optional[logging.Logger] = None,
) -> RouterResult:
    before_metrics = get_system_1_gate_metrics(cognitive_router, log=log)
    route_call = cognitive_router.route_to_system_1(messages, allowed_tools=allowed_tools)
    try:
        if deadline_seconds is None:
            return await route_call
        return await asyncio.wait_for(route_call, timeout=deadline_seconds)
    finally:
        after_metrics = get_system_1_gate_metrics(cognitive_router, log=log)
        log_system_1_gate_pressure(context, before_metrics, after_metrics, log=log)
