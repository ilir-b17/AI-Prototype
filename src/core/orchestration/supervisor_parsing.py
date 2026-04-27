"""Stateless supervisor response parsing helpers."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from src.core.orchestration import plan_normalization

logger = logging.getLogger(__name__)


def extract_workers_payload(response: str) -> tuple[str, Optional[str]]:
    marker = "WORKERS:"
    if marker not in response:
        return response.strip(), None

    marker_index = response.rfind(marker)
    answer = response[:marker_index].strip()
    payload = response[marker_index + len(marker):].strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload).strip()
    return answer, payload or None


def decode_workers_payload(
    workers_payload: str,
    *,
    log: Optional[logging.Logger] = None,
) -> Optional[Any]:
    if not workers_payload:
        return None
    try:
        decoded, end_index = json.JSONDecoder().raw_decode(workers_payload)
    except ValueError:
        return None

    trailing = workers_payload[end_index:].strip()
    if trailing:
        active_logger = log or logger
        active_logger.warning(
            "_decode_workers_payload: model appended trailing text after JSON "
            "(%r ... first 60 chars of payload: %r); ignoring suffix and proceeding.",
            trailing[:60],
            workers_payload[:60],
        )
    return decoded


def is_structured_plan_packet(step: Any) -> bool:
    if not isinstance(step, dict):
        return False

    agent_name = str(step.get("agent") or step.get("name") or "").strip()
    if not agent_name:
        return False

    task = str(step.get("task") or step.get("instructions") or step.get("objective") or "").strip()
    reason = str(step.get("reason") or step.get("why") or "").strip()
    return bool(task or reason)


def is_structured_plan_payload(payload: Any) -> bool:
    if not isinstance(payload, list):
        return False
    return all(is_structured_plan_packet(step) for step in payload)


def parse_supervisor_response(
    response: str,
    state: Dict[str, Any],
    *,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    response_text = response.strip()
    answer, workers_payload = extract_workers_payload(response_text)

    if workers_payload is None:
        state["current_plan"] = []
        state["final_response"] = answer or response.strip()
        return state

    active_logger = log or logger
    decoded_payload = decode_workers_payload(workers_payload, log=active_logger)

    if isinstance(decoded_payload, list) and not decoded_payload:
        state["current_plan"] = []
        state["final_response"] = (
            answer
            or "I wasn't able to find an answer to that. Could you try rephrasing your question?"
        )
        return state

    if not is_structured_plan_payload(decoded_payload):
        active_logger.warning("Supervisor produced invalid or non-structured WORKERS payload: %r", workers_payload)
        state["current_plan"] = []
        fallback_answer = answer if (answer and len(answer.strip()) >= 40) else response_text
        if fallback_answer == response_text:
            fallback_answer = re.sub(r"WORKERS:\s*.*$", "", fallback_answer, flags=re.DOTALL).strip()
        clean_response_text = re.sub(
            r"WORKERS:\s*\[.*?\]\s*", "", response_text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        state["final_response"] = (
            fallback_answer
            or answer
            or clean_response_text
            or "I wasn't able to process that request. Please try again."
        )
        return state

    normalized_plan = plan_normalization.normalize_current_plan(decoded_payload)
    if normalized_plan:
        state["current_plan"] = normalized_plan
        if answer:
            state["worker_outputs"]["supervisor_context"] = answer
    else:
        state["current_plan"] = []
        clean_response_text = re.sub(
            r"WORKERS:\s*\[.*?\]\s*", "", response_text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        state["final_response"] = (
            answer
            or clean_response_text
            or "I wasn't able to process that request. Please try again."
        )
    return state
