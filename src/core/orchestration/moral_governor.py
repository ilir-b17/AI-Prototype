"""Moral governor and critic-node execution helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CriticNodeDeps:
    normalize_state: Callable[[Dict[str, Any]], Dict[str, Any]]
    get_output_to_evaluate: Callable[[Dict[str, Any]], str]
    store_moral_decision_trace: Callable[[Dict[str, Any], Any], None]
    build_local_skip_decision: Callable[[str], Any]
    try_triviality_bypass_decision: Callable[[Dict[str, Any], str], Optional[Any]]
    should_run_critic_review: Callable[[Dict[str, Any], str], bool]
    deduct_energy: Callable[[Dict[str, Any], int, str], Dict[str, Any]]
    refund_energy: Callable[[Dict[str, Any], int, str], Dict[str, Any]]
    build_critic_messages: Callable[[Dict[str, Any], str], list[Dict[str, str]]]
    route_critic_request: Callable[[list[Dict[str, str]]], Awaitable[Any]]
    handle_cognitive_escalation: Callable[[Dict[str, Any], Any], Awaitable[str]]
    apply_critic_response: Callable[[Dict[str, Any], str, str], Dict[str, Any]]
    persist_moral_audit_log: Callable[[Dict[str, Any], str], Awaitable[None]]
    finalize_critic_pass: Callable[[Dict[str, Any], str], Dict[str, Any]]
    suspend_task_for_moral_halt: Callable[[Dict[str, Any]], Awaitable[Optional[int]]]
    notify_admin: Callable[[str], Awaitable[None]]
    spawn_debug_task: Callable[[Dict[str, Any]], Awaitable[None]]
    charter_text: str
    charter_fallback: str
    energy_cost_critic: int
    blocked_key: str
    hitl_error_type: type[Exception]
    log: logging.Logger


async def _enforce_moral_halt(state: Dict[str, Any], deps: CriticNodeDeps) -> None:
    if not state.get("moral_halt_required"):
        return
    halt_summary = str(state.get("moral_halt_summary") or "Severe moral violation detected.").strip()
    suspended_task_id = await deps.suspend_task_for_moral_halt(state)
    hitl_message = f"HITL REQUIRED: Moral governor halt. {halt_summary[:280]}"
    if suspended_task_id is not None:
        hitl_message += f" Task #{suspended_task_id} suspended."

    try:
        await deps.notify_admin(hitl_message)
    except Exception as exc:
        deps.log.warning("Failed to notify admin for moral halt: %s", exc)

    raise deps.hitl_error_type(hitl_message)


async def _enforce_retry_limit(state: Dict[str, Any], deps: CriticNodeDeps) -> None:
    if state["iteration_count"] < 3 or state["critic_feedback"] == "PASS":
        return
    await deps.spawn_debug_task(state)
    raise deps.hitl_error_type(
        "Critic rejected output 3 times. Guidance needed.\n"
        "Question: How should I proceed to satisfy the charter?"
    )


async def critic_node(state: Dict[str, Any], *, deps: CriticNodeDeps) -> Dict[str, Any]:
    state = deps.normalize_state(state)
    output_to_eval = deps.get_output_to_evaluate(state)

    if deps.charter_text == deps.charter_fallback:
        deps.log.warning("Critic disabled because fallback charter is active.")
        deps.store_moral_decision_trace(state, deps.build_local_skip_decision("fallback_charter"))
        state["moral_audit_mode"] = "disabled_fallback_charter"
        state = deps.finalize_critic_pass(state, output_to_eval)
        return state

    if not output_to_eval:
        deps.store_moral_decision_trace(
            state,
            deps.build_local_skip_decision("empty_output"),
        )
        state["critic_feedback"] = "PASS"
        await deps.persist_moral_audit_log(state, output_to_eval)
        return state

    bypass_decision = deps.try_triviality_bypass_decision(state, output_to_eval)
    if bypass_decision is not None:
        deps.store_moral_decision_trace(state, bypass_decision)
        state = deps.finalize_critic_pass(state, output_to_eval)
        await deps.persist_moral_audit_log(state, output_to_eval)
        return state

    if not deps.should_run_critic_review(state, output_to_eval):
        deps.store_moral_decision_trace(
            state,
            deps.build_local_skip_decision("short_or_no_review_needed"),
        )
        state = deps.finalize_critic_pass(state, output_to_eval)
        await deps.persist_moral_audit_log(state, output_to_eval)
        return state

    state = deps.deduct_energy(state, deps.energy_cost_critic, "critic")
    messages = deps.build_critic_messages(state, output_to_eval)

    try:
        router_result = await deps.route_critic_request(messages)
        if router_result.status == "cognitive_escalation":
            solution = await deps.handle_cognitive_escalation(state, router_result)
            state = deps.apply_critic_response(state, output_to_eval, solution)
        elif router_result.status != "ok":
            state[deps.blocked_key] = router_result
            return state
        else:
            state = deps.apply_critic_response(state, output_to_eval, router_result.content)
        await deps.persist_moral_audit_log(state, output_to_eval)
    except Exception as exc:
        deps.log.warning("Critic node failed/timed out: %s. Defaulting to PASS.", exc, exc_info=True)
        state = deps.refund_energy(state, deps.energy_cost_critic, "critic_error")
        deps.store_moral_decision_trace(
            state,
            deps.build_local_skip_decision("critic_error_fallback"),
        )
        state = deps.finalize_critic_pass(state, output_to_eval)
        await deps.persist_moral_audit_log(state, output_to_eval)

    await _enforce_moral_halt(state, deps)
    await _enforce_retry_limit(state, deps)

    return state
