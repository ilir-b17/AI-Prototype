"""Graph execution loop helpers for the Orchestrator facade."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GraphLoopDeps:
    normalize_state: Callable[[Dict[str, Any]], Dict[str, Any]]
    consume_blocked_result: Callable[[Dict[str, Any], str], Awaitable[Optional[str]]]
    apply_critic_retry_instructions: Callable[[Dict[str, Any]], None]
    run_graph_pass: Callable[[Dict[str, Any], str], Awaitable[tuple[Dict[str, Any], Optional[str]]]]
    reset_after_critic_rejection: Callable[[Dict[str, Any]], None]
    ensure_final_response: Callable[[Dict[str, Any], int], None]
    finalize_user_response: Callable[[str, str, str], Awaitable[str]]
    sanitizer: Callable[[str], str]
    log: logging.Logger


@dataclass
class UserTurnDeps:
    extract_audio_bytes: Callable[[Dict[str, Any]], bytes]
    load_state: Callable[..., Awaitable[Dict[str, Any]]]
    sanitizer: Callable[[str], str]
    remember_user_profile: Callable[[str, str], Awaitable[bool]]
    remember_assistant_identity: Callable[[str], Awaitable[Optional[str]]]
    vector_memory: Any
    try_goal_planning_response: Callable[[Dict[str, Any]], Awaitable[Optional[str]]]
    try_fast_path_response: Callable[[Dict[str, Any]], Awaitable[Optional[str]]]
    finalize_user_response: Callable[[str, str, str], Awaitable[str]]
    run_graph_loop: Callable[[Dict[str, Any], str, str], Awaitable[str]]
    is_error_response: Callable[[str], bool]
    ledger_memory: Any
    persist_hitl_state_from_error: Callable[[Dict[str, Any], Exception, str], Awaitable[str]]
    hitl_error_type: type[Exception]
    log: logging.Logger


def has_ready_final_response(state: Dict[str, Any]) -> bool:
    return bool(state.get("final_response") and not state.get("current_plan"))


async def consume_blocked_result(
    state: Dict[str, Any],
    user_id: str,
    *,
    blocked_key: str,
    handle_blocked_result: Callable[[Any, str, Dict[str, Any]], Awaitable[str]],
) -> Optional[str]:
    blocked_result = state.pop(blocked_key, None)
    if blocked_result is None:
        return None
    return await handle_blocked_result(blocked_result, user_id, state)


def apply_critic_retry_instructions(state: Dict[str, Any]) -> None:
    if state["iteration_count"] > 0:
        constraints = [
            str(item).strip()
            for item in (state.get("moral_remediation_constraints", []) or [])
            if str(item).strip()
        ]
        constraint_clause = ""
        if constraints:
            constraint_clause = " Remediation constraints: " + " | ".join(constraints[:6]) + "."
        state["critic_instructions"] = (
            f"[CRITIC FEEDBACK: {state['critic_feedback']}.{constraint_clause} Fix your output.]"
        )


async def run_manual_graph_pass(
    *,
    state: Dict[str, Any],
    user_id: str,
    supervisor_node: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    execute_workers_node: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    critic_node: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    consume_blocked_result: Callable[[Dict[str, Any], str], Awaitable[Optional[str]]],
) -> tuple[Dict[str, Any], Optional[str]]:
    for node in (supervisor_node, execute_workers_node, critic_node):
        state = await node(state)
        blocked_response = await consume_blocked_result(state, user_id)
        if blocked_response is not None:
            return state, blocked_response
    return state, None


async def run_graph_pass(
    *,
    state: Dict[str, Any],
    user_id: str,
    compiled_graph: Any,
    consume_blocked_result: Callable[[Dict[str, Any], str], Awaitable[Optional[str]]],
    run_manual_graph_pass: Callable[[Dict[str, Any], str], Awaitable[tuple[Dict[str, Any], Optional[str]]]],
) -> tuple[Dict[str, Any], Optional[str]]:
    if compiled_graph is not None:
        state = await compiled_graph.ainvoke(state)
        return state, await consume_blocked_result(state, user_id)
    return await run_manual_graph_pass(state, user_id)


def reset_after_critic_rejection(state: Dict[str, Any]) -> None:
    state["final_response"] = ""
    state["worker_outputs"] = {}
    state["current_plan"] = []
    state["moral_halt_required"] = False
    state["moral_halt_summary"] = ""


def ensure_final_response(state: Dict[str, Any], max_iterations: int) -> None:
    if state["iteration_count"] >= max_iterations and state.get("critic_feedback") != "PASS":
        state["final_response"] = "Unable to fulfill this request - output repeatedly failed internal safety checks."
    if not state.get("final_response"):
        state["final_response"] = "No valid response could be generated."


def is_error_response(response: str, prefixes: tuple[str, ...]) -> bool:
    return any(response.startswith(prefix) for prefix in prefixes)


async def _run_text_memory_hooks(
    *,
    user_id: str,
    user_message: str,
    deps: UserTurnDeps,
) -> None:
    profile_updated = await deps.remember_user_profile(user_id, user_message)
    await deps.remember_assistant_identity(user_message)
    lowered_message = user_message.lower()
    explicit_memory_request = any(
        marker in lowered_message
        for marker in ("remember that", "please remember", "remember i ", "remember my ")
    )
    if explicit_memory_request and not profile_updated and deps.vector_memory is not None:
        await deps.vector_memory.add_memory_async(
            text=user_message,
            metadata={"type": "explicit_memory", "source": "user_request"},
        )


async def _try_user_turn_fast_paths(
    *,
    state: Dict[str, Any],
    user_id: str,
    user_message: str,
    deps: UserTurnDeps,
) -> Optional[str]:
    reply = await deps.try_goal_planning_response(state)
    if reply is not None:
        return await deps.finalize_user_response(user_id, user_message, reply)

    reply = await deps.try_fast_path_response(state)
    if reply is not None:
        return await deps.finalize_user_response(user_id, user_message, reply)
    return None


async def _mark_heartbeat_task_complete(
    *,
    state: Dict[str, Any],
    response: str,
    deps: UserTurnDeps,
) -> None:
    heartbeat_task_id = state.get("_heartbeat_origin_task_id")
    if heartbeat_task_id is None or deps.is_error_response(str(response)):
        return
    try:
        await deps.ledger_memory.update_objective_status(int(heartbeat_task_id), "completed")
    except Exception as exc:
        deps.log.warning("Failed to mark heartbeat task %s completed: %s", heartbeat_task_id, exc)


async def run_graph_loop(
    state: Dict[str, Any],
    user_id: str,
    user_message: str,
    *,
    deps: GraphLoopDeps,
) -> str:
    state = deps.normalize_state(state)
    max_iterations = 3

    if has_ready_final_response(state):
        return deps.sanitizer(state["final_response"])

    while state["iteration_count"] < max_iterations:
        blocked_response = await deps.consume_blocked_result(state, user_id)
        if blocked_response is not None:
            return blocked_response

        deps.apply_critic_retry_instructions(state)
        state, blocked_response = await deps.run_graph_pass(state, user_id)
        if blocked_response is not None:
            return blocked_response

        if state["critic_feedback"] == "PASS":
            state["critic_instructions"] = ""
            break
        deps.log.warning("Critic rejected output on iteration %s", state["iteration_count"])
        deps.reset_after_critic_rejection(state)

    deps.ensure_final_response(state, max_iterations)
    return await deps.finalize_user_response(user_id, user_message, state["final_response"])


async def run_user_turn_locked(
    *,
    user_id: str,
    user_message: str,
    user_prompt: Dict[str, Any],
    deps: UserTurnDeps,
) -> str:
    has_audio_prompt = bool(deps.extract_audio_bytes(user_prompt))

    state = await deps.load_state(
        user_id,
        user_message,
        user_prompt=user_prompt,
    )
    if state.get("final_response") and not state.get("current_plan"):
        return deps.sanitizer(state["final_response"])

    if not has_audio_prompt:
        await _run_text_memory_hooks(
            user_id=user_id,
            user_message=user_message,
            deps=deps,
        )
    else:
        deps.log.info("Audio prompt detected for %s; bypassing text-only fast-path memory hooks.", user_id)

    if not has_audio_prompt:
        fast_path_response = await _try_user_turn_fast_paths(
            state=state,
            user_id=user_id,
            user_message=user_message,
            deps=deps,
        )
        if fast_path_response is not None:
            return fast_path_response

    try:
        response = await deps.run_graph_loop(state, user_id, user_message)
        await _mark_heartbeat_task_complete(state=state, response=response, deps=deps)
        return response
    except deps.hitl_error_type as hitl_err:
        return await deps.persist_hitl_state_from_error(state, hitl_err, user_id)
    except Exception as exc:
        deps.log.error("Graph execution failed: %s", exc, exc_info=True)
        return "An internal error occurred."
