"""Pending MFA, HITL, and tool-approval helpers for the Orchestrator facade."""
from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


async def resume_hitl_state(
    *,
    user_id: str,
    user_message: str,
    user_prompt: Optional[Dict[str, Any]],
    pending_hitl_state: Dict[str, Any],
    ledger_memory: Any,
    normalize_state: Callable[[Dict[str, Any]], Dict[str, Any]],
    ttl_seconds: int,
    time_fn: Callable[[], float],
    log: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    if user_id not in pending_hitl_state:
        return None

    active_logger = log or logger
    hitl_entry = pending_hitl_state[user_id]
    age = time_fn() - hitl_entry.get("_hitl_created_at", 0)
    if age > ttl_seconds:
        pending_hitl_state.pop(user_id, None)
        try:
            await ledger_memory.clear_hitl_state(user_id)
        except Exception:
            pass
        active_logger.warning(
            "Expired stale pending_hitl_state for %s (age %.0fs); creating fresh state.",
            user_id,
            age,
        )
        return None

    state = normalize_state(pending_hitl_state.pop(user_id))
    previous_iteration_count = int(state.get("iteration_count", 0) or 0)
    previous_critic_feedback = state.get("critic_feedback")
    try:
        await ledger_memory.clear_hitl_state(user_id)
    except Exception as exc:
        active_logger.warning("Failed to clear HITL state from DB for %s: %s", user_id, exc)
    state["admin_guidance"] = user_message
    state["user_input"] += f"\n[ADMIN GUIDANCE: {user_message}]"
    state["user_prompt"] = dict(user_prompt or {})
    state["iteration_count"] = 0
    state["current_plan"] = []
    state["_energy_gate_cleared"] = False
    state["hitl_count"] = state.get("hitl_count", 0) + 1
    state["energy_remaining"] = min(state.get("energy_remaining", 0) + 50, 75)
    if state["hitl_count"] >= 3 or (
        previous_iteration_count >= 3
        and previous_critic_feedback != "PASS"
    ):
        active_logger.warning(
            "HITL cycle limit (3) reached for user %s. Abandoning task: %s",
            user_id,
            state.get("user_input", "")[:80],
        )
        state["final_response"] = (
            "This task has been attempted 3 times with admin guidance and "
            "could not be completed. The request has been abandoned to prevent "
            "an infinite loop. Please rephrase or break it into smaller steps."
        )
    return normalize_state(state)


async def try_resume_mfa(
    *,
    user_id: str,
    user_message: str,
    pending_mfa: Dict[str, Any],
    ledger_memory: Any,
    execute_tool: Callable[[str, Dict[str, Any]], Awaitable[Any]],
    handle_blocked_result: Callable[[Any, str, Dict[str, Any]], Awaitable[str]],
    verify_mfa_challenge: Callable[[str], bool],
    fire_and_forget: Callable[[Awaitable[Any]], Any],
    ttl_seconds: int,
    time_fn: Callable[[], float],
    log: Optional[logging.Logger] = None,
) -> Optional[str]:
    if user_id not in pending_mfa:
        return None
    active_logger = log or logger
    pending_tool = pending_mfa[user_id]
    age = time_fn() - pending_tool.get("_created_at", 0)
    if age > ttl_seconds:
        del pending_mfa[user_id]
        fire_and_forget(ledger_memory.clear_mfa_state(user_id))
        active_logger.info("Expired stale pending_mfa for %s (age %.0fs).", user_id, age)
        return None
    del pending_mfa[user_id]
    fire_and_forget(ledger_memory.clear_mfa_state(user_id))
    if not verify_mfa_challenge(user_message):
        return "Error: MFA authorization failed. Action aborted."
    exec_result = await execute_tool(pending_tool["name"], pending_tool["arguments"])
    if exec_result.status == "ok":
        return exec_result.content
    return await handle_blocked_result(exec_result, pending_tool.get("user_id", user_id), {})


async def pop_pending_tool_approval_payload(
    *,
    user_id: str,
    pending_tool_approval: Dict[str, Any],
    ledger_memory: Any,
    ttl_seconds: int,
    time_fn: Callable[[], float],
    log: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    if user_id not in pending_tool_approval:
        return None
    active_logger = log or logger
    payload = pending_tool_approval[user_id]
    age = time_fn() - payload.get("_created_at", 0)
    if age > ttl_seconds:
        pending_tool_approval.pop(user_id, None)
        await ledger_memory.clear_pending_approval(user_id)
        active_logger.info("Expired stale pending_tool_approval for %s (age %.0fs).", user_id, age)
        return None
    pending_tool_approval.pop(user_id, None)
    await ledger_memory.clear_pending_approval(user_id)
    return payload


async def try_resume_tool_approval(
    *,
    user_id: str,
    user_message: str,
    pop_payload: Callable[[str], Awaitable[Optional[Dict[str, Any]]]],
    reject_synthesized_tool: Callable[..., Awaitable[str]],
    verify_synthesis_payload_digest: Callable[..., Awaitable[Optional[str]]],
    deploy_approved_synthesized_tool: Callable[..., Awaitable[Any]],
    handle_synthesized_tool_deploy_failure: Callable[..., Awaitable[str]],
) -> Optional[Any]:
    payload = await pop_payload(user_id)
    if payload is None:
        return None

    synthesis = payload["synthesis"]
    original_state = payload["original_state"]
    synthesis_run_id = payload.get("synthesis_run_id") or synthesis.get("synthesis_run_id")
    attempts_used = int(synthesis.get("synthesis_attempts_used") or 0)
    tool_name = str(synthesis.get("tool_name") or "")

    if not user_message.strip().upper().startswith("YES"):
        return await reject_synthesized_tool(
            user_id=user_id,
            synthesis_run_id=synthesis_run_id,
            attempts_used=attempts_used,
            synthesis=synthesis,
        )

    mismatch = await verify_synthesis_payload_digest(
        synthesis=synthesis,
        synthesis_run_id=synthesis_run_id,
        attempts_used=attempts_used,
        tool_name=tool_name,
    )
    if mismatch:
        return mismatch

    try:
        return await deploy_approved_synthesized_tool(
            user_id=user_id,
            tool_name=tool_name,
            synthesis=synthesis,
            synthesis_run_id=synthesis_run_id,
            attempts_used=attempts_used,
            original_state=original_state,
        )
    except Exception as exc:
        return await handle_synthesized_tool_deploy_failure(
            tool_name=tool_name,
            synthesis=synthesis,
            synthesis_run_id=synthesis_run_id,
            attempts_used=attempts_used,
            error=exc,
        )


async def try_resume_tool_approval_compat(
    *,
    user_id: str,
    user_message: str,
    synthesis_pipeline: Any,
    fallback_resume: Callable[[str, str], Awaitable[Optional[Any]]],
) -> Optional[Any]:
    resume_fn = getattr(synthesis_pipeline, "try_resume_tool_approval", None)
    if callable(resume_fn):
        outcome = await resume_fn(user_id, user_message)
        if outcome is None:
            return None
        reply_text = str(getattr(outcome, "reply_text", outcome) or "")
        follow_up_input = getattr(outcome, "follow_up_input", None)
        if follow_up_input is None:
            return reply_text
        follow_up_text = str(follow_up_input or "").strip()
        return {
            "reply_text": reply_text,
            "_deferred_follow_up_input": follow_up_text,
        }
    return await fallback_resume(user_id, user_message)


async def handle_blocked_result(
    *,
    result: Any,
    user_id: str,
    state: Dict[str, Any],
    pending_mfa: Dict[str, Any],
    pending_hitl_state: Dict[str, Any],
    ledger_memory: Any,
    strip_audio_bytes_for_persistence: Callable[[Dict[str, Any]], Dict[str, Any]],
    try_resolve_capability_gap_locally: Callable[[str, Any, Dict[str, Any]], Awaitable[Optional[str]]],
    async_tool_synthesis: Callable[[str, Any, Dict[str, Any]], Awaitable[None]],
    fire_and_forget: Callable[[Awaitable[Any]], Any],
    synthesis_in_progress: Dict[str, float],
    synthesis_lockout_ttl_seconds: int,
    time_fn: Callable[[], float],
) -> str:
    if result.status == "mfa_required":
        pending_mfa[user_id] = {
            "name": result.mfa_tool_name,
            "arguments": result.mfa_arguments,
            "_created_at": time_fn(),
        }
        fire_and_forget(
            ledger_memory.save_mfa_state(user_id, result.mfa_tool_name, result.mfa_arguments)
        )
        return "SECURITY LOCK: Provide the authorization passphrase to continue."

    if result.status == "hitl_required":
        state["_hitl_question"] = result.hitl_message
        state["_hitl_created_at"] = time_fn()
        pending_state_payload = strip_audio_bytes_for_persistence(state)
        pending_hitl_state[user_id] = pending_state_payload
        fire_and_forget(ledger_memory.save_hitl_state(user_id, pending_state_payload))
        return result.hitl_message

    if result.status == "capability_gap":
        local_resolution = await try_resolve_capability_gap_locally(user_id, result, state)
        if local_resolution is not None:
            return local_resolution

        ts = synthesis_in_progress.get(user_id)
        if ts is not None:
            age = time_fn() - ts
            if age < synthesis_lockout_ttl_seconds:
                return (
                    "Tool synthesis already in progress for this request. "
                    "Please wait for the current synthesis to complete."
                )
            del synthesis_in_progress[user_id]

        synthesis_in_progress[user_id] = time_fn()
        fire_and_forget(async_tool_synthesis(user_id, result, state))
        return (
            f"I identified a capability gap: {result.gap_description}. "
            f"Requesting tool synthesis from System 2..."
        )

    return f"An unexpected router status was received: {result.status}"


async def persist_hitl_state_from_error(
    *,
    state: Dict[str, Any],
    hitl_error: Exception,
    user_id: str,
    pending_hitl_state: Dict[str, Any],
    admin_user_id: str,
    ledger_memory: Any,
    strip_audio_bytes_for_persistence: Callable[[Dict[str, Any]], Dict[str, Any]],
    extract_heartbeat_task_id: Callable[[str], Optional[int]],
    fire_and_forget: Callable[[Awaitable[Any]], Any],
    outbound_queue: Any,
    time_fn: Callable[[], float],
) -> str:
    state["_hitl_question"] = str(hitl_error)
    state["_hitl_created_at"] = time_fn()
    pending_owner_id = str(admin_user_id or user_id)
    if user_id == "heartbeat" and pending_owner_id == "heartbeat":
        pending_owner_id = str(os.getenv("ADMIN_USER_ID", "heartbeat"))
    heartbeat_task_id = extract_heartbeat_task_id(str(state.get("user_input") or ""))
    if heartbeat_task_id is not None:
        state["_heartbeat_origin_task_id"] = heartbeat_task_id
    pending_state_payload = strip_audio_bytes_for_persistence(state)
    pending_hitl_state[pending_owner_id] = pending_state_payload
    fire_and_forget(ledger_memory.save_hitl_state(pending_owner_id, pending_state_payload))
    if user_id == "heartbeat" and outbound_queue is not None:
        await outbound_queue.put(str(hitl_error))
    return str(hitl_error)
