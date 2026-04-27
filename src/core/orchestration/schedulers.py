"""Background schedulers and task-management helpers for the Orchestrator facade."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


async def notify_admin(outbound_queue: Optional[asyncio.Queue], message: str, *, log: logging.Logger) -> None:
    if outbound_queue is None:
        log.info("[Admin notification (no queue)]: %s", message)
        return
    if outbound_queue.full():
        try:
            dropped_message = outbound_queue.get_nowait()
            outbound_queue.task_done()
            log.warning(
                "Dropped oldest outbound message due to full queue: %s",
                str(dropped_message)[:180],
            )
        except asyncio.QueueEmpty:
            pass
    try:
        outbound_queue.put_nowait(message)
    except asyncio.QueueFull:
        await outbound_queue.put(message)


def heartbeat_result_indicates_failure(result: str) -> bool:
    lowered = str(result or "").lower()
    return any(
        marker in lowered
        for marker in ("error", "cannot", "unable", "failed", "i cannot", "i am unable")
    )


def build_heartbeat_execution_prompt(task: Dict[str, Any]) -> str:
    return (
        f"[HEARTBEAT TASK #{task['id']}]: {task['title']}\n"
        "You MUST execute this task right now by calling the appropriate tools. "
        "Do NOT describe what you plan to do - use your tools and report the actual results. "
        "If the task requires storing data, call update_core_memory or update_ledger explicitly. "
        "If you cannot complete it, explain exactly why."
    )


def is_executable_heartbeat_task_candidate(
    node: Dict[str, Any],
    unresolved_task_ids: set[int],
    parent_ids: set[int],
) -> bool:
    if str(node.get("tier") or "") != "Task":
        return False
    if str(node.get("status") or "").lower() not in {"pending", "deferred_due_to_energy"}:
        return False

    task_id = int(node.get("id") or 0)
    if task_id <= 0:
        return False
    return task_id not in unresolved_task_ids and task_id not in parent_ids


async def select_executable_heartbeat_tasks(
    *,
    ledger_memory: Any,
    is_candidate: Callable[[Dict[str, Any], set[int], set[int]], bool],
) -> List[Dict[str, Any]]:
    candidates = await ledger_memory.get_energy_evaluation_candidates(
        statuses=["pending", "deferred_due_to_energy"],
    )
    if not candidates:
        return []

    active_nodes = await ledger_memory.get_active_objective_tree()
    unresolved_rows = await ledger_memory.get_tasks_with_unresolved_dependencies(
        statuses=["pending", "deferred_due_to_energy"],
    )
    unresolved_task_ids = {int(row["id"]) for row in unresolved_rows}
    parent_ids = {
        int(node["parent_id"])
        for node in active_nodes
        if node.get("parent_id") is not None
    }

    executable: List[Dict[str, Any]] = []
    for candidate in candidates:
        task = dict(candidate.get("task") or {})
        domain = str(task.get("agent_domain") or "").strip().lower()
        if domain not in {"", "aiden"}:
            continue
        if is_candidate(task, unresolved_task_ids, parent_ids):
            executable.append(candidate)

    executable.sort(
        key=lambda candidate: (
            int((candidate.get("task") or {}).get("priority") or 999),
            int((candidate.get("task") or {}).get("estimated_energy") or 999999),
            int((candidate.get("task") or {}).get("id") or 0),
        )
    )
    return executable


async def select_executable_heartbeat_task(
    select_tasks: Callable[[], Awaitable[List[Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    candidates = await select_tasks()
    if not candidates:
        return None
    return dict(candidates[0].get("task") or {})


def summarize_domain_task_result(result_payload: Dict[str, Any], *, max_length: int) -> str:
    for key in ("summary", "result_summary", "final_answer", "message", "result", "content"):
        value = result_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:max_length]
    compact = json.dumps(result_payload, ensure_ascii=False)
    return compact[:max_length] if compact else "No summary provided"


def extract_domain_task_error(result_payload: Dict[str, Any], *, max_length: int) -> str:
    for key in ("error", "message", "details", "reason"):
        value = result_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:max_length]
    compact = json.dumps(result_payload, ensure_ascii=False)
    return compact[:max_length] if compact else "Unknown error"


async def _query_pending_domain_notifications(*, ledger_memory: Any, log: logging.Logger) -> List[Dict[str, Any]]:
    query_pending = getattr(ledger_memory, "get_completed_tasks_pending_notification", None)
    if not callable(query_pending):
        return []
    try:
        return list(await query_pending(limit=25) or [])
    except Exception as exc:
        log.warning("Domain result monitor: query failed: %s", exc, exc_info=True)
        return []


async def _load_domain_task_result(*, ledger_memory: Any, task_id: int) -> Dict[str, Any]:
    get_task_result = getattr(ledger_memory, "get_task_result", None)
    if not callable(get_task_result):
        return {}
    return dict(await get_task_result(task_id) or {})


async def _notify_one_domain_task(
    *,
    task: Dict[str, Any],
    ledger_memory: Any,
    notify_admin_fn: Callable[[str], Awaitable[None]],
    summarize_result: Callable[[Dict[str, Any]], str],
    extract_error: Callable[[Dict[str, Any]], str],
    notifiable_domains: set[str],
    log: logging.Logger,
) -> Optional[int]:
    task_id = int(task.get("id") or 0)
    if task_id <= 0:
        return None

    domain = str(task.get("agent_domain") or "").strip().lower()
    if domain not in notifiable_domains:
        log.debug(
            "Domain result monitor: skipping task #%s for non-notifiable domain '%s'.",
            task_id,
            domain,
        )
        return None

    title = str(task.get("title") or "").strip() or f"Task #{task_id}"
    status = str(task.get("status") or "").strip().lower()
    result_payload = await _load_domain_task_result(
        ledger_memory=ledger_memory,
        task_id=task_id,
    )
    try:
        if status == "completed":
            await notify_admin_fn(
                f"Google Agent completed: {title} — {summarize_result(result_payload)}"
            )
            return task_id
        if status in {"failed", "blocked"}:
            await notify_admin_fn(
                f"Google Agent blocked: {title} — {extract_error(result_payload)}. Reply to investigate."
            )
            return task_id
    except Exception as exc:
        log.warning(
            "Domain result monitor: notification failed for task #%s: %s",
            task_id,
            exc,
            exc_info=True,
        )
    return None


async def _mark_domain_tasks_notified(
    *,
    ledger_memory: Any,
    notified_ids: List[int],
    log: logging.Logger,
) -> None:
    if not notified_ids:
        return
    mark_notified = getattr(ledger_memory, "mark_tasks_admin_notified", None)
    if not callable(mark_notified):
        return
    try:
        await mark_notified(notified_ids)
    except Exception as exc:
        log.warning("Domain result monitor: failed to mark notified tasks: %s", exc, exc_info=True)


async def notify_completed_domain_tasks(
    *,
    ledger_memory: Any,
    notify_admin_fn: Callable[[str], Awaitable[None]],
    summarize_result: Callable[[Dict[str, Any]], str],
    extract_error: Callable[[Dict[str, Any]], str],
    notifiable_domains: set[str],
    log: logging.Logger,
) -> None:
    notified_ids: List[int] = []
    for task in await _query_pending_domain_notifications(ledger_memory=ledger_memory, log=log):
        notified_id = await _notify_one_domain_task(
            task=task,
            ledger_memory=ledger_memory,
            notify_admin_fn=notify_admin_fn,
            summarize_result=summarize_result,
            extract_error=extract_error,
            notifiable_domains=notifiable_domains,
            log=log,
        )
        if notified_id is not None:
            notified_ids.append(notified_id)
    await _mark_domain_tasks_notified(
        ledger_memory=ledger_memory,
        notified_ids=notified_ids,
        log=log,
    )


async def handle_heartbeat_task_failure(
    *,
    task: Dict[str, Any],
    reason: str,
    result_excerpt: str,
    ledger_memory: Any,
    increment_failure_count: Callable[[int], Awaitable[int]],
    notify_admin_fn: Callable[[str], Awaitable[None]],
    failure_strikes: int,
    log: logging.Logger,
) -> None:
    task_id = int(task["id"])
    strike_count = await increment_failure_count(task_id)

    if strike_count >= failure_strikes:
        await ledger_memory.update_objective_status(task_id, "blocked")
        await ledger_memory.ensure_parent_chain_active(task_id)
        await notify_admin_fn(
            "HITL REQUIRED: Task blocked after repeated heartbeat failures.\n"
            f"  Task #{task_id}: {task['title']}\n"
            f"  Strikes: {strike_count}\n"
            f"  Reason: {reason}\n"
            f"  Latest result: {result_excerpt[:220]}"
        )
        log.warning(
            "Heartbeat: Task #%s blocked after %s failures (%s)",
            task_id,
            strike_count,
            reason,
        )
        return

    await ledger_memory.update_objective_status(task_id, "pending")
    log.warning(
        "Heartbeat: Task #%s failure strike %s/%s (%s)",
        task_id,
        strike_count,
        failure_strikes,
        reason,
    )


async def execute_heartbeat_task(
    *,
    task: Dict[str, Any],
    decision: Any,
    ledger_memory: Any,
    process_message: Callable[..., Awaitable[str]],
    get_predictive_energy_budget_remaining: Callable[[], Awaitable[int]],
    build_prompt: Callable[[Dict[str, Any]], str],
    result_indicates_failure: Callable[[str], bool],
    handle_failure: Callable[..., Awaitable[None]],
    clear_failure_count: Callable[[int], Awaitable[None]],
    notify_admin_fn: Callable[[str], Awaitable[None]],
    heartbeat_interval: int,
    log: logging.Logger,
) -> None:
    task_id = int(task["id"])
    log.info(
        "Heartbeat: Accepting executable task #%s: %s (roi=%.2f cost=%s budget=%s)",
        task_id,
        str(task.get("title") or "")[:60],
        decision.roi,
        decision.predicted_cost,
        await get_predictive_energy_budget_remaining(),
    )
    await ledger_memory.update_objective_status(task_id, "active")

    timeout_seconds = heartbeat_interval * 0.9
    prompt = build_prompt(task)

    try:
        result = await asyncio.wait_for(
            process_message(user_message=prompt, user_id="heartbeat"),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        await handle_failure(
            task,
            reason="timeout",
            result_excerpt=f"Timed out after {int(timeout_seconds)}s",
        )
        return
    except Exception as exc:
        await handle_failure(
            task,
            reason="exception",
            result_excerpt=str(exc),
        )
        return

    if result_indicates_failure(result):
        await handle_failure(
            task,
            reason="execution_failure",
            result_excerpt=result,
        )
        return

    await ledger_memory.update_objective_status(task_id, "completed")
    await clear_failure_count(task_id)

    summary = (
        f"Heartbeat completed task #{task_id}:\n"
        f"  Task: {task.get('title', '')}\n"
        f"  Predicted Energy Cost: {decision.predicted_cost}\n"
        f"  Remaining Predictive Budget: {await get_predictive_energy_budget_remaining()}\n"
        f"  Result: {str(result)[:220]}"
    )
    await notify_admin_fn(summary)
    log.info("Heartbeat: Task #%s completed.", task_id)


@dataclass(frozen=True)
class HeartbeatCycleDeps:
    ensure_energy_budget: Callable[[], None]
    predictive_energy_budget_lock: asyncio.Lock
    apply_predictive_energy_tick_locked: Callable[[int], int]
    resolve_energy_replenish_per_heartbeat: Callable[[], int]
    notify_completed_domain_tasks: Callable[[], Awaitable[None]]
    select_executable_heartbeat_tasks: Callable[[], Awaitable[List[Dict[str, Any]]]]
    evaluate_energy_for_context: Callable[..., Awaitable[tuple[Any, Any, int]]]
    build_energy_evaluation_record: Callable[..., Dict[str, Any]]
    defer_heartbeat_task_due_to_energy: Callable[..., Awaitable[None]]
    try_reserve_predictive_energy_budget: Callable[..., Awaitable[bool]]
    get_predictive_energy_budget_remaining: Callable[[], Awaitable[int]]
    energy_roi_engine: Any
    ledger_memory: Any
    execute_heartbeat_task: Callable[..., Awaitable[None]]
    log: logging.Logger


async def run_heartbeat_cycle(
    deps: HeartbeatCycleDeps,
) -> None:
    deps.ensure_energy_budget()
    async with deps.predictive_energy_budget_lock:
        deps.apply_predictive_energy_tick_locked(deps.resolve_energy_replenish_per_heartbeat())
    await deps.notify_completed_domain_tasks()
    deps.log.info("Heartbeat: Querying objective backlog for executable Tasks...")
    candidate_contexts = await deps.select_executable_heartbeat_tasks()

    if not candidate_contexts:
        deps.log.info("Heartbeat: No executable dependency-resolved tasks available.")
        return

    for candidate in candidate_contexts:
        task = dict(candidate.get("task") or {})
        story = candidate.get("story")
        epic = candidate.get("epic")

        task_id = int(task.get("id") or 0)
        if task_id <= 0:
            continue

        evaluation, decision, available_energy = await deps.evaluate_energy_for_context(
            task=task,
            story=story,
            epic=epic,
            additional_context="heartbeat_dispatch",
        )
        evaluation_record = deps.build_energy_evaluation_record(
            evaluation=evaluation,
            decision=decision,
            available_energy=available_energy,
            context="heartbeat",
        )

        if not decision.should_execute:
            await deps.defer_heartbeat_task_due_to_energy(
                task_id=task_id,
                decision=decision,
                evaluation_record=evaluation_record,
                available_energy=available_energy,
            )
            continue

        reserved = await deps.try_reserve_predictive_energy_budget(
            predicted_cost=decision.predicted_cost,
            min_reserve=decision.min_reserve,
            reason=f"heartbeat_task:{task_id}",
        )
        if not reserved:
            current_budget = await deps.get_predictive_energy_budget_remaining()
            race_decision = deps.energy_roi_engine.evaluate(
                estimated_effort=evaluation.estimated_effort,
                expected_value=evaluation.expected_value,
                available_energy=current_budget,
                defer_count=decision.defer_count,
            )
            race_record = deps.build_energy_evaluation_record(
                evaluation=evaluation,
                decision=race_decision,
                available_energy=current_budget,
                context="heartbeat",
            )
            await deps.defer_heartbeat_task_due_to_energy(
                task_id=task_id,
                decision=race_decision,
                evaluation_record=race_record,
                available_energy=current_budget,
                race=True,
            )
            continue

        await deps.ledger_memory.record_task_energy_evaluation(
            task_id,
            evaluation_record,
            clear_next_eligible=True,
        )
        await deps.execute_heartbeat_task(task=task, decision=decision)
        return

    deps.log.info("Heartbeat: All executable tasks were deferred by energy policy.")


async def heartbeat_loop(
    *,
    run_heartbeat_cycle_fn: Callable[[], Awaitable[None]],
    heartbeat_interval: int,
    log: logging.Logger,
) -> None:
    log.info("Heartbeat loop started.")
    heartbeat_lock = asyncio.Lock()
    while True:
        await asyncio.sleep(heartbeat_interval)
        if heartbeat_lock.locked():
            log.warning("Heartbeat: Previous run still in progress. Skipping this cycle.")
            continue
        async with heartbeat_lock:
            try:
                await run_heartbeat_cycle_fn()
            except Exception as exc:
                log.error("Heartbeat error: %s", exc, exc_info=True)


def prune_email_processed_history(
    processed_timestamps: List[float],
    now_ts: Optional[float] = None,
) -> List[float]:
    cutoff = float(now_ts if now_ts is not None else time.time()) - 86400.0
    return [ts for ts in (processed_timestamps or []) if ts >= cutoff]


def _decode_email_poll_payload(raw_result: str, *, log: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    try:
        result = json.loads(str(raw_result or ""))
    except Exception as exc:
        log.warning("Email poll: read_inbox returned invalid JSON: %s", exc, exc_info=True)
        return None

    if str(result.get("status") or "").strip().lower() != "success":
        log.warning(
            "Email poll: inbox read failed: %s",
            str(result.get("message") or "unknown error"),
        )
        return None

    emails = result.get("emails") or []
    if not isinstance(emails, list):
        log.warning("Email poll: invalid inbox payload type for emails: %s", type(emails).__name__)
        return None
    return [item for item in emails if isinstance(item, dict)]


def _build_email_task_payload(item: Dict[str, Any]) -> tuple[str, str]:
    sender = str(item.get("sender") or "unknown sender").strip() or "unknown sender"
    subject = str(item.get("subject") or "(no subject)").strip() or "(no subject)"
    attachment_paths_raw = item.get("attachment_paths") or []
    attachment_paths = (
        [str(path) for path in attachment_paths_raw]
        if isinstance(attachment_paths_raw, list)
        else []
    )
    acceptance_payload = {
        "sender": sender,
        "subject": subject,
        "date": str(item.get("date") or ""),
        "message_id": str(item.get("message_id") or ""),
        "body": str(item.get("body") or ""),
        "attachment_paths": attachment_paths,
    }
    title = f"Process email request: {subject} from {sender}"
    return title, json.dumps(acceptance_payload, ensure_ascii=False)


async def _create_email_task(
    *,
    ledger_memory: Any,
    item: Dict[str, Any],
    log: logging.Logger,
) -> bool:
    try:
        title, acceptance_criteria = _build_email_task_payload(item)
        objective_id = await ledger_memory.add_objective(
            tier="Task",
            title=title,
            estimated_energy=20,
            origin="email",
            acceptance_criteria=acceptance_criteria,
        )
        await ledger_memory.set_task_agent_domain(objective_id, "google")
        log.info("Email poll: created task #%s for new email: %s", objective_id, title)
        return True
    except Exception as exc:
        log.warning("Email poll: failed to create objective from email: %s", exc, exc_info=True)
        return False


async def run_email_poll_cycle(
    *,
    ledger_memory: Any,
    processed_timestamps: List[float],
    read_inbox_fn: Callable[[], Awaitable[str]],
    now_fn: Callable[[], datetime],
    time_fn: Callable[[], float],
    log: logging.Logger,
) -> tuple[datetime, List[float]]:
    last_run_at = now_fn()
    processed_timestamps = prune_email_processed_history(processed_timestamps)
    try:
        raw_result = await read_inbox_fn()
    except Exception as exc:
        log.warning("Email poll: read_inbox failed: %s", exc, exc_info=True)
        return last_run_at, processed_timestamps

    emails = _decode_email_poll_payload(raw_result, log=log)
    if emails is None:
        return last_run_at, processed_timestamps

    for item in emails:
        if await _create_email_task(ledger_memory=ledger_memory, item=item, log=log):
            processed_timestamps.append(time_fn())

    return last_run_at, prune_email_processed_history(processed_timestamps)


async def email_poll_loop(
    *,
    run_email_poll_cycle_fn: Callable[[], Awaitable[None]],
    email_poll_interval: int,
    log: logging.Logger,
) -> None:
    log.info("Email poll loop started.")
    email_poll_lock = asyncio.Lock()
    while True:
        await asyncio.sleep(email_poll_interval)
        if email_poll_lock.locked():
            log.warning("Email poll: Previous run still in progress. Skipping this cycle.")
            continue
        async with email_poll_lock:
            try:
                await run_email_poll_cycle_fn()
            except Exception as exc:
                log.warning("Email poll loop error: %s", exc, exc_info=True)


async def get_email_poll_status(
    *,
    ledger_memory: Any,
    last_run_at: Optional[datetime],
    processed_timestamps: List[float],
) -> Dict[str, Any]:
    pruned = prune_email_processed_history(processed_timestamps)
    last_poll_text = (
        last_run_at.strftime("%Y-%m-%d %H:%M:%S")
        if last_run_at is not None
        else "never"
    )
    pending_google_tasks = await ledger_memory.get_pending_tasks_for_domain("google", limit=10000)
    return {
        "last_poll_time": last_poll_text,
        "emails_processed_last_24h": len(pruned or []),
        "pending_google_tasks": len(pending_google_tasks or []),
    }


async def memory_consolidation_loop(
    *,
    ledger_memory: Any,
    vector_memory: Any,
    consolidate_memory: Callable[[str], Awaitable[None]],
    interval_seconds: int,
    log: logging.Logger,
) -> None:
    log.info("Memory consolidation loop started.")
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            user_ids = await ledger_memory.get_recent_user_ids(limit=20)
            if not user_ids:
                continue
            for user_id in user_ids:
                await consolidate_memory(user_id)
            try:
                deleted = await ledger_memory.prune_old_chat_history(
                    days=int(os.getenv("CHAT_HISTORY_MAX_DAYS", "90")),
                    keep_minimum=int(os.getenv("CHAT_HISTORY_MIN_ROWS", "20")),
                )
                if deleted:
                    log.info("Consolidation pruning: removed %d chat_history rows.", deleted)
            except Exception as exc:
                log.warning("Periodic chat history pruning failed: %s", exc)
            try:
                await asyncio.to_thread(
                    vector_memory.prune_old_memories,
                    int(os.getenv("VECTOR_MEMORY_MAX_DAYS", "180")),
                )
            except Exception as exc:
                log.warning("Periodic vector memory pruning failed: %s", exc)
        except Exception as exc:
            log.warning("Memory consolidation loop error: %s", exc, exc_info=True)


async def sensory_update_loop(
    *,
    refresh_sensory_state: Callable[[], None],
    interval_seconds: int,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        refresh_sensory_state()


def fire_and_forget(
    coro: Awaitable[Any],
    *,
    background_tasks: set[asyncio.Task],
    log: logging.Logger,
) -> asyncio.Task:
    task_name = getattr(coro, "__name__", "")
    if not task_name and hasattr(coro, "cr_code"):
        task_name = getattr(coro.cr_code, "co_name", "")
    if not task_name:
        task_name = repr(coro)

    task = asyncio.create_task(coro)
    background_tasks.add(task)

    def _on_done(done_task: asyncio.Task) -> None:
        background_tasks.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc is not None:
            log.error(
                "Background task '%s' raised an unhandled exception: %s",
                task_name,
                exc,
                exc_info=exc,
            )

    task.add_done_callback(_on_done)
    return task


async def drain_background_tasks(background_tasks: set[asyncio.Task]) -> None:
    tasks = [
        task
        for task in tuple(background_tasks or set())
        if task is not asyncio.current_task()
    ]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        background_tasks.difference_update(tasks)


async def get_user_lock(
    *,
    user_id: str,
    user_locks: Dict[str, asyncio.Lock],
    user_locks_lock: asyncio.Lock,
    max_size: int,
) -> asyncio.Lock:
    async with user_locks_lock:
        if user_id in user_locks:
            user_locks[user_id] = user_locks.pop(user_id)
            return user_locks[user_id]
        lock = asyncio.Lock()
        user_locks[user_id] = lock
        while len(user_locks) > max(1, max_size):
            evict_key = next(
                (
                    key
                    for key, candidate_lock in user_locks.items()
                    if key != user_id and not candidate_lock.locked()
                ),
                None,
            )
            if evict_key is None:
                break
            user_locks.pop(evict_key, None)
        return lock


async def restore_heartbeat_failure_counts(
    *,
    ledger_memory: Any,
    state_key: str,
    log: logging.Logger,
) -> Dict[int, int]:
    try:
        raw_value = await ledger_memory.get_system_state(state_key)
        if not raw_value:
            return {}

        decoded = json.loads(raw_value)
        if not isinstance(decoded, dict):
            return {}

        restored: Dict[int, int] = {}
        for raw_task_id, raw_count in decoded.items():
            try:
                task_id = int(raw_task_id)
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if task_id > 0 and count > 0:
                restored[task_id] = count
        return restored
    except Exception as exc:
        log.warning("Failed to restore heartbeat failure counts: %s", exc)
        return {}


async def persist_heartbeat_failure_counts(
    *,
    ledger_memory: Any,
    counts: Dict[int, int],
    state_key: str,
) -> None:
    payload = {
        str(task_id): int(count)
        for task_id, count in counts.items()
        if int(count) > 0
    }
    await ledger_memory.set_system_state(
        state_key,
        json.dumps(payload, sort_keys=True),
    )


async def increment_heartbeat_failure_count(
    *,
    ledger_memory: Any,
    counts: Dict[int, int],
    task_id: int,
    state_key: str,
) -> int:
    current = int(counts.get(task_id, 0)) + 1
    counts[task_id] = current
    await persist_heartbeat_failure_counts(
        ledger_memory=ledger_memory,
        counts=counts,
        state_key=state_key,
    )
    return current


async def clear_heartbeat_failure_count(
    *,
    ledger_memory: Any,
    counts: Dict[int, int],
    task_id: int,
    state_key: str,
) -> None:
    if task_id in counts:
        counts.pop(task_id, None)
        await persist_heartbeat_failure_counts(
            ledger_memory=ledger_memory,
            counts=counts,
            state_key=state_key,
        )


async def run_tool_synthesis_background(
    *,
    user_id: str,
    state: Dict[str, Any],
    result: Any,
    tool_synthesis_node: Callable[[Dict[str, Any], Any], Awaitable[Any]],
    notify_admin_fn: Callable[[str], Awaitable[None]],
    synthesis_in_progress: Dict[str, float],
    timeout_seconds: float,
    log: logging.Logger,
) -> None:
    try:
        hitl_prompt = await asyncio.wait_for(
            tool_synthesis_node(state, result),
            timeout=float(timeout_seconds),
        )
        await notify_admin_fn(hitl_prompt)
    except asyncio.TimeoutError:
        log.critical(
            "Tool synthesis timed out after %ss for user '%s'.",
            timeout_seconds,
            user_id,
        )
        await notify_admin_fn(
            f"[CRITICAL] Tool synthesis timed out after {timeout_seconds}s "
            f"for user '{user_id}'."
        )
    except Exception as exc:
        log.error("Background tool synthesis failed: %s", exc, exc_info=True)
    finally:
        synthesis_in_progress.pop(user_id, None)
