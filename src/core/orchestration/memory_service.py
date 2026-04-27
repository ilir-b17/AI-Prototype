"""Memory and state helpers for the Orchestrator facade."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.core.orchestration import pending_state

logger = logging.getLogger(__name__)


async def remember_user_profile(
    *,
    core_memory: Any,
    user_id: str,
    user_message: str,
    extract_updates: Callable[[str], Dict[str, Any]],
) -> bool:
    if not user_id or core_memory is None:
        return False

    updates = extract_updates(user_message)
    if not updates:
        return False

    core_state = await core_memory.get_all()
    profiles = dict(core_state.get("user_profiles", {}) or {})
    profile = dict(profiles.get(user_id, {}) or {})
    changed = False

    for key, value in updates.items():
        if profile.get(key) != value:
            profile[key] = value
            changed = True

    if not changed:
        return False

    profiles[user_id] = profile
    await core_memory.update("user_profiles", profiles)
    return True


async def get_user_profile(*, core_memory: Any, user_id: str) -> Dict[str, Any]:
    if not user_id or core_memory is None:
        return {}
    core_state = await core_memory.get_all()
    profiles = dict(core_state.get("user_profiles", {}) or {})
    return dict(profiles.get(user_id, {}) or {})


async def remember_assistant_identity(
    *,
    core_memory: Any,
    user_message: str,
    extract_update: Callable[[str], Optional[str]],
) -> Optional[str]:
    if core_memory is None:
        return None

    assistant_name = extract_update(user_message)
    if not assistant_name:
        return None

    core_state = await core_memory.get_all()
    assistant_profile = dict(core_state.get("assistant_profile", {}) or {})
    if assistant_profile.get("name") == assistant_name:
        return assistant_name

    assistant_profile["name"] = assistant_name
    await core_memory.update("assistant_profile", assistant_profile)
    return assistant_name


async def get_assistant_name(*, core_memory: Any) -> str:
    if core_memory is None:
        return "AIDEN"

    core_state = await core_memory.get_all()
    assistant_profile = dict(core_state.get("assistant_profile", {}) or {})
    assistant_name = str(assistant_profile.get("name", "")).strip()
    return assistant_name or "AIDEN"


async def get_archival_context(
    *,
    vector_memory: Any,
    query: str,
    max_chunk_chars: int,
    max_total_chars: int,
    log: Optional[logging.Logger] = None,
) -> str:
    if not query:
        return ""
    active_logger = log or logger
    try:
        results = await vector_memory.query_memory_async(query, n_results=3)
        if not results:
            return ""
        lines = ["<Archival_Memory>"]
        total = 0
        for item in results:
            snippet = (item.get("document", "") or "")[:max_chunk_chars]
            if not snippet:
                continue
            if total + len(snippet) > max_total_chars:
                break
            lines.append(f"  <Memory>{snippet}</Memory>")
            total += len(snippet)
        lines.append("</Archival_Memory>")
        return "\n".join(lines)
    except Exception as exc:
        active_logger.warning("Archival memory lookup failed: %s", exc)
        return ""


async def save_memory_async(
    *,
    vector_memory: Any,
    text: str,
    memory_save_threshold: int,
    utc_timestamp_format: str,
    log: Optional[logging.Logger] = None,
) -> None:
    active_logger = log or logger
    try:
        if len(text) < memory_save_threshold:
            return
        await vector_memory.add_memory_async(
            text=text,
            metadata={
                "type": "conversation",
                "timestamp": datetime.now(timezone.utc).strftime(utc_timestamp_format),
            },
        )
    except Exception as exc:
        active_logger.warning("Async memory save failed: %s", exc)


async def consolidate_memory(
    *,
    user_id: str,
    ledger_memory: Any,
    vector_memory: Any,
    core_memory: Any,
    nocturnal_consolidation: Any,
    score_nocturnal_candidates: Callable[[List[Dict[str, str]]], Awaitable[Any]],
    redactor: Callable[..., str],
    threshold: float,
    log: Optional[logging.Logger] = None,
) -> None:
    active_logger = log or logger
    try:
        turns = await ledger_memory.get_chat_history(user_id, limit=20)
        if len(turns) < 6:
            return

        recent_logs = await ledger_memory.get_logs(limit=20)
        critic_logs = [
            entry for entry in recent_logs
            if "critic" in str(entry.get("message", "")).lower()
            or str(entry.get("context", "")).lower().find("critic") != -1
        ]
        try:
            blueprint_entries = await vector_memory.query_memory_async(
                "System 2 Reasoning Blueprint",
                n_results=5,
            )
        except Exception:
            blueprint_entries = []

        filtered_candidates = await nocturnal_consolidation.extract_and_filter_candidates(
            user_id=user_id,
            chat_history=turns,
            worker_outputs={},
            critic_feedback="",
            blueprint_entries=blueprint_entries,
            ledger_logs=critic_logs,
            vector_memory=vector_memory,
        )

        if not filtered_candidates:
            await ledger_memory.trim_chat_history(user_id, keep_last=5)
            return

        scored_candidates = await nocturnal_consolidation.score_candidates_with_system2(
            filtered_candidates,
            route_to_system_2=score_nocturnal_candidates,
            redactor=redactor,
            threshold=threshold,
        )
        write_result = await nocturnal_consolidation.write_back_scored_candidates(
            scored_candidates,
            core_memory=core_memory,
            vector_memory=vector_memory,
            ledger_memory=ledger_memory,
            threshold=threshold,
        )

        await ledger_memory.trim_chat_history(user_id, keep_last=5)
        if write_result.get("stored_total", 0) > 0:
            active_logger.info(
                "Nocturnal consolidation stored memories for %s: %s",
                user_id,
                write_result,
            )
    except asyncio.TimeoutError:
        active_logger.warning("Memory consolidation skipped: LLM did not respond within 90s")
    except Exception as exc:
        active_logger.warning(
            "Memory consolidation failed: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )


async def load_state(
    *,
    user_id: str,
    user_message: str,
    user_prompt: Optional[Dict[str, Any]],
    pending_hitl_state: Dict[str, Any],
    ledger_memory: Any,
    new_state: Callable[..., Dict[str, Any]],
    normalize_state: Callable[[Dict[str, Any]], Dict[str, Any]],
    ttl_seconds: int,
    recent_chat_history_limit: int,
    time_fn: Callable[[], float],
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    resumed = await pending_state.resume_hitl_state(
        user_id=user_id,
        user_message=user_message,
        user_prompt=user_prompt,
        pending_hitl_state=pending_hitl_state,
        ledger_memory=ledger_memory,
        normalize_state=normalize_state,
        ttl_seconds=ttl_seconds,
        time_fn=time_fn,
        log=log,
    )
    if resumed is not None:
        return normalize_state(resumed)

    state = new_state(user_id, user_message, user_prompt=user_prompt)
    state["user_prompt"] = dict(user_prompt or {})
    active_logger = log or logger
    if user_id != "heartbeat":
        try:
            state["chat_history"] = await ledger_memory.get_chat_history(
                user_id,
                limit=recent_chat_history_limit,
            )
        except Exception as exc:
            active_logger.warning("Failed to load chat history for %s: %s", user_id, exc)
    return normalize_state(state)


async def persist_chat_turns(
    *,
    user_id: str,
    user_message: str,
    final_resp: str,
    turn_failed: bool,
    ledger_memory: Any,
    increment_consolidation_turn_count: Callable[[str], bool],
    consolidate_memory: Callable[[str], Awaitable[None]],
    fire_and_forget: Callable[[Awaitable[Any]], Any],
    consolidation_turn_counts: Dict[str, int],
    log: Optional[logging.Logger] = None,
) -> None:
    if user_id == "heartbeat" or turn_failed:
        return
    active_logger = log or logger
    try:
        await ledger_memory.save_chat_turn(user_id, "user", user_message)
        await ledger_memory.save_chat_turn(user_id, "assistant", final_resp)
    except Exception as exc:
        active_logger.warning("Failed to save chat turn for %s: %s", user_id, exc)

    if increment_consolidation_turn_count(user_id):
        fire_and_forget(consolidate_memory(user_id))

    try:
        current = int(consolidation_turn_counts.get(user_id, 0) or 0)
        coro = ledger_memory.set_system_state(
            f"consolidation_turns:{user_id}", str(current)
        )
        if asyncio.iscoroutine(coro):
            fire_and_forget(coro)
    except Exception as exc:
        active_logger.debug("Consolidation counter persistence skipped: %s", exc)


def increment_consolidation_turn_count(
    *,
    counts: Any,
    user_id: str,
    max_size: int,
    trigger_turns: int,
) -> tuple[OrderedDict, bool]:
    if counts is None:
        counts = OrderedDict()
    if not isinstance(counts, OrderedDict):
        counts = OrderedDict(counts)

    current_count = int(counts.get(user_id, 0) or 0)
    if user_id in counts:
        counts.move_to_end(user_id)
    else:
        counts[user_id] = current_count

    counts[user_id] = current_count + 1
    while len(counts) > max(1, max_size):
        counts.popitem(last=False)

    if counts.get(user_id, 0) >= max(1, trigger_turns):
        counts[user_id] = 0
        return counts, True
    return counts, False


def schedule_response_memory_save(
    *,
    fire_and_forget: Callable[[Awaitable[Any]], Any],
    save_memory_async: Callable[[str], Awaitable[None]],
    user_message: str,
    final_resp: str,
) -> None:
    fire_and_forget(save_memory_async(f"User: {user_message}\nAssistant: {final_resp}"))


def recover_displayable_response(raw_response: Any, sanitized_response: Any) -> str:
    sanitized = str(sanitized_response or "").strip()
    if sanitized:
        return sanitized

    raw = str(raw_response or "").strip()
    if raw:
        cleaned = re.sub(r"<agent_output>.*?</agent_output>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"WORKERS:\s*\[.*?\]\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            return cleaned

    return "I couldn't format a response for that request. Please retry."


async def finalize_user_response(
    *,
    user_id: str,
    user_message: str,
    response: str,
    sanitizer: Callable[[str], str],
    persist_chat_turns: Callable[..., Awaitable[None]],
    schedule_response_memory_save: Callable[[str, str], None],
) -> str:
    sanitized = sanitizer(response)
    final_resp = recover_displayable_response(response, sanitized)
    await persist_chat_turns(user_id, user_message, final_resp)
    schedule_response_memory_save(user_message, final_resp)
    return final_resp
