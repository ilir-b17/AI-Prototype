"""Energy accounting helpers for the Orchestrator facade."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def energy_gate_available(owner: Any) -> bool:
    return (
        getattr(owner, "_predictive_energy_budget_lock", None) is not None
        and hasattr(owner, "_predictive_energy_budget_remaining")
    )


async def get_predictive_energy_budget_remaining(
    owner: Any,
    *,
    gate_available: Optional[Callable[[], bool]] = None,
) -> int:
    available = gate_available() if callable(gate_available) else energy_gate_available(owner)
    if not available:
        return max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
    async with owner._predictive_energy_budget_lock:
        return max(0, int(owner._predictive_energy_budget_remaining))


def resolve_energy_replenish_per_hour(*, log: Optional[logging.Logger] = None) -> float:
    raw_rate = str(os.getenv("ENERGY_REPLENISH_PER_HOUR", "30") or "").strip()
    try:
        rate = float(raw_rate)
    except ValueError:
        active_logger = log or logger
        active_logger.warning("Invalid ENERGY_REPLENISH_PER_HOUR=%r. Falling back to 30.", raw_rate)
        rate = 30.0
    return max(0.0, rate)


def resolve_energy_replenish_per_heartbeat(*, log: Optional[logging.Logger] = None) -> int:
    raw_value = str(os.getenv("ENERGY_REPLENISH_PER_HEARTBEAT", "2") or "").strip()
    try:
        amount = int(raw_value)
    except ValueError:
        active_logger = log or logger
        active_logger.warning("Invalid ENERGY_REPLENISH_PER_HEARTBEAT=%r. Falling back to 2.", raw_value)
        amount = 2
    return max(0, amount)


def resolve_energy_replenish_per_turn(owner_cls: type, *, log: Optional[logging.Logger] = None) -> int:
    raw_heartbeat = str(os.getenv("ENERGY_REPLENISH_PER_HEARTBEAT", "") or "").strip()
    if raw_heartbeat:
        return resolve_energy_replenish_per_heartbeat(log=log)

    raw_deprecated = str(os.getenv("ENERGY_REPLENISH_PER_TURN", "") or "").strip()
    if raw_deprecated:
        warn_deprecated_energy_replenish_turn_env_once(owner_cls, log=log)
        try:
            return max(0, int(raw_deprecated))
        except ValueError:
            active_logger = log or logger
            active_logger.warning(
                "Invalid ENERGY_REPLENISH_PER_TURN=%r. Falling back to ENERGY_REPLENISH_PER_HEARTBEAT.",
                raw_deprecated,
            )
    return resolve_energy_replenish_per_heartbeat(log=log)


def warn_deprecated_energy_replenish_turn_env_once(
    owner_cls: type,
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    if getattr(owner_cls, "_deprecated_energy_replenish_turn_warning_logged", False):
        return
    raw_deprecated = str(os.getenv("ENERGY_REPLENISH_PER_TURN", "") or "").strip()
    if not raw_deprecated:
        return
    active_logger = log or logger
    active_logger.warning(
        "ENERGY_REPLENISH_PER_TURN is deprecated and ignored. "
        "Use ENERGY_REPLENISH_PER_HOUR for wall-clock replenishment instead."
    )
    setattr(owner_cls, "_deprecated_energy_replenish_turn_warning_logged", True)


def compute_predictive_energy_replenishment_points_wallclock_locked(
    owner: Any,
    *,
    now: Optional[float] = None,
    resolve_per_hour: Optional[Callable[[], float]] = None,
    time_fn: Optional[Callable[[], float]] = None,
) -> int:
    clock = time_fn or time.time
    current_time = float(clock() if now is None else now)
    last_replenished_at = float(
        getattr(owner, "_predictive_energy_budget_last_replenished_at", current_time)
    )
    if current_time <= last_replenished_at:
        return 0

    replenish_per_hour = resolve_per_hour() if callable(resolve_per_hour) else resolve_energy_replenish_per_hour()
    if replenish_per_hour <= 0:
        owner._predictive_energy_budget_last_replenished_at = current_time
        return 0

    elapsed_seconds = current_time - last_replenished_at
    points = int((elapsed_seconds / 3600.0) * replenish_per_hour)
    if points <= 0:
        return 0

    consumed_seconds = (float(points) / replenish_per_hour) * 3600.0
    owner._predictive_energy_budget_last_replenished_at = last_replenished_at + consumed_seconds
    return points


def apply_predictive_energy_tick_locked(owner: Any, amount: int) -> int:
    if amount <= 0:
        return 0
    cap = max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
    before = max(0, int(getattr(owner, "_predictive_energy_budget_remaining", 0)))
    owner._predictive_energy_budget_remaining = min(cap, before + int(amount))
    if hasattr(owner, "_energy_budget"):
        owner._energy_budget = owner._predictive_energy_budget_remaining
    return int(owner._predictive_energy_budget_remaining) - before


def replenish_predictive_energy_budget_wallclock_locked(
    owner: Any,
    *,
    now: Optional[float] = None,
    warn_deprecated: Optional[Callable[[], None]] = None,
    compute_points: Optional[Callable[..., int]] = None,
    apply_tick: Optional[Callable[[int], int]] = None,
) -> int:
    if callable(warn_deprecated):
        warn_deprecated()
    points = (
        compute_points(now=now)
        if callable(compute_points)
        else compute_predictive_energy_replenishment_points_wallclock_locked(owner, now=now)
    )
    if callable(apply_tick):
        return apply_tick(points)
    return apply_predictive_energy_tick_locked(owner, points)


async def refund_predictive_energy_budget(
    owner: Any,
    amount: int,
    reason: str,
    *,
    gate_available: Optional[Callable[[], bool]] = None,
    log: Optional[logging.Logger] = None,
) -> None:
    amount = max(0, int(amount))
    if amount <= 0:
        return
    available = gate_available() if callable(gate_available) else energy_gate_available(owner)
    if not available:
        return

    cap = max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
    async with owner._predictive_energy_budget_lock:
        owner._predictive_energy_budget_remaining = min(
            cap, owner._predictive_energy_budget_remaining + amount
        )
        if hasattr(owner, "_energy_budget"):
            owner._energy_budget = owner._predictive_energy_budget_remaining

    active_logger = log or logger
    active_logger.info(
        "Energy budget refunded: +%s (%s). Remaining=%s",
        amount,
        reason,
        owner._predictive_energy_budget_remaining,
    )


async def tick_predictive_energy_budget(
    amount: int,
    reason: str,
    *,
    get_remaining: Callable[[], Any],
    refund: Callable[[int, str], Any],
) -> int:
    if amount <= 0:
        return 0
    before = await get_remaining()
    await refund(int(amount), reason)
    after = await get_remaining()
    return max(0, after - before)


async def try_reserve_predictive_energy_budget(
    owner: Any,
    *,
    predicted_cost: int,
    min_reserve: int,
    reason: str,
    gate_available: Optional[Callable[[], bool]] = None,
    log: Optional[logging.Logger] = None,
) -> bool:
    available = gate_available() if callable(gate_available) else energy_gate_available(owner)
    if not available:
        return True

    cost = max(0, int(predicted_cost))
    reserve = max(0, int(min_reserve))
    async with owner._predictive_energy_budget_lock:
        remaining = max(0, int(owner._predictive_energy_budget_remaining))
        if remaining - cost < reserve:
            return False
        owner._predictive_energy_budget_remaining = remaining - cost
        if hasattr(owner, "_energy_budget"):
            owner._energy_budget = owner._predictive_energy_budget_remaining
        active_logger = log or logger
        active_logger.info(
            "Predictive energy budget reserved: -%s (%s). Remaining=%s",
            cost,
            reason,
            owner._predictive_energy_budget_remaining,
        )
        return True


def build_energy_evaluation_record(
    *,
    evaluation: Any,
    decision: Any,
    available_energy: int,
    context: str,
) -> Dict[str, Any]:
    return {
        "context": str(context or ""),
        "available_energy": int(available_energy),
        "estimated_effort": int(evaluation.estimated_effort),
        "expected_value": int(evaluation.expected_value),
        "defer_count": int(decision.defer_count),
        "used_fallback": bool(evaluation.used_fallback),
        "fallback_reason": str(evaluation.fallback_reason or ""),
        "should_execute": bool(decision.should_execute),
        "reason": str(decision.reason or ""),
        "roi": float(decision.roi),
        "base_roi": float(decision.base_roi),
        "effective_roi": float(decision.effective_roi),
        "predicted_cost": int(decision.predicted_cost),
        "reserve_after_execution": int(decision.reserve_after_execution),
        "roi_threshold": float(decision.roi_threshold),
        "min_reserve": int(decision.min_reserve),
        "fairness_boost_multiplier": float(decision.fairness_boost_multiplier),
        "max_defer_count": int(decision.max_defer_count),
        "defer_cooldown_seconds": int(decision.defer_cooldown_seconds),
        "evaluated_at": datetime.now().isoformat(),
    }


async def route_energy_judge_messages(owner: Any, messages: List[Dict[str, str]]) -> Any:
    return await owner._route_to_system_1(
        messages,
        allowed_tools=[],
        deadline_seconds=45.0,
        context="energy_judge",
    )


async def evaluate_energy_for_context(
    owner: Any,
    *,
    task: Dict[str, Any],
    story: Optional[Dict[str, Any]],
    epic: Optional[Dict[str, Any]],
    additional_context: str,
) -> tuple[Any, Any, int]:
    available_energy = await owner._get_predictive_energy_budget_remaining()
    defer_count = int(task.get("defer_count") or 0)
    evaluation = await owner.energy_judge.evaluate_with_system1(
        task=task,
        story=story,
        epic=epic,
        route_to_system_1=owner._route_energy_judge_messages,
        additional_context=additional_context,
    )
    decision = owner.energy_roi_engine.evaluate(
        estimated_effort=evaluation.estimated_effort,
        expected_value=evaluation.expected_value,
        available_energy=available_energy,
        defer_count=defer_count,
    )
    return evaluation, decision, available_energy


def build_synthesized_ad_hoc_energy_context(owner: Any, state: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    user_message = str(state.get("user_input") or "").strip()
    complexity = max(1, owner._estimate_request_complexity(user_message) + 1)
    history_context = owner._build_goal_planning_context(state)

    task = {
        "title": user_message,
        "acceptance_criteria": "Provide a correct and concise answer to the current user request.",
        "estimated_energy": min(10, complexity),
        "status": "pending",
        "depends_on_ids": [],
        "defer_count": 0,
    }
    story = {
        "title": "Ad-hoc user response",
        "status": "active",
        "acceptance_criteria": "Answer the user request with useful and policy-compliant output.",
    }
    epic = {
        "title": "Sustain reliable operator assistance",
        "status": "active",
        "acceptance_criteria": "Serve user requests while preserving energy reserves for critical work.",
    }
    return {
        "task": task,
        "story": story,
        "epic": epic,
        "additional_context": history_context,
    }


def format_ad_hoc_energy_deferral_message(*, decision: Any, available_energy: int) -> str:
    return (
        "Deferred due to energy policy. "
        f"Reason: {decision.reason}. "
        f"ROI={decision.roi:.2f} (base={decision.base_roi:.2f}, defer_count={decision.defer_count}), "
        f"predicted_cost={decision.predicted_cost}, "
        f"available_budget={int(available_energy)}, reserve_floor={decision.min_reserve}."
    )


async def try_ad_hoc_dispatch_energy_gate(
    owner: Any,
    state: Dict[str, Any],
    *,
    dispatch_context: str,
    log: Optional[logging.Logger] = None,
) -> Optional[str]:
    if not owner._energy_gate_available():
        return None

    user_id = str(state.get("user_id") or "")
    if user_id == "heartbeat":
        return None

    user_message = str(state.get("user_input") or "").strip()
    if not user_message:
        return None

    context = owner._build_synthesized_ad_hoc_energy_context(state)
    evaluation, decision, available_energy = await owner._evaluate_energy_for_context(
        task=context["task"] or {},
        story=context["story"],
        epic=context["epic"],
        additional_context=str(context.get("additional_context") or ""),
    )

    active_logger = log or logger
    if not decision.should_execute:
        message = owner._format_ad_hoc_energy_deferral_message(
            decision=decision,
            available_energy=available_energy,
        )
        active_logger.warning(
            "Ad-hoc dispatch deferred by energy gate (%s): %s",
            dispatch_context,
            message,
        )
        return message

    reserved = await owner._try_reserve_predictive_energy_budget(
        predicted_cost=decision.predicted_cost,
        min_reserve=decision.min_reserve,
        reason=f"ad_hoc:{dispatch_context}:{user_id}",
    )
    if not reserved:
        current_budget = await owner._get_predictive_energy_budget_remaining()
        race_decision = owner.energy_roi_engine.evaluate(
            estimated_effort=evaluation.estimated_effort,
            expected_value=evaluation.expected_value,
            available_energy=current_budget,
            defer_count=decision.defer_count,
        )
        message = owner._format_ad_hoc_energy_deferral_message(
            decision=race_decision,
            available_energy=current_budget,
        )
        active_logger.warning(
            "Ad-hoc dispatch deferred after budget reservation race (%s): %s",
            dispatch_context,
            message,
        )
        return message

    active_logger.info(
        "Ad-hoc dispatch approved by energy gate (%s): roi=%.2f cost=%s remaining=%s",
        dispatch_context,
        decision.roi,
        decision.predicted_cost,
        await owner._get_predictive_energy_budget_remaining(),
    )
    return None


async def defer_heartbeat_task_due_to_energy(
    owner: Any,
    *,
    task_id: int,
    decision: Any,
    evaluation_record: Dict[str, Any],
    available_energy: int,
    race: bool = False,
    log: Optional[logging.Logger] = None,
) -> None:
    await owner.ledger_memory.defer_task_due_to_energy(
        task_id,
        evaluation_record,
        cooldown_seconds=int(decision.defer_cooldown_seconds),
    )
    active_logger = log or logger
    active_logger.info(
        "Heartbeat: Deferred task #%s%s (%s). roi=%.2f cost=%s available=%s reserve_floor=%s",
        task_id,
        " after budget race" if race else "",
        decision.reason,
        decision.roi,
        decision.predicted_cost,
        available_energy,
        decision.min_reserve,
    )


def deduct_energy(
    state: Dict[str, Any],
    amount: int,
    reason: str,
    hitl_error_type: type[Exception],
    *,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    state["energy_remaining"] = state.get("energy_remaining", 100) - amount
    active_logger = log or logger
    active_logger.debug("Energy -%d (%s). Remaining: %d", amount, reason, state["energy_remaining"])
    if state["energy_remaining"] <= 0:
        raise hitl_error_type(
            "Energy Budget Exhausted: The system consumed all cognitive energy on this task.\n"
            f"Question: How should I prioritize the remaining work for: '{state.get('user_input', '')}'"
        )
    return state


def refund_energy(
    state: Dict[str, Any],
    amount: int,
    reason: str,
    *,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    state["energy_remaining"] = state.get("energy_remaining", 0) + amount
    active_logger = log or logger
    active_logger.debug("Energy +%d refunded (%s). Remaining: %d", amount, reason, state["energy_remaining"])
    return state
