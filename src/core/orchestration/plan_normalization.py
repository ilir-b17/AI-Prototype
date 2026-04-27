"""Stateless plan normalization helpers for supervisor worker plans."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def merge_plan_text(existing: str, incoming: str) -> str:
    existing = str(existing or "").strip()
    incoming = str(incoming or "").strip()
    if not existing:
        return incoming
    if not incoming or incoming in existing:
        return existing
    return f"{existing}\n{incoming}"


def make_plan_step(
    agent_name: str,
    *,
    task: str = "",
    reason: str = "",
    depends_on: Optional[List[str]] = None,
    preferred_model: str = "",
) -> Dict[str, Any]:
    return {
        "agent": agent_name,
        "task": task,
        "reason": reason,
        "depends_on": list(depends_on or []),
        "preferred_model": preferred_model,
    }


def normalize_plan_dependencies(raw_dependencies: Any) -> List[str]:
    if raw_dependencies is None:
        return []

    items = raw_dependencies if isinstance(raw_dependencies, (list, tuple, set)) else [raw_dependencies]
    normalized: List[str] = []
    for item in items:
        dependency = str(item).strip()
        if dependency and dependency not in normalized:
            normalized.append(dependency)
    return normalized


def merge_plan_dependencies(existing: List[str], incoming: List[str]) -> List[str]:
    merged = list(existing or [])
    for dependency in incoming or []:
        if dependency not in merged:
            merged.append(dependency)
    return merged


def normalize_model_preference(raw_value: Any) -> str:
    preferred_model = str(raw_value or "").strip().lower()
    return preferred_model if preferred_model in {"system_1", "system_2"} else ""


def normalize_plan_step(step: Any) -> Optional[Dict[str, Any]]:
    if isinstance(step, str):
        agent_name = step.strip()
        if not agent_name:
            return None
        return make_plan_step(agent_name)

    if not isinstance(step, dict):
        return None

    agent_name = str(step.get("agent") or step.get("name") or "").strip()
    if not agent_name:
        return None

    task = str(
        step.get("task")
        or step.get("instructions")
        or step.get("objective")
        or ""
    ).strip()
    reason = str(step.get("reason") or step.get("why") or "").strip()
    depends_on = normalize_plan_dependencies(
        step.get("depends_on") or step.get("requires") or step.get("inputs")
    )
    preferred_model = normalize_model_preference(
        step.get("preferred_model") or step.get("model")
    )
    return make_plan_step(
        agent_name,
        task=task,
        reason=reason,
        depends_on=depends_on,
        preferred_model=preferred_model,
    )


def normalize_current_plan(plan: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    by_agent: Dict[str, Dict[str, Any]] = {}

    for raw_step in plan or []:
        step = normalize_plan_step(raw_step)
        if step is None:
            continue

        existing = by_agent.get(step["agent"])
        if existing is None:
            normalized.append(step)
            by_agent[step["agent"]] = step
            continue

        existing["task"] = merge_plan_text(existing["task"], step["task"])
        existing["reason"] = merge_plan_text(existing["reason"], step["reason"])
        existing["depends_on"] = merge_plan_dependencies(
            existing.get("depends_on", []),
            step.get("depends_on", []),
        )
        existing["preferred_model"] = existing.get("preferred_model") or step.get("preferred_model", "")

    return normalized
