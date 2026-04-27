"""Agent execution helpers for the Orchestrator facade."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentExecutorDeps:
    normalize_state: Callable[[Dict[str, Any]], Dict[str, Any]]
    deduct_energy: Callable[[Dict[str, Any], int, str], Dict[str, Any]]
    refund_energy: Callable[[Dict[str, Any], int, str], Dict[str, Any]]
    refund_predictive_energy_budget: Callable[[int, str], Awaitable[None]]
    build_agent_state_snapshot: Callable[[Dict[str, Any]], Dict[str, Any]]
    run_agent: Callable[..., Awaitable[Dict[str, Any]]]
    build_execution_batches: Callable[[List[Dict[str, Any]]], tuple]
    get_requested_plan_steps: Callable[[Dict[str, Any]], List[Dict[str, Any]]]
    build_agent_handoff: Callable[..., str]
    route_agent_request: Callable[..., Awaitable[tuple[Optional[Any], bool]]]
    route_to_system_1: Callable[..., Awaitable[Any]]
    route_to_system_2_redacted: Callable[..., Awaitable[Any]]
    is_system_1_error: Callable[[Any], bool]
    is_system_2_error: Callable[[Any], bool]
    handle_cognitive_escalation: Callable[[Dict[str, Any], Any], Awaitable[str]]
    get_capabilities_string: Callable[[], str]
    get_step_preferred_model: Callable[[Optional[Dict[str, Any]], Any], str]
    core_memory: Any
    cognitive_router: Any
    charter_text: str
    blocked_key: str
    hitl_error_type: type[Exception]
    log: logging.Logger


def build_agent_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = dict(state)
    snapshot["chat_history"] = list(state.get("chat_history", []) or [])
    snapshot["current_plan"] = list(state.get("current_plan", []) or [])
    snapshot["worker_outputs"] = dict(state.get("worker_outputs", {}) or {})
    return snapshot


def _find_blocked_parallel_result(
    batch: List[tuple[Any, Dict[str, Any]]],
    results: List[Any],
    *,
    deps: AgentExecutorDeps,
) -> Optional[Any]:
    blocked_result = None
    for (agent_def, _task_packet), result in zip(batch, results):
        if isinstance(result, deps.hitl_error_type):
            deps.log.warning("Parallel agent %s requested HITL: %s", agent_def.name, result)
            raise result
        if not isinstance(result, Exception) and result.get(deps.blocked_key) and blocked_result is None:
            blocked_result = result[deps.blocked_key]
    return blocked_result


async def _refund_parallel_siblings_after_block(
    batch: List[tuple[Any, Dict[str, Any]]],
    results: List[Any],
    state: Dict[str, Any],
    *,
    deps: AgentExecutorDeps,
) -> Dict[str, Any]:
    refunded_cost = sum(
        int(agent_def.energy_cost)
        for (agent_def, _task_packet), result in zip(batch, results)
        if not isinstance(result, Exception) and not result.get(deps.blocked_key)
    )
    if refunded_cost <= 0:
        return state
    state = deps.refund_energy(
        state,
        refunded_cost,
        "parallel_batch_blocked_unused_siblings",
    )
    await deps.refund_predictive_energy_budget(
        refunded_cost,
        "parallel_batch_blocked_unused_siblings",
    )
    return state


def _merge_parallel_agent_results(
    batch: List[tuple[Any, Dict[str, Any]]],
    results: List[Any],
    state: Dict[str, Any],
    *,
    deps: AgentExecutorDeps,
) -> Dict[str, Any]:
    for (agent_def, _task_packet), result in zip(batch, results):
        if isinstance(result, Exception):
            deps.log.error("Parallel agent %s failed with exception: %s", agent_def.name, result, exc_info=True)
            state["worker_outputs"][agent_def.name] = f"Error: {agent_def.name} failed with exception: {result}"
            continue

        agent_output = dict(result.get("worker_outputs", {}) or {}).get(agent_def.name)
        if agent_output is not None:
            state["worker_outputs"][agent_def.name] = agent_output
    return state


async def run_parallel_agent_batch(
    batch: List[tuple[Any, Dict[str, Any]]],
    state: Dict[str, Any],
    *,
    deps: AgentExecutorDeps,
) -> Dict[str, Any]:
    batch_names = [agent_def.name for agent_def, _ in batch]
    state = deps.deduct_energy(state, sum(agent_def.energy_cost for agent_def, _ in batch), f"parallel_agents:{batch_names}")
    base_state = deps.build_agent_state_snapshot(state)
    results = await asyncio.gather(
        *[
            deps.run_agent(
                agent_def,
                deps.build_agent_state_snapshot(base_state),
                task_packet=task_packet,
                deduct_energy=False,
            )
            for agent_def, task_packet in batch
        ],
        return_exceptions=True,
    )
    result_list = list(results)

    blocked_result = _find_blocked_parallel_result(batch, result_list, deps=deps)

    if blocked_result is not None:
        state = await _refund_parallel_siblings_after_block(batch, result_list, state, deps=deps)
        state[deps.blocked_key] = blocked_result
        return state

    return _merge_parallel_agent_results(batch, result_list, state, deps=deps)


async def try_route_agent_system_1(
    messages: List[Dict[str, str]],
    agent_def: Any,
    *,
    deps: AgentExecutorDeps,
) -> Optional[Any]:
    try:
        router_result = await deps.route_to_system_1(
            messages,
            allowed_tools=agent_def.allowed_tools,
            deadline_seconds=60.0,
            context=agent_def.name,
        )
        if deps.is_system_1_error(router_result):
            return None
        return router_result
    except Exception as exc:
        deps.log.error("%s failed on System 1: %s", agent_def.name, exc)
        return None


async def try_route_agent_system_2(
    messages: List[Dict[str, str]],
    agent_def: Any,
    *,
    deps: AgentExecutorDeps,
) -> Optional[Any]:
    if not deps.cognitive_router.get_system_2_available():
        return None

    try:
        router_result = await asyncio.wait_for(
            deps.route_to_system_2_redacted(
                messages,
                allowed_tools=agent_def.allowed_tools,
                purpose=f"{agent_def.name}_fallback",
                allow_sensitive_context=False,
            ),
            timeout=60.0,
        )
        if deps.is_system_2_error(router_result):
            return None
        return router_result
    except Exception as exc:
        deps.log.error("%s failed on System 2: %s", agent_def.name, exc)
        return None


async def route_agent_request(
    messages: List[Dict[str, str]],
    agent_def: Any,
    *,
    task_packet: Optional[Dict[str, Any]],
    deps: AgentExecutorDeps,
) -> tuple[Optional[Any], bool]:
    preferred_model = deps.get_step_preferred_model(task_packet, agent_def)

    if preferred_model == "system_2":
        attempted_system_2 = deps.cognitive_router.get_system_2_available()
        deps.log.info("Routing %s through System 2 (agent preference)", agent_def.name)
        router_result = await try_route_agent_system_2(messages, agent_def, deps=deps)
        if router_result is not None:
            return router_result, attempted_system_2

        deps.log.info(
            "Falling back %s to System 1 after preferred System 2 path was unavailable or failed",
            agent_def.name,
        )
        return await try_route_agent_system_1(messages, agent_def, deps=deps), attempted_system_2

    deps.log.info("Routing %s through System 1", agent_def.name)
    router_result = await try_route_agent_system_1(messages, agent_def, deps=deps)
    if router_result is not None:
        return router_result, False

    attempted_system_2 = deps.cognitive_router.get_system_2_available()
    if attempted_system_2:
        deps.log.info("Escalating %s to System 2 after System 1 failure", agent_def.name)
    return await try_route_agent_system_2(messages, agent_def, deps=deps), attempted_system_2


async def run_agent(
    agent_def: Any,
    state: Dict[str, Any],
    *,
    task_packet: Optional[Dict[str, Any]],
    deduct_energy: bool,
    deps: AgentExecutorDeps,
) -> Dict[str, Any]:
    state = deps.normalize_state(state)
    if deduct_energy:
        state = deps.deduct_energy(state, agent_def.energy_cost, agent_def.name)
    try:
        core_mem_str = await deps.core_memory.get_context_string()

        capabilities_str = deps.get_capabilities_string()
        handoff = deps.build_agent_handoff(agent_def, state, task_packet=task_packet)
        messages = [
            {
                "role": "system",
                "content": f"{agent_def.system_prompt}\n{deps.charter_text}\n{core_mem_str}\n\n{capabilities_str}",
            },
            {"role": "user", "content": handoff},
        ]
        router_result, attempted_system_2 = await deps.route_agent_request(
            messages,
            agent_def,
            task_packet=task_packet,
            state=state,
        )
    except Exception as agent_exc:
        if deduct_energy:
            state = deps.refund_energy(state, agent_def.energy_cost, f"{agent_def.name}_exception")
        raise agent_exc

    if router_result is None:
        if deduct_energy:
            state = deps.refund_energy(state, agent_def.energy_cost, f"{agent_def.name}_null_result")
        if attempted_system_2:
            state["worker_outputs"][agent_def.name] = (
                f"Error: {agent_def.name} failed after System 1 error and System 2 fallback."
            )
        else:
            state["worker_outputs"][agent_def.name] = (
                f"Error: {agent_def.name} failed and System 2 is not configured."
            )
        return state

    if router_result.status == "cognitive_escalation":
        solution = await deps.handle_cognitive_escalation(state, router_result)
        state["worker_outputs"][agent_def.name] = solution
        return state

    if router_result.status != "ok":
        state[deps.blocked_key] = router_result
    else:
        state["worker_outputs"][agent_def.name] = router_result.content

    return state


async def execute_workers_node(state: Dict[str, Any], *, deps: AgentExecutorDeps) -> Dict[str, Any]:
    state = deps.normalize_state(state)
    if not state.get("current_plan"):
        if not state.get("final_response"):
            state["final_response"] = "No response could be generated. Please try again."
        return state

    state["current_plan"] = deps.get_requested_plan_steps(state)
    executed_agent = False
    execution_batches, skipped_cycles = deps.build_execution_batches(state["current_plan"])
    for skipped in skipped_cycles:
        state["worker_outputs"][skipped] = (
            f"Warning: agent '{skipped}' was skipped due to a cyclic dependency in the plan."
        )
    for batch in execution_batches:
        if not batch:
            continue

        executed_agent = True
        if len(batch) == 1:
            agent_def, task_packet = batch[0]
            state = await deps.run_agent(agent_def, state, task_packet=task_packet)
        else:
            state = await run_parallel_agent_batch(batch, state, deps=deps)

        if state.get(deps.blocked_key):
            return state

    if not executed_agent and not state.get("final_response"):
        state["final_response"] = "No registered agent could execute the current plan."
    return state
