"""Supervisor graph node helpers for the Orchestrator facade."""
from __future__ import annotations

import asyncio
import logging
import platform
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SupervisorRouteDeps:
    route_to_system_1: Callable[..., Awaitable[Any]]
    route_to_system_2_redacted: Callable[..., Awaitable[Any]]
    is_system_1_error: Callable[[Any], bool]
    cognitive_router: Any
    log: logging.Logger


@dataclass
class SupervisorNodeDeps:
    normalize_state: Callable[[Dict[str, Any]], Dict[str, Any]]
    try_ad_hoc_dispatch_energy_gate: Callable[..., Awaitable[Optional[str]]]
    deduct_energy: Callable[[Dict[str, Any], int, str], Dict[str, Any]]
    refund_energy: Callable[[Dict[str, Any], int, str], Dict[str, Any]]
    core_memory: Any
    get_archival_context: Callable[[str], Awaitable[str]]
    get_capabilities_string: Callable[[], str]
    get_agent_descriptions: Callable[[], str]
    get_sensory_context: Callable[[], str]
    build_user_prompt_message: Callable[[Dict[str, Any]], Dict[str, Any]]
    route_supervisor_request: Callable[[List[Dict[str, Any]]], Awaitable[Optional[Any]]]
    handle_cognitive_escalation: Callable[[Dict[str, Any], Any], Awaitable[str]]
    parse_supervisor_response: Callable[[str, Dict[str, Any]], Dict[str, Any]]
    build_supervisor_prompt: Callable[..., str]
    charter_text: str
    prompt_config: Any
    sensory_state: Dict[str, Any]
    system_1_error_prefix: str
    blocked_key: str
    energy_cost_supervisor: int
    cognitive_router: Any
    hitl_error_type: type[Exception]
    log: logging.Logger


async def route_supervisor_request(
    messages: List[Dict[str, Any]],
    *,
    deps: SupervisorRouteDeps,
) -> Optional[Any]:
    router_result: Optional[Any] = None
    try:
        deps.log.info("Routing Supervisor through System 1 (Local Model)")
        router_result = await deps.route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=150.0,
            context="supervisor",
        )
        if deps.is_system_1_error(router_result):
            deps.log.warning("System 1 returned an error payload in supervisor; escalating to System 2.")
            router_result = None
    except asyncio.TimeoutError:
        deps.log.error("System 1 timed out in supervisor (150 s). Escalating to System 2.", exc_info=True)
    except Exception as exc:
        deps.log.error("System 1 raised an exception in supervisor: %r. Escalating.", exc, exc_info=True)

    if router_result is not None:
        return router_result

    if not deps.cognitive_router.get_system_2_available():
        return None

    try:
        deps.log.info("Escalating Supervisor to System 2")
        router_result = await asyncio.wait_for(
            deps.route_to_system_2_redacted(
                messages,
                allowed_tools=[],
                purpose="supervisor_fallback",
                allow_sensitive_context=False,
            ),
            timeout=60.0,
        )
        return router_result
    except asyncio.TimeoutError:
        deps.log.error("System 2 timed out in supervisor (60 s).", exc_info=True)
    except Exception as exc:
        deps.log.error("System 2 raised an exception in supervisor: %r.", exc, exc_info=True)
    return None


async def _apply_supervisor_energy_gate(state: Dict[str, Any], deps: SupervisorNodeDeps) -> bool:
    if not state.get("_energy_gate_cleared"):
        deferred = await deps.try_ad_hoc_dispatch_energy_gate(
            state,
            dispatch_context="graph_dispatch",
        )
        if deferred is not None:
            state["current_plan"] = []
            state["final_response"] = deferred
            return True
        state["_energy_gate_cleared"] = True
    return False


async def _build_supervisor_messages(state: Dict[str, Any], deps: SupervisorNodeDeps) -> List[Dict[str, Any]]:
    user_input = state["user_input"]
    core_mem_str = await deps.core_memory.get_context_string()
    archival_context = await deps.get_archival_context(user_input)
    capabilities_str = deps.get_capabilities_string()
    archival_block = f"{archival_context}\n\n" if archival_context else ""

    system_prompt = deps.build_supervisor_prompt(
        charter_text=deps.charter_text,
        core_mem_str=core_mem_str,
        archival_block=archival_block,
        capabilities_str=capabilities_str,
        agent_descriptions=deps.get_agent_descriptions(),
        sensory_context=deps.get_sensory_context(),
        os_name=deps.sensory_state.get("os", platform.system()),
        downloads_dir=deps.prompt_config.downloads_dir,
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in state.get("chat_history", []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    if state.get("critic_instructions"):
        messages.append({"role": "system", "content": state["critic_instructions"]})
    messages.append(deps.build_user_prompt_message(state))
    return messages


async def _apply_supervisor_router_result(
    state: Dict[str, Any],
    router_result: Optional[Any],
    deps: SupervisorNodeDeps,
) -> Dict[str, Any]:
    if router_result is not None and router_result.status == "cognitive_escalation":
        solution = await deps.handle_cognitive_escalation(state, router_result)
        state["worker_outputs"]["system_2_escalation"] = solution
        state["current_plan"] = [{
            "agent": "system_2_escalation",
            "task": "",
            "reason": "cognitive escalation",
            "depends_on": [],
            "preferred_model": "system_2",
        }]
        return state

    if router_result is not None and router_result.status != "ok":
        state[deps.blocked_key] = router_result
        return state

    response = router_result.content if router_result else None
    if not response or response.startswith(deps.system_1_error_prefix) or response.startswith("[System 2"):
        deps.log.warning("Supervisor received error or no response: %r", response)
        state["current_plan"] = []
        if not deps.cognitive_router.get_system_2_available():
            state["final_response"] = (
                "I was unable to process this request locally. This task likely requires "
                "capabilities beyond my local model (e.g. internet access or complex reasoning). "
                "To enable this, configure a System 2 provider by setting GROQ_API_KEY in your .env file. "
                "Groq offers a free tier at console.groq.com."
            )
        else:
            state["final_response"] = "Both local and cloud reasoning failed on this request. Please try rephrasing or simplifying the task."
        return state

    return deps.parse_supervisor_response(response, state)


async def supervisor_node(state: Dict[str, Any], *, deps: SupervisorNodeDeps) -> Dict[str, Any]:
    state = deps.normalize_state(state)

    if await _apply_supervisor_energy_gate(state, deps):
        return state

    state = deps.deduct_energy(state, deps.energy_cost_supervisor, "supervisor")
    messages = await _build_supervisor_messages(state, deps)

    try:
        router_result = await deps.route_supervisor_request(messages)
        state = await _apply_supervisor_router_result(state, router_result, deps)
    except asyncio.TimeoutError:
        deps.log.error("Supervisor node timeout (60s)")
        state = deps.refund_energy(state, deps.energy_cost_supervisor, "supervisor_timeout")
        state["current_plan"] = []
        state["final_response"] = "Planning timed out. Please try again."
    except deps.hitl_error_type:
        raise
    except Exception as exc:
        deps.log.error("Supervisor Node failed: %s", exc, exc_info=True)
        state = deps.refund_energy(state, deps.energy_cost_supervisor, "supervisor_error")
        state["current_plan"] = []
        state["final_response"] = "Supervisor encountered an error. Please try again."

    return state
