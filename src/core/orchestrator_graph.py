"""LangGraph node functions and agent execution mixin for Orchestrator."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from src.core.orchestrator_constants import (
    ENERGY_COST_SUPERVISOR,
    ENERGY_COST_WORKER,
    ENERGY_COST_CRITIC,
    ENERGY_COST_TOOL,
    _BLOCKED_KEY,
    _SYNTHESIS_LOCKOUT_TTL_SECONDS,
    _SYSTEM_1_ERROR_PREFIX,
    _CRITIC_SHORT_OUTPUT_THRESHOLD,
    _MORAL_TRIVIALITY_READ_ONLY_HINTS,
    _MORAL_TRIVIALITY_BLOCK_HINTS,
)
from src.core.agent_definition import AgentDefinition
from src.core.agent_registry import AgentRegistry
from src.core.llm_router import (
    RouterResult, RequiresHITLError, RequiresMFAError,
)
from src.core.state_model import normalize_state
from src.core.prompt_config import (
    build_supervisor_prompt, build_supervisor_turn_context,
)
from src.core.routing_assessor import RoutingAssessor, _FAST_PATH_SINGLE_TOOL_ALLOWLIST
from src.core.agent_output import (
    AgentOutputType,
    extract_structured_output,
    format_structured_for_handoff,
    get_display_text,
    make_extraction_failed_result,
    resolve_output_type,
    strip_agent_output_block,
    validate_agent_output,
    OUTPUT_FORMAT_PROMPTS,
)
from src.core.moral_ledger import (
    MORAL_DIMENSIONS,
    MORAL_RUBRIC_VERSION,
    MORAL_TIER_1_DIMENSIONS,
    MoralDecision,
    build_legacy_binary_decision,
    build_local_skip_decision,
    build_triviality_bypass_decision,
    parse_moral_decision_response,
)
from src.core.progress import (
    ProgressEmitter,
    ProgressEvent,
    ProgressCallback,
    get_current_emitter,
    reset_emitter,
    set_current_emitter,
)
from src.core import cloud_redaction

logger = logging.getLogger(__name__)


class _GraphNodesMixin:
    """Agent execution, plan parsing, and LangGraph nodes. Mixed into Orchestrator."""

    @staticmethod
    def _normalize_plan_step(step: Any) -> Optional[Dict[str, Any]]:
        if isinstance(step, str):
            agent_name = step.strip()
            if not agent_name:
                return None
            return Orchestrator._make_plan_step(agent_name)

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
        depends_on = Orchestrator._normalize_plan_dependencies(
            step.get("depends_on") or step.get("requires") or step.get("inputs")
        )
        preferred_model = Orchestrator._normalize_model_preference(
            step.get("preferred_model") or step.get("model")
        )
        return Orchestrator._make_plan_step(
            agent_name,
            task=task,
            reason=reason,
            depends_on=depends_on,
            preferred_model=preferred_model,
        )

    def _normalize_current_plan(self, plan: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        by_agent: Dict[str, Dict[str, Any]] = {}

        for raw_step in plan or []:
            step = self._normalize_plan_step(raw_step)
            if step is None:
                continue

            existing = by_agent.get(step["agent"])
            if existing is None:
                normalized.append(step)
                by_agent[step["agent"]] = step
                continue

            existing["task"] = self._merge_plan_text(existing["task"], step["task"])
            existing["reason"] = self._merge_plan_text(existing["reason"], step["reason"])
            existing["depends_on"] = self._merge_plan_dependencies(
                existing.get("depends_on", []),
                step.get("depends_on", []),
            )
            existing["preferred_model"] = existing.get("preferred_model") or step.get("preferred_model", "")

        return normalized

    @staticmethod
    def _normalize_plan_dependencies(raw_dependencies: Any) -> List[str]:
        if raw_dependencies is None:
            return []

        items = raw_dependencies if isinstance(raw_dependencies, (list, tuple, set)) else [raw_dependencies]
        normalized: List[str] = []
        for item in items:
            dependency = str(item).strip()
            if dependency and dependency not in normalized:
                normalized.append(dependency)
        return normalized

    @staticmethod

    def _normalize_model_preference(raw_value: Any) -> str:
        preferred_model = str(raw_value or "").strip().lower()
        return preferred_model if preferred_model in {"system_1", "system_2"} else ""

    @staticmethod

    def _merge_plan_text(existing: str, incoming: str) -> str:
        existing = str(existing or "").strip()
        incoming = str(incoming or "").strip()
        if not existing:
            return incoming
        if not incoming or incoming in existing:
            return existing
        return f"{existing}\n{incoming}"

    @staticmethod

    def _merge_plan_dependencies(existing: List[str], incoming: List[str]) -> List[str]:
        merged = list(existing or [])
        for dependency in incoming or []:
            if dependency not in merged:
                merged.append(dependency)
        return merged

    @staticmethod

    def _make_plan_step(
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

    @staticmethod

    def _is_structured_plan_packet(step: Any) -> bool:
        if not isinstance(step, dict):
            return False

        agent_name = str(step.get("agent") or step.get("name") or "").strip()
        if not agent_name:
            return False

        task = str(step.get("task") or step.get("instructions") or step.get("objective") or "").strip()
        reason = str(step.get("reason") or step.get("why") or "").strip()
        return bool(task or reason)

    def _is_structured_plan_payload(self, payload: Any) -> bool:
        if not isinstance(payload, list):
            return False
        return all(self._is_structured_plan_packet(step) for step in payload)

    @staticmethod
    def _decode_workers_payload(workers_payload: str) -> Optional[Any]:
        if not workers_payload:
            return None
        try:
            decoded, end_index = json.JSONDecoder().raw_decode(workers_payload)
        except ValueError:
            return None

        # Strip trailing markdown fences and punctuation the model may append.
        remainder = workers_payload[end_index:]
        # Allow trailing whitespace, backticks, and sentence-ending punctuation.
        # Trailing prose beyond this is common for smaller local models (e.g. gemma4:e4b
        # frequently appends an explanation sentence after the JSON array).  Since
        # raw_decode() already validated and extracted a complete JSON value we can safely
        # ignore whatever comes after it — discarding the payload here silently drops the
        # entire agent dispatch plan, which is far worse than accepting a noisy suffix.
        trailing = remainder.strip()
        if trailing and not re.fullmatch(r"[\s`.,;!?]*", remainder):
            logger.warning(
                "_decode_workers_payload: model appended trailing text after JSON "
                "(%r … first 60 chars of payload: %r); ignoring suffix and proceeding.",
                trailing[:60],
                workers_payload[:60],
            )
        return decoded

    @staticmethod

    def _extract_workers_payload(response: str) -> tuple[str, Optional[str]]:
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


    def _parse_supervisor_response(self, response: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the WORKERS tag from *response* and update *state* in-place."""
        response_text = response.strip()
        answer, workers_payload = self._extract_workers_payload(response_text)

        if workers_payload is None:
            state["current_plan"] = []
            state["final_response"] = answer or response.strip()
            return state

        decoded_payload = self._decode_workers_payload(workers_payload)
        if not self._is_structured_plan_payload(decoded_payload):
            logger.warning("Supervisor produced invalid or non-structured WORKERS payload: %r", workers_payload)
            state["current_plan"] = []
            fallback_answer = answer if (answer and len(answer.strip()) >= 40) else response_text
            if fallback_answer == response_text:
                fallback_answer = re.sub(r"WORKERS:\s*.*$", "", fallback_answer, flags=re.DOTALL).strip()
            state["final_response"] = fallback_answer or answer or response_text
            return state

        normalized_plan = self._normalize_current_plan(decoded_payload)
        if normalized_plan:
            state["current_plan"] = normalized_plan
            if answer:
                state["worker_outputs"]["supervisor_context"] = answer
        else:
            state["current_plan"] = []
            state["final_response"] = answer or response_text
        return state

    def _get_requested_plan_steps(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._normalize_current_plan(list(state.get("current_plan", []) or []))

    @staticmethod

    def _get_real_worker_outputs(state: Dict[str, Any]) -> Dict[str, str]:
        _INTERNAL_OUTPUT_KEYS = {"supervisor_context", "_s2_blueprint"}
        return {
            name: output
            for name, output in dict(state.get("worker_outputs", {}) or {}).items()
            if name not in _INTERNAL_OUTPUT_KEYS
        }

    def _get_agent_registry(self) -> AgentRegistry:
        registry = getattr(self, "agent_registry", None)
        if registry is None:
            registry = AgentRegistry()
            self.agent_registry = registry
        return registry

    def _get_agent_descriptions(self) -> str:
        registry = self._get_agent_registry()
        agents = registry.all()
        if not agents:
            return "- No registered agents available."
        return "\n".join(f"- {agent.name}: {agent.description}" for agent in agents)

    def _build_agent_handoff(
        self,
        agent_def: AgentDefinition,
        state: Dict[str, Any],
        task_packet: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines = [f"User request: {state.get('user_input', '')}"]
        task_packet = task_packet or self._make_plan_step(agent_def.name)

        supervisor_context = state.get("worker_outputs", {}).get("supervisor_context", "")
        if supervisor_context:
            lines.append(f"Supervisor context: {supervisor_context}")

        # Inject session context so agents can scope their tool calls
        session = state.get("active_session")
        if isinstance(session, dict) and session.get("id"):
            session_line = (
                f"Active session: {session.get('name')} "
                f"(session_id={session['id']}"
            )
            if session.get("epic_id"):
                session_line += f", epic_id={session['epic_id']}"
            session_line += ")"
            lines.append(session_line)

        task_text = str(task_packet.get("task", "")).strip()
        reason_text = str(task_packet.get("reason", "")).strip()
        if task_text:
            lines.append(f"Your task: {task_text}")
        if reason_text:
            lines.append(f"Why you were selected: {reason_text}")

        dependency_outputs = []
        for dependency in self._get_step_dependencies(task_packet, agent_def):
            # Prefer structured dependency output when available
            structured_dep = (
                state.get("structured_outputs", {}).get(dependency)
            )
            if structured_dep is not None:
                formatted = format_structured_for_handoff(structured_dep, dependency)
                if formatted:
                    dependency_outputs.append(formatted)
                    continue
            # Fallback to raw text
            raw_dep = state.get("worker_outputs", {}).get(dependency, "")
            if raw_dep:
                dependency_outputs.append(
                    f"[{dependency} raw output]\n{dependency}: {raw_dep}"
                )

        if dependency_outputs:
            lines.append("Prior agent outputs (use these to inform your work):")
            for dep_output in dependency_outputs:
                lines.append(dep_output)

        return "<supervisor_handoff>\n" + "\n".join(lines) + "\n</supervisor_handoff>"

    def _get_step_dependencies(
        self,
        step: Optional[Dict[str, Any]],
        agent_def: Optional[AgentDefinition] = None,
    ) -> List[str]:
        step = step or self._make_plan_step("")
        if agent_def is None and step.get("agent"):
            agent_def = self._get_agent_registry().get(step["agent"])

        static_dependencies = list(agent_def.depends_on) if agent_def is not None else []
        dynamic_dependencies = self._normalize_plan_dependencies(step.get("depends_on"))
        return self._merge_plan_dependencies(static_dependencies, dynamic_dependencies)

    def _get_step_preferred_model(
        self,
        step: Optional[Dict[str, Any]],
        agent_def: Optional[AgentDefinition],
    ) -> str:
        step_preference = self._normalize_model_preference((step or {}).get("preferred_model"))
        if step_preference:
            return step_preference
        agent_preference = self._normalize_model_preference(agent_def.preferred_model if agent_def else "")
        return agent_preference or "system_1"

    def _collect_agent_dependencies(
        self,
        agent_name: str,
        step_map: Optional[Dict[str, Dict[str, Any]]] = None,
        visiting: Optional[set] = None,
    ) -> set:
        registry = self._get_agent_registry()
        agent_def = registry.get(agent_name)
        step = (step_map or {}).get(agent_name, self._make_plan_step(agent_name))
        if agent_def is None and not step.get("depends_on"):
            return set()

        visiting = visiting or set()
        if agent_name in visiting:
            return set()

        visiting.add(agent_name)
        dependencies = set()
        for dependency in self._get_step_dependencies(step, agent_def):
            dependencies.add(dependency)
            dependencies |= self._collect_agent_dependencies(dependency, step_map, visiting)
        visiting.remove(agent_name)
        return dependencies

    def _should_combine_requested_outputs(self, requested_steps: List[Dict[str, Any]]) -> bool:
        requested_names = [step["agent"] for step in requested_steps]
        if len(requested_names) <= 1:
            return False

        step_map = {step["agent"]: step for step in requested_steps}
        requested_set = set(requested_names)
        for agent_name in requested_names:
            if self._collect_agent_dependencies(agent_name, step_map) & requested_set:
                return False
        return True

    @staticmethod

    def _enrich_display_text(
        raw_text: str,
        state: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> str:
        """Prefer structured display text over raw worker output string."""
        if agent_name is not None:
            structured = state.get("structured_outputs", {}).get(agent_name)
            if structured is not None:
                enriched = get_display_text(structured, raw_text)
                if enriched and enriched != raw_text:
                    return enriched
        return raw_text

    def _get_output_to_evaluate(self, state: Dict[str, Any]) -> str:
        """Return the requested agent output, combining independent outputs when needed."""
        real_outputs = self._get_real_worker_outputs(state)
        if not real_outputs:
            return state.get("final_response", "")

        requested_steps = self._get_requested_plan_steps(state)
        requested_outputs = [
            (step["agent"], real_outputs[step["agent"]])
            for step in requested_steps
            if step["agent"] in real_outputs
        ]

        if not requested_outputs:
            last_worker = list(real_outputs.keys())[-1]
            return self._enrich_display_text(
                real_outputs[last_worker], state, last_worker
            )

        if len(requested_outputs) == 1 or not self._should_combine_requested_outputs(requested_steps):
            _last_agent_name, _last_output = requested_outputs[-1]
            return self._enrich_display_text(
                _last_output, state, _last_agent_name
            )

        return "\n\n".join(
            f"{agent_name}:\n{self._enrich_display_text(output, state, agent_name)}"
            for agent_name, output in requested_outputs
        )

    def _should_run_critic_review(self, state: Dict[str, Any], output_to_eval: str) -> bool:
        if not output_to_eval:
            return False

        real_outputs = self._get_real_worker_outputs(state)
        if not real_outputs:
            return False

        if state.get("critic_instructions"):
            return True

        if output_to_eval.strip().lower().startswith("error:"):
            return False

        requested_steps = self._get_requested_plan_steps(state)
        requested_output_count = sum(1 for step in requested_steps if step["agent"] in real_outputs)

        if requested_output_count > 1:
            return True

        return len(output_to_eval.strip()) >= _CRITIC_SHORT_OUTPUT_THRESHOLD


    async def _run_agent(
        self,
        agent_def: AgentDefinition,
        state: Dict[str, Any],
        *,
        task_packet: Optional[Dict[str, Any]] = None,
        deduct_energy: bool = True,
    ) -> Dict[str, Any]:
        state = normalize_state(state)
        state.setdefault("structured_outputs", {})
        if deduct_energy:
            state = self._deduct_energy(state, agent_def.energy_cost, agent_def.name)
        _agent_emitter = get_current_emitter()
        _agent_start_mono = time.monotonic()
        if _agent_emitter is not None:
            _agent_emitter.record_agent_start(agent_def.name)
            await _agent_emitter.emit(ProgressEvent.agent_start(agent_def.name))
        try:
            core_mem_str = await self.core_memory.get_context_string()

            capabilities_str = self._get_capabilities_string()
            handoff = self._build_agent_handoff(agent_def, state, task_packet=task_packet)
            # Build system prompt with structured output format injection
            _output_type = resolve_output_type(agent_def.output_type)
            _structured_output_enabled = (
                os.getenv("ENABLE_STRUCTURED_AGENT_OUTPUT", "true").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            _format_prompt = (
                OUTPUT_FORMAT_PROMPTS.get(_output_type, "")
                if _structured_output_enabled
                else ""
            )
            _system_content = (
                f"{agent_def.system_prompt}"
                f"{_format_prompt}"
                f"\n{self.charter_text}\n{core_mem_str}\n\n{capabilities_str}"
            )
            messages = [
                {
                    "role": "system",
                    "content": _system_content,
                },
                {"role": "user", "content": handoff},
            ]
            if _agent_emitter is not None:
                if "synthesis" in agent_def.name.lower():
                    await _agent_emitter.emit(ProgressEvent.synthesis_start())
            router_result, attempted_system_2 = await self._route_agent_request(
                messages,
                agent_def,
                task_packet=task_packet,
                state=state,
            )
        except Exception as _agent_exc:
            if deduct_energy:
                state = self._refund_energy(state, agent_def.energy_cost, f"{agent_def.name}_exception")
            if _agent_emitter is not None:
                _dur = time.monotonic() - _agent_start_mono
                await _agent_emitter.emit(
                    ProgressEvent.agent_done(agent_def.name, _dur)
                )
            raise _agent_exc

        if router_result is None:
            if deduct_energy:
                state = self._refund_energy(state, agent_def.energy_cost, f"{agent_def.name}_null_result")
            if attempted_system_2:
                state["worker_outputs"][agent_def.name] = (
                    f"Error: {agent_def.name} failed after System 1 error and System 2 fallback."
                )
                state["structured_outputs"][agent_def.name] = None
            else:
                state["worker_outputs"][agent_def.name] = (
                    f"Error: {agent_def.name} failed and System 2 is not configured."
                )
                state["structured_outputs"][agent_def.name] = None
            if _agent_emitter is not None:
                _dur = time.monotonic() - _agent_start_mono
                await _agent_emitter.emit(
                    ProgressEvent.agent_done(agent_def.name, _dur)
                )
            return state

        if router_result.status == "cognitive_escalation":
            solution = await self._handle_cognitive_escalation(state, router_result)
            state["worker_outputs"][agent_def.name] = solution
            # Structured extraction for escalation result (synthesis-like)
            if _structured_output_enabled and _output_type != AgentOutputType.TEXT:
                _structured = validate_agent_output(
                    extract_structured_output(solution),
                    _output_type,
                    agent_def.name,
                    solution,
                )
                if _structured is None and _output_type != AgentOutputType.TEXT:
                    _structured = make_extraction_failed_result(
                        _output_type, agent_def.name, solution
                    )
                state["structured_outputs"][agent_def.name] = _structured
            if _agent_emitter is not None:
                _dur = time.monotonic() - _agent_start_mono
                await _agent_emitter.emit(
                    ProgressEvent.agent_done(agent_def.name, _dur)
                )
            return state

        if router_result.status != "ok":
            state[_BLOCKED_KEY] = router_result
        else:
            state["worker_outputs"][agent_def.name] = router_result.content
            # Structured extraction from raw response
            if _structured_output_enabled and _output_type != AgentOutputType.TEXT:
                _raw_content = str(router_result.content or "")
                _extracted = extract_structured_output(_raw_content)
                _structured = validate_agent_output(
                    _extracted, _output_type, agent_def.name, _raw_content
                )
                if _structured is None:
                    _structured = make_extraction_failed_result(
                        _output_type, agent_def.name, _raw_content
                    )
                state["structured_outputs"][agent_def.name] = _structured
                # Store clean version (without <agent_output> block) as the raw output
                _clean = strip_agent_output_block(_raw_content)
                if _clean != _raw_content:
                    state["worker_outputs"][agent_def.name] = _clean
                logger.info(
                    "Agent %s structured output: type=%s success=%s confidence=%.2f "
                    "extraction_failed=%s",
                    agent_def.name,
                    _output_type.value,
                    _structured.get("success"),
                    float(_structured.get("confidence", 0)),
                    _structured.get("extraction_failed", False),
                )
            if _agent_emitter is not None:
                _dur = time.monotonic() - _agent_start_mono
                await _agent_emitter.emit(
                    ProgressEvent.agent_done(agent_def.name, _dur)
                )

        return state

    async def _run_parallel_agent_batch(
        self,
        batch: List[tuple[AgentDefinition, Dict[str, Any]]],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        batch_names = [agent_def.name for agent_def, _ in batch]
        state = self._deduct_energy(state, sum(agent_def.energy_cost for agent_def, _ in batch), f"parallel_agents:{batch_names}")
        base_state = self._build_agent_state_snapshot(state)
        results = await asyncio.gather(
            *[
                self._run_agent(
                    agent_def,
                    self._build_agent_state_snapshot(base_state),
                    task_packet=task_packet,
                    deduct_energy=False,
                )
                for agent_def, task_packet in batch
            ],
            return_exceptions=True,
        )

        self._raise_parallel_blocking_exception(results)
        blocked_result, blocked_agent_names = self._find_parallel_blocked_result(batch, results)

        if blocked_result is not None:
            state = await self._refund_blocked_parallel_batch_energy(batch, state, blocked_agent_names)
            state[_BLOCKED_KEY] = blocked_result
            return state

        return self._merge_parallel_agent_results(batch, results, state)

    def _build_execution_batches(
        self,
        plan_steps: List[Dict[str, Any]],
    ) -> tuple:
        """Return (batches, skipped_cycles) where batches is a list of dependency-aware execution
        batches and skipped_cycles lists agents dropped due to cyclic dependencies."""
        registry = self._get_agent_registry()
        normalized_steps = self._normalize_current_plan(plan_steps)
        step_map: Dict[str, Dict[str, Any]] = {}
        step_order: Dict[str, int] = {}
        order_counter = 0
        visiting = set()
        resolving_depth = set()
        skipped_cycles: List[str] = []

        for step in normalized_steps:
            order_counter = self._visit_execution_step(
                step,
                registry,
                step_map,
                step_order,
                visiting,
                order_counter,
                skipped_cycles,
            )

        depth_cache: Dict[str, int] = {}

        batched: Dict[int, List[tuple[AgentDefinition, Dict[str, Any]]]] = {}
        for agent_name, step in step_map.items():
            agent_def = registry.get(agent_name)
            if agent_def is None:
                continue
            depth = self._resolve_execution_depth(
                agent_name,
                registry,
                step_map,
                depth_cache,
                resolving_depth,
            )
            batched.setdefault(depth, []).append((agent_def, step))

        return self._group_execution_batches(batched, step_order), skipped_cycles


    def _visit_execution_step(
        self,
        step: Dict[str, Any],
        registry: AgentRegistry,
        step_map: Dict[str, Dict[str, Any]],
        step_order: Dict[str, int],
        visiting: set,
        order_counter: int,
        skipped_cycles: Optional[List[str]] = None,
    ) -> int:
        agent_name = step["agent"]
        if self._merge_execution_step(step_map, step):
            return order_counter
        if agent_name in visiting:
            logger.warning("Dependency cycle detected while planning agent %r — skipping.", agent_name)
            if skipped_cycles is not None:
                skipped_cycles.append(agent_name)
            return order_counter

        agent_def = registry.get(agent_name)
        if agent_def is None:
            logger.warning("Unknown agent requested by supervisor: %r", agent_name)
            return order_counter

        visiting.add(agent_name)
        for dependency in self._get_step_dependencies(step, agent_def):
            order_counter = self._visit_execution_step(
                self._make_plan_step(dependency),
                registry,
                step_map,
                step_order,
                visiting,
                order_counter,
                skipped_cycles,
            )
        visiting.remove(agent_name)

        step_map[agent_name] = dict(step)
        step_order[agent_name] = order_counter
        return order_counter + 1

    def _resolve_execution_depth(
        self,
        agent_name: str,
        registry: AgentRegistry,
        step_map: Dict[str, Dict[str, Any]],
        depth_cache: Dict[str, int],
        resolving_depth: set,
    ) -> int:
        if agent_name in depth_cache:
            return depth_cache[agent_name]
        if agent_name in resolving_depth:
            logger.warning("Dependency depth cycle detected for agent %r", agent_name)
            return 0

        resolving_depth.add(agent_name)
        agent_def = registry.get(agent_name)
        dependency_depths = []
        if agent_def is not None or agent_name in step_map:
            step = step_map.get(agent_name, self._make_plan_step(agent_name))
            dependency_depths = [
                self._resolve_execution_depth(dependency, registry, step_map, depth_cache, resolving_depth)
                for dependency in self._get_step_dependencies(step, agent_def)
                if dependency in step_map
            ]
        resolving_depth.remove(agent_name)

        depth = (max(dependency_depths) + 1) if dependency_depths else 0
        depth_cache[agent_name] = depth
        return depth

    @staticmethod

    def _group_execution_batches(
        batched: Dict[int, List[tuple[AgentDefinition, Dict[str, Any]]]],
        step_order: Dict[str, int],
    ) -> List[List[tuple[AgentDefinition, Dict[str, Any]]]]:
        return [
            sorted(batch, key=lambda item: step_order[item[0].name])
            for _, batch in sorted(batched.items(), key=lambda item: item[0])
        ]

    def _merge_execution_step(
        self,
        step_map: Dict[str, Dict[str, Any]],
        step: Dict[str, Any],
    ) -> bool:
        agent_name = step["agent"]
        existing = step_map.get(agent_name)
        if existing is None:
            return False
        existing["task"] = self._merge_plan_text(existing["task"], step.get("task", ""))
        existing["reason"] = self._merge_plan_text(existing["reason"], step.get("reason", ""))
        existing["depends_on"] = self._merge_plan_dependencies(
            existing.get("depends_on", []),
            self._normalize_plan_dependencies(step.get("depends_on")),
        )
        existing["preferred_model"] = existing.get("preferred_model") or step.get("preferred_model", "")
        return True

    @staticmethod
    def _build_agent_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = dict(state)
        snapshot["chat_history"] = list(state.get("chat_history", []) or [])
        snapshot["current_plan"] = list(state.get("current_plan", []) or [])
        snapshot["worker_outputs"] = dict(state.get("worker_outputs", {}) or {})
        snapshot["structured_outputs"] = dict(
            state.get("structured_outputs", {}) or {}
        )
        return snapshot

    @staticmethod

    def _raise_parallel_blocking_exception(results: List[Any]) -> None:
        for result in results:
            if isinstance(result, (RequiresHITLError, RequiresMFAError)):
                raise result

    @staticmethod

    def _find_parallel_blocked_result(
        batch: List[tuple[AgentDefinition, Dict[str, Any]]],
        results: List[Any],
    ) -> tuple[Optional[RouterResult], set[str]]:
        blocked_result = None
        blocked_agent_names: set[str] = set()
        for (agent_def, _task_packet), result in zip(batch, results):
            if isinstance(result, Exception):
                continue
            if result.get(_BLOCKED_KEY):
                blocked_result = blocked_result or result[_BLOCKED_KEY]
                blocked_agent_names.add(agent_def.name)
        return blocked_result, blocked_agent_names

    async def _refund_blocked_parallel_batch_energy(
        self,
        batch: List[tuple[AgentDefinition, Dict[str, Any]]],
        state: Dict[str, Any],
        blocked_agent_names: set[str],
    ) -> Dict[str, Any]:
        unused_energy = sum(
            agent_def.energy_cost
            for agent_def, _task_packet in batch
            if agent_def.name not in blocked_agent_names
        )
        state = self._refund_energy(state, unused_energy, "parallel_agents_blocked_unused")
        if state.get("_energy_gate_cleared"):
            await self._refund_predictive_energy_budget(unused_energy, "parallel_agents_blocked_unused")
        return state

    @staticmethod

    def _merge_parallel_agent_results(
        batch: List[tuple[AgentDefinition, Dict[str, Any]]],
        results: List[Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        state.setdefault("structured_outputs", {})
        for (agent_def, _task_packet), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(
                    "Parallel agent %s failed: %s",
                    agent_def.name, result, exc_info=True
                )
                state["worker_outputs"][agent_def.name] = (
                    f"Error: {agent_def.name} failed: {result}"
                )
                state["structured_outputs"][agent_def.name] = None
                continue

            agent_output = dict(
                result.get("worker_outputs", {}) or {}
            ).get(agent_def.name)
            if agent_output is not None:
                state["worker_outputs"][agent_def.name] = agent_output

            structured = dict(
                result.get("structured_outputs", {}) or {}
            ).get(agent_def.name)
            state["structured_outputs"][agent_def.name] = structured
        return state

    async def research_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility wrapper for the built-in research agent."""
        agent_def = self._get_agent_registry().get("research_agent")
        if agent_def is None:
            return normalize_state(state)
        return await self._run_agent(agent_def, state)

    async def coder_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility wrapper for the built-in coder agent."""
        agent_def = self._get_agent_registry().get("coder_agent")
        if agent_def is None:
            return normalize_state(state)
        return await self._run_agent(agent_def, state)

    @staticmethod

    def _store_moral_decision_trace(state: Dict[str, Any], decision: MoralDecision) -> None:
        state["moral_decision"] = decision.to_dict()
        state["moral_audit_mode"] = str(decision.decision_mode)
        state["moral_audit_trace"] = str(decision.bypass_reason or decision.reasoning)
        state["moral_audit_bypassed"] = decision.decision_mode in {"triviality_bypass", "local_skip"}

    @staticmethod

    def _contains_any_hint(text: str, hints: tuple[str, ...]) -> bool:
        lowered = str(text or "").lower()
        return any(hint in lowered for hint in hints)

    def _build_moral_triviality_text(self, state: Dict[str, Any], output_to_eval: str) -> str:
        parts: List[str] = [
            str(state.get("user_input") or ""),
            str(output_to_eval or ""),
        ]
        for step in state.get("current_plan", []) or []:
            if isinstance(step, dict):
                parts.append(str(step.get("agent") or ""))
                parts.append(str(step.get("task") or ""))
                parts.append(str(step.get("reason") or ""))
            else:
                parts.append(str(step))
        return "\n".join(part for part in parts if part)

    def _is_triviality_bypass_blocked(self, state: Dict[str, Any], inspection_text: str) -> bool:
        if state.get("critic_instructions"):
            return True
        if int(state.get("iteration_count") or 0) > 0:
            return True
        if not str(inspection_text or "").strip():
            return True
        if self._contains_any_hint(inspection_text, _MORAL_TRIVIALITY_BLOCK_HINTS):
            return True
        if len(state.get("current_plan", []) or []) > 1:
            return True
        return False

    def _try_route_assessment_triviality_bypass(self, user_input: str) -> Optional[MoralDecision]:
        try:
            assessment = self.routing_assessor.assess_request_route(user_input)
        except Exception:
            return None

        mode = str(assessment.get("mode") or "")
        if mode == "single_tool":
            tool_name = str(assessment.get("tool_name") or "")
            if tool_name in _FAST_PATH_SINGLE_TOOL_ALLOWLIST:
                return build_triviality_bypass_decision(
                    f"single_tool_read_only:{tool_name}"
                )
        if mode == "direct" and RoutingAssessor._is_trivial_direct_intent(user_input):
            return build_triviality_bypass_decision("direct_trivial_read_only")
        return None

    def _try_triviality_bypass_decision(
        self,
        state: Dict[str, Any],
        output_to_eval: str,
    ) -> Optional[MoralDecision]:
        inspection_text = self._build_moral_triviality_text(state, output_to_eval)
        if self._is_triviality_bypass_blocked(state, inspection_text):
            return None

        user_input = str(state.get("user_input") or "").strip()
        if not user_input:
            return None

        route_based_decision = self._try_route_assessment_triviality_bypass(user_input)
        if route_based_decision is not None:
            return route_based_decision

        if self._contains_any_hint(inspection_text, _MORAL_TRIVIALITY_READ_ONLY_HINTS):
            return build_triviality_bypass_decision("keyword_read_only")

        return None

    async def _spawn_debug_task(self, state: Dict[str, Any]) -> None:
        """Inject a high-priority debug Task into the backlog after 3 critic failures."""
        try:
            real_outputs = self._get_real_worker_outputs(state)
            last_worker = list(real_outputs.keys())[-1] if real_outputs else "unknown"
            task_title = f"Debug failing [{last_worker}] logic for: {state.get('user_input', '')[:80]}"
            await self.ledger_memory.add_objective(
                tier="Task",
                title=task_title,
                estimated_energy=20,
                origin="Critic",
                priority=2,
            )
            logger.info(f"Critic: injected debug Task into backlog after 3 failures: {task_title!r}")
        except Exception as spawn_err:
            logger.warning(f"Critic: failed to spawn debug objective: {spawn_err}")


    def _build_critic_messages(self, state: Dict[str, Any], output_to_eval: str) -> List[Dict[str, str]]:
        tier_1 = self._extract_charter_tier_block("Tier_1_Axioms")
        tier_2 = self._extract_charter_tier_block("Tier_2_Strategic")
        tier_3 = self._extract_charter_tier_block("Tier_3_Operational")

        redacted_user_input = cloud_redaction.redact_text_for_cloud(
            str(state.get("user_input") or ""),
            allow_sensitive_context=False,
            max_chars=1600,
        )
        redacted_plan = cloud_redaction.redact_text_for_cloud(
            self._summarize_plan_for_moral_audit(state),
            allow_sensitive_context=False,
            max_chars=1200,
        )
        redacted_output = cloud_redaction.redact_text_for_cloud(
            str(output_to_eval or ""),
            allow_sensitive_context=False,
            max_chars=2200,
        )
        redacted_critic_instructions = cloud_redaction.redact_text_for_cloud(
            str(state.get("critic_instructions") or ""),
            allow_sensitive_context=False,
            max_chars=600,
        )

        schema_example = json.dumps(self._build_moral_json_schema_example(), indent=2)
        system_prompt = (
            "You are the Hierarchical Moral Evaluator for the Critic. "
            "Evaluate the candidate action/output against the charter with explicit tier grounding.\n\n"
            "Tier mapping requirements:\n"
            "- Tier 1 (absolute axioms): harm_reduction, data_privacy, admin_authority_security, data_sovereignty_local_priority\n"
            "- Tier 2 (strategic objectives): epistemic_humility, transparency_logging, alignment_with_user_intent\n"
            "- Tier 3 (operational preferences): output_cleanliness\n"
            "If Tier 1 conflicts with Tier 2/3, Tier 1 takes precedence.\n\n"
            "Return ONLY strict JSON, no markdown, no prose, no extra keys.\n"
            "The JSON must match this exact schema shape and key names:\n"
            f"{schema_example}\n\n"
            "Scoring rules:\n"
            "- Every score must be an integer from 1 to 5.\n"
            "- Set rubric_version exactly to moral_ledger_v1.\n"
            "- Set decision_mode exactly to system2_audit and bypass_reason to an empty string.\n"
            "- is_approved must be false if any Tier 1 directive is materially violated.\n"
            "- If is_approved is false and remediation is possible, include remediation_constraints as concise actionable constraints.\n"
            "- If this is a Tier 1/security violation, include violated_tiers with tier_1 and set security_conflict=true when applicable.\n"
            "- reasoning must cite the most important charter-grounded reason.\n\n"
            "Charter grounding excerpts:\n"
            f"<Tier_1_Axioms>\n{tier_1 or '[MISSING_TIER_1]'}\n</Tier_1_Axioms>\n"
            f"<Tier_2_Strategic>\n{tier_2 or '[MISSING_TIER_2]'}\n</Tier_2_Strategic>\n"
            f"<Tier_3_Operational>\n{tier_3 or '[MISSING_TIER_3]'}\n</Tier_3_Operational>"
        )

        user_payload = {
            "user_request_redacted": redacted_user_input,
            "proposed_plan_redacted": redacted_plan,
            "candidate_output_redacted": redacted_output,
            "critic_instructions_redacted": redacted_critic_instructions,
            "iteration_count": int(state.get("iteration_count") or 0),
        }

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ]

    @staticmethod
    def _build_moral_json_schema_example() -> Dict[str, Any]:
        return {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": dict.fromkeys(MORAL_DIMENSIONS, 1),
            "reasoning": "concise rationale that cites the strongest risk/constraint",
            "is_approved": False,
            "decision_mode": "system2_audit",
            "bypass_reason": "",
            "remediation_constraints": [
                "constraint to satisfy before retry"
            ],
            "violated_tiers": ["tier_2"],
            "security_conflict": False,
        }

    @staticmethod
    def _summarize_plan_for_moral_audit(state: Dict[str, Any]) -> str:
        plan = list(state.get("current_plan", []) or [])
        if not plan:
            return "[NO_EXPLICIT_PLAN]"

        lines: List[str] = []
        for idx, step in enumerate(plan[:6], start=1):
            if isinstance(step, dict):
                agent = str(step.get("agent") or "").strip()
                task = str(step.get("task") or "").strip()
                reason = str(step.get("reason") or "").strip()
                lines.append(f"{idx}. agent={agent}; task={task}; reason={reason}")
            else:
                lines.append(f"{idx}. {str(step).strip()}")
        return "\n".join(lines) if lines else "[NO_EXPLICIT_PLAN]"


    async def _persist_moral_audit_log(self, state: Dict[str, Any], output_to_eval: str) -> None:
        ledger = getattr(self, "ledger_memory", None)
        append_fn = getattr(ledger, "append_moral_audit_log", None)
        if not callable(append_fn):
            return

        decision = dict(state.get("moral_decision", {}) or {})
        if not decision:
            return

        try:
            await append_fn(
                user_id=str(state.get("user_id") or ""),
                audit_mode=str(state.get("moral_audit_mode") or decision.get("decision_mode") or ""),
                audit_trace=str(state.get("moral_audit_trace") or decision.get("reasoning") or ""),
                critic_feedback=str(state.get("critic_feedback") or ""),
                moral_decision=decision,
                request_redacted=cloud_redaction.redact_text_for_cloud(
                    str(state.get("user_input") or ""),
                    allow_sensitive_context=False,
                    max_chars=1600,
                ),
                output_redacted=cloud_redaction.redact_text_for_cloud(
                    str(output_to_eval or ""),
                    allow_sensitive_context=False,
                    max_chars=2200,
                ),
            )
        except Exception as e:
            logger.warning("Failed to persist moral audit log entry: %s", e)


    def _apply_critic_response(
        self,
        state: Dict[str, Any],
        output_to_eval: str,
        response: str,
    ) -> Dict[str, Any]:
        decision = parse_moral_decision_response(response)
        if decision.decision_mode == "validation_failure":
            response_text = str(response or "")
            if "FAIL" in response_text.upper():
                decision = build_legacy_binary_decision(
                    is_approved=False,
                    reasoning=response_text.strip() or "Legacy critic rejected output.",
                )
            elif "PASS" in response_text.upper():
                decision = build_legacy_binary_decision(
                    is_approved=True,
                    reasoning="Legacy critic approved output.",
                )

        self._store_moral_decision_trace(state, decision)
        state["moral_halt_required"] = False
        state["moral_halt_summary"] = ""
        state["moral_remediation_constraints"] = []

        if not decision.is_approved:
            state["iteration_count"] += 1
            severity, summary = self._classify_moral_rejection(decision)
            if severity == "severe":
                state["moral_halt_required"] = True
                state["moral_halt_summary"] = summary
                state["critic_feedback"] = f"FAIL: {summary}"
                return state

            remediation_constraints = [
                str(item).strip()
                for item in tuple(decision.remediation_constraints or ())
                if str(item).strip()
            ]
            if remediation_constraints:
                state["moral_remediation_constraints"] = remediation_constraints
                constraints_text = " | ".join(remediation_constraints[:6])
                state["critic_feedback"] = (
                    f"FAIL: {summary} Remediation constraints: {constraints_text}"
                )
                return state

            state["critic_feedback"] = f"FAIL: {summary}"
            return state
        return self._finalize_critic_pass(state, output_to_eval)

    @staticmethod

    def _apply_fallback_charter_critic_skip(state: Dict[str, Any], output_to_eval: str) -> Dict[str, Any]:
        state["critic_feedback"] = "PASS"
        state["moral_audit_mode"] = "disabled_fallback_charter"
        state["moral_audit_trace"] = "fallback charter active"
        state["moral_audit_bypassed"] = True
        state["moral_remediation_constraints"] = []
        state["moral_halt_required"] = False
        state["moral_halt_summary"] = ""
        if output_to_eval and not state.get("final_response"):
            state["final_response"] = output_to_eval
        return state

    async def _try_pre_review_critic_skip(  # NOSONAR
        self,
        state: Dict[str, Any],
        output_to_eval: str,
    ) -> Optional[Dict[str, Any]]:
        if self._is_fallback_charter_active():
            logger.warning(
                "Critic disabled because fallback charter is active; skipping critic and moral audit for this turn."
            )
            return self._apply_fallback_charter_critic_skip(state, output_to_eval)

        if not output_to_eval:
            self._store_moral_decision_trace(
                state,
                build_local_skip_decision("empty_output"),
            )
            state["critic_feedback"] = "PASS"
            await self._persist_moral_audit_log(state, output_to_eval)
            return state

        return None

    @staticmethod
    def _finalize_critic_pass(state: Dict[str, Any], output_to_eval: str) -> Dict[str, Any]:
        state["critic_feedback"] = "PASS"
        state["moral_remediation_constraints"] = []
        state["moral_halt_required"] = False
        state["moral_halt_summary"] = ""
        if not state.get("final_response"):
            state["final_response"] = output_to_eval
        return state


    def _classify_moral_rejection(self, decision: MoralDecision) -> tuple[str, str]:
        if (
            bool(decision.security_conflict)
            or self._decision_has_tier_1_violation(decision)
            or self._contains_security_conflict_language(decision.reasoning)
        ):
            return "severe", f"Tier 1/security conflict: {decision.reasoning}".strip()

        if tuple(decision.remediation_constraints or ()):
            return "moderate", str(decision.reasoning or "Moderate moral risk; retry with constraints.").strip()

        return "standard", str(decision.reasoning or "Moral review rejected output.").strip()

    @staticmethod
    def _decision_has_tier_1_violation(decision: MoralDecision) -> bool:
        normalized_tiers = {
            str(tier).strip().lower().replace("-", "_")
            for tier in tuple(decision.violated_tiers or ())
        }
        if "tier_1" in normalized_tiers or "tier1" in normalized_tiers:
            return True

        for dimension in MORAL_TIER_1_DIMENSIONS:
            try:
                score = int(decision.scores.get(dimension, 5))
            except Exception:
                score = 5
            if score <= 2:
                return True
        return False

    @staticmethod
    def _contains_security_conflict_language(text: str) -> bool:
        lowered = str(text or "").lower()
        indicators = (
            "security conflict",
            "tier 1 violation",
            "admin authority conflict",
            "unauthorized write",
            "unauthorized modify",
            "unauthorized delete",
            "mfa required",
            "data exfiltration",
        )
        return any(indicator in lowered for indicator in indicators)

    @staticmethod

    def _apply_critic_retry_instructions(state: Dict[str, Any]) -> None:
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

    @staticmethod
    def _reset_after_critic_rejection(state: Dict[str, Any]) -> None:
        state["final_response"] = ""
        state["worker_outputs"] = {}
        state["current_plan"] = []
        state["moral_halt_required"] = False
        state["moral_halt_summary"] = ""

    @staticmethod

    def _ensure_final_response(state: Dict[str, Any], max_iterations: int) -> None:
        if state["iteration_count"] >= max_iterations and state.get("critic_feedback") != "PASS":
            state["_turn_failed"] = True
            state["final_response"] = "Unable to fulfill this request — output repeatedly failed internal safety checks."
        if not state.get("final_response"):
            state["_turn_failed"] = True
            state["final_response"] = "No valid response could be generated."

    async def _get_recent_moral_rejections_for_supervisor(self, user_id: str) -> List[Dict[str, Any]]:
        getter = getattr(self.ledger_memory, "get_recent_moral_rejections", None)
        if not callable(getter):
            return []
        try:
            return await getter(str(user_id or ""), limit=3)
        except Exception as exc:
            logger.warning("Could not fetch recent moral rejections for supervisor context: %s", exc)
            return []

    def _refresh_sensory_state(self) -> None:
        """Snapshot current machine state into self.sensory_state."""
        cpu = f"{psutil.cpu_percent(interval=None):.0f}%" if _PSUTIL_AVAILABLE else "unavailable"

        # platform.release() returns "10" even on Windows 11 — check build number instead
        os_name = platform.system()
        if os_name == "Windows":
            try:
                build = int(platform.version().split(".")[-1])
                win_ver = "11" if build >= 22000 else "10"
                os_str = f"Windows {win_ver} (build {build})"
            except Exception:
                os_str = f"Windows {platform.release()}"
        else:
            os_str = f"{os_name} {platform.release()}"

        self.sensory_state = {
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "os": os_str,
            "cpu_usage": cpu,
            "cwd": os.getcwd(),
        }

    async def _sensory_update_loop(self) -> None:
        """Background task: refresh sensory_state every 60 seconds."""
        while True:
            await asyncio.sleep(60)
            self._refresh_sensory_state()

    def _get_sensory_context(self) -> str:
        s = self.sensory_state
        return (
            f"[Machine Context — {s.get('current_time', 'unknown')} | "
            f"OS: {s.get('os', 'unknown')} | "
            f"CPU: {s.get('cpu_usage', 'unknown')} | "
            f"CWD: {s.get('cwd', 'unknown')}]"
        )


    async def supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plans execution. Tries System 1 (local) first per Directive 1.3; escalates to System 2 only on failure."""
        state = normalize_state(state)
        _sv_emitter = get_current_emitter()
        if _sv_emitter is not None:
            await _sv_emitter.emit(ProgressEvent.supervisor_planning())

        if not state.get("_energy_gate_cleared"):
            deferred = await self._try_ad_hoc_dispatch_energy_gate(
                state,
                dispatch_context="graph_dispatch",
            )
            if deferred is not None:
                state["current_plan"] = []
                state["final_response"] = deferred
                return state
            state["_energy_gate_cleared"] = True

        state = self._deduct_energy(state, ENERGY_COST_SUPERVISOR, "supervisor")
        user_input = state["user_input"]
        user_id = str(state.get("user_id") or "")
        core_mem_str = await self.core_memory.get_context_string()
        _sup_session = state.get("active_session")
        _sup_session_id = int(_sup_session.get("id") or 0) if _sup_session else 0
        _sup_epic_id = int(_sup_session.get("epic_id") or 0) if _sup_session else 0
        archival_context = await self._get_archival_context(
            user_input,
            session_id=_sup_session_id or None,
            epic_id=_sup_epic_id or None,
        )
        capabilities_str = self._get_capabilities_string()
        archival_block = f"{archival_context}\n\n" if archival_context else ""
        recent_rejections = await self._get_recent_moral_rejections_for_supervisor(
            str(state.get("user_id") or "")
        )

        system_prompt = build_supervisor_prompt(
            charter_text=self.charter_text,
            core_mem_str=core_mem_str,
            archival_block=archival_block,
            capabilities_str=capabilities_str,
            agent_descriptions=self._get_agent_descriptions(),
            sensory_context=self._get_sensory_context(),
            os_name=self.sensory_state.get("os", platform.system()),
            downloads_dir=self.prompt_config.downloads_dir,
            recent_rejections=recent_rejections,
        )
        logger.info(
            "Supervisor system prompt sha256=%s bytes=%d",
            hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            len(system_prompt.encode("utf-8")),
        )

        # Inject any in-turn blueprint from prior System 2 escalations
        _blueprint = (state.get("worker_outputs") or {}).get("_s2_blueprint", "")
        _blueprint_section = (
            f"\n<s2_in_turn_blueprint>\n{_blueprint}\n</s2_in_turn_blueprint>"
            if _blueprint else ""
        )

        # Fetch epic rollup for session context block (non-blocking, best-effort)
        _sup_epic_rollup: Optional[Dict[str, Any]] = None
        if _sup_session and _sup_session.get("epic_id"):
            _sup_epic_rollup = await self._get_session_epic_rollup(
                _sup_session.get("epic_id")
            )

        turn_context = build_supervisor_turn_context(
            sensory_context=self._get_sensory_context(),
            archival_block=archival_block,
            active_session=_sup_session if _sup_session else None,
            epic_rollup=_sup_epic_rollup,
        ) + _blueprint_section
        user_prompt_message = self._build_user_prompt_message(state)
        if turn_context:
            user_prompt_message = dict(user_prompt_message)
            user_prompt_message["content"] = f"{turn_context}\n\n{user_prompt_message['content']}"

        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.get("chat_history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        if state.get("critic_instructions"):
            messages.append({"role": "system", "content": state["critic_instructions"]})
        messages.append(user_prompt_message)

        try:
            router_result = await self._route_supervisor_request(messages)

            if router_result is not None and router_result.status == "cognitive_escalation":
                solution = await self._handle_cognitive_escalation(state, router_result)
                # Route through worker_outputs so critic_node evaluates the answer against the charter
                state["worker_outputs"]["system_2_escalation"] = solution
                state["current_plan"] = [{"agent": "system_2_escalation", "task": "", "reason": "cognitive escalation", "depends_on": [], "preferred_model": "system_2"}]
                return state

            if router_result is not None and router_result.status != "ok":
                state[_BLOCKED_KEY] = router_result
                return state

            response = router_result.content if router_result else None

            if not response or response.startswith(_SYSTEM_1_ERROR_PREFIX) or response.startswith("[System 2"):
                logger.warning(f"Supervisor received error or no response: {response!r}")
                state["current_plan"] = []
                if not self.cognitive_router.get_system_2_available():
                    state["_turn_failed"] = True
                    state["final_response"] = (
                        "I was unable to process this request locally. This task likely requires "
                        "capabilities beyond my local model (e.g. internet access or complex reasoning). "
                        "To enable this, configure a System 2 provider by setting GROQ_API_KEY in your .env file. "
                        "Groq offers a free tier at console.groq.com."
                    )
                else:
                    state["_turn_failed"] = True
                    state["final_response"] = "Both local and cloud reasoning failed on this request. Please try rephrasing or simplifying the task."
                return state

            state = self._parse_supervisor_response(response, state)
            _sv_emitter2 = get_current_emitter()
            if _sv_emitter2 is not None:
                _plan_names = [
                    step["agent"]
                    for step in state.get("current_plan", [])
                    if isinstance(step, dict) and step.get("agent")
                ]
                if _plan_names:
                    await _sv_emitter2.emit(
                        ProgressEvent.supervisor_done(_plan_names)
                    )
                elif state.get("final_response"):
                    await _sv_emitter2.emit(ProgressEvent.supervisor_direct())

            # -- Supervisor decision logging for introspection -----------------
            # Heartbeat tasks are never logged; they are background work,
            # not user-initiated turns.
            if user_id != "heartbeat":
                _plan_for_log = list(state.get("current_plan", []) or [])
                _reasoning_for_log = str(
                    state.get("worker_outputs", {}).get("supervisor_context", "")
                    or ""
                )
                # If no plan and a direct response was set, capture a snippet
                # of it as the reasoning so introspection can explain it.
                if not _plan_for_log and state.get("final_response"):
                    _direct_preview = str(state["final_response"] or "")[:300]
                    _reasoning_for_log = _reasoning_for_log or _direct_preview

                _active_session_for_log = state.get("active_session")
                _session_id_for_log: Optional[int] = None
                if isinstance(_active_session_for_log, dict):
                    _session_id_for_log = _active_session_for_log.get("id") or None

                _log_supervisor_decision = getattr(
                    self.ledger_memory,
                    "log_supervisor_decision",
                    None,
                )
                if callable(_log_supervisor_decision):
                    self._fire_and_forget(
                        _log_supervisor_decision(
                            user_id=str(user_id),
                            user_input=str(user_input or "")[:500],
                            plan_json=json.dumps(
                                [
                                    {
                                        "agent": step.get("agent", ""),
                                        "task": str(step.get("task", ""))[:200],
                                        "reason": str(step.get("reason", ""))[:200],
                                    }
                                    for step in _plan_for_log
                                    if isinstance(step, dict)
                                ]
                            ),
                            is_direct=not bool(_plan_for_log),
                            reasoning=_reasoning_for_log[:1000],
                            energy_before=int(state.get("energy_remaining", 0)),
                            worker_count=len(_plan_for_log),
                            session_id=_session_id_for_log,
                        )
                    )
                    state["_supervisor_decision_logged"] = True
            # -- End supervisor decision logging -------------------------------

        except asyncio.TimeoutError:
            logger.error("Supervisor node timeout (60s)")
            state = self._refund_energy(state, ENERGY_COST_SUPERVISOR, "supervisor_timeout")
            state["current_plan"] = []
            state["_turn_failed"] = True
            state["final_response"] = "Planning timed out. Please try again."
        except RequiresHITLError:
            raise
        except Exception as e:
            logger.error(f"Supervisor Node failed: {e}", exc_info=True)
            state = self._refund_energy(state, ENERGY_COST_SUPERVISOR, "supervisor_error")
            state["current_plan"] = []
            state["_turn_failed"] = True
            state["final_response"] = "Supervisor encountered an error. Please try again."

        return state

    async def execute_workers_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph worker-execution node: runs all workers declared in current_plan."""
        state = normalize_state(state)
        if not state.get("current_plan"):
            if not state.get("final_response"):
                state["final_response"] = "No response could be generated. Please try again."
            return state

        state["current_plan"] = self._get_requested_plan_steps(state)
        executed_agent = False
        execution_batches, skipped_cycles = self._build_execution_batches(state["current_plan"])
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
                state = await self._run_agent(agent_def, state, task_packet=task_packet)
            else:
                state = await self._run_parallel_agent_batch(batch, state)

            if state.get(_BLOCKED_KEY):
                return state

        if not executed_agent and not state.get("final_response"):
            state["final_response"] = "No registered agent could execute the current plan."
        return state

    async def critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Checks worker output against the charter. Skipped for direct supervisor responses."""
        state = normalize_state(state)
        output_to_eval = self._get_output_to_evaluate(state)

        pre_review_skip = await self._try_pre_review_critic_skip(state, output_to_eval)
        if pre_review_skip is not None:
            return pre_review_skip

        bypass_decision = self._try_triviality_bypass_decision(state, output_to_eval)
        if bypass_decision is not None:
            self._store_moral_decision_trace(state, bypass_decision)
            state = self._finalize_critic_pass(state, output_to_eval)
            await self._persist_moral_audit_log(state, output_to_eval)
            return state

        _critic_emitter = get_current_emitter()
        if _critic_emitter is not None and output_to_eval:
            await _critic_emitter.emit(ProgressEvent.critic_start())

        if not self._should_run_critic_review(state, output_to_eval):
            self._store_moral_decision_trace(
                state,
                build_local_skip_decision("short_or_no_review_needed"),
            )
            state = self._finalize_critic_pass(state, output_to_eval)
            await self._persist_moral_audit_log(state, output_to_eval)
            return state

        state = self._deduct_energy(state, ENERGY_COST_CRITIC, "critic")
        messages = self._build_critic_messages(state, output_to_eval)

        try:
            router_result = await self._route_critic_request(messages)
            if router_result.status == "cognitive_escalation":
                solution = await self._handle_cognitive_escalation(state, router_result)
                state = self._apply_critic_response(state, output_to_eval, solution)
            elif router_result.status != "ok":
                state[_BLOCKED_KEY] = router_result
                return state
            else:
                state = self._apply_critic_response(state, output_to_eval, router_result.content)
            await self._persist_moral_audit_log(state, output_to_eval)
        except Exception as e:
            logger.warning(f"Critic node failed/timed out: {e}. Defaulting to PASS.", exc_info=True)
            state = self._refund_energy(state, ENERGY_COST_CRITIC, "critic_error")
            self._store_moral_decision_trace(
                state,
                build_local_skip_decision("critic_error_fallback"),
            )
            state = self._finalize_critic_pass(state, output_to_eval)
            await self._persist_moral_audit_log(state, output_to_eval)

        if state.get("moral_halt_required"):
            halt_summary = str(state.get("moral_halt_summary") or "Severe moral violation detected.").strip()
            suspended_task_id = await self._suspend_task_for_moral_halt(state)
            hitl_message = f"HITL REQUIRED: Moral governor halt. {halt_summary[:280]}"
            if suspended_task_id is not None:
                hitl_message += f" Task #{suspended_task_id} suspended."

            try:
                await self._notify_admin(hitl_message)
            except Exception as e:
                logger.warning("Failed to notify admin for moral halt: %s", e)

            raise RequiresHITLError(hitl_message)

        if state["iteration_count"] >= 3 and state["critic_feedback"] != "PASS":
            await self._spawn_debug_task(state)
            raise RequiresHITLError(
                "Critic rejected output 3 times. Guidance needed.\n"
                "Question: How should I proceed to satisfy the charter?"
            )

        return state

    async def _run_graph_loop_impl(self, state: Dict[str, Any], user_id: str, user_message: str) -> str:
        """Execute the supervisor -> workers -> critic loop and return sanitized output."""
        state = normalize_state(state)
        max_iterations = 3

        if self._has_ready_final_response(state):
            return self.cognitive_router.sanitize_response(state["final_response"])

        while state["iteration_count"] < max_iterations:
            blocked_response = await self._consume_blocked_result(state, user_id)
            if blocked_response is not None:
                return blocked_response

            self._apply_critic_retry_instructions(state)
            state, blocked_response = await self._run_graph_pass(state, user_id)
            if blocked_response is not None:
                return blocked_response

            if state["critic_feedback"] == "PASS":
                state["critic_instructions"] = ""
                break
            logger.warning("Critic rejected output on iteration %s", state["iteration_count"])
            self._reset_after_critic_rejection(state)

        self._ensure_final_response(state, max_iterations)
        self._finalizing_turn_failed = bool(state.get("_turn_failed", False))
        try:
            return await self._call_finalize_user_response(
                user_id,
                user_message,
                state["final_response"],
                state=state,
            )
        finally:
            self._finalizing_turn_failed = False

    async def _run_user_turn_locked_impl(
        self,
        user_id: str,
        user_message: str,
        user_prompt: Dict[str, Any],
    ) -> str:
        """Run one user turn while assuming the per-user lock is already held."""
        has_audio_prompt = bool(self._extract_audio_bytes(user_prompt))

        state = await self._load_state(
            user_id,
            user_message,
            user_prompt=user_prompt,
        )
        effective_user_id = user_id
        heartbeat_task_id = self._extract_heartbeat_origin_task_id_from_state(state)
        if heartbeat_task_id is not None:
            effective_user_id = "heartbeat"

        if state.get("final_response") and not state.get("current_plan"):
            return self.cognitive_router.sanitize_response(state["final_response"])

        if not has_audio_prompt:
            await self._apply_text_memory_hooks(effective_user_id, user_message)
        else:
            logger.info(
                "Audio prompt detected for %s; bypassing text-only fast-path memory hooks.",
                effective_user_id,
            )

        if not has_audio_prompt:
            reply = await self._try_goal_planning_response(state)
            if reply is not None:
                return await self._call_finalize_user_response(
                    effective_user_id,
                    user_message,
                    reply,
                    state=state,
                )

            reply = await self._try_fast_path_response(state)
            if reply is not None:
                return await self._call_finalize_user_response(
                    effective_user_id,
                    user_message,
                    reply,
                    state=state,
                )

        try:
            response = await self._run_graph_loop(state, effective_user_id, user_message)
            if bool(state.get("_resumed_from_hitl")) and heartbeat_task_id is not None:
                if not self._has_pending_heartbeat_hitl(heartbeat_task_id) and not self._has_pending_heartbeat_mfa(heartbeat_task_id):
                    await self._finalize_resumed_heartbeat_task(heartbeat_task_id, response)
            return response
        except RequiresHITLError as hitl_err:
            _hitl_emitter = get_current_emitter()
            if _hitl_emitter is not None:
                await _hitl_emitter.emit_immediate(ProgressEvent.hitl_raised())
            state["_hitl_question"] = str(hitl_err)
            state["_hitl_created_at"] = time.time()
            pending_user_id = self._pending_owner_for_heartbeat_origin(user_id, state)
            pending_state = self._strip_audio_bytes_for_persistence(state)
            self.pending_hitl_state[pending_user_id] = pending_state
            self._fire_and_forget(self.ledger_memory.save_hitl_state(pending_user_id, pending_state))
            if heartbeat_task_id is not None:
                await self._notify_admin(str(hitl_err))
            return str(hitl_err)
        except Exception as e:
            logger.error("Graph execution failed: %s", e, exc_info=True)
            state["_turn_failed"] = True
            return "An internal error occurred."

    async def _process_message_impl(
        self,
        user_message: Any,
        user_id: str,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """Main entry point for turn processing with progress emission and lock discipline."""
        user_prompt = self._coerce_user_prompt_payload(user_message)
        normalized_user_message = str(user_prompt.get("text") or "").strip()
        has_audio_prompt = bool(self._extract_audio_bytes(user_prompt))
        approval_outcome: Optional[Any] = None

        if not normalized_user_message and has_audio_prompt:
            audio_bytes = self._extract_audio_bytes(user_prompt)
            audio_mime_type = str(user_prompt.get("audio_mime_type") or "audio/ogg").strip() or "audio/ogg"
            normalized_user_message = f"[Voice note · {len(audio_bytes)} bytes · {audio_mime_type}]"
            user_prompt["text"] = normalized_user_message

        if not normalized_user_message and not has_audio_prompt:
            return "Error: Invalid message"

        _emitter_token: Optional[object] = None
        _emitter: Optional[ProgressEmitter] = None
        if progress_callback is not None and str(user_id or "").strip() != "heartbeat":
            _emitter = ProgressEmitter(progress_callback)
            _emitter_token = set_current_emitter(_emitter)

        try:
            if not hasattr(self, "_ready"):
                self._ready = asyncio.Event()
                self._ready.set()

            if not self._ready.is_set():
                try:
                    await asyncio.wait_for(self._ready.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    return "System is still initializing. Please try again in a moment."

            lock = await self._get_user_lock(user_id)
            async with lock:
                if str(user_id or "").strip() != "heartbeat":
                    async with self._energy_budget_lock:
                        replenishment = int(os.getenv("ENERGY_REPLENISH_PER_TURN", "5"))
                        cap = int(os.getenv("INITIAL_ENERGY_BUDGET", "100"))
                        self._energy_budget = min(self._energy_budget + replenishment, cap)

                reply = await self._try_resume_mfa(user_id, normalized_user_message)
                if reply is not None:
                    return reply

                reply = await self.synthesis_pipeline.try_resume_tool_approval(
                    user_id, normalized_user_message
                )
                if reply is not None:
                    approval_outcome = reply
                else:
                    return await self._run_user_turn_locked(
                        user_id=user_id,
                        user_message=normalized_user_message,
                        user_prompt=user_prompt,
                    )

            if approval_outcome is None:
                return "An internal error occurred."

            follow_up_input = approval_outcome.follow_up_input
            if follow_up_input is None:
                return approval_outcome.reply_text

            follow_up_response = await self.process_message(
                follow_up_input,
                user_id,
            )
            reply_text = str(approval_outcome.reply_text or "").strip()
            if not reply_text:
                return follow_up_response
            return f"{reply_text}\n\n{follow_up_response}"
        finally:
            if _emitter is not None:
                await _emitter.flush_pending()
            if _emitter_token is not None:
                reset_emitter(_emitter_token)

    async def _handle_blocked_result_impl(
        self,
        result: RouterResult,
        user_id: str,
        state: Dict[str, Any],
    ) -> str:
        """Convert a non-ok RouterResult into user-facing output and pending state."""
        if result.status == "mfa_required":
            pending_owner_id = self._pending_owner_for_heartbeat_origin(user_id, state)
            heartbeat_task_id = self._extract_heartbeat_origin_task_id_from_state(state)
            pending_entry = {
                "name": result.mfa_tool_name,
                "arguments": result.mfa_arguments,
                "user_id": str(state.get("user_id") or user_id),
                "_created_at": time.time(),
            }
            if heartbeat_task_id is not None:
                pending_entry["_heartbeat_origin_task_id"] = heartbeat_task_id

            self.pending_mfa[pending_owner_id] = pending_entry
            self._fire_and_forget(
                self.ledger_memory.save_mfa_state(pending_owner_id, result.mfa_tool_name, result.mfa_arguments)
            )
            _mfa_emitter = get_current_emitter()
            if _mfa_emitter is not None:
                await _mfa_emitter.emit_immediate(ProgressEvent.mfa_required())
            return "SECURITY LOCK: Provide the authorization passphrase to continue."

        if result.status == "hitl_required":
            pending_owner_id = self._pending_owner_for_heartbeat_origin(user_id, state)
            heartbeat_task_id = self._extract_heartbeat_origin_task_id_from_state(state)
            state["_hitl_question"] = result.hitl_message
            state["_hitl_created_at"] = time.time()
            if heartbeat_task_id is not None:
                state["_heartbeat_origin_task_id"] = heartbeat_task_id
            pending_state = self._strip_audio_bytes_for_persistence(state)
            self.pending_hitl_state[pending_owner_id] = pending_state
            self._fire_and_forget(self.ledger_memory.save_hitl_state(pending_owner_id, pending_state))
            if heartbeat_task_id is not None:
                await self._notify_admin(result.hitl_message)
            return result.hitl_message

        if result.status == "capability_gap":
            local_resolution = await self._try_resolve_capability_gap_locally(
                user_id,
                result,
                state,
            )
            if local_resolution is not None:
                return local_resolution

            in_progress = getattr(self, "_synthesis_in_progress", None)
            if not isinstance(in_progress, dict):
                in_progress = {}
                self._synthesis_in_progress = in_progress

            now = time.time()
            started_at = in_progress.get(user_id)
            if started_at is not None:
                elapsed_seconds = max(0, int(now - float(started_at)))
                if elapsed_seconds > _SYNTHESIS_LOCKOUT_TTL_SECONDS:
                    logger.warning(
                        "Synthesis run for %s timed out at registry level after %ds; allowing new run",
                        user_id,
                        elapsed_seconds,
                    )
                    in_progress.pop(user_id, None)
                else:
                    return (
                        "A capability synthesis is already in progress for your "
                        "request. Please wait for it to complete before sending "
                        "a new request that requires tool synthesis."
                    )

            in_progress[user_id] = now
            self._fire_and_forget(
                self.synthesis_pipeline.async_tool_synthesis(user_id, result, state)
            )
            _gap_emitter = get_current_emitter()
            if _gap_emitter is not None:
                await _gap_emitter.emit_immediate(ProgressEvent.capability_gap())
            return (
                f"I identified a capability gap: {result.gap_description}. "
                f"Requesting tool synthesis from System 2..."
            )

        return f"An unexpected router status was received: {result.status}"


# Compatibility alias for extracted static helpers that reference Orchestrator explicitly.
Orchestrator = _GraphNodesMixin
