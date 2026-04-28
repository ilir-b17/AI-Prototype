"""LLM routing mixin for Orchestrator."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

from src.core import cloud_redaction
from src.core.llm_router import RouterResult
from src.core.orchestrator_constants import (
    _SYSTEM_1_ERROR_PREFIX,
    _SYSTEM_2_ERROR_PREFIX,
    _SYSTEM_2_EMPTY_PREFIX,
)

logger = logging.getLogger(__name__)


class _LLMGatewayMixin:
    """LLM routing methods. Mixed into Orchestrator."""

    @staticmethod
    def _is_system_1_error(result: Optional[RouterResult]) -> bool:
        """Detect System 1 error payloads that should trigger System 2 fallback."""
        return bool(
            result
            and result.status == "ok"
            and isinstance(result.content, str)
            and result.content.startswith(_SYSTEM_1_ERROR_PREFIX)
        )

    @staticmethod
    def _is_system_2_error(result: Optional[RouterResult]) -> bool:
        """Detect placeholder System 2 error payloads that should trigger fallback."""
        return bool(
            result
            and result.status == "ok"
            and isinstance(result.content, str)
            and result.content.startswith((_SYSTEM_2_ERROR_PREFIX, _SYSTEM_2_EMPTY_PREFIX))
        )

    async def _append_cloud_payload_audit_if_supported(
        self,
        *,
        purpose: str,
        message_count_before: int,
        message_count_after: int,
        allow_sensitive_context: bool,
        payload_sha256: str,
    ) -> None:
        ledger = getattr(self, "ledger_memory", None)
        append_fn = getattr(ledger, "append_cloud_payload_audit", None)
        if not callable(append_fn):
            return
        try:
            await append_fn(
                purpose=purpose,
                message_count_before=message_count_before,
                message_count_after=message_count_after,
                allow_sensitive_context=allow_sensitive_context,
                payload_sha256=payload_sha256,
            )
        except Exception as e:
            logger.warning("Failed to append cloud payload audit entry for %s: %s", purpose, e)

    async def _route_to_system_2_redacted(
        self,
        messages: List[Dict[str, str]],
        *,
        allowed_tools: Optional[List[str]] = None,
        purpose: str = "system_2",
        allow_sensitive_context: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: str = "structured_output",
    ) -> RouterResult:
        minimized_messages = cloud_redaction.redact_messages_for_cloud(
            messages,
            allow_sensitive_context=allow_sensitive_context,
        )
        logger.info(
            "System 2 payload redacted for %s: %d -> %d message(s)",
            purpose,
            len(messages),
            len(minimized_messages),
        )
        await self._append_cloud_payload_audit_if_supported(
            purpose=purpose,
            message_count_before=len(messages),
            message_count_after=len(minimized_messages),
            allow_sensitive_context=allow_sensitive_context,
            payload_sha256=cloud_redaction.compute_payload_sha256(minimized_messages, allowed_tools),
        )
        route_kwargs: Dict[str, Any] = {"allowed_tools": allowed_tools}
        if response_schema is not None:
            route_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema_name,
                    "schema": response_schema,
                    "strict": True,
                },
            }
        return await self.cognitive_router.route_to_system_2(
            minimized_messages,
            **route_kwargs,
        )

    def _get_system_1_gate_metrics(self) -> Dict[str, Any]:
        getter = getattr(self.cognitive_router, "get_system_1_gate_metrics", None)
        if not callable(getter):
            return {}
        try:
            metrics = getter()
        except Exception as e:
            logger.debug(f"Unable to read System 1 gate metrics: {e}")
            return {}
        return metrics if isinstance(metrics, dict) else {}

    def _log_system_1_gate_pressure(
        self,
        context: str,
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
    ) -> None:
        if not after_metrics:
            return

        before_wait_events = int(before_metrics.get("wait_events", 0) or 0)
        after_wait_events = int(after_metrics.get("wait_events", 0) or 0)
        waiting_requests = int(after_metrics.get("waiting_requests", 0) or 0)
        new_wait_events = max(0, after_wait_events - before_wait_events)

        if new_wait_events == 0 and waiting_requests == 0:
            return

        logger.info(
            "System 1 gate pressure after %s: active=%s waiting=%s max=%s new_wait_events=%s total_wait=%.3fs avg_wait=%.3fs peak_waiting=%s",
            context,
            int(after_metrics.get("active_requests", 0) or 0),
            waiting_requests,
            int(after_metrics.get("max_concurrency", 0) or 0),
            new_wait_events,
            float(after_metrics.get("total_wait_seconds", 0.0) or 0.0),
            float(after_metrics.get("average_wait_seconds", 0.0) or 0.0),
            int(after_metrics.get("peak_waiting_requests", 0) or 0),
        )

    async def _route_to_system_1(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None,
        *,
        deadline_seconds: Optional[float] = None,
        context: str = "orchestrator",
        max_output_tokens: Optional[int] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: str = "structured_output",
    ) -> RouterResult:
        before_metrics = self._get_system_1_gate_metrics()
        route_kwargs: Dict[str, Any] = {"allowed_tools": allowed_tools}
        if max_output_tokens is not None:
            route_kwargs["max_output_tokens"] = max_output_tokens
        if response_schema is not None:
            route_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema_name,
                    "schema": response_schema,
                    "strict": True,
                },
            }
        route_call = self.cognitive_router.route_to_system_1(messages, **route_kwargs)
        try:
            if deadline_seconds is None:
                return await route_call
            return await asyncio.wait_for(route_call, timeout=deadline_seconds)
        finally:
            after_metrics = self._get_system_1_gate_metrics()
            self._log_system_1_gate_pressure(context, before_metrics, after_metrics)

    async def _route_memories_to_system_1(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None,
        max_output_tokens: Optional[int] = None,
    ):
        """Lightweight System 1 routing for reranker scoring calls.

        Uses a strict token cap (150 tokens) because reranker output
        is always short JSON. Passes deadline_seconds from reranker config.
        """
        _ = allowed_tools
        _reranker = getattr(self, "memory_reranker", None)
        if _reranker is None:
            return await self._route_to_system_1(
                messages,
                allowed_tools=[],
                deadline_seconds=20.0,
                context="memory_reranker",
                max_output_tokens=max_output_tokens or 150,
            )

        deadline = _reranker.config.timeout_seconds
        return await self._route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=deadline,
            context="memory_reranker",
            max_output_tokens=max_output_tokens or 150,
        )

    async def _route_energy_judge_messages(self, messages: List[Dict[str, str]]) -> RouterResult:
        return await self._route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=45.0,
            context="energy_judge",
        )

    async def _route_supervisor_request(self, messages: List[Dict]) -> Optional["RouterResult"]:
        """Try System 1, fall back to System 2.  Returns None on total failure."""
        router_result: Optional[RouterResult] = None
        try:
            logger.info("Routing Supervisor through System 1 (Local Model)")
            router_result = await self._route_to_system_1(
                messages,
                allowed_tools=[],
                deadline_seconds=150.0,
                context="supervisor",
            )
            if self._is_system_1_error(router_result):
                logger.warning("System 1 returned an error payload in supervisor; escalating to System 2.")
                router_result = None
        except asyncio.TimeoutError:
            logger.error("System 1 timed out in supervisor (150 s). Escalating to System 2.", exc_info=True)
        except Exception as s1_err:
            logger.error(f"System 1 raised an exception in supervisor: {s1_err!r}. Escalating.", exc_info=True)

        if router_result is not None:
            return router_result

        if not self.cognitive_router.get_system_2_available():
            return None

        try:
            logger.info("Escalating Supervisor to System 2")
            router_result = await asyncio.wait_for(
                self._route_to_system_2_redacted(
                    messages,
                    allowed_tools=[],
                    purpose="supervisor_fallback",
                    allow_sensitive_context=False,
                ),
                timeout=60.0,
            )
            return router_result
        except asyncio.TimeoutError:
            logger.error("System 2 timed out in supervisor (60 s).", exc_info=True)
        except Exception as s2_err:
            logger.error(f"System 2 raised an exception in supervisor: {s2_err!r}.", exc_info=True)
        return None

    async def _try_route_agent_system_1(
        self,
        messages: List[Dict[str, str]],
        agent_def: AgentDefinition,
    ) -> Optional[RouterResult]:
        try:
            router_result = await self._route_to_system_1(
                messages,
                allowed_tools=agent_def.allowed_tools,
                deadline_seconds=60.0,
                context=agent_def.name,
            )
            if self._is_system_1_error(router_result):
                return None
            return router_result
        except Exception as e:
            logger.error("%s failed on System 1: %s", agent_def.name, e)
            return None

    async def _try_route_agent_system_2(
        self,
        messages: List[Dict[str, str]],
        agent_def: AgentDefinition,
        *,
        task_packet: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[RouterResult]:
        if not self.cognitive_router.get_system_2_available():
            return None

        _ = (task_packet, state)
        try:
            router_result = await asyncio.wait_for(
                self._route_to_system_2_redacted(
                    messages,
                    allowed_tools=agent_def.allowed_tools,
                    purpose=f"{agent_def.name}_fallback",
                    allow_sensitive_context=False,
                ),
                timeout=60.0,
            )
            if self._is_system_2_error(router_result):
                return None
            return router_result
        except Exception as e:
            logger.error("%s failed on System 2: %s", agent_def.name, e)
            return None

    async def _route_agent_request(
        self,
        messages: List[Dict[str, str]],
        agent_def: AgentDefinition,
        task_packet: Optional[Dict[str, Any]] = None,
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[RouterResult], bool]:
        preferred_model = self._get_step_preferred_model(task_packet, agent_def)

        if preferred_model == "system_2":
            attempted_system_2 = self.cognitive_router.get_system_2_available()
            logger.info("Routing %s through System 2 (agent preference)", agent_def.name)
            router_result = await self._try_route_agent_system_2(
                messages,
                agent_def,
                task_packet=task_packet,
                state=state,
            )
            if router_result is not None:
                return router_result, attempted_system_2

            logger.info(
                "Falling back %s to System 1 after preferred System 2 path was unavailable or failed",
                agent_def.name,
            )
            return await self._try_route_agent_system_1(messages, agent_def), attempted_system_2

        logger.info("Routing %s through System 1", agent_def.name)
        router_result = await self._try_route_agent_system_1(messages, agent_def)
        if router_result is not None:
            return router_result, False

        attempted_system_2 = self.cognitive_router.get_system_2_available()
        if attempted_system_2:
            logger.info("Escalating %s to System 2 after System 1 failure", agent_def.name)
        return await self._try_route_agent_system_2(
            messages,
            agent_def,
            task_packet=task_packet,
            state=state,
        ), attempted_system_2

    async def _score_nocturnal_candidates(self, messages: List[Dict[str, str]]) -> RouterResult:
        if self.cognitive_router.get_system_2_available():
            return await asyncio.wait_for(
                self._route_to_system_2_redacted(
                    messages,
                    allowed_tools=[],
                    purpose="nocturnal_scoring",
                    allow_sensitive_context=False,
                ),
                timeout=45.0,
            )

        return await self._route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=90.0,
            context="nocturnal_scoring_local",
        )

    def _extract_router_content(self, result: RouterResult) -> Optional[str]:
        """
        Return the string content from an "ok" RouterResult, or None if it is
        a non-ok result (MFA / HITL / capability-gap).  Non-ok results should
        be stored in state[_BLOCKED_KEY] by the caller.
        """
        if result.status == "ok":
            return result.content
        return None

    async def _route_critic_request(self, messages: List[Dict[str, str]]) -> RouterResult:
        if self.cognitive_router.get_system_2_available():
            logger.info("Routing Critic through System 2 (Gemini)")
            return await asyncio.wait_for(
                self._route_to_system_2_redacted(
                    messages,
                    allowed_tools=[],
                    purpose="critic_review",
                    allow_sensitive_context=False,
                ),
                timeout=30.0,
            )
        return await self._route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=60.0,
            context="critic",
        )

    @staticmethod
    def _inject_in_turn_s2_blueprint(
        state: Dict[str, Any],
        escalation_problem: str,
        blueprint_text: str,
    ) -> None:
        worker_outputs = state.get("worker_outputs")
        if not isinstance(worker_outputs, dict):
            return

        existing = worker_outputs.get("_s2_blueprint", "")
        separator = "\n\n---\n\n" if existing else ""
        worker_outputs["_s2_blueprint"] = (
            f"{existing}{separator}"
            f"System 2 Blueprint ({escalation_problem[:80]}):\n"
            f"{blueprint_text}"
        )

    async def _handle_cognitive_escalation(
        self,
        state: Dict[str, Any],
        router_result: RouterResult,
    ) -> str:
        """
        Escalate a complex reasoning problem directly to System 2 inline.
        Extracts the solution and blueprint, saves blueprint to memory async,
        and returns the solution string.
        """
        escalation_problem = cloud_redaction.redact_text_for_cloud(
            router_result.escalation_problem,
            allow_sensitive_context=False,
        )
        escalation_context = cloud_redaction.redact_text_for_cloud(
            router_result.escalation_context,
            allow_sensitive_context=False,
        )

        prompt = (
            "You are System 2. System 1 has escalated a complex problem to you.\n\n"
            f"Problem Description: {escalation_problem}\n\n"
            f"Context Scratchpad: {escalation_context}\n\n"
            "Please provide a direct solution to the user's problem. "
            "Additionally, generate a brief 'Reasoning Blueprint' on how to solve this class of problem."
        )

        messages = [{"role": "user", "content": prompt}]
        response_schema = {
            "type": "object",
            "properties": {
                "solution": {"type": "string"},
                "blueprint": {"type": "string"},
            },
            "required": ["solution", "blueprint"],
            "additionalProperties": False,
        }
        sys2_result = await self._route_to_system_2_redacted(
            messages,
            purpose="cognitive_escalation",
            allow_sensitive_context=False,
            response_schema=response_schema,
            response_schema_name="cognitive_escalation_response",
        )

        if sys2_result.status != "ok":
            return "[System 2 - Error]: Escalation failed."

        content = sys2_result.content
        if not isinstance(content, dict):
            return "[System 2 - Error]: Escalation failed."

        solution_text = str(content.get("solution") or "").strip()
        blueprint_text = content.get("blueprint")
        if isinstance(blueprint_text, str):
            blueprint_text = blueprint_text.strip()
        else:
            blueprint_text = None

        if not solution_text:
            solution_text = "[System 2 - Error]: Escalation failed."

        if blueprint_text:
            # Inject into current turn so downstream nodes can use it.
            # Use a reserved key so it never collides with real agent names.
            self._inject_in_turn_s2_blueprint(
                state,
                router_result.escalation_problem,
                blueprint_text,
            )

            logger.info("System 2 blueprint extracted. Triggering non-blocking save to Archival Memory.")
            doc_text = f"System 2 Reasoning Blueprint for: {router_result.escalation_problem}\nBlueprint: {blueprint_text}"
            meta = {"type": "system_2_learned_pattern"}

            async def _save_blueprint_with_retry() -> None:
                for attempt in range(2):
                    try:
                        await self.vector_memory.add_memory_async(text=doc_text, metadata=meta)
                        logger.info("Blueprint saved to archival memory (attempt %d).", attempt + 1)
                        return
                    except Exception as exc:
                        logger.error("Blueprint save failed (attempt %d/2): %s", attempt + 1, exc)
                logger.critical(
                    "Blueprint permanently lost — ChromaDB insert failed twice: %.200s", doc_text
                )

            self._fire_and_forget(_save_blueprint_with_retry())

        return solution_text
