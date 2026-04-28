from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from src.core.orchestrator import Orchestrator

LANGGRAPH_AVAILABLE = False
logger = logging.getLogger(__name__)


class ReActWorkflow:
    """Continuous native tool-calling ReAct loop."""

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self.orchestrator = orchestrator
        self.max_iterations = 12
        self.llm_deadline_seconds = 150.0

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(state or {})
        user_input = str(state.get("user_input") or "").strip()
        if not user_input:
            state["final_response"] = "No valid response could be generated."
            state["current_plan"] = []
            state["critic_feedback"] = "PASS"
            return state

        messages = await self._build_messages(state, user_input)
        for _ in range(self.max_iterations):
            tool_schemas, allowed_tools = self._resolve_tooling()
            router_result = await self._call_llm(messages, tool_schemas, allowed_tools)
            if str(getattr(router_result, "status", "ok")) != "ok":
                state["_blocked_result"] = router_result
                return state

            payload = self._extract_payload(router_result)
            tool_call = self._extract_tool_call(payload)
            if tool_call is None:
                state["final_response"] = self._extract_text(payload) or "No valid response could be generated."
                state["current_plan"] = []
                state["critic_feedback"] = "PASS"
                return state

            tool_name, tool_args = tool_call
            tool_result = await self._execute_tool(tool_name, tool_args)
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args,
                            }
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result,
                }
            )

        state["final_response"] = "Unable to fulfill this request within the tool-call budget."
        state["current_plan"] = []
        state["critic_feedback"] = "PASS"
        return state

    async def _build_messages(self, state: Dict[str, Any], user_input: str) -> List[Dict[str, Any]]:
        core_context = ""
        archival_context = ""
        capabilities = ""
        sensory_context = ""

        core_memory = getattr(self.orchestrator, "core_memory", None)
        if core_memory is not None:
            get_context_string = getattr(core_memory, "get_context_string", None)
            if callable(get_context_string):
                try:
                    core_context = str(await get_context_string())
                except Exception as exc:
                    logger.debug("Unable to fetch core context: %s", exc)

        get_archival_context = getattr(self.orchestrator, "_get_archival_context", None)
        if callable(get_archival_context):
            try:
                archival_context = str(await get_archival_context(user_input))
            except Exception as exc:
                logger.debug("Unable to fetch archival context: %s", exc)

        get_capabilities = getattr(self.orchestrator, "_get_capabilities_string", None)
        if callable(get_capabilities):
            try:
                capabilities = str(get_capabilities())
            except Exception as exc:
                logger.debug("Unable to fetch capabilities: %s", exc)

        get_sensory_context = getattr(self.orchestrator, "_get_sensory_context", None)
        if callable(get_sensory_context):
            try:
                sensory_context = str(get_sensory_context())
            except Exception as exc:
                logger.debug("Unable to fetch sensory context: %s", exc)

        system_prompt = (
            "You are AIDEN in native ReAct mode.\n"
            "Use tool calls when they materially improve correctness.\n"
            "When a tool returns an error, adapt and continue without failing the turn.\n"
            "When you have enough information, return a direct final answer."
        )
        dynamic_context = (
            f"<dynamic_context>\n"
            f"<sensory>{sensory_context}</sensory>\n"
            f"<core_memory>{core_context}</core_memory>\n"
            f"<archival_memory>{archival_context}</archival_memory>\n"
            f"<capabilities>{capabilities}</capabilities>\n"
            f"</dynamic_context>"
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": dynamic_context},
        ]
        for turn in list(state.get("chat_history", []) or []):
            role = str(turn.get("role") or "").strip()
            content = str(turn.get("content") or "")
            if role and content:
                messages.append({"role": role, "content": content})

        build_user_prompt_message = getattr(self.orchestrator, "_build_user_prompt_message", None)
        if callable(build_user_prompt_message):
            messages.append(build_user_prompt_message(state))
        else:
            messages.append({"role": "user", "content": user_input})
        return messages

    def _resolve_tooling(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        registry = getattr(getattr(self.orchestrator, "cognitive_router", None), "registry", None)
        if registry is None:
            return [], []

        get_native_schemas = getattr(registry, "get_native_function_schemas", None)
        if callable(get_native_schemas):
            schemas = list(get_native_schemas())
        else:
            schemas = list(getattr(registry, "get_schemas", lambda: [])())

        allowed_tools: List[str] = []
        for schema in schemas:
            if isinstance(schema, dict) and schema.get("type") == "function":
                name = str((schema.get("function") or {}).get("name") or "").strip()
            else:
                name = str((schema or {}).get("name") or "").strip()
            if name:
                allowed_tools.append(name)
        return schemas, allowed_tools

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tool_schemas: List[Dict[str, Any]],
        allowed_tools: List[str],
    ) -> Any:
        route = getattr(self.orchestrator, "_route_to_system_1", None)
        kwargs: Dict[str, Any] = {
            "context": "react_loop",
            "deadline_seconds": self.llm_deadline_seconds,
        }
        if callable(route):
            params = inspect.signature(route).parameters
            if "allowed_tools" in params:
                kwargs["allowed_tools"] = allowed_tools
            if "tools" in params:
                kwargs["tools"] = tool_schemas
            return await route(messages, **kwargs)

        route = getattr(getattr(self.orchestrator, "cognitive_router", None), "route_to_system_1", None)
        if not callable(route):
            return {"status": "ok", "content": "No router available."}

        params = inspect.signature(route).parameters
        router_kwargs = {
            "context": "react_loop",
            "deadline_seconds": self.llm_deadline_seconds,
        }
        if "context" not in params:
            router_kwargs.pop("context", None)
        if "deadline_seconds" not in params:
            router_kwargs.pop("deadline_seconds", None)
        if "allowed_tools" in params:
            router_kwargs["allowed_tools"] = allowed_tools
        if "tools" in params:
            router_kwargs["tools"] = tool_schemas
        return await route(messages, **router_kwargs)

    @staticmethod
    def _extract_payload(router_result: Any) -> Any:
        if isinstance(router_result, dict):
            return router_result.get("content", router_result)
        return getattr(router_result, "content", router_result)

    @classmethod
    def _extract_tool_call(cls, payload: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        if isinstance(payload, dict):
            message_payload = payload.get("message")
            if message_payload is not None:
                nested = cls._extract_tool_call(message_payload)
                if nested is not None:
                    return nested

            tool_calls = payload.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                parsed = cls._parse_tool_call_dict(tool_calls[0])
                if parsed is not None:
                    return parsed

            for key in ("tool_call", "function_call"):
                candidate = payload.get(key)
                if isinstance(candidate, dict):
                    parsed = cls._parse_tool_call_dict(candidate)
                    if parsed is not None:
                        return parsed
                if isinstance(candidate, str):
                    return candidate, {}

            if "tool_name" in payload:
                tool_name = str(payload.get("tool_name") or "").strip()
                if tool_name:
                    return tool_name, cls._coerce_arguments(payload.get("arguments", {}))
        return None

    @classmethod
    def _parse_tool_call_dict(cls, blob: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        function_blob = blob.get("function") if isinstance(blob.get("function"), dict) else blob
        tool_name = str(
            function_blob.get("name")
            or function_blob.get("tool_name")
            or blob.get("name")
            or blob.get("tool_name")
            or ""
        ).strip()
        if not tool_name:
            return None
        arguments = function_blob.get("arguments", blob.get("arguments", {}))
        return tool_name, cls._coerce_arguments(arguments)

    @staticmethod
    def _coerce_arguments(raw_args: Any) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @classmethod
    def _extract_text(cls, payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            message_payload = payload.get("message")
            if message_payload is not None:
                message_text = cls._extract_text(message_payload)
                if message_text:
                    return message_text
            for key in ("content", "text", "answer", "final_response", "result"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(payload, ensure_ascii=False)
        return str(payload or "").strip()

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        registry = getattr(getattr(self.orchestrator, "cognitive_router", None), "registry", None)
        execute = getattr(registry, "execute", None)
        if not callable(execute):
            return f"Error: Tool '{tool_name}' could not execute because the skill registry is unavailable."
        try:
            result = await execute(tool_name, tool_args)
        except Exception as exc:
            return f"Error: Tool '{tool_name}' failed — {exc}"
        return str(result)


def build_orchestrator_graph(orchestrator: "Orchestrator") -> Optional[Any]:
    """Build the flattened ReAct loop workflow."""
    return ReActWorkflow(orchestrator)
