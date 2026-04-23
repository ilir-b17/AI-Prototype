"""
Orchestrator Module - Implements State Graph Architecture (Sprint 6)

State Graph with Energy Budget, Proactive Heartbeat, and Charter enforcement.
Passes a State Dictionary between specialized nodes until task is complete,
exhausted (energy = 0), or blocked (HITL required).
"""

import os
import platform
import logging
import json
import asyncio
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from src.memory.vector_db import VectorMemory
from src.memory.ledger_db import LedgerMemory
from src.memory.core_memory import CoreMemory
from src.core.agent_definition import AgentDefinition
from src.core.agent_registry import AgentRegistry
from src.core.llm_router import CognitiveRouter, RouterResult, RequiresHITLError
from src.core.prompt_config import load_prompt_config, build_supervisor_prompt
from src.core.runtime_context import set_runtime_context
from src.core.state_model import AgentState, normalize_state
from src.core.workflow_graph import build_orchestrator_graph
from src.core.security import verify_mfa_challenge

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

logger = logging.getLogger(__name__)

# Energy costs per operation
ENERGY_COST_SUPERVISOR = 10
ENERGY_COST_WORKER = 15
ENERGY_COST_CRITIC = 10
ENERGY_COST_TOOL = 5
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))  # default 30 minutes
MEMORY_SAVE_THRESHOLD = int(os.getenv("MEMORY_SAVE_THRESHOLD", "120"))
MEMORY_CONSOLIDATION_INTERVAL = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "21600"))  # 6 hours
# How long an unanswered MFA/HITL/tool-approval entry stays alive before auto-expiry
_PENDING_STATE_TTL_SECONDS = int(os.getenv("PENDING_STATE_TTL_SECONDS", "86400"))  # 24 hours
_SYSTEM_1_ERROR_PREFIX = "[System 1 - Error]"
_ROUTING_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "could", "do", "for", "from",
    "hello", "hey", "hi", "i", "in", "is", "it", "me", "my", "now", "of",
    "on", "or", "please", "right", "tell", "the", "this", "to", "today", "what",
    "would", "you", "your",
}
_NON_UTILITY_TOOL_PREFIXES = ("update_", "request_", "spawn_", "run_", "execute_")
_NON_UTILITY_TOOL_NAMES = {
    "ask_admin_for_guidance",
    "consolidate_memory",
    "manage_file_system",
    "query_highest_priority_task",
}
_DIRECT_ROUTE_MAX_COMPLEXITY = 0
_SINGLE_TOOL_MAX_COMPLEXITY = 2
_SINGLE_TOOL_MIN_SCORE = 1.5
_SINGLE_TOOL_MIN_MARGIN = 0.75
_RECENT_CHAT_HISTORY_LIMIT = 12
_CRITIC_SHORT_OUTPUT_THRESHOLD = 220
_SYSTEM_2_ERROR_PREFIX = "[System 2 - Error]"
_SYSTEM_2_EMPTY_PREFIX = "[System 2 - No Response]"
_ERROR_RESPONSE_PREFIXES = (
    "Supervisor encountered an error",
    "Both local and cloud reasoning failed",
    "I was unable to process this request locally",
    "Planning timed out",
    "Unable to fulfill this request",
    "No valid response could be generated",
    "An internal error occurred",
)

# Key used in the state dict to signal a blocked (non-ok) router result
_BLOCKED_KEY = "_blocked_result"


class Orchestrator:
    """
    Central orchestration engine using a State Graph architecture.

    State Dictionary schema:
        user_id, user_input, current_plan, worker_outputs,
        final_response, iteration_count, admin_guidance, energy_remaining

    Lifecycle:
        1. ``__init__`` — synchronous; creates memory objects (no DB I/O yet).
        2. ``await async_init()`` — must be called once from an async context
           (the Telegram bot's ``post_init`` hook) to open the aiosqlite
           connection and seed initial goals.
    """

    def __init__(
        self,
        vector_db_path: str = "data/chroma_storage",
        ledger_db_path: str = "data/ledger.db",
        core_memory_path: str = "data/core_memory.json",
        gemini_model: str = "gemini-2.0-flash",
        local_model: str = "gemma4:e4b"
    ) -> None:
        logger.info("Initializing Orchestrator (State Graph)")

        try:
            self.vector_memory = VectorMemory(persist_dir=vector_db_path)
            self.ledger_memory = LedgerMemory(db_path=ledger_db_path)
            self.core_memory = CoreMemory(memory_file_path=core_memory_path)
            self.cognitive_router = CognitiveRouter(model_name=gemini_model, local_model=local_model)
            self.prompt_config = load_prompt_config()
            self.agent_registry = AgentRegistry()

            self.charter_text = self._load_charter()
            self.pending_mfa: Dict[str, dict] = {}
            self.pending_hitl_state: Dict[str, dict] = {}
            self.pending_tool_approval: Dict[str, dict] = {}
            self.outbound_queue: Optional[asyncio.Queue] = None
            self.sensory_state: Dict[str, str] = {}
            # Background task registry — holds strong references to prevent GC (ISSUE-002)
            self._background_tasks: set = set()
            # Per-user async locks to serialise concurrent messages (ISSUE-012)
            self._user_locks: Dict[str, asyncio.Lock] = {}
            self._user_locks_lock: asyncio.Lock = asyncio.Lock()
            self._compiled_graph = None
            self._refresh_sensory_state()

            logger.info("Orchestrator __init__ complete — call async_init() to finish setup")
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
            raise

    async def async_init(self) -> None:
        """
        Async second-phase initialisation.  Must be awaited before any messages
        are processed.  Opens the aiosqlite ledger, seeds initial goals, and
        reloads previously approved dynamic tools.
        """
        await self.ledger_memory.initialize()

        # Register Groq cooldown persistence callback and reload any saved cooldown
        async def _save_groq_cooldown(expiry: float) -> None:
            await self.ledger_memory.set_system_state("groq_cooldown_until", str(expiry))

        self.cognitive_router._persist_cooldown_cb = _save_groq_cooldown
        try:
            saved_cooldown = await self.ledger_memory.get_system_state("groq_cooldown_until")
            if saved_cooldown:
                expiry = float(saved_cooldown)
                if expiry > time.time():
                    self.cognitive_router._system2_cooldown_until = expiry
                    logger.info(
                        "Restored Groq rate-limit cooldown from DB: %.0fs remaining.",
                        expiry - time.time(),
                    )
        except Exception as _ce:
            logger.warning("Failed to restore Groq cooldown from DB: %s", _ce)

        await self.ledger_memory.seed_initial_goals()
        await self._load_approved_tools()
        await self._load_pending_approvals()
        await self._load_pending_mfa()
        await self._load_pending_hitl()
        self._compiled_graph = build_orchestrator_graph(self)
        set_runtime_context(self.ledger_memory, self.core_memory)
        # Record the host OS now that we're in an async context
        await self.core_memory.update("host_os", platform.system())
        # Startup pruning: remove stale chat_history rows
        try:
            deleted = await self.ledger_memory.prune_old_chat_history(
                days=int(os.getenv("CHAT_HISTORY_MAX_DAYS", "90")),
                keep_minimum=int(os.getenv("CHAT_HISTORY_MIN_ROWS", "20")),
            )
            if deleted:
                logger.info("Startup pruning: removed %d stale chat_history rows.", deleted)
        except Exception as _prune_err:
            logger.warning("Startup chat history pruning failed: %s", _prune_err)
        if self.charter_text == self._CHARTER_FALLBACK:
            logger.warning(
                "SECURITY: Running without a full charter. Charter enforcement will be minimal."
            )
        logger.info("Orchestrator async_init complete")

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

    @staticmethod
    def _routing_keywords(text: str) -> set:
        return {
            token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 2 and token not in _ROUTING_STOPWORDS
        }

    def _estimate_request_complexity(self, text: str) -> int:
        lowered = (text or "").lower()
        tokens = re.findall(r"[a-z0-9]+", lowered)
        score = 0

        if len(tokens) > 12:
            score += 1
        if len(tokens) > 24:
            score += 1
        if len(re.findall(r"[?.!]", text or "")) > 1:
            score += 1
        if any(marker in (text or "") for marker in ("\n", "```", ":", ";")):
            score += 1
        if re.search(r"\b(and|then|after|before|also|plus|compare|step|steps|plan)\b", lowered):
            score += 1
        if re.search(
            r"\b(build|create|implement|debug|fix|refactor|analyze|analyse|review|modify|update|delete|write|open|extract|read|summari[sz]e|code)\b",
            lowered,
        ):
            score += 1

        return score

    @staticmethod
    def _is_utility_tool_schema(schema: Dict[str, Any]) -> bool:
        name = schema.get("name", "")
        if not name or name in _NON_UTILITY_TOOL_NAMES or name.startswith(_NON_UTILITY_TOOL_PREFIXES):
            return False
        params = schema.get("parameters", {}) or {}
        required = list(params.get("required", []) or [])
        if len(required) > 1:
            return False
        if not required:
            return True
        return required[0] in {"query", "url"}

    @staticmethod
    def _tool_schema_keywords(schema: Dict[str, Any]) -> Dict[str, set]:
        params = (schema.get("parameters") or {}).get("properties", {}) or {}
        name_tokens = Orchestrator._routing_keywords(schema.get("name", "").replace("_", " "))
        desc_tokens = Orchestrator._routing_keywords(schema.get("description", ""))
        param_tokens = set()
        for param_name, param_schema in params.items():
            param_tokens |= Orchestrator._routing_keywords(param_name.replace("_", " "))
            param_tokens |= Orchestrator._routing_keywords(param_schema.get("description", ""))
        return {
            "name": name_tokens,
            "description": desc_tokens,
            "parameters": param_tokens,
        }

    def _prepare_utility_tool_arguments(
        self,
        schema: Dict[str, Any],
        user_message: str,
    ) -> Optional[Dict[str, Any]]:
        params = schema.get("parameters", {}) or {}
        properties = params.get("properties", {}) or {}
        required = list(params.get("required", []) or [])

        if not required:
            return {}

        if len(required) != 1:
            return None

        field_name = required[0]
        if field_name == "query":
            cleaned = re.sub(
                r"^(can|could|would|please|hey|hi|hello|nice|ok|okay|what about)\b[\s,]*",
                "",
                user_message.strip(),
                flags=re.IGNORECASE,
            ).strip(" ?!.,")
            if not cleaned:
                cleaned = user_message.strip()
            args = {"query": cleaned}
            if "max_results" in properties:
                args["max_results"] = 3
            return args

        if field_name == "url":
            url_match = re.search(r"https?://\S+", user_message)
            if not url_match:
                return None
            return {"url": url_match.group(0)}

        return None

    def _score_tool_for_request(self, user_message: str, schema: Dict[str, Any]) -> float:
        query_keywords = self._routing_keywords(user_message)
        if not query_keywords:
            return 0.0
        schema_keywords = self._tool_schema_keywords(schema)
        score = 0.0
        score += 2.5 * len(query_keywords & schema_keywords["name"])
        score += 1.5 * len(query_keywords & schema_keywords["description"])
        score += 0.75 * len(query_keywords & schema_keywords["parameters"])
        if not (schema.get("parameters", {}) or {}).get("required"):
            score += 0.25
        return score

    def _assess_request_route(self, user_message: str) -> Dict[str, Any]:
        complexity = self._estimate_request_complexity(user_message)
        candidates = []

        for schema in self.cognitive_router.registry.get_schemas():
            if not self._is_utility_tool_schema(schema):
                continue
            arguments = self._prepare_utility_tool_arguments(schema, user_message)
            if arguments is None:
                continue
            score = self._score_tool_for_request(user_message, schema)
            if score <= 0:
                continue
            candidates.append({
                "tool_name": schema.get("name", ""),
                "arguments": arguments,
                "score": score,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        top = candidates[0] if candidates else None
        next_score = candidates[1]["score"] if len(candidates) > 1 else 0.0

        if (
            top
            and complexity <= _SINGLE_TOOL_MAX_COMPLEXITY
            and top["score"] >= _SINGLE_TOOL_MIN_SCORE
            and (top["score"] - next_score) >= _SINGLE_TOOL_MIN_MARGIN
        ):
            return {
                "mode": "single_tool",
                "complexity": complexity,
                "tool_name": top["tool_name"],
                "arguments": top["arguments"],
            }

        if complexity <= _DIRECT_ROUTE_MAX_COMPLEXITY and (not top or top["score"] < _SINGLE_TOOL_MIN_SCORE):
            return {"mode": "direct", "complexity": complexity}

        return {"mode": "graph", "complexity": complexity}

    @staticmethod
    def _coerce_fast_path_state(
        state_or_user_message: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(state_or_user_message, dict):
            merged = dict(state_or_user_message)
        else:
            merged = dict(state or {})
            merged["user_input"] = str(state_or_user_message or "")
        merged.setdefault("user_input", "")
        merged.setdefault("chat_history", [])
        return merged

    @staticmethod
    def _normalize_display_name(raw_name: str) -> str:
        cleaned = re.sub(r"\s+", " ", raw_name).strip(" ,.!?")
        return " ".join(part.capitalize() for part in cleaned.split())

    @staticmethod
    def _extract_user_profile_updates(user_message: str) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        text = (user_message or "").strip()
        lowered = text.lower()

        name_match = re.search(
            r"\bmy name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if name_match:
            raw_name = Orchestrator._normalize_display_name(name_match.group(1))
            if raw_name:
                updates["name"] = raw_name

        age_match = re.search(r"\b(\d{1,3})\s*years?\s*old\b", lowered)
        if age_match:
            updates["age"] = int(age_match.group(1))

        if re.search(r"\bi am(?: also)? your developer\b", lowered):
            updates["relationship"] = "developer"

        return updates

    @staticmethod
    def _extract_assistant_identity_update(user_message: str) -> Optional[str]:
        text = (user_message or "").strip()
        if not text or "?" in text:
            return None

        patterns = (
            r"\b(?:just for your information,\s*)?your name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|$)",
            r"\bi am informing you(?: that)? your name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return Orchestrator._normalize_display_name(match.group(1))
        return None

    async def _remember_user_profile(self, user_id: str, user_message: str) -> bool:
        if not user_id or not hasattr(self, "core_memory"):
            return False

        updates = self._extract_user_profile_updates(user_message)
        if not updates:
            return False

        core_state = await self.core_memory.get_all()
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
        await self.core_memory.update("user_profiles", profiles)
        return True

    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        if not user_id or not hasattr(self, "core_memory"):
            return {}
        core_state = await self.core_memory.get_all()
        profiles = dict(core_state.get("user_profiles", {}) or {})
        return dict(profiles.get(user_id, {}) or {})

    async def _remember_assistant_identity(self, user_message: str) -> Optional[str]:
        if not hasattr(self, "core_memory"):
            return None

        assistant_name = self._extract_assistant_identity_update(user_message)
        if not assistant_name:
            return None

        core_state = await self.core_memory.get_all()
        assistant_profile = dict(core_state.get("assistant_profile", {}) or {})
        if assistant_profile.get("name") == assistant_name:
            return assistant_name

        assistant_profile["name"] = assistant_name
        await self.core_memory.update("assistant_profile", assistant_profile)
        return assistant_name

    async def _get_assistant_name(self) -> str:
        if not hasattr(self, "core_memory"):
            return "AIDEN"

        core_state = await self.core_memory.get_all()
        assistant_profile = dict(core_state.get("assistant_profile", {}) or {})
        assistant_name = str(assistant_profile.get("name", "")).strip()
        return assistant_name or "AIDEN"

    def _get_registry_skill_names(self) -> List[str]:
        registry = getattr(self.cognitive_router, "registry", None)
        if registry is None:
            return []
        getter = getattr(registry, "get_skill_names", None)
        if callable(getter):
            return list(getter())
        schema_getter = getattr(registry, "get_schemas", None)
        if callable(schema_getter):
            return sorted(
                schema.get("name", "")
                for schema in schema_getter()
                if schema.get("name")
            )
        return []

    @staticmethod
    def _is_capability_question(user_message: str) -> bool:
        lowered = (user_message or "").lower()
        capability_markers = ("tool", "tools", "skill", "skills", "capabil", "internet", "browse", "web", "repository")
        return any(marker in lowered for marker in capability_markers) and any(
            phrase in lowered
            for phrase in (
                "can you",
                "do you have",
                "are you sure",
                "have access",
                "access the internet",
                "browse the internet",
            )
        )

    def _build_capability_response(self, user_message: str) -> Optional[str]:
        if not self._is_capability_question(user_message):
            return None

        skill_names = self._get_registry_skill_names()
        if not skill_names:
            return "I do not currently have any registered tools available."

        lowered = (user_message or "").lower()
        if "repository" in lowered:
            visible_tools = ", ".join(skill_names[:8])
            extra = len(skill_names) - min(len(skill_names), 8)
            if extra > 0:
                visible_tools = f"{visible_tools}, and {extra} more"
            return f"Yes. I can access my registered tool repository. Available tools include: {visible_tools}."

        if any(term in lowered for term in ("internet", "browse", "web")):
            if "web_search" in skill_names:
                return (
                    "Yes. I can access current internet information by using my web_search tool when a request needs live web data. "
                    "I do not browse continuously on my own, but I can run targeted searches for you."
                )
            return "Not right now. I do not currently have a web-search tool registered."

        visible_tools = ", ".join(skill_names[:8])
        extra = len(skill_names) - min(len(skill_names), 8)
        if extra > 0:
            visible_tools = f"{visible_tools}, and {extra} more"
        return f"Yes. I currently have tool access. Available tools include: {visible_tools}."

    @staticmethod
    def _is_summary_request(user_message: str) -> bool:
        lowered = (user_message or "").lower()
        summary_markers = ("summarize", "summarise", "summerize", "summary", "recap", "sum up")
        context_markers = ("conversation", "chat", "so far", "we had", "what we discussed")
        return any(marker in lowered for marker in summary_markers) and any(
            marker in lowered for marker in context_markers
        )

    @staticmethod
    def _summarize_chat_history(chat_history: List[Dict[str, str]]) -> Optional[str]:
        user_turns = []
        for turn in chat_history:
            if turn.get("role") != "user":
                continue
            content = re.sub(r"\s+", " ", (turn.get("content") or "").strip())
            if not content:
                continue
            if len(content) > 90:
                content = f"{content[:87]}..."
            user_turns.append(content)

        if not user_turns:
            return None

        recent = user_turns[-6:]
        if len(recent) == 1:
            return f"So far you've asked about: {recent[0]}."
        return "So far we've discussed: " + "; ".join(recent[:-1]) + f"; and {recent[-1]}."

    @staticmethod
    def _is_last_reply_question(user_message: str) -> bool:
        lowered = (user_message or "").lower()
        return any(
            phrase in lowered
            for phrase in (
                "what did you just share",
                "what did you just shared",
                "what did you share",
                "what did you just say",
                "what did you just send",
            )
        )

    @staticmethod
    def _recall_last_assistant_message(chat_history: List[Dict[str, str]]) -> Optional[str]:
        for turn in reversed(chat_history):
            if turn.get("role") != "assistant":
                continue
            content = re.sub(r"\s+", " ", (turn.get("content") or "").strip())
            if not content:
                continue
            if len(content) > 280:
                content = f"{content[:277]}..."
            return f"The last thing I shared was: {content}"
        return None

    @staticmethod
    def _recent_user_turn_mentions_name(chat_history: List[Dict[str, str]]) -> bool:
        user_turns = [turn for turn in chat_history if turn.get("role") == "user"]
        for turn in reversed(user_turns[-3:]):
            if "name" in (turn.get("content") or "").lower():
                return True
        return False

    @staticmethod
    def _is_assistant_name_question(lowered: str, chat_history: List[Dict[str, str]]) -> bool:
        if re.search(r"\b(what is|what's|who is|who are) your name\b", lowered):
            return True
        if re.search(r"\bwhat about yours\b", lowered):
            return True
        if re.search(r"\band yours\b", lowered):
            return Orchestrator._recent_user_turn_mentions_name(chat_history)
        return lowered.strip(" ?!.") == "yours" and Orchestrator._recent_user_turn_mentions_name(chat_history)

    @staticmethod
    def _is_profile_memory_request(lowered: str) -> bool:
        return (
            "store" in lowered and "memory" in lowered and "name" in lowered
        ) or (
            "remember" in lowered and "later" in lowered and "name" in lowered
        )

    @staticmethod
    def _is_user_profile_lookup_question(lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "do you know my name",
                "can you tell my name",
                "tell my name",
                "tell me my name",
                "what is my name",
                "what's my name",
                "who am i",
                "do you know my age",
                "how old am i",
            )
        )

    @staticmethod
    def _format_user_profile_response(lowered: str, profile: Dict[str, Any]) -> str:
        name = profile.get("name")
        age = profile.get("age")

        if "age" in lowered or "old am i" in lowered:
            if age:
                return f"You told me that you are {age} years old."
            return "You have not told me your age yet."

        if name and age:
            return f"Yes. Your name is {name}, and you told me that you are {age} years old."
        if name:
            return f"Yes. Your name is {name}."
        return "Not yet. You have not told me your name."

    async def _build_profile_response(self, state: Dict[str, Any]) -> Optional[str]:
        user_message = (state.get("user_input") or "").strip()
        lowered = user_message.lower()
        chat_history = list(state.get("chat_history", []) or [])

        assistant_name = self._extract_assistant_identity_update(user_message)
        if assistant_name:
            return f"Understood. I will use {assistant_name} as my name."

        if self._is_assistant_name_question(lowered, chat_history):
            return f"My name is {await self._get_assistant_name()}."

        if self._is_profile_memory_request(lowered):
            return (
                "Yes. If you share details like your name, I can store them in memory and use them in later replies."
            )

        if not self._is_user_profile_lookup_question(lowered):
            return None

        profile = await self._get_user_profile(state.get("user_id", ""))
        return self._format_user_profile_response(lowered, profile)

    async def _try_meta_fast_path_response(self, state: Dict[str, Any]) -> Optional[str]:
        user_message = state.get("user_input", "")
        chat_history = list(state.get("chat_history", []) or [])

        capability_response = self._build_capability_response(user_message)
        if capability_response:
            return capability_response

        profile_response = await self._build_profile_response(state)
        if profile_response:
            return profile_response

        if self._is_last_reply_question(user_message):
            return self._recall_last_assistant_message(chat_history)

        if self._is_summary_request(user_message):
            return self._summarize_chat_history(chat_history)

        return None

    async def _try_fast_path_response(
        self,
        state_or_user_message: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Handle low-complexity requests before full orchestration.

        The routing decision is generic: score request complexity and match the
        query against tool-schema metadata. If a single safe utility tool is a
        clear fit, execute it and ask System 1 to summarize the result without
        any further tool calls. Otherwise, either answer directly (tool-free) or
        fall back to the full supervisor workflow.
        """
        state = self._coerce_fast_path_state(state_or_user_message, state)
        user_message = state.get("user_input", "")

        meta_response = await self._try_meta_fast_path_response(state)
        if meta_response:
            return meta_response

        assessment = self._assess_request_route(user_message)
        core_mem_str = ""
        if hasattr(self, "core_memory"):
            core_mem_str = await self.core_memory.get_context_string()
        capabilities_str = self._get_capabilities_string()
        history_msgs = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in state.get("chat_history", [])
            if turn.get("role") in {"user", "assistant"} and turn.get("content")
        ]

        if assessment["mode"] == "direct":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are AIDEN. This request is low complexity. Reply directly in 1-2 concise sentences. "
                        "Use the recent chat history and core memory when they are relevant. "
                        "Answer capability questions based on the provided capabilities instead of claiming you have none. "
                        "Do not call tools and do not mention internal routing.\n\n"
                        f"{core_mem_str}\n\n{capabilities_str}"
                    ),
                },
                *history_msgs,
                {"role": "user", "content": user_message},
            ]
            result = await self._route_to_system_1(
                messages,
                allowed_tools=[],
                context="fast_path_direct",
            )
            if (
                result.status == "ok"
                and result.content
                and not result.content.startswith(_SYSTEM_1_ERROR_PREFIX)
            ):
                return result.content.strip()
            return None

        if assessment["mode"] != "single_tool":
            return None

        tool_name = assessment["tool_name"]
        tool_result = await self.cognitive_router._execute_tool(tool_name, assessment["arguments"])
        if tool_result.status != "ok":
            return None
        if tool_result.content.strip().lower().startswith("error:"):
            return tool_result.content.strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are AIDEN. One trusted tool has already been executed. "
                    "Using only the tool output provided, answer the user's request clearly and concisely. "
                    "Do not call tools. If the tool output is inconclusive, say so explicitly.\n\n"
                    f"{core_mem_str}\n\n{capabilities_str}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request: {user_message}\n\n"
                    f"Tool used: {tool_name}\n"
                    f"Tool output:\n{tool_result.content}"
                ),
            },
        ]
        result = await self._route_to_system_1(
            messages,
            allowed_tools=[],
            context="fast_path_single_tool",
        )
        if (
            result.status == "ok"
            and result.content
            and not result.content.startswith(_SYSTEM_1_ERROR_PREFIX)
        ):
            return result.content.strip()
        return tool_result.content.strip()

    _CHARTER_FALLBACK = "Core Directive: Do no harm."

    def _load_charter(self, filepath: str = "charter.md") -> str:
        resolved = os.getenv("CHARTER_PATH", filepath)
        try:
            if os.path.exists(resolved):
                text = open(resolved, "r", encoding="utf-8").read().strip()
                if text:
                    return text
                logger.warning("Charter file at '%s' is empty — using minimal fallback.", resolved)
            else:
                logger.warning(
                    "Charter file not found at '%s'. Agent will operate with minimal directives. "
                    "Set CHARTER_PATH or place charter.md in the working directory.",
                    resolved,
                )
        except Exception as e:
            logger.error("Failed to read charter at '%s': %s", resolved, e)
        return self._CHARTER_FALLBACK

    def _get_capabilities_string(self) -> str:
        """Compact tool names list derived live from the SkillRegistry.

        Full schemas are already sent to Ollama as structured JSON — this
        one-liner is enough for conversational self-reference without
        duplicating ~400 tokens of schema text per call.
        """
        names = ", ".join(s["name"] for s in self.cognitive_router.registry.get_schemas())
        return f"Available tools (use them — do not claim you lack them): {names}"

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
    ) -> RouterResult:
        before_metrics = self._get_system_1_gate_metrics()
        route_call = self.cognitive_router.route_to_system_1(messages, allowed_tools=allowed_tools)
        try:
            if deadline_seconds is None:
                return await route_call
            return await asyncio.wait_for(route_call, timeout=deadline_seconds)
        finally:
            after_metrics = self._get_system_1_gate_metrics()
            self._log_system_1_gate_pressure(context, before_metrics, after_metrics)

    async def _get_archival_context(self, query: str) -> str:
        """Retrieve top archival memory snippets relevant to the user query."""
        if not query:
            return ""
        _max_chunk = int(os.getenv("MAX_ARCHIVAL_CHUNK_CHARS", "2000"))
        _max_total = int(os.getenv("MAX_ARCHIVAL_TOTAL_CHARS", "6000"))
        try:
            results = await self.vector_memory.query_memory_async(query, n_results=3)
            if not results:
                return ""
            lines = ["<Archival_Memory>"]
            total = 0
            for item in results:
                snippet = (item.get("document", "") or "")[:_max_chunk]
                if not snippet:
                    continue
                if total + len(snippet) > _max_total:
                    break
                lines.append(f"  <Memory>{snippet}</Memory>")
                total += len(snippet)
            lines.append("</Archival_Memory>")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Archival memory lookup failed: {e}")
            return ""

    def _new_state(self, user_id: str, user_message: str) -> Dict[str, Any]:
        return AgentState.new(user_id=user_id, user_input=user_message).to_dict()

    def _deduct_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        """Deduct energy and raise HITL if exhausted."""
        state["energy_remaining"] = state.get("energy_remaining", 100) - amount
        logger.debug(f"Energy -{amount} ({reason}). Remaining: {state['energy_remaining']}")
        if state["energy_remaining"] <= 0:
            raise RequiresHITLError(
                f"Energy Budget Exhausted: The system consumed all cognitive energy on this task.\n"
                f"Question: How should I prioritize the remaining work for: '{state.get('user_input', '')}'"
            )
        return state

    def _refund_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        """Refund energy deducted by a failed or timed-out operation."""
        state["energy_remaining"] = state.get("energy_remaining", 0) + amount
        logger.debug("Energy +%d refunded (%s). Remaining: %d", amount, reason, state["energy_remaining"])
        return state

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
    def _merge_plan_dependencies(existing: List[str], incoming: List[str]) -> List[str]:
        merged = list(existing or [])
        for dependency in incoming or []:
            if dependency not in merged:
                merged.append(dependency)
        return merged

    @staticmethod
    def _normalize_model_preference(raw_value: Any) -> str:
        preferred_model = str(raw_value or "").strip().lower()
        return preferred_model if preferred_model in {"system_1", "system_2"} else ""

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

    def _get_requested_plan_steps(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._normalize_current_plan(list(state.get("current_plan", []) or []))

    @staticmethod
    def _get_real_worker_outputs(state: Dict[str, Any]) -> Dict[str, str]:
        return {
            name: output
            for name, output in dict(state.get("worker_outputs", {}) or {}).items()
            if name != "supervisor_context"
        }

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
            return real_outputs[last_worker]

        if len(requested_outputs) == 1 or not self._should_combine_requested_outputs(requested_steps):
            return requested_outputs[-1][1]

        return "\n\n".join(
            f"{agent_name}:\n{output}"
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

    async def _route_supervisor_request(self, messages: List[Dict]) -> Optional["RouterResult"]:
        """Try System 1, fall back to System 2.  Returns None on total failure."""
        router_result: Optional[RouterResult] = None
        try:
            logger.info("Routing Supervisor through System 1 (Local Model)")
            router_result = await self._route_to_system_1(
                messages,
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
                self.cognitive_router.route_to_system_2(messages),
                timeout=60.0,
            )
            return router_result
        except asyncio.TimeoutError:
            logger.error("System 2 timed out in supervisor (60 s).", exc_info=True)
        except Exception as s2_err:
            logger.error(f"System 2 raised an exception in supervisor: {s2_err!r}.", exc_info=True)
        return None

    def _parse_supervisor_response(self, response: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the WORKERS tag from *response* and update *state* in-place."""
        lines = response.strip().split("\n")
        workers_payload = None
        for line in reversed(lines[-3:]):
            if "WORKERS:" in line:
                workers_payload = line.split("WORKERS:", 1)[1].strip()
                break

        if not workers_payload:
            state["current_plan"] = []
            state["final_response"] = response.strip()
            return state

        try:
            plan = json.loads(workers_payload)
            answer = "\n".join(ln for ln in lines if not re.search(r"WORKERS:\s*\[", ln)).strip()
            normalized_plan = self._normalize_current_plan(plan if isinstance(plan, list) else [])
            if normalized_plan:
                state["current_plan"] = normalized_plan
                if answer:
                    state["worker_outputs"]["supervisor_context"] = answer
            else:
                state["current_plan"] = []
                state["final_response"] = answer or response.strip()
        except ValueError:
            state["current_plan"] = []
            answer = "\n".join(ln for ln in lines if not re.search(r"WORKERS:\s*\[", ln)).strip()
            state["final_response"] = answer or response.strip()
        return state

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

        task_text = str(task_packet.get("task", "")).strip()
        reason_text = str(task_packet.get("reason", "")).strip()
        if task_text:
            lines.append(f"Your task: {task_text}")
        if reason_text:
            lines.append(f"Why you were selected: {reason_text}")

        dependency_outputs = []
        for dependency in self._get_step_dependencies(task_packet, agent_def):
            dependency_output = state.get("worker_outputs", {}).get(dependency, "")
            if dependency_output:
                dependency_outputs.append(f"{dependency}: {dependency_output}")

        if dependency_outputs:
            lines.append("Relevant prior agent outputs:")
            lines.extend(dependency_outputs)

        return "<supervisor_handoff>\n" + "\n".join(lines) + "\n</supervisor_handoff>"

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

    @staticmethod
    def _build_agent_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = dict(state)
        snapshot["chat_history"] = list(state.get("chat_history", []) or [])
        snapshot["current_plan"] = list(state.get("current_plan", []) or [])
        snapshot["worker_outputs"] = dict(state.get("worker_outputs", {}) or {})
        return snapshot

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

        blocked_result = None
        for (agent_def, _task_packet), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error("Parallel agent %s failed with exception: %s", agent_def.name, result, exc_info=True)
                state["worker_outputs"][agent_def.name] = f"Error: {agent_def.name} failed with exception: {result}"
                continue

            agent_output = dict(result.get("worker_outputs", {}) or {}).get(agent_def.name)
            if agent_output is not None:
                state["worker_outputs"][agent_def.name] = agent_output

            if blocked_result is None and result.get(_BLOCKED_KEY):
                blocked_result = result[_BLOCKED_KEY]

        if blocked_result is not None:
            state[_BLOCKED_KEY] = blocked_result

        return state

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
    ) -> Optional[RouterResult]:
        if not self.cognitive_router.get_system_2_available():
            return None

        try:
            router_result = await asyncio.wait_for(
                self.cognitive_router.route_to_system_2(
                    messages,
                    allowed_tools=agent_def.allowed_tools,
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
    ) -> tuple[Optional[RouterResult], bool]:
        preferred_model = self._get_step_preferred_model(task_packet, agent_def)

        if preferred_model == "system_2":
            attempted_system_2 = self.cognitive_router.get_system_2_available()
            logger.info("Routing %s through System 2 (agent preference)", agent_def.name)
            router_result = await self._try_route_agent_system_2(messages, agent_def)
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
        return await self._try_route_agent_system_2(messages, agent_def), attempted_system_2

    async def _run_agent(
        self,
        agent_def: AgentDefinition,
        state: Dict[str, Any],
        *,
        task_packet: Optional[Dict[str, Any]] = None,
        deduct_energy: bool = True,
    ) -> Dict[str, Any]:
        state = normalize_state(state)
        if deduct_energy:
            state = self._deduct_energy(state, agent_def.energy_cost, agent_def.name)
        try:
            core_mem_str = await self.core_memory.get_context_string()

            capabilities_str = self._get_capabilities_string()
            handoff = self._build_agent_handoff(agent_def, state, task_packet=task_packet)
            messages = [
                {
                    "role": "system",
                    "content": f"{agent_def.system_prompt}\n{self.charter_text}\n{core_mem_str}\n\n{capabilities_str}",
                },
                {"role": "user", "content": handoff},
            ]
            router_result, attempted_system_2 = await self._route_agent_request(
                messages,
                agent_def,
                task_packet=task_packet,
            )
        except Exception as _agent_exc:
            if deduct_energy:
                state = self._refund_energy(state, agent_def.energy_cost, f"{agent_def.name}_exception")
            raise _agent_exc

        if router_result is None:
            if deduct_energy:
                state = self._refund_energy(state, agent_def.energy_cost, f"{agent_def.name}_null_result")
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
            solution = await self._handle_cognitive_escalation(state, router_result)
            state["worker_outputs"][agent_def.name] = solution
            return state

        if router_result.status != "ok":
            state[_BLOCKED_KEY] = router_result
        else:
            state["worker_outputs"][agent_def.name] = router_result.content

        return state

    async def _notify_admin(self, message: str) -> None:
        """Send a message to the admin via the outbound queue (used by heartbeat)."""
        if self.outbound_queue is not None:
            await self.outbound_queue.put(message)
        else:
            logger.info(f"[Admin notification (no queue)]: {message}")

    async def _heartbeat_loop(self) -> None:
        """
        Proactive Heartbeat: wakes every 30 min, queries the Objective Backlog
        for the highest-priority pending Task, processes it if energy allows,
        then notifies the admin with a summary.

        An asyncio.Lock prevents heartbeat cycles from overlapping: if the
        previous run is still in progress when the next interval fires, the
        new cycle is skipped (ISSUE-004).  A hard timeout of 90% of the
        heartbeat interval caps individual task execution time.
        """
        logger.info("Heartbeat loop started.")
        _heartbeat_lock = asyncio.Lock()
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if _heartbeat_lock.locked():
                logger.warning("Heartbeat: Previous run still in progress. Skipping this cycle.")
                continue
            async with _heartbeat_lock:
                try:
                    logger.info("Heartbeat: Querying objective backlog...")
                    task = await self.ledger_memory.get_highest_priority_task()

                    if not task:
                        logger.info("Heartbeat: Backlog is empty. Nothing to do.")
                        continue

                    energy_available = int(os.getenv("INITIAL_ENERGY_BUDGET", "100"))
                    if task["estimated_energy"] > energy_available:
                        logger.info(
                            f"Heartbeat: Task #{task['id']} needs {task['estimated_energy']} energy "
                            f"but only {energy_available} available. Skipping."
                        )
                        continue

                    logger.info(f"Heartbeat: Accepting task #{task['id']}: {task['title'][:60]}")
                    await self.ledger_memory.update_objective_status(task["id"], "active")

                    try:
                        result = await asyncio.wait_for(
                            self.process_message(
                                user_message=(
                                    f"[HEARTBEAT TASK #{task['id']}]: {task['title']}\n"
                                    f"You MUST execute this task right now by calling the appropriate tools. "
                                    f"Do NOT describe what you plan to do — use your tools and report the actual results. "
                                    f"If the task requires storing data, call update_core_memory or update_ledger explicitly. "
                                    f"If you cannot complete it, explain exactly why."
                                ),
                                user_id="heartbeat"
                            ),
                            timeout=HEARTBEAT_INTERVAL * 0.9,  # 27-minute hard cap
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Heartbeat: Task #{task['id']} timed out after "
                            f"{int(HEARTBEAT_INTERVAL * 0.9)}s. Marking pending."
                        )
                        await self.ledger_memory.update_objective_status(task["id"], "pending")
                        continue

                    # Only mark completed if the result isn't an error or refusal
                    status = "completed"
                    if any(w in result.lower() for w in ("error", "cannot", "unable", "failed", "i cannot", "i am unable")):
                        status = "pending"
                        logger.warning(f"Heartbeat: Task #{task['id']} may not have completed — marking pending. Result: {result[:100]}")
                    await self.ledger_memory.update_objective_status(task["id"], status)

                    summary = (
                        f"Heartbeat completed task #{task['id']}:\n"
                        f"  Task: {task['title']}\n"
                        f"  Result: {result[:200]}"
                    )
                    await self._notify_admin(summary)
                    logger.info(f"Heartbeat: Task #{task['id']} completed.")

                except Exception as e:
                    logger.error(f"Heartbeat error: {e}", exc_info=True)

    async def _memory_consolidation_loop(self) -> None:
        """Background task: periodically consolidate chat history into long-term memory."""
        logger.info("Memory consolidation loop started.")
        while True:
            await asyncio.sleep(MEMORY_CONSOLIDATION_INTERVAL)
            try:
                user_ids = await self.ledger_memory.get_recent_user_ids(limit=20)
                if not user_ids:
                    continue
                for user_id in user_ids:
                    await self._consolidate_memory(user_id)
                # Periodic DB pruning — delete old chat rows and vector memories
                try:
                    deleted = await self.ledger_memory.prune_old_chat_history(
                        days=int(os.getenv("CHAT_HISTORY_MAX_DAYS", "90")),
                        keep_minimum=int(os.getenv("CHAT_HISTORY_MIN_ROWS", "20")),
                    )
                    if deleted:
                        logger.info("Consolidation pruning: removed %d chat_history rows.", deleted)
                except Exception as _p_err:
                    logger.warning("Periodic chat history pruning failed: %s", _p_err)
                try:
                    await asyncio.to_thread(
                        self.vector_memory.prune_old_memories,
                        int(os.getenv("VECTOR_MEMORY_MAX_DAYS", "180")),
                    )
                except Exception as _vp_err:
                    logger.warning("Periodic vector memory pruning failed: %s", _vp_err)
            except Exception as e:
                logger.warning(f"Memory consolidation loop error: {e}", exc_info=True)

    async def _save_memory_async(self, text: str) -> None:
        """Fire-and-forget memory storage gated by response length.

        Skips trivial exchanges (short greetings, one-liners).
        Threshold: combined user+assistant text must exceed MEMORY_SAVE_THRESHOLD.
        Uses asyncio.to_thread() so the blocking ChromaDB call does not
        stall the event loop.
        """
        try:
            if len(text) < MEMORY_SAVE_THRESHOLD:
                return
            await self.vector_memory.add_memory_async(
                text=text,
                metadata={"type": "conversation", "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.warning(f"Async memory save failed: {e}")

    async def _load_approved_tools(self) -> None:
        """Re-register all previously approved dynamic tools from the DB at startup."""
        try:
            approved = await self.ledger_memory.get_approved_tools()
            for tool in approved:
                try:
                    self.cognitive_router.register_dynamic_tool(
                        tool["name"], tool["code"], tool["schema_json"]
                    )
                    logger.info(f"Restored dynamic tool '{tool['name']}' from registry")
                except Exception as e:
                    logger.error(f"Failed to restore tool '{tool['name']}': {e}")
        except Exception as e:
            logger.warning(f"_load_approved_tools failed: {e}")

    async def _load_pending_approvals(self) -> None:
        """Reload any tool synthesis proposals that were pending when the bot last stopped."""
        try:
            pending = await self.ledger_memory.load_pending_approvals()
            for user_id, payload in pending.items():
                self.pending_tool_approval[user_id] = {
                    "synthesis": payload["synthesis"],
                    "original_state": {"user_input": payload["original_input"], "user_id": user_id},
                }
            if pending:
                logger.info(f"Restored {len(pending)} pending tool approval(s) from DB")
        except Exception as e:
            logger.warning(f"_load_pending_approvals failed: {e}")

    async def _load_pending_mfa(self) -> None:
        """Reload persisted MFA states from DB so bot restarts don't drop pending challenges."""
        try:
            states = await self.ledger_memory.load_mfa_states()
            for user_id, entry in states.items():
                self.pending_mfa[user_id] = entry
            if states:
                logger.info("Restored %d pending MFA state(s) from DB", len(states))
        except Exception as e:
            logger.warning("_load_pending_mfa failed: %s", e)

    async def _load_pending_hitl(self) -> None:
        """Reload any HITL states that were persisted before the last shutdown (ISSUE-013)."""
        try:
            states = await self.ledger_memory.load_hitl_states()
            for user_id, state in states.items():
                self.pending_hitl_state[user_id] = state
            if states:
                logger.info(f"Restored {len(states)} pending HITL state(s) from DB")
                # Notify admin so they know the context survived the restart
                await self._notify_admin(
                    f"⚠️ AIDEN restarted with {len(states)} pending HITL workflow(s) restored. "
                    f"Affected user(s): {', '.join(states.keys())}. Reply to continue."
                )
        except Exception as e:
            logger.warning(f"_load_pending_hitl failed: {e}")

    def _fire_and_forget(self, coro) -> asyncio.Task:
        """Schedule a coroutine as a background task with a strong GC-safe reference (ISSUE-002).

        The task is added to ``_background_tasks`` and automatically removed
        when it completes, so the set never grows unboundedly while still
        preventing the garbage collector from destroying mid-flight tasks.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Return the per-user asyncio.Lock with LRU eviction to bound memory usage."""
        _max = int(os.getenv("USER_LOCKS_MAX_SIZE", "500"))
        async with self._user_locks_lock:
            if user_id in self._user_locks:
                # Move to end (most-recently-used)
                self._user_locks[user_id] = self._user_locks.pop(user_id)
                return self._user_locks[user_id]
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
            # Evict oldest entries when over the cap
            while len(self._user_locks) > _max:
                self._user_locks.pop(next(iter(self._user_locks)))
            return lock

    async def tool_synthesis_node(
        self,
        state: Dict[str, Any],
        router_result: RouterResult,
    ) -> str:
        """
        Cognitive handoff: router flagged a gap → System 2 synthesises a tool
        → Admin approves via HITL → tool registered → original query retried.

        Returns a HITL prompt string for the Admin; the orchestrator stores the
        synthesis payload in pending_tool_approval until the Admin replies.
        """
        user_id = state["user_id"]
        gap_description = router_result.gap_description
        suggested_tool_name = router_result.suggested_tool_name
        logger.info(f"Tool synthesis triggered: gap='{gap_description}'")

        if not self.cognitive_router.get_system_2_available():
            return (
                "System 1 identified a capability gap but System 2 is offline — "
                "cannot synthesise a new tool right now. Please configure GROQ_API_KEY."
            )

        try:
            synthesis = await asyncio.wait_for(
                self.cognitive_router.synthesize_tool(
                    gap_description=gap_description,
                    suggested_tool_name=suggested_tool_name,
                    user_query=state["user_input"],
                ),
                timeout=60.0,
            )
        except Exception as e:
            logger.error(f"Tool synthesis failed: {e}", exc_info=True)
            return f"System 2 failed to synthesise a tool for '{gap_description}': {e}"

        # Store payload for approval resumption (in-memory + persisted to DB)
        self.pending_tool_approval[user_id] = {
            "synthesis": synthesis,
            "original_state": state,
            "_created_at": time.time(),
        }
        await self.ledger_memory.save_pending_approval(
            user_id, synthesis, state["user_input"]
        )

        # Build Admin-facing HITL prompt with the proposed code
        hitl_msg = (
            f"🔧 TOOL SYNTHESIS REQUEST\n\n"
            f"System 1 could not answer: \"{state['user_input']}\"\n"
            f"Gap identified: {gap_description}\n\n"
            f"System 2 has drafted the following tool:\n"
            f"─────────────────────────────\n"
            f"Name: {synthesis['tool_name']}\n"
            f"Description: {synthesis['description']}\n\n"
            f"Code:\n```python\n{synthesis['code']}\n```\n"
            f"─────────────────────────────\n"
            f"Reply YES to approve and deploy, or NO to reject."
        )
        return hitl_msg

    def _extract_router_content(self, result: RouterResult) -> Optional[str]:
        """
        Return the string content from an "ok" RouterResult, or None if it is
        a non-ok result (MFA / HITL / capability-gap).  Non-ok results should
        be stored in state[_BLOCKED_KEY] by the caller.
        """
        if result.status == "ok":
            return result.content
        return None

    async def supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plans execution. Tries System 1 (local) first per Directive 1.3; escalates to System 2 only on failure."""
        state = normalize_state(state)
        state = self._deduct_energy(state, ENERGY_COST_SUPERVISOR, "supervisor")
        user_input = state["user_input"]
        core_mem_str = await self.core_memory.get_context_string()
        archival_context = await self._get_archival_context(user_input)
        capabilities_str = self._get_capabilities_string()
        archival_block = f"{archival_context}\n\n" if archival_context else ""

        system_prompt = build_supervisor_prompt(
            charter_text=self.charter_text,
            core_mem_str=core_mem_str,
            archival_block=archival_block,
            capabilities_str=capabilities_str,
            agent_descriptions=self._get_agent_descriptions(),
            sensory_context=self._get_sensory_context(),
            os_name=self.sensory_state.get("os", platform.system()),
            downloads_dir=self.prompt_config.downloads_dir,
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.get("chat_history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        if state.get("critic_instructions"):
            messages.append({"role": "system", "content": state["critic_instructions"]})
        messages.append({"role": "user", "content": f"<user_input>{user_input}</user_input>"})

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
                    state["final_response"] = (
                        "I was unable to process this request locally. This task likely requires "
                        "capabilities beyond my local model (e.g. internet access or complex reasoning). "
                        "To enable this, configure a System 2 provider by setting GROQ_API_KEY in your .env file. "
                        "Groq offers a free tier at console.groq.com."
                    )
                else:
                    state["final_response"] = "Both local and cloud reasoning failed on this request. Please try rephrasing or simplifying the task."
                return state

            state = self._parse_supervisor_response(response, state)

        except asyncio.TimeoutError:
            logger.error("Supervisor node timeout (60s)")
            state = self._refund_energy(state, ENERGY_COST_SUPERVISOR, "supervisor_timeout")
            state["current_plan"] = []
            state["final_response"] = "Planning timed out. Please try again."
        except RequiresHITLError:
            raise
        except Exception as e:
            logger.error(f"Supervisor Node failed: {e}", exc_info=True)
            state = self._refund_energy(state, ENERGY_COST_SUPERVISOR, "supervisor_error")
            state["current_plan"] = []
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
    def _finalize_critic_pass(state: Dict[str, Any], output_to_eval: str) -> Dict[str, Any]:
        state["critic_feedback"] = "PASS"
        if not state.get("final_response"):
            state["final_response"] = output_to_eval
        return state

    def _build_critic_messages(self, output_to_eval: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Evaluate the output. Output PASS if it follows the charter, or FAIL with a reason.\n"
                    f"Charter: {self.charter_text}"
                ),
            },
            {"role": "user", "content": output_to_eval[:1000]},
        ]

    async def _route_critic_request(self, messages: List[Dict[str, str]]) -> RouterResult:
        if self.cognitive_router.get_system_2_available():
            logger.info("Routing Critic through System 2 (Gemini)")
            return await asyncio.wait_for(
                self.cognitive_router.route_to_system_2(messages, allowed_tools=[]),
                timeout=30.0,
            )
        return await self._route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=60.0,
            context="critic",
        )

    @staticmethod
    def _apply_critic_response(
        state: Dict[str, Any],
        output_to_eval: str,
        response: str,
    ) -> Dict[str, Any]:
        if "FAIL" in response.upper():
            state["iteration_count"] += 1
            state["critic_feedback"] = response
            return state
        return Orchestrator._finalize_critic_pass(state, output_to_eval)

    async def critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Checks worker output against the charter. Skipped for direct supervisor responses."""
        state = normalize_state(state)
        output_to_eval = self._get_output_to_evaluate(state)

        if not output_to_eval:
            state["critic_feedback"] = "PASS"
            return state

        if not self._should_run_critic_review(state, output_to_eval):
            return self._finalize_critic_pass(state, output_to_eval)

        state = self._deduct_energy(state, ENERGY_COST_CRITIC, "critic")
        messages = self._build_critic_messages(output_to_eval)

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
        except Exception as e:
            logger.warning(f"Critic node failed/timed out: {e}. Defaulting to PASS.", exc_info=True)
            state = self._refund_energy(state, ENERGY_COST_CRITIC, "critic_error")
            state = self._finalize_critic_pass(state, output_to_eval)

        if state["iteration_count"] >= 3 and state["critic_feedback"] != "PASS":
            await self._spawn_debug_task(state)
            raise RequiresHITLError(
                "Critic rejected output 3 times. Guidance needed.\n"
                "Question: How should I proceed to satisfy the charter?"
            )

        return state

    async def _consolidate_memory(self, user_id: str) -> None:
        """
        Memory Consolidation Skill: summarise the last 10 chat turns for
        *user_id* into a single paragraph, update core_memory under
        ``conversation_summary``, and trim the stored raw turns to the most
        recent 5.  Runs fire-and-forget; errors are swallowed.
        """
        try:
            turns = await self.ledger_memory.get_chat_history(user_id, limit=10)
            if len(turns) < 10:
                return  # Not enough turns to warrant consolidation

            convo_text = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)
            messages = [
                {"role": "system", "content": "You are a memory consolidation engine. Summarise the conversation concisely in 2–3 sentences."},
                {"role": "user", "content": convo_text},
            ]
            result = await self._route_to_system_1(
                messages,
                allowed_tools=[],
                deadline_seconds=90.0,
                context="memory_consolidation",
            )
            if result.status == "ok" and result.content:
                summary = result.content.strip()
                await self.core_memory.update("conversation_summary", summary)
                await self.ledger_memory.trim_chat_history(user_id, keep_last=5)
                await self.vector_memory.add_memory_async(
                    text=f"Conversation summary ({user_id}): {summary}",
                    metadata={"type": "conversation_summary", "user_id": user_id, "timestamp": datetime.now().isoformat()}
                )
                logger.info(f"Memory consolidation complete for user {user_id}")
        except asyncio.TimeoutError:
            logger.warning("Memory consolidation skipped: LLM did not respond within 90s")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {type(e).__name__}: {e}", exc_info=True)

    async def _try_resume_mfa(self, user_id: str, user_message: str) -> Optional[str]:
        """Handle MFA challenge response. Returns a reply string, or None if not in MFA flow."""
        if user_id not in self.pending_mfa:
            return None
        pending_tool = self.pending_mfa[user_id]
        age = time.time() - pending_tool.get("_created_at", 0)
        if age > _PENDING_STATE_TTL_SECONDS:
            del self.pending_mfa[user_id]
            self._fire_and_forget(self.ledger_memory.clear_mfa_state(user_id))
            logger.info("Expired stale pending_mfa for %s (age %.0fs).", user_id, age)
            return None
        del self.pending_mfa[user_id]
        self._fire_and_forget(self.ledger_memory.clear_mfa_state(user_id))
        if not verify_mfa_challenge(user_message):
            return "Error: MFA authorization failed. Action aborted."
        exec_result = await self.cognitive_router._execute_tool(
            pending_tool["name"], pending_tool["arguments"]
        )
        if exec_result.status == "ok":
            return exec_result.content
        return self._handle_blocked_result(exec_result, pending_tool.get("user_id", user_id), {})

    async def _try_resume_tool_approval(self, user_id: str, user_message: str) -> Optional[str]:
        """Handle YES/NO tool synthesis approval. Returns a reply string, or None if not pending."""
        if user_id not in self.pending_tool_approval:
            return None
        payload = self.pending_tool_approval[user_id]
        age = time.time() - payload.get("_created_at", 0)
        if age > _PENDING_STATE_TTL_SECONDS:
            self.pending_tool_approval.pop(user_id, None)
            await self.ledger_memory.clear_pending_approval(user_id)
            logger.info("Expired stale pending_tool_approval for %s (age %.0fs).", user_id, age)
            return None
        self.pending_tool_approval.pop(user_id)
        await self.ledger_memory.clear_pending_approval(user_id)
        synthesis = payload["synthesis"]
        original_state = payload["original_state"]
        if not user_message.strip().upper().startswith("YES"):
            logger.info(f"Admin rejected tool synthesis for user {user_id}")
            return f"Tool proposal rejected. The capability gap remains: {synthesis['description']}"
        tool_name = synthesis["tool_name"]
        try:
            self.cognitive_router.register_dynamic_tool(tool_name, synthesis["code"], synthesis["schema_json"])
            await self.ledger_memory.register_tool(
                name=tool_name, description=synthesis["description"],
                code=synthesis["code"], schema_json=synthesis["schema_json"],
            )
            await self.ledger_memory.approve_tool(tool_name)
            core = await self.core_memory.get_all()
            caps = core.get("known_capabilities", "")
            await self.core_memory.update("known_capabilities", f"{caps}, {tool_name}".lstrip(", "))
            logger.info(f"Tool '{tool_name}' approved, registered, and logged to core memory")
            retry = await self.process_message(original_state["user_input"], user_id)
            return f"✅ Tool '{tool_name}' deployed.\n\n{retry}"
        except Exception as e:
            logger.error(f"Tool registration failed: {e}", exc_info=True)
            return f"Error deploying tool '{tool_name}': {e}"

    async def _load_state(self, user_id: str, user_message: str) -> Dict[str, Any]:
        """Return a state dict: resumes a HITL conversation if pending, else creates fresh."""
        if user_id in self.pending_hitl_state:
            _hitl_entry = self.pending_hitl_state[user_id]
            _age = time.time() - _hitl_entry.get("_hitl_created_at", 0)
            if _age > _PENDING_STATE_TTL_SECONDS:
                self.pending_hitl_state.pop(user_id, None)
                try:
                    await self.ledger_memory.clear_hitl_state(user_id)
                except Exception:
                    pass
                logger.warning("Expired stale pending_hitl_state for %s (age %.0fs); creating fresh state.", user_id, _age)
                # Fall through to fresh state below
            else:
                state = normalize_state(self.pending_hitl_state.pop(user_id))
                # Clear the persisted DB record now that we've resumed (ISSUE-013)
                try:
                    await self.ledger_memory.clear_hitl_state(user_id)
                except Exception as e:
                    logger.warning(f"Failed to clear HITL state from DB for {user_id}: {e}")
                state["admin_guidance"] = user_message
                state["user_input"] += f"\n[ADMIN GUIDANCE: {user_message}]"
                state["iteration_count"] = 0
                state["current_plan"] = []
                # Track how many HITL cycles have been spent on this task (ISSUE-005)
                state["hitl_count"] = state.get("hitl_count", 0) + 1
                # Cap recharge to 75 so energy budget is never fully restored — this
                # ensures a perpetually failing task eventually becomes unrecoverable
                # rather than cycling indefinitely at full energy (ISSUE-005).
                state["energy_remaining"] = min(state.get("energy_remaining", 0) + 50, 75)
                # If the admin has guided this task 3+ times without success, abandon it
                if state["hitl_count"] >= 3:
                    logger.warning(
                        f"HITL cycle limit (3) reached for user {user_id}. "
                        f"Abandoning task: {state.get('user_input', '')[:80]}"
                    )
                    state["final_response"] = (
                        "This task has been attempted 3 times with admin guidance and "
                        "could not be completed. The request has been abandoned to prevent "
                        "an infinite loop. Please rephrase or break it into smaller steps."
                    )
                return normalize_state(state)
        state = self._new_state(user_id, user_message)
        if user_id != "heartbeat":
            try:
                state["chat_history"] = await self.ledger_memory.get_chat_history(
                    user_id,
                    limit=_RECENT_CHAT_HISTORY_LIMIT,
                )
            except Exception as e:
                logger.warning(f"Failed to load chat history for {user_id}: {e}")
        return normalize_state(state)

    @staticmethod
    def _has_ready_final_response(state: Dict[str, Any]) -> bool:
        return bool(state.get("final_response") and not state.get("current_plan"))

    def _consume_blocked_result(self, state: Dict[str, Any], user_id: str) -> Optional[str]:
        blocked_result = state.pop(_BLOCKED_KEY, None)
        if blocked_result is None:
            return None
        return self._handle_blocked_result(blocked_result, user_id, state)

    @staticmethod
    def _apply_critic_retry_instructions(state: Dict[str, Any]) -> None:
        if state["iteration_count"] > 0:
            state["critic_instructions"] = (
                f"[CRITIC FEEDBACK: {state['critic_feedback']}. Fix your output.]"
            )

    async def _run_manual_graph_pass(
        self,
        state: Dict[str, Any],
        user_id: str,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        for node in (self.supervisor_node, self.execute_workers_node, self.critic_node):
            state = await node(state)
            blocked_response = self._consume_blocked_result(state, user_id)
            if blocked_response is not None:
                return state, blocked_response
        return state, None

    async def _run_graph_pass(
        self,
        state: Dict[str, Any],
        user_id: str,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        if self._compiled_graph is not None:
            state = await self._compiled_graph.ainvoke(state)
            return state, self._consume_blocked_result(state, user_id)
        return await self._run_manual_graph_pass(state, user_id)

    @staticmethod
    def _reset_after_critic_rejection(state: Dict[str, Any]) -> None:
        state["final_response"] = ""
        state["worker_outputs"] = {}
        state["current_plan"] = []

    @staticmethod
    def _ensure_final_response(state: Dict[str, Any], max_iterations: int) -> None:
        if state["iteration_count"] >= max_iterations and state.get("critic_feedback") != "PASS":
            state["final_response"] = "Unable to fulfill this request — output repeatedly failed internal safety checks."
        if not state.get("final_response"):
            state["final_response"] = "No valid response could be generated."

    @staticmethod
    def _is_error_response(response: str) -> bool:
        return any(response.startswith(prefix) for prefix in _ERROR_RESPONSE_PREFIXES)

    async def _persist_chat_turns(self, user_id: str, user_message: str, final_resp: str) -> None:
        if user_id == "heartbeat" or self._is_error_response(final_resp):
            return
        try:
            await self.ledger_memory.save_chat_turn(user_id, "user", user_message)
            await self.ledger_memory.save_chat_turn(user_id, "assistant", final_resp)
        except Exception as e:
            logger.warning(f"Failed to save chat turn for {user_id}: {e}")
        self._fire_and_forget(self._consolidate_memory(user_id))

    def _schedule_response_memory_save(self, user_message: str, final_resp: str) -> None:
        self._fire_and_forget(
            self._save_memory_async(f"User: {user_message}\nAssistant: {final_resp}")
        )

    async def _finalize_user_response(self, user_id: str, user_message: str, response: str) -> str:
        final_resp = self.cognitive_router.sanitize_response(response)
        await self._persist_chat_turns(user_id, user_message, final_resp)
        self._schedule_response_memory_save(user_message, final_resp)
        return final_resp

    async def _run_graph_loop(self, state: Dict[str, Any], user_id: str, user_message: str) -> str:
        """Execute the supervisor → workers → critic loop. Returns the final sanitized response."""
        state = normalize_state(state)
        max_iterations = 3

        # Fast-exit if the state already carries a final response before we enter
        # the loop — e.g. when the HITL cycle limit has been reached (ISSUE-005).
        if self._has_ready_final_response(state):
            return self.cognitive_router.sanitize_response(state["final_response"])

        while state["iteration_count"] < max_iterations:
            blocked_response = self._consume_blocked_result(state, user_id)
            if blocked_response is not None:
                return blocked_response

            self._apply_critic_retry_instructions(state)
            state, blocked_response = await self._run_graph_pass(state, user_id)
            if blocked_response is not None:
                return blocked_response

            if state["critic_feedback"] == "PASS":
                state["critic_instructions"] = ""
                break
            logger.warning(f"Critic rejected output on iteration {state['iteration_count']}")
            # Clear the stale plan so the supervisor re-plans with the Critic
            # feedback injected via state["critic_instructions"] (ISSUE-006).
            self._reset_after_critic_rejection(state)

        self._ensure_final_response(state, max_iterations)
        return await self._finalize_user_response(user_id, user_message, state["final_response"])

    async def process_message(self, user_message: str, user_id: str) -> str:
        """Main entry point: State Graph execution with Energy Budget."""
        if not user_message:
            return "Error: Invalid message"

        # Serialise concurrent messages for the same user_id to prevent
        # race conditions on pending_mfa / pending_hitl_state / pending_tool_approval
        # dicts (ISSUE-012).  Different users are still processed concurrently.
        lock = await self._get_user_lock(user_id)
        async with lock:
            reply = await self._try_resume_mfa(user_id, user_message)
            if reply is not None:
                return reply

            reply = await self._try_resume_tool_approval(user_id, user_message)
            if reply is not None:
                return reply

            state = await self._load_state(user_id, user_message)
            if state.get("final_response") and not state.get("current_plan"):
                return self.cognitive_router.sanitize_response(state["final_response"])

            await self._remember_user_profile(user_id, user_message)
            await self._remember_assistant_identity(user_message)

            reply = await self._try_fast_path_response(state)
            if reply is not None:
                return await self._finalize_user_response(user_id, user_message, reply)

            try:
                return await self._run_graph_loop(state, user_id, user_message)
            except RequiresHITLError as hitl_err:
                state["_hitl_question"] = str(hitl_err)
                state["_hitl_created_at"] = time.time()
                self.pending_hitl_state[user_id] = state
                # Persist so the state survives a bot restart (ISSUE-013)
                self._fire_and_forget(self.ledger_memory.save_hitl_state(user_id, state))
                return str(hitl_err)
            except Exception as e:
                logger.error(f"Graph execution failed: {e}", exc_info=True)
                return "An internal error occurred."

    def _handle_blocked_result(
        self,
        result: RouterResult,
        user_id: str,
        state: Dict[str, Any],
    ) -> str:
        """
        Convert a non-ok RouterResult into the appropriate user-facing string,
        registering any pending MFA / HITL / tool-synthesis state as needed.
        """
        if result.status == "mfa_required":
            self.pending_mfa[user_id] = {
                "name": result.mfa_tool_name,
                "arguments": result.mfa_arguments,
                "_created_at": time.time(),
            }
            self._fire_and_forget(
                self.ledger_memory.save_mfa_state(user_id, result.mfa_tool_name, result.mfa_arguments)
            )
            return "SECURITY LOCK: To authorize this core change, complete the phrase: 'The sky is...'"

        if result.status == "hitl_required":
            state["_hitl_question"] = result.hitl_message
            state["_hitl_created_at"] = time.time()
            self.pending_hitl_state[user_id] = state
            # Persist so the HITL state survives a bot restart (ISSUE-013)
            self._fire_and_forget(self.ledger_memory.save_hitl_state(user_id, state))
            return result.hitl_message

        if result.status == "capability_gap":
            # Run synthesis in the background via _fire_and_forget which holds
            # a strong GC-safe reference (ISSUE-002).
            self._fire_and_forget(self._async_tool_synthesis(user_id, result, state))
            return (
                f"I identified a capability gap: {result.gap_description}. "
                f"Requesting tool synthesis from System 2..."
            )

        # Fallback
        return f"An unexpected router status was received: {result.status}"

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
        prompt = (
            "You are System 2. System 1 has escalated a complex problem to you.\n\n"
            f"Problem Description: {router_result.escalation_problem}\n\n"
            f"Context Scratchpad: {router_result.escalation_context}\n\n"
            "Please provide a direct solution to the user's problem. "
            "Additionally, generate a brief 'Reasoning Blueprint' on how to solve this class of problem. "
            "You MUST format your output strictly using XML tags:\n"
            "<solution> The actual answer to the user's problem... </solution>\n"
            "<blueprint> How to solve this problem: Step 1, Step 2... </blueprint>"
        )

        messages = [{"role": "user", "content": prompt}]
        sys2_result = await self.cognitive_router.route_to_system_2(messages)

        content = sys2_result.content if sys2_result.status == "ok" else "[System 2 - Error]: Escalation failed."

        solution_match = re.search(r"<solution>(.*?)</solution>", content, flags=re.DOTALL | re.IGNORECASE)
        blueprint_match = re.search(r"<blueprint>(.*?)</blueprint>", content, flags=re.DOTALL | re.IGNORECASE)

        solution_text = solution_match.group(1).strip() if solution_match else content.strip()
        blueprint_text = blueprint_match.group(1).strip() if blueprint_match else None

        if blueprint_text:
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

    async def _async_tool_synthesis(
        self,
        _user_id: str,
        result: RouterResult,
        state: Dict[str, Any],
    ) -> None:
        """Background task: run tool synthesis and send the HITL prompt to admin."""
        try:
            hitl_prompt = await self.tool_synthesis_node(state, result)
            if self.outbound_queue is not None:
                await self.outbound_queue.put(hitl_prompt)
        except Exception as e:
            logger.error(f"Background tool synthesis failed: {e}", exc_info=True)

    def close(self) -> None:
        try:
            set_runtime_context(None, None)
            if hasattr(self, 'vector_memory') and self.vector_memory:
                self.vector_memory.close()
            if hasattr(self, 'cognitive_router') and self.cognitive_router:
                # Schedule async close — best-effort at shutdown time
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.cognitive_router.close())
                        loop.create_task(self.ledger_memory.close())
                    else:
                        loop.run_until_complete(self.cognitive_router.close())
                        loop.run_until_complete(self.ledger_memory.close())
                except Exception:
                    pass
            logger.info("Orchestrator resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing Orchestrator: {e}", exc_info=True)
