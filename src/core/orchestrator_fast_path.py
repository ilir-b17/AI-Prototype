"""Fast-path response logic mixin for Orchestrator."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from src.core.orchestrator_constants import (
    _INTENT_CLASSIFIER_CACHE_TTL_SECONDS,
    _INTENT_CLASSIFIER_CACHE_MAX_SIZE_DEFAULT,
    _INTENT_CLASSIFIER_TIMEOUT_SECONDS,
    _INTENT_CLASSIFIER_MAX_OUTPUT_TOKENS,
    _USER_INTENTS,
    _SYSTEM_1_ERROR_PREFIX,
    _GOAL_PLANNER_COMPLEXITY_THRESHOLD,
    _CHARTER_TIER_TAGS,
    _CHARTER_PARSE_ERROR_KEY,
    _CATALOG_MATCH_STOPWORDS,
    _CATALOG_META_TOOL_NAMES,
    _VOICE_NOTE_PLACEHOLDER_RE,
    _MULTI_BLANK_LINES_RE,
)
from src.core.routing_assessor import (
    RoutingAssessor,
    _ROUTING_STOPWORDS,
    _ROUTING_TOKEN_RE,
)
from src.core.goal_planner import PlanningResult
from src.core.orchestrator_memory import _MemoryOpsMixin
from src.core.llm_router import RouterResult
from src.core.progress import ProgressEvent, get_current_emitter
from src.core import cloud_redaction

logger = logging.getLogger(__name__)


def _is_voice_placeholder_text(text: str) -> bool:
    return bool(_VOICE_NOTE_PLACEHOLDER_RE.fullmatch(str(text or "").strip()))


class _FastPathMixin:
    """Fast-path response logic. Mixed into Orchestrator."""

    _CHARTER_FALLBACK = "Core Directive: Do no harm."

    _CONFIRMATORY_FOLLOWUP_RE = re.compile(
        r"^\s*(did you|have you|were you|has it|was that|was it|did it|is that right|is that correct)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _should_use_intent_classifier(user_message: str) -> bool:
        lowered = str(user_message or "").lower()
        classifier_hints = (
            "capability",
            "capabilities",
            "do you have",
            "have access",
            "can you access",
            "access the internet",
            "browse the internet",
            "can you browse",
            "are you able to",
            "ability to",
        )
        return any(hint in lowered for hint in classifier_hints)

    @staticmethod

    def _intent_cache_key(user_message: str) -> str:
        normalized = re.sub(r"\s+", " ", str(user_message or "").strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod

    def _parse_intent_classifier_response(content: str) -> Optional[str]:
        raw = str(content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        intent = str(payload.get("intent") or "").strip().lower()
        return intent if intent in _USER_INTENTS else None

    @staticmethod

    def _fallback_classify_user_intent(user_message: str) -> str:
        lowered = str(user_message or "").lower()
        if Orchestrator._is_capability_question(user_message):
            return "capability_query"
        if (
            _MemoryOpsMixin._extract_user_profile_updates(user_message)
            or _MemoryOpsMixin._extract_assistant_identity_update(user_message)
        ):
            return "profile_update"
        if RoutingAssessor._is_trivial_direct_intent(user_message):
            return "casual"
        if Orchestrator._looks_like_task_request(lowered):
            return "task"
        return "casual"

    @staticmethod

    def _is_capability_question(user_message: str) -> bool:
        lowered = (user_message or "").lower()
        capability_markers = ("tool", "tools", "skill", "skills", "capabil", "internet", "browse", "web", "repository")
        if not any(marker in lowered for marker in capability_markers):
            return False
        if not any(
            phrase in lowered
            for phrase in (
                "can you",
                "do you have",
                "are you sure",
                "have access",
                "access the internet",
                "browse the internet",
            )
        ):
            return False
        # Exclude action requests: "can you search the web FOR X" is a task, not a capability query.
        # Require at least 12 chars of content after "for/about" to distinguish a specific topic
        # from a short pronoun like "that" or "it" in a genuine capability question.
        if re.search(r'\b(?:search|browse|web|internet)\b.*\b(?:for|about)\b.{12,}', lowered):
            return False
        if Orchestrator._looks_like_task_request(lowered):
            return False
        return True

    @staticmethod

    def _looks_like_task_request(lowered: str) -> bool:
        task_markers = (
            "what is", "what's", "who is", "who's", "meaning of", "define", "explain",
            "tell me", "summarize", "summarise", "check", "find", "search", "look up",
            "get", "show", "read", "open", "list", "store", "remember", "record",
            "add", "create",
        )
        if not any(marker in lowered for marker in task_markers):
            return False

        meta_tokens = {
            "tool", "tools", "skill", "skills", "capability", "capabilities",
            "internet", "browse", "web", "repository", "access", "search",
            "check", "current", "live", "information", "data", "must", "need",
            "needed", "available", "have", "use",
        }
        informative_tokens = {
            token for token in re.findall(_ROUTING_TOKEN_RE, lowered)
            if len(token) > 3 and token not in _ROUTING_STOPWORDS and token not in meta_tokens
        }
        return bool(informative_tokens)

    @staticmethod

    def _build_intent_classifier_messages(user_message: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Classify the user's intent. Return ONLY strict JSON matching this schema: "
                    '{"intent":"capability_query|task|profile_update|casual"}. '
                    "Use capability_query for questions about what tools, access, skills, or capabilities are available. "
                    "Use task when the user asks you to do or find something."
                ),
            },
            {"role": "user", "content": str(user_message or "")},
        ]

    async def _classify_user_intent(self, user_message: str) -> str:
        cache_key = self._intent_cache_key(user_message)
        cached_intent = self._get_cached_user_intent(cache_key)
        if cached_intent in _USER_INTENTS:
            return cached_intent

        fallback_intent = self._fallback_classify_user_intent(user_message)
        try:
            result = await self._route_to_system_1(
                self._build_intent_classifier_messages(user_message),
                allowed_tools=[],
                deadline_seconds=_INTENT_CLASSIFIER_TIMEOUT_SECONDS,
                context="intent_classifier",
                max_output_tokens=_INTENT_CLASSIFIER_MAX_OUTPUT_TOKENS,
            )
            if result.status == "ok" and result.content and not result.content.startswith(_SYSTEM_1_ERROR_PREFIX):
                classified_intent = self._parse_intent_classifier_response(result.content)
                if classified_intent in _USER_INTENTS:
                    self._cache_user_intent(cache_key, classified_intent)
                    return classified_intent
        except Exception as e:
            logger.debug("Intent classifier failed; falling back to heuristic: %s", e)

        self._cache_user_intent(cache_key, fallback_intent)
        return fallback_intent


    def _get_cached_user_intent(self, cache_key: str) -> Optional[str]:
        cache = self._coerce_intent_cache()
        cached = cache.get(cache_key)
        if not cached:
            return None
        if not isinstance(cached, (tuple, list)) or len(cached) != 2:
            cache.pop(cache_key, None)
            return None
        expires_at, intent = cached
        if float(expires_at) <= time.time():
            cache.pop(cache_key, None)
            return None
        cache.move_to_end(cache_key)
        return str(intent)

    def _cache_user_intent(self, cache_key: str, intent: str) -> None:
        cache = self._coerce_intent_cache()
        cache[cache_key] = (time.time() + _INTENT_CLASSIFIER_CACHE_TTL_SECONDS, intent)
        cache.move_to_end(cache_key)
        self._evict_ordered_mapping_to_size(
            cache,
            self._intent_classifier_cache_max_size(),
            label="Intent classifier cache",
        )


    def _coerce_intent_cache(self) -> "OrderedDict[str, tuple[float, str]]":
        cache = getattr(self, "_intent_classification_cache", None)
        if isinstance(cache, OrderedDict):
            return cache
        if isinstance(cache, dict):
            ordered = OrderedDict(cache.items())
        else:
            ordered = OrderedDict()
        self._intent_classification_cache = ordered
        self._evict_ordered_mapping_to_size(
            ordered,
            self._intent_classifier_cache_max_size(),
            label="Intent classifier cache",
        )
        return ordered

    @classmethod
    def _intent_classifier_cache_max_size(cls) -> int:
        return cls._resolve_positive_int_env(
            "INTENT_CLASSIFIER_CACHE_MAX_SIZE",
            _INTENT_CLASSIFIER_CACHE_MAX_SIZE_DEFAULT,
        )

    @staticmethod

    def _is_skill_list_request(user_message: str) -> bool:
        """Return True when the user wants a full inventory list of AIDEN's skills/tools."""
        lowered = (user_message or "").lower()
        # Exclude clear task-search requests like "search for a list of tools for woodworking"
        if re.search(r"\b(?:search|find|look up|fetch)\b.{0,30}\bfor\b", lowered):
            return False
        # Must mention skills/tools/capabilities
        if not re.search(r"\b(?:skills?|tools?|capabilit\w+|commands?)\b", lowered):
            return False
        # Inventory/listing intent phrases
        if re.search(r"\blist\s+of\b", lowered):
            return True
        if re.search(r"\b(?:list|show|display|enumerate)\b", lowered):
            return True
        if re.search(r"\bwhat\b.{0,30}\b(?:skills?|tools?|capabilit\w+)\b", lowered):
            return True
        if re.search(r"\ball\s+(?:your\s+|of\s+your\s+)", lowered):
            return True
        return False

    def _build_compact_skill_list_response(self) -> str:
        """Return a compact bulleted list of registered skills with brief descriptions."""
        entries = self._load_capability_catalog_entries()
        if not entries:
            return "I don't have any registered skills at the moment."

        lines: List[str] = []
        for item in sorted(entries, key=lambda e: str(e.get("name") or "").lower()):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            desc = str(item.get("description") or "").strip()
            if desc and len(desc) > 70:
                cut = desc[:70].rsplit(" ", 1)[0]
                desc = cut + "\u2026"
            lines.append(f"\u2022 {name}" + (f" \u2014 {desc}" if desc else ""))

        total = len(lines)
        header = f"I have {total} registered skill{'s' if total != 1 else ''}:"
        return header + "\n\n" + "\n".join(lines)

    def _build_capability_response(self, user_message: str, *, classified_intent: Optional[str] = None) -> Optional[str]:
        if classified_intent != "capability_query" and not self._is_capability_question(user_message):
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
            if _is_voice_placeholder_text(content):
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

    # Matches short confirmatory questions about the previous turn, e.g.
    # "Did you search online for it?", "Have you found it?", "Was that correct?"
    _CONFIRMATORY_FOLLOWUP_RE = re.compile(
        r"^\s*(did you|have you|were you|has it|was that|was it|did it|is that right|is that correct)\b",
        re.IGNORECASE,
    )

    @staticmethod

    def _is_confirmatory_followup(user_message: str) -> bool:
        return bool(
            Orchestrator._CONFIRMATORY_FOLLOWUP_RE.match(user_message or "")
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

    def _meta_history_response(user_message: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
        if Orchestrator._is_last_reply_question(user_message):
            return Orchestrator._recall_last_assistant_message(chat_history)

        # Confirmatory follow-ups like "Did you search for it?" / "Have you found it?"
        # should acknowledge the previous action, not re-trigger a tool.
        if Orchestrator._is_confirmatory_followup(user_message):
            return Orchestrator._recall_last_assistant_message(chat_history)

        if Orchestrator._is_summary_request(user_message):
            return Orchestrator._summarize_chat_history(chat_history)

        return None

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
        write_verbs = (
            "store", "save", "record", "remember", "write", "note",
            "add", "keep", "log", "put", "jot",
        )
        memory_markers = ("memory", "long-term", "long term", "remember for later")
        has_write_intent = any(v in lowered for v in write_verbs)
        has_memory_ref = any(m in lowered for m in memory_markers)
        return (has_write_intent and has_memory_ref) or (
            "remember" in lowered and "later" in lowered
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


    async def _build_profile_response(self, state: Dict[str, Any]) -> Optional[str]:
        user_message = (state.get("user_input") or "").strip()
        lowered = user_message.lower()
        chat_history = list(state.get("chat_history", []) or [])
        profile_updates = self._extract_user_profile_updates(user_message)

        assistant_name = self._extract_assistant_identity_update(user_message)
        if assistant_name:
            return f"Understood. I will use {assistant_name} as my name."

        if self._is_assistant_name_question(lowered, chat_history):
            return f"My name is {await self._get_assistant_name()}."

        if profile_updates:
            return self._format_profile_update_ack(profile_updates)

        if self._is_profile_memory_request(lowered):
            return (
                "Understood. Share details like your name or role and I will store them for future reference."
            )

        if not self._is_user_profile_lookup_question(lowered):
            return None

        profile = await self._get_user_profile(state.get("user_id", ""))
        return self._format_user_profile_response(lowered, profile)

    @staticmethod
    def _is_introspection_query(user_message: str) -> bool:
        """Return True if the message is asking AIDEN about itself."""
        lowered = str(user_message or "").lower()

        # Decision / reasoning introspection
        decision_markers = (
            "why did you", "why did aiden", "why was",
            "why chose", "why choose", "why selected",
            "what did you decide", "last turn", "last decision",
            "previous response", "moral audit", "was it rejected",
            "was my response", "was the output",
        )
        if any(m in lowered for m in decision_markers):
            return True

        # Objective / backlog introspection
        backlog_markers = (
            "what tasks", "which tasks", "pending tasks", "deferred tasks",
            "blocked tasks", "show me tasks", "task backlog", "objectives",
            "what is in the backlog", "what's in the backlog",
            "show the backlog", "task queue",
        )
        if any(m in lowered for m in backlog_markers):
            return True

        # Energy introspection
        energy_markers = (
            "energy budget", "cognitive budget", "energy remaining",
            "how much energy", "why was task deferred", "why was it deferred",
            "why deferred", "energy state", "energy level",
        )
        if any(m in lowered for m in energy_markers):
            return True

        # Health / error introspection
        health_markers = (
            "system health", "recent errors", "any errors", "what went wrong",
            "synthesis history", "tool synthesis", "why wasn't the tool",
            "why didn't you create", "failed synthesis", "system log",
            "system errors",
        )
        if any(m in lowered for m in health_markers):
            return True

        return False

    @staticmethod

    def _classify_introspection_type(user_message: str) -> Optional[str]:
        """Classify what kind of introspection is being requested.

        Returns one of: 'decision', 'backlog', 'energy', 'health', None.
        Used by the fast-path to select the right introspection skill.
        """
        lowered = str(user_message or "").lower()

        # Energy must be checked before backlog to avoid "energy task" ambiguity
        if any(m in lowered for m in (
            "energy", "budget", "deferred", "why deferred",
        )):
            return "energy"

        if any(m in lowered for m in (
            "why did you", "why chose", "why selected", "last turn",
            "last decision", "moral audit", "rejected", "previous response",
        )):
            return "decision"

        if any(m in lowered for m in (
            "tasks", "backlog", "pending", "blocked", "objectives",
            "task queue",
        )):
            return "backlog"

        if any(m in lowered for m in (
            "health", "error", "errors", "synthesis", "tool creation",
            "system log", "what went wrong",
        )):
            return "health"

        return None

    @staticmethod

    def _introspection_backlog_status_filter(user_message: str) -> str:
        lowered = str(user_message or "").lower()
        if "deferred" in lowered:
            return "deferred"
        if "blocked" in lowered:
            return "blocked"
        if "pending" in lowered:
            return "pending"
        if "active" in lowered:
            return "active"
        return "all"

    @classmethod

    def _build_introspection_skill_call(
        cls,
        *,
        introspection_type: str,
        user_message: str,
        user_id: str,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        if introspection_type == "decision":
            return "query_decision_log", {
                "scope": "all",
                "limit": 5,
                "user_id": user_id or None,
            }
        if introspection_type == "backlog":
            return "query_objective_status", {
                "status_filter": cls._introspection_backlog_status_filter(user_message),
                "include_energy_context": True,
            }
        if introspection_type == "energy":
            return "query_energy_state", {
                "include_deferred_tasks": True,
                "include_blocked_tasks": True,
            }
        if introspection_type == "health":
            return "query_system_health", {
                "hours": 24,
                "include_synthesis_history": True,
            }
        return None

    @staticmethod

    def _build_introspection_format_messages(
        *,
        user_message: str,
        raw_result: str,
        core_mem_str: str,
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are AIDEN, a local autonomous AI agent. "
                    "The user has asked an introspective question about your own "
                    "state, decisions, or history. You have retrieved the relevant "
                    "data and must now present it in a clear, concise, natural "
                    "language response. "
                    "Rules: "
                    "1. Summarise the key facts — do not dump raw JSON. "
                    "2. For empty results, explain what the data means (e.g. 'no deferred tasks means energy is sufficient'). "
                    "3. Keep responses under 400 words. "
                    "4. Never fabricate data — only use what is in the skill output. "
                    "5. Do not include JSON, XML tags, or WORKERS: in your response.\n\n"
                    f"{core_mem_str}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question: {user_message}\n\n"
                    f"Introspection data (JSON):\n{raw_result}"
                ),
            },
        ]

    async def _execute_introspection_skill(
        self,
        *,
        user_message: str,
        introspection_type: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        skill_registry = self.cognitive_router.registry

        try:
            timeout = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "30.0"))
            user_id = str(state.get("user_id") or "")
            skill_call = self._build_introspection_skill_call(
                introspection_type=introspection_type,
                user_message=user_message,
                user_id=user_id,
            )
            if skill_call is None:
                return None

            skill_name, skill_args = skill_call
            fn = skill_registry.get_function(skill_name)
            if not fn:
                return None

            return await asyncio.wait_for(fn(**skill_args), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Introspection skill timed out for type=%s", introspection_type)
            return None
        except Exception as exc:
            logger.warning("Introspection skill error: %s", exc)
            return None

    async def _handle_introspection_fast_path(
        self,
        user_message: str,
        introspection_type: str,
        state: Dict[str, Any],
    ) -> Optional[str]:
        """Execute an introspection skill and format the result with System 1.

        Returns the formatted response string, or None if the skill fails
        or System 1 cannot produce a useful response.
        """
        logger.info(
            "Introspection fast-path triggered: type=%s message=%r",
            introspection_type,
            user_message[:80],
        )

        raw_result = await self._execute_introspection_skill(
            user_message=user_message,
            introspection_type=introspection_type,
            state=state,
        )

        if not raw_result:
            return None

        # Ask System 1 to format the raw skill output into a natural
        # language response. This is a lightweight formatting call —
        # no tool usage, just text transformation.
        core_mem_str = await self._safe_core_context_string()

        messages = self._build_introspection_format_messages(
            user_message=user_message,
            raw_result=raw_result,
            core_mem_str=core_mem_str,
        )

        result = await self._route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=45.0,
            context="introspection_fast_path",
        )

        if (
            result.status == "ok"
            and result.content
            and not result.content.startswith("[System 1 - Error]")
        ):
            return self.cognitive_router.sanitize_response(result.content)

        return None

    async def _maybe_introspection_meta_response(
        self,
        state: Dict[str, Any],
        user_message: str,
    ) -> Optional[str]:
        if not self._is_introspection_query(user_message):
            return None

        introspection_type = self._classify_introspection_type(user_message)
        if introspection_type is None:
            return None

        return await self._handle_introspection_fast_path(
            user_message=user_message,
            introspection_type=introspection_type,
            state=state,
        )

    async def _maybe_capability_meta_response(self, user_message: str) -> Optional[str]:
        if not self._should_use_intent_classifier(user_message):
            return None

        intent = await self._classify_user_intent(user_message)
        if intent != "capability_query":
            return None

        return self._build_capability_response(
            user_message,
            classified_intent=intent,
        )

    @staticmethod
    def _is_explicit_epic_request(user_message: str) -> bool:
        lowered = str(user_message or "").lower()
        return bool(
            re.search(
                r"\b(epic\s*:|create\s+epic|new\s+epic|define\s+epic|epic\s+goal)\b",
                lowered,
            )
        )

    def _should_invoke_goal_planner(self, user_message: str) -> bool:
        text = str(user_message or "").strip()
        if not text:
            return False

        if self._is_explicit_epic_request(text):
            return True

        complexity = self.routing_assessor._estimate_request_complexity(text)
        if complexity < _GOAL_PLANNER_COMPLEXITY_THRESHOLD:
            return False

        lowered = text.lower()
        planning_markers = (
            "plan",
            "roadmap",
            "phases",
            "milestones",
            "decompose",
            "break down",
            "architecture",
            "multi-step",
            "multi step",
            "workflow",
        )
        marker_hits = sum(1 for marker in planning_markers if marker in lowered)
        if marker_hits >= 2:
            return True

        action_markers = (
            "build",
            "implement",
            "launch",
            "ship",
            "refactor",
            "migrate",
        )
        has_action = any(marker in lowered for marker in action_markers)
        return has_action and bool(re.search(r"\b(and|then|after|before|plus)\b", lowered))

    @staticmethod

    def _build_goal_planning_context(state: Dict[str, Any]) -> str:
        history = list(state.get("chat_history", []) or [])
        if not history:
            return ""

        lines: List[str] = []
        for turn in history[-4:]:
            role = str(turn.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = re.sub(r"\s+", " ", str(turn.get("content") or "")).strip()
            if not content:
                continue
            if len(content) > 240:
                content = content[:237] + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    @staticmethod

    def _format_goal_planner_ack(result: PlanningResult) -> str:
        return (
            "Planning only complete. "
            f"Created Epic #{result.epic_id} ({result.epic_title}) with "
            f"{result.story_count} Stories and {result.task_count} Tasks. "
            "No task execution was performed."
        )

    async def _try_goal_planning_response(self, state: Dict[str, Any]) -> Optional[str]:
        user_id = str(state.get("user_id") or "")
        if user_id == "heartbeat":
            return None

        user_message = str(state.get("user_input") or "").strip()
        if not self._should_invoke_goal_planner(user_message):
            return None

        if not self.cognitive_router.get_system_2_available():
            logger.info("GoalPlanner skipped because System 2 is unavailable.")
            return None

        planning_context = self._build_goal_planning_context(state)

        async def _route(messages: List[Dict[str, str]]) -> RouterResult:
            return await self._route_to_system_2_redacted(
                messages,
                allowed_tools=[],
                purpose="goal_planner",
                allow_sensitive_context=False,
            )

        try:
            result = await self.goal_planner.plan_goal(
                user_message,
                context=planning_context,
                route_to_system_2=_route,
                ledger_memory=self.ledger_memory,
                redactor=cloud_redaction.redact_text_for_cloud,
                origin=f"User:{user_id}",
            )
        except Exception as e:
            logger.warning(f"GoalPlanner failed; falling back to normal execution path: {e}")
            return None

        logger.info(
            "GoalPlanner created planning tree for user %s: epic=%s stories=%s tasks=%s",
            user_id,
            result.epic_id,
            result.story_count,
            result.task_count,
        )
        _gp_emitter = get_current_emitter()
        if _gp_emitter is not None:
            await _gp_emitter.emit(ProgressEvent.goal_planning())
        return self._format_goal_planner_ack(result)

    async def _try_meta_fast_path_response(self, state: Dict[str, Any]) -> Optional[str]:
        user_message = state.get("user_input", "")
        chat_history = list(state.get("chat_history", []) or [])

        profile_response = await self._build_profile_response(state)
        if profile_response:
            return profile_response

        history_response = self._meta_history_response(user_message, chat_history)
        if history_response:
            return history_response

        if self._is_skill_list_request(user_message):
            return self._build_compact_skill_list_response()

        # ── Fast-path: introspection queries ──────────────────────────
        # These are answered by calling introspection skills directly
        # without supervisor planning overhead. Complexity is bounded
        # because the skills are pure DB reads (no LLM iteration needed).
        introspection_result = await self._maybe_introspection_meta_response(
            state,
            user_message,
        )
        if introspection_result is not None:
            return introspection_result
        # ── End fast-path introspection ───────────────────────────────

        capability_response = await self._maybe_capability_meta_response(user_message)
        if capability_response:
            return capability_response

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

        if self._state_has_audio_prompt(state):
            logger.info("Fast path bypassed: multimodal audio prompt detected.")
            return None

        meta_response = await self._try_meta_fast_path_response(state)
        if meta_response:
            return meta_response

        assessment = self.routing_assessor.assess_request_route(
            user_message, state.get("chat_history", [])
        )
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
            return await self._fast_path_direct_mode(
                state, user_message, core_mem_str, capabilities_str, history_msgs
            )

        if assessment["mode"] == "multi_ticker":
            return await self._fast_path_multi_ticker_mode(
                state, user_message, core_mem_str, assessment
            )

        if assessment["mode"] != "single_tool":
            return None

        return await self._fast_path_single_tool_mode(
            state, user_message, core_mem_str, capabilities_str, assessment
        )

    async def _fast_path_direct_mode(
        self,
        state: Dict[str, Any],
        user_message: str,
        core_mem_str: str,
        capabilities_str: str,
        history_msgs: List[Dict[str, str]],
    ) -> Optional[str]:
        deferred = await self._try_ad_hoc_dispatch_energy_gate(
            state,
            dispatch_context="fast_path_direct",
        )
        if deferred is not None:
            return deferred
        state["_energy_gate_cleared"] = True

        archival_ctx = ""
        if hasattr(self, "vector_memory"):
            archival_ctx = await self._get_archival_context(user_message)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are AIDEN. This request is low complexity. Reply directly in 1-2 concise sentences. "
                    "Use the recent chat history and core memory when they are relevant. "
                    "Answer capability questions based on the provided capabilities instead of claiming you have none. "
                    "IMPORTANT: Do not fabricate time-sensitive data (weather, live prices, current news, live scores). "
                    "If a request needs real-time data you cannot provide, say so clearly. "
                    "Do not call tools and do not mention internal routing.\n\n"
                    f"{core_mem_str}\n\n{capabilities_str}"
                    + (f"\n\n{archival_ctx}" if archival_ctx else "")
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
        if result.status == "ok" and result.content and not result.content.startswith(_SYSTEM_1_ERROR_PREFIX):
            return result.content.strip()
        return None

    async def _fast_path_multi_ticker_mode(
        self,
        state: Dict[str, Any],
        user_message: str,
        core_mem_str: str,
        assessment: Dict[str, Any],
    ) -> Optional[str]:
        deferred = await self._try_ad_hoc_dispatch_energy_gate(
            state,
            dispatch_context="fast_path_multi_ticker",
        )
        if deferred is not None:
            return deferred
        state["_energy_gate_cleared"] = True

        tickers = assessment["tickers"]
        ticker_results = []
        for ticker in tickers:
            tr = await self.cognitive_router._execute_tool(
                "get_stock_price", {"ticker": ticker}
            )
            if tr.status == "ok":
                ticker_results.append(f"[{ticker}]\n{tr.content}")
        if not ticker_results:
            return None
        combined = "\n\n".join(ticker_results)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are AIDEN. Multiple stock prices have been fetched. "
                    "Present each result clearly, one per line, in a human-readable format. "
                    "Do not call further tools.\n\n"
                    f"{core_mem_str}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request: {user_message}\n\n"
                    f"Tool results:\n{combined}"
                ),
            },
        ]
        result = await self._route_to_system_1(
            messages,
            allowed_tools=[],
            context="fast_path_multi_ticker",
        )
        if result.status == "ok" and result.content and not result.content.startswith(_SYSTEM_1_ERROR_PREFIX):
            return result.content.strip()
        return combined

    async def _fast_path_single_tool_mode(
        self,
        state: Dict[str, Any],
        user_message: str,
        core_mem_str: str,
        capabilities_str: str,
        assessment: Dict[str, Any],
    ) -> Optional[str]:
        tool_name = assessment["tool_name"]
        deferred = await self._try_ad_hoc_dispatch_energy_gate(
            state,
            dispatch_context=f"fast_path_single_tool:{tool_name}",
        )
        if deferred is not None:
            return deferred
        state["_energy_gate_cleared"] = True

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
                    "Reply in 1-3 sentences unless a list or table is clearly more appropriate. "
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
        if result.status == "ok" and result.content and not result.content.startswith(_SYSTEM_1_ERROR_PREFIX):
            return result.content.strip()
        return tool_result.content.strip()

    _CHARTER_FALLBACK = "Core Directive: Do no harm."

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
        merged.setdefault("user_prompt", {})
        merged.setdefault("chat_history", [])
        return merged


    def _get_capabilities_string(self) -> str:
        cached_value = getattr(self, "_capabilities_string_cache", None)
        if cached_value is not None:
            return cached_value
        rows = self._build_capability_catalog_rows()
        if not rows:
            result = "Available skills catalog (name: description): none loaded"
        else:
            result = ("Available skills catalog (name: description):\n"
                      + "\n".join(rows))
        self._capabilities_string_cache = result
        return result

    def _invalidate_capabilities_cache(self) -> None:
        self._capabilities_string_cache = None

    def _build_capability_catalog_rows(self) -> List[str]:
        rows: List[str] = []
        for item in self._load_capability_catalog_entries():
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "").strip()
            if name and description:
                rows.append(f"- {name}: {description}")
        return rows

    def _load_capability_catalog_entries(self) -> List[Dict[str, str]]:
        catalog_getter = getattr(self.cognitive_router.registry, "get_skill_catalog", None)
        if callable(catalog_getter):
            try:
                raw_catalog = catalog_getter()
                if isinstance(raw_catalog, list):
                    return [item for item in raw_catalog if isinstance(item, dict)]
            except Exception as exc:
                logger.warning("Failed to load skill catalog from registry: %s", exc)

        # Backward-compatible fallback for tests/mocks that still expose get_schemas only.
        fallback: List[Dict[str, str]] = []
        for schema in self.cognitive_router.registry.get_schemas():
            fallback.append(
                {
                    "name": str(schema.get("name") or "").strip(),
                    "description": str(schema.get("description") or "").strip(),
                }
            )
        return fallback

    def _load_executable_capability_catalog_entries(self) -> List[Dict[str, str]]:
        registry = getattr(self.cognitive_router, "registry", None)
        if registry is not None:
            executable_getter = getattr(registry, "get_executable_skill_catalog", None)
            if callable(executable_getter):
                try:
                    entries = executable_getter()
                    if isinstance(entries, list):
                        return [item for item in entries if isinstance(item, dict)]
                except Exception as exc:
                    logger.warning("Failed to load executable skill catalog from registry: %s", exc)
        return self._load_capability_catalog_entries()

    def _build_scoped_skill_runtime_context(self, skill_name: str) -> str:
        """Load full SKILL.md body on demand for the current execution turn only."""
        registry = getattr(self.cognitive_router, "registry", None)
        if registry is None:
            return ""

        get_skill_body = getattr(registry, "get_skill_body", None)
        if not callable(get_skill_body):
            return ""

        raw_body = str(get_skill_body(skill_name) or "").strip()
        if not raw_body:
            return ""

        description = ""
        for item in self._load_capability_catalog_entries():
            if str(item.get("name") or "").strip() == skill_name:
                description = str(item.get("description") or "").strip()
                break

        return (
            "<scoped_skill_context>\n"
            f"Skill: {skill_name}\n"
            f"Description: {description}\n"
            "Scope: immediate execution turn only\n\n"
            "SKILL_BODY:\n"
            f"{raw_body}\n"
            "</scoped_skill_context>"
        )

    @staticmethod

    def _capability_catalog_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(_ROUTING_TOKEN_RE, str(text or "").lower())
            if len(token) > 2 and token not in _CATALOG_MATCH_STOPWORDS
        }

    def _find_local_skill_catalog_match(
        self,
        gap_description: str,
        suggested_tool_name: str,
    ) -> Optional[Dict[str, str]]:
        query_tokens = self._capability_catalog_tokens(gap_description)
        query_tokens |= self._capability_catalog_tokens(str(suggested_tool_name or "").replace("_", " "))
        if not query_tokens:
            return None

        best_item: Optional[Dict[str, str]] = None
        best_score = 0.0

        for item in self._load_executable_capability_catalog_entries():
            name = str(item.get("name") or "").strip()
            if not name or name in _CATALOG_META_TOOL_NAMES:
                continue

            name_tokens = self._capability_catalog_tokens(name.replace("_", " "))
            desc_tokens = self._capability_catalog_tokens(item.get("description", ""))

            score = 0.0
            if suggested_tool_name and name.lower() == str(suggested_tool_name).strip().lower():
                score += 6.0
            score += 3.0 * len(query_tokens & name_tokens)
            score += 1.5 * len(query_tokens & desc_tokens)

            if score > best_score:
                best_item = {"name": name, "description": str(item.get("description") or "").strip()}
                best_score = score

        return best_item if best_score >= 1.5 else None

    async def _try_resolve_capability_gap_locally(
        self,
        user_id: str,
        result: RouterResult,
        state: Dict[str, Any],
    ) -> Optional[str]:
        if not bool(getattr(self, "_enable_local_skill_discovery_gate", True)):
            return None

        local_match = self._find_local_skill_catalog_match(
            result.gap_description,
            result.suggested_tool_name,
        )
        if local_match is None:
            return None

        matched_name = local_match["name"]
        matched_description = local_match["description"]
        user_request = str(state.get("user_input") or result.gap_description or "").strip()

        logger.info(
            "Capability gap pre-check matched local skill '%s' for user %s; attempting local recovery.",
            matched_name,
            user_id,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "A request_capability escalation was triggered, but a matching local skill already exists.\n"
                    f"Matched skill: {matched_name}\n"
                    f"Skill purpose: {matched_description}\n"
                    "Use that existing skill to solve the request now. "
                    "Do not call request_capability for this request."
                ),
            },
            {
                "role": "user",
                "content": user_request,
            },
        ]

        local_result = await self._route_to_system_1(
            messages,
            allowed_tools=[matched_name],
            deadline_seconds=45.0,
            context="capability_gap_local_recovery",
        )

        if local_result.status in {"mfa_required", "hitl_required"}:
            return await self._handle_blocked_result(local_result, user_id, state)

        if local_result.status == "ok" and local_result.content and not self._is_system_1_error(local_result):
            return self.cognitive_router.sanitize_response(local_result.content)

        logger.warning(
            "Local capability recovery with '%s' did not resolve the request; continuing to synthesis path.",
            matched_name,
        )
        return None


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

    @staticmethod

    def _build_charter_tier_cache(charter_text: str) -> Dict[str, str]:
        raw_charter = str(charter_text or "").strip()
        if not raw_charter:
            return {_CHARTER_PARSE_ERROR_KEY: "[MALFORMED_CHARTER_XML: empty charter]"}

        try:
            root = ET.fromstring(raw_charter)
        except ET.ParseError as exc:
            return {_CHARTER_PARSE_ERROR_KEY: f"[MALFORMED_CHARTER_XML: {exc}]"}

        cache: Dict[str, str] = {}
        for tier_tag in _CHARTER_TIER_TAGS:
            target = Orchestrator._find_xml_element(root, tier_tag)
            if target is None:
                cache[tier_tag] = f"[MISSING_CHARTER_TIER: {tier_tag}]"
            else:
                cache[tier_tag] = Orchestrator._serialize_xml_inner(target)
        return cache

    def _ensure_charter_tier_cache(self) -> Dict[str, str]:
        cached = getattr(self, "_charter_tier_blocks", None)
        if isinstance(cached, dict) and cached:
            return cached
        rebuilt = self._build_charter_tier_cache(getattr(self, "charter_text", ""))
        self._charter_tier_blocks = rebuilt
        return rebuilt

    def _is_fallback_charter_active(self) -> bool:
        return str(getattr(self, "charter_text", "") or "").strip() == self._CHARTER_FALLBACK

    def _extract_charter_tier_block(self, tier_tag: str) -> str:
        cache = self._ensure_charter_tier_cache()
        parse_error = str(cache.get(_CHARTER_PARSE_ERROR_KEY) or "").strip()
        if parse_error:
            return parse_error
        return str(cache.get(tier_tag) or f"[MISSING_CHARTER_TIER: {tier_tag}]")

    @staticmethod

    def _extract_charter_tier_block_from_text(charter_text: str, tier_tag: str) -> str:
        cache = Orchestrator._build_charter_tier_cache(charter_text)
        parse_error = str(cache.get(_CHARTER_PARSE_ERROR_KEY) or "").strip()
        if parse_error:
            return parse_error
        return str(cache.get(tier_tag) or f"[MISSING_CHARTER_TIER: {tier_tag}]")

    @staticmethod
    def _find_xml_element(root: ET.Element, tag_name: str) -> Optional[ET.Element]:
        expected = str(tag_name or "").strip().lower()
        for element in root.iter():
            if Orchestrator._strip_xml_namespace(element.tag).lower() == expected:
                return element
        return None

    @staticmethod

    def _serialize_xml_inner(element: ET.Element) -> str:
        parts: List[str] = []
        if element.text and element.text.strip():
            parts.append(element.text.strip())
        for child in element:
            parts.append(ET.tostring(child, encoding="unicode").strip())
        return _MULTI_BLANK_LINES_RE.sub("\n\n", "\n".join(part for part in parts if part).strip())

    @staticmethod

    def _strip_xml_namespace(tag: str) -> str:
        return str(tag or "").rsplit("}", 1)[-1]

# Compatibility alias for extracted static helpers that reference Orchestrator explicitly.
Orchestrator = _FastPathMixin
