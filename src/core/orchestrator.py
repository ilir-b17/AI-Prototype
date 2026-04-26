"""
Orchestrator Module - Implements State Graph Architecture.

State graph with energy budgets, proactive heartbeat execution, and charter
enforcement. A state dictionary is routed through supervisor, workers, and
critic until complete, blocked for HITL/MFA, or exhausted.

Energy model reference (operator-facing):
- Sources:
    - Per-turn state budget in state["energy_remaining"] for the current user turn.
    - Cross-turn shared budget account used for
        pacing ad-hoc and heartbeat execution.
- Sinks:
    - _deduct_energy charges per-turn budget for supervisor/worker/critic/tool work.
    - _try_reserve_predictive_energy_budget reserves predictive points before ad-hoc
        and heartbeat dispatch.
- Replenishment:
    - Per-turn replenishment is recalculated on each process_message turn.
    - Predictive wall-clock replenishment accrues by ENERGY_REPLENISH_PER_HOUR.
    - Heartbeat cadence replenishes predictive budget by
        ENERGY_REPLENISH_PER_HEARTBEAT at cycle start.
- Refunds and reversals:
    - _refund_energy reverses per-turn deductions after failures/timeouts.
    - _refund_predictive_energy_budget restores predictive points for failed/blocked
        branches, reservation races, and successful completion paths.
- Deferrals and retry pacing:
    - If ROI/reserve checks fail, tasks are marked deferred_due_to_energy.
    - defer_count and next_eligible_at are persisted for cooldown-aware fairness
        and re-evaluation.
- Completion bonuses:
    - Completing tasks via update_objective_status can apply predictive completion
        refunds (reward policy), currently implemented as a bounded completion bonus
        on successful completion updates.

HITL retry contract: Maximum 3 critic rejections per HITL cycle,
maximum 3 HITL cycles per task, after which the task is abandoned.
"""

import os
import platform
import logging
import json
import hashlib
import asyncio
import inspect
import math
import re
import time
import weakref
from collections import OrderedDict
import xml.etree.ElementTree as ET
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
from src.core import cloud_redaction
from src.core.synthesis_pipeline import SynthesisPipeline
from src.core.routing_assessor import RoutingAssessor, _FAST_PATH_SINGLE_TOOL_ALLOWLIST, _ROUTING_STOPWORDS
from src.core.routing_assessor import _ROUTING_TOKEN_RE
from src.core.agent_definition import AgentDefinition
from src.core.agent_registry import AgentRegistry
from src.core.llm_router import CognitiveRouter, RouterResult, RequiresHITLError, RequiresMFAError
from src.core.prompt_config import load_prompt_config, build_supervisor_prompt, build_supervisor_turn_context
from src.core.runtime_context import set_runtime_context
from src.core.state_model import AgentState, normalize_state
from src.core.workflow_graph import build_orchestrator_graph
from src.core.security import validate_mfa_configuration, verify_mfa_challenge
from src.core.nocturnal_consolidation import NocturnalConsolidationSlice1
from src.core.goal_planner import GoalPlanner, PlanningResult
from src.core.energy_judge import EnergyJudge, EnergyEvaluation
from src.core.energy_roi_engine import EnergyROIEngine, EnergyDecision
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

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

from src.core.progress import (
    ProgressEmitter,
    ProgressEvent,
    ProgressCallback,
    get_current_emitter,
    reset_emitter,
    set_current_emitter,
)

logger = logging.getLogger(__name__)


class _UserLockSlot:
    def __init__(self) -> None:
        self.lock: asyncio.Lock = asyncio.Lock()
        self.in_flight: int = 0


class _UserLockLease:
    def __init__(self, owner: "Orchestrator", user_id: str, slot: _UserLockSlot) -> None:
        self._owner = owner
        self._user_id = user_id
        self._slot = slot
        self._acquired = False
        self._released = False

    async def __aenter__(self) -> asyncio.Lock:
        try:
            await self._slot.lock.acquire()
        except BaseException:
            await self.aclose()
            raise
        self._acquired = True
        return self._slot.lock

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._acquired:
            self._slot.lock.release()
            self._acquired = False
        await self.aclose()

    async def aclose(self) -> None:
        if self._released:
            return
        self._released = True
        await self._owner._release_user_lock_lease(self._user_id, self._slot)

# Energy costs per operation
ENERGY_COST_SUPERVISOR = 10
ENERGY_COST_WORKER = 15
ENERGY_COST_CRITIC = 10
ENERGY_COST_TOOL = 5
# When the remaining budget exceeds this floor and the only reason for deferral is
# a borderline ROI ("ROI too low"), ad-hoc conversational requests are approved
# unconditionally so that energy accounting never blocks normal user interaction
# while resources are plentiful.  Configurable via AD_HOC_COMFORTABLE_BUDGET_FLOOR.
_AD_HOC_COMFORTABLE_BUDGET_FLOOR = int(os.getenv("AD_HOC_COMFORTABLE_BUDGET_FLOOR", "50"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))  # default 30 minutes
MEMORY_SAVE_THRESHOLD = int(os.getenv("MEMORY_SAVE_THRESHOLD", "120"))
MEMORY_CONSOLIDATION_INTERVAL = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "21600"))  # 6 hours
# How long an unanswered MFA/HITL/tool-approval entry stays alive before auto-expiry
_PENDING_STATE_TTL_SECONDS = int(os.getenv("PENDING_STATE_TTL_SECONDS", "86400"))  # 24 hours
_CONSOLIDATION_TRIGGER_TURNS = int(os.getenv("CONSOLIDATION_TRIGGER_TURNS", "10"))
_SYNTHESIS_LOCKOUT_TTL_SECONDS = int(os.getenv("SYNTHESIS_LOCKOUT_TTL_SECONDS", "600"))
_SYSTEM_1_ERROR_PREFIX = "[System 1 - Error]"
_MORAL_TRIVIALITY_READ_ONLY_HINTS = (
    "time",
    "date",
    "timezone",
    "weather",
    "stock",
    "price",
    "score",
    "headline",
    "system info",
)
_MORAL_TRIVIALITY_BLOCK_HINTS = (
    "modify",
    "delete",
    "write",
    "update_",
    "request_capability",
    "escalate_to_system_2",
    "run_terminal_command",
    "execute_python_sandbox",
    "manage_file_system",
    "spawn_new_objective",
    "approval",
    "mfa",
    "deploy",
)
_RECENT_CHAT_HISTORY_LIMIT = 12
_CRITIC_SHORT_OUTPUT_THRESHOLD = 220
_SYSTEM_2_ERROR_PREFIX = "[System 2 - Error]"
_SYSTEM_2_EMPTY_PREFIX = "[System 2 - No Response]"
_GOAL_PLANNER_COMPLEXITY_THRESHOLD = int(os.getenv("GOAL_PLANNER_COMPLEXITY_THRESHOLD", "4"))
_HEARTBEAT_FAILURE_STRIKES = int(os.getenv("HEARTBEAT_FAILURE_STRIKES", "3"))
_HEARTBEAT_FAILURE_STATE_KEY = "heartbeat_task_failure_counts"
HEARTBEAT_TASK_PREFIX_FMT = "[HEARTBEAT TASK #{task_id}]"
_INTENT_CLASSIFIER_CACHE_TTL_SECONDS = 300.0
_INTENT_CLASSIFIER_CACHE_MAX_SIZE_DEFAULT = 256
_INTENT_CLASSIFIER_TIMEOUT_SECONDS = 10.0
_INTENT_CLASSIFIER_MAX_OUTPUT_TOKENS = 60
_USER_INTENTS = frozenset({"capability_query", "task", "profile_update", "casual"})
_HEARTBEAT_REPLENISH_ENV_DEFAULT = 2
_CONSOLIDATION_TURN_COUNT_MAX_USERS = 100
_EXPLICIT_MEMORY_REQUEST_RE = re.compile(
    r"\b(?:remember\s+that|remember\s+my|don'?t\s+forget|please\s+remember|save\s+that|note\s+that)\b",
    flags=re.IGNORECASE,
)
_VOICE_NOTE_PLACEHOLDER_RE = re.compile(r"^\[Voice note · \d+ bytes · [^\]]+\]$")
_CHARTER_TIER_TAGS = ("Tier_1_Axioms", "Tier_2_Strategic", "Tier_3_Operational")
_CHARTER_PARSE_ERROR_KEY = "__charter_parse_error__"
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Key used in the state dict to signal a blocked (non-ok) router result
_BLOCKED_KEY = "_blocked_result"
_CATALOG_MATCH_STOPWORDS = {
    "a", "an", "and", "the", "to", "for", "of", "in", "on", "or", "is", "are",
    "be", "can", "could", "would", "should", "have", "has", "with", "new", "needed",
    "need", "admin", "user", "function", "tool", "tools", "skill", "skills",
    "capability", "capabilities", "allow", "provide", "more", "than", "just", "use",
    "using", "this", "that", "from", "missing", "cannot", "couldnt", "unable",
}
_CATALOG_META_TOOL_NAMES = {
    "request_capability",
    "request_core_update",
    "ask_admin_for_guidance",
    "escalate_to_system_2",
}


def _format_voice_placeholder(audio_bytes: bytes, mime_type: str) -> str:
    safe_mime_type = str(mime_type or "audio/ogg").strip() or "audio/ogg"
    return f"[Voice note · {len(audio_bytes)} bytes · {safe_mime_type}]"


def _is_voice_placeholder_text(text: str) -> bool:
    return bool(_VOICE_NOTE_PLACEHOLDER_RE.fullmatch(str(text or "").strip()))


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

    _deprecated_energy_replenish_turn_warning_logged = False

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
            self.routing_assessor = RoutingAssessor(self.cognitive_router.registry)
            self._capabilities_string_cache: Optional[str] = None
            self._enable_local_skill_discovery_gate = (
                str(os.getenv("ENABLE_LOCAL_SKILL_DISCOVERY_GATE", "true")).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            self.cognitive_router._skill_context_resolver = self._build_scoped_skill_runtime_context
            self.prompt_config = load_prompt_config()
            self.agent_registry = AgentRegistry()
            self.nocturnal_consolidation = NocturnalConsolidationSlice1(
                min_chars=int(os.getenv("NOCTURNAL_MIN_CHARS", "30")),
                batch_semantic_jaccard_threshold=float(os.getenv("NOCTURNAL_BATCH_DEDUP_THRESHOLD", "0.82")),
                vector_distance_threshold=float(os.getenv("NOCTURNAL_VECTOR_DEDUP_DISTANCE", "0.08")),
            )
            self.goal_planner = GoalPlanner(
                max_context_chars=int(os.getenv("GOAL_PLANNER_CONTEXT_CHARS", "1600"))
            )
            self.energy_judge = EnergyJudge()
            self.energy_roi_engine = EnergyROIEngine()
            # Single energy account — drives both fast-path gating and graph node costs.
            _initial_budget = max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
            self._energy_budget: int = _initial_budget
            self._energy_budget_lock: asyncio.Lock = asyncio.Lock()
            self._warn_deprecated_energy_replenish_turn_env_once()

            self.charter_text = self._load_charter()
            self._charter_tier_blocks = self._build_charter_tier_cache(self.charter_text)
            self._ready: asyncio.Event = asyncio.Event()
            self.pending_mfa: Dict[str, dict] = {}
            self.pending_hitl_state: Dict[str, dict] = {}
            self.pending_tool_approval: Dict[str, dict] = {}
            self._admin_user_id = str(os.getenv("ADMIN_USER_ID", "")).strip()
            self._synthesis_in_progress: Dict[str, float] = {}
            self._consolidation_turn_counts: "OrderedDict[str, int]" = OrderedDict()
            self._intent_classification_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
            self.outbound_queue: Optional[asyncio.Queue] = None
            self.sensory_state: Dict[str, str] = {}
            # Background task registry — holds strong references to prevent GC (ISSUE-002)
            self._background_tasks: set = set()
            # Per-user async locks to serialise concurrent messages (ISSUE-012)
            self._user_locks: Dict[str, weakref.ReferenceType[_UserLockSlot]] = {}
            self._user_locks_lock: asyncio.Lock = asyncio.Lock()
            self._compiled_graph = None
            self._heartbeat_failure_counts: Dict[int, int] = {}
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
        self.cognitive_router.set_dynamic_tool_restart_callback(
            self._reload_dynamic_tools_after_worker_restart
        )
        await self.cognitive_router.start_dynamic_tool_worker()

        # Register Groq cooldown persistence callback and reload any saved cooldown
        def _save_groq_cooldown(expiry: float) -> None:
            self._fire_and_forget(
                self.ledger_memory.set_system_state("groq_cooldown_until", str(expiry))
            )

        self.cognitive_router._persist_cooldown_cb = _save_groq_cooldown
        await self._restore_persisted_groq_cooldown()

        await self.ledger_memory.seed_initial_goals()
        await self._load_approved_tools()
        await self._load_pending_approvals()
        await self._load_pending_mfa()
        await self._load_pending_hitl()
        await self._restore_heartbeat_failure_counts()
        if getattr(self.cognitive_router, "_preload_system_1_on_startup", False):
            try:
                await self.cognitive_router.preload_system_1()
            except Exception as _preload_err:
                logger.warning("System 1 preload skipped: %s", _preload_err)
        self._compiled_graph = build_orchestrator_graph(self)
        await self._restore_consolidation_turn_counts()
        set_runtime_context(self.ledger_memory, self.core_memory, self.vector_memory, self)
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

        self.synthesis_pipeline = SynthesisPipeline(
            cognitive_router=self.cognitive_router,
            ledger_memory=self.ledger_memory,
            core_memory=self.core_memory,
            pending_tool_approval=self.pending_tool_approval,
            on_tools_changed=self._invalidate_capabilities_cache,
            process_message_fn=self.process_message,
            route_to_system_2_redacted=self._route_to_system_2_redacted,
            fire_and_forget=self._fire_and_forget,
            notify_admin=self._notify_admin,
            outbound_queue=self.outbound_queue,
        )
        self.synthesis_pipeline._synthesis_in_progress = self._synthesis_in_progress

        self._enforce_charter_policy()
        self._ready.set()
        logger.info("Orchestrator async_init complete")

    async def _restore_persisted_groq_cooldown(self) -> None:
        saved_cooldown = None
        try:
            saved_cooldown = await self.ledger_memory.get_system_state("groq_cooldown_until")
        except Exception as _ce:
            await self._warn_groq_cooldown_restore_failed(f"DB read failed: {_ce}")
            return

        if not saved_cooldown:
            return

        try:
            expiry = float(saved_cooldown)
            now = time.time()
            max_expiry = now + 7200.0
            if not math.isfinite(expiry):
                await self._warn_groq_cooldown_restore_failed(
                    f"Invalid non-finite groq_cooldown_until value: {saved_cooldown!r}"
                )
            elif now < expiry <= max_expiry:
                self.cognitive_router._system2_cooldown_until = expiry
                logger.info(
                    "Restored Groq rate-limit cooldown: %.0fs remaining.",
                    expiry - now,
                )
            elif expiry > max_expiry:
                await self._warn_groq_cooldown_restore_failed(
                    "Persisted Groq cooldown exceeded 2-hour cap; ignoring."
                )
        except (TypeError, ValueError) as e:
            await self._warn_groq_cooldown_restore_failed(
                f"Invalid groq_cooldown_until value in DB: {e}"
            )

    async def _warn_groq_cooldown_restore_failed(self, reason: str) -> None:
        message = f"WARNING: Groq cooldown restoration failed during startup: {reason}"
        logger.warning(message)
        try:
            await self._notify_admin(message)
        except Exception as notify_error:
            logger.warning("Failed to notify admin about Groq cooldown restoration failure: %s", notify_error)

    def _enforce_charter_policy(self) -> None:
        validate_mfa_configuration()
        allow_missing = os.getenv("ALLOW_MISSING_CHARTER", "false").strip().lower() in {"1", "true", "yes"}
        charter_cache = self._ensure_charter_tier_cache()
        parse_error = str(charter_cache.get(_CHARTER_PARSE_ERROR_KEY) or "").strip()

        if parse_error and self.charter_text != self._CHARTER_FALLBACK:
            if not allow_missing:
                raise RuntimeError(
                    "FATAL: charter.md is malformed. The moral evaluation framework cannot operate. "
                    "Fix the charter XML (Tier_1_Axioms/Tier_2_Strategic/Tier_3_Operational) or set "
                    "ALLOW_MISSING_CHARTER=true to explicitly permit degraded operation."
                )
            logger.warning(
                "SECURITY: charter.md is malformed (%s). Falling back to minimal charter due to "
                "ALLOW_MISSING_CHARTER=true.",
                parse_error,
            )
            self.charter_text = self._CHARTER_FALLBACK
            self._charter_tier_blocks = self._build_charter_tier_cache(self.charter_text)

        if self.charter_text == self._CHARTER_FALLBACK:
            if not allow_missing:
                raise RuntimeError(
                    "FATAL: charter.md not found or empty. The moral evaluation framework cannot operate. "
                    "Place a valid charter.md in the working directory or set ALLOW_MISSING_CHARTER=true "
                    "to explicitly permit degraded operation."
                )
            logger.warning(
                "SECURITY: Running without a full charter. Charter enforcement will be minimal."
            )

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

    @staticmethod
    def _extract_audio_bytes(payload: Dict[str, Any]) -> bytes:
        if not isinstance(payload, dict):
            return b""
        raw = payload.get("audio_bytes")
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, bytearray):
            return bytes(raw)
        if isinstance(raw, memoryview):
            return raw.tobytes()
        return b""

    @classmethod
    def _coerce_user_prompt_payload(cls, user_message: Any) -> Dict[str, Any]:
        if isinstance(user_message, dict):
            payload = dict(user_message)
            text = str(
                payload.get("text")
                or payload.get("user_input")
                or payload.get("content")
                or ""
            ).strip()
            audio_bytes = cls._extract_audio_bytes(payload)
            audio_mime_type = str(payload.get("audio_mime_type") or "audio/ogg")
            if not text and audio_bytes:
                text = _format_voice_placeholder(audio_bytes, audio_mime_type)

            normalized: Dict[str, Any] = {
                "text": text,
                "audio_bytes": audio_bytes,
                "audio_mime_type": audio_mime_type,
                "audio_source": str(payload.get("audio_source") or ""),
                "audio_file_id": str(payload.get("audio_file_id") or ""),
            }
            return normalized

        text = str(user_message or "").strip()
        return {
            "text": text,
            "audio_bytes": b"",
            "audio_mime_type": "",
            "audio_source": "",
            "audio_file_id": "",
        }

    @classmethod
    def _state_has_audio_prompt(cls, state: Dict[str, Any]) -> bool:
        prompt = state.get("user_prompt")
        if not isinstance(prompt, dict):
            return False
        return bool(cls._extract_audio_bytes(prompt))

    @classmethod
    def _build_user_prompt_message(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        user_text = str(state.get("user_input") or "")
        message: Dict[str, Any] = {
            "role": "user",
            "content": f"<user_input>{user_text}</user_input>",
        }
        prompt = state.get("user_prompt")
        if not isinstance(prompt, dict):
            return message

        audio_bytes = cls._extract_audio_bytes(prompt)
        if audio_bytes:
            message["audio_bytes"] = audio_bytes
            message["audio_mime_type"] = str(prompt.get("audio_mime_type") or "audio/ogg")
        return message

    @classmethod
    def _strip_audio_bytes_for_persistence(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = normalize_state(dict(state))
        prompt = sanitized.get("user_prompt")
        if isinstance(prompt, dict) and "audio_bytes" in prompt:
            prompt = dict(prompt)
            prompt.pop("audio_bytes", None)
            sanitized["user_prompt"] = prompt
        sanitized.pop("_energy_gate_cleared", None)
        return sanitized

    @staticmethod
    def _normalize_display_name(raw_name: str) -> str:
        cleaned = re.sub(r"\s+", " ", raw_name).strip(" ,.!?")
        return " ".join(part.capitalize() for part in cleaned.split())

    @staticmethod
    def _extract_user_profile_updates(user_message: str) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        text = (user_message or "").strip()
        lowered = text.lower()

        # Matches "my name is <Name>"; spaces are allowed in the name, and trailing punctuation or end-of-string is required. capitalize() is used, so mixed-case surnames like "McCaffrey" normalize to "Mccaffrey" (known limitation).
        name_match = re.search(
            r"\bmy name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if name_match:
            raw_name = Orchestrator._normalize_display_name(name_match.group(1))
            if raw_name:
                updates["name"] = raw_name

        # Matches "<Name> is me"; question forms are intentionally not matched.
        identity_match = re.search(
            r"(?:^|[.!?,]\s*)([a-z][a-z' -]{0,60})\s+is me\b(?=,|\.|!|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if identity_match and "name" not in updates:
            raw_name = Orchestrator._normalize_display_name(identity_match.group(1))
            if raw_name:
                updates["name"] = raw_name

        age_match = re.search(r"\b(?:i am|i'm)\s+(\d{1,3})\s*years?\s*old\b", lowered)
        if age_match:
            updates["age"] = int(age_match.group(1))

        if re.search(r"\bi am(?: also)? your developer\b", lowered):
            updates["relationship"] = "developer"

        return updates

    @staticmethod
    def _extract_assistant_identity_update(user_message: str) -> Optional[str]:
        text = (user_message or "").strip()
        if not text:
            return None

        # Polite request patterns that may include "?" — must be checked before the "?" guard.
        # e.g. "Can I call you Aiden?", "I'll call you Aiden", "Let's call you Aiden"
        call_patterns = (
            r"\b(?:can i|may i|i(?:'ll| will| want to)|let(?:'s| us)) call you\s+([a-z][a-z' -]{0,40})(?:\?|,|\.|!|$)",
            r"\bi(?:'ll| will) refer to you as\s+([a-z][a-z' -]{0,40})(?:\?|,|\.|!|$)",
        )
        for pattern in call_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return Orchestrator._normalize_display_name(match.group(1))

        # Statement patterns must not be questions to avoid false positives like "Is your name X?"
        if "?" in text:
            return None

        statement_patterns = (
            r"\b(?:just for your information,\s*)?your name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|$)",
            r"\bi am informing you(?: that)? your name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|$)",
        )
        for pattern in statement_patterns:
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

    # ------------------------------------------------------------------
    # Session management helpers
    # ------------------------------------------------------------------

    async def _load_active_session_from_core(self) -> Optional[Dict[str, Any]]:
        """Fast path: read active session from core memory (no DB query)."""
        try:
            state = await self.core_memory.get_all()
            session = state.get("active_session")
            if not isinstance(session, dict):
                return None
            if not session.get("id"):
                return None
            return session
        except Exception as exc:
            logger.warning("Could not read active session from core memory: %s", exc)
            return None

    async def _sync_active_session_to_core(
        self, session_row: Optional[Dict[str, Any]]
    ) -> None:
        """Write active session summary to core memory for fast reads."""
        try:
            if session_row is None:
                await self.core_memory.update("active_session", None)
                return
            # Only store the fields needed by the supervisor prompt and
            # context loading. Do not store full chat history here.
            summary = {
                "id": int(session_row.get("id") or 0),
                "name": str(session_row.get("name") or ""),
                "description": str(session_row.get("description") or ""),
                "epic_id": session_row.get("epic_id"),
                "turn_count": int(session_row.get("turn_count") or 0),
                "memory_count": int(session_row.get("memory_count") or 0),
                "started_at": str(session_row.get("created_at") or ""),
            }
            await self.core_memory.update("active_session", summary)
        except Exception as exc:
            logger.warning("Could not sync active session to core memory: %s", exc)

    async def _get_active_session(self) -> Optional[Dict[str, Any]]:
        """Return active session from core memory cache (fast, no DB call)."""
        return await self._load_active_session_from_core()

    async def _get_session_epic_rollup(
        self, epic_id: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Return Epic title and story/task counts for session context block."""
        if not epic_id:
            return None
        try:
            rollups = await self.ledger_memory.get_objective_hierarchy_rollup(epic_id)
            # Find the epic-level rollup
            epic_rollup = next(
                (r for r in rollups if r.get("tier") == "Epic" and r.get("id") == epic_id),
                None,
            )
            if not epic_rollup:
                # Fall back to just the title from the backlog
                tree = await self.ledger_memory.get_active_objective_tree(epic_id)
                epic_node = next(
                    (n for n in tree if n.get("tier") == "Epic"), None
                )
                if epic_node:
                    return {
                        "title": str(epic_node.get("title") or ""),
                        "status": str(epic_node.get("status") or ""),
                        "total_tasks": 0,
                        "completed_tasks": 0,
                        "pending_tasks": 0,
                        "active_tasks": 0,
                    }
                return None
            return {
                "title": str(epic_rollup.get("title") or ""),
                "status": str(epic_rollup.get("status") or ""),
                "total_tasks": int(epic_rollup.get("total_tasks") or 0),
                "completed_tasks": int(epic_rollup.get("completed_tasks") or 0),
                "pending_tasks": int(epic_rollup.get("pending_tasks") or 0),
                "active_tasks": int(epic_rollup.get("active_tasks") or 0),
            }
        except Exception as exc:
            logger.warning("Could not fetch epic rollup for session: %s", exc)
            return None

    async def _activate_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Activate a session in the DB and sync to core memory.
        Returns the session row or None if not found."""
        row = await self.ledger_memory.activate_session(session_id)
        if row is not None:
            await self._sync_active_session_to_core(row)
            logger.info("Session activated: id=%s name=%r", row["id"], row["name"])
        return row

    async def _deactivate_session(self) -> None:
        """Deactivate all sessions and clear core memory cache."""
        await self.ledger_memory.deactivate_all_sessions()
        await self._sync_active_session_to_core(None)
        logger.info("Session deactivated (no active session)")

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
        if Orchestrator._extract_user_profile_updates(user_message) or Orchestrator._extract_assistant_identity_update(user_message):
            return "profile_update"
        if RoutingAssessor._is_trivial_direct_intent(user_message):
            return "casual"
        if Orchestrator._looks_like_task_request(lowered):
            return "task"
        return "casual"

    @staticmethod
    def _resolve_positive_int_env(name: str, default: int) -> int:
        raw = str(os.getenv(name, str(default))).strip()
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return int(default)
        return value if value > 0 else int(default)

    @classmethod
    def _intent_classifier_cache_max_size(cls) -> int:
        return cls._resolve_positive_int_env(
            "INTENT_CLASSIFIER_CACHE_MAX_SIZE",
            _INTENT_CLASSIFIER_CACHE_MAX_SIZE_DEFAULT,
        )

    @staticmethod
    def _evict_ordered_mapping_to_size(
        mapping: "OrderedDict[Any, Any]",
        max_size: int,
        *,
        label: str,
    ) -> None:
        while len(mapping) > max_size:
            evicted_key, _ = mapping.popitem(last=False)
            logger.warning(
                "%s exceeded max size (%s); evicted least-recently-used key: %s",
                label,
                max_size,
                evicted_key,
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

    @staticmethod
    def _format_profile_update_ack(updates: Dict[str, Any]) -> str:
        parts = []
        if "name" in updates:
            parts.append(f"your name as {updates['name']}")
        if "age" in updates:
            parts.append(f"your age as {updates['age']}")
        if "relationship" in updates:
            parts.append(f"your role as {updates['relationship']}")
        detail = " and ".join(parts) if parts else "that information"
        return f"Noted. I've recorded {detail} in my memory."

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

    async def _try_meta_fast_path_response(self, state: Dict[str, Any]) -> Optional[str]:
        user_message = state.get("user_input", "")
        chat_history = list(state.get("chat_history", []) or [])

        profile_response = await self._build_profile_response(state)
        if profile_response:
            return profile_response

        if self._is_last_reply_question(user_message):
            return self._recall_last_assistant_message(chat_history)

        # Confirmatory follow-ups like "Did you search for it?" / "Have you found it?"
        # should acknowledge the previous action, not re-trigger a tool.
        if self._is_confirmatory_followup(user_message):
            last = self._recall_last_assistant_message(chat_history)
            if last:
                return last

        if self._is_summary_request(user_message):
            return self._summarize_chat_history(chat_history)

        if self._is_skill_list_request(user_message):
            return self._build_compact_skill_list_response()

        if self._should_use_intent_classifier(user_message):
            intent = await self._classify_user_intent(user_message)
            if intent == "capability_query":
                capability_response = self._build_capability_response(
                    user_message,
                    classified_intent=intent,
                )
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
            if (
                result.status == "ok"
                and result.content
                and not result.content.startswith(_SYSTEM_1_ERROR_PREFIX)
            ):
                return result.content.strip()
            return None

        if assessment["mode"] == "multi_ticker":
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

        if assessment["mode"] != "single_tool":
            return None

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
        return await self.cognitive_router.route_to_system_2(
            minimized_messages,
            allowed_tools=allowed_tools,
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
    ) -> RouterResult:
        before_metrics = self._get_system_1_gate_metrics()
        route_kwargs: Dict[str, Any] = {"allowed_tools": allowed_tools}
        if max_output_tokens is not None:
            route_kwargs["max_output_tokens"] = max_output_tokens
        route_call = self.cognitive_router.route_to_system_1(messages, **route_kwargs)
        try:
            if deadline_seconds is None:
                return await route_call
            return await asyncio.wait_for(route_call, timeout=deadline_seconds)
        finally:
            after_metrics = self._get_system_1_gate_metrics()
            self._log_system_1_gate_pressure(context, before_metrics, after_metrics)

    async def _get_archival_context(
        self,
        query: str,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> str:
        """Retrieve top archival memory snippets relevant to the query.
        Scoped to session/epic when provided."""
        if not query:
            return ""
        _max_chunk = int(os.getenv("MAX_ARCHIVAL_CHUNK_CHARS", "2000"))
        _max_total = int(os.getenv("MAX_ARCHIVAL_TOTAL_CHARS", "6000"))

        from src.skills.search_archival_memory import _build_where_clause
        where = _build_where_clause(session_id, epic_id)

        try:
            results = await self.vector_memory.query_memory_async(
                query, n_results=3, where=where
            )
            # Fall back to global if scoped returns nothing useful
            if not results and where is not None:
                results = await self.vector_memory.query_memory_async(
                    query, n_results=3
                )
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
                # Annotate scoped memories so the supervisor knows they are
                # project-specific
                meta = item.get("metadata", {}) or {}
                mem_session = int(meta.get("session_id") or 0)
                scope_prefix = "[session] " if mem_session and mem_session == session_id else ""
                lines.append(f"  <Memory>{scope_prefix}{snippet}</Memory>")
                total += len(snippet)
            lines.append("</Archival_Memory>")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Archival memory lookup failed: {e}")
            return ""

    @classmethod
    def _warn_deprecated_energy_replenish_turn_env_once(cls) -> None:
        if cls._deprecated_energy_replenish_turn_warning_logged:
            return
        if os.getenv("ENERGY_REPLENISH_PER_TURN") is None:
            return
        logger.warning(
            "ENERGY_REPLENISH_PER_TURN is deprecated and ignored. Use ENERGY_REPLENISH_PER_HOUR instead."
        )
        cls._deprecated_energy_replenish_turn_warning_logged = True

    @staticmethod
    def _resolve_energy_replenish_per_hour() -> float:
        raw_rate = str(os.getenv("ENERGY_REPLENISH_PER_HOUR", "30") or "").strip()
        try:
            rate = float(raw_rate)
        except ValueError:
            logger.warning(
                "Invalid ENERGY_REPLENISH_PER_HOUR=%r. Falling back to 30.",
                raw_rate,
            )
            rate = 30.0
        return max(0.0, rate)

    @staticmethod
    def _resolve_energy_replenish_per_heartbeat() -> int:
        raw_value = str(
            os.getenv("ENERGY_REPLENISH_PER_HEARTBEAT", str(_HEARTBEAT_REPLENISH_ENV_DEFAULT)) or ""
        ).strip()
        try:
            amount = int(raw_value)
        except ValueError:
            logger.warning(
                "Invalid ENERGY_REPLENISH_PER_HEARTBEAT=%r. Falling back to %s.",
                raw_value,
                _HEARTBEAT_REPLENISH_ENV_DEFAULT,
            )
            amount = _HEARTBEAT_REPLENISH_ENV_DEFAULT
        return max(0, amount)

    async def _tick_predictive_energy_budget(self, amount: int, reason: str) -> int:
        if amount <= 0:
            return 0

        before = await self._get_predictive_energy_budget_remaining()
        await self._refund_predictive_energy_budget(int(amount), reason)
        after = await self._get_predictive_energy_budget_remaining()
        applied = max(0, after - before)
        return applied

    async def _get_predictive_energy_budget_remaining(self) -> int:
        async with self._energy_budget_lock:
            return max(0, int(self._energy_budget))

    async def _refund_predictive_energy_budget(self, amount: int, reason: str) -> None:
        if amount <= 0:
            return

        cap = max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
        async with self._energy_budget_lock:
            self._energy_budget = min(cap, self._energy_budget + int(amount))
        logger.info(
            "Energy budget refunded: +%s (%s). Remaining=%s",
            amount, reason, self._energy_budget,
        )

    async def _try_reserve_predictive_energy_budget(
        self,
        *,
        predicted_cost: int,
        min_reserve: int,
        reason: str,
    ) -> bool:
        cost = max(0, int(predicted_cost))
        reserve = max(0, int(min_reserve))
        async with self._energy_budget_lock:
            if self._energy_budget - cost < reserve:
                return False
            self._energy_budget -= cost
        logger.info(
            "Energy budget reserved: -%s (%s). Remaining=%s",
            cost, reason, self._energy_budget,
        )
        return True

    @staticmethod
    def _build_energy_evaluation_record(
        *,
        evaluation: EnergyEvaluation,
        decision: EnergyDecision,
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

    async def _route_energy_judge_messages(self, messages: List[Dict[str, str]]) -> RouterResult:
        return await self._route_to_system_1(
            messages,
            allowed_tools=[],
            deadline_seconds=45.0,
            context="energy_judge",
        )

    async def _evaluate_energy_for_context(
        self,
        *,
        task: Dict[str, Any],
        story: Optional[Dict[str, Any]],
        epic: Optional[Dict[str, Any]],
        additional_context: str,
    ) -> tuple[EnergyEvaluation, EnergyDecision, int]:
        available_energy = await self._get_predictive_energy_budget_remaining()
        defer_count = int(task.get("defer_count") or 0)
        evaluation = await self.energy_judge.evaluate_with_system1(
            task=task,
            story=story,
            epic=epic,
            route_to_system_1=self._route_energy_judge_messages,
            additional_context=additional_context,
        )
        decision = self.energy_roi_engine.evaluate(
            estimated_effort=evaluation.estimated_effort,
            expected_value=evaluation.expected_value,
            available_energy=available_energy,
            defer_count=defer_count,
        )
        return evaluation, decision, available_energy

    def _build_synthesized_ad_hoc_energy_context(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        user_message = str(state.get("user_input") or "").strip()
        complexity = max(1, self.routing_assessor._estimate_request_complexity(user_message) + 1)
        history_context = self._build_goal_planning_context(state)

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

    @staticmethod
    def _format_ad_hoc_energy_deferral_message(
        *,
        decision: EnergyDecision,
        available_energy: int,
    ) -> str:
        return (
            "Deferred due to energy policy. "
            f"Reason: {decision.reason}. "
            f"ROI={decision.roi:.2f} (base={decision.base_roi:.2f}, defer_count={decision.defer_count}), "
            f"predicted_cost={decision.predicted_cost}, "
            f"available_budget={int(available_energy)}, reserve_floor={decision.min_reserve}."
        )

    async def _try_ad_hoc_dispatch_energy_gate(
        self,
        state: Dict[str, Any],
        *,
        dispatch_context: str,
    ) -> Optional[str]:
        user_id = str(state.get("user_id") or "")
        if user_id == "heartbeat":
            return None

        user_message = str(state.get("user_input") or "").strip()
        if not user_message:
            return None

        context = self._build_synthesized_ad_hoc_energy_context(state)
        evaluation, decision, available_energy = await self._evaluate_energy_for_context(
            task=context["task"] or {},
            story=context["story"],
            epic=context["epic"],
            additional_context=str(context.get("additional_context") or ""),
        )

        if not decision.should_execute:
            # Comfort-budget floor bypass: when more than half the starting budget
            # remains and the sole reason for deferral is a borderline ROI (not an
            # actual reserve shortage), approve the request unconditionally.  This
            # prevents the energy gate from creating friction during normal interactive
            # sessions where resources are genuinely plentiful.
            if (
                decision.reason == "ROI too low"
                and available_energy > _AD_HOC_COMFORTABLE_BUDGET_FLOOR
            ):
                logger.info(
                    "Ad-hoc dispatch approved by comfort-budget floor (%s): "
                    "ROI=%.2f below threshold but budget=%s > floor=%s",
                    dispatch_context,
                    decision.roi,
                    available_energy,
                    _AD_HOC_COMFORTABLE_BUDGET_FLOOR,
                )
            else:
                message = self._format_ad_hoc_energy_deferral_message(
                    decision=decision,
                    available_energy=available_energy,
                )
                logger.warning(
                    "Ad-hoc dispatch deferred by energy gate (%s): %s",
                    dispatch_context,
                    message,
                )
                return message

        reserved = await self._try_reserve_predictive_energy_budget(
            predicted_cost=decision.predicted_cost,
            min_reserve=decision.min_reserve,
            reason=f"ad_hoc:{dispatch_context}:{user_id}",
        )
        if not reserved:
            current_budget = await self._get_predictive_energy_budget_remaining()
            race_decision = self.energy_roi_engine.evaluate(
                estimated_effort=evaluation.estimated_effort,
                expected_value=evaluation.expected_value,
                available_energy=current_budget,
                defer_count=decision.defer_count,
            )
            message = self._format_ad_hoc_energy_deferral_message(
                decision=race_decision,
                available_energy=current_budget,
            )
            logger.warning(
                "Ad-hoc dispatch deferred after budget reservation race (%s): %s",
                dispatch_context,
                message,
            )
            return message

        logger.info(
            "Ad-hoc dispatch approved by energy gate (%s): roi=%.2f cost=%s remaining=%s",
            dispatch_context,
            decision.roi,
            decision.predicted_cost,
            await self._get_predictive_energy_budget_remaining(),
        )
        return None

    def _new_state(
        self,
        user_id: str,
        user_message: str,
        *,
        user_prompt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = AgentState.new(
            user_id=user_id,
            user_input=user_message,
            user_prompt=user_prompt,
        ).to_dict()
        state["_energy_gate_cleared"] = False
        return state

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
        _INTERNAL_OUTPUT_KEYS = {"supervisor_context", "_s2_blueprint"}
        return {
            name: output
            for name, output in dict(state.get("worker_outputs", {}) or {}).items()
            if name not in _INTERNAL_OUTPUT_KEYS
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
                self._route_to_system_2_redacted(
                    messages,
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
        trailing = remainder.strip()
        if trailing and not re.fullmatch(r"[\s`.,;!?]*", remainder):
            logger.debug(
                "_decode_workers_payload: rejected payload due to trailing "
                "content %r (first 60 chars of payload: %r)",
                trailing[:40],
                workers_payload[:60],
            )
            return None
        return decoded

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
        for (agent_def, _task_packet), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error("Parallel agent %s failed with exception: %s", agent_def.name, result, exc_info=True)
                state["worker_outputs"][agent_def.name] = f"Error: {agent_def.name} failed with exception: {result}"
                continue

            agent_output = dict(result.get("worker_outputs", {}) or {}).get(agent_def.name)
            if agent_output is not None:
                state["worker_outputs"][agent_def.name] = agent_output
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
        _agent_emitter = get_current_emitter()
        _agent_start_mono = time.monotonic()
        if _agent_emitter is not None:
            _agent_emitter.record_agent_start(agent_def.name)
            await _agent_emitter.emit(ProgressEvent.agent_start(agent_def.name))
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
            else:
                state["worker_outputs"][agent_def.name] = (
                    f"Error: {agent_def.name} failed and System 2 is not configured."
                )
            if _agent_emitter is not None:
                _dur = time.monotonic() - _agent_start_mono
                await _agent_emitter.emit(
                    ProgressEvent.agent_done(agent_def.name, _dur)
                )
            return state

        if router_result.status == "cognitive_escalation":
            solution = await self._handle_cognitive_escalation(state, router_result)
            state["worker_outputs"][agent_def.name] = solution
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
            if _agent_emitter is not None:
                _dur = time.monotonic() - _agent_start_mono
                await _agent_emitter.emit(
                    ProgressEvent.agent_done(agent_def.name, _dur)
                )

        return state

    def _enqueue_outbound_payload(self, payload: Any, *, source: str) -> bool:
        queue = getattr(self, "outbound_queue", None)
        if queue is None:
            logger.info("[Admin notification (no queue)]: %s", payload)
            return False

        try:
            queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            dropped_payload = None
            try:
                dropped_payload = queue.get_nowait()
                try:
                    queue.task_done()
                except ValueError:
                    pass
            except asyncio.QueueEmpty:
                pass

            logger.warning(
                "Dropped oldest outbound message because queue is full "
                "(source=%s, maxsize=%s, dropped_type=%s).",
                source,
                getattr(queue, "maxsize", "unknown"),
                type(dropped_payload).__name__ if dropped_payload is not None else "none",
            )

            try:
                queue.put_nowait(payload)
                return True
            except asyncio.QueueFull:
                logger.error(
                    "Failed to enqueue outbound payload after dropping oldest message "
                    "(source=%s).",
                    source,
                )
                return False

    async def _notify_admin(self, message: Any) -> None:
        """Send a message to the admin via the outbound queue (used by heartbeat)."""
        self._enqueue_outbound_payload(message, source="notify_admin")
        await asyncio.sleep(0)

    @staticmethod
    def _heartbeat_result_indicates_failure(result: str) -> bool:
        lowered = str(result or "").lower()
        return any(
            marker in lowered
            for marker in ("error", "cannot", "unable", "failed", "i cannot", "i am unable")
        )

    @staticmethod
    def _build_heartbeat_execution_prompt(task: Dict[str, Any]) -> str:
        prefix = HEARTBEAT_TASK_PREFIX_FMT.format(task_id=task["id"])
        return (
            f"{prefix}: {task['title']}\n"
            f"You MUST execute this task right now by calling the appropriate tools. "
            f"Do NOT describe what you plan to do — use your tools and report the actual results. "
            f"If the task requires storing data, call update_core_memory or update_ledger explicitly. "
            f"If you cannot complete it, explain exactly why."
        )

    @staticmethod
    def _is_executable_heartbeat_task_candidate(
        node: Dict[str, Any],
        unresolved_task_ids: set[int],
        parent_ids: set[int],
    ) -> bool:
        if str(node.get("tier") or "") != "Task":
            return False
        if str(node.get("status") or "").lower() not in {"pending", "deferred_due_to_energy"}:
            return False

        task_id = int(node.get("id") or 0)
        if task_id <= 0:
            return False
        return task_id not in unresolved_task_ids and task_id not in parent_ids

    async def _select_executable_heartbeat_tasks(self) -> List[Dict[str, Any]]:
        """Return executable Task candidates with Story/Epic context for energy scoring."""
        candidates = await self.ledger_memory.get_energy_evaluation_candidates(
            statuses=["pending", "deferred_due_to_energy"],
        )
        if not candidates:
            return []

        active_nodes = await self.ledger_memory.get_active_objective_tree()
        unresolved_rows = await self.ledger_memory.get_tasks_with_unresolved_dependencies(
            statuses=["pending", "active", "deferred_due_to_energy"],
        )
        unresolved_task_ids = {int(row["id"]) for row in unresolved_rows}
        parent_ids = {
            int(node["parent_id"])
            for node in active_nodes
            if node.get("parent_id") is not None
        }

        executable: List[Dict[str, Any]] = []
        for candidate in candidates:
            task = dict(candidate.get("task") or {})
            if self._is_executable_heartbeat_task_candidate(task, unresolved_task_ids, parent_ids):
                executable.append(candidate)

        executable.sort(
            key=lambda candidate: (
                int((candidate.get("task") or {}).get("priority") or 999),
                int((candidate.get("task") or {}).get("estimated_energy") or 999999),
                int((candidate.get("task") or {}).get("id") or 0),
            )
        )
        return executable

    async def _select_executable_heartbeat_task(self) -> Optional[Dict[str, Any]]:
        """Pick the next executable Task leaf with all dependencies resolved."""
        candidates = await self._select_executable_heartbeat_tasks()
        if not candidates:
            return None
        return dict(candidates[0].get("task") or {})

    async def _handle_heartbeat_task_failure(
        self,
        task: Dict[str, Any],
        *,
        reason: str,
        result_excerpt: str,
    ) -> None:
        task_id = int(task["id"])
        strike_count = await self._increment_heartbeat_failure_count(task_id)

        if strike_count >= _HEARTBEAT_FAILURE_STRIKES:
            await self.ledger_memory.update_objective_status(task_id, "blocked")
            await self.ledger_memory.ensure_parent_chain_active(task_id)
            await self._notify_admin(
                "HITL REQUIRED: Task blocked after repeated heartbeat failures.\n"
                f"  Task #{task_id}: {task['title']}\n"
                f"  Strikes: {strike_count}\n"
                f"  Reason: {reason}\n"
                f"  Latest result: {result_excerpt[:220]}"
            )
            logger.warning(
                "Heartbeat: Task #%s blocked after %s failures (%s)",
                task_id,
                strike_count,
                reason,
            )
            return

        await self.ledger_memory.update_objective_status(task_id, "pending")
        logger.warning(
            "Heartbeat: Task #%s failure strike %s/%s (%s)",
            task_id,
            strike_count,
            _HEARTBEAT_FAILURE_STRIKES,
            reason,
        )

    async def _defer_heartbeat_task_due_to_energy(
        self,
        *,
        task_id: int,
        decision: EnergyDecision,
        evaluation_record: Dict[str, Any],
        available_energy: int,
        race: bool = False,
    ) -> None:
        await self.ledger_memory.defer_task_due_to_energy(
            task_id,
            evaluation_record,
            cooldown_seconds=int(decision.defer_cooldown_seconds),
        )
        logger.info(
            "Heartbeat: Deferred task #%s%s (%s). roi=%.2f cost=%s available=%s reserve_floor=%s",
            task_id,
            " after budget race" if race else "",
            decision.reason,
            decision.roi,
            decision.predicted_cost,
            available_energy,
            decision.min_reserve,
        )

    async def _execute_heartbeat_task(
        self,
        *,
        task: Dict[str, Any],
        decision: EnergyDecision,
    ) -> None:
        task_id = int(task["id"])
        logger.info(
            "Heartbeat: Accepting executable task #%s: %s (roi=%.2f cost=%s budget=%s)",
            task_id,
            str(task.get("title") or "")[:60],
            decision.roi,
            decision.predicted_cost,
            await self._get_predictive_energy_budget_remaining(),
        )
        await self.ledger_memory.update_objective_status(task_id, "active")

        timeout_seconds = HEARTBEAT_INTERVAL * 0.9
        prompt = self._build_heartbeat_execution_prompt(task)

        try:
            result = await asyncio.wait_for(
                self.process_message(user_message=prompt, user_id="heartbeat"),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._handle_heartbeat_task_failure(
                task,
                reason="timeout",
                result_excerpt=f"Timed out after {int(timeout_seconds)}s",
            )
            return
        except Exception as e:
            await self._handle_heartbeat_task_failure(
                task,
                reason="exception",
                result_excerpt=str(e),
            )
            return

        if self._has_pending_heartbeat_hitl(task_id) or self._has_pending_heartbeat_mfa(task_id):
            logger.info("Heartbeat: Task #%s paused awaiting admin guidance.", task_id)
            return

        if self._heartbeat_result_indicates_failure(result):
            await self._handle_heartbeat_task_failure(
                task,
                reason="execution_failure",
                result_excerpt=result,
            )
            return

        await self.ledger_memory.update_objective_status(task_id, "completed")
        await self._clear_heartbeat_failure_count(task_id)

        summary = (
            f"Heartbeat completed task #{task_id}:\n"
            f"  Task: {task.get('title', '')}\n"
            f"  Predicted Energy Cost: {decision.predicted_cost}\n"
            f"  Remaining Predictive Budget: {await self._get_predictive_energy_budget_remaining()}\n"
            f"  Result: {str(result)[:220]}"
        )
        await self._notify_admin(summary)
        logger.info("Heartbeat: Task #%s completed.", task_id)

    async def _run_heartbeat_cycle(self) -> None:
        await self._tick_predictive_energy_budget(
            amount=self._resolve_energy_replenish_per_heartbeat(),
            reason="heartbeat_cycle",
        )

        logger.info("Heartbeat: Querying objective backlog for executable Tasks...")
        candidate_contexts = await self._select_executable_heartbeat_tasks()

        if not candidate_contexts:
            # Try to report when the next deferred task becomes eligible.
            try:
                cursor = await self.ledger_memory._db.execute(
                    "SELECT MIN(next_eligible_at) as earliest "
                    "FROM objective_backlog "
                    "WHERE tier='Task' AND status='deferred_due_to_energy' "
                    "AND next_eligible_at IS NOT NULL"
                )
                row = await cursor.fetchone()
                earliest = row["earliest"] if row else None
            except Exception:
                earliest = None

            if earliest:
                logger.info(
                    "Heartbeat: No executable tasks available. "
                    "Next deferred task eligible at: %s",
                    earliest,
                )
            else:
                logger.info("Heartbeat: No executable dependency-resolved tasks available.")
            return

        for candidate in candidate_contexts:
            task = dict(candidate.get("task") or {})
            story = candidate.get("story")
            epic = candidate.get("epic")

            task_id = int(task.get("id") or 0)
            if task_id <= 0:
                continue

            evaluation, decision, available_energy = await self._evaluate_energy_for_context(
                task=task,
                story=story,
                epic=epic,
                additional_context="heartbeat_dispatch",
            )
            evaluation_record = self._build_energy_evaluation_record(
                evaluation=evaluation,
                decision=decision,
                available_energy=available_energy,
                context="heartbeat",
            )

            if not decision.should_execute:
                await self._defer_heartbeat_task_due_to_energy(
                    task_id=task_id,
                    decision=decision,
                    evaluation_record=evaluation_record,
                    available_energy=available_energy,
                )
                continue

            reserved = await self._try_reserve_predictive_energy_budget(
                predicted_cost=decision.predicted_cost,
                min_reserve=decision.min_reserve,
                reason=f"heartbeat_task:{task_id}",
            )
            if not reserved:
                current_budget = await self._get_predictive_energy_budget_remaining()
                race_decision = self.energy_roi_engine.evaluate(
                    estimated_effort=evaluation.estimated_effort,
                    expected_value=evaluation.expected_value,
                    available_energy=current_budget,
                    defer_count=decision.defer_count,
                )
                race_record = self._build_energy_evaluation_record(
                    evaluation=evaluation,
                    decision=race_decision,
                    available_energy=current_budget,
                    context="heartbeat",
                )
                await self._defer_heartbeat_task_due_to_energy(
                    task_id=task_id,
                    decision=race_decision,
                    evaluation_record=race_record,
                    available_energy=current_budget,
                    race=True,
                )
                continue

            await self.ledger_memory.record_task_energy_evaluation(
                task_id,
                evaluation_record,
                clear_next_eligible=True,
            )
            await self._execute_heartbeat_task(task=task, decision=decision)
            return

        logger.info("Heartbeat: All executable tasks were deferred by energy policy.")

    async def _heartbeat_loop(self) -> None:
        """
        Proactive Heartbeat: wakes every 30 min, selects executable Objective
        leaf tasks (dependency-resolved), executes one, and applies remediation
        logic on repeated failures.

        An asyncio.Lock prevents heartbeat cycles from overlapping: if the
        previous run is still in progress when the next interval fires, the
        new cycle is skipped (ISSUE-004).
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
                    await self._run_heartbeat_cycle()
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}", exc_info=True)

    @staticmethod
    def _is_pending_entry_expired(
        entry: Any,
        *,
        created_at_key: str,
        ttl_seconds: int,
        now_epoch: float,
    ) -> bool:
        payload = entry if isinstance(entry, dict) else {}
        try:
            created_at = float(payload.get(created_at_key, 0) or 0)
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at <= 0:
            return True
        return (now_epoch - created_at) > float(ttl_seconds)

    def _purge_expired_pending_in_memory(
        self,
        *,
        attr_name: str,
        created_at_key: str,
        ttl_seconds: int,
        now_epoch: float,
    ) -> int:
        pending_store = getattr(self, attr_name, None)
        if not isinstance(pending_store, dict):
            return 0

        expired_users = [
            user_id
            for user_id, payload in pending_store.items()
            if self._is_pending_entry_expired(
                payload,
                created_at_key=created_at_key,
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            )
        ]
        for user_id in expired_users:
            pending_store.pop(user_id, None)
        if expired_users:
            logger.warning(
                "Swept %d expired in-memory pending entries from %s.",
                len(expired_users),
                attr_name,
            )
        return len(expired_users)

    async def _sweep_expired_pending_state(self) -> None:
        ttl_seconds = self._resolve_positive_int_env(
            "PENDING_STATE_TTL_SECONDS",
            _PENDING_STATE_TTL_SECONDS,
        )

        db_deleted: Dict[str, int] = {}
        purge_pending = getattr(self.ledger_memory, "purge_expired_pending", None)
        if callable(purge_pending):
            try:
                db_deleted = await purge_pending(ttl_seconds=ttl_seconds)
            except Exception as e:
                logger.warning("Pending-state DB sweep failed: %s", e)

        now_epoch = time.time()
        mem_deleted = {
            "pending_tool_approval": self._purge_expired_pending_in_memory(
                attr_name="pending_tool_approval",
                created_at_key="_created_at",
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            ),
            "pending_hitl_state": self._purge_expired_pending_in_memory(
                attr_name="pending_hitl_state",
                created_at_key="_hitl_created_at",
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            ),
            "pending_mfa": self._purge_expired_pending_in_memory(
                attr_name="pending_mfa",
                created_at_key="_created_at",
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            ),
        }

        if any(int(value or 0) > 0 for value in db_deleted.values()) or any(mem_deleted.values()):
            logger.info(
                "Pending-state sweep complete. DB deleted=%s in-memory deleted=%s",
                db_deleted,
                mem_deleted,
            )

    async def _memory_consolidation_loop(self) -> None:
        """Background task: periodically consolidate chat history into long-term memory."""
        logger.info("Memory consolidation loop started.")
        while True:
            await asyncio.sleep(MEMORY_CONSOLIDATION_INTERVAL)
            try:
                await self._sweep_expired_pending_state()
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

    async def _save_memory_async(
        self,
        text: str,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> None:
        """Fire-and-forget memory storage with optional session/epic tagging."""
        try:
            if len(text) < MEMORY_SAVE_THRESHOLD:
                return
            metadata: Dict[str, Any] = {
                "type": "conversation",
                "timestamp": datetime.now().isoformat(),
                # ChromaDB requires scalar values; 0 means "no session/epic"
                "session_id": int(session_id) if session_id else 0,
                "epic_id": int(epic_id) if epic_id else 0,
            }
            await self.vector_memory.add_memory_async(text=text, metadata=metadata)
            if session_id:
                increment_memory_count = getattr(
                    self.ledger_memory,
                    "increment_session_memory_count",
                    None,
                )
                if callable(increment_memory_count) and inspect.iscoroutinefunction(increment_memory_count):
                    self._fire_and_forget(increment_memory_count(session_id))
        except Exception as e:
            logger.warning(f"Async memory save failed: {e}")

    @staticmethod
    def _has_explicit_memory_intent(text: str) -> bool:
        return bool(_EXPLICIT_MEMORY_REQUEST_RE.search(str(text or "")))

    async def _save_explicit_memory_request(self, text: str) -> None:
        try:
            await self.vector_memory.add_memory_async(
                text=text,
                metadata={"type": "explicit_memory", "source": "user_request"},
            )
        except Exception as e:
            logger.warning(f"Explicit memory save failed: {e}")

    async def _apply_text_memory_hooks(self, user_id: str, user_message: str) -> None:
        profile_updated = await self._remember_user_profile(user_id, user_message)
        await self._remember_assistant_identity(user_message)
        if not profile_updated and self._has_explicit_memory_intent(user_message):
            await self._save_explicit_memory_request(user_message)

    async def _load_approved_tools(self) -> None:
        """Re-register all previously approved dynamic tools from the DB at startup."""
        try:
            approved = await self.ledger_memory.get_approved_tools()
            for tool in approved:
                try:
                    await self.cognitive_router.register_dynamic_tool(
                        tool["name"], tool["code"], tool["schema_json"]
                    )
                    logger.info(f"Restored dynamic tool '{tool['name']}' from registry")
                except Exception as e:
                    logger.error(f"Failed to restore tool '{tool['name']}': {e}")
            self._invalidate_capabilities_cache()
        except Exception as e:
            logger.warning(f"_load_approved_tools failed: {e}")

    async def _reload_dynamic_tools_after_worker_restart(self) -> None:
        """Rehydrate the isolated worker from approved dynamic tools after a crash."""
        logger.critical("Dynamic tool worker restarted; re-registering approved tools from ledger.")
        await self._load_approved_tools()

    async def _load_pending_approvals(self) -> None:
        """Reload any tool synthesis proposals that were pending when the bot last stopped."""
        try:
            pending = await self.ledger_memory.load_pending_approvals()
            for user_id, payload in pending.items():
                self.pending_tool_approval[user_id] = {
                    "synthesis": payload["synthesis"],
                    "original_state": {"user_input": payload["original_input"], "user_id": user_id},
                    "_created_at": float(payload.get("_created_at", time.time())),
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

    async def _restore_heartbeat_failure_counts(self) -> None:
        """Restore heartbeat task-failure strike counts from persisted system_state."""
        try:
            raw_value = await self.ledger_memory.get_system_state(_HEARTBEAT_FAILURE_STATE_KEY)
            if not raw_value:
                self._heartbeat_failure_counts = {}
                return

            decoded = json.loads(raw_value)
            if not isinstance(decoded, dict):
                self._heartbeat_failure_counts = {}
                return

            restored: Dict[int, int] = {}
            for raw_task_id, raw_count in decoded.items():
                try:
                    task_id = int(raw_task_id)
                    count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                if task_id > 0 and count > 0:
                    restored[task_id] = count
            self._heartbeat_failure_counts = restored
        except Exception as e:
            logger.warning("Failed to restore heartbeat failure counts: %s", e)
            self._heartbeat_failure_counts = {}

    def _coerce_consolidation_turn_counts(self) -> "OrderedDict[str, int]":
        counts = getattr(self, "_consolidation_turn_counts", None)
        if isinstance(counts, OrderedDict):
            return counts
        if isinstance(counts, dict):
            ordered = OrderedDict(counts.items())
        else:
            ordered = OrderedDict()
        self._consolidation_turn_counts = ordered
        self._evict_ordered_mapping_to_size(
            ordered,
            _CONSOLIDATION_TURN_COUNT_MAX_USERS,
            label="Consolidation turn counts",
        )
        return ordered

    async def _restore_consolidation_turn_counts(self) -> None:
        prefix = "consolidation_turn_count:"
        restored: "OrderedDict[str, int]" = OrderedDict()
        if not hasattr(self, "_consolidation_turn_counts"):
            self._consolidation_turn_counts = OrderedDict()

        getter = getattr(self.ledger_memory, "get_system_state_keys_by_prefix", None)
        if not callable(getter):
            self._consolidation_turn_counts = OrderedDict()
            return

        try:
            rows = await getter(prefix)
            for key, value in rows:
                key_text = str(key or "")
                if not key_text.startswith(prefix):
                    continue

                user_id = key_text[len(prefix):].strip()
                if not user_id:
                    continue

                try:
                    count = int(str(value).strip())
                except (TypeError, ValueError):
                    continue

                restored[user_id] = max(0, count)

            self._consolidation_turn_counts = restored
            bounded_counts = self._coerce_consolidation_turn_counts()
            if bounded_counts:
                logger.info("Restored %d consolidation turn counter(s) from DB", len(bounded_counts))
        except Exception as e:
            logger.warning("Failed to restore consolidation turn counts: %s", e)
            self._consolidation_turn_counts = OrderedDict()

    async def _persist_heartbeat_failure_counts(self) -> None:
        payload = {
            str(task_id): int(count)
            for task_id, count in self._heartbeat_failure_counts.items()
            if int(count) > 0
        }
        await self.ledger_memory.set_system_state(
            _HEARTBEAT_FAILURE_STATE_KEY,
            json.dumps(payload, sort_keys=True),
        )

    async def _increment_heartbeat_failure_count(self, task_id: int) -> int:
        current = int(self._heartbeat_failure_counts.get(task_id, 0)) + 1
        self._heartbeat_failure_counts[task_id] = current
        await self._persist_heartbeat_failure_counts()
        return current

    async def _clear_heartbeat_failure_count(self, task_id: int) -> None:
        if task_id in self._heartbeat_failure_counts:
            self._heartbeat_failure_counts.pop(task_id, None)
            await self._persist_heartbeat_failure_counts()

    def _persist_consolidation_turn_count_async(self, user_id: str, count: int) -> None:
        ledger = getattr(self, "ledger_memory", None)
        scheduler = getattr(self, "_fire_and_forget", None)
        if ledger is None or not callable(scheduler):
            return

        setter = getattr(ledger, "set_system_state", None)
        if not callable(setter):
            return

        key = f"consolidation_turn_count:{user_id}"
        try:
            persistence_coro = setter(key, str(max(0, int(count))))
        except Exception as e:
            logger.warning("Failed to schedule consolidation turn count persistence for %s: %s", user_id, e)
            return

        if asyncio.iscoroutine(persistence_coro):
            scheduler(persistence_coro)

    def _fire_and_forget(self, coro) -> asyncio.Task:
        """Schedule a coroutine as a background task with a strong GC-safe reference (ISSUE-002).

        The task is added to ``_background_tasks`` and automatically removed
        when it completes, so the set never grows unboundedly while still
        preventing the garbage collector from destroying mid-flight tasks.
        """
        task_name = getattr(coro, "__name__", "")
        if not task_name and hasattr(coro, "cr_code"):
            task_name = getattr(coro.cr_code, "co_name", "")
        if not task_name:
            task_name = repr(coro)

        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "Background task '%s' raised an unhandled exception: %s",
                    task_name,
                    exc,
                    exc_info=exc,
                )

        task.add_done_callback(_on_done)
        return task

    @staticmethod
    def _coerce_user_lock_slot(raw_ref: Any) -> Optional[_UserLockSlot]:
        if isinstance(raw_ref, weakref.ReferenceType):
            slot = raw_ref()
            return slot if isinstance(slot, _UserLockSlot) else None
        return None

    def _evict_idle_user_lock_refs(self, max_size: int) -> None:
        for lock_key in tuple(self._user_locks.keys()):
            if self._coerce_user_lock_slot(self._user_locks[lock_key]) is None:
                self._user_locks.pop(lock_key, None)

        while len(self._user_locks) > max_size:
            evicted = False
            for lock_key, raw_ref in tuple(self._user_locks.items()):
                slot = self._coerce_user_lock_slot(raw_ref)
                if slot is None or slot.in_flight == 0:
                    self._user_locks.pop(lock_key, None)
                    evicted = True
                    break
            if not evicted:
                break

    async def _get_user_lock(self, user_id: str) -> _UserLockLease:
        """Return a per-user lock lease with LRU eviction that ignores in-flight entries."""
        _max = int(os.getenv("USER_LOCKS_MAX_SIZE", "500"))
        async with self._user_locks_lock:
            if not isinstance(getattr(self, "_user_locks", None), dict):
                self._user_locks = {}

            raw_ref = self._user_locks.pop(user_id, None)
            slot = self._coerce_user_lock_slot(raw_ref) if raw_ref is not None else None
            if slot is None:
                slot = _UserLockSlot()

            slot.in_flight += 1
            self._user_locks[user_id] = weakref.ref(slot)
            self._evict_idle_user_lock_refs(_max)
            return _UserLockLease(self, user_id, slot)

    async def _release_user_lock_lease(self, user_id: str, slot: _UserLockSlot) -> None:
        _max = int(os.getenv("USER_LOCKS_MAX_SIZE", "500"))
        async with self._user_locks_lock:
            slot.in_flight = max(0, slot.in_flight - 1)
            current_ref = self._user_locks.get(user_id)
            if current_ref is None or current_ref() is slot:
                self._user_locks[user_id] = weakref.ref(slot)
            self._evict_idle_user_lock_refs(_max)

    def _extract_router_content(self, result: RouterResult) -> Optional[str]:
        """
        Return the string content from an "ok" RouterResult, or None if it is
        a non-ok result (MFA / HITL / capability-gap).  Non-ok results should
        be stored in state[_BLOCKED_KEY] by the caller.
        """
        if result.status == "ok":
            return result.content
        return None

    async def _get_recent_moral_rejections_for_supervisor(self, user_id: str) -> List[Dict[str, Any]]:
        getter = getattr(self.ledger_memory, "get_recent_moral_rejections", None)
        if not callable(getter):
            return []
        try:
            return await getter(str(user_id or ""), limit=3)
        except Exception as exc:
            logger.warning("Could not fetch recent moral rejections for supervisor context: %s", exc)
            return []

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
        state["moral_remediation_constraints"] = []
        state["moral_halt_required"] = False
        state["moral_halt_summary"] = ""
        if not state.get("final_response"):
            state["final_response"] = output_to_eval
        return state

    @staticmethod
    def _extract_charter_tier_block_from_text(charter_text: str, tier_tag: str) -> str:
        cache = Orchestrator._build_charter_tier_cache(charter_text)
        parse_error = str(cache.get(_CHARTER_PARSE_ERROR_KEY) or "").strip()
        if parse_error:
            return parse_error
        return str(cache.get(tier_tag) or f"[MISSING_CHARTER_TIER: {tier_tag}]")

    def _extract_charter_tier_block(self, tier_tag: str) -> str:
        cache = self._ensure_charter_tier_cache()
        parse_error = str(cache.get(_CHARTER_PARSE_ERROR_KEY) or "").strip()
        if parse_error:
            return parse_error
        return str(cache.get(tier_tag) or f"[MISSING_CHARTER_TIER: {tier_tag}]")

    @staticmethod
    def _strip_xml_namespace(tag: str) -> str:
        return str(tag or "").rsplit("}", 1)[-1]

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
    def _extract_heartbeat_task_id(user_input: str) -> Optional[int]:
        prefix_pattern = re.escape(HEARTBEAT_TASK_PREFIX_FMT).replace(r"\{task_id\}", r"(\d+)")
        match = re.search(prefix_pattern, str(user_input or ""))
        if not match:
            return None
        try:
            task_id = int(match.group(1))
        except (TypeError, ValueError):
            return None
        return task_id if task_id > 0 else None

    @staticmethod
    def _coerce_positive_task_id(raw_task_id: Any) -> Optional[int]:
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            return None
        return task_id if task_id > 0 else None

    def _get_admin_user_id(self) -> str:
        admin_user_id = str(getattr(self, "_admin_user_id", "") or "").strip()
        if not admin_user_id:
            admin_user_id = str(os.getenv("ADMIN_USER_ID", "") or "").strip()
        if not admin_user_id:
            admin_user_id = "heartbeat"
        self._admin_user_id = admin_user_id
        return admin_user_id

    def _extract_heartbeat_origin_task_id_from_state(self, state: Dict[str, Any]) -> Optional[int]:
        existing_task_id = self._coerce_positive_task_id(state.get("_heartbeat_origin_task_id"))
        if existing_task_id is not None:
            state["_heartbeat_origin_task_id"] = existing_task_id
            return existing_task_id

        if str(state.get("user_id") or "") != "heartbeat":
            return None

        parsed_task_id = self._extract_heartbeat_task_id(str(state.get("user_input") or ""))
        if parsed_task_id is not None:
            state["_heartbeat_origin_task_id"] = parsed_task_id
        return parsed_task_id

    def _pending_owner_for_heartbeat_origin(self, user_id: str, state: Dict[str, Any]) -> str:
        heartbeat_task_id = self._extract_heartbeat_origin_task_id_from_state(state)
        if heartbeat_task_id is None:
            return user_id
        return self._get_admin_user_id()

    def _has_pending_heartbeat_hitl(self, task_id: int) -> bool:
        pending_hitl_state = getattr(self, "pending_hitl_state", {}) or {}
        for pending_state in pending_hitl_state.values():
            if self._coerce_positive_task_id(pending_state.get("_heartbeat_origin_task_id")) == task_id:
                return True
        return False

    def _has_pending_heartbeat_mfa(self, task_id: int) -> bool:
        pending_mfa = getattr(self, "pending_mfa", {}) or {}
        for pending_state in pending_mfa.values():
            if self._coerce_positive_task_id(pending_state.get("_heartbeat_origin_task_id")) == task_id:
                return True
        return False

    async def _finalize_resumed_heartbeat_task(self, task_id: int, response: str) -> None:
        if self._heartbeat_result_indicates_failure(response):
            await self._handle_heartbeat_task_failure(
                {
                    "id": task_id,
                    "title": f"Task #{task_id}",
                },
                reason="post_hitl_resume_failure",
                result_excerpt=response,
            )
            return

        await self.ledger_memory.update_objective_status(task_id, "completed")
        await self._clear_heartbeat_failure_count(task_id)
        logger.info("Heartbeat: Task #%s completed after HITL resume.", task_id)

    async def _suspend_task_for_moral_halt(self, state: Dict[str, Any]) -> Optional[int]:
        task_id = self._extract_heartbeat_task_id(str(state.get("user_input") or ""))
        if task_id is None:
            return None

        ledger = getattr(self, "ledger_memory", None)
        if ledger is None:
            return None

        try:
            await ledger.update_objective_status(task_id, "suspended")
            return task_id
        except Exception as e:
            logger.warning("Could not suspend heartbeat task %s after moral halt: %s", task_id, e)
            return None

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

    def _is_fallback_charter_active(self) -> bool:
        return str(getattr(self, "charter_text", "") or "").strip() == self._CHARTER_FALLBACK

    async def _consolidate_memory(self, user_id: str) -> None:
        """Run deep nocturnal consolidation pipeline (extract -> filter -> score -> write-back)."""
        try:
            turns = await self.ledger_memory.get_chat_history(user_id, limit=20)
            if len(turns) < 6:
                return

            recent_logs = await self.ledger_memory.get_logs(limit=20)
            critic_logs = [
                entry for entry in recent_logs
                if "critic" in str(entry.get("message", "")).lower()
                or str(entry.get("context", "")).lower().find("critic") != -1
            ]
            try:
                blueprint_entries = await self.vector_memory.query_memory_async(
                    "System 2 Reasoning Blueprint",
                    n_results=5,
                )
            except Exception:
                blueprint_entries = []

            filtered_candidates = await self.nocturnal_consolidation.extract_and_filter_candidates(
                user_id=user_id,
                chat_history=turns,
                worker_outputs={},
                critic_feedback="",
                blueprint_entries=blueprint_entries,
                ledger_logs=critic_logs,
                vector_memory=self.vector_memory,
            )

            if not filtered_candidates:
                await self.ledger_memory.trim_chat_history(user_id, keep_last=5)
                return

            scored_candidates = await self.nocturnal_consolidation.score_candidates_with_system2(
                filtered_candidates,
                route_to_system_2=self._score_nocturnal_candidates,
                redactor=cloud_redaction.redact_text_for_cloud,
                threshold=float(os.getenv("NOCTURNAL_Q_THRESHOLD", "3.0")),
            )
            write_result = await self.nocturnal_consolidation.write_back_scored_candidates(
                scored_candidates,
                core_memory=self.core_memory,
                vector_memory=self.vector_memory,
                ledger_memory=self.ledger_memory,
                threshold=float(os.getenv("NOCTURNAL_Q_THRESHOLD", "3.0")),
            )

            await self.ledger_memory.trim_chat_history(user_id, keep_last=5)
            if write_result.get("stored_total", 0) > 0:
                logger.info(
                    "Nocturnal consolidation stored memories for %s: %s",
                    user_id,
                    write_result,
                )
        except asyncio.TimeoutError:
            logger.warning("Memory consolidation skipped: LLM did not respond within 90s")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {type(e).__name__}: {e}", exc_info=True)

    async def _try_resume_mfa(self, user_id: str, user_message: str) -> Optional[str]:
        """Handle MFA challenge response. Returns a reply string, or None if not in MFA flow."""
        if user_id not in self.pending_mfa:
            return None
        pending_tool = self.pending_mfa[user_id]
        heartbeat_task_id = self._coerce_positive_task_id(pending_tool.get("_heartbeat_origin_task_id"))
        age = time.time() - pending_tool.get("_created_at", 0)
        if age > _PENDING_STATE_TTL_SECONDS:
            del self.pending_mfa[user_id]
            self._fire_and_forget(self.ledger_memory.clear_mfa_state(user_id))
            logger.info("Expired stale pending_mfa for %s (age %.0fs).", user_id, age)
            return None
        del self.pending_mfa[user_id]
        self._fire_and_forget(self.ledger_memory.clear_mfa_state(user_id))
        if not verify_mfa_challenge(user_message):
            failure_msg = "Error: MFA authorization failed. Action aborted."
            if heartbeat_task_id is not None:
                await self._finalize_resumed_heartbeat_task(heartbeat_task_id, failure_msg)
            return failure_msg
        exec_result = await self.cognitive_router._execute_tool(
            pending_tool["name"], pending_tool["arguments"]
        )
        if exec_result.status == "ok":
            if heartbeat_task_id is not None:
                await self._finalize_resumed_heartbeat_task(heartbeat_task_id, exec_result.content)
            return exec_result.content
        blocked_state = {
            "user_id": pending_tool.get("user_id", user_id),
        }
        if heartbeat_task_id is not None:
            blocked_state["_heartbeat_origin_task_id"] = heartbeat_task_id
        return await self._handle_blocked_result(
            exec_result,
            pending_tool.get("user_id", user_id),
            blocked_state,
        )

    async def _load_state(
        self,
        user_id: str,
        user_message: str,
        *,
        user_prompt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
                state["user_prompt"] = dict(user_prompt or {})
                state["current_plan"] = []
                state["_energy_gate_cleared"] = False
                state["_resumed_from_hitl"] = True
                heartbeat_task_id = self._extract_heartbeat_origin_task_id_from_state(state)
                if heartbeat_task_id is not None:
                    state["_heartbeat_origin_task_id"] = heartbeat_task_id
                # Track how many HITL cycles have been spent on this task (ISSUE-005)
                state["hitl_count"] = state.get("hitl_count", 0) + 1
                # Cap recharge to 75 so energy budget is never fully restored — this
                # ensures a perpetually failing task eventually becomes unrecoverable
                # rather than cycling indefinitely at full energy (ISSUE-005).
                state["energy_remaining"] = min(state.get("energy_remaining", 0) + 50, 75)
                self._apply_hitl_resume_limits(state, user_id)
                return normalize_state(state)
        state = AgentState.new(user_id=user_id, user_input=user_message, user_prompt=user_prompt).to_dict()
        state["energy_remaining"] = await self._get_predictive_energy_budget_remaining()
        state["user_prompt"] = dict(user_prompt or {})
        if user_id != "heartbeat":
            try:
                active_session = await self._get_active_session()
                get_session_history = getattr(
                    self.ledger_memory,
                    "get_session_chat_history",
                    None,
                )
                if (
                    active_session
                    and active_session.get("id")
                    and callable(get_session_history)
                    and inspect.iscoroutinefunction(get_session_history)
                ):
                    state["chat_history"] = (
                        await get_session_history(
                            user_id=user_id,
                            session_id=int(active_session["id"]),
                            limit=max(_RECENT_CHAT_HISTORY_LIMIT, 20),
                        )
                    )
                    state["active_session"] = active_session
                else:
                    state["chat_history"] = await self.ledger_memory.get_chat_history(
                        user_id,
                        limit=_RECENT_CHAT_HISTORY_LIMIT,
                    )
                    state["active_session"] = None
            except Exception as e:
                logger.warning(f"Failed to load chat history for {user_id}: {e}")
                state["active_session"] = None
        return normalize_state(state)

    @staticmethod
    def _apply_hitl_resume_limits(state: Dict[str, Any], user_id: str) -> None:
        if int(state.get("iteration_count") or 0) >= 3:
            state["final_response"] = (
                "This task reached the critic rejection limit and has been abandoned "
                "to prevent repeated unsafe retries. Please rephrase or break it into smaller steps."
            )
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

    @staticmethod
    def _has_ready_final_response(state: Dict[str, Any]) -> bool:
        return bool(state.get("final_response") and not state.get("current_plan"))

    async def _consume_blocked_result(self, state: Dict[str, Any], user_id: str) -> Optional[str]:
        blocked_result = state.pop(_BLOCKED_KEY, None)
        if blocked_result is None:
            return None
        return await self._handle_blocked_result(blocked_result, user_id, state)

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

    async def _run_manual_graph_pass(
        self,
        state: Dict[str, Any],
        user_id: str,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        for node in (self.supervisor_node, self.execute_workers_node, self.critic_node):
            state = await node(state)
            blocked_response = await self._consume_blocked_result(state, user_id)
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
            return state, await self._consume_blocked_result(state, user_id)
        return await self._run_manual_graph_pass(state, user_id)

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

    async def _persist_chat_turns(
        self,
        user_id: str,
        user_message: str,
        final_resp: str,
        turn_failed: bool = False,
        session_id: Optional[int] = None,
    ) -> None:
        if user_id == "heartbeat" or turn_failed:
            return
        try:
            save_with_session = getattr(
                self.ledger_memory,
                "save_chat_turn_with_session",
                None,
            )
            if callable(save_with_session) and inspect.iscoroutinefunction(save_with_session):
                await save_with_session(
                    user_id,
                    "user",
                    user_message,
                    session_id=session_id,
                )
                await save_with_session(
                    user_id,
                    "assistant",
                    final_resp,
                    session_id=session_id,
                )
            else:
                await self.ledger_memory.save_chat_turn(user_id, "user", user_message)
                await self.ledger_memory.save_chat_turn(user_id, "assistant", final_resp)
            if session_id:
                increment_turn_count = getattr(
                    self.ledger_memory,
                    "increment_session_turn_count",
                    None,
                )
                if callable(increment_turn_count) and inspect.iscoroutinefunction(increment_turn_count):
                    self._fire_and_forget(increment_turn_count(session_id))
        except Exception as e:
            logger.warning(f"Failed to save chat turn for {user_id}: {e}")

        counts = self._coerce_consolidation_turn_counts()

        trigger_turns = max(1, _CONSOLIDATION_TRIGGER_TURNS)
        new_count = int(counts.get(user_id, 0)) + 1
        counts[user_id] = new_count
        counts.move_to_end(user_id)
        self._evict_ordered_mapping_to_size(
            counts,
            _CONSOLIDATION_TURN_COUNT_MAX_USERS,
            label="Consolidation turn counts",
        )
        self._persist_consolidation_turn_count_async(user_id, new_count)

        if new_count >= trigger_turns:
            counts[user_id] = 0
            counts.move_to_end(user_id)
            self._persist_consolidation_turn_count_async(user_id, 0)
            self._fire_and_forget(self._consolidate_memory(user_id))

    def _schedule_response_memory_save(
        self,
        user_message: str,
        final_resp: str,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> None:
        self._fire_and_forget(
            self._save_memory_async(
                f"User: {user_message}\nAssistant: {final_resp}",
                session_id=session_id,
                epic_id=epic_id,
            )
        )

    async def _call_finalize_user_response(
        self,
        user_id: str,
        user_message: str,
        response: str,
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call _finalize_user_response with backward-compatible positional args only.

        Some tests monkeypatch _finalize_user_response with the legacy 3-arg
        callable signature. Session context is provided via an internal state
        hint so the default implementation can still resolve session metadata.
        """
        previous_hint = getattr(self, "_finalize_state_hint", None)
        if isinstance(state, dict):
            self._finalize_state_hint = state
        try:
            return await self._finalize_user_response(user_id, user_message, response)
        finally:
            self._finalize_state_hint = previous_hint

    async def _finalize_user_response(
        self,
        user_id: str,
        user_message: str,
        response: str,
        state: Optional[Dict[str, Any]] = None,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> str:
        final_resp = self.cognitive_router.sanitize_response(response)
        resolved_state: Optional[Dict[str, Any]] = state if isinstance(state, dict) else None
        if resolved_state is None:
            hinted_state = getattr(self, "_finalize_state_hint", None)
            if isinstance(hinted_state, dict):
                resolved_state = hinted_state

        turn_failed = bool((resolved_state or {}).get("_turn_failed", False))
        if resolved_state is None:
            turn_failed = bool(getattr(self, "_finalizing_turn_failed", False))

        resolved_session_id = session_id
        resolved_epic_id = epic_id
        if isinstance(resolved_state, dict):
            active_session = resolved_state.get("active_session")
            if isinstance(active_session, dict):
                if resolved_session_id is None:
                    raw_session_id = active_session.get("id")
                    try:
                        resolved_session_id = int(raw_session_id) if raw_session_id else None
                    except (TypeError, ValueError):
                        resolved_session_id = None
                if resolved_epic_id is None:
                    raw_epic_id = active_session.get("epic_id")
                    try:
                        resolved_epic_id = int(raw_epic_id) if raw_epic_id else None
                    except (TypeError, ValueError):
                        resolved_epic_id = None

        await self._persist_chat_turns(
            user_id,
            user_message,
            final_resp,
            turn_failed=turn_failed,
            session_id=resolved_session_id,
        )
        self._schedule_response_memory_save(
            user_message,
            final_resp,
            session_id=resolved_session_id,
            epic_id=resolved_epic_id,
        )
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
            logger.warning(f"Critic rejected output on iteration {state['iteration_count']}")
            # Clear the stale plan so the supervisor re-plans with the Critic
            # feedback injected via state["critic_instructions"] (ISSUE-006).
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

    async def _run_user_turn_locked(
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
            logger.info("Audio prompt detected for %s; bypassing text-only fast-path memory hooks.", effective_user_id)

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
            # Persist so the state survives a bot restart (ISSUE-013)
            self._fire_and_forget(self.ledger_memory.save_hitl_state(pending_user_id, pending_state))
            if heartbeat_task_id is not None:
                await self._notify_admin(str(hitl_err))
            return str(hitl_err)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            state["_turn_failed"] = True
            return "An internal error occurred."

    async def process_message(
        self,
        user_message: Any,
        user_id: str,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """Main entry point: State Graph execution with Energy Budget."""
        user_prompt = self._coerce_user_prompt_payload(user_message)
        normalized_user_message = str(user_prompt.get("text") or "").strip()
        has_audio_prompt = bool(self._extract_audio_bytes(user_prompt))
        approval_outcome: Optional[Any] = None

        if not normalized_user_message and has_audio_prompt:
            normalized_user_message = _format_voice_placeholder(
                self._extract_audio_bytes(user_prompt),
                str(user_prompt.get("audio_mime_type") or "audio/ogg"),
            )
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

            # Serialise concurrent messages for the same user_id to prevent
            # race conditions on pending_mfa / pending_hitl_state / pending_tool_approval
            # dicts (ISSUE-012).  Different users are still processed concurrently.
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

    async def _handle_blocked_result(
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
            # Persist so the HITL state survives a bot restart (ISSUE-013)
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

            # Run synthesis in the background via _fire_and_forget which holds
            # a strong GC-safe reference (ISSUE-002).
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

        # Fallback
        return f"An unexpected router status was received: {result.status}"

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
            "Additionally, generate a brief 'Reasoning Blueprint' on how to solve this class of problem. "
            "You MUST format your output strictly using XML tags:\n"
            "<solution> The actual answer to the user's problem... </solution>\n"
            "<blueprint> How to solve this problem: Step 1, Step 2... </blueprint>"
        )

        messages = [{"role": "user", "content": prompt}]
        sys2_result = await self._route_to_system_2_redacted(
            messages,
            purpose="cognitive_escalation",
            allow_sensitive_context=False,
        )

        content = sys2_result.content if sys2_result.status == "ok" else "[System 2 - Error]: Escalation failed."

        solution_match = re.search(r"<solution>(.*?)</solution>", content, flags=re.DOTALL | re.IGNORECASE)
        blueprint_match = re.search(r"<blueprint>(.*?)</blueprint>", content, flags=re.DOTALL | re.IGNORECASE)

        solution_text = solution_match.group(1).strip() if solution_match else content.strip()
        blueprint_text = blueprint_match.group(1).strip() if blueprint_match else None

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

    def close(self) -> None:
        try:
            set_runtime_context(None, None, None, None)
            if hasattr(self, 'vector_memory') and self.vector_memory:
                self.vector_memory.close()
            # cognitive_router and ledger_memory are async resources.
            # They are closed by _async_run's finally block in telegram_bot.py.
            # This synchronous close() path is only used in test/script contexts
            # where those resources were never opened asynchronously.
            logger.info(
                "Synchronous close(): async resources (router, ledger) "
                "must be closed by the async shutdown path."
            )
            logger.info("Orchestrator resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing Orchestrator: {e}", exc_info=True)
