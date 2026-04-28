from __future__ import annotations

"""
Orchestrator Module - Implements State Graph Architecture (Sprint 6)

State Graph with Energy Budget, Proactive Heartbeat, and Charter enforcement.
Passes a State Dictionary between specialized nodes until task is complete,
exhausted (energy = 0), or blocked (HITL required).
"""

import os
import platform
import logging
import hashlib
import asyncio
import math
import re
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable
import xml.etree.ElementTree as ET

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from src.memory.vector_db import VectorMemory
from src.memory.ledger_db import LedgerMemory
from src.memory.core_memory import CoreMemory
from src.skills.read_inbox import read_inbox
from src.core.agent_definition import AgentDefinition
from src.core.agent_registry import AgentRegistry
from src.core.prompt_config import load_prompt_config, build_supervisor_prompt
from src.core.runtime_context import set_runtime_context
from src.core.state_model import AgentState, normalize_state
from src.core.workflow_graph import build_orchestrator_graph
from src.core.security import validate_mfa_configuration, verify_mfa_challenge
from src.core.orchestration import capabilities as capability_helpers
from src.core.orchestration import energy as energy_service, llm_gateway
from src.core.orchestration import memory_service, pending_state, schedulers
from src.core.orchestration import prompt_payload
from src.core.progress import ProgressEmitter, reset_emitter, set_current_emitter
from src.core.nocturnal_consolidation import NocturnalConsolidationSlice1
from src.core.goal_planner import GoalPlanner, PlanningResult
from src.core.energy_judge import EnergyJudge, EnergyEvaluation
from src.core.energy_roi_engine import EnergyROIEngine, EnergyDecision
from src.core.synthesis_pipeline import SynthesisPipeline
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

logger = logging.getLogger(__name__)

class RequiresHITLError(RuntimeError):
    """Raised when a request requires human-in-the-loop approval."""

# Energy costs per operation
ENERGY_COST_SUPERVISOR = 10
ENERGY_COST_WORKER = 15
ENERGY_COST_CRITIC = 10
ENERGY_COST_TOOL = 5
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))  # default 30 minutes
EMAIL_POLL_INTERVAL = int(os.getenv("AIDEN_EMAIL_POLL_INTERVAL_SECONDS", "300"))  # default 5 minutes
MEMORY_SAVE_THRESHOLD = int(os.getenv("MEMORY_SAVE_THRESHOLD", "120"))
MEMORY_CONSOLIDATION_INTERVAL = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "21600"))  # 6 hours
# UTC timestamp format shared with VectorMemory.add_memory / prune_old_memories so that
# lexical comparisons inside ChromaDB remain consistent across all write paths.
_UTC_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S"
# How long an unanswered MFA/HITL/tool-approval entry stays alive before auto-expiry
_PENDING_STATE_TTL_SECONDS = int(os.getenv("PENDING_STATE_TTL_SECONDS", "86400"))  # 24 hours
_SYNTHESIS_SELF_TEST_TIMEOUT_DEFAULT_SECONDS = 12.0
_MAX_SYNTHESIS_RETRIES = int(os.getenv("MAX_SYNTHESIS_RETRIES", "3"))
_CONSOLIDATION_TRIGGER_TURNS = int(os.getenv("CONSOLIDATION_TRIGGER_TURNS", "10"))
_CONSOLIDATION_TURN_COUNTS_MAX_SIZE_DEFAULT = 100
_SYNTHESIS_LOCKOUT_TTL_SECONDS = int(os.getenv("SYNTHESIS_LOCKOUT_TTL_SECONDS", "600"))
_MAX_SYNTHESIS_FAILURES_BEFORE_MANUAL_INTERVENTION = 2
_SAFE_SUBPROCESS_ENV_KEYS = {
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
}
_BLOCKED_ENV_PREFIXES = ("TELEGRAM_", "GROQ_", "GEMINI_", "ANTHROPIC_", "OPENAI_", "OLLAMA_CLOUD_", "ADMIN_")
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
_HEARTBEAT_FAILURE_STRIKES = int(os.getenv("HEARTBEAT_FAILURE_STRIKES", "3"))
_HEARTBEAT_FAILURE_STATE_KEY = "heartbeat_task_failure_counts"
HEARTBEAT_TASK_PREFIX_FMT = "[HEARTBEAT TASK #{task_id}]"
_NOTIFIABLE_DOMAIN_RESULTS = {"google"}
_DOMAIN_RESULT_SUMMARY_MAX_LENGTH = 220
_ERROR_RESPONSE_PREFIXES = (
    "Supervisor encountered an error",
    "Both local and cloud reasoning failed",
    "I was unable to process this request locally",
    "Planning timed out",
    "Unable to fulfill this request",
    "No valid response could be generated",
    "An internal error occurred",
)
_VOICE_INPUT_PLACEHOLDER = "[Voice note]"
_CLOUD_REDACTION_BLOCK_PATTERNS = (
    r"<context_and_memory>.*?</context_and_memory>",
    r"<Core_Working_Memory>.*?</Core_Working_Memory>",
    r"<Archival_Memory>.*?</Archival_Memory>",
    r"<chat_history>.*?</chat_history>",
)
_CLOUD_REDACTION_LINE_PATTERNS = (
    (r"\[Machine Context[^\]]*\].*?(?=\n\s*\n|$)", "[REDACTED_SENSORY_STATE]"),
    (
        r"^\s*(Host[_ ]?OS|OS|CPU(?:[_ ]?usage)?|CWD|Platform|Memory Usage|Disk Usage)\s*[:=].*$",
        "[REDACTED_SENSORY_STATE]",
    ),
    (
        r"<\s*(host_os|os|cpu_usage|cwd|machine|platform)\s*>.*?<\s*/\s*(host_os|os|cpu_usage|cwd|machine|platform)\s*>",
        "[REDACTED_SENSORY_STATE]",
    ),
)
_CLOUD_REDACTION_PII_PATTERNS = (
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]"),
    (r"\+?\d[\d\-\s().]{7,}\d", "[REDACTED_PHONE]"),
    (
        r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
        "[REDACTED_ADDRESS]",
    ),
    (r"\b(?:my name is|i am|i'm)\s+[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2}\b", "[REDACTED_NAME]"),
    (r"\b(?:i live in|located in|from)\s+[A-Za-z][A-Za-z'\-]+(?:[\s,]+[A-Za-z][A-Za-z'\-]+){0,2}\b", "[REDACTED_LOCATION]"),
    (r"\b(?:user[_ ]?id|ssn|social security|passport|account number)\b[^\n]*", "[REDACTED_PII_FIELD]"),
    (r"[A-Za-z]:\\[^\n]*", "[REDACTED_PATH]"),
)
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

def _build_safe_subprocess_env() -> Dict[str, str]:
    safe_env: Dict[str, str] = {
        key: value
        for key in _SAFE_SUBPROCESS_ENV_KEYS
        if (value := os.environ.get(key))
    }
    safe_env["PYTHONPATH"] = os.getcwd()
    safe_env.setdefault("PYTHONIOENCODING", "utf-8")
    safe_env.setdefault("PYTHONUNBUFFERED", "1")
    return safe_env

class Orchestrator:
    """
    Central orchestration engine using a State Graph architecture.

    State Dictionary schema:
        user_id, user_input, final_response, iteration_count,
        admin_guidance, energy_remaining

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
            router_factory = getattr(llm_gateway, "CognitiveRouter", None)
            if not callable(router_factory):
                raise RuntimeError(
                    "CognitiveRouter is unavailable. Configure LLM gateway wiring before initializing Orchestrator."
                )
            self.cognitive_router = router_factory(model_name=gemini_model, local_model=local_model)
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
            self._predictive_energy_budget_remaining = max(
                0,
                int(os.getenv("INITIAL_ENERGY_BUDGET", "100")),
            )
            self._predictive_energy_budget_last_replenished_at = time.time()
            self._predictive_energy_budget_lock: asyncio.Lock = asyncio.Lock()

            self.charter_text = self._load_charter()
            self._ready: asyncio.Event = asyncio.Event()
            self.pending_mfa: Dict[str, dict] = {}
            self.pending_hitl_state: Dict[str, dict] = {}
            self.pending_tool_approval: Dict[str, dict] = {}
            self._consolidation_turn_counts: OrderedDict = OrderedDict()
            self._synthesis_in_progress: Dict[str, float] = {}
            self.outbound_queue: Optional[asyncio.Queue] = None
            self.synthesis_pipeline = self._build_synthesis_pipeline()
            self.sensory_state: Dict[str, str] = {}
            # Background task registry — holds strong references to prevent GC (ISSUE-002)
            self._background_tasks: set = set()
            # Per-user async locks to serialise concurrent messages (ISSUE-012)
            self._user_locks: Dict[str, asyncio.Lock] = {}
            self._user_locks_lock: asyncio.Lock = asyncio.Lock()
            self._compiled_graph = None
            self._heartbeat_failure_counts: Dict[int, int] = {}
            self._email_poll_last_run_at: Optional[datetime] = None
            self._email_processed_timestamps: List[float] = []
            self._refresh_sensory_state()

            logger.info("Orchestrator __init__ complete — call async_init() to finish setup")
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
            raise

    async def _noop_async(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _build_synthesis_pipeline(self) -> SynthesisPipeline:
        on_tools_changed = getattr(self, "_invalidate_capabilities_cache", None)
        if not callable(on_tools_changed):
            on_tools_changed = lambda: None

        process_message_fn = getattr(self, "process_message", None)
        if not callable(process_message_fn):
            process_message_fn = self._noop_async

        route_to_system_2_redacted = getattr(self, "_route_to_system_2_redacted", None)
        if not callable(route_to_system_2_redacted):
            route_to_system_2_redacted = self._noop_async

        fire_and_forget = getattr(self, "_fire_and_forget", None)
        if not callable(fire_and_forget):
            fire_and_forget = lambda coro: asyncio.create_task(coro)

        notify_admin = getattr(self, "_notify_admin", None)
        if not callable(notify_admin):
            notify_admin = self._noop_async

        if not hasattr(self, "pending_tool_approval"):
            self.pending_tool_approval = {}

        return SynthesisPipeline(
            cognitive_router=getattr(self, "cognitive_router", None),
            ledger_memory=getattr(self, "ledger_memory", None),
            core_memory=getattr(self, "core_memory", None),
            pending_tool_approval=self.pending_tool_approval,
            on_tools_changed=on_tools_changed,
            process_message_fn=process_message_fn,
            route_to_system_2_redacted=route_to_system_2_redacted,
            fire_and_forget=fire_and_forget,
            notify_admin=notify_admin,
            outbound_queue=getattr(self, "outbound_queue", None),
        )

    def _get_synthesis_pipeline(self) -> SynthesisPipeline:
        pipeline = getattr(self, "synthesis_pipeline", None)
        if pipeline is None:
            pipeline = self._build_synthesis_pipeline()
            self.synthesis_pipeline = pipeline

        pipeline.cognitive_router = getattr(self, "cognitive_router", None)
        pipeline.ledger_memory = getattr(self, "ledger_memory", None)
        pipeline.core_memory = getattr(self, "core_memory", None)
        pipeline.pending_tool_approval = getattr(self, "pending_tool_approval", {})
        pipeline.outbound_queue = getattr(self, "outbound_queue", None)

        instance_dict = getattr(self, "__dict__", {})
        if "_run_synthesis_self_test" in instance_dict:
            pipeline._run_synthesis_self_test = instance_dict["_run_synthesis_self_test"]
        if "_execute_synthesis_repair_loop" in instance_dict:
            pipeline._execute_synthesis_repair_loop = instance_dict["_execute_synthesis_repair_loop"]
        return pipeline

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
        await self._restore_persisted_groq_cooldown()

        await self.ledger_memory.seed_initial_goals()
        restart_callback = getattr(self.cognitive_router, "set_dynamic_tool_restart_callback", None)
        if callable(restart_callback):
            restart_callback(self._reload_dynamic_tools_after_worker_restart)
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
        set_runtime_context(
            self.ledger_memory,
            self.core_memory,
            self.vector_memory,
            orchestrator=self,
        )
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

        # Restore per-user consolidation counters persisted by _persist_chat_turns.
        try:
            stored = await self.ledger_memory.get_system_states_by_prefix("consolidation_turns:")
            if not hasattr(self, "_consolidation_turn_counts") or self._consolidation_turn_counts is None:
                self._consolidation_turn_counts = OrderedDict()
            for key, val in stored.items():
                uid = key[len("consolidation_turns:"):]
                if uid:
                    self._consolidation_turn_counts[uid] = int(val or 0)
        except Exception as _ce:
            logger.warning("Consolidation turn count restore failed: %s", _ce)

        self._enforce_charter_policy()
        self._ready.set()
        logger.info("Orchestrator async_init complete")

    async def _restore_persisted_groq_cooldown(self) -> None:
        saved_cooldown = None
        try:
            saved_cooldown = await self.ledger_memory.get_system_state("groq_cooldown_until")
        except Exception as _ce:
            msg = f"WARNING: Groq cooldown restoration failed: {_ce}"
            logger.warning("Failed to restore Groq cooldown from DB: %s", _ce)
            if hasattr(self, "outbound_queue"):
                try:
                    self.outbound_queue.put_nowait(msg)
                except Exception:
                    pass
            return

        if not saved_cooldown:
            return

        try:
            expiry = float(saved_cooldown)
            now = time.time()
            max_expiry = now + 7200.0
            if not math.isfinite(expiry):
                logger.warning("Invalid groq_cooldown_until value in DB: non-finite (%r)", saved_cooldown)
            elif now < expiry <= max_expiry:
                self.cognitive_router._system2_cooldown_until = expiry
                logger.info(
                    "Restored Groq rate-limit cooldown: %.0fs remaining.",
                    expiry - now,
                )
            elif expiry > max_expiry:
                logger.warning("Persisted Groq cooldown exceeded 2-hour cap; ignoring.")
        except (TypeError, ValueError) as e:
            logger.warning("Invalid groq_cooldown_until value in DB: %s", e)

    def _enforce_charter_policy(self) -> None:
        validate_mfa_configuration()
        allow_missing = os.getenv("ALLOW_MISSING_CHARTER", "false").strip().lower() in {"1", "true", "yes"}
        if getattr(self, "charter_text", "") == self._CHARTER_FALLBACK:
            if not allow_missing:
                raise RuntimeError(
                    "FATAL: charter.md not found or empty. The moral evaluation framework cannot operate. "
                    "Place a valid charter.md in the working directory or set ALLOW_MISSING_CHARTER=true "
                    "to explicitly permit degraded operation."
                )
            logger.warning(
                "SECURITY: Running without a full charter. Charter enforcement will be minimal."
            )
            return
        try:
            ET.fromstring(str(self.charter_text or ""))
        except ET.ParseError as exc:
            if not allow_missing:
                raise RuntimeError(f"FATAL: malformed charter.md XML: {exc}") from exc
            logger.warning("SECURITY: Malformed charter allowed by override; using fallback charter.")
            self.charter_text = self._CHARTER_FALLBACK

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
        await schedulers.sensory_update_loop(
            refresh_sensory_state=self._refresh_sensory_state,
            interval_seconds=60,
        )

    def _get_sensory_context(self) -> str:
        s = self.sensory_state
        return (
            f"[Machine Context — {s.get('current_time', 'unknown')} | "
            f"OS: {s.get('os', 'unknown')} | "
            f"CPU: {s.get('cpu_usage', 'unknown')} | "
            f"CWD: {s.get('cwd', 'unknown')}]"
        )

    @staticmethod
    def _extract_audio_bytes(payload: Dict[str, Any]) -> bytes:
        return prompt_payload.extract_audio_bytes(payload)

    @classmethod
    def _coerce_user_prompt_payload(cls, user_message: Any) -> Dict[str, Any]:
        return prompt_payload.coerce_user_prompt_payload(user_message)

    @classmethod
    def _build_user_prompt_message(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        return prompt_payload.build_user_prompt_message(state)

    @classmethod
    def _strip_audio_bytes_for_persistence(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        return prompt_payload.strip_audio_bytes_for_persistence(state)

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
            # Guard against negated phrases like "no, my name is not there" or
            # "my name is not available" by checking for a negation word that
            # immediately precedes the match, and by rejecting names whose
            # first word is a common non-name stop word.
            _STOP_NAME_WORDS = frozenset({
                "not", "there", "here", "anywhere", "nowhere",
                "nothing", "everything", "something", "available",
                "known", "given", "provided",
            })
            match_start = name_match.start()
            preceding = text[max(0, match_start - 30):match_start].lower()
            is_negated = bool(re.search(r"\b(no|not|n't|never|nope)\s*$", preceding.rstrip()))
            raw_name = Orchestrator._normalize_display_name(name_match.group(1))
            first_word = raw_name.split()[0].lower() if raw_name else ""
            if raw_name and not is_negated and first_word not in _STOP_NAME_WORDS:
                updates["name"] = raw_name

        identity_match = re.search(
            r"(?:^|[.!?,]\s*)([a-z][a-z' -]{0,60})\s+is me\b(?=,|\.|!|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if identity_match and "name" not in updates:
            raw_name = Orchestrator._normalize_display_name(identity_match.group(1))
            if raw_name:
                updates["name"] = raw_name

        age_match = re.search(
            r"\b(?:i(?:'m|\s+am)(?:\s+also)?)\s+(\d{1,3})\s*years?\s*old\b",
            lowered,
        )
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
        return await memory_service.remember_user_profile(
            core_memory=getattr(self, "core_memory", None),
            user_id=user_id,
            user_message=user_message,
            extract_updates=self._extract_user_profile_updates,
        )

    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return await memory_service.get_user_profile(
            core_memory=getattr(self, "core_memory", None),
            user_id=user_id,
        )

    async def _remember_assistant_identity(self, user_message: str) -> Optional[str]:
        return await memory_service.remember_assistant_identity(
            core_memory=getattr(self, "core_memory", None),
            user_message=user_message,
            extract_update=self._extract_assistant_identity_update,
        )

    async def _get_assistant_name(self) -> str:
        return await memory_service.get_assistant_name(
            core_memory=getattr(self, "core_memory", None),
        )

    _CHARTER_FALLBACK = "Core Directive: Do no harm."

    def _load_charter(self, filepath: str = "charter.md") -> str:
        resolved = os.getenv("CHARTER_PATH", filepath)
        try:
            if os.path.exists(resolved):
                with open(resolved, "r", encoding="utf-8") as fh:
                    text = fh.read().strip()
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
        """Build an ultra-lean Level-1 catalog: name + description only."""
        return capability_helpers.get_capabilities_string(
            self._load_capability_catalog_entries()
        )

    def _build_capability_catalog_rows(self) -> List[str]:
        return capability_helpers.build_capability_catalog_rows(
            self._load_capability_catalog_entries()
        )

    def _load_capability_catalog_entries(self) -> List[Dict[str, str]]:
        registry = getattr(getattr(self, "cognitive_router", None), "registry", None)
        return capability_helpers.load_capability_catalog_entries(registry, log=logger)

    def _load_executable_capability_catalog_entries(self) -> List[Dict[str, str]]:
        registry = getattr(getattr(self, "cognitive_router", None), "registry", None)
        return capability_helpers.load_executable_capability_catalog_entries(registry, log=logger)

    def _build_scoped_skill_runtime_context(self, skill_name: str) -> str:
        """Load full SKILL.md body on demand for the current execution turn only."""
        registry = getattr(getattr(self, "cognitive_router", None), "registry", None)
        return capability_helpers.build_scoped_skill_runtime_context(
            registry,
            skill_name,
            self._load_capability_catalog_entries(),
        )

    @staticmethod
    def _capability_catalog_tokens(text: str) -> set[str]:
        return capability_helpers.capability_catalog_tokens(text)

    def _find_local_skill_catalog_match(
        self,
        gap_description: str,
        suggested_tool_name: str,
    ) -> Optional[Dict[str, str]]:
        return capability_helpers.find_local_skill_catalog_match(
            gap_description,
            suggested_tool_name,
            self._load_executable_capability_catalog_entries(),
        )

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
        return llm_gateway.is_system_1_error(result, _SYSTEM_1_ERROR_PREFIX)

    @staticmethod
    def _is_system_2_error(result: Optional[RouterResult]) -> bool:
        """Detect placeholder System 2 error payloads that should trigger fallback."""
        return llm_gateway.is_system_2_error(
            result,
            (_SYSTEM_2_ERROR_PREFIX, _SYSTEM_2_EMPTY_PREFIX),
        )

    @staticmethod
    def _redact_text_for_cloud(
        text: str,
        *,
        allow_sensitive_context: bool = False,
        max_chars: int = 12000,
    ) -> str:
        return llm_gateway.redact_text_for_cloud(
            text,
            allow_sensitive_context=allow_sensitive_context,
            max_chars=max_chars,
        )

    def _redact_messages_for_cloud(
        self,
        messages: List[Dict[str, str]],
        *,
        allow_sensitive_context: bool = False,
    ) -> List[Dict[str, str]]:
        return llm_gateway.redact_messages_for_cloud(
            messages,
            allow_sensitive_context=allow_sensitive_context,
        )

    @staticmethod
    def _cloud_payload_audit_sha256(
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]],
    ) -> str:
        return llm_gateway.cloud_payload_audit_sha256(messages, allowed_tools)

    async def _append_cloud_payload_audit_if_supported(
        self,
        *,
        purpose: str,
        message_count_before: int,
        message_count_after: int,
        allow_sensitive_context: bool,
        payload_sha256: str,
    ) -> None:
        await llm_gateway.append_cloud_payload_audit_if_supported(
            getattr(self, "ledger_memory", None),
            purpose=purpose,
            message_count_before=message_count_before,
            message_count_after=message_count_after,
            allow_sensitive_context=allow_sensitive_context,
            payload_sha256=payload_sha256,
            log=logger,
        )

    async def _route_to_system_2_redacted(
        self,
        messages: List[Dict[str, str]],
        *,
        allowed_tools: Optional[List[str]] = None,
        purpose: str = "system_2",
        allow_sensitive_context: bool = False,
    ) -> RouterResult:
        return await llm_gateway.route_to_system_2_redacted(
            self.cognitive_router,
            getattr(self, "ledger_memory", None),
            messages,
            allowed_tools=allowed_tools,
            purpose=purpose,
            allow_sensitive_context=allow_sensitive_context,
            log=logger,
        )

    def _get_system_1_gate_metrics(self) -> Dict[str, Any]:
        return llm_gateway.get_system_1_gate_metrics(
            getattr(self, "cognitive_router", None),
            log=logger,
        )

    def _log_system_1_gate_pressure(
        self,
        context: str,
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
    ) -> None:
        llm_gateway.log_system_1_gate_pressure(
            context,
            before_metrics,
            after_metrics,
            log=logger,
        )

    async def _route_to_system_1(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None,
        *,
        deadline_seconds: Optional[float] = None,
        context: str = "orchestrator",
    ) -> RouterResult:
        return await llm_gateway.route_to_system_1(
            self.cognitive_router,
            messages,
            allowed_tools=allowed_tools,
            deadline_seconds=deadline_seconds,
            context=context,
            log=logger,
        )

    async def _get_archival_context(self, query: str) -> str:
        """Retrieve top archival memory snippets relevant to the user query."""
        return await memory_service.get_archival_context(
            vector_memory=self.vector_memory,
            query=query,
            max_chunk_chars=int(os.getenv("MAX_ARCHIVAL_CHUNK_CHARS", "2000")),
            max_total_chars=int(os.getenv("MAX_ARCHIVAL_TOTAL_CHARS", "6000")),
            log=logger,
        )

    def _energy_gate_available(self) -> bool:
        return energy_service.energy_gate_available(self)

    async def _get_predictive_energy_budget_remaining(self) -> int:
        return await energy_service.get_predictive_energy_budget_remaining(
            self,
            gate_available=self._energy_gate_available,
        )

    @staticmethod
    def _resolve_energy_replenish_per_hour() -> float:
        return energy_service.resolve_energy_replenish_per_hour(log=logger)

    @staticmethod
    def _resolve_energy_replenish_per_heartbeat() -> int:
        return energy_service.resolve_energy_replenish_per_heartbeat(log=logger)

    @classmethod
    def _resolve_energy_replenish_per_turn(cls) -> int:
        return energy_service.resolve_energy_replenish_per_turn(cls, log=logger)

    @classmethod
    def _warn_deprecated_energy_replenish_turn_env_once(cls) -> None:
        energy_service.warn_deprecated_energy_replenish_turn_env_once(cls, log=logger)

    def _compute_predictive_energy_replenishment_points_wallclock_locked(self, *, now: Optional[float] = None) -> int:
        return energy_service.compute_predictive_energy_replenishment_points_wallclock_locked(
            self,
            now=now,
            resolve_per_hour=self._resolve_energy_replenish_per_hour,
            time_fn=time.time,
        )

    def _apply_predictive_energy_tick_locked(self, amount: int) -> int:
        return energy_service.apply_predictive_energy_tick_locked(self, amount)

    def _replenish_predictive_energy_budget_wallclock_locked(self, *, now: Optional[float] = None) -> int:
        return energy_service.replenish_predictive_energy_budget_wallclock_locked(
            self,
            now=now,
            warn_deprecated=self._warn_deprecated_energy_replenish_turn_env_once,
            compute_points=self._compute_predictive_energy_replenishment_points_wallclock_locked,
            apply_tick=self._apply_predictive_energy_tick_locked,
        )

    async def _refund_predictive_energy_budget(self, amount: int, reason: str) -> None:
        await energy_service.refund_predictive_energy_budget(
            self,
            amount,
            reason,
            gate_available=self._energy_gate_available,
            log=logger,
        )

    async def _tick_predictive_energy_budget(self, amount: int, reason: str) -> int:
        return await energy_service.tick_predictive_energy_budget(
            amount,
            reason,
            get_remaining=self._get_predictive_energy_budget_remaining,
            refund=self._refund_predictive_energy_budget,
        )

    async def _try_reserve_predictive_energy_budget(
        self,
        *,
        predicted_cost: int,
        min_reserve: int,
        reason: str,
    ) -> bool:
        return await energy_service.try_reserve_predictive_energy_budget(
            self,
            predicted_cost=predicted_cost,
            min_reserve=min_reserve,
            reason=reason,
            gate_available=self._energy_gate_available,
            log=logger,
        )

    @staticmethod
    def _build_energy_evaluation_record(
        *,
        evaluation: EnergyEvaluation,
        decision: EnergyDecision,
        available_energy: int,
        context: str,
    ) -> Dict[str, Any]:
        return energy_service.build_energy_evaluation_record(
            evaluation=evaluation,
            decision=decision,
            available_energy=available_energy,
            context=context,
        )

    async def _route_energy_judge_messages(self, messages: List[Dict[str, str]]) -> RouterResult:
        return await energy_service.route_energy_judge_messages(self, messages)

    async def _evaluate_energy_for_context(
        self,
        *,
        task: Dict[str, Any],
        story: Optional[Dict[str, Any]],
        epic: Optional[Dict[str, Any]],
        additional_context: str,
    ) -> tuple[EnergyEvaluation, EnergyDecision, int]:
        return await energy_service.evaluate_energy_for_context(
            self,
            task=task,
            story=story,
            epic=epic,
            additional_context=additional_context,
        )

    def _build_synthesized_ad_hoc_energy_context(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        return energy_service.build_synthesized_ad_hoc_energy_context(self, state)

    @staticmethod
    def _format_ad_hoc_energy_deferral_message(
        *,
        decision: EnergyDecision,
        available_energy: int,
    ) -> str:
        return energy_service.format_ad_hoc_energy_deferral_message(
            decision=decision,
            available_energy=available_energy,
        )

    async def _try_ad_hoc_dispatch_energy_gate(
        self,
        state: Dict[str, Any],
        *,
        dispatch_context: str,
    ) -> Optional[str]:
        return await energy_service.try_ad_hoc_dispatch_energy_gate(
            self,
            state,
            dispatch_context=dispatch_context,
            log=logger,
        )

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
                redactor=self._redact_text_for_cloud,
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
        return self._format_goal_planner_ack(result)

    def _deduct_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        """Deduct energy and raise HITL if exhausted."""
        return energy_service.deduct_energy(
            state,
            amount,
            reason,
            RequiresHITLError,
            log=logger,
        )

    def _refund_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        """Refund energy deducted by a failed or timed-out operation."""
        return energy_service.refund_energy(state, amount, reason, log=logger)

    @staticmethod

    def _contains_any_hint(text: str, hints: tuple[str, ...]) -> bool:
        lowered = str(text or "").lower()
        return any(hint in lowered for hint in hints)

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

    async def _notify_admin(self, message: str) -> None:
        """Send a message to the admin via the outbound queue (used by heartbeat)."""
        await schedulers.notify_admin(getattr(self, "outbound_queue", None), message, log=logger)

    @staticmethod
    def _heartbeat_result_indicates_failure(result: str) -> bool:
        return schedulers.heartbeat_result_indicates_failure(result)

    @staticmethod
    def _build_heartbeat_execution_prompt(task: Dict[str, Any]) -> str:
        return schedulers.build_heartbeat_execution_prompt(task)

    @staticmethod
    def _is_executable_heartbeat_task_candidate(
        node: Dict[str, Any],
        unresolved_task_ids: set[int],
        parent_ids: set[int],
    ) -> bool:
        return schedulers.is_executable_heartbeat_task_candidate(
            node,
            unresolved_task_ids,
            parent_ids,
        )

    async def _select_executable_heartbeat_tasks(self) -> List[Dict[str, Any]]:
        """Return executable Task candidates with Story/Epic context for energy scoring."""
        return await schedulers.select_executable_heartbeat_tasks(
            ledger_memory=self.ledger_memory,
            is_candidate=self._is_executable_heartbeat_task_candidate,
        )

    async def _select_executable_heartbeat_task(self) -> Optional[Dict[str, Any]]:
        """Pick the next executable Task leaf with all dependencies resolved."""
        return await schedulers.select_executable_heartbeat_task(
            self._select_executable_heartbeat_tasks,
        )

    @staticmethod
    def _summarize_domain_task_result(result_payload: Dict[str, Any]) -> str:
        return schedulers.summarize_domain_task_result(
            result_payload,
            max_length=_DOMAIN_RESULT_SUMMARY_MAX_LENGTH,
        )

    @staticmethod
    def _extract_domain_task_error(result_payload: Dict[str, Any]) -> str:
        return schedulers.extract_domain_task_error(
            result_payload,
            max_length=_DOMAIN_RESULT_SUMMARY_MAX_LENGTH,
        )

    async def _notify_completed_domain_tasks(self) -> None:
        await schedulers.notify_completed_domain_tasks(
            ledger_memory=self.ledger_memory,
            notify_admin_fn=self._notify_admin,
            summarize_result=self._summarize_domain_task_result,
            extract_error=self._extract_domain_task_error,
            notifiable_domains=_NOTIFIABLE_DOMAIN_RESULTS,
            log=logger,
        )

    async def _handle_heartbeat_task_failure(
        self,
        task: Dict[str, Any],
        *,
        reason: str,
        result_excerpt: str,
    ) -> None:
        await schedulers.handle_heartbeat_task_failure(
            task=task,
            reason=reason,
            result_excerpt=result_excerpt,
            ledger_memory=self.ledger_memory,
            increment_failure_count=self._increment_heartbeat_failure_count,
            notify_admin_fn=self._notify_admin,
            failure_strikes=_HEARTBEAT_FAILURE_STRIKES,
            log=logger,
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
        await energy_service.defer_heartbeat_task_due_to_energy(
            self,
            task_id=task_id,
            decision=decision,
            evaluation_record=evaluation_record,
            available_energy=available_energy,
            race=race,
            log=logger,
        )

    async def _execute_heartbeat_task(
        self,
        *,
        task: Dict[str, Any],
        decision: EnergyDecision,
    ) -> None:
        await schedulers.execute_heartbeat_task(
            task=task,
            decision=decision,
            ledger_memory=self.ledger_memory,
            process_message=self.process_message,
            get_predictive_energy_budget_remaining=self._get_predictive_energy_budget_remaining,
            build_prompt=self._build_heartbeat_execution_prompt,
            result_indicates_failure=self._heartbeat_result_indicates_failure,
            handle_failure=self._handle_heartbeat_task_failure,
            clear_failure_count=self._clear_heartbeat_failure_count,
            notify_admin_fn=self._notify_admin,
            heartbeat_interval=HEARTBEAT_INTERVAL,
            log=logger,
        )

    async def _run_heartbeat_cycle(self) -> None:
        def _ensure_energy_budget() -> None:
            if not hasattr(self, "_predictive_energy_budget_lock"):
                self._predictive_energy_budget_lock = asyncio.Lock()
            if not hasattr(self, "_predictive_energy_budget_remaining"):
                self._predictive_energy_budget_remaining = max(
                    0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100"))
                )

        _ensure_energy_budget()
        await schedulers.run_heartbeat_cycle(
            schedulers.HeartbeatCycleDeps(
                ensure_energy_budget=_ensure_energy_budget,
                predictive_energy_budget_lock=self._predictive_energy_budget_lock,
                apply_predictive_energy_tick_locked=self._apply_predictive_energy_tick_locked,
                resolve_energy_replenish_per_heartbeat=self._resolve_energy_replenish_per_heartbeat,
                notify_completed_domain_tasks=self._notify_completed_domain_tasks,
                select_executable_heartbeat_tasks=self._select_executable_heartbeat_tasks,
                evaluate_energy_for_context=self._evaluate_energy_for_context,
                build_energy_evaluation_record=self._build_energy_evaluation_record,
                defer_heartbeat_task_due_to_energy=self._defer_heartbeat_task_due_to_energy,
                try_reserve_predictive_energy_budget=self._try_reserve_predictive_energy_budget,
                get_predictive_energy_budget_remaining=self._get_predictive_energy_budget_remaining,
                energy_roi_engine=getattr(self, "energy_roi_engine", None),
                ledger_memory=self.ledger_memory,
                execute_heartbeat_task=self._execute_heartbeat_task,
                log=logger,
            )
        )

    async def _heartbeat_loop(self) -> None:
        """
        Proactive Heartbeat: wakes every 30 min, selects executable Objective
        leaf tasks (dependency-resolved), executes one, and applies remediation
        logic on repeated failures.

        An asyncio.Lock prevents heartbeat cycles from overlapping: if the
        previous run is still in progress when the next interval fires, the
        new cycle is skipped (ISSUE-004).
        """
        await schedulers.heartbeat_loop(
            run_heartbeat_cycle_fn=self._run_heartbeat_cycle,
            heartbeat_interval=HEARTBEAT_INTERVAL,
            log=logger,
        )

    def _prune_email_processed_history(self, now_ts: Optional[float] = None) -> None:
        self._email_processed_timestamps = schedulers.prune_email_processed_history(
            getattr(self, "_email_processed_timestamps", []),
            now_ts,
        )

    async def _run_email_poll_cycle(self) -> None:
        self._email_poll_last_run_at, self._email_processed_timestamps = await schedulers.run_email_poll_cycle(
            ledger_memory=self.ledger_memory,
            processed_timestamps=getattr(self, "_email_processed_timestamps", []),
            read_inbox_fn=read_inbox,
            now_fn=datetime.now,
            time_fn=time.time,
            log=logger,
        )

    async def _email_poll_loop(self) -> None:
        await schedulers.email_poll_loop(
            run_email_poll_cycle_fn=self._run_email_poll_cycle,
            email_poll_interval=EMAIL_POLL_INTERVAL,
            log=logger,
        )

    async def get_email_poll_status(self) -> Dict[str, Any]:
        self._prune_email_processed_history()
        return await schedulers.get_email_poll_status(
            ledger_memory=self.ledger_memory,
            last_run_at=getattr(self, "_email_poll_last_run_at", None),
            processed_timestamps=getattr(self, "_email_processed_timestamps", []),
        )

    async def _memory_consolidation_loop(self) -> None:
        """Background task: periodically consolidate chat history into long-term memory."""
        await schedulers.memory_consolidation_loop(
            ledger_memory=self.ledger_memory,
            vector_memory=self.vector_memory,
            consolidate_memory=self._consolidate_memory,
            interval_seconds=MEMORY_CONSOLIDATION_INTERVAL,
            log=logger,
        )

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

    async def _save_memory_async(self, text: str) -> None:
        """Fire-and-forget memory storage gated by response length.

        Skips trivial exchanges (short greetings, one-liners).
        Threshold: combined user+assistant text must exceed MEMORY_SAVE_THRESHOLD.
        Uses asyncio.to_thread() so the blocking ChromaDB call does not
        stall the event loop.
        """
        await memory_service.save_memory_async(
            vector_memory=self.vector_memory,
            text=text,
            memory_save_threshold=MEMORY_SAVE_THRESHOLD,
            utc_timestamp_format=_UTC_TIMESTAMP_FMT,
            log=logger,
        )

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
        except Exception as e:
            logger.warning(f"_load_approved_tools failed: {e}")

    async def _reload_dynamic_tools_after_worker_restart(self) -> None:
        """Reload approved dynamic tools after the isolated worker is respawned."""
        await self._load_approved_tools()
        invalidate = getattr(self, "_invalidate_capabilities_cache", None)
        if callable(invalidate):
            invalidate()

    async def _load_pending_approvals(self) -> None:
        """Reload any tool synthesis proposals that were pending when the bot last stopped."""
        try:
            pending = await self.ledger_memory.load_pending_approvals()
            for user_id, payload in pending.items():
                self.pending_tool_approval[user_id] = {
                    "synthesis": payload["synthesis"],
                    "original_state": {"user_input": payload["original_input"], "user_id": user_id},
                    "_created_at": float(payload.get("_created_at") or time.time()),
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
        self._heartbeat_failure_counts = await schedulers.restore_heartbeat_failure_counts(
            ledger_memory=self.ledger_memory,
            state_key=_HEARTBEAT_FAILURE_STATE_KEY,
            log=logger,
        )

    async def _persist_heartbeat_failure_counts(self) -> None:
        await schedulers.persist_heartbeat_failure_counts(
            ledger_memory=self.ledger_memory,
            counts=getattr(self, "_heartbeat_failure_counts", {}),
            state_key=_HEARTBEAT_FAILURE_STATE_KEY,
        )

    async def _increment_heartbeat_failure_count(self, task_id: int) -> int:
        if not hasattr(self, "_heartbeat_failure_counts"):
            self._heartbeat_failure_counts = {}
        return await schedulers.increment_heartbeat_failure_count(
            ledger_memory=self.ledger_memory,
            counts=self._heartbeat_failure_counts,
            task_id=task_id,
            state_key=_HEARTBEAT_FAILURE_STATE_KEY,
        )

    async def _clear_heartbeat_failure_count(self, task_id: int) -> None:
        if not hasattr(self, "_heartbeat_failure_counts"):
            self._heartbeat_failure_counts = {}
        await schedulers.clear_heartbeat_failure_count(
            ledger_memory=self.ledger_memory,
            counts=self._heartbeat_failure_counts,
            task_id=task_id,
            state_key=_HEARTBEAT_FAILURE_STATE_KEY,
        )

    def _fire_and_forget(self, coro) -> asyncio.Task:
        """Schedule a coroutine as a background task with a strong GC-safe reference (ISSUE-002).

        The task is added to ``_background_tasks`` and automatically removed
        when it completes, so the set never grows unboundedly while still
        preventing the garbage collector from destroying mid-flight tasks.
        """
        if not hasattr(self, "_background_tasks"):
            self._background_tasks = set()
        return schedulers.fire_and_forget(
            coro,
            background_tasks=self._background_tasks,
            log=logger,
        )

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Return the per-user asyncio.Lock with LRU eviction to bound memory usage."""
        if not hasattr(self, "_user_locks"):
            self._user_locks = OrderedDict()
        if not hasattr(self, "_user_locks_lock"):
            self._user_locks_lock = asyncio.Lock()
        return await schedulers.get_user_lock(
            user_id=user_id,
            user_locks=self._user_locks,
            user_locks_lock=self._user_locks_lock,
            max_size=max(1, int(os.getenv("USER_LOCKS_MAX_SIZE", "500"))),
        )

    @staticmethod
    def _get_synthesis_self_test_timeout_seconds() -> float:
        """Return a bounded timeout for generated-tool self-tests."""
        raw_value = os.getenv(
            "SYNTHESIS_SELF_TEST_TIMEOUT_SECONDS",
            str(_SYNTHESIS_SELF_TEST_TIMEOUT_DEFAULT_SECONDS),
        )
        try:
            timeout_seconds = float(raw_value)
        except ValueError:
            timeout_seconds = _SYNTHESIS_SELF_TEST_TIMEOUT_DEFAULT_SECONDS
        # Keep timeout strict to avoid orchestration hangs from infinite loops.
        return min(max(timeout_seconds, 1.0), 15.0)

    @staticmethod
    def _extract_pytest_counts(output_text: str) -> Dict[str, int]:
        """Extract pass/fail/error counts from pytest textual output."""
        counts = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
        }
        patterns = {
            "passed": r"(\d+)\s+passed",
            "failed": r"(\d+)\s+failed",
            "errors": r"(\d+)\s+error(?:s)?",
            "skipped": r"(\d+)\s+skipped",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, output_text)
            if match:
                counts[key] = int(match.group(1))
        return counts

    @staticmethod
    def _write_text_file(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(content)

    @staticmethod
    def _compute_synthesis_proof_sha256(tool_code: str, pytest_code: str) -> str:
        """Return SHA-256 digest for the exact tool+test artifact pair."""
        normalized_tool = str(tool_code or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized_pytest = str(pytest_code or "").replace("\r\n", "\n").replace("\r", "\n")
        proof_payload = (
            "SYNTHESIZED_TOOL_CODE\n"
            f"{normalized_tool}\n\n"
            "SYNTHESIZED_PYTEST_CODE\n"
            f"{normalized_pytest}\n"
        )
        return hashlib.sha256(proof_payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_synthesis_test_summary(
        attempt_number: int,
        max_retries: int,
        self_test_result: Dict[str, Any],
    ) -> str:
        status = str(self_test_result.get("status") or "failed").upper()
        passed = int(self_test_result.get("passed") or 0)
        failed = int(self_test_result.get("failed") or 0)
        errors = int(self_test_result.get("errors") or 0)
        duration_ms = int(self_test_result.get("duration_ms") or 0)
        return (
            f"{status} on attempt {attempt_number}/{max_retries} "
            f"(passed={passed}, failed={failed}, errors={errors}, duration={duration_ms} ms)"
        )

    @staticmethod
    def _extract_synthesis_failure_trace(self_test_result: Dict[str, Any], max_chars: int = 3000) -> str:
        stderr_text = str(self_test_result.get("stderr") or "").strip()
        stdout_text = str(self_test_result.get("stdout") or "").strip()
        error_text = str(self_test_result.get("error") or "").strip()

        trace_text = "\n\n".join(
            chunk for chunk in [stderr_text, stdout_text, error_text] if chunk
        ).strip()
        if not trace_text:
            return "(no traceback captured)"
        if len(trace_text) > max_chars:
            return trace_text[-max_chars:]
        return trace_text

    @staticmethod
    def _build_synthesis_generation_failure_result(error_message: str) -> Dict[str, Any]:
        return {
            "status": "failed",
            "timed_out": False,
            "timeout_seconds": 0,
            "exit_code": None,
            "duration_ms": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "stdout": "",
            "stderr": "",
            "error": str(error_message or "Tool generation failed."),
        }

    async def _create_synthesis_run_if_supported(
        self,
        *,
        user_id: str,
        gap_description: str,
        suggested_tool_name: str,
        original_input: str,
        max_retries: int,
    ) -> Optional[int]:
        creator = getattr(self.ledger_memory, "create_synthesis_run", None)
        if not callable(creator):
            return None
        try:
            return await creator(
                user_id=user_id,
                gap_description=gap_description,
                suggested_tool_name=suggested_tool_name,
                original_input=original_input,
                max_retries=max_retries,
            )
        except Exception as e:
            logger.warning("Could not create synthesis run record: %s", e)
            return None

    async def _append_synthesis_attempt_if_supported(
        self,
        *,
        run_id: Optional[int],
        attempt_number: int,
        phase: str,
        synthesis_payload: Dict[str, Any],
        self_test_result: Dict[str, Any],
        code_sha256: str,
    ) -> None:
        if run_id is None:
            return
        appender = getattr(self.ledger_memory, "append_synthesis_attempt", None)
        if not callable(appender):
            return
        try:
            await appender(
                run_id=run_id,
                attempt_number=attempt_number,
                phase=phase,
                synthesis_payload=synthesis_payload,
                self_test_result=self_test_result,
                code_sha256=code_sha256,
            )
        except Exception as e:
            logger.warning("Could not append synthesis attempt %s for run %s: %s", attempt_number, run_id, e)

    async def _update_synthesis_run_status_if_supported(
        self,
        run_id: Optional[int],
        **kwargs: Any,
    ) -> None:
        if run_id is None:
            return
        updater = getattr(self.ledger_memory, "update_synthesis_run_status", None)
        if not callable(updater):
            return
        try:
            await updater(run_id, **kwargs)
        except Exception as e:
            logger.warning("Could not update synthesis run %s status: %s", run_id, e)

    async def _run_synthesis_self_test(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """Run generated pytest code in an isolated subprocess with a hard timeout."""
        tool_name = synthesis["tool_name"]
        tool_code = synthesis["code"]
        pytest_code = synthesis["pytest_code"]
        timeout_seconds = self._get_synthesis_self_test_timeout_seconds()
        started = time.perf_counter()

        tool_file_name = f"{tool_name}.py"
        test_file_name = f"test_{tool_name}.py"

        process = None
        stdout_text = ""
        stderr_text = ""

        try:
            with tempfile.TemporaryDirectory(prefix="synth_selftest_") as temp_dir:
                tool_path = os.path.join(temp_dir, tool_file_name)
                test_path = os.path.join(temp_dir, test_file_name)

                await asyncio.to_thread(self._write_text_file, tool_path, tool_code)
                await asyncio.to_thread(self._write_text_file, test_path, pytest_code)

                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    test_file_name,
                    cwd=temp_dir,
                    env=_build_safe_subprocess_env(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=timeout_seconds
                    )
                    stdout_text = stdout_bytes.decode(errors="replace")
                    stderr_text = stderr_bytes.decode(errors="replace")
                except asyncio.TimeoutError:
                    if process.returncode is None:
                        process.kill()
                    stdout_bytes, stderr_bytes = await process.communicate()
                    stdout_text = stdout_bytes.decode(errors="replace")
                    stderr_text = stderr_bytes.decode(errors="replace")
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    counts = self._extract_pytest_counts(f"{stdout_text}\n{stderr_text}")
                    return {
                        "status": "timeout",
                        "timed_out": True,
                        "timeout_seconds": timeout_seconds,
                        "exit_code": process.returncode,
                        "duration_ms": elapsed_ms,
                        "passed": counts["passed"],
                        "failed": counts["failed"],
                        "errors": counts["errors"],
                        "skipped": counts["skipped"],
                        "stdout": stdout_text,
                        "stderr": stderr_text,
                        "error": (
                            f"Sandboxed self-test timed out after {timeout_seconds:.1f}s "
                            f"(possible infinite loop or blocking operation)."
                        ),
                    }
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {
                "status": "failed",
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "exit_code": process.returncode if process is not None else None,
                "duration_ms": elapsed_ms,
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": f"Sandboxed self-test harness crashed: {exc}",
            }

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        counts = self._extract_pytest_counts(f"{stdout_text}\n{stderr_text}")
        passed = process.returncode == 0

        return {
            "status": "passed" if passed else "failed",
            "timed_out": False,
            "timeout_seconds": timeout_seconds,
            "exit_code": process.returncode,
            "duration_ms": elapsed_ms,
            "passed": counts["passed"],
            "failed": counts["failed"],
            "errors": counts["errors"],
            "skipped": counts["skipped"],
            "stdout": stdout_text,
            "stderr": stderr_text,
            "error": "" if passed else "Sandboxed self-test reported pytest failures.",
        }

    @staticmethod
    def _build_synthesis_fallback_payload(
        failed_candidate: Optional[Dict[str, Any]],
        suggested_tool_name: str,
    ) -> Dict[str, Any]:
        if isinstance(failed_candidate, dict):
            return dict(failed_candidate)
        return {
            "tool_name": suggested_tool_name,
            "description": "",
            "code": "",
            "pytest_code": "",
            "schema_json": "",
            "test_manifest_json": "",
        }

    async def _request_synthesis_candidate(
        self,
        *,
        attempt_number: int,
        suggested_tool_name: str,
        redacted_gap_description: str,
        redacted_user_query: str,
        failed_candidate: Optional[Dict[str, Any]],
        last_failure_summary: str,
        last_failure_trace: str,
    ) -> Dict[str, Any]:
        if attempt_number == 1:
            return await asyncio.wait_for(
                self.cognitive_router.synthesize_tool(
                    gap_description=redacted_gap_description,
                    suggested_tool_name=suggested_tool_name,
                    user_query=redacted_user_query,
                ),
                timeout=60.0,
            )

        if failed_candidate is None:
            raise RuntimeError("Repair loop missing failed candidate payload.")

        redacted_failure_summary = self._redact_text_for_cloud(
            last_failure_summary,
            allow_sensitive_context=False,
            max_chars=1200,
        )
        redacted_failure_trace = self._redact_text_for_cloud(
            last_failure_trace,
            allow_sensitive_context=False,
            max_chars=2800,
        )

        return await asyncio.wait_for(
            self.cognitive_router.repair_synthesized_tool(
                gap_description=redacted_gap_description,
                suggested_tool_name=suggested_tool_name,
                user_query=redacted_user_query,
                previous_tool_name=str(failed_candidate.get("tool_name") or suggested_tool_name),
                previous_code=str(failed_candidate.get("code") or ""),
                previous_pytest_code=str(failed_candidate.get("pytest_code") or ""),
                failure_summary=redacted_failure_summary,
                failure_trace=redacted_failure_trace,
            ),
            timeout=60.0,
        )

    async def _run_single_synthesis_attempt(
        self,
        *,
        run_id: Optional[int],
        attempt_number: int,
        max_retries: int,
        suggested_tool_name: str,
        redacted_gap_description: str,
        redacted_user_query: str,
        failed_candidate: Optional[Dict[str, Any]],
        last_failure_summary: str,
        last_failure_trace: str,
    ) -> Dict[str, Any]:
        phase = "synthesis" if attempt_number == 1 else "repair"
        try:
            candidate = await self._request_synthesis_candidate(
                attempt_number=attempt_number,
                suggested_tool_name=suggested_tool_name,
                redacted_gap_description=redacted_gap_description,
                redacted_user_query=redacted_user_query,
                failed_candidate=failed_candidate,
                last_failure_summary=last_failure_summary,
                last_failure_trace=last_failure_trace,
            )
        except Exception as e:
            error_message = (
                f"{phase.capitalize()} generation failed on attempt {attempt_number}/{max_retries}: {e}"
            )
            logger.warning(error_message)
            failure_result = self._build_synthesis_generation_failure_result(error_message)
            await self._append_synthesis_attempt_if_supported(
                run_id=run_id,
                attempt_number=attempt_number,
                phase=phase,
                synthesis_payload=self._build_synthesis_fallback_payload(
                    failed_candidate,
                    suggested_tool_name,
                ),
                self_test_result=failure_result,
                code_sha256="",
            )
            return {
                "status": "generation_failed",
                "candidate": failed_candidate,
                "test_result": failure_result,
                "failure_summary": error_message,
                "failure_trace": error_message,
            }

        test_result = await self._run_synthesis_self_test(candidate)
        proof_sha256 = self._compute_synthesis_proof_sha256(
            candidate.get("code", ""),
            candidate.get("pytest_code", ""),
        )
        test_summary = self._build_synthesis_test_summary(
            attempt_number,
            max_retries,
            test_result,
        )
        candidate["self_test_result"] = test_result
        candidate["self_test_summary"] = test_summary
        candidate["synthesis_proof_sha256"] = proof_sha256

        await self._append_synthesis_attempt_if_supported(
            run_id=run_id,
            attempt_number=attempt_number,
            phase=phase,
            synthesis_payload=candidate,
            self_test_result=test_result,
            code_sha256=proof_sha256,
        )

        if str(test_result.get("status") or "") == "passed":
            return {
                "status": "passed",
                "candidate": candidate,
                "test_result": test_result,
                "proof_sha256": proof_sha256,
                "test_summary": test_summary,
            }

        return {
            "status": "failed",
            "candidate": candidate,
            "test_result": test_result,
            "failure_summary": f"{test_summary}. {str(test_result.get('error') or '').strip()}".strip(),
            "failure_trace": self._extract_synthesis_failure_trace(test_result),
        }

    async def _build_blocked_synthesis_result(
        self,
        *,
        run_id: Optional[int],
        attempts_used: int,
        max_retries: int,
        failed_candidate: Optional[Dict[str, Any]],
        last_test_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        trace_excerpt = self._extract_synthesis_failure_trace(last_test_result)
        blocked_reason = (
            f"Synthesis run blocked after {attempts_used}/{max_retries} attempts; no passing candidate produced."
        )
        blocked_message = (
            "System 2 synthesis run was blocked after bounded repair attempts. "
            "The main orchestrator remains active.\n\n"
            f"Attempts: {attempts_used}/{max_retries}\n"
            f"Last status: {str(last_test_result.get('status') or 'failed')}\n"
            f"Details: {str(last_test_result.get('error') or blocked_reason)}\n\n"
            f"Trace excerpt:\n{trace_excerpt}"
        )
        await self._update_synthesis_run_status_if_supported(
            run_id,
            status="blocked",
            total_attempts=attempts_used,
            blocked_reason=blocked_reason,
            synthesis_payload=failed_candidate,
        )
        return {
            "status": "blocked",
            "run_id": run_id,
            "attempts_used": attempts_used,
            "max_retries": max_retries,
            "synthesis": failed_candidate,
            "self_test_result": last_test_result,
            "blocked_reason": blocked_reason,
            "blocked_message": blocked_message,
        }

    async def _execute_synthesis_repair_loop(
        self,
        *,
        user_id: str,
        gap_description: str,
        suggested_tool_name: str,
        user_input: str,
    ) -> Dict[str, Any]:
        """Run synthesis + bounded repair retries until tests pass or run is blocked."""
        max_retries = max(1, int(_MAX_SYNTHESIS_RETRIES))
        run_id = await self._create_synthesis_run_if_supported(
            user_id=user_id,
            gap_description=gap_description,
            suggested_tool_name=suggested_tool_name,
            original_input=user_input,
            max_retries=max_retries,
        )

        redacted_gap_description = self._redact_text_for_cloud(gap_description)
        redacted_user_query = self._redact_text_for_cloud(user_input)

        failed_candidate: Optional[Dict[str, Any]] = None
        last_test_result: Dict[str, Any] = {}
        last_failure_summary = ""
        last_failure_trace = ""
        attempts_used = 0

        for attempt_number in range(1, max_retries + 1):
            attempts_used = attempt_number
            attempt_result = await self._run_single_synthesis_attempt(
                run_id=run_id,
                attempt_number=attempt_number,
                max_retries=max_retries,
                suggested_tool_name=suggested_tool_name,
                redacted_gap_description=redacted_gap_description,
                redacted_user_query=redacted_user_query,
                failed_candidate=failed_candidate,
                last_failure_summary=last_failure_summary,
                last_failure_trace=last_failure_trace,
            )

            status = str(attempt_result.get("status") or "failed")
            if status == "passed":
                return {
                    "status": "passed",
                    "run_id": run_id,
                    "attempts_used": attempt_number,
                    "max_retries": max_retries,
                    "proof_sha256": str(attempt_result.get("proof_sha256") or ""),
                    "test_summary": str(attempt_result.get("test_summary") or ""),
                    "synthesis": dict(attempt_result.get("candidate") or {}),
                    "self_test_result": dict(attempt_result.get("test_result") or {}),
                }

            failed_candidate = dict(attempt_result.get("candidate") or failed_candidate or {})
            last_test_result = dict(attempt_result.get("test_result") or {})
            last_failure_summary = str(attempt_result.get("failure_summary") or "")
            last_failure_trace = str(attempt_result.get("failure_trace") or "")
            if status == "generation_failed":
                break

        return await self._build_blocked_synthesis_result(
            run_id=run_id,
            attempts_used=attempts_used,
            max_retries=max_retries,
            failed_candidate=failed_candidate,
            last_test_result=last_test_result,
        )

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

        # Gate 1: repeated failures for the same capability require manual intervention.
        if hasattr(self, "ledger_memory"):
            try:
                failure_count = await self.ledger_memory.count_synthesis_failures_fuzzy(
                    suggested_tool_name, window_hours=24
                )
                if failure_count >= _MAX_SYNTHESIS_FAILURES_BEFORE_MANUAL_INTERVENTION:
                    return (
                        f"This capability has been attempted {failure_count} times without success. "
                        "Manual intervention needed."
                    )
            except Exception as _e:
                logger.warning("Synthesis failure count check failed: %s", _e)

            # Gate 2: per-user hourly synthesis budget.
            try:
                max_per_hour = int(os.getenv("MAX_SYNTHESES_PER_USER_PER_HOUR", "10"))
                runs_this_hour = await self.ledger_memory.count_synthesis_runs_in_window(
                    user_id, hours=1
                )
                if runs_this_hour >= max_per_hour:
                    return (
                        f"⚠️ HITL REQUIRED: Synthesis budget of {max_per_hour} per hour reached for user "
                        f"'{user_id}'. Please wait before requesting more tool synthesis."
                    )
            except Exception as _e:
                logger.warning("Synthesis budget check failed: %s", _e)

        loop_result = await self._execute_synthesis_repair_loop(
            user_id=user_id,
            gap_description=gap_description,
            suggested_tool_name=suggested_tool_name,
            user_input=str(state.get("user_input") or ""),
        )
        if loop_result.get("status") != "passed":
            return str(loop_result.get("blocked_message") or "Tool synthesis run blocked.")

        synthesis = dict(loop_result["synthesis"])
        proof_sha256 = str(loop_result.get("proof_sha256") or "")
        attempts_used = int(loop_result.get("attempts_used") or 1)
        max_retries = int(loop_result.get("max_retries") or 1)
        test_summary = str(loop_result.get("test_summary") or "")
        synthesis_run_id = loop_result.get("run_id")

        synthesis["synthesis_run_id"] = synthesis_run_id
        synthesis["synthesis_attempts_used"] = attempts_used
        synthesis["synthesis_max_retries"] = max_retries
        synthesis["synthesis_proof_sha256"] = proof_sha256
        synthesis["self_test_summary"] = test_summary

        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="pending_approval",
            total_attempts=attempts_used,
            successful_attempt=attempts_used,
            final_tool_name=str(synthesis.get("tool_name") or ""),
            code_sha256=proof_sha256,
            test_summary=test_summary,
            synthesis_payload=synthesis,
        )

        # Store payload for approval resumption (in-memory + persisted to DB)
        self.pending_tool_approval[user_id] = {
            "synthesis": synthesis,
            "original_state": state,
            "synthesis_run_id": synthesis_run_id,
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
            f"Sandboxed self-test: {test_summary}\n"
            f"Cryptographic proof (SHA-256 tool+tests): {proof_sha256}\n"
            f"Audit run id: {synthesis_run_id if synthesis_run_id is not None else 'n/a'}\n\n"
            f"Code:\n```python\n{synthesis['code']}\n```\n"
            f"─────────────────────────────\n"
            f"Reply YES to approve and deploy, or NO to reject."
        )
        if len(hitl_msg) > 3500:
            attachment = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=f"_{synthesis['tool_name']}.py",
                delete=False,
            )
            with attachment:
                attachment.write(str(synthesis["code"]))
            compact_msg = (
                f"🔧 TOOL SYNTHESIS REQUEST\n\n"
                f"System 1 could not answer: \"{state['user_input']}\"\n"
                f"Gap identified: {gap_description}\n\n"
                f"Name: {synthesis['tool_name']}\n"
                f"Description: {synthesis['description']}\n"
                f"Sandboxed self-test: {test_summary}\n"
                f"Cryptographic proof (SHA-256 tool+tests): {proof_sha256}\n"
                f"Audit run id: {synthesis_run_id if synthesis_run_id is not None else 'n/a'}\n\n"
                "Code attached as Telegram document.\n"
                "Reply YES to approve and deploy, or NO to reject."
            )
            return {"text": compact_msg[:3500], "document_path": attachment.name}
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

    def _extract_charter_tier_block(self, charter_text: str, tier_tag: Optional[str] = None) -> str:
        if tier_tag is None:
            tier_tag = str(charter_text)
            charter_text = getattr(self, "charter_text", "")
        pattern = rf"<{re.escape(tier_tag)}[^>]*>(.*?)</{re.escape(tier_tag)}>"
        match = re.search(pattern, str(charter_text or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\n{3,}", "\n\n", match.group(1).strip())

    @staticmethod
    def _extract_charter_tier_block_from_text(charter_text: str, tier_tag: str) -> str:
        try:
            ET.fromstring(str(charter_text or ""))
        except ET.ParseError as exc:
            return f"[MALFORMED_CHARTER_XML: {exc}]"
        pattern = rf"<{re.escape(tier_tag)}[^>]*>(.*?)</{re.escape(tier_tag)}>"
        match = re.search(pattern, str(charter_text or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\n{3,}", "\n\n", match.group(1).strip())

    @staticmethod

    def _extract_heartbeat_task_id(user_input: str) -> Optional[int]:
        match = re.search(r"\[HEARTBEAT TASK #(\d+)\]", str(user_input or ""))
        if not match:
            return None
        try:
            task_id = int(match.group(1))
        except (TypeError, ValueError):
            return None
        return task_id if task_id > 0 else None

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

    async def _consolidate_memory(self, user_id: str) -> None:
        """Run deep nocturnal consolidation pipeline (extract -> filter -> score -> write-back)."""
        await memory_service.consolidate_memory(
            user_id=user_id,
            ledger_memory=self.ledger_memory,
            vector_memory=self.vector_memory,
            core_memory=self.core_memory,
            nocturnal_consolidation=self.nocturnal_consolidation,
            score_nocturnal_candidates=self._score_nocturnal_candidates,
            redactor=self._redact_text_for_cloud,
            threshold=float(os.getenv("NOCTURNAL_Q_THRESHOLD", "3.0")),
            log=logger,
        )

    async def _try_resume_mfa(self, user_id: str, user_message: str) -> Optional[str]:
        """Handle MFA challenge response. Returns a reply string, or None if not in MFA flow."""
        return await pending_state.try_resume_mfa(
            user_id=user_id,
            user_message=user_message,
            pending_mfa=self.pending_mfa,
            ledger_memory=self.ledger_memory,
            execute_tool=self.cognitive_router._execute_tool,
            handle_blocked_result=self._handle_blocked_result,
            verify_mfa_challenge=verify_mfa_challenge,
            fire_and_forget=self._fire_and_forget,
            ttl_seconds=_PENDING_STATE_TTL_SECONDS,
            time_fn=time.time,
            log=logger,
        )

    async def _pop_pending_tool_approval_payload(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await pending_state.pop_pending_tool_approval_payload(
            user_id=user_id,
            pending_tool_approval=self.pending_tool_approval,
            ledger_memory=self.ledger_memory,
            ttl_seconds=_PENDING_STATE_TTL_SECONDS,
            time_fn=time.time,
            log=logger,
        )

    async def _reject_synthesized_tool(
        self,
        *,
        user_id: str,
        synthesis_run_id: Optional[int],
        attempts_used: int,
        synthesis: Dict[str, Any],
    ) -> str:
        logger.info("Admin rejected tool synthesis for user %s", user_id)
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="rejected",
            total_attempts=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(synthesis.get("tool_name") or ""),
            code_sha256=str(synthesis.get("synthesis_proof_sha256") or ""),
            test_summary=str(synthesis.get("self_test_summary") or ""),
            blocked_reason="Rejected by admin during HITL approval.",
            synthesis_payload=synthesis,
        )
        return f"Tool proposal rejected. The capability gap remains: {synthesis['description']}"

    async def _verify_synthesis_payload_digest(
        self,
        *,
        synthesis: Dict[str, Any],
        synthesis_run_id: Optional[int],
        attempts_used: int,
        tool_name: str,
    ) -> Optional[str]:
        expected_digest = str(synthesis.get("synthesis_proof_sha256") or "").strip()
        if not expected_digest:
            return None

        actual_digest = self._compute_synthesis_proof_sha256(
            str(synthesis.get("code") or ""),
            str(synthesis.get("pytest_code") or ""),
        )
        if actual_digest == expected_digest:
            return None

        mismatch_reason = (
            "Tool approval blocked: synthesis proof digest mismatch between sandbox-passed "
            "artifact and pending deployment payload."
        )
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="blocked",
            total_attempts=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(tool_name),
            code_sha256=actual_digest,
            test_summary=str(synthesis.get("self_test_summary") or ""),
            blocked_reason=mismatch_reason,
            synthesis_payload=synthesis,
        )
        logger.error(mismatch_reason)
        return mismatch_reason

    async def _deploy_approved_synthesized_tool(
        self,
        *,
        user_id: str,
        tool_name: str,
        synthesis: Dict[str, Any],
        synthesis_run_id: Optional[int],
        attempts_used: int,
        original_state: Dict[str, Any],
    ) -> Any:
        await self.cognitive_router.register_dynamic_tool(tool_name, synthesis["code"], synthesis["schema_json"])
        await self.ledger_memory.register_tool(
            name=tool_name,
            description=synthesis["description"],
            code=synthesis["code"],
            schema_json=synthesis["schema_json"],
        )
        await self.ledger_memory.approve_tool(tool_name)
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="approved",
            total_attempts=attempts_used if attempts_used > 0 else None,
            successful_attempt=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(tool_name),
            code_sha256=str(synthesis.get("synthesis_proof_sha256") or ""),
            test_summary=str(synthesis.get("self_test_summary") or ""),
            synthesis_payload=synthesis,
        )
        core = await self.core_memory.get_all()
        caps = core.get("known_capabilities", "")
        await self.core_memory.update("known_capabilities", f"{caps}, {tool_name}".lstrip(", "))
        logger.info("Tool '%s' approved, registered, and logged to core memory", tool_name)
        retry_input = str(original_state["user_input"])
        return {
            "reply_text": f"✅ Tool '{tool_name}' deployed.",
            "_deferred_follow_up_input": retry_input,
        }

    async def _handle_synthesized_tool_deploy_failure(
        self,
        *,
        tool_name: str,
        synthesis: Dict[str, Any],
        synthesis_run_id: Optional[int],
        attempts_used: int,
        error: Exception,
    ) -> str:
        await self._update_synthesis_run_status_if_supported(
            synthesis_run_id,
            status="blocked",
            total_attempts=attempts_used if attempts_used > 0 else None,
            final_tool_name=str(tool_name),
            code_sha256=str(synthesis.get("synthesis_proof_sha256") or ""),
            test_summary=str(synthesis.get("self_test_summary") or ""),
            blocked_reason=f"Deployment failed after approval: {error}",
            synthesis_payload=synthesis,
        )
        logger.error("Tool registration failed: %s", error, exc_info=True)
        return f"Error deploying tool '{tool_name}': {error}"

    async def _try_resume_tool_approval(self, user_id: str, user_message: str) -> Optional[Any]:
        """Handle YES/NO tool synthesis approval. Returns a reply string, or None if not pending."""
        return await pending_state.try_resume_tool_approval(
            user_id=user_id,
            user_message=user_message,
            pop_payload=self._pop_pending_tool_approval_payload,
            reject_synthesized_tool=self._reject_synthesized_tool,
            verify_synthesis_payload_digest=self._verify_synthesis_payload_digest,
            deploy_approved_synthesized_tool=self._deploy_approved_synthesized_tool,
            handle_synthesized_tool_deploy_failure=self._handle_synthesized_tool_deploy_failure,
        )

    async def _load_state(
        self,
        user_id: str,
        user_message: str,
        *,
        user_prompt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a state dict: resumes a HITL conversation if pending, else creates fresh."""
        return await memory_service.load_state(
            user_id=user_id,
            user_message=user_message,
            user_prompt=user_prompt,
            pending_hitl_state=self.pending_hitl_state,
            ledger_memory=self.ledger_memory,
            new_state=self._new_state,
            normalize_state=normalize_state,
            ttl_seconds=_PENDING_STATE_TTL_SECONDS,
            recent_chat_history_limit=_RECENT_CHAT_HISTORY_LIMIT,
            time_fn=time.time,
            log=logger,
        )

    async def _consume_blocked_result(self, state: Dict[str, Any], user_id: str) -> Optional[str]:
        blocked_result = state.pop(_BLOCKED_KEY, None)
        if blocked_result is None:
            return None
        return await self._handle_blocked_result(blocked_result, user_id, state)

    @staticmethod
    def _is_error_response(response: str) -> bool:
        return any(str(response or "").startswith(prefix) for prefix in _ERROR_RESPONSE_PREFIXES)

    async def _persist_chat_turns(
        self,
        user_id: str,
        user_message: str,
        final_resp: str,
        *,
        turn_failed: bool = False,
    ) -> None:
        if not hasattr(self, "_consolidation_turn_counts"):
            self._consolidation_turn_counts = OrderedDict()
        await memory_service.persist_chat_turns(
            user_id=user_id,
            user_message=user_message,
            final_resp=final_resp,
            turn_failed=turn_failed,
            ledger_memory=self.ledger_memory,
            increment_consolidation_turn_count=self._increment_consolidation_turn_count,
            consolidate_memory=self._consolidate_memory,
            fire_and_forget=self._fire_and_forget,
            consolidation_turn_counts=self._consolidation_turn_counts,
            log=logger,
        )

    def _increment_consolidation_turn_count(self, user_id: str) -> bool:
        counts, triggered = memory_service.increment_consolidation_turn_count(
            counts=getattr(self, "_consolidation_turn_counts", OrderedDict()),
            user_id=user_id,
            max_size=int(os.getenv(
                "CONSOLIDATION_TURN_COUNTS_MAX_SIZE",
                str(_CONSOLIDATION_TURN_COUNTS_MAX_SIZE_DEFAULT),
            )),
            trigger_turns=_CONSOLIDATION_TRIGGER_TURNS,
        )
        self._consolidation_turn_counts = counts
        return triggered

    def _schedule_response_memory_save(self, user_message: str, final_resp: str) -> None:
        memory_service.schedule_response_memory_save(
            fire_and_forget=self._fire_and_forget,
            save_memory_async=self._save_memory_async,
            user_message=user_message,
            final_resp=final_resp,
        )

    @staticmethod
    def _ensure_non_empty_response_text(value: Any) -> str:
        text = str(value or "").strip()
        if text:
            return text
        return "I couldn't generate a response. Please retry."

    @staticmethod
    def _recover_displayable_response(raw_response: Any, sanitized_response: Any) -> str:
        return memory_service.recover_displayable_response(raw_response, sanitized_response)

    async def _finalize_user_response(self, user_id: str, user_message: str, response: str) -> str:
        return await memory_service.finalize_user_response(
            user_id=user_id,
            user_message=user_message,
            response=response,
            sanitizer=self.cognitive_router.sanitize_response,
            persist_chat_turns=self._persist_chat_turns,
            schedule_response_memory_save=self._schedule_response_memory_save,
        )

    async def _run_graph_loop(self, state: Dict[str, Any], user_id: str, user_message: str) -> str:
        """Execute the flattened ReAct tool-calling loop and finalize the response."""
        state = normalize_state(state)
        compiled_graph = getattr(self, "_compiled_graph", None)
        if compiled_graph is None:
            compiled_graph = build_orchestrator_graph(self)
            self._compiled_graph = compiled_graph

        if compiled_graph is None:
            return await self._finalize_user_response(
                user_id,
                user_message,
                "No valid response could be generated.",
            )

        state = await compiled_graph.ainvoke(state)
        blocked_response = await self._consume_blocked_result(state, user_id)
        if blocked_response is not None:
            return blocked_response

        final_response = str(state.get("final_response") or "No valid response could be generated.")
        return await self._finalize_user_response(user_id, user_message, final_response)

    async def _try_resume_tool_approval_compat(self, user_id: str, user_message: str) -> Optional[Any]:
        return await pending_state.try_resume_tool_approval_compat(
            user_id=user_id,
            user_message=user_message,
            synthesis_pipeline=self._get_synthesis_pipeline(),
            fallback_resume=self._try_resume_tool_approval,
        )

    async def _persist_hitl_state_from_error(
        self,
        state: Dict[str, Any],
        hitl_error: Exception,
        user_id: str,
    ) -> str:
        return await pending_state.persist_hitl_state_from_error(
            state=state,
            hitl_error=hitl_error,
            user_id=user_id,
            pending_hitl_state=self.pending_hitl_state,
            admin_user_id=str(getattr(self, "_admin_user_id", "") or ""),
            ledger_memory=self.ledger_memory,
            strip_audio_bytes_for_persistence=self._strip_audio_bytes_for_persistence,
            extract_heartbeat_task_id=self._extract_heartbeat_task_id,
            fire_and_forget=self._fire_and_forget,
            outbound_queue=getattr(self, "outbound_queue", None),
            time_fn=time.time,
        )

    async def _run_user_turn_locked(
        self,
        *,
        user_id: str,
        user_message: str,
        user_prompt: Dict[str, Any],
    ) -> str:
        has_audio_prompt = bool(self._extract_audio_bytes(user_prompt))

        state = await self._load_state(
            user_id,
            user_message,
            user_prompt=user_prompt,
        )
        if state.get("final_response"):
            return self.cognitive_router.sanitize_response(state["final_response"])

        if not has_audio_prompt:
            profile_updated = await self._remember_user_profile(user_id, user_message)
            await self._remember_assistant_identity(user_message)
            lowered_message = user_message.lower()
            explicit_memory_request = any(
                marker in lowered_message
                for marker in ("remember that", "please remember", "remember i ", "remember my ")
            )
            if explicit_memory_request and not profile_updated and getattr(self, "vector_memory", None) is not None:
                await self.vector_memory.add_memory_async(
                    text=user_message,
                    metadata={"type": "explicit_memory", "source": "user_request"},
                )

            reply = await self._try_goal_planning_response(state)
            if reply is not None:
                return await self._finalize_user_response(user_id, user_message, reply)
        else:
            logger.info("Audio prompt detected for %s; bypassing text-only fast-path memory hooks.", user_id)

        try:
            response = await self._run_graph_loop(state, user_id, user_message)
            heartbeat_task_id = state.get("_heartbeat_origin_task_id")
            if heartbeat_task_id is not None and not self._is_error_response(str(response)):
                try:
                    await self.ledger_memory.update_objective_status(int(heartbeat_task_id), "completed")
                except Exception as exc:
                    logger.warning("Failed to mark heartbeat task %s completed: %s", heartbeat_task_id, exc)
            return response
        except RequiresHITLError as hitl_err:
            return await self._persist_hitl_state_from_error(state, hitl_err, user_id)
        except Exception as exc:
            logger.error("Graph execution failed: %s", exc, exc_info=True)
            return "An internal error occurred."

    async def process_message(
        self,
        user_message: Any,
        user_id: str,
        *,
        progress_callback: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> str:
        """Main entry point: State Graph execution with Energy Budget."""
        user_prompt = self._coerce_user_prompt_payload(user_message)
        normalized_user_message = str(user_prompt.get("text") or "").strip()
        has_audio_prompt = bool(self._extract_audio_bytes(user_prompt))

        if not normalized_user_message and has_audio_prompt:
            normalized_user_message = _VOICE_INPUT_PLACEHOLDER
            user_prompt["text"] = normalized_user_message

        if not normalized_user_message and not has_audio_prompt:
            return "Error: Invalid message"

        emitter_token: Optional[object] = None
        emitter: Optional[ProgressEmitter] = None
        if progress_callback is not None and str(user_id or "").strip() != "heartbeat":
            emitter = ProgressEmitter(progress_callback)
            emitter_token = set_current_emitter(emitter)

        try:
            deferred_follow_up: Optional[Dict[str, str]] = None
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
                if not hasattr(self, "_predictive_energy_budget_lock"):
                    self._predictive_energy_budget_lock = asyncio.Lock()
                if not hasattr(self, "_predictive_energy_budget_remaining"):
                    self._predictive_energy_budget_remaining = max(
                        0,
                        int(os.getenv("INITIAL_ENERGY_BUDGET", "100")),
                    )

                if not hasattr(self, "_predictive_energy_budget_last_replenished_at"):
                    self._predictive_energy_budget_last_replenished_at = time.time()

                if str(user_id or "").strip() != "heartbeat":
                    async with self._predictive_energy_budget_lock:
                        self._replenish_predictive_energy_budget_wallclock_locked()

                reply = await self._try_resume_mfa(user_id, normalized_user_message)
                if reply is not None:
                    return self._ensure_non_empty_response_text(reply)

                reply = await self._try_resume_tool_approval_compat(user_id, normalized_user_message)
                if reply is not None:
                    if isinstance(reply, dict) and reply.get("_deferred_follow_up_input") is not None:
                        deferred_follow_up = {
                            "reply_text": str(reply.get("reply_text") or ""),
                            "follow_up_input": str(reply.get("_deferred_follow_up_input") or ""),
                        }
                    else:
                        return self._ensure_non_empty_response_text(reply)

                if deferred_follow_up is None:
                    response = await self._run_user_turn_locked(
                        user_id=user_id,
                        user_message=normalized_user_message,
                        user_prompt=user_prompt,
                    )
                    return self._ensure_non_empty_response_text(response)

            follow_up = deferred_follow_up
            follow_up_response = await self.process_message(
                follow_up["follow_up_input"],
                user_id,
            )
            combined = f"{follow_up['reply_text']}\n\n{follow_up_response}".strip()
            return self._ensure_non_empty_response_text(combined)
        finally:
            if emitter is not None:
                await emitter.flush_pending()
            if emitter_token is not None:
                reset_emitter(emitter_token)

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
        if not hasattr(self, "_synthesis_in_progress"):
            self._synthesis_in_progress = {}
        return await pending_state.handle_blocked_result(
            result=result,
            user_id=user_id,
            state=state,
            pending_mfa=self.pending_mfa,
            pending_hitl_state=self.pending_hitl_state,
            ledger_memory=self.ledger_memory,
            strip_audio_bytes_for_persistence=self._strip_audio_bytes_for_persistence,
            try_resolve_capability_gap_locally=self._try_resolve_capability_gap_locally,
            async_tool_synthesis=self._async_tool_synthesis,
            fire_and_forget=self._fire_and_forget,
            synthesis_in_progress=self._synthesis_in_progress,
            synthesis_lockout_ttl_seconds=_SYNTHESIS_LOCKOUT_TTL_SECONDS,
            time_fn=time.time,
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
        escalation_problem = self._redact_text_for_cloud(
            router_result.escalation_problem,
            allow_sensitive_context=False,
        )
        escalation_context = self._redact_text_for_cloud(
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
        user_id: str,
        result: RouterResult,
        state: Dict[str, Any],
    ) -> None:
        """Background task: run tool synthesis with a timeout and send the HITL prompt to admin."""
        try:
            hitl_prompt = await asyncio.wait_for(
                self.tool_synthesis_node(state, result),
                timeout=float(_SYNTHESIS_LOCKOUT_TTL_SECONDS),
            )
            if self.outbound_queue is not None:
                await self.outbound_queue.put(hitl_prompt)
        except asyncio.TimeoutError:
            logger.critical(
                "Tool synthesis timed out after %ss for user '%s'.",
                _SYNTHESIS_LOCKOUT_TTL_SECONDS,
                user_id,
            )
            if self.outbound_queue is not None:
                await self.outbound_queue.put(
                    f"[CRITICAL] Tool synthesis timed out after {_SYNTHESIS_LOCKOUT_TTL_SECONDS}s "
                    f"for user '{user_id}'."
                )
        except Exception as e:
            logger.error(f"Background tool synthesis failed: {e}", exc_info=True)
        finally:
            if hasattr(self, "_synthesis_in_progress"):
                self._synthesis_in_progress.pop(user_id, None)

    @staticmethod
    def _get_synthesis_self_test_timeout_seconds() -> float:
        return SynthesisPipeline._get_synthesis_self_test_timeout_seconds()

    @staticmethod
    def _extract_pytest_counts(output_text: str) -> Dict[str, int]:
        return SynthesisPipeline._extract_pytest_counts(output_text)

    @staticmethod
    def _write_text_file(path: str, content: str) -> None:
        SynthesisPipeline._write_text_file(path, content)

    @staticmethod
    def _compute_synthesis_proof_sha256(tool_code: str, pytest_code: str) -> str:
        return SynthesisPipeline._compute_synthesis_proof_sha256(tool_code, pytest_code)

    @staticmethod
    def _build_synthesis_test_summary(
        attempt_number: int,
        max_retries: int,
        self_test_result: Dict[str, Any],
    ) -> str:
        return SynthesisPipeline._build_synthesis_test_summary(
            attempt_number,
            max_retries,
            self_test_result,
        )

    @staticmethod
    def _extract_synthesis_failure_trace(self_test_result: Dict[str, Any], max_chars: int = 3000) -> str:
        return SynthesisPipeline._extract_synthesis_failure_trace(self_test_result, max_chars)

    @staticmethod
    def _build_synthesis_generation_failure_result(error_message: str) -> Dict[str, Any]:
        return SynthesisPipeline._build_synthesis_generation_failure_result(error_message)

    @staticmethod
    def _build_synthesis_fallback_payload(
        failed_candidate: Optional[Dict[str, Any]],
        suggested_tool_name: str,
    ) -> Dict[str, Any]:
        return SynthesisPipeline._build_synthesis_fallback_payload(
            failed_candidate,
            suggested_tool_name,
        )

    async def _create_synthesis_run_if_supported(
        self,
        *,
        user_id: str,
        gap_description: str,
        suggested_tool_name: str,
        original_input: str,
        max_retries: int,
    ) -> Optional[int]:
        return await self._get_synthesis_pipeline()._create_synthesis_run_if_supported(
            user_id=user_id,
            gap_description=gap_description,
            suggested_tool_name=suggested_tool_name,
            original_input=original_input,
            max_retries=max_retries,
        )

    async def _append_synthesis_attempt_if_supported(
        self,
        *,
        run_id: Optional[int],
        attempt_number: int,
        phase: str,
        synthesis_payload: Dict[str, Any],
        self_test_result: Dict[str, Any],
        code_sha256: str,
    ) -> None:
        await self._get_synthesis_pipeline()._append_synthesis_attempt_if_supported(
            run_id=run_id,
            attempt_number=attempt_number,
            phase=phase,
            synthesis_payload=synthesis_payload,
            self_test_result=self_test_result,
            code_sha256=code_sha256,
        )

    async def _update_synthesis_run_status_if_supported(
        self,
        run_id: Optional[int],
        **kwargs: Any,
    ) -> None:
        await self._get_synthesis_pipeline()._update_synthesis_run_status_if_supported(
            run_id,
            **kwargs,
        )

    async def _run_synthesis_self_test(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        return await self._get_synthesis_pipeline()._run_synthesis_self_test(synthesis)

    async def _request_synthesis_candidate(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._get_synthesis_pipeline()._request_synthesis_candidate(**kwargs)

    async def _run_single_synthesis_attempt(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._get_synthesis_pipeline()._run_single_synthesis_attempt(**kwargs)

    async def _build_blocked_synthesis_result(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._get_synthesis_pipeline()._build_blocked_synthesis_result(**kwargs)

    async def _execute_synthesis_repair_loop(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._get_synthesis_pipeline()._execute_synthesis_repair_loop(**kwargs)

    async def tool_synthesis_node(
        self,
        state: Dict[str, Any],
        router_result: RouterResult,
    ) -> str:
        return await self._get_synthesis_pipeline().tool_synthesis_node(state, router_result)

    async def _pop_pending_tool_approval_payload(self, user_id: str) -> Optional[Dict[str, Any]]:
        return await self._get_synthesis_pipeline()._pop_pending_tool_approval_payload(user_id)

    async def _reject_synthesized_tool(self, **kwargs: Any) -> str:
        return await self._get_synthesis_pipeline()._reject_synthesized_tool(**kwargs)

    async def _verify_synthesis_payload_digest(self, **kwargs: Any) -> Optional[str]:
        return await self._get_synthesis_pipeline()._verify_synthesis_payload_digest(**kwargs)

    async def _deploy_approved_synthesized_tool(self, **kwargs: Any) -> Any:
        kwargs.pop("user_id", None)
        return await self._get_synthesis_pipeline()._deploy_approved_synthesized_tool(**kwargs)

    async def _handle_synthesized_tool_deploy_failure(self, **kwargs: Any) -> str:
        return await self._get_synthesis_pipeline()._handle_synthesized_tool_deploy_failure(**kwargs)

    async def _try_resume_tool_approval(self, user_id: str, user_message: str) -> Optional[Any]:
        outcome = await self._get_synthesis_pipeline().try_resume_tool_approval(user_id, user_message)
        if outcome is None:
            return None
        reply_text = str(getattr(outcome, "reply_text", outcome) or "")
        follow_up_input = getattr(outcome, "follow_up_input", None)
        if follow_up_input is None:
            return reply_text
        return {
            "reply_text": reply_text,
            "_deferred_follow_up_input": follow_up_input,
        }

    async def _async_tool_synthesis(
        self,
        user_id: str,
        result: RouterResult,
        state: Dict[str, Any],
    ) -> None:
        if not hasattr(self, "_synthesis_in_progress"):
            self._synthesis_in_progress = {}
        await schedulers.run_tool_synthesis_background(
            user_id=user_id,
            state=state,
            result=result,
            tool_synthesis_node=self.tool_synthesis_node,
            notify_admin_fn=self._notify_admin,
            synthesis_in_progress=self._synthesis_in_progress,
            timeout_seconds=float(_SYNTHESIS_LOCKOUT_TTL_SECONDS),
            log=logger,
        )

    async def aclose(self) -> None:
        """Close async and sync resources owned by the orchestrator."""
        set_runtime_context(None, None, None, orchestrator=None)

        await schedulers.drain_background_tasks(getattr(self, "_background_tasks", set()))

        if hasattr(self, "cognitive_router") and self.cognitive_router:
            await self.cognitive_router.close()
        if hasattr(self, "ledger_memory") and self.ledger_memory:
            await self.ledger_memory.close()
        if hasattr(self, "vector_memory") and self.vector_memory:
            self.vector_memory.close()

        logger.info("Orchestrator resources cleaned up")

    def close(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Schedule cleanup and keep a strong reference so the GC cannot
            # destroy the task before it completes (same pattern as _fire_and_forget).
            task = loop.create_task(self.aclose())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            try:
                asyncio.run(self.aclose())
            except Exception as e:
                logger.error(f"Error closing Orchestrator: {e}", exc_info=True)
