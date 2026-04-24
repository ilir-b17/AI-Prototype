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
import hashlib
import asyncio
import math
import re
import sys
import tempfile
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
_SYNTHESIS_SELF_TEST_TIMEOUT_DEFAULT_SECONDS = 12.0
_MAX_SYNTHESIS_RETRIES = int(os.getenv("MAX_SYNTHESIS_RETRIES", "3"))
_CONSOLIDATION_TRIGGER_TURNS = int(os.getenv("CONSOLIDATION_TRIGGER_TURNS", "10"))
_SAFE_ENV_KEYS = {"PATH", "PYTHONPATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USER", "LANG", "LC_ALL"}
_BLOCKED_ENV_PREFIXES = ("TELEGRAM_", "GROQ_", "GEMINI_", "ANTHROPIC_", "OPENAI_", "OLLAMA_CLOUD_", "ADMIN_")
_SYSTEM_1_ERROR_PREFIX = "[System 1 - Error]"
_ROUTING_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "could", "do", "for", "from",
    "hello", "hey", "hi", "i", "in", "is", "it", "me", "my", "now", "of",
    "on", "or", "please", "right", "tell", "the", "this", "to", "today", "what",
    "would", "you", "your",
    # Generic action verbs — too common to discriminate between tools
    "get", "give", "find", "show", "fetch", "list", "like", "have", "has",
    "want", "need", "also", "just", "did", "how",
    # Common discourse words that appear in tool descriptions but carry no routing signal
    "that", "which", "where", "when", "there", "then", "them", "they",
    "these", "those", "with", "been", "will", "from", "use", "not",
}
_NON_UTILITY_TOOL_PREFIXES = ("update_", "request_", "spawn_", "run_", "execute_")
_NON_UTILITY_TOOL_NAMES = {
    "ask_admin_for_guidance",
    "consolidate_memory",
    "manage_file_system",
    "query_highest_priority_task",
    # Archival memory search requires supervisor-level reasoning about WHEN to use it;
    # auto-selecting it in fast-path on any message with 'memory' or 'information' is wrong.
    "search_archival_memory",
}
_ROUTING_TOKEN_RE = r"[a-z0-9]+"
_DIRECT_ROUTE_MAX_COMPLEXITY = 0
_SINGLE_TOOL_MAX_COMPLEXITY = 2
_SINGLE_TOOL_MIN_SCORE = 1.5
_SINGLE_TOOL_MIN_MARGIN = 0.75
_FAST_PATH_DIRECT_MAX_TOKENS = 12
_FAST_PATH_SINGLE_TOOL_MAX_TOKENS = 14
_FAST_PATH_SINGLE_TOOL_ALLOWLIST = frozenset({"get_system_info", "get_stock_price", "web_search"})
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
_SENSITIVE_CONTEXT_HINT_RE = re.compile(
    r"\b(use|include|review|analy[sz]e|consider|reference)\b.{0,120}\b(chat[_\s-]*history|sensory[_\s-]*state|core[_\s-]*memory|user\s+profile|full\s+name|location|email|phone|pii)\b",
    flags=re.IGNORECASE | re.DOTALL,
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
    safe_env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _SAFE_ENV_KEYS:
            safe_env[key] = value
            continue
        if any(key.startswith(prefix) for prefix in _BLOCKED_ENV_PREFIXES):
            continue
        safe_env[key] = value
    # Ensure Python can find packages
    safe_env.setdefault("PYTHONPATH", os.getcwd())
    return safe_env


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
            self._predictive_energy_budget_lock: asyncio.Lock = asyncio.Lock()

            self.charter_text = self._load_charter()
            self._ready: asyncio.Event = asyncio.Event()
            self.pending_mfa: Dict[str, dict] = {}
            self.pending_hitl_state: Dict[str, dict] = {}
            self.pending_tool_approval: Dict[str, dict] = {}
            self._consolidation_turn_counts: Dict[str, int] = {}
            self.outbound_queue: Optional[asyncio.Queue] = None
            self.sensory_state: Dict[str, str] = {}
            # Background task registry — holds strong references to prevent GC (ISSUE-002)
            self._background_tasks: set = set()
            # Per-user async locks to serialise concurrent messages (ISSUE-012)
            self._user_locks: Dict[str, asyncio.Lock] = {}
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

        # Register Groq cooldown persistence callback and reload any saved cooldown
        async def _save_groq_cooldown(expiry: float) -> None:
            await self.ledger_memory.set_system_state("groq_cooldown_until", str(expiry))

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
        set_runtime_context(self.ledger_memory, self.core_memory, self.vector_memory)
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

        self._enforce_charter_policy()
        self._ready.set()
        logger.info("Orchestrator async_init complete")

    async def _restore_persisted_groq_cooldown(self) -> None:
        saved_cooldown = None
        try:
            saved_cooldown = await self.ledger_memory.get_system_state("groq_cooldown_until")
        except Exception as _ce:
            logger.warning("Failed to restore Groq cooldown from DB: %s", _ce)
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
        allow_missing = os.getenv("ALLOW_MISSING_CHARTER", "false").strip().lower() in {"1", "true", "yes"}
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
    def _routing_keywords(text: str) -> set:
        return {
            token for token in re.findall(_ROUTING_TOKEN_RE, (text or "").lower())
            if len(token) > 2 and token not in _ROUTING_STOPWORDS
        }

    def _estimate_request_complexity(self, text: str) -> int:
        lowered = (text or "").lower()
        tokens = re.findall(_ROUTING_TOKEN_RE, lowered)
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
        # "ticker" is also handled by _prepare_utility_tool_arguments
        return required[0] in {"query", "url", "ticker"}

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

    # Uppercase tokens that are common words, not ticker symbols.
    _TICKER_STOPWORDS: frozenset = frozenset({
        "A", "I", "AN", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT",
        "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO", "UP", "WE",
        "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HAS", "HIM", "HIS", "HOW",
        "ITS", "MAY", "NOT", "NOW", "OFF", "OLD", "OUR", "OUT", "OWN", "THE",
        "TOO", "TWO", "USE", "WAS", "WHO", "WHY", "YES", "YET", "YOU",
        "ALSO", "BEEN", "BOTH", "DOES", "DONE", "EACH", "FROM", "GIVE",
        "GOOD", "HAVE", "HERE", "HIGH", "JUST", "KNOW", "LAST", "LIKE", "LIVE",
        "LONG", "LOOK", "MAKE", "MANY", "MORE", "MOST", "MUCH", "NEED", "NEXT",
        "ONCE", "ONLY", "OPEN", "OVER", "PAST", "REAL", "SAME", "SEND", "SHOW",
        "SOME", "SUCH", "SURE", "TAKE", "TELL", "THAN", "THAT", "THEM", "THEN",
        "THEY", "THIS", "TIME", "TOLD", "VERY", "WANT", "WELL", "WERE", "WHAT",
        "WHEN", "WITH", "WORK", "YEAR", "YOUR",
    })

    @staticmethod
    def _extract_multiple_tickers(user_message: str) -> List[str]:
        """Return all valid uppercase ticker symbols found in the message (2+ expected for multi-ticker path)."""
        return [
            tok for tok in re.findall(r'\b([A-Z]{1,5})\b', user_message)
            if tok not in Orchestrator._TICKER_STOPWORDS
        ]

    @staticmethod
    def _strip_optional_tool_fallback_clause(user_message: str) -> str:
        text = (user_message or "").strip()
        stripped = re.sub(
            r"(?:[,.!?]\s*|\s+)(?:please\s+)?(?:search|browse|check|look(?:\s+up)?|find|get|use)\s+(?:the\s+)?(?:web|internet|online)\b[^.?!]*?\bif\s+(?:you\s+)?(?:must|need(?:\s+to)?|have\s+to|required|necessary)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ,.!?;")
        return stripped or text

    def _prepare_utility_tool_arguments(
        self,
        schema: Dict[str, Any],
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
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
            # Strip residual "you [action-verb] [for/about]?" prefix left after removing
            # polite openers. e.g. "Can you search for X" → "you search for X" → "X"
            cleaned = re.sub(
                r"^you\s+(?:search|look|find|browse|check|tell|get|show|fetch|look up|search for)\s+(?:for\s+|about\s+|up\s+)?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip(" ?!.,")
            if not cleaned:
                cleaned = user_message.strip()
            # If reduced to a bare action verb (e.g. "Please search" → "search"),
            # look back at the last substantive user turn for the real topic.
            _bare_verbs = {"search", "find", "look", "do", "go", "try", "run", "proceed", "continue"}
            if cleaned.lower() in _bare_verbs and chat_history:
                for turn in reversed(chat_history):
                    if turn.get("role") == "user":
                        prev = turn.get("content", "").strip()
                        if len(prev) > 15:
                            cleaned = prev
                            break
            args = {"query": cleaned}
            if "max_results" in properties:
                args["max_results"] = 3
            return args

        if field_name == "url":
            url_match = re.search(r"https?://\S+", user_message)
            if not url_match:
                return None
            return {"url": url_match.group(0)}

        if field_name == "ticker":
            # Extract uppercase ticker symbols (1–5 chars), excluding common words.
            candidates = [
                tok for tok in re.findall(r'\b([A-Z]{1,5})\b', user_message)
                if tok not in Orchestrator._TICKER_STOPWORDS
            ]
            if len(candidates) == 1:
                return {"ticker": candidates[0]}
            # Zero or multiple tickers: let the supervisor handle it.
            return None

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

    @staticmethod
    def _is_trivial_direct_intent(user_message: str) -> bool:
        lowered = (user_message or "").strip().lower()
        if not lowered:
            return False

        if re.fullmatch(
            r"(?:hi|hello|hey)(?:\s+[a-z0-9]{1,20})?[!.? ]*|(?:thanks|thank you|ok|okay|good morning|good evening|good night)[!.? ]*",
            lowered,
        ):
            return True

        token_count = len(re.findall(_ROUTING_TOKEN_RE, lowered))
        if token_count > _FAST_PATH_DIRECT_MAX_TOKENS:
            return False

        return bool(re.search(r"\b(what is|what's|who is|define|meaning of)\b", lowered))

    @staticmethod
    def _is_trivial_single_tool_intent(tool_name: str, user_message: str, complexity: int) -> bool:
        if tool_name not in _FAST_PATH_SINGLE_TOOL_ALLOWLIST:
            return False
        if complexity > _SINGLE_TOOL_MAX_COMPLEXITY:
            return False

        lowered = (user_message or "").lower()
        token_count = len(re.findall(_ROUTING_TOKEN_RE, lowered))
        if token_count > _FAST_PATH_SINGLE_TOOL_MAX_TOKENS:
            return False

        if re.search(r"\b(and then|then|after|before|also|plus|compare|step|steps|plan)\b", lowered):
            return False

        if tool_name == "get_system_info":
            return bool(re.search(r"\b(time|date|timezone|system info|system information|platform|cpu|memory|os)\b", lowered))
        if tool_name == "get_stock_price":
            return bool(re.search(r"\b(stock|price|quote|ticker|market)\b", lowered))
        if tool_name == "web_search":
            return bool(re.search(r"\b(weather|news|latest|current|today|now|headline|score)\b", lowered))

        return False

    def _assess_request_route(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        routing_message = self._strip_optional_tool_fallback_clause(user_message)
        complexity = self._estimate_request_complexity(routing_message)

        # Multi-ticker: two or more valid stock symbols → sequential get_stock_price calls.
        # Checked before general scoring so the supervisor never sees this pattern.
        tickers = self._extract_multiple_tickers(routing_message)
        if len(tickers) >= 2:
            return {
                "mode": "multi_ticker",
                "tickers": tickers,
                "complexity": complexity,
            }

        candidates = []

        for schema in self.cognitive_router.registry.get_schemas():
            if not self._is_utility_tool_schema(schema):
                continue
            arguments = self._prepare_utility_tool_arguments(schema, routing_message, chat_history)
            if arguments is None:
                continue
            score = self._score_tool_for_request(routing_message, schema)
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
        trivial_direct = self._is_trivial_direct_intent(routing_message)

        if (
            top
            and complexity <= _SINGLE_TOOL_MAX_COMPLEXITY
            and top["score"] >= _SINGLE_TOOL_MIN_SCORE
            and (top["score"] - next_score) >= _SINGLE_TOOL_MIN_MARGIN
            and self._is_trivial_single_tool_intent(top["tool_name"], routing_message, complexity)
        ):
            return {
                "mode": "single_tool",
                "complexity": complexity,
                "tool_name": top["tool_name"],
                "arguments": top["arguments"],
            }

        if (
            trivial_direct
            and complexity <= _DIRECT_ROUTE_MAX_COMPLEXITY
            and (not top or top["score"] < _SINGLE_TOOL_MIN_SCORE)
        ):
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
            if not text and audio_bytes:
                text = _VOICE_INPUT_PLACEHOLDER

            normalized: Dict[str, Any] = {
                "text": text,
                "audio_bytes": audio_bytes,
                "audio_mime_type": str(payload.get("audio_mime_type") or "audio/ogg"),
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

        name_match = re.search(
            r"\bmy name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if name_match:
            raw_name = Orchestrator._normalize_display_name(name_match.group(1))
            if raw_name:
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

        age_match = re.search(r"\b(\d{1,3})\s*years?\s*old\b", lowered)
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

        capability_response = self._build_capability_response(user_message)
        if capability_response:
            return capability_response

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

        assessment = self._assess_request_route(user_message, state.get("chat_history", []))
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
        """Build an ultra-lean Level-1 catalog: name + description only."""
        rows = self._build_capability_catalog_rows()
        if not rows:
            return "Available skills catalog (name: description): none loaded"
        return "Available skills catalog (name: description):\n" + "\n".join(rows)

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

    @staticmethod
    def _requires_sensitive_cloud_context(*segments: str) -> bool:
        combined = "\n".join(str(segment or "") for segment in segments)
        if not combined.strip():
            return False
        return bool(_SENSITIVE_CONTEXT_HINT_RE.search(combined))

    @staticmethod
    def _redact_text_for_cloud(
        text: str,
        *,
        allow_sensitive_context: bool = False,
        max_chars: int = 3500,
    ) -> str:
        raw_text = str(text or "")
        if not raw_text:
            return ""

        if allow_sensitive_context:
            compact = raw_text.strip()
            if len(compact) > max_chars:
                return compact[:max_chars].rstrip() + "\n[TRUNCATED_FOR_SIZE]"
            return compact

        redacted = raw_text
        for pattern in _CLOUD_REDACTION_BLOCK_PATTERNS:
            redacted = re.sub(pattern, "[REDACTED_CONTEXT_BLOCK]", redacted, flags=re.IGNORECASE | re.DOTALL)

        for pattern, replacement in _CLOUD_REDACTION_LINE_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)

        for pattern, replacement in _CLOUD_REDACTION_PII_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

        redacted = re.sub(r"\n{3,}", "\n\n", redacted).strip()
        if not redacted:
            return "[REDACTED_EMPTY_PAYLOAD]"
        if len(redacted) > max_chars:
            return redacted[:max_chars].rstrip() + "\n[TRUNCATED_FOR_PRIVACY]"
        return redacted

    def _redact_messages_for_cloud(
        self,
        messages: List[Dict[str, str]],
        *,
        allow_sensitive_context: bool = False,
    ) -> List[Dict[str, str]]:
        if not messages:
            return [{"role": "user", "content": "[REDACTED_EMPTY_PAYLOAD]"}]

        source_messages: List[Dict[str, Any]]
        if allow_sensitive_context:
            source_messages = list(messages)
        else:
            system_messages = [msg for msg in messages if str(msg.get("role", "")).lower() == "system"]
            user_messages = [msg for msg in messages if str(msg.get("role", "")).lower() == "user"]

            source_messages = []
            if system_messages:
                source_messages.append(system_messages[0])
                if len(system_messages) > 1:
                    source_messages.append(system_messages[-1])
            if user_messages:
                source_messages.append(user_messages[-1])
            elif messages:
                source_messages.append(messages[-1])

        sanitized: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for message in source_messages:
            role = str(message.get("role") or "user")
            content = self._redact_text_for_cloud(
                str(message.get("content") or ""),
                allow_sensitive_context=allow_sensitive_context,
            )
            if not content:
                continue
            key = (role, content)
            if key in seen:
                continue
            seen.add(key)
            sanitized.append({"role": role, "content": content})

        return sanitized or [{"role": "user", "content": "[REDACTED_EMPTY_PAYLOAD]"}]

    async def _route_to_system_2_redacted(
        self,
        messages: List[Dict[str, str]],
        *,
        allowed_tools: Optional[List[str]] = None,
        purpose: str = "system_2",
        allow_sensitive_context: bool = False,
    ) -> RouterResult:
        minimized_messages = self._redact_messages_for_cloud(
            messages,
            allow_sensitive_context=allow_sensitive_context,
        )
        logger.info(
            "System 2 payload redacted for %s: %d -> %d message(s)",
            purpose,
            len(messages),
            len(minimized_messages),
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

    def _energy_gate_available(self) -> bool:
        return (
            getattr(self, "energy_judge", None) is not None
            and getattr(self, "energy_roi_engine", None) is not None
            and getattr(self, "_predictive_energy_budget_lock", None) is not None
            and hasattr(self, "_predictive_energy_budget_remaining")
        )

    async def _get_predictive_energy_budget_remaining(self) -> int:
        if not self._energy_gate_available():
            return max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
        async with self._predictive_energy_budget_lock:
            return max(0, int(self._predictive_energy_budget_remaining))

    async def _try_reserve_predictive_energy_budget(
        self,
        *,
        predicted_cost: int,
        min_reserve: int,
        reason: str,
    ) -> bool:
        if not self._energy_gate_available():
            return True

        cost = max(0, int(predicted_cost))
        reserve = max(0, int(min_reserve))
        async with self._predictive_energy_budget_lock:
            remaining = max(0, int(self._predictive_energy_budget_remaining))
            if remaining - cost < reserve:
                return False
            self._predictive_energy_budget_remaining = remaining - cost
            logger.info(
                "Predictive energy budget reserved: -%s (%s). Remaining=%s",
                cost,
                reason,
                self._predictive_energy_budget_remaining,
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
        complexity = max(1, self._estimate_request_complexity(user_message) + 1)
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
        if not self._energy_gate_available():
            return None

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

        complexity = self._estimate_request_complexity(text)
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
        allow_sensitive_context = self._requires_sensitive_cloud_context(
            user_message,
            planning_context,
        )

        async def _route(messages: List[Dict[str, str]]) -> RouterResult:
            return await self._route_to_system_2_redacted(
                messages,
                allowed_tools=[],
                purpose="goal_planner",
                allow_sensitive_context=allow_sensitive_context,
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
            assessment = self._assess_request_route(user_input)
        except Exception:
            return None

        mode = str(assessment.get("mode") or "")
        if mode == "single_tool":
            tool_name = str(assessment.get("tool_name") or "")
            if tool_name in _FAST_PATH_SINGLE_TOOL_ALLOWLIST:
                return build_triviality_bypass_decision(
                    f"single_tool_read_only:{tool_name}"
                )
        if mode == "direct" and self._is_trivial_direct_intent(user_input):
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
            allow_sensitive_context = self._requires_sensitive_cloud_context(
                messages[-1].get("content", "") if messages else "",
            )
            router_result = await asyncio.wait_for(
                self._route_to_system_2_redacted(
                    messages,
                    purpose="supervisor_fallback",
                    allow_sensitive_context=allow_sensitive_context,
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

        # If extra non-whitespace text appears after the JSON payload, reject it.
        if workers_payload[end_index:].strip():
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
        *,
        task_packet: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[RouterResult]:
        if not self.cognitive_router.get_system_2_available():
            return None

        try:
            allow_sensitive_context = self._requires_sensitive_cloud_context(
                (task_packet or {}).get("task", ""),
                (task_packet or {}).get("reason", ""),
                (state or {}).get("user_input", ""),
            )
            router_result = await asyncio.wait_for(
                self._route_to_system_2_redacted(
                    messages,
                    allowed_tools=agent_def.allowed_tools,
                    purpose=f"{agent_def.name}_fallback",
                    allow_sensitive_context=allow_sensitive_context,
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
                state=state,
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

    @staticmethod
    def _heartbeat_result_indicates_failure(result: str) -> bool:
        lowered = str(result or "").lower()
        return any(
            marker in lowered
            for marker in ("error", "cannot", "unable", "failed", "i cannot", "i am unable")
        )

    @staticmethod
    def _build_heartbeat_execution_prompt(task: Dict[str, Any]) -> str:
        return (
            f"[HEARTBEAT TASK #{task['id']}]: {task['title']}\n"
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
            statuses=["pending", "deferred_due_to_energy"],
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
        logger.info("Heartbeat: Querying objective backlog for executable Tasks...")
        candidate_contexts = await self._select_executable_heartbeat_tasks()

        if not candidate_contexts:
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

    async def _score_nocturnal_candidates(self, messages: List[Dict[str, str]]) -> RouterResult:
        allow_sensitive_context = self._requires_sensitive_cloud_context(
            *(message.get("content", "") for message in messages),
        )
        if self.cognitive_router.get_system_2_available():
            return await asyncio.wait_for(
                self._route_to_system_2_redacted(
                    messages,
                    allowed_tools=[],
                    purpose="nocturnal_scoring",
                    allow_sensitive_context=allow_sensitive_context,
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
        messages.append(self._build_user_prompt_message(state))

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
        state["moral_remediation_constraints"] = []
        state["moral_halt_required"] = False
        state["moral_halt_summary"] = ""
        if not state.get("final_response"):
            state["final_response"] = output_to_eval
        return state

    @staticmethod
    def _extract_charter_tier_block(charter_text: str, tier_tag: str) -> str:
        pattern = rf"<{re.escape(tier_tag)}[^>]*>(.*?)</{re.escape(tier_tag)}>"
        match = re.search(pattern, str(charter_text or ""), flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\n{3,}", "\n\n", match.group(1).strip())

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
        tier_1 = self._extract_charter_tier_block(self.charter_text, "Tier_1_Axioms")
        tier_2 = self._extract_charter_tier_block(self.charter_text, "Tier_2_Strategic")
        tier_3 = self._extract_charter_tier_block(self.charter_text, "Tier_3_Operational")

        redacted_user_input = self._redact_text_for_cloud(
            str(state.get("user_input") or ""),
            allow_sensitive_context=False,
            max_chars=1600,
        )
        redacted_plan = self._redact_text_for_cloud(
            self._summarize_plan_for_moral_audit(state),
            allow_sensitive_context=False,
            max_chars=1200,
        )
        redacted_output = self._redact_text_for_cloud(
            str(output_to_eval or ""),
            allow_sensitive_context=False,
            max_chars=2200,
        )
        redacted_critic_instructions = self._redact_text_for_cloud(
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
                request_redacted=self._redact_text_for_cloud(
                    str(state.get("user_input") or ""),
                    allow_sensitive_context=False,
                    max_chars=1600,
                ),
                output_redacted=self._redact_text_for_cloud(
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

    async def critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Checks worker output against the charter. Skipped for direct supervisor responses."""
        state = normalize_state(state)
        output_to_eval = self._get_output_to_evaluate(state)

        if not output_to_eval:
            self._store_moral_decision_trace(
                state,
                build_local_skip_decision("empty_output"),
            )
            state["critic_feedback"] = "PASS"
            await self._persist_moral_audit_log(state, output_to_eval)
            return state

        bypass_decision = self._try_triviality_bypass_decision(state, output_to_eval)
        if bypass_decision is not None:
            self._store_moral_decision_trace(state, bypass_decision)
            state = self._finalize_critic_pass(state, output_to_eval)
            await self._persist_moral_audit_log(state, output_to_eval)
            return state

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
                redactor=self._redact_text_for_cloud,
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
        return await self._handle_blocked_result(exec_result, pending_tool.get("user_id", user_id), {})

    async def _pop_pending_tool_approval_payload(self, user_id: str) -> Optional[Dict[str, Any]]:
        if user_id not in self.pending_tool_approval:
            return None
        payload = self.pending_tool_approval[user_id]
        age = time.time() - payload.get("_created_at", 0)
        if age > _PENDING_STATE_TTL_SECONDS:
            self.pending_tool_approval.pop(user_id, None)
            await self.ledger_memory.clear_pending_approval(user_id)
            logger.info("Expired stale pending_tool_approval for %s (age %.0fs).", user_id, age)
            return None
        self.pending_tool_approval.pop(user_id, None)
        await self.ledger_memory.clear_pending_approval(user_id)
        return payload

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
    ) -> str:
        self.cognitive_router.register_dynamic_tool(tool_name, synthesis["code"], synthesis["schema_json"])
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
        retry = await self.process_message(original_state["user_input"], user_id)
        return f"✅ Tool '{tool_name}' deployed.\n\n{retry}"

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

    async def _try_resume_tool_approval(self, user_id: str, user_message: str) -> Optional[str]:
        """Handle YES/NO tool synthesis approval. Returns a reply string, or None if not pending."""
        payload = await self._pop_pending_tool_approval_payload(user_id)
        if payload is None:
            return None

        synthesis = payload["synthesis"]
        original_state = payload["original_state"]
        synthesis_run_id = payload.get("synthesis_run_id") or synthesis.get("synthesis_run_id")
        attempts_used = int(synthesis.get("synthesis_attempts_used") or 0)
        tool_name = str(synthesis.get("tool_name") or "")

        if not user_message.strip().upper().startswith("YES"):
            return await self._reject_synthesized_tool(
                user_id=user_id,
                synthesis_run_id=synthesis_run_id,
                attempts_used=attempts_used,
                synthesis=synthesis,
            )

        mismatch = await self._verify_synthesis_payload_digest(
            synthesis=synthesis,
            synthesis_run_id=synthesis_run_id,
            attempts_used=attempts_used,
            tool_name=tool_name,
        )
        if mismatch:
            return mismatch

        try:
            return await self._deploy_approved_synthesized_tool(
                user_id=user_id,
                tool_name=tool_name,
                synthesis=synthesis,
                synthesis_run_id=synthesis_run_id,
                attempts_used=attempts_used,
                original_state=original_state,
            )
        except Exception as e:
            return await self._handle_synthesized_tool_deploy_failure(
                tool_name=tool_name,
                synthesis=synthesis,
                synthesis_run_id=synthesis_run_id,
                attempts_used=attempts_used,
                error=e,
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
                state["iteration_count"] = 0
                state["current_plan"] = []
                state["_energy_gate_cleared"] = False
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
        state = self._new_state(user_id, user_message, user_prompt=user_prompt)
        state["user_prompt"] = dict(user_prompt or {})
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

        if not hasattr(self, "_consolidation_turn_counts"):
            self._consolidation_turn_counts = {}

        trigger_turns = max(1, _CONSOLIDATION_TRIGGER_TURNS)
        self._consolidation_turn_counts[user_id] = self._consolidation_turn_counts.get(user_id, 0) + 1
        if self._consolidation_turn_counts[user_id] >= trigger_turns:
            self._consolidation_turn_counts[user_id] = 0
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
        return await self._finalize_user_response(user_id, user_message, state["final_response"])

    async def process_message(self, user_message: Any, user_id: str) -> str:
        """Main entry point: State Graph execution with Energy Budget."""
        user_prompt = self._coerce_user_prompt_payload(user_message)
        normalized_user_message = str(user_prompt.get("text") or "").strip()
        has_audio_prompt = bool(self._extract_audio_bytes(user_prompt))

        if not normalized_user_message and has_audio_prompt:
            normalized_user_message = _VOICE_INPUT_PLACEHOLDER
            user_prompt["text"] = normalized_user_message

        if not normalized_user_message and not has_audio_prompt:
            return "Error: Invalid message"

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
            if not hasattr(self, "_predictive_energy_budget_lock"):
                self._predictive_energy_budget_lock = asyncio.Lock()
            if not hasattr(self, "_predictive_energy_budget_remaining"):
                self._predictive_energy_budget_remaining = max(
                    0,
                    int(os.getenv("INITIAL_ENERGY_BUDGET", "100")),
                )

            async with self._predictive_energy_budget_lock:
                replenishment = int(os.getenv("ENERGY_REPLENISH_PER_TURN", "5"))
                cap = int(os.getenv("INITIAL_ENERGY_BUDGET", "100"))
                self._predictive_energy_budget_remaining = min(
                    self._predictive_energy_budget_remaining + replenishment, cap
                )

            reply = await self._try_resume_mfa(user_id, normalized_user_message)
            if reply is not None:
                return reply

            reply = await self._try_resume_tool_approval(user_id, normalized_user_message)
            if reply is not None:
                return reply

            state = await self._load_state(
                user_id,
                normalized_user_message,
                user_prompt=user_prompt,
            )
            if state.get("final_response") and not state.get("current_plan"):
                return self.cognitive_router.sanitize_response(state["final_response"])

            if not has_audio_prompt:
                await self._remember_user_profile(user_id, normalized_user_message)
                await self._remember_assistant_identity(normalized_user_message)
            else:
                logger.info("Audio prompt detected for %s; bypassing text-only fast-path memory hooks.", user_id)

            if not has_audio_prompt:
                reply = await self._try_goal_planning_response(state)
                if reply is not None:
                    return await self._finalize_user_response(user_id, normalized_user_message, reply)

                reply = await self._try_fast_path_response(state)
                if reply is not None:
                    return await self._finalize_user_response(user_id, normalized_user_message, reply)

            try:
                return await self._run_graph_loop(state, user_id, normalized_user_message)
            except RequiresHITLError as hitl_err:
                state["_hitl_question"] = str(hitl_err)
                state["_hitl_created_at"] = time.time()
                pending_state = self._strip_audio_bytes_for_persistence(state)
                self.pending_hitl_state[user_id] = pending_state
                # Persist so the state survives a bot restart (ISSUE-013)
                self._fire_and_forget(self.ledger_memory.save_hitl_state(user_id, pending_state))
                return str(hitl_err)
            except Exception as e:
                logger.error(f"Graph execution failed: {e}", exc_info=True)
                return "An internal error occurred."

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
            pending_state = self._strip_audio_bytes_for_persistence(state)
            self.pending_hitl_state[user_id] = pending_state
            # Persist so the HITL state survives a bot restart (ISSUE-013)
            self._fire_and_forget(self.ledger_memory.save_hitl_state(user_id, pending_state))
            return result.hitl_message

        if result.status == "capability_gap":
            local_resolution = await self._try_resolve_capability_gap_locally(
                user_id,
                result,
                state,
            )
            if local_resolution is not None:
                return local_resolution

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
        allow_sensitive_context = self._requires_sensitive_cloud_context(
            router_result.escalation_problem,
            router_result.escalation_context,
            state.get("user_input", ""),
        )
        escalation_problem = self._redact_text_for_cloud(
            router_result.escalation_problem,
            allow_sensitive_context=allow_sensitive_context,
        )
        escalation_context = self._redact_text_for_cloud(
            router_result.escalation_context,
            allow_sensitive_context=allow_sensitive_context,
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
            allow_sensitive_context=allow_sensitive_context,
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
            set_runtime_context(None, None, None)
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
