"""
LLM Router Module - Routes decisions to System 1 (Local) or System 2 (Cloud).

System 1: Local Ollama model (Gemma 4) — fast, private, always available.
System 2: Cloud LLM for complex reasoning. Provider priority:
  1. Ollama Cloud — set OLLAMA_CLOUD_API_KEY + optionally SYSTEM_2_MODEL (default: deepseek-v3.2)
  2. Groq (free, fast) — set GROQ_API_KEY
  3. Gemini (Google AI Studio free tier) — set GEMINI_API_KEY + USE_GEMINI=True
"""

import ast
import base64
import difflib
import os
import logging
import json
import asyncio
import io
import math
import re
import time
import tokenize
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, AsyncIterator

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False

import ollama

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from src.core.skill_manager import SkillRegistry
from src.core.dynamic_tool_worker import DynamicToolWorkerClient

# ── Structured result returned by all router methods ─────────────────────────

@dataclass
class RouterResult:
    """
    Structured result from the CognitiveRouter.

    ``status`` is always one of:
      - "ok"               — normal text response in ``content``
      - "mfa_required"     — the model requested a privileged core update
      - "hitl_required"    — human-in-the-loop guidance is needed
      - "capability_gap"   — System 1 cannot fulfil the request; synthesis needed
      - "cognitive_escalation" — escalate complex reasoning to System 2
    """
    status: str  # "ok" | "mfa_required" | "hitl_required" | "capability_gap" | "cognitive_escalation"
    content: str = ""
    # MFA fields
    mfa_tool_name: str = ""
    mfa_arguments: dict = field(default_factory=dict)
    # HITL fields
    hitl_message: str = ""
    # Capability-gap fields
    gap_description: str = ""
    suggested_tool_name: str = ""
    # Escalation fields
    escalation_problem: str = ""
    escalation_context: str = ""


# ── Legacy exception classes kept for backward compatibility ─────────────────
# These are no longer raised by the router itself; they may still be raised
# by orchestrator-internal code (e.g. energy exhaustion in _deduct_energy).

class RequiresMFAError(Exception):
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments
        super().__init__(f"MFA required for tool: {tool_name}")

class RequiresHITLError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class RequiresCapabilitySynthesisError(Exception):
    """Raised by orchestrator-internal code (e.g. energy exhaustion)."""
    def __init__(self, gap_description: str, suggested_tool_name: str):
        self.gap_description = gap_description
        self.suggested_tool_name = suggested_tool_name
        super().__init__(f"Capability gap: {gap_description}")

# ── AST sandbox configuration ─────────────────────────────────────────────────

_BLOCKED_TOP_LEVEL_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket",
    "importlib", "builtins", "ctypes", "multiprocessing", "threading",
    "signal", "pty", "popen", "pexpect", "atexit", "gc",
    # Additional dangerous modules blocked to close sandbox escape paths:
    "concurrent",   # concurrent.futures.ProcessPoolExecutor / ThreadPoolExecutor
    "runpy",        # runpy.run_path() executes arbitrary code files
    "code",         # interactive Python interpreter
    "codeop",       # compile/execute helper used by interactive interpreter
    "compileall",   # mass-compilation of Python source trees
    "tempfile",     # can write arbitrary files to the filesystem
    "mmap",         # raw memory-mapped file access
    "pickle",       # arbitrary code execution via __reduce__ / __getstate__
    "marshal",      # low-level serialization; can reconstruct code objects
    "shelve",       # uses pickle internally
    "ast",          # can parse/compile/reconstruct blocked expressions at runtime
}

# Generated pytest scripts are intentionally constrained to a tight import
# allowlist. They may import the generated module itself, pytest, and unittest.mock.
_ALLOWED_PYTEST_TOP_LEVEL_MODULES = {
    "pytest",
    "unittest",
    "json",
    "re",
    "math",
    "typing",
    "collections",
    "dataclasses",
}

_BLOCKED_DUNDER_ATTRIBUTES = {
    "__import__", "__loader__", "__builtins__", "__code__",
    "__globals__", "__subclasses__", "__mro__", "__dict__",
}

_BLOCKED_TOOL_BUILTINS = {
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars",
}
_BLOCKED_DYNAMIC_TOOL_TOKENS = {
    "__subclasses__", "__mro__", "__globals__", "__builtins__", "__import__",
}
_BLOCKED_DYNAMIC_TOOL_DOTTED_TOKENS = {"sys.modules"}
_DYNAMIC_TOOL_IGNORED_TOKEN_TYPES = frozenset({
    tokenize.COMMENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NL,
    tokenize.NEWLINE,
})

_BLOCKED_ASYNCIO_CALLS = {
    "create_subprocess_exec",
    "create_subprocess_shell",
}

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
USE_GEMINI = os.getenv('USE_GEMINI', 'False').strip().lower() == 'true'
OLLAMA_CLOUD_API_KEY = os.getenv('OLLAMA_CLOUD_API_KEY', '').strip()
OLLAMA_CLOUD_HOST = os.getenv('OLLAMA_CLOUD_HOST', 'https://api.ollama.com').strip()
SYSTEM_2_MODEL = os.getenv('SYSTEM_2_MODEL', 'deepseek-v3.2').strip()
TOOL_EXEC_TIMEOUT_SECONDS = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "30"))
DEFAULT_TARGET_CONTEXT_TOKENS = 131072
DEFAULT_CONTEXT_RESERVE_TOKENS = 4096
DEFAULT_MIN_INGESTION_CHARS = 4000
DEFAULT_PARALLEL_READ_ONLY_TOOL_CONCURRENCY = 4
_APPROX_CHARS_PER_TOKEN = 4
_READ_ONLY_PARALLEL_TOOL_NAMES = {
    "analyze_table_file",
    "ask_admin_for_guidance",
    "extract_pdf_text",
    "extract_web_article",
    "get_stock_price",
    "get_system_info",
    "query_highest_priority_task",
    "search_archival_memory",
    "web_search",
}
_CAPABILITY_MATCH_STOPWORDS = {
    "a", "an", "and", "the", "to", "for", "of", "in", "on", "or", "is", "are",
    "be", "can", "could", "would", "should", "have", "has", "with", "new", "needed",
    "need", "admin", "user", "function", "tool", "tools", "skill", "skills",
    "capability", "capabilities", "allow", "easily", "provide", "more", "than",
    "just", "calling", "underlying", "use", "using", "this", "that", "from",
}
_CAPABILITY_META_TOOL_NAMES = {
    "request_capability",
    "request_core_update",
    "ask_admin_for_guidance",
    "escalate_to_system_2",
}


class CognitiveRouter:
    """
    Routes prompts to appropriate LLM systems.
    System 1: Local Ollama (fast, private)
    System 2: Ollama Cloud (preferred when OLLAMA_CLOUD_API_KEY set) → Groq → Gemini
    """

    @staticmethod
    def _capability_match_tokens(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 2 and token not in _CAPABILITY_MATCH_STOPWORDS
        }

    @staticmethod
    def _schema_match_tokens(schema: Dict[str, Any]) -> Dict[str, set[str]]:
        params = (schema.get("parameters") or {}).get("properties", {}) or {}
        name_tokens = CognitiveRouter._capability_match_tokens(schema.get("name", "").replace("_", " "))
        desc_tokens = CognitiveRouter._capability_match_tokens(schema.get("description", ""))
        param_tokens: set[str] = set()
        for param_name, param_schema in params.items():
            param_tokens |= CognitiveRouter._capability_match_tokens(param_name.replace("_", " "))
            param_tokens |= CognitiveRouter._capability_match_tokens(param_schema.get("description", ""))
        return {
            "name": name_tokens,
            "description": desc_tokens,
            "parameters": param_tokens,
        }

    def _find_existing_capability_match(
        self,
        gap_description: str,
        suggested_tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        query_tokens = self._capability_match_tokens(gap_description)
        query_tokens |= self._capability_match_tokens((suggested_tool_name or "").replace("_", " "))
        if not query_tokens:
            return None

        best_schema: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for schema in self.registry.get_schemas():
            name = schema.get("name", "")
            if not name or name in _CAPABILITY_META_TOOL_NAMES:
                continue
            schema_tokens = self._schema_match_tokens(schema)
            score = 0.0
            score += 3.0 * len(query_tokens & schema_tokens["name"])
            score += 1.5 * len(query_tokens & schema_tokens["description"])
            score += 0.75 * len(query_tokens & schema_tokens["parameters"])
            if score > best_score:
                best_schema = schema
                best_score = score

        return best_schema if best_score >= 2.5 else None

    async def _execute_tool(self, tool_name: str, arguments: dict) -> RouterResult:
        """Dispatch a tool call through the SkillRegistry.

        Security intercepts (MFA / HITL / capability synthesis) are handled
        here, returning a structured :class:`RouterResult` rather than raising.
        """
        # Normalize hallucinated / malformed tool names
        tool_name = self._normalize_tool_name(tool_name)

        if arguments is None:
            arguments = {}

        logger.info(f"Executing tool: {tool_name} with args: {list(arguments.keys())}")

        # ── Security intercepts ──────────────────────────────────────────────
        if tool_name == "request_core_update":
            return RouterResult(
                status="mfa_required",
                mfa_tool_name=tool_name,
                mfa_arguments=arguments,
            )

        if tool_name == "ask_admin_for_guidance":
            msg = (
                f"Guidance Needed: {arguments.get('context_summary', '')}\n"
                f"Question: {arguments.get('specific_question', '')}"
            )
            return RouterResult(status="hitl_required", hitl_message=msg)

        if tool_name == "request_capability":
            gap_description = arguments.get("gap_description", "unspecified gap")
            suggested_tool_name = arguments.get("suggested_tool_name", "new_tool")
            existing_tool = self._find_existing_capability_match(
                gap_description,
                suggested_tool_name,
            )
            if existing_tool is not None:
                matched_name = existing_tool.get("name", "existing_tool")
                logger.info(
                    "Capability-gap request matched existing tool '%s'; skipping synthesis.",
                    matched_name,
                )
                return RouterResult(
                    status="ok",
                    content=(
                        f"Existing capability already available: use the '{matched_name}' tool instead of requesting a new one. "
                        f"Tool purpose: {existing_tool.get('description', '').strip()}"
                    ),
                )
            return RouterResult(
                status="capability_gap",
                gap_description=gap_description,
                suggested_tool_name=suggested_tool_name,
            )

        if tool_name == "escalate_to_system_2":
            return RouterResult(
                status="cognitive_escalation",
                escalation_problem=arguments.get("problem_description", "unspecified problem"),
                escalation_context=arguments.get("context_scratchpad", ""),
            )

        # ── Registry dispatch ────────────────────────────────────────────────
        try:
            content = await asyncio.wait_for(
                self.registry.execute(tool_name, arguments),
                timeout=TOOL_EXEC_TIMEOUT_SECONDS,
            )
            return RouterResult(status="ok", content=content)
        except asyncio.TimeoutError:
            return RouterResult(
                status="ok",
                content=(
                    f"Error: Tool '{tool_name}' exceeded "
                    f"{int(TOOL_EXEC_TIMEOUT_SECONDS)}s timeout."
                ),
            )

    @staticmethod
    def sanitize_response(text: str) -> str:
        """
        Strip internal reasoning, tool schemas, and critic annotations from
        LLM output before it reaches the end user.

        Removes:
        - <think>…</think> and <reasoning>…</reasoning> blocks (local model CoT)
        - [CRITIC FEEDBACK: …] injected annotations
        - WORKERS: […] supervisor tags
        - [ADMIN GUIDANCE: …] and [HEARTBEAT TASK …] prefixes
        - Standalone JSON objects that look like tool-call payloads
        """
        if not text:
            return text

        # Chain-of-thought blocks produced by some local models
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Internal annotation tags injected by the orchestrator loop
        text = re.sub(r'\[CRITIC FEEDBACK[^\n]*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[ADMIN GUIDANCE[^\n]*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[HEARTBEAT TASK[^\n]*\n?', '', text, flags=re.IGNORECASE)

        # Supervisor planning tag
        text = re.sub(r'WORKERS:\s*\[.*?\]\s*\n?', '', text, flags=re.IGNORECASE)

        # Standalone JSON blobs that look like tool-call payloads
        # Matches a top-level {...} block containing known internal keys
        text = re.sub(
            r'(?m)^\s*\{[^{}]*"(?:tool_call|tool_name|function_call|function|name)"[^{}]*\}\s*$',
            '',
            text,
            flags=re.IGNORECASE,
        )

        # Internal planning/scratchpad sections the model sometimes generates
        # Strips: ## ⚙️ Section Title, [Output Draft], [Internal Critique], [Finalized Deliverable]
        text = re.sub(r'^#{1,3}\s.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[Output Draft\][^\[]*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\[Internal Critique\][^\[]*', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\[Finalized Deliverable\]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'---+\s*\n', '', text)  # horizontal rule separators

        # Collapse leftover blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        logged = text[:120].replace('\n', ' ')
        logger.debug(f"sanitize_response output preview: {logged!r}")
        return text.strip()

    @staticmethod
    def _parse_bool_env(name: str, default: bool = False) -> bool:
        raw = os.getenv(name, "").strip().lower()
        if not raw:
            return default
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        logger.warning("Invalid %s=%r. Falling back to %s.", name, raw, default)
        return default

    @staticmethod
    def _parse_int_env(name: str) -> Optional[int]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid %s=%r. Ignoring.", name, raw)
            return None

    @staticmethod
    def _parse_float_env(name: str) -> Optional[float]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid %s=%r. Ignoring.", name, raw)
            return None

    @staticmethod
    def _parse_positive_int_env(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value <= 0:
                raise ValueError(raw)
            return value
        except ValueError:
            logger.warning("Invalid %s=%r. Falling back to %s.", name, raw, default)
            return default

    @staticmethod
    def _resolve_parallel_read_only_tool_concurrency() -> int:
        raw = os.getenv(
            "PARALLEL_READ_ONLY_TOOL_CONCURRENCY",
            str(DEFAULT_PARALLEL_READ_ONLY_TOOL_CONCURRENCY),
        ).strip()
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "Invalid PARALLEL_READ_ONLY_TOOL_CONCURRENCY=%r. Falling back to %s.",
                raw,
                DEFAULT_PARALLEL_READ_ONLY_TOOL_CONCURRENCY,
            )
            return DEFAULT_PARALLEL_READ_ONLY_TOOL_CONCURRENCY
        return max(1, min(5, value))

    @staticmethod
    def _resolve_ollama_keep_alive() -> Optional[str]:
        raw = os.getenv("OLLAMA_KEEP_ALIVE", "15m").strip()
        return raw or None

    @staticmethod
    def _resolve_ollama_think() -> Optional[Any]:
        raw = os.getenv("OLLAMA_THINK", "false").strip().lower()
        if not raw:
            return None
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        if raw in {"low", "medium", "high"}:
            return raw
        logger.warning("Invalid OLLAMA_THINK=%r. Falling back to False.", raw)
        return False

    @staticmethod
    def _extract_context_length(modelinfo: Dict[str, Any]) -> Optional[int]:
        for key, value in (modelinfo or {}).items():
            if not str(key).endswith(".context_length"):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _resolve_ollama_runtime_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {}

        num_gpu = self._parse_int_env("OLLAMA_NUM_GPU")
        if num_gpu is not None and num_gpu != -1:
            options["num_gpu"] = num_gpu

        int_options = {
            "OLLAMA_NUM_CTX": "num_ctx",
            "OLLAMA_REPEAT_LAST_N": "repeat_last_n",
            "OLLAMA_NUM_PREDICT": "num_predict",
            "OLLAMA_TOP_K": "top_k",
            "OLLAMA_SEED": "seed",
            "OLLAMA_NUM_BATCH": "num_batch",
            "OLLAMA_NUM_THREAD": "num_thread",
        }
        for env_name, option_name in int_options.items():
            value = self._parse_int_env(env_name)
            if value is not None:
                options[option_name] = value

        float_options = {
            "OLLAMA_TEMPERATURE": "temperature",
            "OLLAMA_TOP_P": "top_p",
            "OLLAMA_MIN_P": "min_p",
            "OLLAMA_REPEAT_PENALTY": "repeat_penalty",
            "OLLAMA_PRESENCE_PENALTY": "presence_penalty",
            "OLLAMA_FREQUENCY_PENALTY": "frequency_penalty",
        }
        for env_name, option_name in float_options.items():
            value = self._parse_float_env(env_name)
            if value is not None:
                options[option_name] = value

        return options

    def _inspect_system_1_model(self) -> None:
        try:
            show = ollama.Client().show(self.local_model)
            self._system_1_capabilities = set(getattr(show, "capabilities", []) or [])
            self._system_1_modelinfo = dict(getattr(show, "modelinfo", {}) or {})
            self._system_1_context_length = self._extract_context_length(self._system_1_modelinfo)

            if self._ollama_think is not None and "thinking" not in self._system_1_capabilities:
                logger.warning(
                    "OLLAMA_THINK is configured but model '%s' does not advertise thinking support. Disabling think override.",
                    self.local_model,
                )
                self._ollama_think = None

            if "tools" not in self._system_1_capabilities:
                logger.warning(
                    "System 1 model '%s' does not advertise tool support. Tool-calling reliability may be degraded.",
                    self.local_model,
                )

            logger.info(
                "System 1 model inspected: capabilities=%s context_length=%s options=%s",
                ", ".join(sorted(self._system_1_capabilities)) or "unknown",
                self._system_1_context_length or "unknown",
                self._ollama_options or {},
            )
        except Exception as exc:
            logger.warning("Unable to inspect System 1 model '%s': %s", self.local_model, exc)

    def _build_system_1_chat_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools=None,
    ) -> Dict[str, Any]:
        ollama_options = getattr(self, "_ollama_options", {}) or {}
        ollama_keep_alive = getattr(self, "_ollama_keep_alive", None)
        ollama_think = getattr(self, "_ollama_think", None)
        system_1_capabilities = getattr(self, "_system_1_capabilities", set()) or set()
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": ollama_options,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if ollama_keep_alive is not None:
            kwargs["keep_alive"] = ollama_keep_alive
        if ollama_think is not None and (
            not system_1_capabilities or "thinking" in system_1_capabilities
        ):
            kwargs["think"] = ollama_think
        return kwargs

    @staticmethod
    def _coerce_message_audio_bytes(message: Dict[str, Any]) -> bytes:
        raw = message.get("audio_bytes")
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, bytearray):
            return bytes(raw)
        if isinstance(raw, memoryview):
            return raw.tobytes()
        return b""

    @staticmethod
    def _coerce_message_text(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        return str(raw)

    def _prepare_system_1_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for msg in messages or []:
            role = str(msg.get("role") or "user")
            content = self._coerce_message_text(msg.get("content", ""))
            normalized: Dict[str, Any] = {
                "role": role,
                "content": content,
            }

            audio_bytes = self._coerce_message_audio_bytes(msg)
            if audio_bytes and role == "user":
                mime_type = str(msg.get("audio_mime_type") or "audio/ogg")
                if self._enable_native_audio and "audio" in self._system_1_capabilities:
                    normalized["audios"] = [base64.b64encode(audio_bytes).decode("ascii")]
                    normalized["content"] = (f"{content}\n\n[Voice note attached]").strip()
                else:
                    fallback = (
                        f"[Voice note attached: {len(audio_bytes)} bytes ({mime_type}). "
                        "Native audio decoding is unavailable in this runtime. "
                        "Use the textual context and ask for clarification if needed.]"
                    )
                    normalized["content"] = (f"{content}\n\n{fallback}").strip()

            prepared.append(normalized)
        return prepared

    def _prepare_system_2_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for msg in messages or []:
            normalized: Dict[str, Any] = {
                "role": str(msg.get("role") or "user"),
                "content": self._coerce_message_text(msg.get("content", "")),
            }
            if "tool_calls" in msg:
                normalized["tool_calls"] = msg.get("tool_calls")
            if "tool_call_id" in msg:
                normalized["tool_call_id"] = msg.get("tool_call_id")
            sanitized.append(normalized)
        return sanitized

    def __init__(self, model_name: str = "gemini-2.0-flash", local_model: str = "gemma4:e4b") -> None:
        self.model_name = model_name
        self.local_model = os.getenv("OLLAMA_MODEL", local_model)
        env_timeout = os.getenv("OLLAMA_TIMEOUT", "").strip()
        if env_timeout:
            self._ollama_timeout = float(env_timeout)
        else:
            self._ollama_timeout = 120.0 if "26b" in self.local_model else 60.0
        self._system_1_max_concurrency = self._resolve_system_1_max_concurrency()
        self._system_1_semaphore = asyncio.Semaphore(self._system_1_max_concurrency)
        self._system_1_active_requests = 0
        self._system_1_waiting_requests = 0
        self._system_1_wait_events = 0
        self._system_1_total_wait_seconds = 0.0
        self._system_1_peak_waiting_requests = 0
        self._system2_cooldown_until = 0.0
        # Optional callback to persist cooldown expiry to DB; set by Orchestrator.async_init
        self._persist_cooldown_cb = None  # Optional[Callable[[float], Awaitable[None]]]
        # Optional callback to lazily resolve scoped SKILL.md context per tool call.
        self._skill_context_resolver = None

        # SkillRegistry — single source of truth for all tool schemas and callables
        self.registry = SkillRegistry()
        self._dynamic_tool_worker = DynamicToolWorkerClient(
            call_timeout_seconds=TOOL_EXEC_TIMEOUT_SECONDS,
        )
        self.registry.set_dynamic_tool_worker(self._dynamic_tool_worker)

        # System 1: cached Ollama client — created once, reused across all calls
        self._ollama_client: Optional[ollama.AsyncClient] = None
        self._ollama_keep_alive = self._resolve_ollama_keep_alive()
        self._ollama_think = self._resolve_ollama_think()
        self._preload_system_1_on_startup = self._parse_bool_env("SYSTEM_1_PRELOAD_ON_STARTUP", False)
        self._enable_128k_context = self._parse_bool_env("ENABLE_128K_CONTEXT", False)
        self._enable_native_audio = self._parse_bool_env("ENABLE_NATIVE_AUDIO", False)
        self._enable_parallel_tools = self._parse_bool_env("ENABLE_PARALLEL_TOOLS", False)
        self._enable_scoped_skill_context = self._parse_bool_env("ENABLE_SCOPED_SKILL_CONTEXT", True)
        self._parallel_read_only_tool_concurrency = self._resolve_parallel_read_only_tool_concurrency()
        self._target_context_tokens = self._parse_positive_int_env(
            "SYSTEM_1_TARGET_CONTEXT_TOKENS",
            DEFAULT_TARGET_CONTEXT_TOKENS,
        )
        self._ingestion_context_reserve_tokens = self._parse_positive_int_env(
            "INGESTION_CONTEXT_RESERVE_TOKENS",
            DEFAULT_CONTEXT_RESERVE_TOKENS,
        )
        self._ingestion_min_context_chars = self._parse_positive_int_env(
            "INGESTION_MIN_CONTEXT_CHARS",
            DEFAULT_MIN_INGESTION_CHARS,
        )
        self._system_1_capabilities: set[str] = set()
        self._system_1_modelinfo: Dict[str, Any] = {}
        self._system_1_context_length: Optional[int] = None

        # Ollama runtime options — these map directly to Modelfile/API controls.
        self._ollama_options = self._resolve_ollama_runtime_options()
        if self._ollama_options.get("num_gpu") == 0:
            logger.info("System 1: CPU-only mode (OLLAMA_NUM_GPU=0)")
        self._inspect_system_1_model()

        # System 2 provider selection:
        #   Priority: Ollama Cloud > Groq > Gemini
        self.groq_client = None
        self.gemini_client = None
        self.ollama_cloud_client: Optional[ollama.AsyncClient] = None
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.system_2_model = SYSTEM_2_MODEL

        if OLLAMA_CLOUD_API_KEY:
            self.ollama_cloud_client = ollama.AsyncClient(
                host=OLLAMA_CLOUD_HOST,
                headers={'Authorization': f'Bearer {OLLAMA_CLOUD_API_KEY}'},
            )
            logger.info(f"System 2: Ollama Cloud ({self.system_2_model} @ {OLLAMA_CLOUD_HOST}) — API key configured")
        elif GROQ_AVAILABLE and GROQ_API_KEY:
            self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)
            logger.info(f"System 2: Groq ({self.groq_model}) — API key configured")
        elif not GROQ_AVAILABLE and GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is set but the 'groq' package is not installed. "
                "Run: pip install groq"
            )
        elif GEMINI_API_KEY and USE_GEMINI and GENAI_AVAILABLE:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"System 2: Gemini ({model_name}) — API key configured")
        elif GEMINI_API_KEY and USE_GEMINI and not GENAI_AVAILABLE:
            logger.warning(
                "GEMINI_API_KEY + USE_GEMINI=True is set but the 'google-genai' package "
                "is not installed. Run: pip install google-genai"
            )
        elif GEMINI_API_KEY and not USE_GEMINI:
            logger.warning(
                "GEMINI_API_KEY is set but USE_GEMINI is not 'True'. "
                "Set USE_GEMINI=True in .env to activate Gemini as System 2."
            )
        else:
            logger.warning(
                "No System 2 provider configured — capability synthesis will be unavailable. "
                "Set OLLAMA_CLOUD_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY+USE_GEMINI=True."
            )

        logger.info(
            "CognitiveRouter initialized. System 1: %s (max concurrency: %s, keep_alive=%s, think=%s)",
            self.local_model,
            self._system_1_max_concurrency,
            self._ollama_keep_alive or "default",
            self._ollama_think,
        )
        logger.info(
            "Feature flags: ENABLE_128K_CONTEXT=%s ENABLE_NATIVE_AUDIO=%s ENABLE_PARALLEL_TOOLS=%s ENABLE_SCOPED_SKILL_CONTEXT=%s",
            self._enable_128k_context,
            self._enable_native_audio,
            self._enable_parallel_tools,
            self._enable_scoped_skill_context,
        )
        logger.info(
            "Parallel tool execution: read-only concurrency=%s",
            self._parallel_read_only_tool_concurrency,
        )

    @staticmethod
    def _resolve_system_1_max_concurrency() -> int:
        env_concurrency = os.getenv("SYSTEM_1_MAX_CONCURRENCY", "1").strip()
        try:
            return max(1, int(env_concurrency or "1"))
        except ValueError:
            logger.warning(
                "Invalid SYSTEM_1_MAX_CONCURRENCY=%r. Falling back to 1.",
                env_concurrency,
            )
            return 1

    @asynccontextmanager
    async def _system_1_slot(self) -> AsyncIterator[None]:
        wait_started = time.perf_counter()
        self._system_1_waiting_requests += 1
        self._system_1_peak_waiting_requests = max(
            self._system_1_peak_waiting_requests,
            self._system_1_waiting_requests,
        )
        gate_busy = self._system_1_semaphore.locked()
        if gate_busy:
            logger.info(
                "System 1 concurrency gate busy; queuing local request (active=%s waiting=%s max=%s)",
                self._system_1_active_requests,
                self._system_1_waiting_requests,
                self._system_1_max_concurrency,
            )

        _queue_timeout = float(os.environ.get("SYSTEM_1_QUEUE_TIMEOUT_SECONDS", "120"))
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._system_1_semaphore.acquire(), timeout=_queue_timeout)
            except asyncio.TimeoutError:
                self._system_1_waiting_requests = max(0, self._system_1_waiting_requests - 1)
                logger.error(
                    "System 1 queue timeout after %.0fs — dropping request (active=%s waiting=%s)",
                    _queue_timeout,
                    self._system_1_active_requests,
                    self._system_1_waiting_requests,
                )
                raise
            acquired = True
            wait_seconds = time.perf_counter() - wait_started
            self._system_1_waiting_requests -= 1
            self._system_1_active_requests += 1

            if gate_busy or wait_seconds >= 0.001:
                self._system_1_wait_events += 1
                self._system_1_total_wait_seconds += wait_seconds
                logger.info(
                    "System 1 slot acquired after %.3fs wait (active=%s waiting=%s max=%s)",
                    wait_seconds,
                    self._system_1_active_requests,
                    self._system_1_waiting_requests,
                    self._system_1_max_concurrency,
                )

            yield
        finally:
            if acquired:
                self._system_1_active_requests = max(0, self._system_1_active_requests - 1)
                self._system_1_semaphore.release()
                logger.debug(
                    "System 1 slot released (active=%s waiting=%s max=%s)",
                    self._system_1_active_requests,
                    self._system_1_waiting_requests,
                    self._system_1_max_concurrency,
                )
            else:
                self._system_1_waiting_requests = max(0, self._system_1_waiting_requests - 1)

    def get_system_1_gate_metrics(self) -> Dict[str, float]:
        average_wait = (
            self._system_1_total_wait_seconds / self._system_1_wait_events
            if self._system_1_wait_events else 0.0
        )
        return {
            "active_requests": self._system_1_active_requests,
            "waiting_requests": self._system_1_waiting_requests,
            "max_concurrency": self._system_1_max_concurrency,
            "wait_events": self._system_1_wait_events,
            "total_wait_seconds": self._system_1_total_wait_seconds,
            "average_wait_seconds": average_wait,
            "peak_waiting_requests": self._system_1_peak_waiting_requests,
        }

    async def close(self) -> None:
        """Release cached client and dynamic worker resources."""
        dynamic_worker = getattr(self, "_dynamic_tool_worker", None)
        if dynamic_worker is not None:
            try:
                await dynamic_worker.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down dynamic tool worker: {e}")
            self._dynamic_tool_worker = None
        if getattr(self, "_ollama_client", None) is not None:
            try:
                await self._ollama_client.close()
            except Exception as e:
                logger.debug(f"Error closing Ollama client: {e}")
            self._ollama_client = None

    def set_dynamic_tool_restart_callback(self, callback) -> None:
        worker = getattr(self, "_dynamic_tool_worker", None)
        if worker is not None:
            worker.set_restart_callback(callback)

    @staticmethod
    def _find_json_blobs(text: str) -> List[str]:
        """Extract all balanced top-level JSON objects from *text*."""
        blobs: List[str] = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    blobs.append(text[start:i + 1])
                    start = -1
        return blobs

    @staticmethod
    def _is_pdf_extraction_tool(tool_name: str) -> bool:
        _pdf_variants = {
            "extract_pdf_text", "extract_text_from_file", "extract_text",
            "extract_pdf_text_from_file", "read_pdf", "pdf_extract",
        }
        return tool_name in _pdf_variants

    @staticmethod
    def _is_web_extraction_tool(tool_name: str) -> bool:
        _web_variants = {
            "extract_web_article", "extract_article", "fetch_article", "scrape_web",
        }
        return tool_name in _web_variants

    @classmethod
    def _is_context_ingestion_tool(cls, tool_name: str) -> bool:
        return cls._is_pdf_extraction_tool(tool_name) or cls._is_web_extraction_tool(tool_name)

    @staticmethod
    def _coerce_tool_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _resolve_effective_context_limit_tokens(self) -> int:
        inspected = self._coerce_positive_int(getattr(self, "_system_1_context_length", None))
        configured = self._coerce_positive_int((getattr(self, "_ollama_options", {}) or {}).get("num_ctx"))
        target = self._coerce_positive_int(getattr(self, "_target_context_tokens", None)) or DEFAULT_TARGET_CONTEXT_TOKENS
        enable_128k = bool(getattr(self, "_enable_128k_context", False))

        limits = [value for value in (inspected, configured) if value is not None]
        if enable_128k:
            limits.append(target)

        if limits:
            return min(limits)
        return target if enable_128k else 8192

    @staticmethod
    def _estimate_content_chars(content: Any) -> int:
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))
                else:
                    total += len(str(part))
            return total
        if content is None:
            return 0
        return len(str(content))

    @staticmethod
    def _estimate_tool_calls_chars(tool_calls: Any) -> int:
        if not tool_calls:
            return 0
        try:
            return len(json.dumps(tool_calls, default=str))
        except Exception:
            return len(str(tool_calls))

    @classmethod
    def _estimate_messages_char_usage(cls, messages: List[dict]) -> int:
        total_chars = 0
        for msg in messages:
            total_chars += cls._estimate_content_chars(msg.get("content", ""))
            total_chars += cls._estimate_tool_calls_chars(msg.get("tool_calls"))
            total_chars += 32
        return total_chars

    def _estimate_remaining_context_chars(self, current_messages: List[dict]) -> int:
        token_limit = self._resolve_effective_context_limit_tokens()
        reserve_tokens = self._coerce_positive_int(
            getattr(self, "_ingestion_context_reserve_tokens", DEFAULT_CONTEXT_RESERVE_TOKENS)
        ) or DEFAULT_CONTEXT_RESERVE_TOKENS
        used_chars = self._estimate_messages_char_usage(current_messages)
        used_tokens = int(math.ceil(used_chars / _APPROX_CHARS_PER_TOKEN))
        remaining_tokens = max(0, token_limit - reserve_tokens - used_tokens)
        return remaining_tokens * _APPROX_CHARS_PER_TOKEN

    def _prepare_ingestion_tool_arguments(
        self,
        tool_name: str,
        arguments: dict,
        current_messages: List[dict],
    ) -> tuple[dict, Optional[RouterResult]]:
        prepared = dict(arguments or {})
        if not self._is_context_ingestion_tool(tool_name):
            return prepared, None

        full_context = self._coerce_tool_bool(prepared.get("full_context", False), default=False)
        prepared["full_context"] = full_context
        if not full_context:
            return prepared, None

        if not bool(getattr(self, "_enable_128k_context", False)):
            return prepared, RouterResult(
                status="ok",
                content=(
                    "Error: full_context document ingestion is disabled. "
                    "Set ENABLE_128K_CONTEXT=True in .env to enable 128K-context ingestion."
                ),
            )

        remaining_chars = self._estimate_remaining_context_chars(current_messages)
        min_chars = self._coerce_positive_int(
            getattr(self, "_ingestion_min_context_chars", DEFAULT_MIN_INGESTION_CHARS)
        ) or DEFAULT_MIN_INGESTION_CHARS
        if remaining_chars < min_chars:
            context_limit = self._resolve_effective_context_limit_tokens()
            return prepared, RouterResult(
                status="ok",
                content=(
                    "Error: Insufficient context budget for full_context ingestion "
                    f"(remaining~{remaining_chars} chars, minimum required~{min_chars}, "
                    f"context_limit~{context_limit} tokens). Please narrow the request "
                    "or start a fresh turn before ingesting the full document."
                ),
            )

        requested_max_chars = self._coerce_positive_int(prepared.get("max_chars"))
        dynamic_max_chars = remaining_chars if requested_max_chars is None else min(remaining_chars, requested_max_chars)
        prepared["max_chars"] = max(200, dynamic_max_chars)

        if self._is_pdf_extraction_tool(tool_name) and prepared.get("max_pages") is None:
            # Full-context mode reads all pages unless the caller requests a stricter page cap.
            prepared["max_pages"] = 0

        logger.info(
            "Context handshake for %s: full_context=%s max_chars=%s remaining~%s chars",
            tool_name,
            full_context,
            prepared.get("max_chars"),
            remaining_chars,
        )
        return prepared, None

    def _normalize_tool_name(self, raw_name: str) -> str:
        """Best-effort normalization of hallucinated tool names.

        Steps:
        1. Strip namespace prefixes (e.g. ``skill:func`` → ``func``).
        2. Check the skill registry alias map (exact match).
        3. Fuzzy-match against registered skill names using difflib.
        """
        if not raw_name:
            return raw_name

        # Strip colon-separated namespace prefix
        if ":" in raw_name:
            raw_name = raw_name.rsplit(":", 1)[-1].strip()

        # Already a registered name?
        registry = getattr(self, "registry", None)
        skill_names = registry.get_skill_names() if registry is not None else []
        if raw_name in skill_names:
            return raw_name

        # Underscore-normalized lookup (e.g., "extractPdfText" won't match,
        # but "extract_pdf_text_from_file" might after alias map in registry)
        # Let the registry alias map handle it — but try fuzzy match first
        matches = difflib.get_close_matches(raw_name, skill_names, n=1, cutoff=0.6)
        if matches:
            resolved = matches[0]
            logger.info(f"Fuzzy-matched tool name '{raw_name}' → '{resolved}'")
            return resolved

        # Return as-is; registry.execute() will return an error for unknowns
        logger.debug(
            "_normalize_tool_name: no registry match for %r - returning as-is "
            "(will likely result in 'Unknown tool' error)",
            raw_name,
        )
        return raw_name

    @staticmethod
    def _is_invalid_tool_result(text: str) -> bool:
        """Detect tool-execution failures caused by hallucinated names/args."""
        if not text:
            return False
        lowered = text.strip().lower()
        return (
            lowered.startswith("error: unknown tool")
            or lowered.startswith("error: bad arguments")
        )

    @staticmethod
    def _extract_failed_generation(err_text: str) -> str:
        if not err_text:
            return ""
        m = re.search(r"failed_generation':\s*'(.+?)'\s*}\s*$", err_text, re.DOTALL)
        if not m:
            return ""
        raw = m.group(1)
        return raw.replace("\\n", "\n")

    @staticmethod
    def _extract_failed_generation_tool_call(text: str) -> Optional[tuple]:
        if not text:
            return None

        start = text.find("<function=")
        if start == -1:
            return None

        end = text.find("</function>", start)
        if end == -1:
            return None

        payload = text[start + len("<function="):end].strip()
        name_match = re.match(r"(\w+)", payload)
        if name_match is None:
            return None

        name = name_match.group(1).strip()
        remainder = payload[name_match.end():].strip()
        while remainder.startswith(("[", "]")):
            remainder = remainder[1:].lstrip()

        if not remainder.startswith("{"):
            return name, {}

        try:
            args, _ = json.JSONDecoder().raw_decode(remainder)
        except (json.JSONDecodeError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        return name, args

    @staticmethod
    def _extract_args_from_blob(obj: dict) -> dict:
        """Parse tool arguments from an inline tool call JSON object."""
        # Look for an explicit argument container key.  Use 'in' so that an
        # empty dict value is still honoured and not silently skipped.
        for key in ("arguments", "parameters", "args"):
            if key not in obj:
                continue
            raw_args = obj[key]
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except ValueError:
                    raw_args = {}
            if isinstance(raw_args, dict):
                return raw_args
            break  # key found but unusable type — fall through to flat extraction

        # No usable container key found: treat every non-meta key as an argument.
        known_meta = {"tool_name", "name", "function_call", "tool_call",
                      "arguments", "parameters", "args"}
        return {k: v for k, v in obj.items() if k not in known_meta}

    @staticmethod
    def _extract_inline_tool_call(text: str) -> Optional[tuple]:
        """
        Fallback parser for models that embed tool call JSON inside the text
        ``content`` field instead of using the native ``tool_calls`` array.

        Looks for a top-level JSON object containing a ``tool_name`` (or
        ``name``) key whose value matches a known meta-tool pattern, plus an
        optional ``arguments`` / ``parameters`` block.

        Returns ``(tool_name, arguments_dict)`` on success, or ``None`` if no
        recognisable inline tool call is found.
        """
        if not text:
            return None

        candidate_text = str(text or "").strip()
        if candidate_text.startswith("```"):
            candidate_text = re.sub(r"^```(?:json)?\s*", "", candidate_text, flags=re.IGNORECASE)
            candidate_text = re.sub(r"\s*```$", "", candidate_text).strip()

        if not (candidate_text.startswith("{") and candidate_text.endswith("}")):
            return None

        for raw in CognitiveRouter._find_json_blobs(candidate_text):
            if raw.strip() != candidate_text:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue

            if not isinstance(obj, dict):
                continue

            name_val = (
                obj.get("tool_name") or obj.get("name")
                or obj.get("function_call") or obj.get("tool_call")
            )
            if not isinstance(name_val, str) or not name_val.strip():
                continue

            raw_args = CognitiveRouter._extract_args_from_blob(obj)
            logger.debug(f"_extract_inline_tool_call: found '{name_val}' in content text")
            return (name_val.strip(), raw_args)

        return None

    def _get_or_create_ollama_client(self):
        """Return the cached Ollama async client, creating it if needed."""
        if self._ollama_client is None:
            self._ollama_client = ollama.AsyncClient()
        return self._ollama_client

    def _format_ollama_tools(self, allowed_tools: Optional[List[str]]) -> List[dict]:
        """Build Ollama-format tool descriptors from the SkillRegistry."""
        result = []
        for tool in self.registry.get_schemas():
            if allowed_tools is not None and tool["name"] not in allowed_tools:
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            })
        return result

    def _format_cooldown_message(self) -> str:
        remaining = max(0, int(self._system2_cooldown_until - time.time()))
        if remaining <= 0:
            return ""
        minutes, seconds = divmod(remaining, 60)
        return f"Rate limited. Retry in {minutes}m{seconds}s."

    async def _call_ollama_with_model_fallback(self, client, messages, tools):
        """Call Ollama chat with the configured local model.

        Raises on failure so the orchestrator's S1→S2 escalation logic takes over.
        Returns ``(response, model_name_used)``.
        """
        model = self.local_model
        response = await asyncio.wait_for(
            client.chat(**self._build_system_1_chat_kwargs(model, messages, tools=tools)),
            timeout=self._ollama_timeout,
        )
        return response, model

    @staticmethod
    def _extract_tool_call_function_data(raw_tc: Any) -> Any:
        if hasattr(raw_tc, "function"):
            return raw_tc.function
        if isinstance(raw_tc, dict):
            return raw_tc.get("function", raw_tc)
        return raw_tc

    @staticmethod
    def _coerce_tool_arguments(raw_args: Any) -> dict:
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except ValueError:
                logger.warning(f"Tool call args could not be JSON-parsed: {raw_args!r}")
                return {}
        if isinstance(raw_args, dict):
            return raw_args
        try:
            return dict(raw_args) if raw_args else {}
        except Exception:
            return {}

    @staticmethod
    def _parse_native_tool_call_from_raw(raw_tc: Any) -> tuple:
        """Extract (tool_name, arguments, success) from a single Ollama tool_call item."""
        try:
            fn_data = CognitiveRouter._extract_tool_call_function_data(raw_tc)
            tool_name = (
                fn_data.get("name", "") if isinstance(fn_data, dict)
                else getattr(fn_data, "name", "")
            )
            raw_args = (
                fn_data.get("arguments", {}) if isinstance(fn_data, dict)
                else getattr(fn_data, "arguments", {})
            )
            raw_args = CognitiveRouter._coerce_tool_arguments(raw_args)
            if not tool_name:
                raise ValueError(f"Tool call missing 'name': {raw_tc!r}")
            # Strip namespace prefix (e.g. "extract_pdf_text:extract_pdf_text" → "extract_pdf_text")
            if ":" in tool_name:
                tool_name = tool_name.rsplit(":", 1)[-1].strip()
            return tool_name, raw_args, True
        except Exception as exc:
            logger.error(f"Failed to parse tool_calls: {exc!r}", exc_info=True)
            return "", {}, False

    @staticmethod
    def _parse_native_tool_call(message: dict) -> tuple:
        """Extract (tool_name, arguments, success) from an Ollama tool_calls message."""
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return "", {}, False
        return CognitiveRouter._parse_native_tool_call_from_raw(tool_calls[0])

    @staticmethod
    def _parse_native_tool_calls(message: dict) -> List[tuple[str, dict]]:
        parsed: List[tuple[str, dict]] = []
        for raw_tc in message.get("tool_calls") or []:
            tool_name, arguments, ok = CognitiveRouter._parse_native_tool_call_from_raw(raw_tc)
            if ok:
                parsed.append((tool_name, arguments))
        return parsed

    @staticmethod
    def _message_has_tool_calls(message: dict) -> bool:
        return bool("tool_calls" in message and message["tool_calls"])

    def _should_continue_tool_loop(self, message: dict, iters: int) -> bool:
        return self._message_has_tool_calls(message) and iters < 10

    @staticmethod
    def _is_read_only_tool_call(tool_name: str, arguments: dict) -> bool:
        if tool_name == "manage_file_system":
            action = str((arguments or {}).get("action") or "").strip().lower()
            return action in {"read", "list"}
        return tool_name in _READ_ONLY_PARALLEL_TOOL_NAMES

    async def _execute_tool_calls_sequential(self, tool_calls: List[tuple[str, dict]]) -> List[RouterResult]:
        results: List[RouterResult] = []
        for tool_name, arguments in tool_calls:
            results.append(await self._execute_tool(tool_name, arguments))
        return results

    async def _execute_read_only_tool_calls_parallel(
        self,
        tool_calls: List[tuple[str, dict]],
    ) -> List[RouterResult]:
        if not tool_calls:
            return []

        max_concurrency = max(
            1,
            min(
                5,
                int(getattr(
                    self,
                    "_parallel_read_only_tool_concurrency",
                    DEFAULT_PARALLEL_READ_ONLY_TOOL_CONCURRENCY,
                )),
            ),
        )
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_single(tool_name: str, arguments: dict) -> RouterResult:
            async with semaphore:
                return await self._execute_tool(tool_name, arguments)

        return list(
            await asyncio.gather(
                *(_run_single(tool_name, arguments) for tool_name, arguments in tool_calls)
            )
        )

    async def _execute_tool_calls_with_partition(
        self,
        tool_calls: List[tuple[str, dict]],
    ) -> List[RouterResult]:
        if not tool_calls:
            return []

        if not bool(getattr(self, "_enable_parallel_tools", False)):
            return await self._execute_tool_calls_sequential(tool_calls)

        results: List[RouterResult] = []
        idx = 0
        while idx < len(tool_calls):
            tool_name, arguments = tool_calls[idx]
            if self._is_read_only_tool_call(tool_name, arguments):
                segment: List[tuple[str, dict]] = []
                end_idx = idx
                while end_idx < len(tool_calls):
                    seg_name, seg_args = tool_calls[end_idx]
                    if not self._is_read_only_tool_call(seg_name, seg_args):
                        break
                    segment.append((seg_name, seg_args))
                    end_idx += 1

                if len(segment) > 1:
                    logger.info(
                        "Parallel tool batch: %s read-only calls (concurrency=%s)",
                        len(segment),
                        getattr(
                            self,
                            "_parallel_read_only_tool_concurrency",
                            DEFAULT_PARALLEL_READ_ONLY_TOOL_CONCURRENCY,
                        ),
                    )
                    results.extend(await self._execute_read_only_tool_calls_parallel(segment))
                else:
                    results.extend(await self._execute_tool_calls_sequential(segment))

                idx = end_idx
                continue

            # Mutating/unknown calls always execute serially to avoid races.
            results.extend(await self._execute_tool_calls_sequential([(tool_name, arguments)]))
            idx += 1

        return results

    @staticmethod
    def _build_tool_signature(tool_name: str, arguments: dict) -> str:
        try:
            return json.dumps(
                {"tool_name": tool_name, "arguments": arguments or {}},
                sort_keys=True,
                default=str,
            )
        except Exception:
            return f"{tool_name}:{arguments!r}"

    def _track_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        seen_tool_calls: set,
    ) -> str:
        tool_signature = self._build_tool_signature(tool_name, arguments)
        if tool_signature in seen_tool_calls:
            logger.info(
                f"System 1 repeated tool call '{tool_name}' — forcing text synthesis instead"
            )
            return (
                "Do not call any more tools. Using only the successful tool results above, "
                "answer the user's request directly."
            )

        seen_tool_calls.add(tool_signature)
        return ""

    def _handle_invalid_tool_request(
        self,
        exec_result: RouterResult,
        tool_name: str,
        current_messages: List[dict],
        original_messages: List[dict],
    ) -> tuple[Optional[RouterResult], str]:
        if not self._is_invalid_tool_result(exec_result.content):
            return None, ""

        logger.warning(
            f"System 1 requested invalid tool usage for '{tool_name}': {exec_result.content}"
        )
        if len(current_messages) == len(original_messages):
            return RouterResult(
                status="ok",
                content=f"[System 1 - Error]: {exec_result.content}",
            ), ""

        return None, (
            "A later tool request was invalid. Do not call any more tools. "
            "Using only the successful tool results above, answer the user's request directly."
        )

    def _process_tool_execution_result(
        self,
        exec_result: RouterResult,
        tool_name: str,
        current_messages: List[dict],
        original_messages: List[dict],
    ) -> tuple[Optional[RouterResult], str]:
        if exec_result.status != "ok":
            return exec_result, ""
        return self._handle_invalid_tool_request(
            exec_result,
            tool_name,
            current_messages,
            original_messages,
        )

    @staticmethod
    def _append_tool_result_messages(
        current_messages: List[dict],
        tool_name: str,
        arguments: dict,
        tool_output: str,
    ) -> List[dict]:
        return current_messages + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}],
            },
            {"role": "tool", "content": tool_output},
        ]

    async def _chat_with_ollama(
        self,
        client,
        model: str,
        messages: List[dict],
        *,
        tools=None,
    ):
        async with asyncio.timeout(self._ollama_timeout):
            return await client.chat(**self._build_system_1_chat_kwargs(model, messages, tools=tools))

    async def _chat_with_ollama_with_timeout(
        self,
        client,
        model: str,
        messages: List[dict],
        timeout_seconds: float,
        *,
        tools=None,
    ):
        async with asyncio.timeout(timeout_seconds):
            return await client.chat(**self._build_system_1_chat_kwargs(model, messages, tools=tools))

    async def preload_system_1(self) -> None:
        """Warm-load the local model so the first live turn avoids cold-start latency."""
        client = self._get_or_create_ollama_client()
        preload_timeout = min(self._ollama_timeout, 20.0)
        async with asyncio.timeout(preload_timeout):
            await client.chat(
                model=self.local_model,
                messages=[],
                keep_alive=self._ollama_keep_alive,
            )
        logger.info("System 1 preloaded: %s", self.local_model)

    async def _summarize_pdf_tool_output(
        self,
        client,
        model: str,
        tool_name: str,
        tool_output: str,
    ) -> Optional[RouterResult]:
        if not self._is_pdf_extraction_tool(tool_name):
            return None
        if tool_output.strip().lower().startswith("error:"):
            return RouterResult(status="ok", content=tool_output.strip())

        summary_messages = [
            {
                "role": "system",
                "content": "You are a summarization engine. Summarize the provided PDF text faithfully."
            },
            {
                "role": "user",
                "content": (
                    "PDF text:\n"
                    f"{tool_output}\n\n"
                    "Summarize in a concise report. Start with a 3–5 sentence abstract, "
                    "then 5–8 bullet points. If the text ends with [TRUNCATED], mention "
                    "the summary is based on a truncated excerpt."
                ),
            },
        ]
        try:
            force_resp = await self._chat_with_ollama_with_timeout(
                client,
                model,
                summary_messages,
                60.0,
            )
            if force_resp and "message" in force_resp:
                content = (force_resp["message"].get("content") or "").strip()
                if content:
                    logger.info("System 1 PDF summary generated")
                    return RouterResult(status="ok", content=content)
        except Exception as e:
            logger.warning(f"PDF summary synthesis failed: {e}")
        return RouterResult(status="ok", content="Error: Failed to summarize extracted PDF text.")

    async def _fetch_tool_followup_message(
        self,
        client,
        model: str,
        current_messages: List[dict],
        active_tools,
    ) -> Optional[dict]:
        response = await self._chat_with_ollama(
            client,
            model,
            current_messages,
            tools=active_tools,
        )
        if not (response and "message" in response):
            logger.warning("Empty response during tool chain — stopping loop")
            return None
        return response["message"]

    async def _advance_tool_loop(
        self,
        client,
        model: str,
        current_messages: List[dict],
        tool_name: str,
        arguments: dict,
        tool_output: str,
        active_tools,
    ) -> tuple[List[dict], Optional[dict]]:
        updated_messages = self._append_tool_result_messages(
            current_messages,
            tool_name,
            arguments,
            tool_output,
        )
        next_message = await self._fetch_tool_followup_message(
            client,
            model,
            updated_messages,
            active_tools,
        )
        return updated_messages, next_message

    async def _force_tool_result_synthesis(
        self,
        client,
        model: str,
        current_messages: List[dict],
        force_synthesis_prompt: str,
        content: str,
        iters: int,
    ) -> str:
        if not (force_synthesis_prompt or not content) or iters <= 0:
            return content

        current_messages.append({
            "role": "user",
            "content": (
                force_synthesis_prompt
                or "Please summarise all the tool results above in a clear reply."
            ),
        })
        try:
            force_resp = await self._chat_with_ollama(client, model, current_messages)
            if force_resp and "message" in force_resp:
                return (force_resp["message"].get("content") or "").strip()
        except Exception as e:
            logger.warning(f"Force synthesis failed: {e}")
        return content

    async def _inject_scoped_skill_context_if_needed(
        self,
        current_messages: List[dict],
        tool_name: str,
        injected_skill_names: set[str],
    ) -> List[dict]:
        if not bool(getattr(self, "_enable_scoped_skill_context", True)):
            return current_messages

        if tool_name in injected_skill_names:
            return current_messages

        resolver = getattr(self, "_skill_context_resolver", None)
        if not callable(resolver):
            return current_messages

        try:
            maybe_context = resolver(tool_name)
            if asyncio.iscoroutine(maybe_context):
                maybe_context = await maybe_context
        except Exception as exc:
            logger.warning("Failed to resolve scoped skill context for %s: %s", tool_name, exc)
            return current_messages

        context_text = str(maybe_context or "").strip()
        if not context_text:
            return current_messages

        injected_skill_names.add(tool_name)
        updated_messages = list(current_messages)
        updated_messages.append(
            {
                "role": "system",
                "content": context_text,
            }
        )
        logger.info("Injected scoped SKILL.md context for tool '%s' in current turn.", tool_name)
        return updated_messages

    async def _run_tool_loop(self, client, model: str, first_message: dict,
                             messages: List[dict], active_tools) -> RouterResult:
        """Execute a chain of tool calls until the model returns plain text.

        Each iteration: parse tool call → execute → feed result back → re-prompt.
        Caps at 10 iterations; forces a text synthesis call if the limit is hit.
        """
        current_messages = list(messages)
        message = first_message
        iters = 0
        seen_tool_calls = set()
        injected_skill_names: set[str] = set()
        force_synthesis_prompt = ""

        while self._should_continue_tool_loop(message, iters):
            iters += 1
            raw_tool_calls = self._parse_native_tool_calls(message)
            if not raw_tool_calls:
                break

            normalized_calls: List[tuple[str, dict]] = []
            duplicate_forced = False
            for raw_tool_name, raw_arguments in raw_tool_calls:
                tool_name = self._normalize_tool_name(raw_tool_name)
                current_messages = await self._inject_scoped_skill_context_if_needed(
                    current_messages,
                    tool_name,
                    injected_skill_names,
                )
                arguments, budget_result = self._prepare_ingestion_tool_arguments(
                    tool_name,
                    raw_arguments,
                    current_messages,
                )
                if budget_result is not None:
                    return budget_result

                duplicate_prompt = self._track_tool_call(
                    tool_name,
                    arguments,
                    seen_tool_calls,
                )
                if duplicate_prompt:
                    force_synthesis_prompt = duplicate_prompt
                    duplicate_forced = True
                    break

                normalized_calls.append((tool_name, arguments))

            if duplicate_forced and not normalized_calls:
                message = {"content": ""}
                break

            if not normalized_calls:
                break

            exec_results = await self._execute_tool_calls_with_partition(normalized_calls)
            force_after_execution = False

            for batch_idx, ((tool_name, arguments), exec_result) in enumerate(
                zip(normalized_calls, exec_results),
                start=1,
            ):
                invalid_result, invalid_force_prompt = self._process_tool_execution_result(
                    exec_result,
                    tool_name,
                    current_messages,
                    messages,
                )
                if invalid_result is not None:
                    return invalid_result
                if invalid_force_prompt:
                    force_synthesis_prompt = invalid_force_prompt
                    force_after_execution = True

                logger.info(
                    "System 1 tool call #%s.%s: %s → %s chars",
                    iters,
                    batch_idx,
                    tool_name,
                    len(exec_result.content),
                )
                current_messages = self._append_tool_result_messages(
                    current_messages,
                    tool_name,
                    arguments,
                    exec_result.content,
                )

                if force_after_execution:
                    break

            if duplicate_forced or force_after_execution:
                message = {"content": ""}
                break

            next_message = await self._fetch_tool_followup_message(
                client,
                model,
                current_messages,
                active_tools,
            )
            if next_message is None:
                break
            message = next_message

        content = (message.get("content") or "").strip()
        content = await self._force_tool_result_synthesis(
            client,
            model,
            current_messages,
            force_synthesis_prompt,
            content,
            iters,
        )

        if content:
            logger.info(f"System 1 response after {iters} tool call(s) ({len(content)} chars)")
            return RouterResult(status="ok", content=content)
        return RouterResult(status="ok", content="Tool(s) executed but model produced no response.")

    async def _handle_text_response(self, message: dict) -> RouterResult:
        """Handle a plain-text (no tool_calls) response message from Ollama."""
        if "content" not in message:
            return RouterResult(status="ok", content="[System 1 - Error]: Unexpected response format")

        result = (message["content"] or "").strip()
        if not result:
            logger.warning("System 1 returned empty response")
            return RouterResult(status="ok", content="[System 1 - Error]: Empty response from model")

        inline_tc = self._extract_inline_tool_call(result)
        if inline_tc:
            if getattr(self, "_inline_tool_call_streak", 0) >= 1:
                self._last_inline_tool_executed = False
                logger.warning("Inline tool call hard cap reached; returning content as text.")
                return RouterResult(status="ok", content=result)
            inline_name, inline_args = inline_tc
            logger.info(f"System 1 in-text tool call detected: {inline_name}")
            self._inline_tool_call_streak = getattr(self, "_inline_tool_call_streak", 0) + 1
            self._last_inline_tool_executed = True
            exec_result = await self._execute_tool(inline_name, inline_args)
            if exec_result.status != "ok":
                return exec_result
            return RouterResult(status="ok", content=exec_result.content)

        self._last_inline_tool_executed = False
        logger.info(f"System 1 response received ({len(result)} chars)")
        return RouterResult(status="ok", content=result)

    async def route_to_system_1(
        self,
        messages: List[Dict[str, Any]],
        allowed_tools: Optional[List[str]] = None
    ) -> RouterResult:
        """
        Route to System 1 (Local Model) - Fast, pattern-based responses.

        Returns a :class:`RouterResult` whose ``status`` is "ok" on success,
        or a non-ok status when a security intercept fires.
        """
        logger.info(f"Routing to System 1 (Local Model - {self.local_model})")
        try:
            async with self._system_1_slot():
                client = self._get_or_create_ollama_client()
                active_tools = self._format_ollama_tools(allowed_tools) or None
                prepared_messages = self._prepare_system_1_messages(messages)
                response, available_model = await self._call_ollama_with_model_fallback(
                    client, prepared_messages, active_tools
                )

                if not (response and "message" in response):
                    return RouterResult(status="ok", content="[System 1 - Error]: Unexpected response format from Ollama")

                message = response["message"]
                if "tool_calls" in message and message["tool_calls"]:
                    return await self._run_tool_loop(
                        client,
                        available_model,
                        message,
                        list(prepared_messages),
                        active_tools,
                    )

                self._inline_tool_call_streak = 0
                text_result = await self._handle_text_response(message)
                if getattr(self, "_last_inline_tool_executed", False):
                    follow_up = await self._chat_with_ollama(
                        client,
                        available_model,
                        list(prepared_messages),
                        tools=active_tools,
                    )
                    if follow_up and "message" in follow_up:
                        text_result = await self._handle_text_response(follow_up["message"])
                return text_result

        except asyncio.TimeoutError:
            error_msg = (
                f"System 1 (Local Model) error: timed out after {int(self._ollama_timeout)}s. "
                f"Make sure Ollama is running with '{self.local_model}' available."
            )
            logger.warning(error_msg)
            return RouterResult(status="ok", content=f"[System 1 - Error]: {error_msg}")
        except Exception as e:
            error_msg = (
                f"System 1 (Local Model) error: {str(e)}. "
                f"Make sure Ollama is running with '{self.local_model}' available."
            )
            logger.error(error_msg, exc_info=True)
            return RouterResult(status="ok", content=f"[System 1 - Error]: {error_msg}")

    async def _route_to_gemini(
        self,
        messages: List[Dict[str, str]]
    ) -> RouterResult:
        """
        Route to System 2 (Gemini API) via the google-genai SDK.
        Tools are disabled on this path — Groq handles tool calling.
        """
        if genai_types is None:
            return RouterResult(status="ok", content="[System 2 - Error]: google-genai package not installed.")

        if not self.gemini_client:
            raise RuntimeError("Gemini not configured.")

        logger.info("Routing to System 2 (Gemini API / google-genai)")

        try:
            system_instruction = None
            contents: List[genai_types.Content] = []

            for msg in messages:
                role = msg["role"]
                text = msg["content"]
                if role == "system":
                    system_instruction = text
                elif role == "user":
                    contents.append(
                        genai_types.Content(role="user", parts=[genai_types.Part(text=text)])
                    )
                elif role in ("assistant", "model"):
                    contents.append(
                        genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
                    )

            config = genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
            ) if system_instruction else None

            response = await self.gemini_client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            if response and response.text:
                result = response.text.strip()
                logger.info(f"System 2 (Gemini) response received ({len(result)} chars)")
                return RouterResult(status="ok", content=result)

            logger.warning("Empty response from Gemini API")
            return RouterResult(status="ok", content="[System 2 - No Response]: Empty response from Gemini API")

        except Exception as e:
            logger.error(f"System 2 (Gemini) error: {str(e)}", exc_info=True)
            return RouterResult(status="ok", content=f"[System 2 - Error]: {str(e)[:200]}")

    async def synthesize_tool(
        self,
        gap_description: str,
        suggested_tool_name: str,
        user_query: str,
    ) -> Dict[str, str]:
        """
        Ask System 2 to draft a new Python tool for a capability gap.

        Returns a dict with keys: tool_name, description, code, schema_json,
        pytest_code, test_manifest_json.
        Raises RuntimeError if System 2 is unavailable or parsing fails.
        """
        logger.info(f"Synthesising tool for gap: {gap_description!r}")

        synthesis_prompt = f"""You are the Tool Synthesis Engine for a local autonomous AI agent.

The agent's System 1 (local model) identified a capability gap:
  Gap: {gap_description}
  Suggested tool name: {suggested_tool_name}
  Original user query that triggered the gap: {user_query}

Write a safe, minimal, async Python tool to fill this gap.

STRICT OUTPUT FORMAT — use exactly these delimiters, no extra text:

TOOL_NAME: {suggested_tool_name}
DESCRIPTION: <one-sentence description>
PYTHON_CODE:
```python
import asyncio  # add any stdlib imports needed
async def {suggested_tool_name}() -> str:
    \"\"\"Docstring here.\"\"\"
    try:
        # implementation
        return "result string"
    except Exception as e:
        return f"Error: {{e}}"
```
TOOL_SCHEMA:
```json
{{
  "name": "{suggested_tool_name}",
  "description": "<same one-sentence description>",
  "parameters": {{
    "type": "object",
    "properties": {{}},
    "required": []
  }}
}}
```
PYTEST_CODE:
```python
import pytest
from unittest.mock import AsyncMock, patch
from {suggested_tool_name} import {suggested_tool_name}


@pytest.mark.asyncio
async def test_{suggested_tool_name}_basic():
        result = await {suggested_tool_name}()
        assert isinstance(result, str)
```
TEST_MANIFEST:
```json
{{
    "version": "synthesis_contract_v2",
    "test_targets": ["{suggested_tool_name}"],
    "cases": [
        "happy_path"
    ],
    "notes": "Generated tests must be deterministic and side-effect free."
}}
```

Rules:
- Only use Python standard library (no pip installs).
- Function must be async.
- Must return a plain string.
- No file writes, no network calls to external services unless the gap explicitly requires it.
- Parameters dict may have properties if the tool needs arguments; otherwise keep empty.
- Tests must be deterministic and must not perform file writes, shell execution, or network access.
- Test imports are restricted to pytest, unittest.mock, and the generated tool module.
"""

        messages = [
            {"role": "system", "content": "You are a precise code generation engine. Follow the output format exactly."},
            {"role": "user", "content": synthesis_prompt},
        ]

        raw_result = await self.route_to_system_2(messages)
        parsed = self._parse_synthesis_output(raw_result.content, suggested_tool_name)
        # Fail closed: both tool and generated pytest code must pass AST checks.
        self._validate_tool_code_ast(parsed["code"], parsed["tool_name"])
        self._validate_pytest_code_ast(parsed["pytest_code"], parsed["tool_name"])
        return parsed

    async def repair_synthesized_tool(
        self,
        *,
        gap_description: str,
        suggested_tool_name: str,
        user_query: str,
        previous_tool_name: str,
        previous_code: str,
        previous_pytest_code: str,
        failure_summary: str,
        failure_trace: str,
    ) -> Dict[str, str]:
        """Ask System 2 to repair a synthesized tool that failed sandbox tests."""
        logger.info(
            "Repairing synthesized tool '%s' for gap: %r",
            previous_tool_name,
            gap_description,
        )

        repair_prompt = f"""You are the Tool Repair Engine for a local autonomous AI agent.

The previous synthesized candidate failed sandboxed pytest execution and must be repaired.

Capability context:
  Gap: {gap_description}
  Suggested tool name: {suggested_tool_name}
  Original user query: {user_query}

Failed candidate:
  Tool name: {previous_tool_name}

BROKEN_TOOL_CODE:
```python
{previous_code}
```

BROKEN_PYTEST_CODE:
```python
{previous_pytest_code}
```

Failure summary:
{failure_summary}

Failure trace:
```text
{failure_trace}
```

Return a corrected candidate using STRICT OUTPUT FORMAT (exact delimiters, no extra text):

TOOL_NAME: {suggested_tool_name}
DESCRIPTION: <one-sentence description>
PYTHON_CODE:
```python
import asyncio
async def {suggested_tool_name}() -> str:
    \"\"\"Docstring here.\"\"\"
    try:
        return "result string"
    except Exception as e:
        return f"Error: {{e}}"
```
TOOL_SCHEMA:
```json
{{
  "name": "{suggested_tool_name}",
  "description": "<same one-sentence description>",
  "parameters": {{
    "type": "object",
    "properties": {{}},
    "required": []
  }}
}}
```
PYTEST_CODE:
```python
import pytest
from unittest.mock import AsyncMock, patch
from {suggested_tool_name} import {suggested_tool_name}


@pytest.mark.asyncio
async def test_{suggested_tool_name}_basic():
    result = await {suggested_tool_name}()
    assert isinstance(result, str)
```
TEST_MANIFEST:
```json
{{
  "version": "synthesis_contract_v2",
  "test_targets": ["{suggested_tool_name}"],
  "cases": [
    "happy_path"
  ],
  "notes": "Repaired candidate after sandbox failure."
}}
```

Rules:
- Only use Python standard library (no pip installs).
- Function must be async and return plain string.
- No file writes, no shell execution, no external network calls unless gap explicitly requires it.
- Tests must be deterministic and side-effect free.
- Test imports are restricted to pytest, unittest.mock, and the generated tool module.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a strict code-repair engine. Follow the output format exactly.",
            },
            {"role": "user", "content": repair_prompt},
        ]

        raw_result = await self.route_to_system_2(messages)
        parsed = self._parse_synthesis_output(raw_result.content, suggested_tool_name)
        self._validate_tool_code_ast(parsed["code"], parsed["tool_name"])
        self._validate_pytest_code_ast(parsed["pytest_code"], parsed["tool_name"])
        return parsed

    @staticmethod
    def _parse_synthesis_output(raw: str, fallback_name: str) -> Dict[str, str]:
        """Extract v2 synthesis artifacts from the model output.

        Required fields:
          - tool_name
          - description
          - code
          - schema_json
          - pytest_code
          - test_manifest_json
        """
        import json as _json

        result = {
            "tool_name": fallback_name,
            "description": "",
            "code": "",
            "schema_json": "",
            "pytest_code": "",
            "test_manifest_json": "",
        }

        # tool_name
        m = re.search(r'TOOL_NAME:\s*(\S+)', raw)
        if m:
            result["tool_name"] = m.group(1).strip()

        # description
        m = re.search(r'DESCRIPTION:\s*(.+)', raw)
        if m:
            result["description"] = m.group(1).strip()

        # python code block
        m = re.search(r'PYTHON_CODE:\s*```python\s*(.*?)```', raw, re.DOTALL)
        if m:
            result["code"] = m.group(1).strip()

        # json schema block
        m = re.search(r'TOOL_SCHEMA:\s*```json\s*(.*?)```', raw, re.DOTALL)
        if m:
            schema_str = m.group(1).strip()
            try:
                schema_obj = _json.loads(schema_str)
                if not isinstance(schema_obj, dict):
                    raise _json.JSONDecodeError("schema must be a JSON object", schema_str, 0)
                result["schema_json"] = schema_str
            except _json.JSONDecodeError:
                raise RuntimeError("Tool synthesis failed: TOOL_SCHEMA must be valid JSON object.")

        # pytest code block
        m = re.search(r'PYTEST_CODE:\s*```python\s*(.*?)```', raw, re.DOTALL)
        if m:
            result["pytest_code"] = m.group(1).strip()

        # test manifest block
        m = re.search(r'TEST_MANIFEST:\s*```json\s*(.*?)```', raw, re.DOTALL)
        if m:
            manifest_str = m.group(1).strip()
            try:
                manifest_obj = _json.loads(manifest_str)
            except _json.JSONDecodeError as exc:
                raise RuntimeError("Tool synthesis failed: TEST_MANIFEST must be valid JSON object.") from exc
            if not isinstance(manifest_obj, dict):
                raise RuntimeError("Tool synthesis failed: TEST_MANIFEST must be a JSON object.")
            result["test_manifest_json"] = manifest_str

        if not result["description"]:
            raise RuntimeError("Tool synthesis failed: DESCRIPTION field missing.")
        if not result["code"]:
            raise RuntimeError(f"Tool synthesis failed: no Python code block found in output.\nRaw:\n{raw[:500]}")
        if not result["schema_json"]:
            raise RuntimeError("Tool synthesis failed: TOOL_SCHEMA block missing.")
        if not result["pytest_code"]:
            raise RuntimeError("Tool synthesis failed: PYTEST_CODE block missing.")
        if not result["test_manifest_json"]:
            raise RuntimeError("Tool synthesis failed: TEST_MANIFEST block missing.")

        return result

    @staticmethod
    def _check_blocked_import(module_name: str, tool_name: str) -> None:
        """Raise ValueError if *module_name*'s top-level package is blocked."""
        base = module_name.split(".")[0]
        if base in _BLOCKED_TOP_LEVEL_MODULES:
            raise ValueError(
                f"Synthesised tool '{tool_name}' imports blocked module '{module_name}'. "
                f"Blocked modules: {sorted(_BLOCKED_TOP_LEVEL_MODULES)}"
            )

    @staticmethod
    def _validate_ast_import_node(node: ast.AST, tool_name: str) -> bool:
        if isinstance(node, ast.Import):
            for alias in node.names:
                CognitiveRouter._check_blocked_import(alias.name, tool_name)
            return True
        if isinstance(node, ast.ImportFrom) and node.module:
            CognitiveRouter._check_blocked_import(node.module, tool_name)
            return True
        return False

    @staticmethod
    def _check_pytest_import(module_name: str, tool_name: str) -> None:
        """Validate generated pytest imports with an explicit allowlist."""
        base = module_name.split(".")[0]
        if base in _BLOCKED_TOP_LEVEL_MODULES:
            raise ValueError(
                f"Synthesised pytest for '{tool_name}' imports blocked module '{module_name}'. "
                f"Blocked modules: {sorted(_BLOCKED_TOP_LEVEL_MODULES)}"
            )

        if base == tool_name:
            return

        if module_name == "unittest.mock":
            return

        if base == "asyncio":
            return

        if base not in _ALLOWED_PYTEST_TOP_LEVEL_MODULES:
            raise ValueError(
                f"Synthesised pytest for '{tool_name}' imports non-allowlisted module '{module_name}'. "
                f"Allowed modules: pytest, unittest.mock, {tool_name}"
            )

    @staticmethod
    def _validate_pytest_import_node(node: ast.AST, tool_name: str) -> bool:
        if isinstance(node, ast.Import):
            for alias in node.names:
                CognitiveRouter._check_pytest_import(alias.name, tool_name)
            return True

        if isinstance(node, ast.ImportFrom):
            if node.module:
                CognitiveRouter._check_pytest_import(node.module, tool_name)
            for alias in node.names:
                if alias.name == "*":
                    raise ValueError(
                        f"Synthesised pytest for '{tool_name}' uses wildcard import, which is not allowed."
                    )
            return True

        return False

    @staticmethod
    def _validate_ast_attribute_node(node: ast.AST, tool_name: str) -> bool:
        if not isinstance(node, ast.Attribute):
            return False
        if node.attr in _BLOCKED_DUNDER_ATTRIBUTES:
            raise ValueError(
                f"Synthesised tool '{tool_name}' uses blocked dunder attribute access '{node.attr}'."
            )
        return True

    @staticmethod
    def _validate_ast_asyncio_call(node: ast.AST, tool_name: str) -> bool:
        """Block asyncio.create_subprocess_* specifically."""
        if not isinstance(node, ast.Attribute):
            return False
        if node.attr in _BLOCKED_ASYNCIO_CALLS:
            if isinstance(node.value, ast.Name) and node.value.id == "asyncio":
                raise ValueError(
                    f"Synthesised tool '{tool_name}' calls blocked asyncio function '{node.attr}'. "
                    "Subprocess creation is not permitted in synthesised tools."
                )
        return True

    @staticmethod
    def _validate_ast_call_node(node: ast.AST, tool_name: str) -> bool:
        if not isinstance(node, ast.Call):
            return False

        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _BLOCKED_TOOL_BUILTINS:
            raise ValueError(
                f"Synthesised tool '{tool_name}' calls blocked builtin '{fn.id}'."
            )
        return True

    @staticmethod
    def _validate_ast_string_constant_node(node: ast.AST, tool_name: str) -> bool:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            return False
        if node.value in _BLOCKED_TOP_LEVEL_MODULES:
            raise ValueError(
                f"Synthesised tool '{tool_name}' references blocked module string literal '{node.value}'."
            )
        return True

    @staticmethod
    def _validate_tool_code_ast(code: str, tool_name: str) -> None:
        """
        Parse the synthesised code as an AST and reject any import that
        touches a dangerous top-level module.

        Raises ``ValueError`` with a human-readable message if the code fails
        the sandbox check.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"Synthesised code for '{tool_name}' has a syntax error: {exc}") from exc

        for node in ast.walk(tree):
            if CognitiveRouter._validate_ast_import_node(node, tool_name):
                continue
            CognitiveRouter._validate_ast_asyncio_call(node, tool_name)
            if CognitiveRouter._validate_ast_attribute_node(node, tool_name):
                continue
            if CognitiveRouter._validate_ast_string_constant_node(node, tool_name):
                continue
            CognitiveRouter._validate_ast_call_node(node, tool_name)

    @staticmethod
    def _validate_pytest_code_ast(pytest_code: str, tool_name: str) -> None:
        """Parse and validate generated pytest code using strict sandbox checks."""
        try:
            tree = ast.parse(pytest_code)
        except SyntaxError as exc:
            raise ValueError(f"Synthesised pytest for '{tool_name}' has a syntax error: {exc}") from exc

        for node in ast.walk(tree):
            if CognitiveRouter._validate_pytest_import_node(node, tool_name):
                continue
            CognitiveRouter._validate_ast_asyncio_call(node, tool_name)
            if CognitiveRouter._validate_ast_attribute_node(node, tool_name):
                continue
            if CognitiveRouter._validate_ast_string_constant_node(node, tool_name):
                continue
            CognitiveRouter._validate_ast_call_node(node, tool_name)

    @staticmethod
    def _dynamic_tool_significant_tokens(code: str, tool_name: str) -> List[tokenize.TokenInfo]:
        try:
            tokens = list(tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline))
        except tokenize.TokenError as exc:
            raise ValueError(f"Synthesised code for '{tool_name}' could not be tokenized: {exc}") from exc
        return [token for token in tokens if token.type not in _DYNAMIC_TOOL_IGNORED_TOKEN_TYPES]

    @staticmethod
    def _string_token_literal_value(token: tokenize.TokenInfo) -> Optional[str]:
        if token.type != tokenize.STRING:
            return None
        try:
            literal_value = ast.literal_eval(token.string)
        except (SyntaxError, ValueError):
            return None
        return literal_value if isinstance(literal_value, str) else None

    @staticmethod
    def _validate_dynamic_tool_bare_token(token: tokenize.TokenInfo, tool_name: str) -> None:
        token_value = token.string
        literal_value = CognitiveRouter._string_token_literal_value(token)
        blocked_value = token_value if token_value in _BLOCKED_DYNAMIC_TOOL_TOKENS else literal_value
        if blocked_value in _BLOCKED_DYNAMIC_TOOL_TOKENS:
            raise ValueError(
                f"Synthesised tool '{tool_name}' contains blocked runtime token '{blocked_value}'."
            )

    @staticmethod
    def _validate_dynamic_tool_dotted_token(
        tokens: List[tokenize.TokenInfo],
        index: int,
        tool_name: str,
    ) -> None:
        if tokens[index].string != "sys" or index + 2 >= len(tokens):
            return
        dotted = "".join(next_token.string for next_token in tokens[index:index + 3])
        if dotted in _BLOCKED_DYNAMIC_TOOL_DOTTED_TOKENS:
            raise ValueError(
                f"Synthesised tool '{tool_name}' contains blocked runtime token '{dotted}'."
            )

    @staticmethod
    def _validate_dynamic_tool_token_scan(code: str, tool_name: str) -> None:
        significant_tokens = CognitiveRouter._dynamic_tool_significant_tokens(code, tool_name)
        for index, token in enumerate(significant_tokens):
            CognitiveRouter._validate_dynamic_tool_bare_token(token, tool_name)
            CognitiveRouter._validate_dynamic_tool_dotted_token(significant_tokens, index, tool_name)

    async def register_dynamic_tool(self, tool_name: str, code: str, schema_json: str) -> None:
        """Load synthesised Python code into the runtime via the SkillRegistry.

        The code is first validated through an AST sandbox that blocks imports
        of dangerous modules (os, sys, subprocess, etc.) before execution.
        """
        import json as _json

        # ── AST sandbox check ──────────────────────────────────────────────
        self._validate_tool_code_ast(code, tool_name)
        self._validate_dynamic_tool_token_scan(code, tool_name)

        try:
            schema = _json.loads(schema_json)
        except Exception as exc:
            raise RuntimeError(f"Invalid schema JSON for '{tool_name}': {exc}")

        worker = getattr(self, "_dynamic_tool_worker", None)
        if worker is None:
            worker = DynamicToolWorkerClient(call_timeout_seconds=TOOL_EXEC_TIMEOUT_SECONDS)
            self._dynamic_tool_worker = worker
            self.registry.set_dynamic_tool_worker(worker)
        response = await worker.register_tool(tool_name, code, schema)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or f"Failed to register dynamic tool '{tool_name}'"))

        self.registry.register_dynamic(tool_name, schema)
        logger.info(f"Dynamic tool '{tool_name}' registered via SkillRegistry")

    def get_system_1_available(self) -> bool:
        """System 1 (local Ollama) is always considered available."""
        return True

    def get_system_2_available(self) -> bool:
        """True if any System 2 provider (Ollama Cloud, Groq, or Gemini) is configured."""
        if self._system2_cooldown_until and time.time() < self._system2_cooldown_until:
            return False
        return (
            self.ollama_cloud_client is not None
            or self.groq_client is not None
            or self.gemini_client is not None
        )

    async def route_to_system_2(
        self,
        messages: List[Dict[str, Any]],
        allowed_tools: Optional[List[str]] = None
    ) -> RouterResult:
        """
        Route to System 2. Priority: Ollama Cloud > Groq > Gemini.
        Returns a :class:`RouterResult` — never raises for security intercepts.
        """
        if self._system2_cooldown_until and time.time() < self._system2_cooldown_until:
            msg = self._format_cooldown_message() or "Rate limited. Please try again later."
            return RouterResult(status="ok", content=f"[System 2 - Error]: {msg}")

        sanitized_messages = self._prepare_system_2_messages(messages)

        if self.ollama_cloud_client is not None:
            return await self._route_to_ollama_cloud(sanitized_messages, allowed_tools=allowed_tools)
        elif self.groq_client is not None:
            return await self._route_to_groq(sanitized_messages, allowed_tools=allowed_tools)
        elif self.gemini_client is not None:
            return await self._route_to_gemini(sanitized_messages)
        else:
            raise RuntimeError(
                "No System 2 provider configured. "
                "Set OLLAMA_CLOUD_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY+USE_GEMINI=True."
            )

    def _format_groq_tools(self, allowed_tools: Optional[List[str]]) -> List[dict]:
        groq_tools = []
        for tool in self.registry.get_schemas():
            if allowed_tools is not None and tool["name"] not in allowed_tools:
                continue
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            })
        return groq_tools

    async def _create_groq_completion(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None,
    ):
        tool_kwargs = {}
        groq_tools = self._format_groq_tools(allowed_tools)
        if groq_tools:
            tool_kwargs = {"tools": groq_tools, "tool_choice": "auto"}
        return await self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            **tool_kwargs,
            max_tokens=2048,
        )

    async def _create_groq_text_completion(self, messages: List[Dict[str, str]]):
        return await self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            max_tokens=2048,
        )

    @staticmethod
    def _choice_has_groq_tool_call(choice) -> bool:
        return bool(choice.finish_reason == "tool_calls" and choice.message.tool_calls)

    @staticmethod
    def _parse_groq_tool_call(tool_call) -> tuple[str, dict]:
        tool_name = tool_call.function.name
        arguments = CognitiveRouter._coerce_tool_arguments(tool_call.function.arguments)
        return tool_name, arguments

    @staticmethod
    def _build_pdf_summary_messages(tool_output: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a summarization engine. Summarize the provided PDF text faithfully."
            },
            {
                "role": "user",
                "content": (
                    "PDF text:\n"
                    f"{tool_output}\n\n"
                    "Summarize in a concise report. Start with a 3–5 sentence abstract, "
                    "then 5–8 bullet points. If the text ends with [TRUNCATED], mention "
                    "the summary is based on a truncated excerpt."
                ),
            },
        ]

    async def _summarize_pdf_with_groq(self, tool_output: str) -> RouterResult:
        if tool_output.strip().lower().startswith("error:"):
            return RouterResult(status="ok", content=tool_output.strip())

        final_response = await self._create_groq_text_completion(
            self._build_pdf_summary_messages(tool_output)
        )
        result = (final_response.choices[0].message.content or "").strip()
        logger.info(f"System 2 (Groq) PDF summary generated ({len(result)} chars)")
        return RouterResult(status="ok", content=result)

    @staticmethod
    def _build_groq_followup_messages(
        messages: List[Dict[str, Any]],
        executed_calls: List[dict],
    ) -> List[dict]:
        assistant_tool_calls = []
        tool_messages = []
        for idx, executed in enumerate(executed_calls, start=1):
            tool_call = executed["tool_call"]
            raw_tool_name = executed["raw_tool_name"]
            raw_arguments = executed["raw_arguments"]
            tool_output = executed["tool_output"]
            tool_call_id = getattr(tool_call, "id", "") or f"groq_call_{idx}"

            assistant_tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": raw_tool_name,
                    "arguments": raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments),
                },
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_output,
            })

        return list(messages) + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            },
            *tool_messages,
        ]

    async def _handle_groq_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
    ) -> RouterResult:
        if not tool_calls:
            return RouterResult(status="ok", content="[System 2 - Error]: Empty Groq tool call batch")

        parsed_calls = []
        for tool_call in tool_calls:
            raw_tool_name = getattr(getattr(tool_call, "function", None), "name", "")
            raw_arguments = getattr(getattr(tool_call, "function", None), "arguments", {})
            tool_name, arguments = self._parse_groq_tool_call(tool_call)
            tool_name = self._normalize_tool_name(tool_name)
            parsed_calls.append({
                "tool_call": tool_call,
                "raw_tool_name": raw_tool_name or tool_name,
                "raw_arguments": raw_arguments,
                "tool_name": tool_name,
                "arguments": arguments,
            })

        exec_results = await self._execute_tool_calls_with_partition(
            [(item["tool_name"], item["arguments"]) for item in parsed_calls]
        )

        executed_calls = []
        for item, exec_result in zip(parsed_calls, exec_results):
            if exec_result.status != "ok":
                return exec_result
            logger.info(
                "Groq tool '%s' executed: %s",
                item["tool_name"],
                exec_result.content[:100],
            )
            executed_calls.append({
                **item,
                "tool_output": exec_result.content,
            })

        if len(executed_calls) == 1 and executed_calls[0]["tool_name"] == "extract_pdf_text":
            return await self._summarize_pdf_with_groq(executed_calls[0]["tool_output"])

        followup_messages = self._build_groq_followup_messages(messages, executed_calls)
        final_response = await self._create_groq_text_completion(followup_messages)
        result = (final_response.choices[0].message.content or "").strip()
        logger.info(f"System 2 (Groq) response after tool call ({len(result)} chars)")
        return RouterResult(status="ok", content=result)

    async def _handle_groq_tool_call(self, messages: List[Dict[str, Any]], tool_call) -> RouterResult:
        return await self._handle_groq_tool_calls(messages, [tool_call])

    @staticmethod
    def _extract_last_user_message(messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @staticmethod
    def _extract_system_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [msg for msg in messages if msg.get("role") == "system"]

    @staticmethod
    def _is_groq_rate_limit_error(err_text: str) -> bool:
        lowered = err_text.lower()
        return (
            "rate limit" in lowered
            or "rate_limit" in lowered
            or "code: 429" in lowered
        )

    def _handle_groq_rate_limit(self, err_text: str) -> Optional[RouterResult]:
        if not self._is_groq_rate_limit_error(err_text):
            return None

        cooldown_seconds = 1800
        match = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", err_text)
        if match:
            cooldown_seconds = int(match.group(1)) * 60 + int(float(match.group(2)))
        self._system2_cooldown_until = max(
            float(getattr(self, "_system2_cooldown_until", 0.0) or 0.0),
            time.time() + cooldown_seconds,
        )
        # Persist cooldown so it survives a bot restart
        if self._persist_cooldown_cb is not None:
            import asyncio as _asyncio
            try:
                task = _asyncio.create_task(self._persist_cooldown_cb(self._system2_cooldown_until))
                self._cooldown_persist_task = task
            except RuntimeError:
                pass  # No running event loop (e.g., during tests)
        msg = self._format_cooldown_message() or "Rate limited. Please try again later."
        logger.warning(f"Groq rate limit hit: {msg}")
        return RouterResult(status="ok", content=f"[System 2 - Error]: {msg}")

    @staticmethod
    def _is_groq_tool_failure(err_text: str) -> bool:
        return "tool_use_failed" in err_text or "failed_generation" in err_text

    async def _retry_groq_after_tool_failure(
        self,
        messages: List[Dict[str, str]],
        err_text: str,
    ) -> Optional[RouterResult]:
        failed_gen = self._extract_failed_generation(err_text)
        parsed = self._extract_failed_generation_tool_call(failed_gen)
        if not parsed:
            return None

        tool_name, arguments = parsed
        logger.warning(f"Groq tool_use_failed; replaying tool call: {tool_name}")
        exec_result = await self._execute_tool(tool_name, arguments)
        if exec_result.status != "ok":
            return exec_result

        retry_messages = self._extract_system_messages(messages)
        retry_messages.append({
            "role": "user",
            "content": (
                f"User request: {self._extract_last_user_message(messages)}\n\n"
                f"Tool output:\n{exec_result.content}\n\n"
                "Answer using the tool output above. Do not call tools."
            )
        })
        final_response = await self._create_groq_text_completion(retry_messages)
        result = (final_response.choices[0].message.content or "").strip()
        logger.info("System 2 (Groq) response generated after tool_use_failed recovery")
        return RouterResult(status="ok", content=result)

    async def _fallback_groq_without_tools(self, messages: List[Dict[str, str]]) -> RouterResult:
        fallback_messages = self._extract_system_messages(messages)
        fallback_messages.append({
            "role": "user",
            "content": "Tool calling failed. Respond without tools and explain any limitations."
        })
        final_response = await self._create_groq_text_completion(fallback_messages)
        result = (final_response.choices[0].message.content or "").strip()
        logger.info("System 2 (Groq) fallback response generated without tools")
        return RouterResult(status="ok", content=result)

    # ── Ollama Cloud (System 2 primary) ──────────────────────────────────────

    def _format_ollama_cloud_tools(self, allowed_tools: Optional[List[str]]) -> List[dict]:
        """Build the tool list in the format expected by the Ollama chat API."""
        tools = []
        for schema in self.registry.get_schemas():
            if allowed_tools is not None and schema["name"] not in allowed_tools:
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            })
        return tools

    async def _route_to_ollama_cloud(
        self,
        messages: List[Dict[str, Any]],
        allowed_tools: Optional[List[str]] = None,
    ) -> RouterResult:
        """Route to Ollama Cloud (primary System 2) with full tool calling support."""
        logger.info(f"Routing to System 2 (Ollama Cloud/{self.system_2_model})")
        try:
            tools = self._format_ollama_cloud_tools(allowed_tools)
            kwargs: Dict[str, Any] = {"model": self.system_2_model, "messages": messages}
            if tools:
                kwargs["tools"] = tools

            resp = await self.ollama_cloud_client.chat(**kwargs)
            msg = resp.message if hasattr(resp, "message") else resp["message"]
            tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") else msg.get("tool_calls")

            if tool_calls:
                return await self._handle_ollama_cloud_tool_calls(messages, list(tool_calls))

            content = (msg.content if hasattr(msg, "content") else msg.get("content", "")) or ""
            logger.info(f"System 2 (Ollama Cloud) response received ({len(content)} chars)")
            return RouterResult(status="ok", content=content.strip())

        except Exception as e:
            err_text = str(e)
            logger.error(f"Ollama Cloud error: {e}", exc_info=True)
            # Fall back to Groq when available
            if self.groq_client is not None:
                logger.warning("Ollama Cloud failed, falling back to Groq")
                return await self._route_to_groq(messages, allowed_tools=allowed_tools)
            return RouterResult(status="ok", content=f"[System 2 - Error]: {err_text[:200]}")

    @staticmethod
    def _extract_ollama_tool_call_function(tool_call: Any) -> Any:
        if hasattr(tool_call, "function"):
            return tool_call.function
        if isinstance(tool_call, dict):
            return tool_call.get("function", {})
        return {}

    @staticmethod
    def _parse_ollama_cloud_tool_call(tool_call: Any) -> tuple[str, dict]:
        fn = CognitiveRouter._extract_ollama_tool_call_function(tool_call)
        tool_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
        raw_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", {})
        arguments = CognitiveRouter._coerce_tool_arguments(raw_args)
        return tool_name, arguments

    @staticmethod
    def _build_ollama_cloud_followup_messages(
        messages: List[Dict[str, Any]],
        executed_calls: List[dict],
    ) -> List[Dict[str, Any]]:
        assistant_tool_calls = []
        for item in executed_calls:
            assistant_tool_calls.append({
                "function": {
                    "name": item["raw_tool_name"],
                    "arguments": item["raw_arguments"],
                }
            })

        followup_messages: List[Dict[str, Any]] = list(messages) + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": assistant_tool_calls,
            }
        ]
        for item in executed_calls:
            followup_messages.append(
                {
                    "role": "tool",
                    "content": item["tool_output"],
                }
            )
        return followup_messages

    async def _handle_ollama_cloud_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Any],
    ) -> RouterResult:
        """Execute tool call batch requested by Ollama Cloud and feed results back."""
        if not tool_calls:
            return RouterResult(status="ok", content="[System 2 - Error]: Empty Ollama Cloud tool call batch")

        parsed_calls = []
        for tool_call in tool_calls:
            raw_tool_name, arguments = self._parse_ollama_cloud_tool_call(tool_call)
            tool_name = self._normalize_tool_name(raw_tool_name)
            raw_function = self._extract_ollama_tool_call_function(tool_call)
            raw_arguments = {}
            if hasattr(raw_function, "arguments"):
                raw_arguments = raw_function.arguments
            elif isinstance(raw_function, dict):
                raw_arguments = raw_function.get("arguments", {})
            parsed_calls.append(
                {
                    "raw_tool_name": raw_tool_name or tool_name,
                    "raw_arguments": raw_arguments,
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
            )

        exec_results = await self._execute_tool_calls_with_partition(
            [(item["tool_name"], item["arguments"]) for item in parsed_calls]
        )

        executed_calls = []
        for item, exec_result in zip(parsed_calls, exec_results):
            if exec_result.status != "ok":
                return exec_result

            logger.info(
                "Ollama Cloud tool '%s' executed: %s",
                item["tool_name"],
                exec_result.content[:100],
            )
            executed_calls.append({
                **item,
                "tool_output": exec_result.content,
            })

        if len(executed_calls) == 1 and executed_calls[0]["tool_name"] == "extract_pdf_text":
            return await self._summarize_pdf_with_ollama_cloud(executed_calls[0]["tool_output"])

        followup_messages = self._build_ollama_cloud_followup_messages(messages, executed_calls)
        resp = await self.ollama_cloud_client.chat(
            model=self.system_2_model,
            messages=followup_messages,
        )
        msg = resp.message if hasattr(resp, "message") else resp["message"]
        content = (msg.content if hasattr(msg, "content") else msg.get("content", "")) or ""
        logger.info(f"System 2 (Ollama Cloud) response after tool call ({len(content)} chars)")
        return RouterResult(status="ok", content=content.strip())

    async def _handle_ollama_cloud_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
    ) -> RouterResult:
        return await self._handle_ollama_cloud_tool_calls(messages, [tool_call])

    async def _summarize_pdf_with_ollama_cloud(self, tool_output: str) -> RouterResult:
        if tool_output.strip().lower().startswith("error:"):
            return RouterResult(status="ok", content=tool_output.strip())
        resp = await self.ollama_cloud_client.chat(
            model=self.system_2_model,
            messages=self._build_pdf_summary_messages(tool_output),
        )
        msg = resp.message if hasattr(resp, "message") else resp["message"]
        content = (msg.content if hasattr(msg, "content") else msg.get("content", "")) or ""
        logger.info(f"System 2 (Ollama Cloud) PDF summary generated ({len(content)} chars)")
        return RouterResult(status="ok", content=content.strip())

    # ── Groq (System 2 fallback) ──────────────────────────────────────────────

    async def _route_to_groq(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None
    ) -> RouterResult:
        """Route to Groq API (llama-3.3-70b or configured model) with full tool calling."""
        logger.info(f"Routing to System 2 (Groq/{self.groq_model})")
        if getattr(self, "_system2_cooldown_until", 0.0) and time.time() < self._system2_cooldown_until:
            msg = self._format_cooldown_message() or "Rate limited. Please try again later."
            return RouterResult(status="ok", content=f"[System 2 - Error]: {msg}")
        try:
            response = await self._create_groq_completion(messages, allowed_tools=allowed_tools)
            choice = response.choices[0]

            if self._choice_has_groq_tool_call(choice):
                return await self._handle_groq_tool_calls(messages, list(choice.message.tool_calls))

            result = (choice.message.content or "").strip()
            logger.info(f"System 2 (Groq) response received ({len(result)} chars)")
            return RouterResult(status="ok", content=result)

        except Exception as e:
            err_text = str(e)
            rate_limit_result = self._handle_groq_rate_limit(err_text)
            if rate_limit_result is not None:
                return rate_limit_result

            if self._is_groq_tool_failure(err_text):
                retry_result = await self._retry_groq_after_tool_failure(messages, err_text)
                if retry_result is not None:
                    return retry_result
                return await self._fallback_groq_without_tools(messages)

            logger.error(f"Groq error: {e}", exc_info=True)
            return RouterResult(status="ok", content=f"[System 2 - Error]: {str(e)[:200]}")
