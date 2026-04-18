"""
LLM Router Module - Routes decisions to System 1 (Local) or System 2 (Cloud).

System 1: Local Ollama model (Gemma 4) — fast, private, always available.
System 2: Cloud LLM for complex reasoning. Provider priority:
  1. Groq (free, fast) — set GROQ_API_KEY
  2. Gemini (Google AI Studio free tier) — set GEMINI_API_KEY + USE_GEMINI=True
"""

import ast
import difflib
import os
import logging
import json
import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types as genai_types
import ollama

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from src.core.skill_manager import SkillRegistry

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
    """
    status: str  # "ok" | "mfa_required" | "hitl_required" | "capability_gap"
    content: str = ""
    # MFA fields
    mfa_tool_name: str = ""
    mfa_arguments: dict = field(default_factory=dict)
    # HITL fields
    hitl_message: str = ""
    # Capability-gap fields
    gap_description: str = ""
    suggested_tool_name: str = ""


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
    "asyncio",      # asyncio.create_subprocess_shell/exec can launch child processes
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

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
USE_GEMINI = os.getenv('USE_GEMINI', 'False').strip().lower() == 'true'


class CognitiveRouter:
    """
    Routes prompts to appropriate LLM systems.
    System 1: Local Ollama (fast, private)
    System 2: Groq (preferred if key set) or Gemini (fallback)
    """

    async def _execute_tool(self, tool_name: str, arguments: dict) -> RouterResult:
        """Dispatch a tool call through the SkillRegistry.

        Security intercepts (MFA / HITL / capability synthesis) are handled
        here, returning a structured :class:`RouterResult` rather than raising.
        """
        # Normalize hallucinated / malformed tool names
        tool_name = self._normalize_tool_name(tool_name)

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
            return RouterResult(
                status="capability_gap",
                gap_description=arguments.get("gap_description", "unspecified gap"),
                suggested_tool_name=arguments.get("suggested_tool_name", "new_tool"),
            )

        # ── Registry dispatch ────────────────────────────────────────────────
        content = await self.registry.execute(tool_name, arguments)
        return RouterResult(status="ok", content=content)

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

    def __init__(self, model_name: str = "gemini-2.0-flash", local_model: str = "gemma4:26b") -> None:
        self.model_name = model_name
        self.local_model = os.getenv("OLLAMA_MODEL", local_model)
        env_timeout = os.getenv("OLLAMA_TIMEOUT", "").strip()
        if env_timeout:
            self._ollama_timeout = float(env_timeout)
        else:
            self._ollama_timeout = 120.0 if "26b" in self.local_model else 60.0
        self._system2_cooldown_until = 0.0

        # SkillRegistry — single source of truth for all tool schemas and callables
        self.registry = SkillRegistry()

        # System 1: cached Ollama client — created once, reused across all calls
        self._ollama_client: Optional[ollama.AsyncClient] = None

        # Ollama runtime options — num_gpu=0 forces CPU-only, -1 means all layers on GPU
        _num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "-1"))
        self._ollama_options = {"num_gpu": _num_gpu} if _num_gpu != -1 else {}
        if _num_gpu == 0:
            logger.info("System 1: CPU-only mode (OLLAMA_NUM_GPU=0)")

        # System 2 provider selection: Groq takes priority over Gemini
        self.groq_client = None
        self.gemini_client = None
        self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

        if GROQ_AVAILABLE and GROQ_API_KEY:
            self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)
            logger.info(f"System 2: Groq ({self.groq_model}) — API key configured")
        elif not GROQ_AVAILABLE and GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is set but the 'groq' package is not installed. "
                "Run: pip install groq"
            )
        elif GEMINI_API_KEY and USE_GEMINI:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"System 2: Gemini ({model_name}) — API key configured")
        elif GEMINI_API_KEY and not USE_GEMINI:
            logger.warning(
                "GEMINI_API_KEY is set but USE_GEMINI is not 'True'. "
                "Set USE_GEMINI=True in .env to activate Gemini as System 2."
            )
        else:
            logger.warning(
                "No System 2 provider configured — capability synthesis will be unavailable. "
                "Set GROQ_API_KEY (free at console.groq.com) or GEMINI_API_KEY + USE_GEMINI=True."
            )

        logger.info(f"CognitiveRouter initialized. System 1: {local_model}")

    async def close(self) -> None:
        """Release the cached Ollama client connection."""
        if self._ollama_client is not None:
            try:
                await self._ollama_client.close()
            except Exception as e:
                logger.debug(f"Error closing Ollama client: {e}")
            self._ollama_client = None

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
        skill_names = self.registry.get_skill_names()
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
        return raw_name

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
        m = re.search(
            r"<function=([a-zA-Z0-9_]+)\s*\[?\]?\s*(\{.*?\})\s*</function>",
            text,
            re.DOTALL
        )
        if not m:
            return None
        name = m.group(1).strip()
        args_raw = m.group(2).strip()
        try:
            args = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
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

        for raw in CognitiveRouter._find_json_blobs(text):
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
            client.chat(model=model, messages=messages, tools=tools,
                        options=self._ollama_options),
            timeout=self._ollama_timeout,
        )
        return response, model

    @staticmethod
    def _parse_native_tool_call(message: dict) -> tuple:
        """Extract (tool_name, arguments, success) from an Ollama tool_calls message."""
        try:
            raw_tc = message["tool_calls"][0]
            fn_data = (
                raw_tc.function if hasattr(raw_tc, "function")
                else raw_tc.get("function", raw_tc) if isinstance(raw_tc, dict)
                else raw_tc
            )
            tool_name = (
                fn_data.get("name", "") if isinstance(fn_data, dict)
                else getattr(fn_data, "name", "")
            )
            raw_args = (
                fn_data.get("arguments", {}) if isinstance(fn_data, dict)
                else getattr(fn_data, "arguments", {})
            )
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except ValueError:
                    logger.warning(f"Tool call args could not be JSON-parsed: {raw_args!r}")
                    raw_args = {}
            elif not isinstance(raw_args, dict):
                raw_args = dict(raw_args) if raw_args else {}
            if not tool_name:
                raise ValueError(f"Tool call missing 'name': {raw_tc!r}")
            # Strip namespace prefix (e.g. "extract_pdf_text:extract_pdf_text" → "extract_pdf_text")
            if ":" in tool_name:
                tool_name = tool_name.rsplit(":", 1)[-1].strip()
            return tool_name, raw_args, True
        except Exception as exc:
            logger.error(f"Failed to parse tool_calls: {exc!r}", exc_info=True)
            return "", {}, False

    async def _run_tool_loop(self, client, model: str, first_message: dict,
                             messages: List[dict], active_tools) -> RouterResult:
        """Execute a chain of tool calls until the model returns plain text.

        Each iteration: parse tool call → execute → feed result back → re-prompt.
        Caps at 10 iterations; forces a text synthesis call if the limit is hit.
        """
        current_messages = list(messages)
        message = first_message
        iters = 0

        while "tool_calls" in message and message["tool_calls"] and iters < 10:
            iters += 1
            tool_name, arguments, ok = self._parse_native_tool_call(message)
            if not ok:
                break

            exec_result = await self._execute_tool(tool_name, arguments)
            if exec_result.status != "ok":
                return exec_result

            logger.info(f"System 1 tool call #{iters}: {tool_name} → {len(exec_result.content)} chars")
            current_messages = current_messages + [
                {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}]},
                {"role": "tool", "content": exec_result.content},
            ]

            if self._is_pdf_extraction_tool(tool_name):
                if exec_result.content.strip().lower().startswith("error:"):
                    return RouterResult(status="ok", content=exec_result.content.strip())
                summary_messages = [
                    {
                        "role": "system",
                        "content": "You are a summarization engine. Summarize the provided PDF text faithfully."
                    },
                    {
                        "role": "user",
                        "content": (
                            "PDF text:\n"
                            f"{exec_result.content}\n\n"
                            "Summarize in a concise report. Start with a 3–5 sentence abstract, "
                            "then 5–8 bullet points. If the text ends with [TRUNCATED], mention "
                            "the summary is based on a truncated excerpt."
                        ),
                    },
                ]
                try:
                    force_resp = await asyncio.wait_for(
                        client.chat(model=model, messages=summary_messages,
                                    options=self._ollama_options),
                        timeout=60.0,
                    )
                    if force_resp and "message" in force_resp:
                        content = (force_resp["message"].get("content") or "").strip()
                        if content:
                            logger.info("System 1 PDF summary generated")
                            return RouterResult(status="ok", content=content)
                except Exception as e:
                    logger.warning(f"PDF summary synthesis failed: {e}")
                return RouterResult(status="ok", content="Error: Failed to summarize extracted PDF text.")
            response = await asyncio.wait_for(
                client.chat(model=model, messages=current_messages, tools=active_tools,
                            options=self._ollama_options),
                timeout=self._ollama_timeout,
            )
            if not (response and "message" in response):
                logger.warning("Empty response during tool chain — stopping loop")
                break
            message = response["message"]

        content = (message.get("content") or "").strip()
        # If tools were executed but the model returned no text, force one synthesis
        # pass regardless of whether the final message still carries tool_calls.
        if not content and iters > 0:
            current_messages.append({
                "role": "user",
                "content": "Please summarise all the tool results above in a clear reply.",
            })
            try:
                force_resp = await asyncio.wait_for(
                    client.chat(model=model, messages=current_messages,
                                options=self._ollama_options),
                    timeout=self._ollama_timeout,
                )
                if force_resp and "message" in force_resp:
                    content = (force_resp["message"].get("content") or "").strip()
            except Exception as e:
                logger.warning(f"Force synthesis failed: {e}")

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
            inline_name, inline_args = inline_tc
            logger.info(f"System 1 in-text tool call detected: {inline_name}")
            exec_result = await self._execute_tool(inline_name, inline_args)
            if exec_result.status != "ok":
                return exec_result
            return RouterResult(status="ok", content=exec_result.content)

        logger.info(f"System 1 response received ({len(result)} chars)")
        return RouterResult(status="ok", content=result)

    async def route_to_system_1(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None
    ) -> RouterResult:
        """
        Route to System 1 (Local Model) - Fast, pattern-based responses.

        Returns a :class:`RouterResult` whose ``status`` is "ok" on success,
        or a non-ok status when a security intercept fires.
        """
        logger.info(f"Routing to System 1 (Local Model - {self.local_model})")
        try:
            client = self._get_or_create_ollama_client()
            active_tools = self._format_ollama_tools(allowed_tools) or None
            response, available_model = await self._call_ollama_with_model_fallback(
                client, messages, active_tools
            )

            if not (response and "message" in response):
                return RouterResult(status="ok", content="[System 1 - Error]: Unexpected response format from Ollama")

            message = response["message"]
            if "tool_calls" in message and message["tool_calls"]:
                return await self._run_tool_loop(client, available_model, message, list(messages), active_tools)

            return await self._handle_text_response(message)

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

        Returns a dict with keys: tool_name, description, code, schema_json.
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

Rules:
- Only use Python standard library (no pip installs).
- Function must be async.
- Must return a plain string.
- No file writes, no network calls to external services unless the gap explicitly requires it.
- Parameters dict may have properties if the tool needs arguments; otherwise keep empty.
"""

        messages = [
            {"role": "system", "content": "You are a precise code generation engine. Follow the output format exactly."},
            {"role": "user", "content": synthesis_prompt},
        ]

        raw_result = await self.route_to_system_2(messages)
        return self._parse_synthesis_output(raw_result.content, suggested_tool_name)

    @staticmethod
    def _parse_synthesis_output(raw: str, fallback_name: str) -> Dict[str, str]:
        """Extract tool_name, description, code, and schema_json from synthesis output."""
        import json as _json

        result = {"tool_name": fallback_name, "description": "", "code": "", "schema_json": ""}

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
                _json.loads(schema_str)  # validate
                result["schema_json"] = schema_str
            except _json.JSONDecodeError:
                logger.warning("Synthesis: schema JSON invalid — using minimal fallback")
                result["schema_json"] = _json.dumps({
                    "name": result["tool_name"],
                    "description": result["description"],
                    "parameters": {"type": "object", "properties": {}, "required": []}
                })

        if not result["code"]:
            raise RuntimeError(f"Tool synthesis failed: no Python code block found in output.\nRaw:\n{raw[:500]}")

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
            if isinstance(node, ast.Import):
                for alias in node.names:
                    CognitiveRouter._check_blocked_import(alias.name, tool_name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                CognitiveRouter._check_blocked_import(node.module, tool_name)

    def register_dynamic_tool(self, tool_name: str, code: str, schema_json: str) -> None:
        """Load synthesised Python code into the runtime via the SkillRegistry.

        The code is first validated through an AST sandbox that blocks imports
        of dangerous modules (os, sys, subprocess, etc.) before execution.
        """
        import types
        import json as _json

        # ── AST sandbox check ──────────────────────────────────────────────
        self._validate_tool_code_ast(code, tool_name)

        module = types.ModuleType(f"dynamic_tool_{tool_name}")
        # Restrict built-ins to a safe subset so that exec() cannot reach
        # open(), eval(), __import__(), exec(), compile(), etc. even if the
        # AST check is somehow bypassed.
        _safe_builtins = {
            "None": None, "True": True, "False": False,
            "abs": abs, "all": all, "any": any, "bool": bool,
            "bytes": bytes, "chr": chr, "dict": dict, "dir": dir,
            "divmod": divmod, "enumerate": enumerate, "filter": filter,
            "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map,
            "max": max, "min": min, "next": next, "object": object,
            "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round,
            "set": set, "slice": slice, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "type": type, "vars": vars,
            "zip": zip,
        }
        module.__dict__["__builtins__"] = _safe_builtins
        exec(compile(code, f"<dynamic:{tool_name}>", "exec"), module.__dict__)

        fn = getattr(module, tool_name, None)
        if fn is None:
            raise RuntimeError(f"Synthesised code does not define function '{tool_name}'")

        try:
            schema = _json.loads(schema_json)
        except Exception as exc:
            raise RuntimeError(f"Invalid schema JSON for '{tool_name}': {exc}")

        self.registry.register_dynamic(tool_name, fn, schema)
        logger.info(f"Dynamic tool '{tool_name}' registered via SkillRegistry")

    def get_system_1_available(self) -> bool:
        """System 1 (local Ollama) is always considered available."""
        return True

    def get_system_2_available(self) -> bool:
        """True if any System 2 provider (Groq or Gemini) is configured."""
        if self._system2_cooldown_until and time.time() < self._system2_cooldown_until:
            return False
        return self.groq_client is not None or self.gemini_client is not None

    async def route_to_system_2(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None
    ) -> RouterResult:
        """
        Route to System 2. Uses Groq if available, falls back to Gemini.
        Groq is preferred: free tier, very fast, high quality (llama-3.3-70b).
        Returns a :class:`RouterResult` — never raises for security intercepts.
        """
        if self._system2_cooldown_until and time.time() < self._system2_cooldown_until:
            msg = self._format_cooldown_message() or "Rate limited. Please try again later."
            return RouterResult(status="ok", content=f"[System 2 - Error]: {msg}")

        if self.groq_client is not None:
            return await self._route_to_groq(messages, allowed_tools=allowed_tools)
        elif self.gemini_client is not None:
            return await self._route_to_gemini(messages)
        else:
            raise RuntimeError("No System 2 provider configured. Set GROQ_API_KEY or GEMINI_API_KEY+USE_GEMINI=True.")

    async def _route_to_groq(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None
    ) -> RouterResult:
        """Route to Groq API (llama-3.3-70b or configured model) with full tool calling."""
        logger.info(f"Routing to System 2 (Groq/{self.groq_model})")
        try:
            # Format tools in OpenAI-compatible format for Groq
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

            tool_kwargs = {}
            if groq_tools:
                tool_kwargs = {"tools": groq_tools, "tool_choice": "auto"}

            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                **tool_kwargs,
                max_tokens=2048,
            )

            choice = response.choices[0]

            # Handle tool calls
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                logger.info(f"Groq requested tool call: {tool_name}")

                # Delegate to _execute_tool — returns RouterResult
                exec_result = await self._execute_tool(tool_name, arguments)
                if exec_result.status != "ok":
                    return exec_result  # Propagate MFA / HITL / capability-gap

                logger.info(f"Groq tool '{tool_name}' executed: {exec_result.content[:100]}")

                if tool_name == "extract_pdf_text":
                    if exec_result.content.strip().lower().startswith("error:"):
                        return RouterResult(status="ok", content=exec_result.content.strip())
                    summary_messages = [
                        {
                            "role": "system",
                            "content": "You are a summarization engine. Summarize the provided PDF text faithfully."
                        },
                        {
                            "role": "user",
                            "content": (
                                "PDF text:\n"
                                f"{exec_result.content}\n\n"
                                "Summarize in a concise report. Start with a 3–5 sentence abstract, "
                                "then 5–8 bullet points. If the text ends with [TRUNCATED], mention "
                                "the summary is based on a truncated excerpt."
                            ),
                        },
                    ]
                    final_response = await self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=summary_messages,
                        max_tokens=2048,
                    )
                    result = final_response.choices[0].message.content.strip()
                    logger.info(f"System 2 (Groq) PDF summary generated ({len(result)} chars)")
                    return RouterResult(status="ok", content=result)

                # Follow-up in proper OpenAI tool-result format
                followup_messages = list(messages) + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_call.function.arguments}
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": exec_result.content
                    }
                ]
                if tool_name == "extract_pdf_text":
                    followup_messages.append({
                        "role": "user",
                        "content": (
                            "Summarize the extracted PDF text above in a concise report. "
                            "Start with a 3–5 sentence abstract, then 5–8 bullet points. "
                            "If the text ends with [TRUNCATED], mention that the summary is based on a truncated excerpt."
                        ),
                    })

                final_response = await self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=followup_messages,
                    max_tokens=2048,
                )
                result = final_response.choices[0].message.content.strip()
                logger.info(f"System 2 (Groq) response after tool call ({len(result)} chars)")
                return RouterResult(status="ok", content=result)

            # Standard text response
            result = choice.message.content.strip()
            logger.info(f"System 2 (Groq) response received ({len(result)} chars)")
            return RouterResult(status="ok", content=result)

        except Exception as e:
            err_text = str(e)
            if "rate limit" in err_text.lower() or "rate_limit" in err_text.lower() or "code: 429" in err_text.lower():
                cooldown_seconds = 1800
                m = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", err_text)
                if m:
                    cooldown_seconds = int(m.group(1)) * 60 + int(float(m.group(2)))
                self._system2_cooldown_until = time.time() + cooldown_seconds
                msg = self._format_cooldown_message() or "Rate limited. Please try again later."
                logger.warning(f"Groq rate limit hit: {msg}")
                return RouterResult(status="ok", content=f"[System 2 - Error]: {msg}")

            if "tool_use_failed" in err_text or "failed_generation" in err_text:
                failed_gen = self._extract_failed_generation(err_text)
                parsed = self._extract_failed_generation_tool_call(failed_gen)
                if parsed:
                    tool_name, arguments = parsed
                    logger.warning(f"Groq tool_use_failed; replaying tool call: {tool_name}")
                    exec_result = await self._execute_tool(tool_name, arguments)
                    if exec_result.status != "ok":
                        return exec_result

                    last_user = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            last_user = msg.get("content", "")
                            break

                    retry_messages = []
                    for msg in messages:
                        if msg.get("role") == "system":
                            retry_messages.append(msg)
                    retry_messages.append({
                        "role": "user",
                        "content": (
                            f"User request: {last_user}\n\n"
                            f"Tool output:\n{exec_result.content}\n\n"
                            "Answer using the tool output above. Do not call tools."
                        )
                    })
                    final_response = await self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=retry_messages,
                        max_tokens=2048,
                    )
                    result = final_response.choices[0].message.content.strip()
                    logger.info("System 2 (Groq) response generated after tool_use_failed recovery")
                    return RouterResult(status="ok", content=result)

                # Fallback: respond without tools
                fallback_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        fallback_messages.append(msg)
                fallback_messages.append({
                    "role": "user",
                    "content": "Tool calling failed. Respond without tools and explain any limitations."
                })
                final_response = await self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=fallback_messages,
                    max_tokens=2048,
                )
                result = final_response.choices[0].message.content.strip()
                logger.info("System 2 (Groq) fallback response generated without tools")
                return RouterResult(status="ok", content=result)

            logger.error(f"Groq error: {e}", exc_info=True)
            return RouterResult(status="ok", content=f"[System 2 - Error]: {str(e)[:200]}")
