"""
LLM Router Module - Routes decisions to System 1 (Local) or System 2 (Cloud).

System 1: Local Ollama model (Gemma 4) — fast, private, always available.
System 2: Cloud LLM for complex reasoning. Provider priority:
  1. Groq (free, fast) — set GROQ_API_KEY
  2. Gemini (Google AI Studio free tier) — set GEMINI_API_KEY + USE_GEMINI=True
"""

import ast
import os
import logging
import json
import asyncio
import re
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
        text = re.sub(r'\[Output Draft\].*?(?=\[|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\[Internal Critique\].*?(?=\[|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\[Finalized Deliverable\]\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'---+\s*\n', '', text)  # horizontal rule separators

        # Collapse leftover blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        logged = text[:120].replace('\n', ' ')
        logger.debug(f"sanitize_response output preview: {logged!r}")
        return text.strip()

    def __init__(self, model_name: str = "gemini-2.0-flash", local_model: str = "gemma4:e4b") -> None:
        self.model_name = model_name
        self.local_model = local_model

        # SkillRegistry — single source of truth for all tool schemas and callables
        self.registry = SkillRegistry()

        # System 1: cached Ollama client — created once, reused across all calls
        self._ollama_client: Optional[ollama.AsyncClient] = None

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

        # Extract all top-level JSON objects, including those with nested braces,
        # by scanning for balanced brace pairs.
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

        for raw in blobs:
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            if not isinstance(obj, dict):
                continue

            # Accept keys like "tool_name", "name", "function_call", "tool_call"
            name_val = (
                obj.get("tool_name")
                or obj.get("name")
                or obj.get("function_call")
                or obj.get("tool_call")
            )
            if not isinstance(name_val, str) or not name_val.strip():
                continue

            # Extract arguments from common key variations
            raw_args = (
                obj.get("arguments")
                or obj.get("parameters")
                or obj.get("args")
                or {}
            )
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    raw_args = {}
            if not isinstance(raw_args, dict):
                # Arguments might be spread flat in the same object
                # (e.g. {"tool_name": "x", "gap_description": "...", ...})
                known_meta = {"tool_name", "name", "function_call", "tool_call",
                              "arguments", "parameters", "args"}
                raw_args = {k: v for k, v in obj.items() if k not in known_meta}

            logger.debug(f"_extract_inline_tool_call: found '{name_val}' in content text")
            return (name_val.strip(), raw_args)

        return None

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
            # Reuse cached client — avoids connection overhead on every call
            if self._ollama_client is None:
                self._ollama_client = ollama.AsyncClient()
            client = self._ollama_client

            # Format tools for Ollama from the SkillRegistry
            ollama_tools = []
            for tool in self.registry.get_schemas():
                if allowed_tools is not None and tool["name"] not in allowed_tools:
                    continue
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })

            # Use configured local model (default: gemma4:e4b, fallback: gemma2)
            available_model = self.local_model
            active_tools = ollama_tools or None
            try:
                response = await asyncio.wait_for(
                    client.chat(
                        model=available_model,
                        messages=messages,
                        tools=active_tools
                    ),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Ollama request timeout for model '{available_model}'. Trying gemma2...")
                available_model = 'gemma2'
                try:
                    response = await asyncio.wait_for(
                        client.chat(
                            model=available_model,
                            messages=messages,
                            tools=active_tools
                        ),
                        timeout=60.0
                    )
                except Exception as e2:
                    logger.warning(f"Model 'gemma2' also failed or timed out: {str(e2)}")
                    raise
            except Exception as e:
                if '404' in str(e) or 'not found' in str(e).lower():
                    logger.warning(f"Model '{available_model}' not found. Trying gemma2...")
                    available_model = 'gemma2'
                    try:
                        response = await asyncio.wait_for(
                            client.chat(
                                model=available_model,
                                messages=messages,
                                tools=active_tools
                            ),
                            timeout=60.0
                        )
                    except Exception as e2:
                        logger.warning(f"Model 'gemma2' also not found: {str(e2)}")
                        raise
                else:
                    raise

            # Check for tool calls
            if response and 'message' in response:
                message = response['message']

                if 'tool_calls' in message and message['tool_calls']:
                    # Robust extraction: Ollama returns ToolCall Pydantic objects that
                    # support both attribute access (.function) and dict-like access
                    # (['function']).  Some model/library combinations may omit the
                    # 'function' wrapper or serialize 'arguments' as a JSON string
                    # rather than a plain dict.  Handle all variants gracefully.
                    tool_name = ""
                    arguments: dict = {}
                    tc_parse_ok = False
                    try:
                        raw_tc = message['tool_calls'][0]
                        # Support both {'function': {...}} and bare {'name': ..., 'arguments': ...}
                        if hasattr(raw_tc, 'function'):
                            fn_data = raw_tc.function
                        elif isinstance(raw_tc, dict) and 'function' in raw_tc:
                            fn_data = raw_tc['function']
                        else:
                            fn_data = raw_tc  # bare format

                        tool_name = (
                            fn_data.get('name', '') if isinstance(fn_data, dict)
                            else getattr(fn_data, 'name', '')
                        )
                        raw_args = (
                            fn_data.get('arguments', {}) if isinstance(fn_data, dict)
                            else getattr(fn_data, 'arguments', {})
                        )
                        # Some models return arguments as a JSON string instead of a dict
                        if isinstance(raw_args, str):
                            try:
                                arguments = json.loads(raw_args)
                            except (json.JSONDecodeError, ValueError):
                                logger.warning(
                                    f"System 1 tool call arguments could not be JSON-parsed: {raw_args!r}"
                                )
                                arguments = {}
                        elif isinstance(raw_args, dict):
                            arguments = raw_args
                        else:
                            arguments = dict(raw_args) if raw_args else {}

                        if not tool_name:
                            raise ValueError(f"Tool call is missing the 'name' field: {raw_tc!r}")

                        tc_parse_ok = True

                    except Exception as tc_parse_err:
                        logger.error(
                            f"Failed to parse tool_calls from System 1 response: {tc_parse_err!r}",
                            exc_info=True,
                        )

                    if tc_parse_ok:
                        # Delegate to _execute_tool — returns RouterResult
                        exec_result = await self._execute_tool(tool_name, arguments)
                        if exec_result.status != "ok":
                            return exec_result  # Propagate MFA / HITL / capability-gap

                        # Build follow-up messages and get a final answer
                        followup_messages = messages.copy()
                        followup_messages.append({
                            "role": "assistant",
                            "content": json.dumps({"tool_call": tool_name, "arguments": arguments})
                        })
                        followup_messages.append({
                            "role": "user",
                            "content": f"Tool result: {exec_result.content}"
                        })

                        final_response = await asyncio.wait_for(
                            client.chat(
                                model=available_model,
                                messages=followup_messages,
                                tools=active_tools
                            ),
                            timeout=60.0
                        )

                        if final_response and 'message' in final_response and 'content' in final_response['message']:
                            result = final_response['message']['content'].strip()
                            if result:
                                logger.info(f"System 1 response received after tool call ({len(result)} chars)")
                                return RouterResult(status="ok", content=result)
                        return RouterResult(status="ok", content="Tool executed but model produced no response.")
                    # tc_parse_ok is False: fall through to content handling below

                # Standard text response.
                # Note: intentionally `if` (not `elif`) so that a tool_calls parse
                # failure above can fall through here and still return the content.
                # When a tool_calls parse succeeds the block above always returns
                # early, so double-processing cannot occur.
                if 'content' in message:
                    result = message['content'].strip() if message['content'] else ""
                    if not result:
                        logger.warning("System 1 returned empty response")
                        return RouterResult(status="ok", content="[System 1 - Error]: Empty response from model")

                    # Fallback: detect in-text tool calls that some models embed as JSON
                    # in the content field instead of using the native tool_calls array.
                    inline_tc = self._extract_inline_tool_call(result)
                    if inline_tc:
                        inline_name, inline_args = inline_tc
                        logger.info(
                            f"System 1 in-text tool call detected: {inline_name} — delegating to _execute_tool"
                        )
                        exec_result = await self._execute_tool(inline_name, inline_args)
                        if exec_result.status != "ok":
                            return exec_result  # Propagate MFA / HITL / capability-gap
                        return RouterResult(status="ok", content=exec_result.content)

                    logger.info(f"System 1 response received ({len(result)} chars)")
                    logger.debug(f"System 1 response: {result[:200]}...")
                    return RouterResult(status="ok", content=result)

            error_msg = "Unexpected response format from Ollama"
            logger.warning(error_msg)
            return RouterResult(status="ok", content=f"[System 1 - Error]: {error_msg}")

        except Exception as e:
            error_msg = f"System 1 (Local Model) error: {str(e)}. Make sure Ollama is running with a model like 'gemma4:e4b' or 'gemma2' available."
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
            raise

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
                    base = alias.name.split(".")[0]
                    if base in _BLOCKED_TOP_LEVEL_MODULES:
                        raise ValueError(
                            f"Synthesised tool '{tool_name}' imports blocked module '{alias.name}'. "
                            f"Blocked modules: {sorted(_BLOCKED_TOP_LEVEL_MODULES)}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.split(".")[0]
                    if base in _BLOCKED_TOP_LEVEL_MODULES:
                        raise ValueError(
                            f"Synthesised tool '{tool_name}' imports from blocked module '{node.module}'. "
                            f"Blocked modules: {sorted(_BLOCKED_TOP_LEVEL_MODULES)}"
                        )

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
        return self.groq_client is not None or self.gemini_client is not None

    async def route_to_system_2(self, messages: List[Dict[str, str]]) -> RouterResult:
        """
        Route to System 2. Uses Groq if available, falls back to Gemini.
        Groq is preferred: free tier, very fast, high quality (llama-3.3-70b).
        Returns a :class:`RouterResult` — never raises for security intercepts.
        """
        if self.groq_client is not None:
            return await self._route_to_groq(messages)
        elif self.gemini_client is not None:
            return await self._route_to_gemini(messages)
        else:
            raise RuntimeError("No System 2 provider configured. Set GROQ_API_KEY or GEMINI_API_KEY+USE_GEMINI=True.")

    async def _route_to_groq(self, messages: List[Dict[str, str]]) -> RouterResult:
        """Route to Groq API (llama-3.3-70b or configured model) with full tool calling."""
        logger.info(f"Routing to System 2 (Groq/{self.groq_model})")
        try:
            # Format tools in OpenAI-compatible format for Groq
            groq_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                }
                for tool in self.registry.get_schemas()
            ]

            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                tools=groq_tools,
                tool_choice="auto",
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
            logger.error(f"Groq error: {e}", exc_info=True)
            raise
