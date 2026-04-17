"""
LLM Router Module - Routes decisions to System 1 (Local) or System 2 (Cloud).

System 1: Local Ollama model (Gemma 4) — fast, private, always available.
System 2: Cloud LLM for complex reasoning. Provider priority:
  1. Groq (free, fast) — set GROQ_API_KEY
  2. Gemini (Google AI Studio free tier) — set GEMINI_API_KEY + USE_GEMINI=True
"""

import os
import logging
import json
import asyncio
import re
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
    """Raised when System 1 calls request_capability — signals a tool gap to the orchestrator."""
    def __init__(self, gap_description: str, suggested_tool_name: str):
        self.gap_description = gap_description
        self.suggested_tool_name = suggested_tool_name
        super().__init__(f"Capability gap: {gap_description}")

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

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call through the SkillRegistry.

        Security intercepts (MFA / HITL / capability synthesis) are raised
        here before the registry is consulted, keeping security logic centralised.
        """
        logger.info(f"Executing tool: {tool_name} with args: {list(arguments.keys())}")

        # ── Security intercepts ──────────────────────────────────────────────
        if tool_name == "request_core_update":
            raise RequiresMFAError(tool_name, arguments)

        if tool_name == "ask_admin_for_guidance":
            raise RequiresHITLError(
                f"Guidance Needed: {arguments.get('context_summary', '')}\n"
                f"Question: {arguments.get('specific_question', '')}"
            )

        if tool_name == "request_capability":
            raise RequiresCapabilitySynthesisError(
                gap_description=arguments.get("gap_description", "unspecified gap"),
                suggested_tool_name=arguments.get("suggested_tool_name", "new_tool"),
            )

        # ── Registry dispatch ────────────────────────────────────────────────
        return await self.registry.execute(tool_name, arguments)

    @staticmethod
    def _try_parse_text_tool_call(text: str) -> Optional[Dict[str, Any]]:
        """Detect if the LLM returned a text-based JSON tool call instead of a structured call.

        Local models (e.g. Gemma) sometimes emit the tool invocation as raw JSON text
        rather than a structured tool_calls object. This method detects that pattern.
        Returns a dict with 'tool_call' and 'arguments' keys, or None.
        """
        stripped = text.strip()
        if not (stripped.startswith('{') and stripped.endswith('}')):
            return None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and 'tool_call' in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return None

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
        # First try: entire response is a JSON tool call (handles nested objects)
        stripped = text.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                _parsed = json.loads(stripped)
                if isinstance(_parsed, dict) and any(
                    k in _parsed for k in ('tool_call', 'tool_name', 'function_call')
                ):
                    return ""  # entire response was a raw tool-call JSON blob
            except (json.JSONDecodeError, ValueError):
                pass
        # Second try: scan line-by-line for single-line JSON tool-call blobs
        _clean_lines = []
        for _line in text.split('\n'):
            _s = _line.strip()
            if (
                _s.startswith('{') and _s.endswith('}')
                and any(k in _s for k in ('"tool_call"', '"tool_name"', '"function_call"'))
            ):
                try:
                    _lp = json.loads(_s)
                    if isinstance(_lp, dict) and any(
                        k in _lp for k in ('tool_call', 'tool_name', 'function_call')
                    ):
                        continue  # skip this line
                except (json.JSONDecodeError, ValueError):
                    pass
            _clean_lines.append(_line)
        text = '\n'.join(_clean_lines)

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

        if GROQ_AVAILABLE and GROQ_API_KEY:
            self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)
            self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            logger.info(f"System 2: Groq ({self.groq_model}) — fast free inference")
        elif GEMINI_API_KEY and USE_GEMINI:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"System 2: Gemini ({model_name}) via google-genai SDK")
        else:
            logger.warning("No System 2 available. Set GROQ_API_KEY (free) or GEMINI_API_KEY + USE_GEMINI=True.")

        logger.info(f"CognitiveRouter initialized. System 1: {local_model}")

    async def close(self) -> None:
        """Release the cached Ollama client connection."""
        if self._ollama_client is not None:
            try:
                await self._ollama_client.close()
            except Exception as e:
                logger.debug(f"Error closing Ollama client: {e}")
            self._ollama_client = None

    async def route_to_system_1(
        self,
        messages: List[Dict[str, str]],
        allowed_tools: Optional[List[str]] = None
    ) -> str:
        """
        Route to System 1 (Local Model) - Fast, pattern-based responses.

        Uses local Gemma 4 model via Ollama.

        Args:
            messages: List of message dictionaries, formatted for chat.
                e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            allowed_tools: Optional list of tool names allowed for this call.

        Returns:
            str: The model's response.
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

            # ReAct loop — max 5 tool executions per turn to prevent infinite chaining
            MAX_TOOL_ITERATIONS = 5
            tool_iterations = 0
            current_messages = list(messages)
            current_response = response

            while True:
                if not (current_response and 'message' in current_response):
                    error_msg = "Unexpected response format from Ollama"
                    logger.warning(error_msg)
                    return f"[System 1 - Error]: {error_msg}"

                message = current_response['message']

                if 'tool_calls' in message and message['tool_calls']:
                    if tool_iterations >= MAX_TOOL_ITERATIONS:
                        logger.warning("ReAct loop: max tool iterations (%d) reached", MAX_TOOL_ITERATIONS)
                        return "Tool execution limit reached without a final response."

                    tool_call = message['tool_calls'][0]['function']
                    tool_name = tool_call['name']
                    arguments = tool_call.get('arguments', {})

                    # Intercept sensitive core updates
                    if tool_name == "request_core_update":
                        raise RequiresMFAError(tool_name, arguments)

                    # Intercept HITL requests
                    if tool_name == "ask_admin_for_guidance":
                        context_summary = arguments.get("context_summary", "")
                        specific_question = arguments.get("specific_question", "")
                        raise RequiresHITLError(f"Guidance Needed: {context_summary}\nQuestion: {specific_question}")

                    # Intercept capability gap signals — hand off to orchestrator
                    if tool_name == "request_capability":
                        raise RequiresCapabilitySynthesisError(
                            gap_description=arguments.get("gap_description", "unspecified gap"),
                            suggested_tool_name=arguments.get("suggested_tool_name", "new_tool"),
                        )

                    tool_result = await self._execute_tool(tool_name, arguments)
                    tool_iterations += 1

                    current_messages.append({
                        "role": "assistant",
                        "content": json.dumps({"tool_call": tool_name, "arguments": arguments})
                    })
                    current_messages.append({
                        "role": "user",
                        "content": f"Tool result: {tool_result}"
                    })

                    current_response = await asyncio.wait_for(
                        client.chat(
                            model=available_model,
                            messages=current_messages,
                            tools=active_tools
                        ),
                        timeout=60.0
                    )
                    continue

                elif 'content' in message:
                    result = message['content'].strip() if message['content'] else ''

                    # Detect text-based tool call (Gemma/local models sometimes emit JSON instead of structured calls)
                    if result:
                        text_call = self._try_parse_text_tool_call(result)
                        if text_call:
                            text_tool_name = text_call.get('tool_call', '')
                            text_arguments = text_call.get('arguments', {})
                            if text_tool_name and tool_iterations < MAX_TOOL_ITERATIONS:
                                logger.info(f"ReAct loop: detected text-based tool call for '{text_tool_name}'")
                                if text_tool_name == "request_core_update":
                                    raise RequiresMFAError(text_tool_name, text_arguments)
                                if text_tool_name == "ask_admin_for_guidance":
                                    raise RequiresHITLError(
                                        f"Guidance Needed: {text_arguments.get('context_summary', '')}\n"
                                        f"Question: {text_arguments.get('specific_question', '')}"
                                    )
                                if text_tool_name == "request_capability":
                                    raise RequiresCapabilitySynthesisError(
                                        gap_description=text_arguments.get('gap_description', 'unspecified gap'),
                                        suggested_tool_name=text_arguments.get('suggested_tool_name', 'new_tool'),
                                    )
                                tool_result = await self._execute_tool(text_tool_name, text_arguments)
                                tool_iterations += 1
                                current_messages.append({"role": "assistant", "content": result})
                                current_messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
                                current_response = await asyncio.wait_for(
                                    client.chat(
                                        model=available_model,
                                        messages=current_messages,
                                        tools=active_tools
                                    ),
                                    timeout=60.0
                                )
                                continue
                            elif not text_tool_name:
                                logger.warning("ReAct loop: text-based call missing tool_call name, skipping")

                    if not result and tool_iterations > 0:
                        # Silent crash fix: LLM returned empty after a tool call — force continuation
                        logger.info("ReAct loop: empty response after tool call, injecting continuation prompt")
                        if tool_iterations >= MAX_TOOL_ITERATIONS:
                            return "Tool execution limit reached without a final response."
                        current_messages.append({
                            "role": "user",
                            "content": "Tool execution successful. Please continue with the next logical step or provide your final response to the user."
                        })
                        tool_iterations += 1
                        current_response = await asyncio.wait_for(
                            client.chat(
                                model=available_model,
                                messages=current_messages,
                                tools=active_tools
                            ),
                            timeout=60.0
                        )
                        continue

                    if not result:
                        logger.warning("System 1 returned empty response")
                        return "[System 1 - Error]: Empty response from model"

                    logger.info(f"System 1 response received after {tool_iterations} tool call(s) ({len(result)} chars)")
                    logger.debug(f"System 1 response: {result[:200]}...")
                    return result

                else:
                    error_msg = "Unexpected response format from Ollama"
                    logger.warning(error_msg)
                    return f"[System 1 - Error]: {error_msg}"

        except Exception as e:
            error_msg = f"System 1 (Local Model) error: {str(e)}. Make sure Ollama is running with a model like 'gemma4:e4b' or 'gemma2' available."
            logger.error(error_msg, exc_info=True)
            return f"[System 1 - Error]: {error_msg}"

    async def _route_to_gemini(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
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
                return result

            logger.warning("Empty response from Gemini API")
            return "[System 2 - No Response]: Empty response from Gemini API"

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

        raw = await self.route_to_system_2(messages)
        return self._parse_synthesis_output(raw, suggested_tool_name)

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

    def register_dynamic_tool(self, tool_name: str, code: str, schema_json: str) -> None:
        """Load synthesised Python code into the runtime via the SkillRegistry."""
        import types
        import json as _json

        module = types.ModuleType(f"dynamic_tool_{tool_name}")
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

    async def route_to_system_2(self, messages: List[Dict[str, str]]) -> str:
        """
        Route to System 2. Uses Groq if available, falls back to Gemini.
        Groq is preferred: free tier, very fast, high quality (llama-3.3-70b).
        """
        if self.groq_client is not None:
            return await self._route_to_groq(messages)
        elif self.gemini_client is not None:
            return await self._route_to_gemini(messages)
        else:
            raise RuntimeError("No System 2 provider configured. Set GROQ_API_KEY or GEMINI_API_KEY+USE_GEMINI=True.")

    async def _route_to_groq(self, messages: List[Dict[str, str]]) -> str:
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

            # ReAct loop — max 5 tool executions per turn to prevent infinite chaining
            MAX_TOOL_ITERATIONS = 5
            tool_iterations = 0
            current_messages = list(messages)

            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=current_messages,
                tools=groq_tools,
                tool_choice="auto",
                max_tokens=2048,
            )

            while True:
                choice = response.choices[0]

                # Handle tool calls
                if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    if tool_iterations >= MAX_TOOL_ITERATIONS:
                        logger.warning("ReAct loop: max tool iterations (%d) reached", MAX_TOOL_ITERATIONS)
                        return "Tool execution limit reached without a final response."

                    tool_call = choice.message.tool_calls[0]
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}

                    logger.info(f"Groq requested tool call: {tool_name}")

                    # Security intercepts (same as System 1)
                    if tool_name == "request_core_update":
                        raise RequiresMFAError(tool_name, arguments)
                    if tool_name == "ask_admin_for_guidance":
                        raise RequiresHITLError(
                            f"Guidance Needed: {arguments.get('context_summary', '')}\n"
                            f"Question: {arguments.get('specific_question', '')}"
                        )
                    if tool_name == "request_capability":
                        raise RequiresCapabilitySynthesisError(
                            gap_description=arguments.get("gap_description", "unspecified gap"),
                            suggested_tool_name=arguments.get("suggested_tool_name", "new_tool"),
                        )

                    tool_result = await self._execute_tool(tool_name, arguments)
                    tool_iterations += 1
                    logger.info(f"Groq tool '{tool_name}' executed: {tool_result[:100]}")

                    # Append tool call + result in proper OpenAI format
                    current_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_call.function.arguments}
                        }]
                    })
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                    response = await self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=current_messages,
                        tools=groq_tools,
                        tool_choice="auto",
                        max_tokens=2048,
                    )
                    continue

                # Standard text response
                result = choice.message.content
                if result:
                    result = result.strip()

                if not result and tool_iterations > 0:
                    # Silent crash fix: LLM returned empty after a tool call — force continuation
                    logger.info("ReAct loop: empty response after tool call, injecting continuation prompt")
                    if tool_iterations >= MAX_TOOL_ITERATIONS:
                        return "Tool execution limit reached without a final response."
                    current_messages.append({
                        "role": "user",
                        "content": "Tool execution successful. Please continue with the next logical step or provide your final response to the user."
                    })
                    tool_iterations += 1
                    response = await self.groq_client.chat.completions.create(
                        model=self.groq_model,
                        messages=current_messages,
                        tools=groq_tools,
                        tool_choice="auto",
                        max_tokens=2048,
                    )
                    continue

                logger.info(f"System 2 (Groq) response received after {tool_iterations} tool call(s) ({len(result or '')} chars)")
                return result or ""

        except (RequiresMFAError, RequiresHITLError, RequiresCapabilitySynthesisError):
            raise
        except Exception as e:
            logger.error(f"Groq error: {e}", exc_info=True)
            raise
