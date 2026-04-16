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
import google.generativeai as genai
import ollama

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from src.tools.system_tools import (
    SYSTEM_TOOLS_SCHEMA, update_ledger, request_core_update,
    update_core_memory, search_archival_memory,
    query_highest_priority_task, spawn_new_objective, update_objective_status,
)

class RequiresMFAError(Exception):
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments
        super().__init__(f"MFA required for tool: {tool_name}")

class RequiresHITLError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
USE_GEMINI = os.getenv('USE_GEMINI', 'False').strip().lower() == 'true'

if GEMINI_API_KEY and USE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)


class CognitiveRouter:
    """
    Routes prompts to appropriate LLM systems.
    System 1: Local Ollama (fast, private)
    System 2: Groq (preferred if key set) or Gemini (fallback)
    """

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Executes the specified tool locally and returns the result.
        """
        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")

        try:
            if tool_name == "update_ledger":
                result = await update_ledger(
                    task_description=arguments.get("task_description", ""),
                    priority=arguments.get("priority", 5)
                )
                return result
            elif tool_name == "request_core_update":
                result = await request_core_update(
                    component=arguments.get("component", ""),
                    proposed_change=arguments.get("proposed_change", "")
                )
                return result
            elif tool_name == "update_core_memory":
                result = await update_core_memory(
                    key=arguments.get("key", ""),
                    value=arguments.get("value", "")
                )
                return result
            elif tool_name == "search_archival_memory":
                return await search_archival_memory(query=arguments.get("query", ""))
            elif tool_name == "query_highest_priority_task":
                return await query_highest_priority_task()
            elif tool_name == "spawn_new_objective":
                return await spawn_new_objective(
                    tier=arguments.get("tier", "Task"),
                    title=arguments.get("title", ""),
                    estimated_energy=arguments.get("estimated_energy", 10)
                )
            elif tool_name == "update_objective_status":
                return await update_objective_status(
                    task_id=arguments.get("task_id", 0),
                    new_status=arguments.get("new_status", "completed")
                )
            else:
                return f"Error: Unknown tool {tool_name}."
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error: Failed to execute tool due to [{str(e)}]."

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

        # Collapse leftover blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        logged = text[:120].replace('\n', ' ')
        logger.debug(f"sanitize_response output preview: {logged!r}")
        return text.strip()

    def __init__(self, model_name: str = "gemini-2.0-flash", local_model: str = "gemma4:e4b") -> None:        self.model_name = model_name
        self.local_model = local_model

        # System 2 provider selection: Groq takes priority over Gemini
        self.groq_client = None
        self.gemini_model = None

        if GROQ_AVAILABLE and GROQ_API_KEY:
            self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)
            self.groq_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            logger.info(f"System 2: Groq ({self.groq_model}) — fast free inference")
        elif GEMINI_API_KEY and USE_GEMINI:
            self.gemini_model = genai.GenerativeModel(model_name)
            logger.info(f"System 2: Gemini ({model_name})")
        else:
            logger.warning("No System 2 available. Set GROQ_API_KEY (free) or GEMINI_API_KEY + USE_GEMINI=True.")

        logger.info(f"CognitiveRouter initialized. System 1: {local_model}")

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

        client = None
        try:
            client = ollama.AsyncClient()

            # Format tools for Ollama
            ollama_tools = []
            for tool in SYSTEM_TOOLS_SCHEMA:
                if allowed_tools is not None and tool["name"] not in allowed_tools:
                    continue
                # Ollama format requires 'type': 'function' and 'function': {schema}
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
            try:
                response = await asyncio.wait_for(
                    client.chat(
                        model=available_model,
                        messages=messages
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
                            messages=messages
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
                                messages=messages
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
                    # Handle tool calls
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

                    # Pause, call execute tool
                    tool_result = await self._execute_tool(tool_name, arguments)

                    # Build new messages for follow-up call (avoid malformed assistant messages)
                    followup_messages = messages.copy()
                    followup_messages.append({
                        "role": "assistant",
                        "content": json.dumps({"tool_call": tool_name, "arguments": arguments})
                    })
                    followup_messages.append({
                        "role": "user",
                        "content": f"Tool result: {tool_result}"
                    })

                    # Call model again with the correct model name
                    final_response = await asyncio.wait_for(
                        client.chat(
                            model=available_model,
                            messages=followup_messages
                        ),
                        timeout=60.0
                    )

                    if final_response and 'message' in final_response and 'content' in final_response['message']:
                        result = final_response['message']['content'].strip()
                        if result:
                            logger.info(f"System 1 response received after tool call ({len(result)} chars)")
                            return result
                    return "Tool executed but model produced no response."

                # Standard text response
                elif 'content' in message:
                    result = message['content'].strip()
                    if not result:
                        logger.warning("System 1 returned empty response")
                        return "[System 1 - Error]: Empty response from model"
                    logger.info(f"System 1 response received ({len(result)} chars)")
                    logger.debug(f"System 1 response: {result[:200]}...")
                    return result

            error_msg = "Unexpected response format from Ollama"
            logger.warning(error_msg)
            return f"[System 1 - Error]: {error_msg}"

        except Exception as e:
            error_msg = f"System 1 (Local Model) error: {str(e)}. Make sure Ollama is running with a model like 'gemma4:e4b' or 'gemma2' available."
            logger.error(error_msg, exc_info=True)
            return f"[System 1 - Error]: {error_msg}"
        finally:
            # Properly close the client to avoid resource leak
            try:
                if client:
                    await client.close()
            except Exception as e:
                logger.debug(f"Error closing Ollama client: {e}")

    async def _route_to_gemini(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Route to System 2 (Gemini API) - Complex reasoning and analysis.

        Sends the messages to the Gemini API for advanced reasoning tasks.

        Args:
            messages: List of message dictionaries.

        Returns:
            str: The model's response from Gemini.

        Raises:
            RuntimeError: If GEMINI_API_KEY is not configured.
            Exception: If the API call fails.
        """
        if not self.gemini_model:
            error_msg = "Gemini not configured."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("Routing to System 2 (Gemini API)")

        try:
            system_instruction = None
            history = []

            # Extract system instruction
            if messages and messages[0].get("role") == "system":
                system_instruction = messages[0].get("content")
                # Gemini generative model supports system_instruction on init,
                # but we can also prepend it to the first user message or configure it if needed.
                # For `GenerativeModel`, we can set `system_instruction` parameter in `generate_content` or constructor.
                # Since we initialize it in __init__ without system prompt, we will use it here.
                # A common pattern is to just prepend it to the first user message if system_instruction is not configured globally.

            # Format messages for Gemini
            gemini_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    continue # handled above

                role = "user" if msg["role"] == "user" else "model"
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

            if system_instruction and gemini_messages and gemini_messages[0]["role"] == "user":
                 gemini_messages[0]["parts"][0]["text"] = f"System Instruction: {system_instruction}\n\n{gemini_messages[0]['parts'][0]['text']}"

            # NOTE: Tools disabled due to google-generativeai schema compatibility issues
            # gemini_tools = [{"function_declarations": SYSTEM_TOOLS_SCHEMA}]

            # Call Gemini API
            response = await self.gemini_model.generate_content_async(
                gemini_messages
            )

            # Handle tool calls
            if response and response.candidates and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts

                # Check if it's a function call
                for part in parts:
                    if "function_call" in part:
                        func_call = part.function_call
                        tool_name = func_call.name

                        # Convert protobuf mapping to dict
                        arguments = {k: v for k, v in func_call.args.items()}

                        logger.info(f"System 2 requested tool call: {tool_name}")

                        if tool_name == "request_core_update":
                            raise RequiresMFAError(tool_name, arguments)

                        if tool_name == "ask_admin_for_guidance":
                            context_summary = arguments.get("context_summary", "")
                            specific_question = arguments.get("specific_question", "")
                            raise RequiresHITLError(f"Guidance Needed: {context_summary}\nQuestion: {specific_question}")

                        # Execute tool
                        tool_result = await self._execute_tool(tool_name, arguments)

                        # Append the function call from the model
                        gemini_messages.append({
                            "role": "model",
                            "parts": [part]
                        })

                        # Append the function response
                        gemini_messages.append({
                            "role": "user",
                            "parts": [{
                                "function_response": {
                                    "name": tool_name,
                                    "response": {"result": tool_result}
                                }
                            }]
                        })

                        # Get final response
                        final_response = await self.gemini_model.generate_content_async(
                            gemini_messages
                        )

                        if final_response and final_response.text:
                            return final_response.text.strip()
                        return "Error extracting final response from Gemini."

            # Extract text from response (if no tool calls)
            if response and response.text:
                result = response.text.strip()
                logger.info(f"System 2 response received ({len(result)} chars)")
                logger.debug(f"System 2 response: {result[:100]}...")
                return result
            else:
                error_msg = "Empty response from Gemini API"
                logger.warning(error_msg)
                return f"[System 2 - No Response]: {error_msg}"

        except Exception as e:
            error_msg = f"System 2 error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def get_system_1_available(self) -> bool:
        """
        Check if System 1 (local model) is available.

        Returns:
            bool: True if System 1 is available (always true for placeholder).
        """
        return True

    def get_system_2_available(self) -> bool:
        """True if any System 2 provider (Groq or Gemini) is configured."""
        return self.groq_client is not None or self.gemini_model is not None

    async def route_to_system_2(self, messages: List[Dict[str, str]]) -> str:
        """
        Route to System 2. Uses Groq if available, falls back to Gemini.
        Groq is preferred: free tier, very fast, high quality (llama-3.3-70b).
        """
        if self.groq_client is not None:
            return await self._route_to_groq(messages)
        elif self.gemini_model is not None:
            return await self._route_to_gemini(messages)
        else:
            raise RuntimeError("No System 2 provider configured. Set GROQ_API_KEY or GEMINI_API_KEY+USE_GEMINI=True.")

    async def _route_to_groq(self, messages: List[Dict[str, str]]) -> str:
        """Route to Groq API (llama-3.3-70b or configured model)."""
        logger.info(f"Routing to System 2 (Groq/{self.groq_model})")
        try:
            # Groq uses OpenAI-compatible format — messages pass through as-is
            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                max_tokens=2048,
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"System 2 (Groq) response received ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"Groq error: {e}", exc_info=True)
            raise
