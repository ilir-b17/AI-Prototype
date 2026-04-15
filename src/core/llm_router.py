"""
LLM Router Module - Routes decisions to System 1 (Local) or System 2 (Gemini API).

This module implements the cognitive routing logic that directs prompts to either
the local language model (System 1) for fast, intuitive processing, or the Gemini
API (System 2) for complex, analytical reasoning.
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List
import google.generativeai as genai
import ollama

from src.tools.system_tools import SYSTEM_TOOLS_SCHEMA, update_ledger, request_core_update, update_core_memory, search_archival_memory

class RequiresMFAError(Exception):
    """Exception raised when a tool call requires MFA."""
    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments
        super().__init__(f"MFA required for tool: {tool_name}")

class RequiresHITLError(Exception):
    """Exception raised when the orchestrator needs to pause and ask the Admin for guidance."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

logger = logging.getLogger(__name__)

# Load Gemini API key from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class CognitiveRouter:
    """
    Routes prompts to appropriate LLM systems based on complexity and requirements.

    System 1 (Local): Fast, pattern-matching based responses
    System 2 (Gemini): Complex reasoning, multi-step analysis, creative tasks
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
                result = await search_archival_memory(
                    query=arguments.get("query", "")
                )
                return result
            else:
                return f"Error: Unknown tool {tool_name}."
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return f"Error: Failed to execute tool due to [{str(e)}]."

    def __init__(self, model_name: str = "gemini-3.1-pro-preview") -> None:
        """
        Initialize the CognitiveRouter.

        Args:
            model_name: The Gemini model to use for System 2. Defaults to "gemini-3.1-pro-preview".

        Raises:
            ValueError: If GEMINI_API_KEY is not set in environment.
        """
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set. System 2 will not be available.")

        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name) if GEMINI_API_KEY else None
        logger.info(f"CognitiveRouter initialized with model: {model_name}")

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
        logger.info("Routing to System 1 (Local Model - gemma4)")

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

            response = await client.chat(
                model='gemma4',
                messages=messages,
                tools=ollama_tools if ollama_tools else None
            )

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

                    # Append result back to messages to get final response
                    messages.append(message) # Append assistant's tool call
                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "name": tool_name
                    })

                    # Call model again
                    final_response = await client.chat(
                        model='gemma4',
                        messages=messages,
                        tools=ollama_tools
                    )

                    if final_response and 'message' in final_response and 'content' in final_response['message']:
                        return final_response['message']['content'].strip()
                    return "Error parsing final response after tool call."

                # Standard text response
                elif 'content' in message:
                    result = message['content'].strip()
                    logger.info(f"System 1 response received ({len(result)} chars)")
                    logger.debug(f"System 1 response: {result[:100]}...")
                    return result

            error_msg = "Unexpected response format from Ollama"
            logger.warning(error_msg)
            return f"[System 1 - Error]: {error_msg}"

        except Exception as e:
            error_msg = f"System 1 (Ollama) error: {str(e)}. Is the local server running at localhost:11434?"
            logger.error(error_msg, exc_info=True)
            return f"[System 1 - Error]: {error_msg}"

    async def route_to_system_2(
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
        if not self.model:
            error_msg = "GEMINI_API_KEY not set. Cannot route to System 2."
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

            # Format tools for Gemini API
            gemini_tools = [{"function_declarations": SYSTEM_TOOLS_SCHEMA}]

            # Call Gemini API
            response = await self.model.generate_content_async(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=512,
                ),
                tools=gemini_tools
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
                        final_response = await self.model.generate_content_async(
                            gemini_messages,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.7,
                                top_p=0.95,
                                max_output_tokens=512,
                            ),
                            tools=gemini_tools
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
        """
        Check if System 2 (Gemini API) is available.

        Returns:
            bool: True if GEMINI_API_KEY is configured.
        """
        return self.model is not None
