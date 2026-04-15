"""
LLM Router Module - Routes decisions to System 1 (Local) or System 2 (Gemini API).

This module implements the cognitive routing logic that directs prompts to either
the local language model (System 1) for fast, intuitive processing, or the Gemini
API (System 2) for complex, analytical reasoning.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import google.generativeai as genai
import ollama

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
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Route to System 1 (Local Model) - Fast, pattern-based responses.

        Uses local Gemma 4 model via Ollama.

        Args:
            messages: List of message dictionaries, formatted for chat.
                e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Returns:
            str: The model's response.
        """
        logger.info("Routing to System 1 (Local Model - gemma4)")

        try:
            client = ollama.AsyncClient()
            response = await client.chat(model='gemma4', messages=messages)

            if response and 'message' in response and 'content' in response['message']:
                result = response['message']['content'].strip()
                logger.info(f"System 1 response received ({len(result)} chars)")
                logger.debug(f"System 1 response: {result[:100]}...")
                return result
            else:
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

            # Call Gemini API
            response = await self.model.generate_content_async(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=512,
                )
            )

            # Extract text from response
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
