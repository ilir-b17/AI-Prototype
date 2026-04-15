"""
LLM Router Module - Routes decisions to System 1 (Local) or System 2 (Gemini API).

This module implements the cognitive routing logic that directs prompts to either
the local language model (System 1) for fast, intuitive processing, or the Gemini
API (System 2) for complex, analytical reasoning.
"""

import os
import logging
from typing import Optional, Dict, Any
import google.generativeai as genai

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

    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        """
        Initialize the CognitiveRouter.

        Args:
            model_name: The Gemini model to use for System 2. Defaults to "gemini-1.5-pro".

        Raises:
            ValueError: If GEMINI_API_KEY is not set in environment.
        """
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set. System 2 will not be available.")

        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name) if GEMINI_API_KEY else None
        logger.info(f"CognitiveRouter initialized with model: {model_name}")

    def route_to_system_1(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Route to System 1 (Local Model) - Fast, pattern-based responses.

        This is a placeholder for the local Gemma 4 model. Currently returns
        a mock response. Will be replaced with actual local inference.

        Args:
            prompt: The user's prompt/question.
            context: Optional context dictionary (e.g., retrieved memories).

        Returns:
            str: The model's response.
        """
        logger.info("Routing to System 1 (Local Model)")

        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
            logger.debug(f"System 1 context: {context_str}")

        # Placeholder response for local model
        response = f"[System 1 - Local Model] Processing: {prompt[:50]}..."
        logger.debug(f"System 1 response: {response}")
        return response

    def route_to_system_2(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Route to System 2 (Gemini API) - Complex reasoning and analysis.

        Sends the prompt to the Gemini API for advanced reasoning tasks.

        Args:
            prompt: The user's prompt/question.
            context: Optional context dictionary (e.g., retrieved memories).

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
            # Construct the full prompt with context if provided
            full_prompt = prompt

            if context:
                # Include relevant context in the prompt
                context_str = "\n".join(
                    [f"- {k}: {v}" for k, v in context.items()]
                )
                full_prompt = f"{prompt}\n\nContext:\n{context_str}"

            logger.debug(f"System 2 prompt length: {len(full_prompt)}")

            # Call Gemini API
            response = self.model.generate_content(
                full_prompt,
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
