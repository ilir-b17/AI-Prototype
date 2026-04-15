"""
Executive Agent Module - The Prefrontal Cortex.

This module implements the central routing engine that processes user messages,
retrieves context from memory, and routes decisions to the appropriate cognitive
system (System 1 or System 2).
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from src.memory.vector_db import VectorMemory
from src.memory.ledger_db import LedgerMemory, LogLevel
from src.core.llm_router import CognitiveRouter

logger = logging.getLogger(__name__)


class ExecutiveAgent:
    """
    Executive Agent - The central decision-making and routing engine (Prefrontal Cortex).

    This agent processes incoming messages by:
    1. Logging the message to the ledger
    2. Querying memory for context
    3. Routing to the appropriate LLM system based on complexity
    4. Storing the response for future reference
    """

    def __init__(
        self,
        vector_db_path: str = "data/chroma_storage",
        ledger_db_path: str = "data/ledger.db",
        gemini_model: str = "gemini-3.1-pro-preview"
    ) -> None:
        """
        Initialize the ExecutiveAgent with all necessary subsystems.

        Args:
            vector_db_path: Path to the vector database storage.
            ledger_db_path: Path to the ledger database file.
            gemini_model: The Gemini model to use for System 2.

        Raises:
            Exception: If memory systems fail to initialize.
        """
        logger.info("Initializing ExecutiveAgent (Prefrontal Cortex)")

        try:
            # Initialize memory systems
            self.vector_memory = VectorMemory(persist_dir=vector_db_path)
            self.ledger_memory = LedgerMemory(db_path=ledger_db_path)

            # Initialize cognitive router
            self.cognitive_router = CognitiveRouter(model_name=gemini_model)

            # Load Charter
            self.charter_text = self._load_charter()

            logger.info("ExecutiveAgent initialized successfully")
            self.ledger_memory.log_event(
                LogLevel.INFO,
                "ExecutiveAgent initialized",
                {"subsystems": ["VectorMemory", "LedgerMemory", "CognitiveRouter", "Charter"]}
            )

        except Exception as e:
            logger.error(f"Failed to initialize ExecutiveAgent: {e}", exc_info=True)
            raise

    def _load_charter(self, filepath: str = "charter.md") -> str:
        """
        Loads the core identity charter.
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    charter = f.read().strip()
                logger.info(f"Loaded charter from {filepath}")
                return charter
            else:
                logger.critical(f"Charter file '{filepath}' not found! Using fallback directive.")
                return "Core Directive: You are a rational, autonomous AI. You must prioritize logic, log your actions, and do no harm."
        except Exception as e:
            logger.critical(f"Failed to read charter file '{filepath}': {e}. Using fallback directive.")
            return "Core Directive: You are a rational, autonomous AI. You must prioritize logic, log your actions, and do no harm."

    def _retrieve_context(self, user_message: str, n_results: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector memory.

        Args:
            user_message: The user's message to use as a query.
            n_results: Number of context snippets to retrieve. Defaults to 2.

        Returns:
            List of relevant memory entries with metadata.
        """
        try:
            memories = self.vector_memory.query_memory(
                query_text=user_message,
                n_results=n_results
            )
            logger.debug(f"Retrieved {len(memories)} relevant memories")
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []

    def _construct_master_prompt(
        self,
        user_message: str,
        context_memories: List[Dict[str, Any]]
    ) -> str:
        """
        Construct a comprehensive prompt that includes user message and context.

        Args:
            user_message: The original user message.
            context_memories: List of retrieved memories with context.

        Returns:
            str: The full prompt with context included.
        """
        prompt = f"User Query: {user_message}\n"

        if context_memories:
            prompt += "\n--- Relevant Context from Memory ---\n"
            for i, memory in enumerate(context_memories, 1):
                prompt += f"{i}. {memory['document']}\n"
            prompt += "--- End Context ---\n"

        return prompt

    def _route_decision(self, user_message: str) -> str:
        """
        Determine which cognitive system to route the message to.

        Basic routing logic:
        - If message contains "deep" or "code": Route to System 2 (Gemini - complex reasoning)
        - Otherwise: Route to System 1 (Local - fast responses)

        Args:
            user_message: The user's message.

        Returns:
            str: Either "system_1" or "system_2".
        """
        routing_keywords = ["deep", "code"]
        message_lower = user_message.lower()

        for keyword in routing_keywords:
            if keyword in message_lower:
                logger.debug(f"Routing to System 2 (keyword: '{keyword}' found)")
                return "system_2"

        logger.debug("Routing to System 1 (default)")
        return "system_1"

    async def process_message(self, user_message: str) -> str:
        """
        Process a user message through the executive pipeline.

        Flow:
        1. Log the incoming message to the ledger
        2. Query vector memory for relevant context
        3. Construct a master prompt with context
        4. Route based on complexity (keyword-based routing)
        5. Get response from appropriate system
        6. Save response to vector memory
        7. Return final response

        Args:
            user_message: The user's input message.

        Returns:
            str: The final response from the AI system.

        Raises:
            Exception: If processing fails critically.
        """
        if not user_message or not isinstance(user_message, str):
            error_msg = "Invalid user message"
            logger.warning(error_msg)
            return f"Error: {error_msg}"

        try:
            logger.info(f"Processing message: {user_message[:50]}...")

            # Step 1: Log to ledger
            self.ledger_memory.log_event(
                LogLevel.INFO,
                f"User message received",
                {"message": user_message[:100]}
            )

            # Step 2: Retrieve context from memory
            context_memories = self._retrieve_context(user_message, n_results=2)

            # Step 3: Construct master prompt
            master_prompt = self._construct_master_prompt(user_message, context_memories)

            # Step 4: Route decision
            system_choice = self._route_decision(user_message)

            # Step 5: Get response from appropriate system
            try:
                # Format messages array
                messages = [
                    {"role": "system", "content": self.charter_text}
                ]

                # Add context as an assistant or system thought
                if context_memories:
                    context_str = "\n".join([f"- {m['document']}" for m in context_memories])
                    messages.append({
                        "role": "user",
                        "content": f"[Context Retrieval System]: Retrieved relevant memories:\n{context_str}"
                    })
                    messages.append({
                        "role": "assistant",
                        "content": "I have acknowledged the retrieved context and will use it to inform my response if relevant."
                    })

                messages.append({"role": "user", "content": user_message})

                if system_choice == "system_2" and self.cognitive_router.get_system_2_available():
                    ai_response = await self.cognitive_router.route_to_system_2(messages)
                else:
                    ai_response = await self.cognitive_router.route_to_system_1(messages)

                logger.info(f"Response generated ({len(ai_response)} chars)")

            except Exception as e:
                error_response = f"System error: {str(e)}"
                logger.error(f"Failed to generate response: {e}", exc_info=True)
                self.ledger_memory.log_event(
                    LogLevel.ERROR,
                    "Failed to generate response",
                    {"error": str(e), "system": system_choice}
                )
                return error_response

            # Step 6: Store response in memory
            try:
                memory_id = self.vector_memory.add_memory(
                    text=f"User: {user_message}\nAssistant: {ai_response}",
                    metadata={
                        "type": "conversation",
                        "system_used": system_choice,
                        "timestamp": datetime.now().isoformat(),
                        "context_retrieved": len(context_memories) > 0
                    }
                )
                logger.debug(f"Response stored with memory ID: {memory_id}")
            except Exception as e:
                logger.warning(f"Failed to store response in memory: {e}")
                # Continue despite failure - still return response

            # Step 7: Log successful processing and return response
            self.ledger_memory.log_event(
                LogLevel.INFO,
                "Message processed successfully",
                {
                    "input_length": len(user_message),
                    "output_length": len(ai_response),
                    "system_used": system_choice
                }
            )

            return ai_response

        except Exception as e:
            error_msg = f"Critical error in process_message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def close(self) -> None:
        """
        Close all database connections gracefully.

        Should be called during application shutdown.
        """
        try:
            logger.info("Closing ExecutiveAgent")
            self.ledger_memory.close()
            logger.info("ExecutiveAgent shutdown complete")
        except Exception as e:
            logger.error(f"Error closing ExecutiveAgent: {e}")
