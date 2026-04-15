"""
Orchestrator Module - Implements State Graph Architecture

This module replaces the linear executive flow with a state-based orchestration
model. It routes the user input through a Supervisor Node (System 2),
Worker Nodes (System 1), and a Critic Node before finalizing the output.
"""

import os
import logging
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List

from src.memory.vector_db import VectorMemory
from src.memory.ledger_db import LedgerMemory, LogLevel
from src.memory.core_memory import CoreMemory
from src.core.llm_router import CognitiveRouter, RequiresMFAError, RequiresHITLError
from src.core.security import verify_mfa_challenge

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Central orchestration engine using a State Graph architecture.
    """

    def __init__(
        self,
        vector_db_path: str = "data/chroma_storage",
        ledger_db_path: str = "data/ledger.db",
        core_memory_path: str = "data/core_memory.json",
        gemini_model: str = "gemini-3.1-pro-preview"
    ) -> None:
        logger.info("Initializing Orchestrator (State Graph)")

        try:
            self.vector_memory = VectorMemory(persist_dir=vector_db_path)
            self.ledger_memory = LedgerMemory(db_path=ledger_db_path)
            self.core_memory = CoreMemory(memory_file_path=core_memory_path)
            self.cognitive_router = CognitiveRouter(model_name=gemini_model)

            self.charter_text = self._load_charter()
            self.pending_mfa = {}
            self.pending_hitl_state = {}

            logger.info("Orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
            raise

    def _load_charter(self, filepath: str = "charter.md") -> str:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return f.read().strip()
            return "Core Directive: Do no harm."
        except Exception:
            return "Core Directive: Do no harm."

    async def _evaluate_salience(self, text_to_save: str) -> int:
        """
        Evaluates informational value for long-term storage (1-10).
        """
        try:
            system_prompt = "Evaluate the informational value of this text for long-term memory storage. Output a single integer from 1 to 10. 1 is conversational filler or noise. 10 is highly valuable factual, strategic, or environmental data."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_save}
            ]

            # Using asyncio.wait_for for timeout
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages),
                timeout=10.0
            )

            match = re.search(r'\d+', response)
            if match:
                score = int(match.group())
                return min(max(score, 1), 10)
            return 10
        except Exception as e:
            logger.warning(f"Salience filter error/timeout: {e}. Defaulting to 10.")
            return 10

    async def supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        System 2: Plans the execution.
        """
        user_input = state["user_input"]
        core_mem_str = self.core_memory.get_context_string()

        system_prompt = f"You are the Supervisor Node.\n{self.charter_text}\n{core_mem_str}\nAnalyze the user input and decide the next steps. Output a plan as a JSON list of worker nodes to call, e.g. ['research_agent'] or ['coder_agent'] or ['research_agent', 'coder_agent']. If it's a simple conversation, you can just output an empty list [] and provide a final response directly."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        try:
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_2(messages),
                timeout=20.0
            )

            # Simple extraction for JSON array
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                plan_str = "[" + match.group(1) + "]"
                try:
                    plan = json.loads(plan_str)
                    state["current_plan"] = plan
                except json.JSONDecodeError:
                    state["current_plan"] = []
            else:
                state["current_plan"] = []
                state["final_response"] = response # direct response

        except Exception as e:
            logger.error(f"Supervisor Node failed: {e}")
            state["current_plan"] = []
            state["final_response"] = "Supervisor node failed."

        return state

    async def research_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker Node: System 1 with search_archival_memory tool.
        """
        user_input = state["user_input"]
        core_mem_str = self.core_memory.get_context_string()

        system_prompt = f"You are the Research Agent.\n{self.charter_text}\n{core_mem_str}\nYour job is to search the archival memory for relevant context and summarize findings. You must provide a final conversational response directly to the user detailing your findings."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        try:
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=["search_archival_memory"]),
                timeout=30.0
            )
            state["worker_outputs"]["research"] = response
        except RequiresHITLError:
            raise
        except Exception as e:
            logger.error(f"Research Agent failed: {e}")
            state["worker_outputs"]["research"] = f"Error: {e}"

        return state

    async def coder_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker Node: System 1 with update_ledger and update_core_memory tools.
        """
        user_input = state["user_input"]
        core_mem_str = self.core_memory.get_context_string()
        research_context = state["worker_outputs"].get("research", "")

        system_prompt = f"You are the Coder Agent.\n{self.charter_text}\n{core_mem_str}\nYour job is to execute coding tasks, update core memory, or update the ledger. You must provide a final conversational response to the user."

        content = f"User Input: {user_input}\n"
        if research_context:
            content += f"Research Context: {research_context}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        try:
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=["update_ledger", "update_core_memory", "request_core_update"]),
                timeout=30.0
            )
            state["worker_outputs"]["coder"] = response
        except RequiresHITLError:
            raise
        except Exception as e:
            logger.error(f"Coder Agent failed: {e}")
            state["worker_outputs"]["coder"] = f"Error: {e}"

        return state

    async def critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        System 1: Checks output against charter.
        """
        # Formulate output to evaluate
        output_to_eval = ""
        if state.get("worker_outputs"):
            # Grab the last worker's output to evaluate
            last_worker = list(state["worker_outputs"].keys())[-1]
            output_to_eval = state["worker_outputs"][last_worker]
        elif state.get("final_response"):
            output_to_eval = state["final_response"]

        if not output_to_eval:
             state["critic_feedback"] = "PASS" # nothing to evaluate
             return state

        system_prompt = f"You are the Critic Node. Evaluate the following output against the charter. Output PASS if it strictly follows the charter. Output FAIL and a reason if it violates the charter.\nCharter:\n{self.charter_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": output_to_eval}
        ]

        try:
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=[]),
                timeout=10.0
            )
            if "FAIL" in response.upper():
                state["iteration_count"] += 1
                state["critic_feedback"] = response
            else:
                state["critic_feedback"] = "PASS"
                if not state.get("final_response"):
                     state["final_response"] = output_to_eval
        except Exception as e:
            logger.warning(f"Critic node failed/timed out: {e}. Defaulting to PASS.")
            state["critic_feedback"] = "PASS"
            if not state.get("final_response"):
                state["final_response"] = output_to_eval

        if state["iteration_count"] >= 3 and state["critic_feedback"] != "PASS":
            raise RequiresHITLError(
                f"Guidance Needed: The Critic node continuously rejected the worker's output after 3 iterations.\nQuestion: How should I proceed to satisfy the charter?"
            )

        return state

    async def process_message(self, user_message: str, user_id: str) -> str:
        """
        Main entry point for State Graph Orchestration.
        """
        if not user_message:
            return "Error: Invalid message"

        if user_id in self.pending_mfa:
            pending_tool = self.pending_mfa[user_id]
            if verify_mfa_challenge(user_message):
                del self.pending_mfa[user_id]
                result = await self.cognitive_router._execute_tool(
                    pending_tool["name"],
                    pending_tool["arguments"]
                )
                return result
            else:
                del self.pending_mfa[user_id]
                return "Error: MFA authorization failed. Action aborted."

        if user_id in self.pending_hitl_state:
            # Resume state
            state = self.pending_hitl_state.pop(user_id)
            state["admin_guidance"] = user_message
            # Append guidance so the next node run sees it
            state["user_input"] += f"\n[ADMIN GUIDANCE: {user_message}]"
            # Reset iteration count to allow processing the new guidance
            state["iteration_count"] = 0
            # Force re-planning in the next step
            state["current_plan"] = []
        else:
            # New state
            state = {
                "user_id": user_id,
                "user_input": user_message,
                "current_plan": [],
                "worker_outputs": {},
                "final_response": "",
                "iteration_count": 0,
                "admin_guidance": ""
            }

        try:
            max_iterations = 3

            while state["iteration_count"] < max_iterations:

                # 1. Supervisor (only run on first iteration, and if we don't have a plan)
                if not state["current_plan"] and state["iteration_count"] == 0:
                    state = await self.supervisor_node(state)
                else:
                    # Append critic feedback to user input to force correction
                    state["user_input"] += f"\n[CRITIC FEEDBACK on previous attempt: {state['critic_feedback']}. Fix your output.]"

                # 2. Workers
                for task in state["current_plan"]:
                    if "research" in task.lower():
                        state = await self.research_agent(state)
                    elif "coder" in task.lower():
                        state = await self.coder_agent(state)

                # If no plan, we assume supervisor directly handled it (or nothing to do)
                if not state["current_plan"] and not state["final_response"]:
                     state["final_response"] = "I don't have a plan for that."

                # 3. Critic
                state = await self.critic_node(state)

                if state["critic_feedback"] == "PASS":
                     break
                else:
                     logger.warning(f"Critic rejected output on iteration {state['iteration_count']}")
                     # Clean up final response and outputs for retry
                     state["final_response"] = ""
                     state["worker_outputs"] = {}


            # 4. Fallback if hit max iterations and failed
            if state["iteration_count"] >= max_iterations and state["critic_feedback"] != "PASS":
                state["final_response"] = "I am unable to fulfill this request as my output continuously failed the internal safety checks."

            if not state["final_response"]:
                state["final_response"] = "No valid response could be generated."

            final_resp = state["final_response"]

            # 5. Salience Filter
            text_to_save = f"User: {user_message}\nAssistant: {final_resp}"
            salience_score = await self._evaluate_salience(text_to_save)

            if salience_score >= 6:
                try:
                    self.vector_memory.add_memory(
                        text=text_to_save,
                        metadata={
                            "type": "conversation",
                            "timestamp": datetime.now().isoformat(),
                            "salience": salience_score
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")

            return final_resp

        except RequiresMFAError as mfa_err:
            self.pending_mfa[user_id] = {
                "name": mfa_err.tool_name,
                "arguments": mfa_err.arguments
            }
            return "SECURITY LOCK: To authorize this core change, please complete the phrase: 'The sky is...'"
        except RequiresHITLError as hitl_err:
            self.pending_hitl_state[user_id] = state
            return str(hitl_err)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            return "An internal error occurred."

    def close(self) -> None:
        try:
            self.ledger_memory.close()
        except Exception as e:
            logger.error(f"Error closing Orchestrator: {e}")
