"""
Orchestrator Module - Implements State Graph Architecture (Sprint 6)

State Graph with Energy Budget, Proactive Heartbeat, and Charter enforcement.
Passes a State Dictionary between specialized nodes until task is complete,
exhausted (energy = 0), or blocked (HITL required).
"""

import os
import logging
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.memory.vector_db import VectorMemory
from src.memory.ledger_db import LedgerMemory, LogLevel
from src.memory.core_memory import CoreMemory
from src.core.llm_router import CognitiveRouter, RequiresMFAError, RequiresHITLError
from src.core.security import verify_mfa_challenge

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = Exception

logger = logging.getLogger(__name__)

# Energy costs per operation
ENERGY_COST_SUPERVISOR = 10
ENERGY_COST_WORKER = 15
ENERGY_COST_CRITIC = 10
ENERGY_COST_TOOL = 5
HEARTBEAT_INTERVAL = 1800  # 30 minutes


class Orchestrator:
    """
    Central orchestration engine using a State Graph architecture.

    State Dictionary schema:
        user_id, user_input, current_plan, worker_outputs,
        final_response, iteration_count, admin_guidance, energy_remaining
    """

    def __init__(
        self,
        vector_db_path: str = "data/chroma_storage",
        ledger_db_path: str = "data/ledger.db",
        core_memory_path: str = "data/core_memory.json",
        gemini_model: str = "gemini-2.0-flash",
        local_model: str = "gemma4:e4b"
    ) -> None:
        logger.info("Initializing Orchestrator (State Graph)")

        try:
            self.vector_memory = VectorMemory(persist_dir=vector_db_path)
            self.ledger_memory = LedgerMemory(db_path=ledger_db_path)
            self.core_memory = CoreMemory(memory_file_path=core_memory_path)
            self.cognitive_router = CognitiveRouter(model_name=gemini_model, local_model=local_model)

            self.charter_text = self._load_charter()
            self.pending_mfa = {}
            self.pending_hitl_state = {}
            self.outbound_queue: Optional[asyncio.Queue] = None

            # Seed initial goals if backlog is empty (Sprint 7)
            self.ledger_memory.seed_initial_goals()

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

    def _new_state(self, user_id: str, user_message: str) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "user_input": user_message,
            "chat_history": [],
            "current_plan": [],
            "worker_outputs": {},
            "final_response": "",
            "iteration_count": 0,
            "admin_guidance": "",
            "energy_remaining": 100,
        }

    async def _deduct_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        """Deduct energy and raise HITL if exhausted."""
        state["energy_remaining"] = state.get("energy_remaining", 100) - amount
        logger.debug(f"Energy -{amount} ({reason}). Remaining: {state['energy_remaining']}")
        if state["energy_remaining"] <= 0:
            raise RequiresHITLError(
                f"Energy Budget Exhausted: The system consumed all cognitive energy on this task.\n"
                f"Question: How should I prioritize the remaining work for: '{state.get('user_input', '')}'"
            )
        return state

    async def _notify_admin(self, message: str) -> None:
        """Send a message to the admin via the outbound queue (used by heartbeat)."""
        if self.outbound_queue is not None:
            await self.outbound_queue.put(message)
        else:
            logger.info(f"[Admin notification (no queue)]: {message}")

    async def _heartbeat_loop(self) -> None:
        """
        Proactive Heartbeat: wakes every 30 min, queries the Objective Backlog
        for the highest-priority pending Task, processes it if energy allows,
        then notifies the admin with a summary.
        """
        logger.info("Heartbeat loop started.")
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                logger.info("Heartbeat: Querying objective backlog...")
                task = self.ledger_memory.get_highest_priority_task()

                if not task:
                    logger.info("Heartbeat: Backlog is empty. Nothing to do.")
                    continue

                energy_available = 100  # Heartbeat starts with full budget
                if task["estimated_energy"] > energy_available:
                    logger.info(
                        f"Heartbeat: Task #{task['id']} needs {task['estimated_energy']} energy "
                        f"but only {energy_available} available. Skipping."
                    )
                    continue

                logger.info(f"Heartbeat: Accepting task #{task['id']}: {task['title'][:60]}")
                self.ledger_memory.update_objective_status(task["id"], "active")

                result = await self.process_message(
                    user_message=(
                        f"[HEARTBEAT TASK #{task['id']}]: {task['title']}\n"
                        f"Execute this task autonomously and report what you did."
                    ),
                    user_id="heartbeat"
                )

                self.ledger_memory.update_objective_status(task["id"], "completed")

                summary = (
                    f"Heartbeat completed task #{task['id']}:\n"
                    f"  Task: {task['title']}\n"
                    f"  Result: {result[:200]}"
                )
                await self._notify_admin(summary)
                logger.info(f"Heartbeat: Task #{task['id']} completed.")

            except Exception as e:
                logger.error(f"Heartbeat error: {e}", exc_info=True)

    async def _evaluate_salience(self, text_to_save: str) -> int:
        """Score informational value for long-term storage (1-10)."""
        try:
            messages = [
                {"role": "system", "content": "Score this text for long-term memory value. Output a number 1-10 only. 1=conversational filler, 10=valuable facts/decisions."},
                {"role": "user", "content": text_to_save[:500]}  # Truncate to reduce latency
            ]
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages),
                timeout=60.0
            )
            match = re.search(r'\d+', response)
            if match:
                return min(max(int(match.group()), 1), 10)
            return 10
        except Exception as e:
            logger.warning(f"Salience filter error: {e}. Defaulting to 10.")
            return 10

    async def _save_memory_async(self, text: str) -> None:
        """Fire-and-forget memory storage with salience gating."""
        try:
            score = await self._evaluate_salience(text)
            if score >= 6:
                self.vector_memory.add_memory(
                    text=text,
                    metadata={"type": "conversation", "timestamp": datetime.now().isoformat(), "salience": score}
                )
        except Exception as e:
            logger.warning(f"Async memory save failed: {e}")

    async def supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plans execution. Uses System 2 (Gemini) if available, falls back to System 1."""
        state = await self._deduct_energy(state, ENERGY_COST_SUPERVISOR, "supervisor")
        user_input = state["user_input"]
        core_mem_str = self.core_memory.get_context_string()

        system_prompt = (
            f"You are the Supervisor. Respond to the user, then on the very last line "
            f"declare which workers are needed.\n"
            f"{self.charter_text}\n{core_mem_str}\n\n"
            f"Format — write your response, then end with exactly:\n"
            f"WORKERS: []\n"
            f"or WORKERS: [\"research_agent\"] or WORKERS: [\"coder_agent\"]\n"
            f"Use [] for simple chat. Only use workers for research or coding tasks."
        )

        # Build messages: system prompt → chat history → current user turn
        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.get("chat_history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_input})

        try:
            use_system2 = self.cognitive_router.get_system_2_available()
            response = None
            if use_system2:
                try:
                    logger.info("Routing Supervisor through System 2")
                    response = await asyncio.wait_for(
                        self.cognitive_router.route_to_system_2(messages),
                        timeout=30.0
                    )
                except Exception as s2_err:
                    logger.warning(f"System 2 failed in supervisor ({s2_err}). Falling back to System 1.")
                    response = None

            if not response or response.startswith("[System 2"):
                logger.info("Routing Supervisor through System 1 (Local Model)")
                response = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_1(messages),
                    timeout=60.0
                )

            if not response or response.startswith("[System 1 - Error]") or response.startswith("[System 2"):
                logger.warning(f"Supervisor received error: {response}")
                state["current_plan"] = []
                state["final_response"] = "Supervisor encountered an error. Please try again."
                return state

            # Parse WORKERS tag — search last 3 lines to handle trailing newlines/notes
            lines = response.strip().split('\n')
            workers_match = None
            for line in reversed(lines[-3:]):
                workers_match = re.search(r'WORKERS:\s*(\[.*?\])', line)
                if workers_match:
                    break
            if workers_match:
                try:
                    plan = json.loads(workers_match.group(1))
                    # Strip all lines that contain the WORKERS tag to extract the clean answer
                    answer = '\n'.join(
                        ln for ln in lines if not re.search(r'WORKERS:\s*\[', ln)
                    ).strip()
                    if isinstance(plan, list) and len(plan) > 0 and all(isinstance(x, str) for x in plan):
                        state["current_plan"] = plan
                        if answer:
                            state["worker_outputs"]["supervisor_context"] = answer
                    else:
                        # Empty workers list — direct answer
                        state["current_plan"] = []
                        state["final_response"] = answer if answer else response.strip()
                except (json.JSONDecodeError, ValueError):
                    state["current_plan"] = []
                    answer = '\n'.join(
                        ln for ln in lines if not re.search(r'WORKERS:\s*\[', ln)
                    ).strip()
                    state["final_response"] = answer or response.strip()
            else:
                # No WORKERS tag — treat entire response as direct answer
                state["current_plan"] = []
                state["final_response"] = response.strip()

        except asyncio.TimeoutError:
            logger.error("Supervisor node timeout (60s)")
            state["current_plan"] = []
            state["final_response"] = "Planning timed out. Please try again."
        except RequiresHITLError:
            raise
        except Exception as e:
            logger.error(f"Supervisor Node failed: {e}", exc_info=True)
            state["current_plan"] = []
            state["final_response"] = "Supervisor encountered an error. Please try again."

        return state

    async def research_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Worker: searches archival memory for relevant context."""
        state = await self._deduct_energy(state, ENERGY_COST_WORKER, "research_agent")
        core_mem_str = self.core_memory.get_context_string()

        messages = [
            {"role": "system", "content": f"You are the Research Agent. Search archival memory for relevant context and provide findings.\n{self.charter_text}\n{core_mem_str}"},
            {"role": "user", "content": state["user_input"]}
        ]
        try:
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=["search_archival_memory"]),
                timeout=60.0
            )
            state["worker_outputs"]["research"] = response
        except RequiresHITLError:
            raise
        except Exception as e:
            logger.error(f"Research Agent failed: {e}")
            state["worker_outputs"]["research"] = f"Error: {e}"

        return state

    async def coder_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Worker: executes coding tasks, updates memory/ledger."""
        state = await self._deduct_energy(state, ENERGY_COST_WORKER, "coder_agent")
        core_mem_str = self.core_memory.get_context_string()
        research_context = state["worker_outputs"].get("research", "")

        content = f"User Input: {state['user_input']}\n"
        if research_context:
            content += f"Research Context: {research_context}\n"

        messages = [
            {"role": "system", "content": f"You are the Coder Agent. Execute coding tasks and update memory as needed.\n{self.charter_text}\n{core_mem_str}"},
            {"role": "user", "content": content}
        ]
        try:
            response = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=["update_ledger", "update_core_memory", "request_core_update", "spawn_new_objective", "update_objective_status"]),
                timeout=60.0
            )
            state["worker_outputs"]["coder"] = response
        except RequiresHITLError:
            raise
        except Exception as e:
            logger.error(f"Coder Agent failed: {e}")
            state["worker_outputs"]["coder"] = f"Error: {e}"

        return state

    async def critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Checks worker output against the charter. Skipped for direct supervisor responses."""
        state = await self._deduct_energy(state, ENERGY_COST_CRITIC, "critic")

        output_to_eval = ""
        if state.get("worker_outputs"):
            last_worker = list(state["worker_outputs"].keys())[-1]
            output_to_eval = state["worker_outputs"][last_worker]
        elif state.get("final_response"):
            output_to_eval = state["final_response"]

        if not output_to_eval:
            state["critic_feedback"] = "PASS"
            return state

        messages = [
            {"role": "system", "content": f"Evaluate the output. Output PASS if it follows the charter, or FAIL with a reason.\nCharter: {self.charter_text}"},
            {"role": "user", "content": output_to_eval[:1000]}  # Truncate to reduce latency
        ]

        try:
            use_system2 = self.cognitive_router.get_system_2_available()
            if use_system2:
                logger.info("Routing Critic through System 2 (Gemini)")
                response = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_2(messages),
                    timeout=30.0
                )
            else:
                response = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_1(messages, allowed_tools=[]),
                    timeout=60.0
                )
            if "FAIL" in response.upper():
                state["iteration_count"] += 1
                state["critic_feedback"] = response
            else:
                state["critic_feedback"] = "PASS"
                if not state.get("final_response"):
                    state["final_response"] = output_to_eval
        except Exception as e:
            logger.warning(f"Critic node failed/timed out: {e}. Defaulting to PASS.", exc_info=True)
            state["critic_feedback"] = "PASS"
            if not state.get("final_response"):
                state["final_response"] = output_to_eval

        if state["iteration_count"] >= 3 and state["critic_feedback"] != "PASS":
            raise RequiresHITLError(
                f"Critic rejected output 3 times. Guidance needed.\n"
                f"Question: How should I proceed to satisfy the charter?"
            )

        return state

    async def process_message(self, user_message: str, user_id: str) -> str:
        """Main entry point: State Graph execution with Energy Budget."""
        if not user_message:
            return "Error: Invalid message"

        # MFA resumption
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

        # HITL resumption
        if user_id in self.pending_hitl_state:
            state = self.pending_hitl_state.pop(user_id)
            state["admin_guidance"] = user_message
            state["user_input"] += f"\n[ADMIN GUIDANCE: {user_message}]"
            state["iteration_count"] = 0
            state["current_plan"] = []
            state["energy_remaining"] = min(state.get("energy_remaining", 0) + 50, 100)  # Refuel on guidance
        else:
            state = self._new_state(user_id, user_message)
            # Load last 10 conversational turns for this user
            if user_id != "heartbeat":
                try:
                    state["chat_history"] = self.ledger_memory.get_chat_history(user_id, limit=10)
                except Exception as e:
                    logger.warning(f"Failed to load chat history for {user_id}: {e}")

        try:
            max_iterations = 3

            while state["iteration_count"] < max_iterations:

                # 1. Supervisor (first iteration only, or after HITL guidance)
                if not state["current_plan"] and state["iteration_count"] == 0:
                    state = await self.supervisor_node(state)
                else:
                    state["user_input"] += f"\n[CRITIC FEEDBACK: {state['critic_feedback']}. Fix your output.]"

                # If supervisor handled it directly (no workers), skip critic to reduce latency
                if not state["current_plan"] and state.get("final_response"):
                    break

                # 2. Workers
                for task in state["current_plan"]:
                    if "research" in task.lower():
                        state = await self.research_agent(state)
                    elif "coder" in task.lower():
                        state = await self.coder_agent(state)

                if not state["current_plan"] and not state.get("final_response"):
                    state["final_response"] = "No response could be generated. Please try again."
                    break

                # 3. Critic (only when workers were called)
                state = await self.critic_node(state)

                if state["critic_feedback"] == "PASS":
                    break
                else:
                    logger.warning(f"Critic rejected output on iteration {state['iteration_count']}")
                    state["final_response"] = ""
                    state["worker_outputs"] = {}

            if state["iteration_count"] >= max_iterations and state.get("critic_feedback") != "PASS":
                state["final_response"] = "Unable to fulfill this request — output repeatedly failed internal safety checks."

            if not state.get("final_response"):
                state["final_response"] = "No valid response could be generated."

            final_resp = self.cognitive_router.sanitize_response(state["final_response"])

            # Persist this conversation turn to chat history (skip heartbeat tasks)
            if user_id != "heartbeat":
                try:
                    self.ledger_memory.save_chat_turn(user_id, "user", user_message)
                    self.ledger_memory.save_chat_turn(user_id, "assistant", final_resp)
                except Exception as e:
                    logger.warning(f"Failed to save chat turn for {user_id}: {e}")

            # 4. Memory storage (non-blocking fire-and-forget)
            asyncio.create_task(
                self._save_memory_async(f"User: {user_message}\nAssistant: {final_resp}")
            )

            return final_resp

        except RequiresMFAError as mfa_err:
            self.pending_mfa[user_id] = {"name": mfa_err.tool_name, "arguments": mfa_err.arguments}
            return "SECURITY LOCK: To authorize this core change, complete the phrase: 'The sky is...'"
        except RequiresHITLError as hitl_err:
            self.pending_hitl_state[user_id] = state
            return str(hitl_err)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            return "An internal error occurred."

    def close(self) -> None:
        try:
            if hasattr(self, 'vector_memory') and self.vector_memory:
                self.vector_memory.close()
            if hasattr(self, 'ledger_memory') and self.ledger_memory:
                self.ledger_memory.close()
            logger.info("Orchestrator resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing Orchestrator: {e}", exc_info=True)
