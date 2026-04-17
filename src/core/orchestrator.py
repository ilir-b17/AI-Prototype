"""
Orchestrator Module - Implements State Graph Architecture (Sprint 6)

State Graph with Energy Budget, Proactive Heartbeat, and Charter enforcement.
Passes a State Dictionary between specialized nodes until task is complete,
exhausted (energy = 0), or blocked (HITL required).
"""

import os
import platform
import logging
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from src.memory.vector_db import VectorMemory
from src.memory.ledger_db import LedgerMemory, LogLevel
from src.memory.core_memory import CoreMemory
from src.core.llm_router import CognitiveRouter, RouterResult, RequiresMFAError, RequiresHITLError, RequiresCapabilitySynthesisError
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

# Key used in the state dict to signal a blocked (non-ok) router result
_BLOCKED_KEY = "_blocked_result"


class Orchestrator:
    """
    Central orchestration engine using a State Graph architecture.

    State Dictionary schema:
        user_id, user_input, current_plan, worker_outputs,
        final_response, iteration_count, admin_guidance, energy_remaining

    Lifecycle:
        1. ``__init__`` — synchronous; creates memory objects (no DB I/O yet).
        2. ``await async_init()`` — must be called once from an async context
           (the Telegram bot's ``post_init`` hook) to open the aiosqlite
           connection and seed initial goals.
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
            self.pending_mfa: Dict[str, dict] = {}
            self.pending_hitl_state: Dict[str, dict] = {}
            self.pending_tool_approval: Dict[str, dict] = {}
            self.outbound_queue: Optional[asyncio.Queue] = None
            self.sensory_state: Dict[str, str] = {}
            self._refresh_sensory_state()

            logger.info("Orchestrator __init__ complete — call async_init() to finish setup")
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
            raise

    async def async_init(self) -> None:
        """
        Async second-phase initialisation.  Must be awaited before any messages
        are processed.  Opens the aiosqlite ledger, seeds initial goals, and
        reloads previously approved dynamic tools.
        """
        await self.ledger_memory.initialize()
        await self.ledger_memory.seed_initial_goals()
        await self._load_approved_tools()
        logger.info("Orchestrator async_init complete")

    def _refresh_sensory_state(self) -> None:
        """Snapshot current machine state into self.sensory_state."""
        cpu = f"{psutil.cpu_percent(interval=None):.0f}%" if _PSUTIL_AVAILABLE else "unavailable"

        # platform.release() returns "10" even on Windows 11 — check build number instead
        os_name = platform.system()
        if os_name == "Windows":
            try:
                build = int(platform.version().split(".")[-1])
                win_ver = "11" if build >= 22000 else "10"
                os_str = f"Windows {win_ver} (build {build})"
            except Exception:
                os_str = f"Windows {platform.release()}"
        else:
            os_str = f"{os_name} {platform.release()}"

        self.sensory_state = {
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "os": os_str,
            "cpu_usage": cpu,
        }

    async def _sensory_update_loop(self) -> None:
        """Background task: refresh sensory_state every 60 seconds."""
        while True:
            await asyncio.sleep(60)
            self._refresh_sensory_state()

    def _get_sensory_context(self) -> str:
        s = self.sensory_state
        return (
            f"[Machine Context — {s.get('current_time', 'unknown')} | "
            f"OS: {s.get('os', 'unknown')} | "
            f"CPU: {s.get('cpu_usage', 'unknown')}]"
        )

    def _load_charter(self, filepath: str = "charter.md") -> str:
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return f.read().strip()
            return "Core Directive: Do no harm."
        except Exception:
            return "Core Directive: Do no harm."

    def _get_capabilities_string(self) -> str:
        """Compact tool names list derived live from the SkillRegistry.

        Full schemas are already sent to Ollama as structured JSON — this
        one-liner is enough for conversational self-reference without
        duplicating ~400 tokens of schema text per call.
        """
        names = ", ".join(s["name"] for s in self.cognitive_router.registry.get_schemas())
        return f"Available tools (use them — do not claim you lack them): {names}"

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
                task = await self.ledger_memory.get_highest_priority_task()

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
                await self.ledger_memory.update_objective_status(task["id"], "active")

                result = await self.process_message(
                    user_message=(
                        f"[HEARTBEAT TASK #{task['id']}]: {task['title']}\n"
                        f"Execute this task autonomously and report what you did."
                    ),
                    user_id="heartbeat"
                )

                await self.ledger_memory.update_objective_status(task["id"], "completed")

                summary = (
                    f"Heartbeat completed task #{task['id']}:\n"
                    f"  Task: {task['title']}\n"
                    f"  Result: {result[:200]}"
                )
                await self._notify_admin(summary)
                logger.info(f"Heartbeat: Task #{task['id']} completed.")

            except Exception as e:
                logger.error(f"Heartbeat error: {e}", exc_info=True)

    async def _save_memory_async(self, text: str) -> None:
        """Fire-and-forget memory storage gated by response length.

        Skips trivial exchanges (short greetings, one-liners).
        Threshold: combined user+assistant text must exceed 200 chars.
        Uses asyncio.to_thread() so the blocking ChromaDB call does not
        stall the event loop.
        """
        try:
            if len(text) < 200:
                return
            await self.vector_memory.add_memory_async(
                text=text,
                metadata={"type": "conversation", "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.warning(f"Async memory save failed: {e}")

    async def _load_approved_tools(self) -> None:
        """Re-register all previously approved dynamic tools from the DB at startup."""
        try:
            approved = await self.ledger_memory.get_approved_tools()
            for tool in approved:
                try:
                    self.cognitive_router.register_dynamic_tool(
                        tool["name"], tool["code"], tool["schema_json"]
                    )
                    logger.info(f"Restored dynamic tool '{tool['name']}' from registry")
                except Exception as e:
                    logger.error(f"Failed to restore tool '{tool['name']}': {e}")
        except Exception as e:
            logger.warning(f"_load_approved_tools failed: {e}")

    async def tool_synthesis_node(
        self,
        state: Dict[str, Any],
        router_result: RouterResult,
    ) -> str:
        """
        Cognitive handoff: router flagged a gap → System 2 synthesises a tool
        → Admin approves via HITL → tool registered → original query retried.

        Returns a HITL prompt string for the Admin; the orchestrator stores the
        synthesis payload in pending_tool_approval until the Admin replies.
        """
        user_id = state["user_id"]
        gap_description = router_result.gap_description
        suggested_tool_name = router_result.suggested_tool_name
        logger.info(f"Tool synthesis triggered: gap='{gap_description}'")

        if not self.cognitive_router.get_system_2_available():
            return (
                "System 1 identified a capability gap but System 2 is offline — "
                "cannot synthesise a new tool right now. Please configure GROQ_API_KEY."
            )

        try:
            synthesis = await asyncio.wait_for(
                self.cognitive_router.synthesize_tool(
                    gap_description=gap_description,
                    suggested_tool_name=suggested_tool_name,
                    user_query=state["user_input"],
                ),
                timeout=60.0,
            )
        except Exception as e:
            logger.error(f"Tool synthesis failed: {e}", exc_info=True)
            return f"System 2 failed to synthesise a tool for '{gap_description}': {e}"

        # Store payload for approval resumption
        self.pending_tool_approval[user_id] = {
            "synthesis": synthesis,
            "original_state": state,
        }

        # Build Admin-facing HITL prompt with the proposed code
        hitl_msg = (
            f"🔧 TOOL SYNTHESIS REQUEST\n\n"
            f"System 1 could not answer: \"{state['user_input']}\"\n"
            f"Gap identified: {gap_description}\n\n"
            f"System 2 has drafted the following tool:\n"
            f"─────────────────────────────\n"
            f"Name: {synthesis['tool_name']}\n"
            f"Description: {synthesis['description']}\n\n"
            f"Code:\n```python\n{synthesis['code']}\n```\n"
            f"─────────────────────────────\n"
            f"Reply YES to approve and deploy, or NO to reject."
        )
        return hitl_msg

    def _extract_router_content(self, result: RouterResult) -> Optional[str]:
        """
        Return the string content from an "ok" RouterResult, or None if it is
        a non-ok result (MFA / HITL / capability-gap).  Non-ok results should
        be stored in state[_BLOCKED_KEY] by the caller.
        """
        if result.status == "ok":
            return result.content
        return None

    async def supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plans execution. Tries System 1 (local) first per Directive 1.3; escalates to System 2 only on failure."""
        state = await self._deduct_energy(state, ENERGY_COST_SUPERVISOR, "supervisor")
        user_input = state["user_input"]
        core_mem_str = await self.core_memory.get_context_string()
        capabilities_str = self._get_capabilities_string()

        system_prompt = (
            f"You are AIDEN — a local autonomous AI agent on the Admin's machine (Ollama). "
            f"You are NOT ChatGPT/Claude/Gemini. "
            f"You have persistent memory: the chat history above is real (SQLite). "
            f"You have tools — never deny capabilities listed below. "
            f"Reply in plain conversational text, no markdown headers. "
            f"Private reasoning: wrap in <think>...</think> — it will be stripped.\n\n"
            f"{self._get_sensory_context()}\n\n"
            f"{self.charter_text}\n{core_mem_str}\n\n"
            f"{capabilities_str}\n\n"
            f"Respond to the user, then on the very last line declare which workers are needed.\n"
            f"Format: WORKERS: [] for chat, WORKERS: [\"research_agent\"] or [\"coder_agent\"] for tasks."
        )

        # Build messages: system prompt → chat history → sanitized user turn
        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.get("chat_history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        # Wrap user input in XML tags to isolate it from system directives
        messages.append({"role": "user", "content": f"<user_input>{user_input}</user_input>"})

        try:
            # Local-first per Directive 1.3 — try System 1, escalate to System 2 only on failure
            router_result: Optional[RouterResult] = None
            try:
                logger.info("Routing Supervisor through System 1 (Local Model)")
                router_result = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_1(messages),
                    timeout=60.0
                )
            except Exception as s1_err:
                logger.warning(f"System 1 failed in supervisor ({s1_err}). Escalating to System 2.")
                router_result = None

            # Propagate non-ok RouterResult immediately (MFA / HITL / capability gap)
            if router_result is not None and router_result.status != "ok":
                state[_BLOCKED_KEY] = router_result
                return state

            response = router_result.content if router_result else None

            if not response or response.startswith("[System 1 - Error]"):
                if self.cognitive_router.get_system_2_available():
                    try:
                        logger.info("Escalating Supervisor to System 2")
                        router_result = await asyncio.wait_for(
                            self.cognitive_router.route_to_system_2(messages),
                            timeout=30.0
                        )
                        # Propagate non-ok results from System 2 as well
                        if router_result.status != "ok":
                            state[_BLOCKED_KEY] = router_result
                            return state
                        response = router_result.content
                    except Exception as s2_err:
                        logger.warning(f"System 2 also failed in supervisor ({s2_err}).")
                        response = None

            if not response or response.startswith("[System 1 - Error]") or response.startswith("[System 2"):
                logger.warning(f"Supervisor received error: {response}")
                state["current_plan"] = []
                if not self.cognitive_router.get_system_2_available():
                    state["final_response"] = (
                        "I was unable to process this request locally. This task likely requires "
                        "capabilities beyond my local model (e.g. internet access or complex reasoning). "
                        "To enable this, configure a System 2 provider by setting GROQ_API_KEY in your .env file. "
                        "Groq offers a free tier at console.groq.com."
                    )
                else:
                    state["final_response"] = "Both local and cloud reasoning failed on this request. Please try rephrasing or simplifying the task."
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
        core_mem_str = await self.core_memory.get_context_string()

        capabilities_str = self._get_capabilities_string()
        messages = [
            {"role": "system", "content": f"You are the Research Agent. Search archival memory for relevant context and provide findings.\n{self.charter_text}\n{core_mem_str}\n\n{capabilities_str}"},
            {"role": "user", "content": f"<user_input>{state['user_input']}</user_input>"}
        ]
        try:
            router_result = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=["search_archival_memory"]),
                timeout=60.0
            )
            if router_result.status != "ok":
                state[_BLOCKED_KEY] = router_result
            else:
                state["worker_outputs"]["research"] = router_result.content
        except Exception as e:
            logger.error(f"Research Agent failed: {e}")
            state["worker_outputs"]["research"] = f"Error: {e}"

        return state

    async def coder_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Worker: executes coding tasks, updates memory/ledger."""
        state = await self._deduct_energy(state, ENERGY_COST_WORKER, "coder_agent")
        core_mem_str = await self.core_memory.get_context_string()
        research_context = state["worker_outputs"].get("research", "")

        content = f"User Input: {state['user_input']}\n"
        if research_context:
            content += f"Research Context: {research_context}\n"

        capabilities_str = self._get_capabilities_string()
        messages = [
            {"role": "system", "content": f"You are the Coder Agent. Execute coding tasks and update memory as needed.\n{self.charter_text}\n{core_mem_str}\n\n{capabilities_str}"},
            {"role": "user", "content": f"<user_input>{content}</user_input>"}
        ]
        try:
            router_result = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=["update_ledger", "update_core_memory", "request_core_update", "spawn_new_objective", "update_objective_status"]),
                timeout=60.0
            )
            if router_result.status != "ok":
                state[_BLOCKED_KEY] = router_result
            else:
                state["worker_outputs"]["coder"] = router_result.content
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
                router_result = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_2(messages),
                    timeout=30.0
                )
            else:
                router_result = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_1(messages, allowed_tools=[]),
                    timeout=60.0
                )
            # Non-ok router results (e.g. HITL during critic) are propagated via state
            if router_result.status != "ok":
                state[_BLOCKED_KEY] = router_result
                return state
            response = router_result.content
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
            # Sprint 11: inject a high-priority debug Task into the backlog
            try:
                last_worker = list(state["worker_outputs"].keys())[-1] if state.get("worker_outputs") else "unknown"
                task_title = (
                    f"Debug failing [{last_worker}] logic for: "
                    f"{state.get('user_input', '')[:80]}"
                )
                await self.ledger_memory.add_objective(
                    tier="Task",
                    title=task_title,
                    estimated_energy=20,
                    origin="Critic",
                    priority=2,
                )
                logger.info(f"Critic: injected debug Task into backlog after 3 failures: {task_title!r}")
            except Exception as spawn_err:
                logger.warning(f"Critic: failed to spawn debug objective: {spawn_err}")
            raise RequiresHITLError(
                f"Critic rejected output 3 times. Guidance needed.\n"
                f"Question: How should I proceed to satisfy the charter?"
            )

        return state

    async def _consolidate_memory(self, user_id: str) -> None:
        """
        Memory Consolidation Skill: summarise the last 10 chat turns for
        *user_id* into a single paragraph, update core_memory under
        ``conversation_summary``, and trim the stored raw turns to the most
        recent 5.  Runs fire-and-forget; errors are swallowed.
        """
        try:
            turns = await self.ledger_memory.get_chat_history(user_id, limit=10)
            if len(turns) < 10:
                return  # Not enough turns to warrant consolidation

            convo_text = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)
            messages = [
                {"role": "system", "content": "You are a memory consolidation engine. Summarise the conversation concisely in 2–3 sentences."},
                {"role": "user", "content": convo_text},
            ]
            result = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages, allowed_tools=[]),
                timeout=30.0,
            )
            if result.status == "ok" and result.content:
                await self.core_memory.update("conversation_summary", result.content.strip())
                logger.info(f"Memory consolidation complete for user {user_id}")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {e}")

    async def process_message(self, user_message: str, user_id: str) -> str:
        """Main entry point: State Graph execution with Energy Budget."""
        if not user_message:
            return "Error: Invalid message"

        # MFA resumption
        if user_id in self.pending_mfa:
            pending_tool = self.pending_mfa[user_id]
            if verify_mfa_challenge(user_message):
                del self.pending_mfa[user_id]
                exec_result = await self.cognitive_router._execute_tool(
                    pending_tool["name"],
                    pending_tool["arguments"]
                )
                # _execute_tool returns RouterResult; extract content
                if exec_result.status == "ok":
                    return exec_result.content
                # If another security gate fires during MFA execution, surface it
                return self._handle_blocked_result(exec_result, pending_tool.get("user_id", user_id), {})
            else:
                del self.pending_mfa[user_id]
                return "Error: MFA authorization failed. Action aborted."

        # Tool approval resumption (Admin replies YES/NO to a synthesis proposal)
        if user_id in self.pending_tool_approval:
            payload = self.pending_tool_approval.pop(user_id)
            synthesis = payload["synthesis"]
            original_state = payload["original_state"]

            if user_message.strip().upper().startswith("YES"):
                tool_name = synthesis["tool_name"]
                try:
                    # 1. Register in runtime (includes AST sandbox)
                    self.cognitive_router.register_dynamic_tool(
                        tool_name, synthesis["code"], synthesis["schema_json"]
                    )
                    # 2. Persist to DB
                    await self.ledger_memory.register_tool(
                        name=tool_name,
                        description=synthesis["description"],
                        code=synthesis["code"],
                        schema_json=synthesis["schema_json"],
                    )
                    await self.ledger_memory.approve_tool(tool_name)
                    # 3. Log capability to core memory
                    core = await self.core_memory.get_all()
                    caps = core.get("known_capabilities", "")
                    updated_caps = f"{caps}, {tool_name}".lstrip(", ")
                    await self.core_memory.update("known_capabilities", updated_caps)
                    logger.info(f"Tool '{tool_name}' approved, registered, and logged to core memory")
                    # 4. Retry original query with the new tool available
                    retry_response = await self.process_message(
                        original_state["user_input"], user_id
                    )
                    return f"✅ Tool '{tool_name}' deployed.\n\n{retry_response}"
                except Exception as e:
                    logger.error(f"Tool registration failed: {e}", exc_info=True)
                    return f"Error deploying tool '{tool_name}': {e}"
            else:
                logger.info(f"Admin rejected tool synthesis for user {user_id}")
                return f"Tool proposal rejected. The capability gap remains: {synthesis['description']}"

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
                    state["chat_history"] = await self.ledger_memory.get_chat_history(user_id, limit=5)
                except Exception as e:
                    logger.warning(f"Failed to load chat history for {user_id}: {e}")

        try:
            max_iterations = 3

            while state["iteration_count"] < max_iterations:

                # Check for a blocked result set by a previous node
                if state.get(_BLOCKED_KEY):
                    return self._handle_blocked_result(
                        state.pop(_BLOCKED_KEY), user_id, state
                    )

                # 1. Supervisor (first iteration only, or after HITL guidance)
                if not state["current_plan"] and state["iteration_count"] == 0:
                    state = await self.supervisor_node(state)
                else:
                    state["user_input"] += f"\n[CRITIC FEEDBACK: {state['critic_feedback']}. Fix your output.]"

                # Check for blocked result after supervisor
                if state.get(_BLOCKED_KEY):
                    return self._handle_blocked_result(
                        state.pop(_BLOCKED_KEY), user_id, state
                    )

                # If supervisor handled it directly (no workers), skip critic to reduce latency
                if not state["current_plan"] and state.get("final_response"):
                    break

                # 2. Workers
                for task in state["current_plan"]:
                    if "research" in task.lower():
                        state = await self.research_agent(state)
                    elif "coder" in task.lower():
                        state = await self.coder_agent(state)
                    # Check after each worker
                    if state.get(_BLOCKED_KEY):
                        return self._handle_blocked_result(
                            state.pop(_BLOCKED_KEY), user_id, state
                        )

                if not state["current_plan"] and not state.get("final_response"):
                    state["final_response"] = "No response could be generated. Please try again."
                    break

                # 3. Critic (only when workers were called)
                state = await self.critic_node(state)

                if state.get(_BLOCKED_KEY):
                    return self._handle_blocked_result(
                        state.pop(_BLOCKED_KEY), user_id, state
                    )

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

            # Persist this conversation turn to chat history (skip heartbeat tasks and error responses)
            _error_prefixes = (
                "Supervisor encountered an error",
                "Both local and cloud reasoning failed",
                "I was unable to process this request locally",
                "Planning timed out",
                "Unable to fulfill this request",
                "No valid response could be generated",
                "An internal error occurred",
            )
            is_error_response = any(final_resp.startswith(p) for p in _error_prefixes)

            if user_id != "heartbeat" and not is_error_response:
                try:
                    await self.ledger_memory.save_chat_turn(user_id, "user", user_message)
                    await self.ledger_memory.save_chat_turn(user_id, "assistant", final_resp)
                except Exception as e:
                    logger.warning(f"Failed to save chat turn for {user_id}: {e}")

                # Fire-and-forget memory consolidation and vector store write
                asyncio.create_task(self._consolidate_memory(user_id))

            # Vector memory storage (non-blocking fire-and-forget)
            asyncio.create_task(
                self._save_memory_async(f"User: {user_message}\nAssistant: {final_resp}")
            )

            return final_resp

        except RequiresHITLError as hitl_err:
            self.pending_hitl_state[user_id] = state
            return str(hitl_err)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            return "An internal error occurred."

    def _handle_blocked_result(
        self,
        result: RouterResult,
        user_id: str,
        state: Dict[str, Any],
    ) -> str:
        """
        Convert a non-ok RouterResult into the appropriate user-facing string,
        registering any pending MFA / HITL / tool-synthesis state as needed.
        """
        if result.status == "mfa_required":
            self.pending_mfa[user_id] = {
                "name": result.mfa_tool_name,
                "arguments": result.mfa_arguments,
            }
            return "SECURITY LOCK: To authorize this core change, complete the phrase: 'The sky is...'"

        if result.status == "hitl_required":
            self.pending_hitl_state[user_id] = state
            return result.hitl_message

        if result.status == "capability_gap":
            # Run synthesis synchronously via create_task — result will come back
            # in a follow-up message after the Admin approves/rejects.
            asyncio.create_task(self._async_tool_synthesis(user_id, result, state))
            return (
                f"I identified a capability gap: {result.gap_description}. "
                f"Requesting tool synthesis from System 2..."
            )

        # Fallback
        return f"An unexpected router status was received: {result.status}"

    async def _async_tool_synthesis(
        self,
        user_id: str,
        result: RouterResult,
        state: Dict[str, Any],
    ) -> None:
        """Background task: run tool synthesis and send the HITL prompt to admin."""
        try:
            hitl_prompt = await self.tool_synthesis_node(state, result)
            if self.outbound_queue is not None:
                await self.outbound_queue.put(hitl_prompt)
        except Exception as e:
            logger.error(f"Background tool synthesis failed: {e}", exc_info=True)

    def close(self) -> None:
        try:
            if hasattr(self, 'vector_memory') and self.vector_memory:
                self.vector_memory.close()
            if hasattr(self, 'cognitive_router') and self.cognitive_router:
                # Schedule async close — best-effort at shutdown time
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.cognitive_router.close())
                        loop.create_task(self.ledger_memory.close())
                    else:
                        loop.run_until_complete(self.cognitive_router.close())
                        loop.run_until_complete(self.ledger_memory.close())
                except Exception:
                    pass
            logger.info("Orchestrator resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing Orchestrator: {e}", exc_info=True)
