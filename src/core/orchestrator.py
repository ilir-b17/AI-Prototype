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
MEMORY_SAVE_THRESHOLD = int(os.getenv("MEMORY_SAVE_THRESHOLD", "120"))
MEMORY_CONSOLIDATION_INTERVAL = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "21600"))  # 6 hours

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
        local_model: str = "gemma4:26b"
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
            # Background task registry — holds strong references to prevent GC (ISSUE-002)
            self._background_tasks: set = set()
            # Per-user async locks to serialise concurrent messages (ISSUE-012)
            self._user_locks: Dict[str, asyncio.Lock] = {}
            self._user_locks_lock: asyncio.Lock = asyncio.Lock()
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
        await self._load_pending_approvals()
        await self._load_pending_hitl()
        # Record the host OS now that we're in an async context
        await self.core_memory.update("host_os", platform.system())
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
            "cwd": os.getcwd(),
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
            f"CPU: {s.get('cpu_usage', 'unknown')} | "
            f"CWD: {s.get('cwd', 'unknown')}]"
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

    @staticmethod
    def _is_system_1_error(result: Optional[RouterResult]) -> bool:
        """Detect System 1 error payloads that should trigger System 2 fallback."""
        return bool(
            result
            and result.status == "ok"
            and isinstance(result.content, str)
            and result.content.startswith("[System 1 - Error]")
        )

    async def _get_archival_context(self, query: str) -> str:
        """Retrieve top archival memory snippets relevant to the user query."""
        if not query:
            return ""
        try:
            results = await self.vector_memory.query_memory_async(query, n_results=3)
            if not results:
                return ""
            lines = ["<Archival_Memory>"]
            for item in results:
                snippet = item.get("document", "")
                if snippet:
                    lines.append(f"  <Memory>{snippet}</Memory>")
            lines.append("</Archival_Memory>")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Archival memory lookup failed: {e}")
            return ""

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
            "hitl_count": 0,
        }

    def _deduct_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        """Deduct energy and raise HITL if exhausted."""
        state["energy_remaining"] = state.get("energy_remaining", 100) - amount
        logger.debug(f"Energy -{amount} ({reason}). Remaining: {state['energy_remaining']}")
        if state["energy_remaining"] <= 0:
            raise RequiresHITLError(
                f"Energy Budget Exhausted: The system consumed all cognitive energy on this task.\n"
                f"Question: How should I prioritize the remaining work for: '{state.get('user_input', '')}'"
            )
        return state

    @staticmethod
    def _get_output_to_evaluate(state: Dict[str, Any]) -> str:
        """Return the most recent worker output, or the supervisor's final response."""
        if state.get("worker_outputs"):
            last_worker = list(state["worker_outputs"].keys())[-1]
            return state["worker_outputs"][last_worker]
        return state.get("final_response", "")

    async def _spawn_debug_task(self, state: Dict[str, Any]) -> None:
        """Inject a high-priority debug Task into the backlog after 3 critic failures."""
        try:
            last_worker = list(state["worker_outputs"].keys())[-1] if state.get("worker_outputs") else "unknown"
            task_title = f"Debug failing [{last_worker}] logic for: {state.get('user_input', '')[:80]}"
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

    async def _route_supervisor_request(self, messages: List[Dict]) -> Optional["RouterResult"]:
        """Try System 1, fall back to System 2.  Returns None on total failure."""
        router_result: Optional[RouterResult] = None
        try:
            logger.info("Routing Supervisor through System 1 (Local Model)")
            router_result = await asyncio.wait_for(
                self.cognitive_router.route_to_system_1(messages),
                timeout=150.0,
            )
            if self._is_system_1_error(router_result):
                logger.warning("System 1 returned an error payload in supervisor; escalating to System 2.")
                router_result = None
        except asyncio.TimeoutError:
            logger.error("System 1 timed out in supervisor (150 s). Escalating to System 2.", exc_info=True)
        except Exception as s1_err:
            logger.error(f"System 1 raised an exception in supervisor: {s1_err!r}. Escalating.", exc_info=True)

        if router_result is not None:
            return router_result

        if not self.cognitive_router.get_system_2_available():
            return None

        try:
            logger.info("Escalating Supervisor to System 2")
            router_result = await asyncio.wait_for(
                self.cognitive_router.route_to_system_2(messages),
                timeout=60.0,
            )
            return router_result
        except asyncio.TimeoutError:
            logger.error("System 2 timed out in supervisor (60 s).", exc_info=True)
        except Exception as s2_err:
            logger.error(f"System 2 raised an exception in supervisor: {s2_err!r}.", exc_info=True)
        return None

    def _parse_supervisor_response(self, response: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the WORKERS tag from *response* and update *state* in-place."""
        lines = response.strip().split("\n")
        workers_match = None
        for line in reversed(lines[-3:]):
            workers_match = re.search(r"WORKERS:\s*(\[[^\]]*\])", line)
            if workers_match:
                break

        if not workers_match:
            state["current_plan"] = []
            state["final_response"] = response.strip()
            return state

        try:
            plan = json.loads(workers_match.group(1))
            answer = "\n".join(ln for ln in lines if not re.search(r"WORKERS:\s*\[", ln)).strip()
            if isinstance(plan, list) and plan and all(isinstance(x, str) for x in plan):
                state["current_plan"] = plan
                if answer:
                    state["worker_outputs"]["supervisor_context"] = answer
            else:
                state["current_plan"] = []
                state["final_response"] = answer or response.strip()
        except ValueError:
            state["current_plan"] = []
            answer = "\n".join(ln for ln in lines if not re.search(r"WORKERS:\s*\[", ln)).strip()
            state["final_response"] = answer or response.strip()
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

        An asyncio.Lock prevents heartbeat cycles from overlapping: if the
        previous run is still in progress when the next interval fires, the
        new cycle is skipped (ISSUE-004).  A hard timeout of 90% of the
        heartbeat interval caps individual task execution time.
        """
        logger.info("Heartbeat loop started.")
        _heartbeat_lock = asyncio.Lock()
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if _heartbeat_lock.locked():
                logger.warning("Heartbeat: Previous run still in progress. Skipping this cycle.")
                continue
            async with _heartbeat_lock:
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

                    try:
                        result = await asyncio.wait_for(
                            self.process_message(
                                user_message=(
                                    f"[HEARTBEAT TASK #{task['id']}]: {task['title']}\n"
                                    f"You MUST execute this task right now by calling the appropriate tools. "
                                    f"Do NOT describe what you plan to do — use your tools and report the actual results. "
                                    f"If the task requires storing data, call update_core_memory or update_ledger explicitly. "
                                    f"If you cannot complete it, explain exactly why."
                                ),
                                user_id="heartbeat"
                            ),
                            timeout=HEARTBEAT_INTERVAL * 0.9,  # 27-minute hard cap
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Heartbeat: Task #{task['id']} timed out after "
                            f"{int(HEARTBEAT_INTERVAL * 0.9)}s. Marking pending."
                        )
                        await self.ledger_memory.update_objective_status(task["id"], "pending")
                        continue

                    # Only mark completed if the result isn't an error or refusal
                    status = "completed"
                    if any(w in result.lower() for w in ("error", "cannot", "unable", "failed", "i cannot", "i am unable")):
                        status = "pending"
                        logger.warning(f"Heartbeat: Task #{task['id']} may not have completed — marking pending. Result: {result[:100]}")
                    await self.ledger_memory.update_objective_status(task["id"], status)

                    summary = (
                        f"Heartbeat completed task #{task['id']}:\n"
                        f"  Task: {task['title']}\n"
                        f"  Result: {result[:200]}"
                    )
                    await self._notify_admin(summary)
                    logger.info(f"Heartbeat: Task #{task['id']} completed.")

                except Exception as e:
                    logger.error(f"Heartbeat error: {e}", exc_info=True)

    async def _memory_consolidation_loop(self) -> None:
        """Background task: periodically consolidate chat history into long-term memory."""
        logger.info("Memory consolidation loop started.")
        while True:
            await asyncio.sleep(MEMORY_CONSOLIDATION_INTERVAL)
            try:
                user_ids = await self.ledger_memory.get_recent_user_ids(limit=20)
                if not user_ids:
                    continue
                for user_id in user_ids:
                    await self._consolidate_memory(user_id)
            except Exception as e:
                logger.warning(f"Memory consolidation loop error: {e}", exc_info=True)

    async def _save_memory_async(self, text: str) -> None:
        """Fire-and-forget memory storage gated by response length.

        Skips trivial exchanges (short greetings, one-liners).
        Threshold: combined user+assistant text must exceed MEMORY_SAVE_THRESHOLD.
        Uses asyncio.to_thread() so the blocking ChromaDB call does not
        stall the event loop.
        """
        try:
            if len(text) < MEMORY_SAVE_THRESHOLD:
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

    async def _load_pending_approvals(self) -> None:
        """Reload any tool synthesis proposals that were pending when the bot last stopped."""
        try:
            pending = await self.ledger_memory.load_pending_approvals()
            for user_id, payload in pending.items():
                self.pending_tool_approval[user_id] = {
                    "synthesis": payload["synthesis"],
                    "original_state": {"user_input": payload["original_input"], "user_id": user_id},
                }
            if pending:
                logger.info(f"Restored {len(pending)} pending tool approval(s) from DB")
        except Exception as e:
            logger.warning(f"_load_pending_approvals failed: {e}")

    async def _load_pending_hitl(self) -> None:
        """Reload any HITL states that were persisted before the last shutdown (ISSUE-013)."""
        try:
            states = await self.ledger_memory.load_hitl_states()
            for user_id, state in states.items():
                self.pending_hitl_state[user_id] = state
            if states:
                logger.info(f"Restored {len(states)} pending HITL state(s) from DB")
                # Notify admin so they know the context survived the restart
                await self._notify_admin(
                    f"⚠️ AIDEN restarted with {len(states)} pending HITL workflow(s) restored. "
                    f"Affected user(s): {', '.join(states.keys())}. Reply to continue."
                )
        except Exception as e:
            logger.warning(f"_load_pending_hitl failed: {e}")

    def _fire_and_forget(self, coro) -> asyncio.Task:
        """Schedule a coroutine as a background task with a strong GC-safe reference (ISSUE-002).

        The task is added to ``_background_tasks`` and automatically removed
        when it completes, so the set never grows unboundedly while still
        preventing the garbage collector from destroying mid-flight tasks.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Return the per-user asyncio.Lock, creating it lazily (ISSUE-012)."""
        async with self._user_locks_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

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

        # Store payload for approval resumption (in-memory + persisted to DB)
        self.pending_tool_approval[user_id] = {
            "synthesis": synthesis,
            "original_state": state,
        }
        await self.ledger_memory.save_pending_approval(
            user_id, synthesis, state["user_input"]
        )

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
        state = self._deduct_energy(state, ENERGY_COST_SUPERVISOR, "supervisor")
        user_input = state["user_input"]
        core_mem_str = await self.core_memory.get_context_string()
        archival_context = await self._get_archival_context(user_input)
        capabilities_str = self._get_capabilities_string()
        archival_block = f"{archival_context}\n\n" if archival_context else ""

        system_prompt = (
            f"You are AIDEN — a local autonomous AI agent on the Admin's machine (Ollama). "
            f"You are NOT ChatGPT/Claude/Gemini. "
            f"You have persistent memory: the chat history above is real (SQLite). "
            f"You have tools — never deny capabilities listed below. "
            f"Reply in plain conversational text, no markdown headers. "
            f"Private reasoning: wrap in <think>...</think> — it will be stripped.\n\n"
            f"CRITICAL RULES:\n"
            f"- NEVER fabricate or simulate tool results. If a tool returns an error, report the exact error to the user honestly. Do NOT invent summaries, page counts, or any content that the tool did not actually return.\n"
            f"- If you cannot complete a task due to a tool error, say so clearly and ask the user for help.\n\n"
            f"FILE ACCESS: The Admin's downloads folder for you to read files from is: "
            f"C:\\Users\\iboci\\Live-Trading-bot\\AI_Prototype\\downloads\\\n\n"
            f"PDF RULE: If the user asks to read or summarize a PDF, you MUST call "
            f"`extract_pdf_text` first (use the downloads folder if only a filename is given).\n\n"
            f"WEB RULE: After `web_search`, call `extract_web_article` to read the chosen URL before summarizing.\n\n"
            f"DATA RULE: For CSV/Excel analysis, use `analyze_table_file` instead of reading raw text.\n\n"
            f"{self._get_sensory_context()}\n\n"
            f"OS CONTEXT: You are running on {self.sensory_state.get('os', platform.system())}. "
            f"Use OS-appropriate shell commands (e.g., 'dir' instead of 'ls' on Windows). "
            f"Preferentially use your `manage_file_system` Python tool for OS-agnostic file exploration.\n\n"
            f"{self.charter_text}\n{core_mem_str}\n\n"
            f"{archival_block}"
            f"{capabilities_str}\n\n"
            f"Respond to the user, then on the very last line declare which workers are needed.\n"
            f'Format: WORKERS: [] for chat, WORKERS: ["research_agent"] or ["coder_agent"] for tasks.'
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.get("chat_history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"<user_input>{user_input}</user_input>"})

        try:
            router_result = await self._route_supervisor_request(messages)

            if router_result is not None and router_result.status != "ok":
                state[_BLOCKED_KEY] = router_result
                return state

            response = router_result.content if router_result else None

            if not response or response.startswith("[System 1 - Error]") or response.startswith("[System 2"):
                logger.warning(f"Supervisor received error or no response: {response!r}")
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

            state = self._parse_supervisor_response(response, state)

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
        state = self._deduct_energy(state, ENERGY_COST_WORKER, "research_agent")
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
            if self._is_system_1_error(router_result):
                router_result = None
        except Exception as e:
            logger.error(f"Research Agent failed on System 1: {e}")
            router_result = None

        attempted_system_2 = False
        if router_result is None and self.cognitive_router.get_system_2_available():
            attempted_system_2 = True
            try:
                router_result = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_2(
                        messages, allowed_tools=["search_archival_memory"]
                    ),
                    timeout=60.0
                )
            except Exception as e:
                logger.error(f"Research Agent failed on System 2: {e}")
                router_result = None

        if router_result is None:
            if attempted_system_2:
                state["worker_outputs"]["research"] = "Error: Research agent failed after System 1 error and System 2 fallback."
            else:
                state["worker_outputs"]["research"] = "Error: Research agent failed and System 2 is not configured."
            return state

        if router_result.status != "ok":
            state[_BLOCKED_KEY] = router_result
        else:
            state["worker_outputs"]["research"] = router_result.content

        return state

    async def coder_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Worker: executes coding tasks, updates memory/ledger."""
        state = self._deduct_energy(state, ENERGY_COST_WORKER, "coder_agent")
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
                self.cognitive_router.route_to_system_1(messages, allowed_tools=["update_ledger", "update_core_memory", "request_core_update", "spawn_new_objective", "update_objective_status", "extract_pdf_text"]),
                timeout=60.0
            )
            if self._is_system_1_error(router_result):
                router_result = None
        except Exception as e:
            logger.error(f"Coder Agent failed on System 1: {e}")
            router_result = None

        attempted_system_2 = False
        if router_result is None and self.cognitive_router.get_system_2_available():
            attempted_system_2 = True
            try:
                router_result = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_2(
                        messages,
                        allowed_tools=["update_ledger", "update_core_memory", "request_core_update", "spawn_new_objective", "update_objective_status", "extract_pdf_text"]
                    ),
                    timeout=60.0
                )
            except Exception as e:
                logger.error(f"Coder Agent failed on System 2: {e}")
                router_result = None

        if router_result is None:
            if attempted_system_2:
                state["worker_outputs"]["coder"] = "Error: Coder agent failed after System 1 error and System 2 fallback."
            else:
                state["worker_outputs"]["coder"] = "Error: Coder agent failed and System 2 is not configured."
            return state

        if router_result.status != "ok":
            state[_BLOCKED_KEY] = router_result
        else:
            state["worker_outputs"]["coder"] = router_result.content

        return state

    async def critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Checks worker output against the charter. Skipped for direct supervisor responses."""
        state = self._deduct_energy(state, ENERGY_COST_CRITIC, "critic")
        output_to_eval = self._get_output_to_evaluate(state)

        if not output_to_eval:
            state["critic_feedback"] = "PASS"
            return state

        messages = [
            {"role": "system", "content": f"Evaluate the output. Output PASS if it follows the charter, or FAIL with a reason.\nCharter: {self.charter_text}"},
            {"role": "user", "content": output_to_eval[:1000]},
        ]

        try:
            if self.cognitive_router.get_system_2_available():
                logger.info("Routing Critic through System 2 (Gemini)")
                router_result = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_2(messages), timeout=30.0
                )
            else:
                router_result = await asyncio.wait_for(
                    self.cognitive_router.route_to_system_1(messages, allowed_tools=[]), timeout=60.0
                )
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
            await self._spawn_debug_task(state)
            raise RequiresHITLError(
                "Critic rejected output 3 times. Guidance needed.\n"
                "Question: How should I proceed to satisfy the charter?"
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
                timeout=90.0,
            )
            if result.status == "ok" and result.content:
                summary = result.content.strip()
                await self.core_memory.update("conversation_summary", summary)
                await self.ledger_memory.trim_chat_history(user_id, keep_last=5)
                await self.vector_memory.add_memory_async(
                    text=f"Conversation summary ({user_id}): {summary}",
                    metadata={"type": "conversation_summary", "user_id": user_id, "timestamp": datetime.now().isoformat()}
                )
                logger.info(f"Memory consolidation complete for user {user_id}")
        except asyncio.TimeoutError:
            logger.warning("Memory consolidation skipped: LLM did not respond within 90s")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {type(e).__name__}: {e}", exc_info=True)

    async def _try_resume_mfa(self, user_id: str, user_message: str) -> Optional[str]:
        """Handle MFA challenge response. Returns a reply string, or None if not in MFA flow."""
        if user_id not in self.pending_mfa:
            return None
        pending_tool = self.pending_mfa[user_id]
        del self.pending_mfa[user_id]
        if not verify_mfa_challenge(user_message):
            return "Error: MFA authorization failed. Action aborted."
        exec_result = await self.cognitive_router._execute_tool(
            pending_tool["name"], pending_tool["arguments"]
        )
        if exec_result.status == "ok":
            return exec_result.content
        return await self._handle_blocked_result(exec_result, pending_tool.get("user_id", user_id), {})

    async def _try_resume_tool_approval(self, user_id: str, user_message: str) -> Optional[str]:
        """Handle YES/NO tool synthesis approval. Returns a reply string, or None if not pending."""
        if user_id not in self.pending_tool_approval:
            return None
        payload = self.pending_tool_approval.pop(user_id)
        await self.ledger_memory.clear_pending_approval(user_id)
        synthesis = payload["synthesis"]
        original_state = payload["original_state"]
        if not user_message.strip().upper().startswith("YES"):
            logger.info(f"Admin rejected tool synthesis for user {user_id}")
            return f"Tool proposal rejected. The capability gap remains: {synthesis['description']}"
        tool_name = synthesis["tool_name"]
        try:
            self.cognitive_router.register_dynamic_tool(tool_name, synthesis["code"], synthesis["schema_json"])
            await self.ledger_memory.register_tool(
                name=tool_name, description=synthesis["description"],
                code=synthesis["code"], schema_json=synthesis["schema_json"],
            )
            await self.ledger_memory.approve_tool(tool_name)
            core = await self.core_memory.get_all()
            caps = core.get("known_capabilities", "")
            await self.core_memory.update("known_capabilities", f"{caps}, {tool_name}".lstrip(", "))
            logger.info(f"Tool '{tool_name}' approved, registered, and logged to core memory")
            retry = await self.process_message(original_state["user_input"], user_id)
            return f"✅ Tool '{tool_name}' deployed.\n\n{retry}"
        except Exception as e:
            logger.error(f"Tool registration failed: {e}", exc_info=True)
            return f"Error deploying tool '{tool_name}': {e}"

    async def _load_state(self, user_id: str, user_message: str) -> Dict[str, Any]:
        """Return a state dict: resumes a HITL conversation if pending, else creates fresh."""
        if user_id in self.pending_hitl_state:
            state = self.pending_hitl_state.pop(user_id)
            # Clear the persisted DB record now that we've resumed (ISSUE-013)
            try:
                await self.ledger_memory.clear_hitl_state(user_id)
            except Exception as e:
                logger.warning(f"Failed to clear HITL state from DB for {user_id}: {e}")
            state["admin_guidance"] = user_message
            state["user_input"] += f"\n[ADMIN GUIDANCE: {user_message}]"
            state["iteration_count"] = 0
            state["current_plan"] = []
            # Track how many HITL cycles have been spent on this task (ISSUE-005)
            state["hitl_count"] = state.get("hitl_count", 0) + 1
            # Cap recharge to 75 so energy budget is never fully restored — this
            # ensures a perpetually failing task eventually becomes unrecoverable
            # rather than cycling indefinitely at full energy (ISSUE-005).
            state["energy_remaining"] = min(state.get("energy_remaining", 0) + 50, 75)
            # If the admin has guided this task 3+ times without success, abandon it
            if state["hitl_count"] >= 3:
                logger.warning(
                    f"HITL cycle limit (3) reached for user {user_id}. "
                    f"Abandoning task: {state.get('user_input', '')[:80]}"
                )
                state["final_response"] = (
                    "This task has been attempted 3 times with admin guidance and "
                    "could not be completed. The request has been abandoned to prevent "
                    "an infinite loop. Please rephrase or break it into smaller steps."
                )
            return state
        state = self._new_state(user_id, user_message)
        if user_id != "heartbeat":
            try:
                state["chat_history"] = await self.ledger_memory.get_chat_history(user_id, limit=5)
            except Exception as e:
                logger.warning(f"Failed to load chat history for {user_id}: {e}")
        return state

    async def _run_graph_loop(self, state: Dict[str, Any], user_id: str, user_message: str) -> str:
        """Execute the supervisor → workers → critic loop. Returns the final sanitized response."""
        max_iterations = 3

        # Fast-exit if the state already carries a final response before we enter
        # the loop — e.g. when the HITL cycle limit has been reached (ISSUE-005).
        if state.get("final_response") and not state.get("current_plan"):
            return self.cognitive_router.sanitize_response(state["final_response"])

        while state["iteration_count"] < max_iterations:
            if state.get(_BLOCKED_KEY):
                return await self._handle_blocked_result(state.pop(_BLOCKED_KEY), user_id, state)

            if not state["current_plan"] and state["iteration_count"] == 0:
                state = await self.supervisor_node(state)
            else:
                state["user_input"] += f"\n[CRITIC FEEDBACK: {state['critic_feedback']}. Fix your output.]"

            if state.get(_BLOCKED_KEY):
                return await self._handle_blocked_result(state.pop(_BLOCKED_KEY), user_id, state)

            if not state["current_plan"] and state.get("final_response"):
                break

            for task in state["current_plan"]:
                if "research" in task.lower():
                    state = await self.research_agent(state)
                elif "coder" in task.lower():
                    state = await self.coder_agent(state)
                if state.get(_BLOCKED_KEY):
                    return await self._handle_blocked_result(state.pop(_BLOCKED_KEY), user_id, state)

            if not state["current_plan"] and not state.get("final_response"):
                state["final_response"] = "No response could be generated. Please try again."
                break

            state = await self.critic_node(state)

            if state.get(_BLOCKED_KEY):
                return await self._handle_blocked_result(state.pop(_BLOCKED_KEY), user_id, state)

            if state["critic_feedback"] == "PASS":
                break
            logger.warning(f"Critic rejected output on iteration {state['iteration_count']}")
            state["final_response"] = ""
            state["worker_outputs"] = {}
            # Clear the stale plan so the supervisor re-plans with the Critic
            # feedback injected into user_input (ISSUE-006).
            state["current_plan"] = []

        if state["iteration_count"] >= max_iterations and state.get("critic_feedback") != "PASS":
            state["final_response"] = "Unable to fulfill this request — output repeatedly failed internal safety checks."

        if not state.get("final_response"):
            state["final_response"] = "No valid response could be generated."

        final_resp = self.cognitive_router.sanitize_response(state["final_response"])

        _error_prefixes = (
            "Supervisor encountered an error", "Both local and cloud reasoning failed",
            "I was unable to process this request locally", "Planning timed out",
            "Unable to fulfill this request", "No valid response could be generated",
            "An internal error occurred",
        )
        is_error_response = any(final_resp.startswith(p) for p in _error_prefixes)

        if user_id != "heartbeat" and not is_error_response:
            try:
                await self.ledger_memory.save_chat_turn(user_id, "user", user_message)
                await self.ledger_memory.save_chat_turn(user_id, "assistant", final_resp)
            except Exception as e:
                logger.warning(f"Failed to save chat turn for {user_id}: {e}")
            # Use _fire_and_forget to keep a strong reference preventing GC (ISSUE-002)
            self._fire_and_forget(self._consolidate_memory(user_id))

        # Use _fire_and_forget to keep a strong reference preventing GC (ISSUE-002)
        self._fire_and_forget(
            self._save_memory_async(f"User: {user_message}\nAssistant: {final_resp}")
        )
        return final_resp

    async def process_message(self, user_message: str, user_id: str) -> str:
        """Main entry point: State Graph execution with Energy Budget."""
        if not user_message:
            return "Error: Invalid message"

        # Serialise concurrent messages for the same user_id to prevent
        # race conditions on pending_mfa / pending_hitl_state / pending_tool_approval
        # dicts (ISSUE-012).  Different users are still processed concurrently.
        lock = await self._get_user_lock(user_id)
        async with lock:
            reply = await self._try_resume_mfa(user_id, user_message)
            if reply is not None:
                return reply

            reply = await self._try_resume_tool_approval(user_id, user_message)
            if reply is not None:
                return reply

            state = await self._load_state(user_id, user_message)

            try:
                return await self._run_graph_loop(state, user_id, user_message)
            except RequiresHITLError as hitl_err:
                self.pending_hitl_state[user_id] = state
                # Persist so the state survives a bot restart (ISSUE-013)
                self._fire_and_forget(self.ledger_memory.save_hitl_state(user_id, state))
                return str(hitl_err)
            except Exception as e:
                logger.error(f"Graph execution failed: {e}", exc_info=True)
                return "An internal error occurred."

    async def _handle_blocked_result(
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
            # Persist so the HITL state survives a bot restart (ISSUE-013)
            self._fire_and_forget(self.ledger_memory.save_hitl_state(user_id, state))
            return result.hitl_message

        if result.status == "capability_gap":
            # Run synthesis in the background via _fire_and_forget which holds
            # a strong GC-safe reference (ISSUE-002).
            self._fire_and_forget(self._async_tool_synthesis(user_id, result, state))
            return (
                f"I identified a capability gap: {result.gap_description}. "
                f"Requesting tool synthesis from System 2..."
            )

        # Fallback
        return f"An unexpected router status was received: {result.status}"

    async def _async_tool_synthesis(
        self,
        _user_id: str,
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
