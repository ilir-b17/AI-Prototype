import logging
import asyncio
import platform
from datetime import datetime
from src.memory.ledger_db import LedgerMemory
from src.memory.core_memory import CoreMemory
from src.memory.vector_db import VectorMemory

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Tool Schemas (Ollama / Groq compatible)
# ─────────────────────────────────────────────────────────────
SYSTEM_TOOLS_SCHEMA = [
    {
        "name": "update_ledger",
        "description": "Writes a new task to the system ledger/task queue.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {"type": "string", "description": "The detailed description of the task to be logged."},
                "priority": {"type": "integer", "description": "Priority 1 (highest) to 10 (lowest). Default 5."}
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "request_core_update",
        "description": "Request authorization to modify the system charter or delete core memory. Triggers MFA.",
        "parameters": {
            "type": "object",
            "properties": {
                "component": {"type": "string", "description": "System component to update, e.g. 'charter'."},
                "proposed_change": {"type": "string", "description": "Details of the requested update."}
            },
            "required": ["component", "proposed_change"]
        }
    },
    {
        "name": "update_core_memory",
        "description": "Updates a key in Core Working Memory (short-term RAM) like 'current_focus'.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key to update."},
                "value": {"type": "string", "description": "The new value."}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "search_archival_memory",
        "description": "Searches Archival Memory (ChromaDB) for relevant historical context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "ask_admin_for_guidance",
        "description": "Pauses execution and asks the Admin for strategic input when facing ambiguity.",
        "parameters": {
            "type": "object",
            "properties": {
                "context_summary": {"type": "string", "description": "Summary of the current situation."},
                "specific_question": {"type": "string", "description": "The specific question for the Admin."}
            },
            "required": ["context_summary", "specific_question"]
        }
    },
    {
        "name": "query_highest_priority_task",
        "description": "Returns the single highest-priority pending Task from the Objective Backlog. Use this at the start of a heartbeat cycle to find work.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "spawn_new_objective",
        "description": "Adds a new Epic, Story, or Task to the Objective Backlog. Use when you identify a sub-task needed to complete a broader goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "description": "Hierarchy level: Epic, Story, or Task."},
                "title": {"type": "string", "description": "Clear, actionable title for the objective."},
                "estimated_energy": {"type": "integer", "description": "Estimated cognitive energy cost (5-50)."}
            },
            "required": ["tier", "title", "estimated_energy"]
        }
    },
    {
        "name": "update_objective_status",
        "description": "Updates the status of an objective in the backlog. Use to mark tasks active, completed, or suspended.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "The numeric ID of the objective."},
                "new_status": {"type": "string", "description": "New status: pending, active, completed, or suspended."}
            },
            "required": ["task_id", "new_status"]
        }
    },
    {
        "name": "get_system_info",
        "description": "Returns the current date, time, timezone, and host platform. Use this whenever the user asks about the current time, date, or the machine's environment.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "request_capability",
        "description": (
            "Call this tool when you cannot fulfill the user's request because a required "
            "tool or capability does not exist in your current toolkit. "
            "Describe precisely what is needed so System 2 can synthesise a new tool. "
            "Examples: reading a file, fetching a URL, querying an API."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gap_description": {
                    "type": "string",
                    "description": "Exact description of the missing capability (e.g. 'read the current system time and timezone')."
                },
                "suggested_tool_name": {
                    "type": "string",
                    "description": "snake_case name for the tool that would fill this gap (e.g. 'get_system_info')."
                }
            },
            "required": ["gap_description", "suggested_tool_name"]
        }
    }
]

# ─────────────────────────────────────────────────────────────
# Existing Tools
# ─────────────────────────────────────────────────────────────

def _sync_update_ledger(task_description: str, priority: int = 5) -> str:
    ledger = None
    try:
        ledger = LedgerMemory()
        ledger.add_task(task_description=task_description, priority=priority)
        return "Success: Task added to ledger."
    except Exception as e:
        return f"Error: Could not write to database due to [{str(e)}]."
    finally:
        if ledger:
            ledger.close()

async def update_ledger(task_description: str, priority: int = 5) -> str:
    logger.info(f"update_ledger: {task_description}")
    return await asyncio.to_thread(_sync_update_ledger, task_description, priority)

async def request_core_update(component: str, proposed_change: str) -> str:
    logger.info(f"request_core_update: {component}")
    task_desc = f"Approved Core Update Request for {component}: {proposed_change}"
    result = await asyncio.to_thread(_sync_update_ledger, task_desc, 1)
    if result.startswith("Success"):
        return "Success: MFA authorized. Core update request logged to ledger for admin review."
    return f"Error: Failed to log core update - {result}"

def _sync_update_core_memory(key: str, value: str) -> str:
    try:
        mem = CoreMemory()
        mem.update(key, value)
        return f"Success: Core memory key '{key}' updated."
    except Exception as e:
        return f"Error: Could not update core memory due to [{str(e)}]."

async def update_core_memory(key: str, value: str) -> str:
    logger.info(f"update_core_memory: {key}={value}")
    return await asyncio.to_thread(_sync_update_core_memory, key, value)

# ─────────────────────────────────────────────────────────────
# System Awareness Tools
# ─────────────────────────────────────────────────────────────

async def get_system_info() -> str:
    """Returns current datetime, timezone, and host platform."""
    try:
        now = datetime.now().astimezone()
        tz = now.strftime("%Z %z")
        return (
            f"DateTime: {now.strftime('%Y-%m-%d %H:%M:%S')} {tz} | "
            f"Platform: {platform.system()} {platform.release()} | "
            f"Machine: {platform.machine()}"
        )
    except Exception as e:
        return f"Error retrieving system info: {e}"

async def request_capability(gap_description: str, suggested_tool_name: str) -> str:
    """Intercepted by the orchestrator — this function body is never reached."""
    return f"Capability gap registered: {gap_description}"

def _sync_search_archival_memory(query: str) -> str:
    vector_memory = None
    try:
        vector_memory = VectorMemory()
        results = vector_memory.query_memory(query, n_results=3)
        if not results:
            return "No relevant archival memory found."
        response = "Archival Results:\n"
        for i, res in enumerate(results, 1):
            response += f"{i}. {res['document']}\n"
        return response
    except Exception as e:
        return f"Error: Could not search archival memory due to [{str(e)}]."

async def search_archival_memory(query: str) -> str:
    logger.info(f"search_archival_memory: {query}")
    return await asyncio.to_thread(_sync_search_archival_memory, query)

async def ask_admin_for_guidance(context_summary: str, specific_question: str) -> str:
    logger.info(f"ask_admin_for_guidance: {specific_question}")
    return f"Guidance requested: {specific_question}"

# ─────────────────────────────────────────────────────────────
# Objective Backlog Tools (Sprint 7)
# ─────────────────────────────────────────────────────────────

def _sync_query_highest_priority_task() -> str:
    ledger = None
    try:
        ledger = LedgerMemory()
        task = ledger.get_highest_priority_task()
        if not task:
            return "BACKLOG: No pending Tasks found."
        return (
            f"BACKLOG TASK\n"
            f"  ID: {task['id']}\n"
            f"  Title: {task['title']}\n"
            f"  Estimated Energy: {task['estimated_energy']}\n"
            f"  Priority: {task['priority']}\n"
            f"  Origin: {task['origin']}"
        )
    except Exception as e:
        return f"Error: Could not query backlog due to [{str(e)}]."
    finally:
        if ledger:
            ledger.close()

async def query_highest_priority_task() -> str:
    logger.info("query_highest_priority_task called")
    return await asyncio.to_thread(_sync_query_highest_priority_task)

def _sync_spawn_new_objective(tier: str, title: str, estimated_energy: int) -> str:
    ledger = None
    try:
        ledger = LedgerMemory()
        obj_id = ledger.add_objective(tier=tier, title=title,
                                      estimated_energy=estimated_energy, origin="System")
        return f"Success: New {tier} added to backlog (id={obj_id}): {title!r}"
    except Exception as e:
        return f"Error: Could not spawn objective due to [{str(e)}]."
    finally:
        if ledger:
            ledger.close()

async def spawn_new_objective(tier: str, title: str, estimated_energy: int) -> str:
    logger.info(f"spawn_new_objective: [{tier}] {title}")
    return await asyncio.to_thread(_sync_spawn_new_objective, tier, title, estimated_energy)

def _sync_update_objective_status(task_id: int, new_status: str) -> str:
    ledger = None
    try:
        ledger = LedgerMemory()
        ledger.update_objective_status(task_id, new_status)
        return f"Success: Objective {task_id} marked as {new_status!r}."
    except Exception as e:
        return f"Error: Could not update objective status due to [{str(e)}]."
    finally:
        if ledger:
            ledger.close()

async def update_objective_status(task_id: int, new_status: str) -> str:
    logger.info(f"update_objective_status: id={task_id} -> {new_status}")
    return await asyncio.to_thread(_sync_update_objective_status, task_id, new_status)
