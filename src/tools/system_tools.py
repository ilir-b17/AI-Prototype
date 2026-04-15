import logging
import asyncio
from src.memory.ledger_db import LedgerMemory

logger = logging.getLogger(__name__)

from src.memory.core_memory import CoreMemory
from src.memory.vector_db import VectorMemory

# Define schemas for the tools
SYSTEM_TOOLS_SCHEMA = [
    {
        "name": "update_ledger",
        "description": "Writes a new task to the system ledger/task queue.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "The detailed description of the task to be logged."
                },
                "priority": {
                    "type": "integer",
                    "description": "The priority of the task from 1 (highest) to 10 (lowest). Default is 5."
                }
            },
            "required": ["task_description"]
        }
    },
    {
        "name": "request_core_update",
        "description": "Request authorization to modify the system charter or delete core memory. This triggers a security check.",
        "parameters": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "description": "The system component to update, e.g., 'charter' or 'memory'."
                },
                "proposed_change": {
                    "type": "string",
                    "description": "The details of the requested update or deletion."
                }
            },
            "required": ["component", "proposed_change"]
        }
    },
    {
        "name": "update_core_memory",
        "description": "Updates a specific key in the Core Working Memory (short-term RAM) like 'current_focus' or 'user_preferences'.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key to update in core memory."
                },
                "value": {
                    "type": "string",
                    "description": "The new value for the key."
                }
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "search_archival_memory",
        "description": "Searches the Archival Memory (ChromaDB) for relevant historical context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in archival memory."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ask_admin_for_guidance",
        "description": "Pauses the current execution and asks the Admin for strategic input or guidance when facing ambiguity or when requested by the Critic node.",
        "parameters": {
            "type": "object",
            "properties": {
                "context_summary": {
                    "type": "string",
                    "description": "A summary of the current situation and why guidance is needed."
                },
                "specific_question": {
                    "type": "string",
                    "description": "The specific question to ask the Admin."
                }
            },
            "required": ["context_summary", "specific_question"]
        }
    }
]

def _sync_update_ledger(task_description: str, priority: int = 5) -> str:
    """
    Synchronous inner function to handle DB operations.
    """
    ledger = None
    try:
        ledger = LedgerMemory()
        task_id = ledger.add_task(task_description=task_description, priority=priority)
        return "Success: Task added to ledger."
    except Exception as e:
        return f"Error: Could not write to database due to [{str(e)}]."
    finally:
        if ledger is not None:
            ledger.close()

async def update_ledger(task_description: str, priority: int = 5) -> str:
    """
    Asynchronously writes a new task to the ledger.

    Args:
        task_description: The description of the task to be logged.
        priority: Task priority.

    Returns:
        str: Success or error message.
    """
    logger.info(f"update_ledger called with task_description: {task_description}")
    # Wrap synchronous DB calls in a thread to avoid blocking the event loop
    result = await asyncio.to_thread(_sync_update_ledger, task_description, priority)
    return result

async def request_core_update(component: str, proposed_change: str) -> str:
    """
    Dummy tool for core updates to trigger MFA.
    If executed (which implies MFA passed), logs the update to the ledger.
    """
    logger.info(f"request_core_update called for component: {component}")
    # This acts as the action AFTER MFA.
    task_desc = f"Approved Core Update Request for {component}: {proposed_change}"
    result = await asyncio.to_thread(_sync_update_ledger, task_desc, 1)
    if result.startswith("Success"):
        return "Success: MFA authorized. Core update request has been securely logged to the ledger for admin review."
    else:
        return f"Error: Failed to log core update - {result}"

def _sync_update_core_memory(key: str, value: str) -> str:
    """Synchronous internal function for core memory."""
    try:
        mem = CoreMemory()
        mem.update(key, value)
        return f"Success: Core memory key '{key}' updated."
    except Exception as e:
        return f"Error: Could not update core memory due to [{str(e)}]."

async def update_core_memory(key: str, value: str) -> str:
    """Asynchronously updates a core memory key."""
    logger.info(f"update_core_memory called with key: {key}, value: {value}")
    result = await asyncio.to_thread(_sync_update_core_memory, key, value)
    return result

def _sync_search_archival_memory(query: str) -> str:
    """Synchronous internal function for archival search."""
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
    """Asynchronously searches archival memory."""
    logger.info(f"search_archival_memory called with query: {query}")
    result = await asyncio.to_thread(_sync_search_archival_memory, query)
    return result

async def ask_admin_for_guidance(context_summary: str, specific_question: str) -> str:
    """
    Dummy tool for ask_admin_for_guidance.
    The real interruption logic happens in llm_router.py by raising RequiresHITLError.
    """
    logger.info(f"ask_admin_for_guidance called. Summary: {context_summary}, Question: {specific_question}")
    return f"Guidance requested: {specific_question}"
