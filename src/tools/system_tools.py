import logging
import asyncio
from src.memory.ledger_db import LedgerMemory

logger = logging.getLogger(__name__)

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
