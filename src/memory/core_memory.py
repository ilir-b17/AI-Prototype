"""
Core Memory Module - Implements Working Memory (OS-Level RAM)

This module handles short-term context like current focus and user preferences,
stored in a JSON file to act as temporary working memory (RAM) for the Orchestrator
and Worker Nodes.
"""

import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CoreMemory:
    """
    Manages short-term working memory stored in a local JSON file.
    """

    def __init__(self, memory_file_path: str = "data/core_memory.json"):
        self.memory_file_path = memory_file_path
        self._initialize_memory()

    def _initialize_memory(self):
        """Ensures the core memory file exists with default schema."""
        os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)

        if not os.path.exists(self.memory_file_path):
            default_state = {
                "current_focus": "",
                "user_preferences": ""
            }
            self._save_memory(default_state)
            logger.info(f"Initialized new Core Working Memory at {self.memory_file_path}")
        else:
            try:
                self.get_all()
            except json.JSONDecodeError:
                # Re-initialize if corrupted
                logger.warning(f"Core Working Memory at {self.memory_file_path} corrupted. Re-initializing.")
                default_state = {
                    "current_focus": "",
                    "user_preferences": ""
                }
                self._save_memory(default_state)

    def _save_memory(self, state: Dict[str, Any]):
        """Saves state dict to JSON file."""
        try:
            with open(self.memory_file_path, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save core memory: {e}")

    def get_all(self) -> Dict[str, Any]:
        """Retrieves entire memory state."""
        try:
            with open(self.memory_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load core memory: {e}")
            return {"current_focus": "", "user_preferences": ""}

    def get_context_string(self) -> str:
        """Returns the core memory as a string formatted for Prompt injection."""
        state = self.get_all()
        host_os = state.get('host_os', '')
        os_line = f"\n  <Host_OS>{host_os}</Host_OS>" if host_os else ""
        return (
            f"<Core_Working_Memory>"
            f"\n  <Current_Focus>{state.get('current_focus', '')}</Current_Focus>"
            f"\n  <User_Preferences>{state.get('user_preferences', '')}</User_Preferences>"
            f"{os_line}"
            f"\n</Core_Working_Memory>"
        )

    def update(self, key: str, value: Any) -> bool:
        """Updates a specific key in core memory."""
        state = self.get_all()
        if key not in state:
            logger.warning(f"Attempted to update unknown core memory key: {key}")
            # we can choose to allow new keys, but let's stick to schema
            state[key] = value
        else:
            state[key] = value

        self._save_memory(state)
        logger.info(f"Updated Core Memory: {key} = {value}")
        return True
