"""Module-level constants for the AIDEN orchestrator.
Centralised here to avoid circular imports between orchestrator.py
and its mixin files."""

import os
import re

# Energy costs per operation
ENERGY_COST_SUPERVISOR = 10
ENERGY_COST_WORKER = 15
ENERGY_COST_CRITIC = 10
ENERGY_COST_TOOL = 5
# When the remaining budget exceeds this floor and the only reason for deferral is
# a borderline ROI ("ROI too low"), ad-hoc conversational requests are approved
# unconditionally so that energy accounting never blocks normal user interaction
# while resources are plentiful.  Configurable via AD_HOC_COMFORTABLE_BUDGET_FLOOR.
_AD_HOC_COMFORTABLE_BUDGET_FLOOR = int(os.getenv("AD_HOC_COMFORTABLE_BUDGET_FLOOR", "50"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "1800"))  # default 30 minutes
MEMORY_SAVE_THRESHOLD = int(os.getenv("MEMORY_SAVE_THRESHOLD", "120"))
MEMORY_CONSOLIDATION_INTERVAL = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "21600"))  # 6 hours
# How long an unanswered MFA/HITL/tool-approval entry stays alive before auto-expiry
_PENDING_STATE_TTL_SECONDS = int(os.getenv("PENDING_STATE_TTL_SECONDS", "86400"))  # 24 hours
_CONSOLIDATION_TRIGGER_TURNS = int(os.getenv("CONSOLIDATION_TRIGGER_TURNS", "10"))
_SYNTHESIS_LOCKOUT_TTL_SECONDS = int(os.getenv("SYNTHESIS_LOCKOUT_TTL_SECONDS", "600"))
_SYSTEM_1_ERROR_PREFIX = "[System 1 - Error]"
_MORAL_TRIVIALITY_READ_ONLY_HINTS = (
    "time",
    "date",
    "timezone",
    "weather",
    "stock",
    "price",
    "score",
    "headline",
    "system info",
)
_MORAL_TRIVIALITY_BLOCK_HINTS = (
    "modify",
    "delete",
    "write",
    "update_",
    "request_capability",
    "escalate_to_system_2",
    "run_terminal_command",
    "execute_python_sandbox",
    "manage_file_system",
    "spawn_new_objective",
    "approval",
    "mfa",
    "deploy",
)
_RECENT_CHAT_HISTORY_LIMIT = 12
_CRITIC_SHORT_OUTPUT_THRESHOLD = 220
_SYSTEM_2_ERROR_PREFIX = "[System 2 - Error]"
_SYSTEM_2_EMPTY_PREFIX = "[System 2 - No Response]"
_GOAL_PLANNER_COMPLEXITY_THRESHOLD = int(os.getenv("GOAL_PLANNER_COMPLEXITY_THRESHOLD", "4"))
_HEARTBEAT_FAILURE_STRIKES = int(os.getenv("HEARTBEAT_FAILURE_STRIKES", "3"))
_HEARTBEAT_FAILURE_STATE_KEY = "heartbeat_task_failure_counts"
HEARTBEAT_TASK_PREFIX_FMT = "[HEARTBEAT TASK #{task_id}]"
_INTENT_CLASSIFIER_CACHE_TTL_SECONDS = 300.0
_INTENT_CLASSIFIER_CACHE_MAX_SIZE_DEFAULT = 256
_INTENT_CLASSIFIER_TIMEOUT_SECONDS = 10.0
_INTENT_CLASSIFIER_MAX_OUTPUT_TOKENS = 60
_USER_INTENTS = frozenset({"capability_query", "task", "profile_update", "casual"})
_HEARTBEAT_REPLENISH_ENV_DEFAULT = 2
_CONSOLIDATION_TURN_COUNT_MAX_USERS = 100
_EXPLICIT_MEMORY_REQUEST_RE = re.compile(
    r"\b(?:remember\s+that|remember\s+my|don'?t\s+forget|please\s+remember|save\s+that|note\s+that)\b",
    flags=re.IGNORECASE,
)
_VOICE_NOTE_PLACEHOLDER_RE = re.compile(r"^\[Voice note · \d+ bytes · [^\]]+\]$")
_CHARTER_TIER_TAGS = ("Tier_1_Axioms", "Tier_2_Strategic", "Tier_3_Operational")
_CHARTER_PARSE_ERROR_KEY = "__charter_parse_error__"
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Key used in the state dict to signal a blocked (non-ok) router result
_BLOCKED_KEY = "_blocked_result"
_CATALOG_MATCH_STOPWORDS = {
    "a", "an", "and", "the", "to", "for", "of", "in", "on", "or", "is", "are",
    "be", "can", "could", "would", "should", "have", "has", "with", "new", "needed",
    "need", "admin", "user", "function", "tool", "tools", "skill", "skills",
    "capability", "capabilities", "allow", "provide", "more", "than", "just", "use",
    "using", "this", "that", "from", "missing", "cannot", "couldnt", "unable",
}
_CATALOG_META_TOOL_NAMES = {
    "request_capability",
    "request_core_update",
    "ask_admin_for_guidance",
    "escalate_to_system_2",
}
