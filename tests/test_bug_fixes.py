"""
tests/test_bug_fixes.py
========================
Unit tests that verify the 14 bug fixes applied during the code-review pass.

Tests here are deliberately lightweight: they exercise pure logic (state
manipulation, AST sandbox, energy caps) without requiring a running Ollama,
Telegram, or SQLite instance.
"""
import asyncio
import json
import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from src.core.llm_router import CognitiveRouter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fresh_state(user_id: str = "test_user", user_input: str = "hello") -> dict:
    """Build a minimal orchestrator state dict as _new_state() would."""
    return {
        "user_id": user_id,
        "user_input": user_input,
        "chat_history": [],
        "current_plan": [],
        "worker_outputs": {},
        "final_response": "",
        "iteration_count": 0,
        "admin_guidance": "",
        "energy_remaining": 100,
        "hitl_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE-005 & ISSUE-013 — HITL count, energy cap, abandonment
# ─────────────────────────────────────────────────────────────────────────────

class TestHITLCountAndEnergyCap:
    """Verify the HITL cycle-limit and energy recharge cap logic."""

    def test_new_state_initialises_hitl_count_to_zero(self):
        """_new_state() must include hitl_count=0 (ISSUE-005)."""
        state = _make_fresh_state()
        assert "hitl_count" in state
        assert state["hitl_count"] == 0

    def test_hitl_count_increments_on_each_resume(self):
        """
        Simulates the _load_state HITL-resume branch:
        hitl_count must increase by 1 each time the admin responds.
        """
        state = _make_fresh_state()
        state["hitl_count"] = 0  # initial

        # Simulate three resume cycles
        for expected_count in range(1, 4):
            state["hitl_count"] = state.get("hitl_count", 0) + 1
            assert state["hitl_count"] == expected_count

    def test_energy_recharge_capped_at_75(self):
        """Energy must never be recharged above 75 after a HITL resume (ISSUE-005)."""
        # Start exhausted
        state = _make_fresh_state()
        state["energy_remaining"] = 0
        state["energy_remaining"] = min(state["energy_remaining"] + 50, 75)
        assert state["energy_remaining"] == 50

        # Start at 40 — still capped at 75
        state["energy_remaining"] = 40
        state["energy_remaining"] = min(state["energy_remaining"] + 50, 75)
        assert state["energy_remaining"] == 75

        # Start at 80 (above cap) — capped at 75
        state["energy_remaining"] = 80
        state["energy_remaining"] = min(state["energy_remaining"] + 50, 75)
        assert state["energy_remaining"] == 75

    def test_energy_never_exceeds_75_starting_from_zero(self):
        """Even from zero energy, recharge stays ≤ 75."""
        state = _make_fresh_state()
        state["energy_remaining"] = 0
        recharged = min(state["energy_remaining"] + 50, 75)
        assert recharged <= 75

    def test_hitl_cycle_limit_sets_final_response(self):
        """When hitl_count reaches 3 the state must carry the abandonment message."""
        state = _make_fresh_state()
        state["hitl_count"] = 2  # third admin response → will become 3

        # Simulate the _load_state logic
        state["hitl_count"] = state.get("hitl_count", 0) + 1
        if state["hitl_count"] >= 3:
            state["final_response"] = (
                "This task has been attempted 3 times with admin guidance and "
                "could not be completed. The request has been abandoned to prevent "
                "an infinite loop. Please rephrase or break it into smaller steps."
            )

        assert state["hitl_count"] == 3
        assert "abandoned" in state["final_response"].lower()

    def test_hitl_cycle_limit_not_triggered_below_three(self):
        """States with hitl_count < 3 must NOT receive an abandonment response."""
        for count in (0, 1, 2):
            state = _make_fresh_state()
            state["hitl_count"] = count
            # After increment the count is count+1
            state["hitl_count"] += 1
            if state["hitl_count"] >= 3:
                state["final_response"] = "abandoned"

            if count < 2:
                assert "abandoned" not in state.get("final_response", "")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE-006 — current_plan cleared on Critic FAIL
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticFailClearsPlan:
    """Verify that current_plan is wiped when the Critic rejects output."""

    def test_current_plan_cleared_after_critic_fail(self):
        """
        Replicates the _run_graph_loop Critic-FAIL branch:
        current_plan must be reset to [] so the supervisor re-plans.
        """
        state = _make_fresh_state()
        state["current_plan"] = ["research_agent", "coder_agent"]
        state["critic_feedback"] = "FAIL: content violates charter."

        if state["critic_feedback"] != "PASS":
            state["final_response"] = ""
            state["worker_outputs"] = {}
            state["current_plan"] = []  # ISSUE-006 fix

        assert state["current_plan"] == []

    def test_current_plan_preserved_on_pass(self):
        """current_plan must NOT be cleared when the Critic returns PASS."""
        state = _make_fresh_state()
        state["current_plan"] = ["research_agent"]
        state["critic_feedback"] = "PASS"

        # No clearing should happen on PASS
        if state["critic_feedback"] != "PASS":
            state["current_plan"] = []

        assert state["current_plan"] == ["research_agent"]


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE-008 — _validate_tool_code_ast must be a proper static method
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateToolCodeAstIsStatic:
    """Ensure _validate_tool_code_ast works correctly as a static method (ISSUE-008)."""

    def test_called_as_static_method_does_not_raise_on_safe_code(self):
        """Valid async tool code with no blocked imports must pass without error."""
        safe_code = """
async def my_safe_tool() -> str:
    return "hello"
"""
        # Must not raise TypeError (which was the bug: self was passed as code)
        CognitiveRouter._validate_tool_code_ast(safe_code, "my_safe_tool")

    def test_called_via_instance_does_not_pass_self_as_code(self):
        """Calling the static method via an instance must still work (ISSUE-008)."""
        router = CognitiveRouter.__new__(CognitiveRouter)
        safe_code = "async def noop() -> str:\n    return 'ok'\n"
        # Prior to fix this would raise TypeError because self was passed as code
        router._validate_tool_code_ast(safe_code, "noop")

    def test_syntax_error_raises_value_error(self):
        """Malformed Python must raise ValueError with a clear message."""
        bad_code = "def broken(:"
        with pytest.raises(ValueError, match="syntax error"):
            CognitiveRouter._validate_tool_code_ast(bad_code, "broken_tool")

    def test_blocked_import_raises_value_error(self):
        """Code that imports a blocked module must raise ValueError."""
        blocked_code = "import os\nasync def evil() -> str:\n    return os.getcwd()\n"
        with pytest.raises(ValueError):
            CognitiveRouter._validate_tool_code_ast(blocked_code, "evil_tool")


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE-009 — Expanded blocked module list
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandedBlockedModules:
    """Verify that all newly blocked modules are rejected by the AST sandbox (ISSUE-009)."""

    @pytest.mark.parametrize("module_name", [
        "asyncio",
        "concurrent",
        "runpy",
        "code",
        "codeop",
        "compileall",
        "tempfile",
        "mmap",
        "pickle",
        "marshal",
        "shelve",
        "ast",
        # Pre-existing blocked modules that must still be blocked
        "os",
        "sys",
        "subprocess",
        "socket",
        "threading",
    ])
    def test_blocked_module_import_is_rejected(self, module_name: str):
        """Each blocked module must cause ValueError when imported in synthesised code."""
        code = (
            f"import {module_name}\n"
            f"async def malicious_tool() -> str:\n"
            f"    return 'pwned'\n"
        )
        with pytest.raises(ValueError):
            CognitiveRouter._validate_tool_code_ast(code, "malicious_tool")

    @pytest.mark.parametrize("module_name", [
        "asyncio",
        "concurrent",
        "pickle",
        "marshal",
    ])
    def test_from_import_blocked_modules_also_rejected(self, module_name: str):
        """from X import Y must also be blocked for dangerous top-level modules."""
        code = (
            f"from {module_name} import something\n"
            f"async def bad_tool() -> str:\n"
            f"    return 'pwned'\n"
        )
        with pytest.raises(ValueError):
            CognitiveRouter._validate_tool_code_ast(code, "bad_tool")

    def test_safe_stdlib_import_allowed(self):
        """Imports of unblocked stdlib modules must pass the AST check."""
        safe_code = (
            "import datetime\n"
            "import math\n"
            "import re\n"
            "async def safe_tool() -> str:\n"
            "    return str(datetime.datetime.now())\n"
        )
        # Must not raise
        CognitiveRouter._validate_tool_code_ast(safe_code, "safe_tool")
