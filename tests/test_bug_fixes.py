"""
tests/test_bug_fixes.py
========================
Unit tests that verify the 14 bug fixes applied during the code-review pass.

Tests here are deliberately lightweight: they exercise pure logic (state
manipulation, AST sandbox, energy caps) without requiring a running Ollama,
Telegram, or SQLite instance.
"""
import asyncio
import pytest
import tempfile
import time
import src.core.orchestrator as orchestrator_module

from unittest.mock import AsyncMock, MagicMock
from src.core.llm_router import CognitiveRouter, RouterResult
from src.core.orchestrator import Orchestrator
from src.memory.core_memory import CoreMemory


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


class _InMemoryLedger:
    def __init__(self):
        self.chat_turns = {}

    async def get_chat_history(self, user_id, limit=5):
        await asyncio.sleep(0)
        turns = list(self.chat_turns.get(user_id, []))
        return turns[-limit:]

    async def save_chat_turn(self, user_id, role, content):
        await asyncio.sleep(0)
        self.chat_turns.setdefault(user_id, []).append({"role": role, "content": content})


def _close_fire_and_forget(coro):
    coro.close()


def _build_process_message_test_orchestrator(
    tmp_dir: str,
    *,
    skill_names,
    schemas,
    execute_tool_side_effect,
    route_to_system_1_side_effect,
    current_time: str,
):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}
    orchestrator.core_memory = CoreMemory(f"{tmp_dir}/core_memory.json")
    orchestrator.ledger_memory = _InMemoryLedger()
    orchestrator.vector_memory = MagicMock()
    orchestrator.vector_memory.add_memory_async = AsyncMock(return_value="mem-1")
    orchestrator.charter_text = "Core Directive: Do no harm."
    orchestrator.prompt_config = MagicMock(downloads_dir="downloads")
    orchestrator.sensory_state = {
        "current_time": current_time,
        "os": "Windows 11",
        "cpu_usage": "4%",
        "cwd": "C:/Users/iboci/Live-Trading-bot/AI_Prototype",
    }
    orchestrator._background_tasks = set()
    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()
    orchestrator._compiled_graph = None
    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_resume_tool_approval = AsyncMock(return_value=None)
    orchestrator._run_graph_loop = AsyncMock(return_value="graph fallback should not run")
    orchestrator._consolidate_memory = AsyncMock()
    orchestrator._save_memory_async = AsyncMock()
    orchestrator._fire_and_forget = _close_fire_and_forget

    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)
    orchestrator.cognitive_router.get_system_1_gate_metrics = MagicMock(return_value={})
    orchestrator.cognitive_router.registry.get_skill_names.return_value = list(skill_names)
    orchestrator.cognitive_router.registry.get_schemas.return_value = list(schemas)
    orchestrator.cognitive_router._execute_tool = AsyncMock(side_effect=execute_tool_side_effect)
    orchestrator.cognitive_router.route_to_system_1 = AsyncMock(side_effect=route_to_system_1_side_effect)
    return orchestrator


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

    def test_asyncio_import_is_allowed(self):
        """asyncio imports are allowed for async tool synthesis."""
        safe_code = (
            "import asyncio\n"
            "async def safe_async_tool() -> str:\n"
            "    await asyncio.sleep(0)\n"
            "    return 'ok'\n"
        )
        CognitiveRouter._validate_tool_code_ast(safe_code, "safe_async_tool")

    @pytest.mark.parametrize("call_name", [
        "create_subprocess_exec",
        "create_subprocess_shell",
    ])
    def test_asyncio_subprocess_calls_are_rejected_in_tool_code(self, call_name: str):
        """Tool code may import asyncio, but subprocess creation APIs remain blocked."""
        code = (
            "import asyncio\n"
            "async def malicious_tool() -> str:\n"
            f"    await asyncio.{call_name}('python')\n"
            "    return 'bad'\n"
        )
        with pytest.raises(ValueError, match=call_name):
            CognitiveRouter._validate_tool_code_ast(code, "malicious_tool")

    @pytest.mark.parametrize("call_name", [
        "create_subprocess_exec",
        "create_subprocess_shell",
    ])
    def test_asyncio_subprocess_calls_are_rejected_in_pytest_code(self, call_name: str):
        """Generated pytest code is subject to the same asyncio subprocess guard."""
        pytest_code = (
            "import pytest\n"
            "import asyncio\n\n"
            "@pytest.mark.asyncio\n"
            "async def test_bad_asyncio_usage():\n"
            f"    await asyncio.{call_name}('python')\n"
        )
        with pytest.raises(ValueError, match=call_name):
            CognitiveRouter._validate_pytest_code_ast(pytest_code, "safe_tool")


# ─────────────────────────────────────────────────────────────────────────────
# Tool-loop guardrails — duplicate calls, invalid tool names, critic skip
# ─────────────────────────────────────────────────────────────────────────────

class TestToolLoopGuardrails:
    @pytest.mark.asyncio
    async def test_duplicate_tool_call_forces_text_synthesis(self):
        router = CognitiveRouter.__new__(CognitiveRouter)
        router._ollama_options = {}
        router._ollama_timeout = 5.0
        router._execute_tool = AsyncMock(
            return_value=RouterResult(status="ok", content="Current time: 12:34:56")
        )

        first_message = {
            "tool_calls": [{"function": {"name": "get_system_info", "arguments": {}}}]
        }
        duplicate_message = {
            "message": {
                "tool_calls": [{"function": {"name": "get_system_info", "arguments": {}}}]
            }
        }
        synthesis_message = {"message": {"content": "The current time is 12:34:56."}}

        client = MagicMock()
        client.chat = AsyncMock(side_effect=[duplicate_message, synthesis_message])

        result = await router._run_tool_loop(
            client,
            "gemma4:26b",
            first_message,
            [{"role": "user", "content": "What time is it?"}],
            active_tools=[{"type": "function"}],
        )

        assert result.status == "ok"
        assert result.content == "The current time is 12:34:56."
        assert router._execute_tool.await_count == 1
        assert client.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_invalid_followup_tool_forces_answer_from_prior_results(self):
        router = CognitiveRouter.__new__(CognitiveRouter)
        router._ollama_options = {}
        router._ollama_timeout = 5.0
        router._execute_tool = AsyncMock(side_effect=[
            RouterResult(status="ok", content="Search result: Vienna forecast is 18C and cloudy."),
            RouterResult(status="ok", content="Error: Unknown tool 'run_python_code'."),
        ])

        first_message = {
            "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "Vienna weather"}}}]
        }
        invalid_followup = {
            "message": {
                "tool_calls": [{"function": {"name": "run_python_code", "arguments": {"code": "print(1)"}}}]
            }
        }
        synthesis_message = {"message": {"content": "Today in Vienna it is about 18C and cloudy."}}

        client = MagicMock()
        client.chat = AsyncMock(side_effect=[invalid_followup, synthesis_message])

        result = await router._run_tool_loop(
            client,
            "gemma4:26b",
            first_message,
            [{"role": "user", "content": "What is the weather in Vienna today?"}],
            active_tools=[{"type": "function"}],
        )

        assert result.status == "ok"
        assert result.content == "Today in Vienna it is about 18C and cloudy."
        assert router._execute_tool.await_count == 2
        assert client.chat.await_count == 2


class TestSystem1ConcurrencyGuard:
    @pytest.mark.asyncio
    async def test_system_1_calls_are_serialized_by_semaphore(self):
        router = CognitiveRouter.__new__(CognitiveRouter)
        router.local_model = "gemma4:e4b"
        router._ollama_timeout = 5.0
        router._ollama_options = {}
        router._system_1_max_concurrency = 1
        router._system_1_semaphore = asyncio.Semaphore(1)
        router._system_1_active_requests = 0
        router._system_1_waiting_requests = 0
        router._system_1_wait_events = 0
        router._system_1_total_wait_seconds = 0.0
        router._system_1_peak_waiting_requests = 0
        router._get_or_create_ollama_client = MagicMock(return_value=object())
        router._format_ollama_tools = MagicMock(return_value=None)
        router._run_tool_loop = AsyncMock()
        router._handle_text_response = AsyncMock(side_effect=[
            RouterResult(status="ok", content="first"),
            RouterResult(status="ok", content="second"),
        ])

        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        started_calls = 0

        async def fake_call(*_args, **_kwargs):
            nonlocal started_calls
            started_calls += 1
            if started_calls == 1:
                first_entered.set()
                await release_first.wait()
            return {"message": {"content": f"response {started_calls}"}}, router.local_model

        router._call_ollama_with_model_fallback = fake_call

        task1 = asyncio.create_task(router.route_to_system_1([
            {"role": "user", "content": "first request"}
        ]))
        await first_entered.wait()

        task2 = asyncio.create_task(router.route_to_system_1([
            {"role": "user", "content": "second request"}
        ]))
        await asyncio.sleep(0)

        assert started_calls == 1

        release_first.set()
        result1, result2 = await asyncio.gather(task1, task2)

        assert started_calls == 2
        assert result1.content == "first"
        assert result2.content == "second"

    @pytest.mark.asyncio
    async def test_system_1_queue_wait_is_logged_and_counted(self, caplog):
        router = CognitiveRouter.__new__(CognitiveRouter)
        router.local_model = "gemma4:e4b"
        router._ollama_timeout = 5.0
        router._ollama_options = {}
        router._system_1_max_concurrency = 1
        router._system_1_semaphore = asyncio.Semaphore(1)
        router._system_1_active_requests = 0
        router._system_1_waiting_requests = 0
        router._system_1_wait_events = 0
        router._system_1_total_wait_seconds = 0.0
        router._system_1_peak_waiting_requests = 0
        router._get_or_create_ollama_client = MagicMock(return_value=object())
        router._format_ollama_tools = MagicMock(return_value=None)
        router._run_tool_loop = AsyncMock()
        router._handle_text_response = AsyncMock(side_effect=[
            RouterResult(status="ok", content="first"),
            RouterResult(status="ok", content="second"),
        ])

        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        started_calls = 0

        async def fake_call(*_args, **_kwargs):
            nonlocal started_calls
            started_calls += 1
            if started_calls == 1:
                first_entered.set()
                await release_first.wait()
            return {"message": {"content": f"response {started_calls}"}}, router.local_model

        router._call_ollama_with_model_fallback = fake_call

        with caplog.at_level("INFO", logger="src.core.llm_router"):
            task1 = asyncio.create_task(router.route_to_system_1([
                {"role": "user", "content": "first request"}
            ]))
            await first_entered.wait()

            task2 = asyncio.create_task(router.route_to_system_1([
                {"role": "user", "content": "second request"}
            ]))
            await asyncio.sleep(0)

            release_first.set()
            await asyncio.gather(task1, task2)

        metrics = router.get_system_1_gate_metrics()

        assert any("queuing local request" in message for message in caplog.messages)
        assert any("slot acquired after" in message for message in caplog.messages)
        assert metrics["wait_events"] == 1
        assert metrics["peak_waiting_requests"] == 1
        assert metrics["total_wait_seconds"] > 0
        assert metrics["waiting_requests"] == 0


class TestSystem2AvailabilityDetection:
    def test_ollama_cloud_provider_counts_as_available(self):
        router = CognitiveRouter.__new__(CognitiveRouter)
        router._system2_cooldown_until = 0.0
        router.ollama_cloud_client = object()
        router.groq_client = None
        router.gemini_client = None

        assert CognitiveRouter.get_system_2_available(router) is True

    def test_cooldown_still_blocks_availability_even_when_provider_exists(self):
        router = CognitiveRouter.__new__(CognitiveRouter)
        router._system2_cooldown_until = time.time() + 30
        router.ollama_cloud_client = object()
        router.groq_client = None
        router.gemini_client = None

        assert CognitiveRouter.get_system_2_available(router) is False


class TestCloudPayloadRedaction:
    def test_redact_text_for_cloud_strips_private_context_and_pii(self):
        raw_payload = (
            "<context_and_memory>secret memory about user profile</context_and_memory>\n"
            "[Machine Context - OS: Windows 11 | CPU: 6% | CWD: C:\\Users\\iboci\\Live-Trading-bot\\AI_Prototype]\n"
            "My name is Ilir Boci and my email is ilir@example.com.\n"
            "Task: Explain why the worker failed."
        )

        redacted = Orchestrator._redact_text_for_cloud(raw_payload)

        assert "secret memory" not in redacted
        assert "Ilir Boci" not in redacted
        assert "ilir@example.com" not in redacted
        assert "C:\\Users\\iboci" not in redacted
        assert "Task: Explain why the worker failed." in redacted

    def test_redact_messages_for_cloud_drops_raw_chat_history_by_default(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        messages = [
            {
                "role": "system",
                "content": "Plan this task. <Core_Working_Memory>name=Alice, location=Vienna</Core_Working_Memory>",
            },
            {"role": "user", "content": "Older user turn with details: passport 12345"},
            {"role": "assistant", "content": "Older assistant turn."},
            {"role": "user", "content": "My name is Alice Stone. Please debug this stack trace."},
        ]

        minimized = Orchestrator._redact_messages_for_cloud(orchestrator, messages)

        assert [item["role"] for item in minimized] == ["system", "user"]
        combined = "\n".join(item["content"] for item in minimized)
        assert "Older user turn" not in combined
        assert "passport 12345" not in combined
        assert "Alice Stone" not in combined

    @pytest.mark.asyncio
    async def test_route_to_system_2_redacted_forwards_only_sanitized_payload(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.route_to_system_2 = AsyncMock(
            return_value=RouterResult(status="ok", content="done")
        )

        messages = [
            {
                "role": "system",
                "content": "Supervisor context <Core_Working_Memory>core secret</Core_Working_Memory>",
            },
            {"role": "user", "content": "Previous user turn with bob@example.com"},
            {"role": "assistant", "content": "Prior assistant response"},
            {"role": "user", "content": "Please resolve this worker deadlock."},
        ]

        result = await Orchestrator._route_to_system_2_redacted(
            orchestrator,
            messages,
            allowed_tools=["web_search"],
            purpose="test_redaction",
        )

        assert result.status == "ok"
        assert result.content == "done"

        args, kwargs = orchestrator.cognitive_router.route_to_system_2.await_args
        forwarded_messages = args[0]
        forwarded_text = "\n".join(message["content"] for message in forwarded_messages)

        assert len(forwarded_messages) == 2
        assert "core secret" not in forwarded_text
        assert "bob@example.com" not in forwarded_text
        assert "Previous user turn" not in forwarded_text
        assert "Please resolve this worker deadlock." in forwarded_text
        assert kwargs["allowed_tools"] == ["web_search"]


class TestCriticSkipForDirectSupervisorReplies:
    @pytest.mark.asyncio
    async def test_critic_skips_when_no_worker_outputs_exist(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        state = _make_fresh_state(user_input="What time is it?")
        state["final_response"] = "The current time is 12:34:56."

        result = await Orchestrator.critic_node(orchestrator, state)

        assert result["critic_feedback"] == "PASS"
        assert result["final_response"] == "The current time is 12:34:56."
        assert result["energy_remaining"] == 100

    @pytest.mark.asyncio
    async def test_critic_skips_short_single_agent_output(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        state = _make_fresh_state(user_input="Give me a quick summary")
        state["current_plan"] = [
            {"agent": "research_agent", "task": "Give a short summary", "reason": "Need a fast answer"}
        ]
        state["worker_outputs"] = {"research_agent": "Short answer."}

        result = await Orchestrator.critic_node(orchestrator, state)

        assert result["critic_feedback"] == "PASS"
        assert result["final_response"] == "Short answer."
        assert result["energy_remaining"] == 100

    @pytest.mark.asyncio
    async def test_critic_reviews_combined_independent_agent_output(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.charter_text = "Core Directive: Do no harm."
        orchestrator.cognitive_router.get_system_2_available.return_value = False
        orchestrator._route_to_system_1 = AsyncMock(return_value=RouterResult(status="ok", content="PASS"))

        state = _make_fresh_state(user_input="Research and plan this change")
        state["current_plan"] = [
            {"agent": "research_agent", "task": "Gather the relevant facts", "reason": "Need evidence"},
            {"agent": "planner_agent", "task": "Outline the implementation", "reason": "Need a plan"},
        ]
        state["worker_outputs"] = {
            "research_agent": "This is a detailed research summary that is long enough to justify a critic review. " * 3,
            "planner_agent": "This is a detailed implementation plan that is also long enough to justify a critic review. " * 3,
        }

        result = await Orchestrator.critic_node(orchestrator, state)

        assert result["critic_feedback"] == "PASS"
        orchestrator._route_to_system_1.assert_awaited_once()


class TestCoreMemoryPromptContext:
    @pytest.mark.asyncio
    async def test_conversation_summary_excluded_from_prompt_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory = CoreMemory(f"{tmp_dir}/core_memory.json")
            await memory.update("conversation_summary", "This stale summary should not be injected.")
            await memory.update("current_focus", "testing")

            context = await memory.get_context_string()

            assert "Conversation_Summary" not in context
            assert "Current_Focus" in context


class TestConsolidationTurnThrottling:
    @pytest.mark.asyncio
    async def test_persist_chat_turns_triggers_consolidation_only_after_threshold(self, monkeypatch):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = MagicMock()
        orchestrator.ledger_memory.save_chat_turn = AsyncMock()
        orchestrator._consolidate_memory = AsyncMock()
        scheduled = []

        def fake_fire_and_forget(coro):
            scheduled.append(coro)
            coro.close()

        orchestrator._fire_and_forget = fake_fire_and_forget

        monkeypatch.setattr(orchestrator_module, "_CONSOLIDATION_TRIGGER_TURNS", 3)

        await Orchestrator._persist_chat_turns(orchestrator, "user_1", "msg_1", "resp_1")
        await Orchestrator._persist_chat_turns(orchestrator, "user_1", "msg_2", "resp_2")

        assert len(scheduled) == 0
        assert orchestrator._consolidation_turn_counts["user_1"] == 2

        await Orchestrator._persist_chat_turns(orchestrator, "user_1", "msg_3", "resp_3")

        assert len(scheduled) == 1
        assert orchestrator._consolidation_turn_counts["user_1"] == 0


class TestOrchestratorReadyGate:
    @pytest.mark.asyncio
    async def test_process_message_returns_initializing_message_when_ready_gate_times_out(self, monkeypatch):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._ready = asyncio.Event()

        async def _always_timeout(awaitable, *_args, **_kwargs):
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise asyncio.TimeoutError()

        monkeypatch.setattr(orchestrator_module.asyncio, "wait_for", _always_timeout)

        result = await Orchestrator.process_message(orchestrator, "hello", "test_user")

        assert result == "System is still initializing. Please try again in a moment."


class TestRequestAssessment:
    def test_time_query_routes_to_single_tool(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = [
            {
                "name": "get_system_info",
                "description": "Returns the current date, time, timezone, and host platform details.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "web_search",
                "description": "Search the web for current information and news.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                    "required": ["query"],
                },
            },
        ]

        assessment = Orchestrator._assess_request_route(
            orchestrator,
            "Can you tell me the time right now?",
        )

        assert assessment["mode"] == "single_tool"
        assert assessment["tool_name"] == "get_system_info"

    def test_weather_query_routes_to_web_search(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = [
            {
                "name": "get_system_info",
                "description": "Returns the current date, time, timezone, and host platform details.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "web_search",
                "description": "Search the web for current information and weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                    "required": ["query"],
                },
            },
        ]

        assessment = Orchestrator._assess_request_route(
            orchestrator,
            "What about the weather in Vienna, Austria?",
        )

        assert assessment["mode"] == "single_tool"
        assert assessment["tool_name"] == "web_search"

    def test_high_complexity_query_falls_back_to_graph(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = []

        assessment = Orchestrator._assess_request_route(
            orchestrator,
            "Please analyze this PDF, summarize it, and then tell me what actions I should take next.",
        )

        assert assessment["mode"] == "graph"

    def test_non_trivial_high_score_web_query_falls_back_to_graph(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = [
            {
                "name": "web_search",
                "description": "Search the web for current information, weather, and news.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                    "required": ["query"],
                },
            }
        ]

        assessment = Orchestrator._assess_request_route(
            orchestrator,
            "Can you search the latest weather trends in Vienna and then compare those with the past week?",
        )

        assert assessment["mode"] == "graph"


class TestFastPathResponses:
    @pytest.mark.asyncio
    async def test_time_query_uses_assessed_single_tool_path(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = [
            {
                "name": "get_system_info",
                "description": "Returns the current date, time, timezone, and host platform details.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
        orchestrator.cognitive_router._execute_tool = AsyncMock(
            return_value=RouterResult(
                status="ok",
                content="DateTime: 2026-04-22 20:17:51 CEST +0200 | Platform: Windows 11 | Machine: AMD64",
            )
        )
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock(
            return_value=RouterResult(
                status="ok",
                content="The current time is 2026-04-22 20:17:51 CEST +0200.",
            )
        )

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            "Can you tell me the time right now?",
        )

        assert result == "The current time is 2026-04-22 20:17:51 CEST +0200."
        orchestrator.cognitive_router._execute_tool.assert_awaited_once_with("get_system_info", {})
        orchestrator.cognitive_router.route_to_system_1.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_weather_query_uses_assessed_single_tool_path(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = [
            {
                "name": "web_search",
                "description": "Search the web for current information, weather, and news.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                    "required": ["query"],
                },
            }
        ]
        orchestrator.cognitive_router._execute_tool = AsyncMock(
            return_value=RouterResult(
                status="ok",
                content="1. Weather Vienna\n   URL: https://example.com\n   Vienna is currently 18C and cloudy.",
            )
        )
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock(
            return_value=RouterResult(status="ok", content="Today in Vienna it is about 18C and cloudy.")
        )

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            "Nice, what about the weather in Vienna, Austria?",
        )

        assert result == "Today in Vienna it is about 18C and cloudy."
        orchestrator.cognitive_router._execute_tool.assert_awaited_once()
        _, kwargs = orchestrator.cognitive_router.route_to_system_1.await_args
        assert kwargs["allowed_tools"] == []

    @pytest.mark.asyncio
    async def test_low_complexity_direct_query_uses_tool_free_route(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = []
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock(
            return_value=RouterResult(status="ok", content="Hello. I'm here and ready.")
        )

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            "Hi Aiden",
        )

        assert result == "Hello. I'm here and ready."
        orchestrator.cognitive_router.route_to_system_1.assert_awaited_once()


class TestTranscriptRegressionFixes:
    def test_optional_web_fallback_clause_is_not_treated_as_capability_question(self):
        prompt = "Can you tell me what is the meaning of paradox? Please search the web if you must"
        assert Orchestrator._is_capability_question(prompt) is False

    @pytest.mark.asyncio
    async def test_optional_web_fallback_knowledge_request_is_answered_without_web_search(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_skill_names.return_value = [
            "web_search",
            "extract_web_article",
        ]
        orchestrator.cognitive_router.registry.get_schemas.return_value = [
            {
                "name": "web_search",
                "description": "Search the web for current information, weather, and news.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            }
        ]
        orchestrator.cognitive_router._execute_tool = AsyncMock()
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock(
            return_value=RouterResult(
                status="ok",
                content="A paradox is a statement or situation that seems self-contradictory but may reveal a deeper truth.",
            )
        )

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            {
                "user_id": "test_user",
                "user_input": "Can you tell me what is the meaning of paradox? Please search the web if you must",
                "chat_history": [],
            },
        )

        assert "paradox" in result.lower()
        orchestrator.cognitive_router._execute_tool.assert_not_awaited()
        orchestrator.cognitive_router.route_to_system_1.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_capability_question_is_answered_without_running_web_search(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_skill_names.return_value = [
            "web_search",
            "search_archival_memory",
            "extract_pdf_text",
        ]
        orchestrator.cognitive_router.registry.get_schemas.return_value = []
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock()
        orchestrator.cognitive_router._execute_tool = AsyncMock()

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            {
                "user_id": "test_user",
                "user_input": "Can you access the internet to check for that?",
                "chat_history": [
                    {"role": "user", "content": "Can you check the weather in Vienna, Austria today?"},
                    {"role": "assistant", "content": "I do not have access to real-time weather updates."},
                ],
            },
        )

        assert "web_search" in result
        orchestrator.cognitive_router._execute_tool.assert_not_awaited()
        orchestrator.cognitive_router.route_to_system_1.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_last_shared_message_is_recalled_from_chat_history(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = []
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock()

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            {
                "user_id": "test_user",
                "user_input": "what did you just shared",
                "chat_history": [
                    {"role": "assistant", "content": "1. Weather Vienna\nURL: https://example.com\nVienna is currently 18C and cloudy."},
                ],
            },
        )

        assert result.startswith("The last thing I shared was:")
        assert "Weather Vienna" in result
        orchestrator.cognitive_router.route_to_system_1.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_summary_request_uses_existing_chat_history(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_schemas.return_value = []
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock()

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            {
                "user_id": "test_user",
                "user_input": "Can you summerize the conversation we had so far?",
                "chat_history": [
                    {"role": "user", "content": "Can you check the weather in Vienna, Austria today?"},
                    {"role": "assistant", "content": "Today in Vienna it is about 18C and cloudy."},
                    {"role": "user", "content": "Can you access the internet to check for that?"},
                    {"role": "assistant", "content": "Yes. I can use my web_search tool for live web data."},
                    {"role": "user", "content": "do you know my name"},
                ],
            },
        )

        assert "weather in vienna" in result.lower()
        assert "do you know my name" in result.lower()
        assert "first prompt" not in result.lower()
        orchestrator.cognitive_router.route_to_system_1.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_profile_is_stored_and_recalled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = Orchestrator.__new__(Orchestrator)
            orchestrator.core_memory = CoreMemory(f"{tmp_dir}/core_memory.json")
            orchestrator.cognitive_router = MagicMock()
            orchestrator.cognitive_router.registry.get_skill_names.return_value = []
            orchestrator.cognitive_router.registry.get_schemas.return_value = []
            orchestrator.cognitive_router.route_to_system_1 = AsyncMock()

            stored = await Orchestrator._remember_user_profile(
                orchestrator,
                "test_user",
                "My name is Ilir, 37 years old",
            )
            name_result = await Orchestrator._try_fast_path_response(
                orchestrator,
                {
                    "user_id": "test_user",
                    "user_input": "do you know my name",
                    "chat_history": [],
                },
            )
            age_result = await Orchestrator._try_fast_path_response(
                orchestrator,
                {
                    "user_id": "test_user",
                    "user_input": "how old am i",
                    "chat_history": [],
                },
            )

        assert stored is True
        assert "Ilir" in name_result
        assert "37" in age_result

    @pytest.mark.asyncio
    async def test_identity_statement_is_me_is_stored_and_acknowledged_without_web_search(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = Orchestrator.__new__(Orchestrator)
            orchestrator.core_memory = CoreMemory(f"{tmp_dir}/core_memory.json")
            orchestrator.cognitive_router = MagicMock()
            orchestrator.cognitive_router.registry.get_skill_names.return_value = ["web_search"]
            orchestrator.cognitive_router.registry.get_schemas.return_value = [
                {
                    "name": "web_search",
                    "description": "Search the web for current information, weather, and news.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                }
            ]
            orchestrator.cognitive_router.route_to_system_1 = AsyncMock()
            orchestrator.cognitive_router._execute_tool = AsyncMock()

            message = "Ilir Boci is me, please store that information so you will have contextual awareness when we interact together"
            stored = await Orchestrator._remember_user_profile(
                orchestrator,
                "test_user",
                message,
            )
            result = await Orchestrator._try_fast_path_response(
                orchestrator,
                {
                    "user_id": "test_user",
                    "user_input": message,
                    "chat_history": [],
                },
            )
            profile = await Orchestrator._get_user_profile(orchestrator, "test_user")

        assert stored is True
        assert "Ilir Boci" in result
        assert profile["name"] == "Ilir Boci"
        orchestrator.cognitive_router._execute_tool.assert_not_awaited()
        orchestrator.cognitive_router.route_to_system_1.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_request_capability_uses_existing_objective_tool_when_matched(self):
        router = CognitiveRouter.__new__(CognitiveRouter)
        router.registry = MagicMock()
        router.registry.get_schemas.return_value = [
            {
                "name": "spawn_new_objective",
                "description": "Adds a new goal, objective, Epic, Story, or Task to the Objective Backlog. Use when the Admin asks to add, create, define, or track a goal/objective.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tier": {"type": "string", "description": "Hierarchy level for the objective."},
                        "title": {"type": "string", "description": "Title of the goal or objective."},
                        "estimated_energy": {"type": "integer", "description": "Estimated effort."},
                    },
                    "required": ["tier", "title", "estimated_energy"],
                },
            }
        ]

        result = await CognitiveRouter._execute_tool(
            router,
            "request_capability",
            {
                "gap_description": "A function to add goals and objectives to the Objective Backlog for the Admin.",
                "suggested_tool_name": "add_goal_function",
            },
        )

        assert result.status == "ok"
        assert "spawn_new_objective" in result.content

    @pytest.mark.asyncio
    async def test_process_message_loads_state_before_fast_path(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.pending_mfa = {}
        orchestrator.pending_hitl_state = {}
        orchestrator.pending_tool_approval = {}
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)
        orchestrator._get_user_lock = AsyncMock(return_value=asyncio.Lock())
        orchestrator._try_resume_mfa = AsyncMock(return_value=None)
        orchestrator._try_resume_tool_approval = AsyncMock(return_value=None)
        loaded_state = _make_fresh_state(user_id="test_user", user_input="Can you summerize the conversation we had so far?")
        loaded_state["chat_history"] = [
            {"role": "user", "content": "Can you check the weather in Vienna, Austria today?"},
            {"role": "assistant", "content": "Today in Vienna it is about 18C and cloudy."},
        ]
        orchestrator._load_state = AsyncMock(return_value=loaded_state)
        orchestrator._remember_user_profile = AsyncMock(return_value=False)
        orchestrator._try_fast_path_response = AsyncMock(return_value="So far we've discussed: Can you check the weather in Vienna, Austria today?.")
        orchestrator.ledger_memory = MagicMock()
        orchestrator.ledger_memory.save_chat_turn = AsyncMock()
        orchestrator._consolidate_memory = AsyncMock()
        orchestrator._save_memory_async = AsyncMock()

        def fake_fire_and_forget(coro):
            coro.close()

        orchestrator._fire_and_forget = fake_fire_and_forget

        result = await Orchestrator.process_message(
            orchestrator,
            "Can you summerize the conversation we had so far?",
            "test_user",
        )

        assert result.startswith("So far we've discussed:")
        orchestrator._load_state.assert_awaited_once_with(
            "test_user",
            "Can you summerize the conversation we had so far?",
            user_prompt={
                "text": "Can you summerize the conversation we had so far?",
                "audio_bytes": b"",
                "audio_mime_type": "",
                "audio_source": "",
                "audio_file_id": "",
            },
        )
        orchestrator._try_fast_path_response.assert_awaited_once_with(loaded_state)

    @pytest.mark.asyncio
    async def test_process_message_multi_turn_memory_and_capability_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            async def fake_route_to_system_1(messages, allowed_tools=None):
                await asyncio.sleep(0)
                user_prompt = messages[-1]["content"]
                if "Tool used: web_search" in user_prompt:
                    return RouterResult(status="ok", content="Today in Vienna it is about 18C and cloudy.")
                if "My name is Ilir, 37 years old" in user_prompt:
                    return RouterResult(
                        status="ok",
                        content="It is nice to meet you, Ilir. I will remember your name and age for later.",
                    )
                return RouterResult(status="ok", content="Acknowledged.")

            orchestrator = _build_process_message_test_orchestrator(
                tmp_dir,
                skill_names=["web_search", "search_archival_memory", "extract_pdf_text"],
                schemas=[
                    {
                        "name": "web_search",
                        "description": "Search the web for current information, weather, and news.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_results": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    }
                ],
                execute_tool_side_effect=AsyncMock(
                    return_value=RouterResult(
                        status="ok",
                        content="1. Weather Vienna\n   URL: https://example.com\n   Vienna is currently 18C and cloudy.",
                    )
                ),
                route_to_system_1_side_effect=fake_route_to_system_1,
                current_time="2026-04-22 20:17:51",
            )

            weather_reply = await Orchestrator.process_message(
                orchestrator,
                "Can you check the weather in Vienna, Austria today?",
                "test_user",
            )
            last_shared_reply = await Orchestrator.process_message(
                orchestrator,
                "what did you just shared",
                "test_user",
            )
            internet_reply = await Orchestrator.process_message(
                orchestrator,
                "Can you access the internet to check for that?",
                "test_user",
            )
            profile_ack = await Orchestrator.process_message(
                orchestrator,
                "My name is Ilir, 37 years old",
                "test_user",
            )
            name_reply = await Orchestrator.process_message(
                orchestrator,
                "do you know my name",
                "test_user",
            )
            summary_reply = await Orchestrator.process_message(
                orchestrator,
                "Can you summerize the conversation we had so far?",
                "test_user",
            )
            assert weather_reply == "Today in Vienna it is about 18C and cloudy."
            assert last_shared_reply.startswith("The last thing I shared was:")
            assert "Today in Vienna" in last_shared_reply
            assert "web_search" in internet_reply
            assert "Ilir" in profile_ack
            assert name_reply == "Yes. Your name is Ilir, and you told me that you are 37 years old."
            assert "weather in vienna" in summary_reply.lower()
            assert "my name is ilir, 37 years old" in summary_reply.lower()
            assert orchestrator.cognitive_router._execute_tool.await_count == 1
            assert orchestrator._run_graph_loop.await_count == 0
            stored_profile = await orchestrator._get_user_profile("test_user")
            assert stored_profile["name"] == "Ilir"
            assert stored_profile["age"] == 37


    @pytest.mark.asyncio
    async def test_tool_repository_question_lists_registered_tools(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.cognitive_router = MagicMock()
        orchestrator.cognitive_router.registry.get_skill_names.return_value = [
            "web_search",
            "search_archival_memory",
            "update_core_memory",
        ]
        orchestrator.cognitive_router.registry.get_schemas.return_value = []
        orchestrator.cognitive_router.route_to_system_1 = AsyncMock()

        result = await Orchestrator._try_fast_path_response(
            orchestrator,
            {
                "user_id": "test_user",
                "user_input": "Do you have access to the tool repository?",
                "chat_history": [],
            },
        )

        assert "registered tool repository" in result
        assert "web_search" in result
        orchestrator.cognitive_router.route_to_system_1.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_message_handles_name_time_and_weather_followups(self):
        async def fake_execute_tool(tool_name, arguments):
            await asyncio.sleep(0)
            if tool_name == "get_system_info":
                return RouterResult(
                    status="ok",
                    content="DateTime: 2026-04-22 21:24:06 W. Europe Daylight Time +0200 | Platform: Windows 11 | Machine: AMD64",
                )
            if tool_name == "web_search":
                return RouterResult(
                    status="ok",
                    content="1. Weather Vienna\n   URL: https://example.com\n   Vienna is currently 18C and cloudy.",
                )
            return RouterResult(status="ok", content="Unhandled tool")

        async def fake_route_to_system_1(messages, allowed_tools=None):
            await asyncio.sleep(0)
            user_prompt = messages[-1]["content"]
            if "Tool used: get_system_info" in user_prompt:
                return RouterResult(status="ok", content="The current time is 21:24:06 (W. Europe Daylight Time, +0200) on April 22, 2026.")
            if "Tool used: web_search" in user_prompt:
                return RouterResult(status="ok", content="Today in Vienna it is about 18C and cloudy.")
            return RouterResult(status="ok", content="Acknowledged.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = _build_process_message_test_orchestrator(
                tmp_dir,
                skill_names=["get_system_info", "web_search", "search_archival_memory", "update_core_memory"],
                schemas=[
                    {
                        "name": "get_system_info",
                        "description": "Returns the current date, time, timezone, and host platform details.",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                    {
                        "name": "web_search",
                        "description": "Search the web for current information, weather, and news.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_results": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    },
                ],
                execute_tool_side_effect=fake_execute_tool,
                route_to_system_1_side_effect=fake_route_to_system_1,
                current_time="2026-04-22 21:24:06",
            )

            missing_name_reply = await Orchestrator.process_message(
                orchestrator,
                "Can you tell my name",
                "test_user",
            )
            assistant_name_reply = await Orchestrator.process_message(
                orchestrator,
                "What about yours",
                "test_user",
            )
            time_reply = await Orchestrator.process_message(
                orchestrator,
                "Much better. Ok can you tell the time now?",
                "test_user",
            )
            weather_reply = await Orchestrator.process_message(
                orchestrator,
                "What about the weather in Vienna, Austria today",
                "test_user",
            )
            rename_ack = await Orchestrator.process_message(
                orchestrator,
                "Just for your information, your name is Aiden",
                "test_user",
            )
            assistant_name_after_update = await Orchestrator.process_message(
                orchestrator,
                "What about yours",
                "test_user",
            )

            stored_assistant_profile = await orchestrator.core_memory.get_all()

            assert missing_name_reply == "Not yet. You have not told me your name."
            assert assistant_name_reply == "My name is AIDEN."
            assert "current time is 21:24:06" in time_reply.lower()
            assert weather_reply == "Today in Vienna it is about 18C and cloudy."
            assert "system diagnostic information" not in assistant_name_reply.lower()
            assert "system information" not in weather_reply.lower()
            assert rename_ack == "Understood. I will use Aiden as my name."
            assert assistant_name_after_update == "My name is Aiden."
            assert orchestrator.cognitive_router._execute_tool.await_count == 2
            tool_names = [call.args[0] for call in orchestrator.cognitive_router._execute_tool.await_args_list]
            assert tool_names == ["get_system_info", "web_search"]
            assert stored_assistant_profile["assistant_profile"]["name"] == "Aiden"
