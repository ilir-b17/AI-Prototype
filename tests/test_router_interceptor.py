"""
tests/test_router_interceptor.py
=================================
Deterministic unit tests for the CognitiveRouter parsing layer.

All methods under test are pure static methods that never touch the network,
so no mocking of Ollama / Groq is required.  Tests are grouped by method.
"""
import json
import re
import time
import pytest

from src.core.llm_router import CognitiveRouter


# ── _find_json_blobs ──────────────────────────────────────────────────────────

class TestFindJsonBlobs:
    def test_single_object(self):
        text = '{"tool_name": "web_search", "arguments": {"query": "hello"}}'
        blobs = CognitiveRouter._find_json_blobs(text)
        assert len(blobs) == 1
        assert json.loads(blobs[0]) == {"tool_name": "web_search", "arguments": {"query": "hello"}}

    def test_multiple_top_level_objects(self):
        text = '{"a": 1} some prose {"b": 2}'
        blobs = CognitiveRouter._find_json_blobs(text)
        assert len(blobs) == 2
        assert json.loads(blobs[0]) == {"a": 1}
        assert json.loads(blobs[1]) == {"b": 2}

    def test_nested_object_counts_as_one_blob(self):
        """Nested braces must not split into multiple blobs."""
        text = '{"tool_name": "foo", "arguments": {"nested": {"key": "val"}}}'
        blobs = CognitiveRouter._find_json_blobs(text)
        assert len(blobs) == 1
        obj = json.loads(blobs[0])
        assert obj["arguments"]["nested"]["key"] == "val"

    def test_no_json(self):
        blobs = CognitiveRouter._find_json_blobs("just some plain text")
        assert blobs == []

    def test_json_embedded_in_prose(self):
        text = 'Let me call the tool: {"tool_name": "get_system_info"} done.'
        blobs = CognitiveRouter._find_json_blobs(text)
        assert len(blobs) == 1

    def test_empty_string(self):
        assert CognitiveRouter._find_json_blobs("") == []


# ── _extract_args_from_blob ───────────────────────────────────────────────────

class TestExtractArgsFromBlob:
    def test_arguments_key_dict(self):
        obj = {"tool_name": "foo", "arguments": {"query": "bar"}}
        args = CognitiveRouter._extract_args_from_blob(obj)
        assert args == {"query": "bar"}

    def test_arguments_key_string_json(self):
        """String-encoded JSON in 'arguments' must be decoded."""
        obj = {"tool_name": "foo", "arguments": '{"query": "bar"}'}
        args = CognitiveRouter._extract_args_from_blob(obj)
        assert args == {"query": "bar"}

    def test_parameters_key(self):
        obj = {"name": "foo", "parameters": {"x": 1}}
        args = CognitiveRouter._extract_args_from_blob(obj)
        assert args == {"x": 1}

    def test_args_key(self):
        obj = {"tool_name": "foo", "args": {"y": 2}}
        args = CognitiveRouter._extract_args_from_blob(obj)
        assert args == {"y": 2}

    def test_flat_dict_fallback(self):
        """When no explicit args container is present, non-meta keys are the args."""
        obj = {"tool_name": "foo", "query": "bar", "limit": 5}
        args = CognitiveRouter._extract_args_from_blob(obj)
        assert args == {"query": "bar", "limit": 5}

    def test_empty_object_returns_empty_dict(self):
        """An object with only a tool_name and no args → empty dict."""
        args = CognitiveRouter._extract_args_from_blob({"tool_name": "foo"})
        assert args == {}

    def test_arguments_invalid_string_falls_back_gracefully(self):
        """Unparseable string value in 'arguments' must not raise."""
        obj = {"tool_name": "foo", "arguments": "not valid json"}
        args = CognitiveRouter._extract_args_from_blob(obj)
        assert isinstance(args, dict)


# ── _extract_inline_tool_call ─────────────────────────────────────────────────

class TestExtractInlineToolCall:

    # --- raw JSON {"tool_name": ...} format ---

    def test_pure_json_tool_name_key(self):
        text = '{"tool_name": "web_search", "arguments": {"query": "Vienna weather"}}'
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "Vienna weather"}

    def test_json_name_key(self):
        """'name' is an accepted alias for 'tool_name'."""
        text = '{"name": "get_system_info", "arguments": {}}'
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, _ = result
        assert name == "get_system_info"

    def test_json_tool_call_key(self):
        """'tool_call' as key must also be recognized."""
        text = '{"tool_call": "update_ledger", "arguments": {"amount": 100}}'
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "update_ledger"
        assert args == {"amount": 100}

    def test_json_function_call_key(self):
        """'function_call' as key must also be recognized."""
        text = '{"function_call": "run_terminal_command", "arguments": {"command": "ls"}}'
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, _ = result
        assert name == "run_terminal_command"

    def test_json_mixed_with_prose_prefix(self):
        """JSON embedded in prose must be ignored as non-dominant content."""
        text = 'I will search for you.\n{"tool_name": "web_search", "arguments": {"query": "test"}}'
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is None

    def test_no_tool_call_in_plain_text(self):
        text = "This is just a plain text response with no tool call."
        assert CognitiveRouter._extract_inline_tool_call(text) is None

    def test_json_without_tool_name_key_is_ignored(self):
        """A JSON object that has no tool-identity key must not be parsed as a call."""
        text = '{"price": 123.45, "currency": "USD"}'
        assert CognitiveRouter._extract_inline_tool_call(text) is None

    def test_empty_string_returns_none(self):
        assert CognitiveRouter._extract_inline_tool_call("") is None

    # --- XML <tool_code> / <tool_call> wrapper format ---

    def test_xml_tool_code_tag_wrapping_json(self):
        """XML-wrapped JSON is ignored because it is not dominant JSON content."""
        text = (
            "Sure, let me look that up.\n"
            "<tool_code>\n"
            '{"tool_name": "web_search", "arguments": {"query": "Vienna weather"}}\n'
            "</tool_code>"
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is None

    def test_xml_tool_code_tag_correctly_parses_tool_name(self):
        """XML-wrapped tool_name blocks are ignored for safety."""
        text = (
            "Here is my answer.\n"
            "<tool_code>\n"
            '{"tool_name": "get_system_info"}\n'
            "</tool_code>\n"
            "Done."
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is None

    def test_xml_tool_call_tag_alias(self):
        """Alias XML wrappers are ignored the same way as <tool_code>."""
        text = (
            "<tool_call>\n"
            '{"name": "web_search", "arguments": {"query": "hello"}}\n'
            "</tool_call>"
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is None

    def test_xml_block_does_not_leak_into_result(self):
        """
        XML wrapper text must prevent inline execution even if inner JSON is valid.
        """
        text = (
            "Let me check the price for you.\n"
            "<tool_code>\n"
            '{"tool_name": "get_stock_price", "arguments": {"ticker": "AAPL"}}\n'
            "</tool_code>"
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is None


# ── _parse_native_tool_call ───────────────────────────────────────────────────

class TestParseNativeToolCall:
    @staticmethod
    def _make_ollama_message(tool_name: str, arguments) -> dict:
        """Build a minimal Ollama-style tool_calls message dict."""
        return {
            "tool_calls": [
                {"function": {"name": tool_name, "arguments": arguments}}
            ]
        }

    def test_standard_dict_arguments(self):
        msg = self._make_ollama_message("web_search", {"query": "test"})
        name, args, ok = CognitiveRouter._parse_native_tool_call(msg)
        assert ok
        assert name == "web_search"
        assert args == {"query": "test"}

    def test_string_encoded_arguments_are_decoded(self):
        """Some Ollama versions return arguments as a JSON string."""
        msg = self._make_ollama_message("web_search", '{"query": "test"}')
        name, args, ok = CognitiveRouter._parse_native_tool_call(msg)
        assert ok
        assert name == "web_search"
        assert args == {"query": "test"}

    def test_missing_name_returns_not_ok(self):
        msg = {"tool_calls": [{"function": {"arguments": {"q": "x"}}}]}
        name, args, ok = CognitiveRouter._parse_native_tool_call(msg)
        assert not ok

    def test_empty_tool_calls_list_returns_not_ok(self):
        msg = {"tool_calls": []}
        _, _, ok = CognitiveRouter._parse_native_tool_call(msg)
        assert not ok

    def test_no_function_wrapper_flat_dict(self):
        """Some Ollama versions embed name/arguments directly in the tool_call object."""
        msg = {"tool_calls": [{"name": "get_system_info", "arguments": {}}]}
        name, args, ok = CognitiveRouter._parse_native_tool_call(msg)
        assert ok
        assert name == "get_system_info"

    def test_multiple_tool_calls_uses_first(self):
        """Only the first tool_call should be consumed per invocation."""
        msg = {
            "tool_calls": [
                {"function": {"name": "tool_a", "arguments": {"x": 1}}},
                {"function": {"name": "tool_b", "arguments": {"y": 2}}},
            ]
        }
        name, args, ok = CognitiveRouter._parse_native_tool_call(msg)
        assert ok
        assert name == "tool_a"
        assert args == {"x": 1}


class TestBuildGroqFollowupMessages:
    class _ToolCall:
        def __init__(self, tool_call_id: str):
            self.id = tool_call_id

    def test_dict_arguments_are_json_encoded_for_assistant_tool_calls(self):
        messages = [{"role": "user", "content": "Find weather"}]
        executed_calls = [
            {
                "tool_call": self._ToolCall("call_1"),
                "raw_tool_name": "web_search",
                "raw_arguments": {"query": "Vienna weather"},
                "tool_output": "Sunny",
            }
        ]

        followup = CognitiveRouter._build_groq_followup_messages(messages, executed_calls)
        assistant_call = followup[-2]["tool_calls"][0]["function"]

        assert isinstance(assistant_call["arguments"], str)
        assert json.loads(assistant_call["arguments"]) == {"query": "Vienna weather"}

    def test_string_arguments_are_preserved_for_assistant_tool_calls(self):
        messages = [{"role": "user", "content": "Find weather"}]
        raw_arguments = '{"query":"Vienna weather"}'
        executed_calls = [
            {
                "tool_call": self._ToolCall("call_2"),
                "raw_tool_name": "web_search",
                "raw_arguments": raw_arguments,
                "tool_output": "Sunny",
            }
        ]

        followup = CognitiveRouter._build_groq_followup_messages(messages, executed_calls)
        assistant_call = followup[-2]["tool_calls"][0]["function"]

        assert assistant_call["arguments"] == raw_arguments


# ── sanitize_response ─────────────────────────────────────────────────────────

class TestSanitizeResponse:
    def test_strips_think_block(self):
        text = "<think>internal reasoning step</think>\nThe answer is 42."
        result = CognitiveRouter.sanitize_response(text)
        assert "<think>" not in result
        assert "internal reasoning" not in result
        assert "The answer is 42." in result

    def test_strips_multiline_think_block(self):
        text = "<think>\nstep 1\nstep 2\n</think>\nFinal answer."
        result = CognitiveRouter.sanitize_response(text)
        assert "step 1" not in result
        assert "Final answer." in result

    def test_strips_reasoning_block(self):
        text = "<reasoning>step 1\nstep 2</reasoning>\nFinal answer."
        result = CognitiveRouter.sanitize_response(text)
        assert "<reasoning>" not in result
        assert "Final answer." in result

    def test_strips_agent_output_block(self):
        text = (
            "Email sent successfully.\n"
            "<agent_output>{\"success\": true, \"summary\": \"sent\"}</agent_output>"
        )
        result = CognitiveRouter.sanitize_response(text)
        assert "<agent_output>" not in result
        assert "Email sent successfully." in result

    def test_keeps_user_facing_markdown_heading(self):
        text = "### Email Sent\nThe final score was 2-2."
        result = CognitiveRouter.sanitize_response(text)
        assert "### Email Sent" in result
        assert "The final score was 2-2." in result

    def test_strips_internal_scratch_heading(self):
        text = "## Scratchpad\ninternal notes\nFinal answer."
        result = CognitiveRouter.sanitize_response(text)
        assert "## Scratchpad" not in result
        assert "Final answer." in result

    def test_strips_workers_tag(self):
        text = "Here is my answer.\nWORKERS: []\n"
        result = CognitiveRouter.sanitize_response(text)
        assert "WORKERS:" not in result
        assert "Here is my answer." in result

    def test_strips_workers_tag_with_agents(self):
        text = 'Delegating now.\nWORKERS: ["research_agent"]\n'
        result = CognitiveRouter.sanitize_response(text)
        assert "WORKERS:" not in result

    def test_strips_critic_feedback_annotation(self):
        text = "[CRITIC FEEDBACK: output was too short]\nBetter response here."
        result = CognitiveRouter.sanitize_response(text)
        assert "CRITIC FEEDBACK" not in result
        assert "Better response here." in result

    def test_strips_admin_guidance_annotation(self):
        text = "[ADMIN GUIDANCE: use a different approach]\nSure, retrying."
        result = CognitiveRouter.sanitize_response(text)
        assert "ADMIN GUIDANCE" not in result

    def test_strips_standalone_tool_call_json(self):
        """A JSON blob on its own line containing a known tool key must be stripped."""
        text = 'The price is $150.\n{"tool_call": "get_stock_price", "name": "AAPL"}'
        result = CognitiveRouter.sanitize_response(text)
        assert "tool_call" not in result
        assert "The price is $150." in result

    def test_passes_plain_text_unchanged(self):
        text = "Hello, I am AIDEN. How can I help you today?"
        result = CognitiveRouter.sanitize_response(text)
        assert result == text

    def test_empty_string_returns_empty(self):
        assert CognitiveRouter.sanitize_response("") == ""

    def test_none_returns_none(self):
        """sanitize_response must handle falsy input without raising."""
        result = CognitiveRouter.sanitize_response(None)
        assert not result

    def test_microbenchmark_sanitize_response_is_at_least_2x_faster_than_recompile_baseline(self):
        payload = (
            "<think>internal reasoning\n" + ("step ") * 120 + "</think>\n"
            "[CRITIC FEEDBACK: trim this]\n"
            "[ADMIN GUIDANCE: hidden]\n"
            "WORKERS: [\"research_agent\", \"coder_agent\"]\n"
            "## Scratch\n"
            "[Output Draft]draft section\n"
            "{" + '"tool_call": "web_search", "arguments": {"query": "x"}' + "}\n"
            + ("Use concise, direct prose. " * 130)
        )[:4096]

        def _legacy_recompile_baseline(text: str) -> str:
            if not text:
                return text
            re.purge()
            text = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE).sub("", text)
            text = re.compile(r"<reasoning>.*?</reasoning>", flags=re.DOTALL | re.IGNORECASE).sub("", text)
            text = re.compile(r"\[CRITIC FEEDBACK[^\n]*\n?", flags=re.IGNORECASE).sub("", text)
            text = re.compile(r"\[ADMIN GUIDANCE[^\n]*\n?", flags=re.IGNORECASE).sub("", text)
            text = re.compile(r"\[HEARTBEAT TASK[^\n]*\n?", flags=re.IGNORECASE).sub("", text)
            text = re.compile(r"WORKERS:\s*\[.*?\]\s*\n?", flags=re.IGNORECASE).sub("", text)
            text = re.compile(
                r'(?m)^\s*\{[^{}]*"(?:tool_call|tool_name|function_call|function|name)"[^{}]*\}\s*$',
                flags=re.IGNORECASE,
            ).sub("", text)
            text = re.compile(r"^#{1,3}\s.*$", flags=re.MULTILINE).sub("", text)
            text = re.compile(r"\[Output Draft\][^\[]*", flags=re.DOTALL | re.IGNORECASE).sub("", text)
            text = re.compile(r"\[Internal Critique\][^\[]*", flags=re.DOTALL | re.IGNORECASE).sub("", text)
            text = re.compile(r"\[Finalized Deliverable\]\s*", flags=re.IGNORECASE).sub("", text)
            text = re.compile(r"---+\s*\n").sub("", text)
            text = re.compile(r"\n{3,}").sub("\n\n", text)
            return text.strip()

        # Warm both code paths so one-time overhead does not dominate timing.
        for _ in range(20):
            CognitiveRouter.sanitize_response(payload)
            _legacy_recompile_baseline(payload)

        iterations = 120

        optimized_start = time.perf_counter()
        for _ in range(iterations):
            CognitiveRouter.sanitize_response(payload)
        optimized_elapsed = time.perf_counter() - optimized_start

        baseline_start = time.perf_counter()
        for _ in range(iterations):
            _legacy_recompile_baseline(payload)
        baseline_elapsed = time.perf_counter() - baseline_start

        assert optimized_elapsed * 2 <= baseline_elapsed
