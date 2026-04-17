"""
tests/test_router_interceptor.py
=================================
Deterministic unit tests for the CognitiveRouter parsing layer.

All methods under test are pure static methods that never touch the network,
so no mocking of Ollama / Groq is required.  Tests are grouped by method.
"""
import json
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
        """JSON blob after prose text should still be found."""
        text = 'I will search for you.\n{"tool_name": "web_search", "arguments": {"query": "test"}}'
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, _ = result
        assert name == "web_search"

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
        """Model wraps tool call JSON in <tool_code>…</tool_code> tags."""
        text = (
            "Sure, let me look that up.\n"
            "<tool_code>\n"
            '{"tool_name": "web_search", "arguments": {"query": "Vienna weather"}}\n'
            "</tool_code>"
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "Vienna weather"

    def test_xml_tool_code_tag_correctly_parses_tool_name(self):
        """<tool_code> block with only tool_name (no args) must be parsed."""
        text = (
            "Here is my answer.\n"
            "<tool_code>\n"
            '{"tool_name": "get_system_info"}\n'
            "</tool_code>\n"
            "Done."
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, _ = result
        assert name == "get_system_info"

    def test_xml_tool_call_tag_alias(self):
        """Some models use <tool_call> instead of <tool_code>."""
        text = (
            "<tool_call>\n"
            '{"name": "web_search", "arguments": {"query": "hello"}}\n'
            "</tool_call>"
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "hello"

    def test_xml_block_does_not_leak_into_result(self):
        """
        When the tool call is embedded in XML tags, the JSON inside must be
        found — verifying the parser doesn't choke on the surrounding XML.
        """
        text = (
            "Let me check the price for you.\n"
            "<tool_code>\n"
            '{"tool_name": "get_stock_price", "arguments": {"ticker": "AAPL"}}\n'
            "</tool_code>"
        )
        result = CognitiveRouter._extract_inline_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "get_stock_price"
        assert args == {"ticker": "AAPL"}


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
