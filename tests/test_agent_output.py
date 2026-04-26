"""
Tests for the formal agent communication protocol.
Tier 1: pure function tests - no LLM, no IO.
Tier 2: integration with orchestrator state (no live Ollama).
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Dict, List, Optional


# -- AgentOutputType resolution ---------------------------------------

def test_resolve_output_type_valid():
    from src.core.agent_output import AgentOutputType, resolve_output_type
    assert resolve_output_type("research") == AgentOutputType.RESEARCH
    assert resolve_output_type("CODER") == AgentOutputType.CODER
    assert resolve_output_type("synthesis") == AgentOutputType.SYNTHESIS
    assert resolve_output_type("text") == AgentOutputType.TEXT


def test_resolve_output_type_fallback():
    from src.core.agent_output import AgentOutputType, resolve_output_type
    assert resolve_output_type("unknown") == AgentOutputType.TEXT
    assert resolve_output_type(None) == AgentOutputType.TEXT
    assert resolve_output_type("") == AgentOutputType.TEXT
    assert resolve_output_type(42) == AgentOutputType.TEXT


# -- XML extraction ---------------------------------------------------

def test_extract_valid_research_output():
    from src.core.agent_output import extract_structured_output
    text = '''
I found relevant information in archival memory.
<agent_output>
{"success": true, "confidence": 0.85, "summary": "Found energy docs",
 "facts": ["Fact A", "Fact B"], "sources": ["search_archival_memory"],
 "gaps": ["Could not find X"], "suggested_followup_tools": []}
</agent_output>
'''
    result = extract_structured_output(text)
    assert result is not None
    assert result["confidence"] == pytest.approx(0.85)
    assert "Fact A" in result["facts"]
    assert "Could not find X" in result["gaps"]


def test_extract_no_block_returns_none():
    from src.core.agent_output import extract_structured_output
    assert extract_structured_output("Just plain text") is None
    assert extract_structured_output("") is None
    assert extract_structured_output(None) is None


def test_extract_empty_tags_returns_none():
    from src.core.agent_output import extract_structured_output
    assert extract_structured_output("<agent_output></agent_output>") is None


def test_extract_invalid_json_returns_none():
    from src.core.agent_output import extract_structured_output
    text = "<agent_output>not valid json{{{</agent_output>"
    assert extract_structured_output(text) is None


def test_extract_markdown_wrapped_json():
    from src.core.agent_output import extract_structured_output
    text = (
        "<agent_output>\n```json\n"
        '{"success": true, "confidence": 0.7, "summary": "ok"}\n'
        "```\n</agent_output>"
    )
    result = extract_structured_output(text)
    assert result is not None
    assert result["confidence"] == pytest.approx(0.7)


def test_extract_case_insensitive_tags():
    from src.core.agent_output import extract_structured_output
    text = (
        "<AGENT_OUTPUT>"
        '{"success": true, "confidence": 0.5, "summary": "test"}'
        "</AGENT_OUTPUT>"
    )
    result = extract_structured_output(text)
    assert result is not None


def test_extract_non_dict_returns_none():
    from src.core.agent_output import extract_structured_output
    text = "<agent_output>[1, 2, 3]</agent_output>"
    assert extract_structured_output(text) is None


# -- strip_agent_output_block -----------------------------------------

def test_strip_removes_block():
    from src.core.agent_output import strip_agent_output_block
    text = "Useful text.\n<agent_output>{\"x\": 1}</agent_output>\nEnd."
    stripped = strip_agent_output_block(text)
    assert "<agent_output>" not in stripped
    assert "Useful text." in stripped


def test_strip_no_block_unchanged():
    from src.core.agent_output import strip_agent_output_block
    text = "Plain text with no block."
    assert strip_agent_output_block(text) == text.rstrip()


# -- validate_agent_output --------------------------------------------

def test_validate_research_result():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    raw = {
        "success": True,
        "confidence": 0.9,
        "summary": "Found data",
        "facts": ["Fact 1", "Fact 2"],
        "sources": ["search_archival_memory"],
        "gaps": [],
        "suggested_followup_tools": ["web_search"],
    }
    result = validate_agent_output(raw, AgentOutputType.RESEARCH, "research_agent")
    assert result is not None
    assert result["output_type"] == "research"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["facts"] == ["Fact 1", "Fact 2"]
    assert result["extraction_failed"] is False
    # Must be JSON-serialisable
    json.dumps(result)


def test_validate_coder_result():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    raw = {
        "success": True,
        "confidence": 0.85,
        "summary": "Updated memory",
        "actions_taken": ["Stored key X", "Spawned objective"],
        "memory_updates": ["core_memory.current_focus = 'energy'"],
        "objectives_created": ["Task #7: Run tests"],
        "errors": [],
        "files_modified": [],
    }
    result = validate_agent_output(raw, AgentOutputType.CODER, "coder_agent")
    assert result is not None
    assert result["actions_taken"] == ["Stored key X", "Spawned objective"]
    json.dumps(result)


def test_validate_synthesis_result():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    raw = {
        "success": True,
        "confidence": 0.88,
        "summary": "Combined answer",
        "final_answer": "The energy ROI threshold is 1.25.",
        "sources_cited": ["research_agent"],
        "caveats": [],
        "conflicts_noted": [],
    }
    result = validate_agent_output(raw, AgentOutputType.SYNTHESIS, "synthesis_agent")
    assert result is not None
    assert result["final_answer"] == "The energy ROI threshold is 1.25."
    json.dumps(result)


def test_validate_clips_confidence_out_of_range():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    raw = {
        "success": True,
        "confidence": 99.0,  # way out of range
        "summary": "ok",
        "facts": [],
        "sources": [],
        "gaps": [],
        "suggested_followup_tools": [],
    }
    result = validate_agent_output(raw, AgentOutputType.RESEARCH, "research_agent")
    assert result["confidence"] <= 1.0


def test_validate_none_returns_none():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    assert validate_agent_output(None, AgentOutputType.RESEARCH, "agent") is None


def test_validate_missing_fields_use_defaults():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    raw = {"success": True, "confidence": 0.5, "summary": "minimal"}
    result = validate_agent_output(raw, AgentOutputType.RESEARCH, "agent")
    assert result is not None
    assert result["facts"] == []
    assert result["gaps"] == []


# -- make_extraction_failed_result ------------------------------------

def test_extraction_failed_result_is_json_safe():
    from src.core.agent_output import AgentOutputType, make_extraction_failed_result
    result = make_extraction_failed_result(
        AgentOutputType.RESEARCH, "research_agent", "raw text here"
    )
    assert result["extraction_failed"] is True
    assert result["success"] is True
    json.dumps(result)


def test_extraction_failed_result_all_types():
    from src.core.agent_output import AgentOutputType, make_extraction_failed_result
    for output_type in AgentOutputType:
        result = make_extraction_failed_result(output_type, "agent", "raw")
        assert "extraction_failed" in result
        json.dumps(result)


# -- format_structured_for_handoff ------------------------------------

def test_format_research_handoff_shows_facts():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    from src.core.agent_output import format_structured_for_handoff
    raw = {
        "success": True,
        "confidence": 0.85,
        "summary": "Found energy docs",
        "facts": ["Energy ROI uses value/effort", "Default threshold 1.25"],
        "sources": ["search_archival_memory"],
        "gaps": ["Could not find RLIMIT details"],
        "suggested_followup_tools": ["web_search"],
    }
    structured = validate_agent_output(raw, AgentOutputType.RESEARCH, "research_agent")
    handoff = format_structured_for_handoff(structured, "research_agent")
    assert "research_agent" in handoff
    assert "Energy ROI" in handoff
    assert "Could not find RLIMIT" in handoff
    assert "85%" in handoff


def test_format_coder_handoff_shows_actions():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    from src.core.agent_output import format_structured_for_handoff
    raw = {
        "success": True,
        "confidence": 0.9,
        "summary": "Done",
        "actions_taken": ["Stored X to memory", "Created objective #5"],
        "memory_updates": ["core_memory.current_focus = 'energy'"],
        "objectives_created": ["#5: Run tests"],
        "errors": [],
        "files_modified": [],
    }
    structured = validate_agent_output(raw, AgentOutputType.CODER, "coder_agent")
    handoff = format_structured_for_handoff(structured, "coder_agent")
    assert "Stored X to memory" in handoff
    assert "Created objective #5" in handoff


def test_format_none_returns_empty():
    from src.core.agent_output import format_structured_for_handoff
    assert format_structured_for_handoff(None, "agent") == ""


def test_format_extraction_failed_notes_it():
    from src.core.agent_output import AgentOutputType, make_extraction_failed_result
    from src.core.agent_output import format_structured_for_handoff
    failed = make_extraction_failed_result(
        AgentOutputType.RESEARCH, "research_agent", "unstructured output"
    )
    handoff = format_structured_for_handoff(failed, "research_agent")
    assert "structured extraction failed" in handoff.lower()


# -- get_display_text -------------------------------------------------

def test_get_display_text_prefers_final_answer():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    from src.core.agent_output import get_display_text
    raw = {
        "success": True, "confidence": 0.9,
        "summary": "Summary text",
        "final_answer": "The real answer for the user.",
        "sources_cited": [], "caveats": [], "conflicts_noted": [],
    }
    structured = validate_agent_output(raw, AgentOutputType.SYNTHESIS, "synthesis_agent")
    result = get_display_text(structured, "Raw text fallback")
    assert result == "The real answer for the user."


def test_get_display_text_prefers_summary_over_raw():
    from src.core.agent_output import AgentOutputType, validate_agent_output
    from src.core.agent_output import get_display_text
    raw = {
        "success": True, "confidence": 0.8,
        "summary": "Clear summary of findings",
        "facts": ["Fact A"], "sources": [], "gaps": [],
        "suggested_followup_tools": [],
    }
    structured = validate_agent_output(raw, AgentOutputType.RESEARCH, "research_agent")
    research_text = "Full raw text with <agent_output>...</agent_output>"
    result = get_display_text(structured, research_text)
    assert result == "Clear summary of findings"


def test_get_display_text_falls_back_on_none():
    from src.core.agent_output import get_display_text
    result = get_display_text(None, "Raw text content")
    assert result == "Raw text content"


def test_get_display_text_strips_block_from_raw():
    from src.core.agent_output import get_display_text
    raw_text = "Useful part.\n<agent_output>{}</agent_output>"
    result = get_display_text(None, raw_text)
    assert "<agent_output>" not in result
    assert "Useful part." in result


# -- Format prompt injection ------------------------------------------

def test_output_format_prompts_all_types():
    from src.core.agent_output import AgentOutputType, OUTPUT_FORMAT_PROMPTS
    assert OUTPUT_FORMAT_PROMPTS[AgentOutputType.TEXT] == ""
    for output_type in [
        AgentOutputType.RESEARCH,
        AgentOutputType.CODER,
        AgentOutputType.SYNTHESIS,
    ]:
        prompt = OUTPUT_FORMAT_PROMPTS[output_type]
        assert "<agent_output>" in prompt
        assert "confidence" in prompt
        assert "success" in prompt


def test_format_prompt_contains_schema_fields():
    from src.core.agent_output import AgentOutputType, OUTPUT_FORMAT_PROMPTS
    research_prompt = OUTPUT_FORMAT_PROMPTS[AgentOutputType.RESEARCH]
    assert "facts" in research_prompt
    assert "gaps" in research_prompt
    assert "sources" in research_prompt

    coder_prompt = OUTPUT_FORMAT_PROMPTS[AgentOutputType.CODER]
    assert "actions_taken" in coder_prompt
    assert "errors" in coder_prompt

    synthesis_prompt = OUTPUT_FORMAT_PROMPTS[AgentOutputType.SYNTHESIS]
    assert "final_answer" in synthesis_prompt
    assert "caveats" in synthesis_prompt


# -- AgentDefinition output_type field -------------------------------

def test_agent_definition_default_output_type():
    from src.core.agent_definition import AgentDefinition
    ad = AgentDefinition(name="test", description="t", system_prompt="t")
    assert ad.output_type == "text"


def test_agent_registry_loads_output_type():
    from src.core.agent_registry import AgentRegistry
    registry = AgentRegistry()
    assert registry.get("research_agent").output_type == "research"
    assert registry.get("coder_agent").output_type == "coder"
    assert registry.get("synthesis_agent").output_type == "synthesis"


# -- State model integration ------------------------------------------

def test_state_model_structured_outputs_field():
    from src.core.state_model import AgentState, normalize_state
    state = AgentState.new("u1", "test")
    assert state.structured_outputs == {}
    d = state.to_dict()
    assert "structured_outputs" in d
    assert d["structured_outputs"] == {}


def test_state_model_structured_outputs_roundtrip():
    from src.core.state_model import AgentState
    d = {
        "user_id": "u1",
        "user_input": "test",
        "structured_outputs": {
            "research_agent": {
                "output_type": "research",
                "success": True,
                "confidence": 0.8,
                "facts": ["F1"],
                "extraction_failed": False,
            }
        }
    }
    state = AgentState.from_dict(d)
    assert state.structured_outputs["research_agent"]["confidence"] == pytest.approx(0.8)
    # Round-trip
    import json
    json.dumps(state.to_dict())  # must not raise


def test_state_model_structured_outputs_non_dict_dropped():
    from src.core.state_model import AgentState
    d = {
        "user_id": "u1",
        "user_input": "test",
        "structured_outputs": {
            "bad_agent": "this is a string not a dict",
            "good_agent": {"success": True},
        }
    }
    state = AgentState.from_dict(d)
    # Non-dict value replaced with None
    assert state.structured_outputs["bad_agent"] is None
    assert state.structured_outputs["good_agent"]["success"] is True


# -- End-to-end extraction pipeline ----------------------------------

def test_full_extraction_pipeline_research():
    """Simulate the full _run_agent extraction pipeline for research_agent."""
    from src.core.agent_output import (
        AgentOutputType, resolve_output_type, extract_structured_output,
        validate_agent_output, make_extraction_failed_result,
        strip_agent_output_block, get_display_text,
    )
    import json

    raw_response = """
I searched archival memory for energy-related context.

I found 2 relevant memories about the energy system.

<agent_output>
{"success": true, "confidence": 0.88, "summary": "Energy ROI uses value/effort",
 "facts": ["ROI threshold default is 1.25", "Effort multiplier is 3"],
 "sources": ["search_archival_memory"],
 "gaps": ["Could not find RLIMIT documentation"],
 "suggested_followup_tools": []}
</agent_output>
"""
    output_type = resolve_output_type("research")
    extracted = extract_structured_output(raw_response)
    assert extracted is not None

    validated = validate_agent_output(extracted, output_type, "research_agent", raw_response)
    assert validated is not None
    assert validated["extraction_failed"] is False
    assert len(validated["facts"]) == 2
    assert json.dumps(validated)  # JSON-safe

    clean_response = strip_agent_output_block(raw_response)
    assert "<agent_output>" not in clean_response

    display = get_display_text(validated, clean_response)
    assert "Energy ROI" in display


def test_full_extraction_pipeline_failure_fallback():
    """Extraction failure produces safe fallback result."""
    from src.core.agent_output import (
        AgentOutputType, extract_structured_output,
        validate_agent_output, make_extraction_failed_result,
        get_display_text,
    )
    import json

    raw_response = "I searched but could not produce structured output."
    extracted = extract_structured_output(raw_response)
    assert extracted is None

    fallback = make_extraction_failed_result(
        AgentOutputType.RESEARCH, "research_agent", raw_response
    )
    assert fallback["extraction_failed"] is True
    assert fallback["success"] is True
    assert json.dumps(fallback)  # JSON-safe

    display = get_display_text(fallback, raw_response)
    assert display  # should not be empty
