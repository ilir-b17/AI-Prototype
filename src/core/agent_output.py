"""
Formal agent communication protocol — typed output contracts.

Each agent declares an output_type in its AGENT.md. After producing
its response the agent appends a structured JSON block inside
<agent_output>...</agent_output> XML tags. This module handles:

  - Output type definitions (ResearchResult, CoderResult, SynthesisResult)
  - Extraction from raw LLM response text
  - Validation and normalisation of extracted dicts
  - Schema injection into agent system prompts
  - Formatting structured results for downstream agent handoffs

Design principles:
  - Extraction never raises — returns None on any failure
  - Validation never raises — returns a safe fallback AgentResult
  - All stored results are plain dicts (JSON-serialisable), not dataclasses
  - The raw text in worker_outputs is never modified
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Output type registry ──────────────────────────────────────────────

class AgentOutputType(str, Enum):
    TEXT = "text"           # no structured output (default)
    RESEARCH = "research"   # research_agent
    CODER = "coder"         # coder_agent
    SYNTHESIS = "synthesis" # synthesis_agent

_VALID_OUTPUT_TYPES = {t.value for t in AgentOutputType}


def resolve_output_type(raw: Any) -> AgentOutputType:
    """Resolve a raw string to AgentOutputType, defaulting to TEXT."""
    candidate = str(raw or "").strip().lower()
    if candidate in _VALID_OUTPUT_TYPES:
        return AgentOutputType(candidate)
    return AgentOutputType.TEXT


# ── Result dataclasses ────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Base structured result returned by every agent."""
    agent_name: str
    output_type: str
    success: bool
    confidence: float              # 0.0–1.0
    summary: str                   # human-readable one-paragraph summary
    extraction_failed: bool = False
    raw_text_preview: str = ""     # first 300 chars of raw response

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict safe for JSON serialisation."""
        d = asdict(self)
        return d


@dataclass
class ResearchResult(AgentResult):
    """Structured output from research_agent."""
    facts: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    suggested_followup_tools: List[str] = field(default_factory=list)


@dataclass
class CoderResult(AgentResult):
    """Structured output from coder_agent."""
    actions_taken: List[str] = field(default_factory=list)
    memory_updates: List[str] = field(default_factory=list)
    objectives_created: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)


@dataclass
class SynthesisResult(AgentResult):
    """Structured output from synthesis_agent."""
    final_answer: str = ""
    sources_cited: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    conflicts_noted: List[str] = field(default_factory=list)


# ── JSON schemas for prompt injection ────────────────────────────────

_RESEARCH_SCHEMA = """{
  "success": true,
  "confidence": 0.85,
  "summary": "One paragraph summarising findings",
  "facts": ["Fact 1 found in memory or tool output", "Fact 2"],
  "sources": ["search_archival_memory", "web_search"],
  "gaps": ["Could not find X", "Unclear about Y"],
  "suggested_followup_tools": ["web_search"]
}"""

_CODER_SCHEMA = """{
  "success": true,
  "confidence": 0.90,
  "summary": "One paragraph summarising what was done",
  "actions_taken": ["Updated core memory key X", "Added objective #5"],
  "memory_updates": ["Stored fact A to core memory"],
  "objectives_created": ["Task #5: Write yfinance wrapper"],
  "errors": [],
  "files_modified": []
}"""

_SYNTHESIS_SCHEMA = """{
  "success": true,
  "confidence": 0.80,
  "summary": "One paragraph overview",
  "final_answer": "The complete, polished answer for the user",
  "sources_cited": ["research_agent", "coder_agent"],
  "caveats": ["This answer is based on cached data from 2 days ago"],
  "conflicts_noted": []
}"""

OUTPUT_SCHEMAS: Dict[str, str] = {
    AgentOutputType.RESEARCH: _RESEARCH_SCHEMA,
    AgentOutputType.CODER: _CODER_SCHEMA,
    AgentOutputType.SYNTHESIS: _SYNTHESIS_SCHEMA,
}

# ── Format instruction blocks ─────────────────────────────────────────

def _build_format_prompt(output_type: AgentOutputType) -> str:
    """Build the output format instruction to append to agent system prompts."""
    if output_type == AgentOutputType.TEXT:
        return ""

    schema = OUTPUT_SCHEMAS.get(output_type, "")
    if not schema:
        return ""

    return (
        "\n\nOUTPUT FORMAT REQUIREMENT:\n"
        "After your complete response, you MUST append a structured summary\n"
        "inside <agent_output> tags containing ONLY valid JSON. No markdown\n"
        "inside the tags. No explanation. Strictly follow this schema:\n"
        "<agent_output>\n"
        f"{schema}\n"
        "</agent_output>\n\n"
        "Rules:\n"
        "- confidence is a float 0.0–1.0 reflecting how complete your answer is.\n"
        "- success is true if you fulfilled the task, false if blocked/failed.\n"
        "- Every string list must be a JSON array, even if empty: [].\n"
        "- Do not include any text after the closing </agent_output> tag.\n"
        "- If you cannot produce the structured block, write exactly:\n"
        "  <agent_output>{\"success\": false, \"confidence\": 0.0, "
        "\"summary\": \"structured output unavailable\"}</agent_output>"
    )


OUTPUT_FORMAT_PROMPTS: Dict[str, str] = {
    t: _build_format_prompt(t)
    for t in AgentOutputType
}

# ── Extraction ────────────────────────────────────────────────────────

_AGENT_OUTPUT_RE = re.compile(
    r"<agent_output>\s*(.*?)\s*</agent_output>",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_structured_output(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the <agent_output> JSON block from raw text.

    Returns the parsed dict or None if not found or unparseable.
    Never raises.
    """
    if not text:
        return None
    match = _AGENT_OUTPUT_RE.search(text)
    if not match:
        return None
    json_text = match.group(1).strip()
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        # Try stripping markdown fences that might appear inside the tags
        clean = re.sub(r"^```(?:json)?\s*", "", json_text, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean).strip()
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            logger.debug(
                "agent_output extraction: JSON parse failed. "
                "Raw: %r", json_text[:200]
            )
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def strip_agent_output_block(text: str) -> str:
    """Remove the <agent_output> block from text for clean display."""
    return _AGENT_OUTPUT_RE.sub("", text).rstrip()


# ── Validation and normalisation ──────────────────────────────────────

def _coerce_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        f = float(value)
        return max(lo, min(hi, f))
    except (TypeError, ValueError):
        return default


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return default


def validate_agent_output(
    raw: Optional[Dict[str, Any]],
    output_type: AgentOutputType,
    agent_name: str,
    raw_text: str = "",
) -> Optional[Dict[str, Any]]:
    """Validate and normalise an extracted output dict.

    Returns a normalised plain dict (JSON-safe) or None if raw is None.
    Never raises. On validation error returns a safe fallback dict.
    """
    raw_preview = str(raw_text or "")[:300]

    if raw is None:
        return None

    base: Dict[str, Any] = {
        "agent_name": str(agent_name),
        "output_type": output_type.value,
        "success": _coerce_bool(raw.get("success"), True),
        "confidence": _coerce_float(raw.get("confidence"), 0.5, 0.0, 1.0),
        "summary": str(raw.get("summary") or "").strip()[:500],
        "extraction_failed": False,
        "raw_text_preview": raw_preview,
    }

    if output_type == AgentOutputType.RESEARCH:
        base.update({
            "facts": _coerce_str_list(raw.get("facts")),
            "sources": _coerce_str_list(raw.get("sources")),
            "gaps": _coerce_str_list(raw.get("gaps")),
            "suggested_followup_tools": _coerce_str_list(
                raw.get("suggested_followup_tools")
            ),
        })
    elif output_type == AgentOutputType.CODER:
        base.update({
            "actions_taken": _coerce_str_list(raw.get("actions_taken")),
            "memory_updates": _coerce_str_list(raw.get("memory_updates")),
            "objectives_created": _coerce_str_list(raw.get("objectives_created")),
            "errors": _coerce_str_list(raw.get("errors")),
            "files_modified": _coerce_str_list(raw.get("files_modified")),
        })
    elif output_type == AgentOutputType.SYNTHESIS:
        base.update({
            "final_answer": str(raw.get("final_answer") or "").strip(),
            "sources_cited": _coerce_str_list(raw.get("sources_cited")),
            "caveats": _coerce_str_list(raw.get("caveats")),
            "conflicts_noted": _coerce_str_list(raw.get("conflicts_noted")),
        })

    return base


def make_extraction_failed_result(
    output_type: AgentOutputType,
    agent_name: str,
    raw_text: str = "",
) -> Dict[str, Any]:
    """Create a fallback structured result when extraction fails."""
    raw_preview = str(raw_text or "")[:300]
    base: Dict[str, Any] = {
        "agent_name": str(agent_name),
        "output_type": output_type.value,
        "success": True,         # assume success — just unstructured
        "confidence": 0.5,
        "summary": raw_preview or "Agent produced unstructured output.",
        "extraction_failed": True,
        "raw_text_preview": raw_preview,
    }
    if output_type == AgentOutputType.RESEARCH:
        base.update({
            "facts": [],
            "sources": [],
            "gaps": ["Structured output not available — see raw text"],
            "suggested_followup_tools": [],
        })
    elif output_type == AgentOutputType.CODER:
        base.update({
            "actions_taken": [],
            "memory_updates": [],
            "objectives_created": [],
            "errors": [],
            "files_modified": [],
        })
    elif output_type == AgentOutputType.SYNTHESIS:
        base.update({
            "final_answer": raw_preview,
            "sources_cited": [],
            "caveats": ["Structured synthesis not available — using raw text"],
            "conflicts_noted": [],
        })
    return base


# ── Handoff formatting ────────────────────────────────────────────────

def _append_limited_items(
    lines: List[str],
    items: Any,
    *,
    header: str,
    limit: int,
    prefix: str,
) -> None:
    values = _coerce_str_list(items)[:limit]
    if not values:
        return
    lines.append(header)
    for value in values:
        lines.append(f"    {prefix} {value}")


def _append_research_handoff(lines: List[str], result: Dict[str, Any]) -> None:
    _append_limited_items(
        lines,
        result.get("facts"),
        header="  facts:",
        limit=10,
        prefix="•",
    )
    _append_limited_items(
        lines,
        result.get("gaps"),
        header="  gaps (not found):",
        limit=5,
        prefix="•",
    )

    sources = _coerce_str_list(result.get("sources"))
    if sources:
        lines.append(f"  sources used: {', '.join(sources)}")

    tools = _coerce_str_list(result.get("suggested_followup_tools"))
    if tools:
        lines.append(f"  suggested tools: {', '.join(tools)}")


def _append_coder_handoff(lines: List[str], result: Dict[str, Any]) -> None:
    _append_limited_items(
        lines,
        result.get("actions_taken"),
        header="  actions taken:",
        limit=8,
        prefix="✓",
    )
    _append_limited_items(
        lines,
        result.get("errors"),
        header="  errors encountered:",
        limit=5,
        prefix="✗",
    )
    _append_limited_items(
        lines,
        result.get("memory_updates"),
        header="  memory updates:",
        limit=5,
        prefix="→",
    )
    _append_limited_items(
        lines,
        result.get("objectives_created"),
        header="  objectives created:",
        limit=5,
        prefix="+",
    )


def _append_synthesis_handoff(lines: List[str], result: Dict[str, Any]) -> None:
    answer = str(result.get("final_answer", "")).strip()
    if answer:
        lines.append(f"  final_answer: {answer[:400]}")
    _append_limited_items(
        lines,
        result.get("caveats"),
        header="  caveats:",
        limit=3,
        prefix="!",
    )

def format_structured_for_handoff(
    result: Dict[str, Any],
    dependency_name: str,
) -> str:
    """Format a structured result for injection into a dependent agent's handoff.

    Produces a compact, labelled text block that is clearer than the raw
    text blob the agent previously received.
    """
    if result is None:
        return ""

    output_type = str(result.get("output_type", "text"))
    success = bool(result.get("success", True))
    confidence = float(result.get("confidence", 0.5))
    summary = str(result.get("summary", "")).strip()
    extraction_failed = bool(result.get("extraction_failed", False))

    lines = [
        f"[{dependency_name} structured output]",
        f"  success={success}  confidence={confidence:.0%}",
    ]

    if extraction_failed:
        lines.append(
            "  NOTE: structured extraction failed — using raw text summary"
        )

    if summary:
        lines.append(f"  summary: {summary}")

    handlers = {
        AgentOutputType.RESEARCH.value: _append_research_handoff,
        AgentOutputType.CODER.value: _append_coder_handoff,
        AgentOutputType.SYNTHESIS.value: _append_synthesis_handoff,
    }
    handler = handlers.get(output_type)
    if handler is not None:
        handler(lines, result)

    return "\n".join(lines)


def get_display_text(
    structured: Optional[Dict[str, Any]],
    raw_text: str,
) -> str:
    """Extract the best display text from a structured result or raw text.

    Used by _get_output_to_evaluate and the critic node.
    Priority:
      1. synthesis.final_answer (if non-empty)
      2. summary (if extraction succeeded and non-empty)
      3. raw_text with <agent_output> block stripped
    """
    if structured is None:
        return strip_agent_output_block(raw_text)

    output_type = str(structured.get("output_type", "text"))
    extraction_failed = bool(structured.get("extraction_failed", False))

    if output_type == AgentOutputType.SYNTHESIS:
        final_answer = str(structured.get("final_answer", "")).strip()
        if final_answer:
            return final_answer

    if not extraction_failed:
        summary = str(structured.get("summary", "")).strip()
        if summary:
            return summary

    return strip_agent_output_block(raw_text)
