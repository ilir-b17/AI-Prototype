"""
Ground-truth test cases for the moral audit system.

Covers:
  - validate_moral_decision_payload: valid and invalid payloads
  - parse_moral_decision_response: JSON extraction from raw LLM output
  - Triviality bypass detection: which outputs bypass the full audit
  - Severity classification: moderate vs severe violations
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.core.moral_ledger import MORAL_DIMENSIONS, MORAL_RUBRIC_VERSION


def _valid_scores(override: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Return valid scores (all 4) with optional overrides."""
    scores = dict.fromkeys(MORAL_DIMENSIONS, 4)
    if override:
        scores.update(override)
    return scores


def _as_markdown_json(payload: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def _mcase(
    id: str,
    description: str,
    payload: Any,
    expect_valid: bool,
    expect_approved: Optional[bool] = None,
    expect_decision_mode: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": id,
        "description": description,
        "payload": payload,
        "expect_valid": expect_valid,
        "expect_approved": expect_approved,
        "expect_decision_mode": expect_decision_mode,
        "tags": tags or [],
    }


# Valid, approved payloads
VALID_APPROVED = [
    _mcase(
        "moral_001",
        "all scores 4, approved",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores(),
            "reasoning": "Output is safe and aligned.",
            "is_approved": True,
            "decision_mode": "system2_audit",
            "bypass_reason": "",
            "remediation_constraints": [],
            "violated_tiers": [],
            "security_conflict": False,
        },
        expect_valid=True,
        expect_approved=True,
        expect_decision_mode="system2_audit",
        tags=["valid", "approved"],
    ),
    _mcase(
        "moral_002",
        "scores in markdown code block",
        _as_markdown_json(
            {
                "rubric_version": MORAL_RUBRIC_VERSION,
                "scores": _valid_scores(),
                "reasoning": "Safe",
                "is_approved": True,
                "decision_mode": "system2_audit",
                "bypass_reason": "",
                "remediation_constraints": [],
                "violated_tiers": [],
                "security_conflict": False,
            }
        ),
        expect_valid=True,
        expect_approved=True,
        expect_decision_mode="system2_audit",
        tags=["valid", "approved", "markdown"],
    ),
]

# Valid, rejected payloads
VALID_REJECTED = [
    _mcase(
        "moral_003",
        "tier 2 violation, remediation constraints",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores({"epistemic_humility": 2}),
            "reasoning": "Overconfident claim without evidence.",
            "is_approved": False,
            "decision_mode": "system2_audit",
            "bypass_reason": "",
            "remediation_constraints": ["Add uncertainty qualifier"],
            "violated_tiers": ["tier_2"],
            "security_conflict": False,
        },
        expect_valid=True,
        expect_approved=False,
        expect_decision_mode="system2_audit",
        tags=["valid", "rejected", "tier2"],
    ),
    _mcase(
        "moral_004",
        "tier 1 security conflict",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores({"admin_authority_security": 1}),
            "reasoning": "Output bypasses admin authority.",
            "is_approved": False,
            "decision_mode": "system2_audit",
            "bypass_reason": "",
            "remediation_constraints": [],
            "violated_tiers": ["tier_1"],
            "security_conflict": True,
        },
        expect_valid=True,
        expect_approved=False,
        expect_decision_mode="system2_audit",
        tags=["valid", "rejected", "tier1", "security"],
    ),
]

# Invalid payloads - should produce rejection decisions
INVALID_PAYLOADS = [
    _mcase(
        "moral_005",
        "missing rubric_version",
        {"scores": _valid_scores(), "reasoning": "ok", "is_approved": True},
        expect_valid=False,
        tags=["invalid", "missing_field"],
    ),
    _mcase(
        "moral_006",
        "missing scores",
        {"rubric_version": MORAL_RUBRIC_VERSION, "reasoning": "ok", "is_approved": True},
        expect_valid=False,
        tags=["invalid", "missing_field"],
    ),
    _mcase(
        "moral_007",
        "score out of range (6)",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores({"harm_reduction": 6}),
            "reasoning": "ok",
            "is_approved": True,
        },
        expect_valid=False,
        tags=["invalid", "bad_score"],
    ),
    _mcase(
        "moral_008",
        "score = 0 (below minimum)",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores({"harm_reduction": 0}),
            "reasoning": "ok",
            "is_approved": True,
        },
        expect_valid=False,
        tags=["invalid", "bad_score"],
    ),
    _mcase(
        "moral_009",
        "is_approved is a string not bool",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores(),
            "reasoning": "ok",
            "is_approved": "true",
        },
        expect_valid=False,
        tags=["invalid", "type_error"],
    ),
    _mcase(
        "moral_010",
        "entirely non-JSON string",
        "This is not JSON at all",
        expect_valid=False,
        tags=["invalid", "not_json"],
    ),
    _mcase(
        "moral_011",
        "empty dict",
        {},
        expect_valid=False,
        tags=["invalid", "empty"],
    ),
    _mcase(
        "moral_012",
        "missing reasoning",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores(),
            "reasoning": "",
            "is_approved": True,
        },
        expect_valid=False,
        tags=["invalid", "missing_field"],
    ),
    _mcase(
        "moral_013",
        "invalid violated_tier value",
        {
            "rubric_version": MORAL_RUBRIC_VERSION,
            "scores": _valid_scores(),
            "reasoning": "ok",
            "is_approved": False,
            "violated_tiers": ["tier_99"],
        },
        expect_valid=False,
        tags=["invalid", "bad_tier"],
    ),
]

ALL_MORAL_CASES = VALID_APPROVED + VALID_REJECTED + INVALID_PAYLOADS


# Triviality bypass cases
# (state_snapshot, output_to_eval, expected_bypass: True/False)
TRIVIALITY_CASES = [
    {
        "id": "triv_001",
        "description": "System info single_tool - should bypass",
        "user_input": "What time is it?",
        "output": '{"datetime": "2026-04-26 10:00"}',
        "current_plan": [{"agent": "get_system_info"}],
        "iteration_count": 0,
        "critic_instructions": "",
        "expected_bypass": True,
        "tags": ["bypass", "single_tool"],
    },
    {
        "id": "triv_002",
        "description": "Multi-agent plan - must NOT bypass",
        "user_input": "Research and summarize AI trends",
        "output": "AI is evolving rapidly...",
        "current_plan": [
            {"agent": "research_agent"},
            {"agent": "coder_agent"},
        ],
        "iteration_count": 0,
        "critic_instructions": "",
        "expected_bypass": False,
        "tags": ["no_bypass", "multi_agent"],
    },
    {
        "id": "triv_003",
        "description": "Critic instruction present - must NOT bypass",
        "user_input": "What time is it?",
        "output": "It is 10am",
        "current_plan": [],
        "iteration_count": 1,
        "critic_instructions": "[CRITIC: fix output]",
        "expected_bypass": False,
        "tags": ["no_bypass", "critic_active"],
    },
    {
        "id": "triv_004",
        "description": "Output contains modify - must NOT bypass",
        "user_input": "Update my core memory",
        "output": "Updated memory key X",
        "current_plan": [{"agent": "update_core_memory"}],
        "iteration_count": 0,
        "critic_instructions": "",
        "expected_bypass": False,
        "tags": ["no_bypass", "mutating"],
    },
    {
        "id": "triv_005",
        "description": "Weather keyword - should bypass",
        "user_input": "Current weather in Vienna",
        "output": '{"temperature": "22 C"}',
        "current_plan": [{"agent": "weather_current"}],
        "iteration_count": 0,
        "critic_instructions": "",
        "expected_bypass": True,
        "tags": ["bypass", "weather"],
    },
    {
        "id": "triv_006",
        "description": "Empty output - must NOT bypass",
        "user_input": "Hello",
        "output": "",
        "current_plan": [],
        "iteration_count": 0,
        "critic_instructions": "",
        "expected_bypass": False,
        "tags": ["no_bypass", "empty_output"],
    },
]
