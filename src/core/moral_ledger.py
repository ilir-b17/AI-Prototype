from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


MORAL_RUBRIC_VERSION = "moral_ledger_v1"
MORAL_SCORE_MIN = 1
MORAL_SCORE_MAX = 5
MORAL_TIER_1_DIMENSIONS: Tuple[str, ...] = (
    "harm_reduction",
    "data_privacy",
    "admin_authority_security",
    "data_sovereignty_local_priority",
)
MORAL_TIER_2_DIMENSIONS: Tuple[str, ...] = (
    "epistemic_humility",
    "transparency_logging",
    "alignment_with_user_intent",
)
MORAL_TIER_3_DIMENSIONS: Tuple[str, ...] = (
    "output_cleanliness",
)
MORAL_DIMENSIONS: Tuple[str, ...] = (
    *MORAL_TIER_1_DIMENSIONS,
    *MORAL_TIER_2_DIMENSIONS,
    *MORAL_TIER_3_DIMENSIONS,
)
VALID_MORAL_TIERS: Tuple[str, ...] = ("tier_1", "tier_2", "tier_3")


@dataclass(frozen=True)
class MoralDecision:
    rubric_version: str
    scores: Dict[str, int]
    reasoning: str
    is_approved: bool
    decision_mode: str = "system2_audit"
    bypass_reason: str = ""
    remediation_constraints: Tuple[str, ...] = ()
    violated_tiers: Tuple[str, ...] = ()
    security_conflict: bool = False
    validation_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rubric_version": str(self.rubric_version),
            "scores": dict(self.scores),
            "reasoning": str(self.reasoning),
            "is_approved": bool(self.is_approved),
            "decision_mode": str(self.decision_mode),
            "bypass_reason": str(self.bypass_reason),
            "remediation_constraints": list(self.remediation_constraints),
            "violated_tiers": list(self.violated_tiers),
            "security_conflict": bool(self.security_conflict),
            "validation_error": str(self.validation_error),
        }


def _extract_json_text(raw_response: str) -> str:
    text = str(raw_response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _default_scores(score: int) -> Dict[str, int]:
    bounded = max(MORAL_SCORE_MIN, min(MORAL_SCORE_MAX, int(score)))
    return dict.fromkeys(MORAL_DIMENSIONS, bounded)


def _validate_required_payload_fields(payload: Dict[str, Any]) -> str:
    required_fields = {"rubric_version", "scores", "reasoning", "is_approved"}
    present_fields = set(payload.keys())
    missing_fields = sorted(required_fields - present_fields)
    if missing_fields:
        return "missing_required_fields:" + ",".join(missing_fields)
    return ""


def _validate_scores_dimensions(scores_raw: Dict[str, Any]) -> str:
    expected_dimensions = set(MORAL_DIMENSIONS)
    provided_dimensions = set(scores_raw.keys())
    missing_dimensions = sorted(expected_dimensions - provided_dimensions)
    extra_dimensions = sorted(provided_dimensions - expected_dimensions)
    if not missing_dimensions and not extra_dimensions:
        return ""

    details = []
    if missing_dimensions:
        details.append("missing_dimensions=" + ",".join(missing_dimensions))
    if extra_dimensions:
        details.append("extra_dimensions=" + ",".join(extra_dimensions))
    return "invalid_score_dimensions:" + "|".join(details)


def _normalize_scores(scores_raw: Dict[str, Any]) -> tuple[Dict[str, int], str]:
    normalized_scores: Dict[str, int] = {}
    try:
        for dimension in MORAL_DIMENSIONS:
            normalized_scores[dimension] = _parse_score(scores_raw.get(dimension))
    except Exception as exc:
        return {}, f"invalid_score_value:{exc}"
    return normalized_scores, ""


def _normalize_string_list(value: Any, field_name: str) -> tuple[Tuple[str, ...], str]:
    if value is None:
        return (), ""

    if isinstance(value, str):
        candidate_values = [value]
    elif isinstance(value, list):
        candidate_values = value
    else:
        return (), f"invalid_{field_name}:expected_list_or_string"

    normalized: List[str] = []
    for raw_item in candidate_values:
        item = str(raw_item or "").strip()
        if not item:
            continue
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized), ""


def _normalize_violated_tiers(value: Any) -> tuple[Tuple[str, ...], str]:
    tiers, error = _normalize_string_list(value, "violated_tiers")
    if error:
        return (), error

    normalized: List[str] = []
    for raw_tier in tiers:
        tier = str(raw_tier).lower().replace("-", "_").replace(" ", "")
        if tier == "tier1":
            tier = "tier_1"
        elif tier == "tier2":
            tier = "tier_2"
        elif tier == "tier3":
            tier = "tier_3"
        if tier not in VALID_MORAL_TIERS:
            return (), f"invalid_violated_tier:{raw_tier}"
        if tier not in normalized:
            normalized.append(tier)
    return tuple(normalized), ""


def build_safe_rejection_decision(reason: str, *, decision_mode: str = "validation_failure") -> MoralDecision:
    return MoralDecision(
        rubric_version=MORAL_RUBRIC_VERSION,
        scores=_default_scores(MORAL_SCORE_MIN),
        reasoning=str(reason or "Moral decision rejected due to invalid payload.").strip(),
        is_approved=False,
        decision_mode=str(decision_mode or "validation_failure"),
        bypass_reason="",
        remediation_constraints=(),
        violated_tiers=(),
        security_conflict=False,
        validation_error=str(reason or "invalid_payload").strip(),
    )


def build_triviality_bypass_decision(reason: str) -> MoralDecision:
    explanation = str(reason or "trivial_read_only").strip()
    return MoralDecision(
        rubric_version=MORAL_RUBRIC_VERSION,
        scores=_default_scores(MORAL_SCORE_MAX),
        reasoning="Moral audit bypassed by triviality gate for benign read-only action.",
        is_approved=True,
        decision_mode="triviality_bypass",
        bypass_reason=explanation,
        remediation_constraints=(),
        violated_tiers=(),
        security_conflict=False,
        validation_error="",
    )


def build_local_skip_decision(reason: str) -> MoralDecision:
    explanation = str(reason or "local_skip").strip()
    return MoralDecision(
        rubric_version=MORAL_RUBRIC_VERSION,
        scores=_default_scores(4),
        reasoning="Moral audit was locally skipped before System 2 evaluation.",
        is_approved=True,
        decision_mode="local_skip",
        bypass_reason=explanation,
        remediation_constraints=(),
        violated_tiers=(),
        security_conflict=False,
        validation_error="",
    )


def build_legacy_binary_decision(*, is_approved: bool, reasoning: str) -> MoralDecision:
    approved = bool(is_approved)
    return MoralDecision(
        rubric_version=MORAL_RUBRIC_VERSION,
        scores=_default_scores(4 if approved else MORAL_SCORE_MIN),
        reasoning=str(reasoning or "Legacy critic decision.").strip(),
        is_approved=approved,
        decision_mode="legacy_binary_critic",
        bypass_reason="",
        remediation_constraints=(),
        violated_tiers=(),
        security_conflict=False,
        validation_error="",
    )


def _parse_score(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("scores must be integers in range 1-5")
    parsed = int(value)
    if parsed < MORAL_SCORE_MIN or parsed > MORAL_SCORE_MAX:
        raise ValueError("scores must be integers in range 1-5")
    return parsed


def _parse_optional_moral_metadata(
    payload: Dict[str, Any],
) -> tuple[Tuple[str, ...], Tuple[str, ...], bool, str]:
    remediation_constraints, remediation_error = _normalize_string_list(
        payload.get("remediation_constraints"),
        "remediation_constraints",
    )
    if remediation_error:
        return (), (), False, remediation_error

    violated_tiers, violated_tiers_error = _normalize_violated_tiers(
        payload.get("violated_tiers"),
    )
    if violated_tiers_error:
        return (), (), False, violated_tiers_error

    security_conflict_raw = payload.get("security_conflict", False)
    if not isinstance(security_conflict_raw, bool):
        return (), (), False, "invalid_security_conflict"

    return remediation_constraints, violated_tiers, bool(security_conflict_raw), ""


def validate_moral_decision_payload(payload: Any) -> MoralDecision:
    if not isinstance(payload, dict):
        return build_safe_rejection_decision("invalid_payload: expected object")

    required_error = _validate_required_payload_fields(payload)
    if required_error:
        return build_safe_rejection_decision(required_error)

    rubric_version = str(payload.get("rubric_version") or "").strip()
    if not rubric_version:
        return build_safe_rejection_decision("invalid_rubric_version")

    scores_raw = payload.get("scores")
    if not isinstance(scores_raw, dict):
        return build_safe_rejection_decision("invalid_scores_object")

    dimensions_error = _validate_scores_dimensions(scores_raw)
    if dimensions_error:
        return build_safe_rejection_decision(dimensions_error)

    normalized_scores, score_error = _normalize_scores(scores_raw)
    if score_error:
        return build_safe_rejection_decision(score_error)

    reasoning = str(payload.get("reasoning") or "").strip()
    if not reasoning:
        return build_safe_rejection_decision("invalid_reasoning")

    is_approved_raw = payload.get("is_approved")
    if not isinstance(is_approved_raw, bool):
        return build_safe_rejection_decision("invalid_is_approved")

    decision_mode = str(payload.get("decision_mode") or "system2_audit").strip() or "system2_audit"
    bypass_reason = str(payload.get("bypass_reason") or "").strip()
    remediation_constraints, violated_tiers, security_conflict, metadata_error = _parse_optional_moral_metadata(payload)
    if metadata_error:
        return build_safe_rejection_decision(metadata_error)

    return MoralDecision(
        rubric_version=rubric_version,
        scores=normalized_scores,
        reasoning=reasoning,
        is_approved=bool(is_approved_raw),
        decision_mode=decision_mode,
        bypass_reason=bypass_reason,
        remediation_constraints=remediation_constraints,
        violated_tiers=violated_tiers,
        security_conflict=security_conflict,
        validation_error="",
    )


def parse_moral_decision_response(response_content: Any) -> MoralDecision:
    if isinstance(response_content, dict):
        return validate_moral_decision_payload(response_content)

    response_text = _extract_json_text(str(response_content or ""))
    if not response_text:
        return build_safe_rejection_decision("invalid_json:empty_response")

    try:
        payload = json.loads(response_text)
    except Exception:
        return build_safe_rejection_decision("invalid_json:decode_failure")

    return validate_moral_decision_payload(payload)
