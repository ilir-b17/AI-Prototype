from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentState:
    """Formal runtime state model passed between orchestration nodes."""

    user_id: str
    user_input: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    current_plan: List[Any] = field(default_factory=list)
    worker_outputs: Dict[str, str] = field(default_factory=dict)
    final_response: str = ""
    iteration_count: int = 0
    admin_guidance: str = ""
    energy_remaining: int = 100
    hitl_count: int = 0
    critic_feedback: str = ""
    critic_instructions: str = ""
    moral_decision: Dict[str, Any] = field(default_factory=dict)
    moral_audit_mode: str = ""
    moral_audit_trace: str = ""
    moral_audit_bypassed: bool = False
    moral_remediation_constraints: List[str] = field(default_factory=list)
    moral_halt_required: bool = False
    moral_halt_summary: str = ""

    @classmethod
    def new(cls, user_id: str, user_input: str) -> "AgentState":
        return cls(user_id=user_id, user_input=user_input)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AgentState":
        return cls(
            user_id=str(raw.get("user_id", "")),
            user_input=str(raw.get("user_input", "")),
            chat_history=list(raw.get("chat_history", []) or []),
            current_plan=list(raw.get("current_plan", []) or []),
            worker_outputs=dict(raw.get("worker_outputs", {}) or {}),
            final_response=str(raw.get("final_response", "") or ""),
            iteration_count=int(raw.get("iteration_count", 0) or 0),
            admin_guidance=str(raw.get("admin_guidance", "") or ""),
            energy_remaining=int(raw.get("energy_remaining", 100) or 100),
            hitl_count=int(raw.get("hitl_count", 0) or 0),
            critic_feedback=str(raw.get("critic_feedback", "") or ""),
            critic_instructions=str(raw.get("critic_instructions", "") or ""),
            moral_decision=dict(raw.get("moral_decision", {}) or {}),
            moral_audit_mode=str(raw.get("moral_audit_mode", "") or ""),
            moral_audit_trace=str(raw.get("moral_audit_trace", "") or ""),
            moral_audit_bypassed=bool(raw.get("moral_audit_bypassed", False)),
            moral_remediation_constraints=[
                str(item).strip()
                for item in (raw.get("moral_remediation_constraints", []) or [])
                if str(item).strip()
            ],
            moral_halt_required=bool(raw.get("moral_halt_required", False)),
            moral_halt_summary=str(raw.get("moral_halt_summary", "") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize arbitrary dict input to a complete AgentState-backed dict."""
    normalized = AgentState.from_dict(raw).to_dict()
    for key, value in raw.items():
        if key not in normalized:
            normalized[key] = value
    return normalized
