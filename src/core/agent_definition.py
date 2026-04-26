from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    system_prompt: str
    allowed_tools: List[str] = field(default_factory=list)
    preferred_model: str = "system_1"
    max_tool_calls: int = 5
    energy_cost: int = 15
    depends_on: List[str] = field(default_factory=list)
    output_type: str = "text"   # "text" | "research" | "coder" | "synthesis"