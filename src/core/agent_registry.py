from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)


_FALLBACK_AGENT_DEFINITIONS = (
    AgentDefinition(
        name="research_agent",
        description=(
            "Use this agent when the task requires searching memory, gathering "
            "background information, or retrieving facts before acting."
        ),
        system_prompt=(
            "You are the Research Agent. Search archival memory for relevant "
            "context and provide findings. If uncertainty is high or the task "
            "needs deeper reasoning, call escalate_to_system_2 with a concise "
            "problem description and scratchpad."
        ),
        allowed_tools=["search_archival_memory", "escalate_to_system_2"],
        preferred_model="system_1",
        max_tool_calls=4,
        energy_cost=15,
    ),
    AgentDefinition(
        name="coder_agent",
        description=(
            "Use this agent when the task requires implementation, file changes, "
            "memory updates, or structured execution after research is complete."
        ),
        system_prompt=(
            "You are the Coder Agent. Execute coding tasks and update memory as needed. "
            "Escalate early via escalate_to_system_2 when local reasoning is "
            "insufficient for correctness."
        ),
        allowed_tools=[
            "update_ledger",
            "update_core_memory",
            "request_core_update",
            "spawn_new_objective",
            "update_objective_status",
            "extract_pdf_text",
            "search_archival_memory",
            "escalate_to_system_2",
        ],
        preferred_model="system_1",
        max_tool_calls=5,
        energy_cost=15,
        depends_on=["research_agent"],
    ),
    AgentDefinition(
        name="synthesis_agent",
        description=(
            "Use this agent when multiple agent outputs need to be combined into "
            "one final, user-facing response without adding new facts."
        ),
        system_prompt=(
            "You are the Synthesis Agent. Combine prior agent outputs into one clear "
            "final response for the user. Reconcile overlaps, preserve uncertainties, "
            "and do not invent facts that were not present in the supplied inputs."
        ),
        allowed_tools=[],
        preferred_model="system_2",
        max_tool_calls=0,
        energy_cost=10,
    ),
)


class AgentRegistry:
    def __init__(self, agents_dir: Optional[Path | str] = None) -> None:
        self.agents_dir = Path(agents_dir) if agents_dir is not None else Path(__file__).resolve().parent.parent / "agents"
        self._agents: Dict[str, AgentDefinition] = {}
        self.reload()

    def reload(self) -> None:
        self._agents = {}
        self._load_from_disk()
        self._register_missing_fallbacks()

    def register(self, agent: AgentDefinition) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[AgentDefinition]:
        return self._agents.get(name)

    def all(self) -> List[AgentDefinition]:
        return sorted(self._agents.values(), key=lambda agent: agent.name)

    def _load_from_disk(self) -> None:
        if not self.agents_dir.exists():
            return

        for agent_file in sorted(self.agents_dir.glob("*/AGENT.md")):
            try:
                self.register(self._parse_agent_file(agent_file))
            except ValueError as exc:
                logger.warning("Skipping invalid agent definition %s: %s", agent_file, exc)

    def _register_missing_fallbacks(self) -> None:
        for agent in _FALLBACK_AGENT_DEFINITIONS:
            self._agents.setdefault(agent.name, agent)

    def _parse_agent_file(self, agent_file: Path) -> AgentDefinition:
        raw_text = agent_file.read_text(encoding="utf-8")
        front_matter, body = self._split_front_matter(raw_text)
        if not body.strip():
            raise ValueError("agent prompt body is empty")

        name = str(front_matter.get("name") or agent_file.parent.name)
        description = str(front_matter.get("description") or "").strip()
        if not description:
            raise ValueError("description is required")

        allowed_tools = self._coerce_list(front_matter.get("allowed_tools"))
        depends_on = self._coerce_list(front_matter.get("depends_on"))

        return AgentDefinition(
            name=name,
            description=description,
            system_prompt=body.strip(),
            allowed_tools=allowed_tools,
            preferred_model=str(front_matter.get("preferred_model") or "system_1"),
            max_tool_calls=int(front_matter.get("max_tool_calls") or 5),
            energy_cost=int(front_matter.get("energy_cost") or 15),
            depends_on=depends_on,
        )

    @staticmethod
    def _split_front_matter(text: str) -> tuple[Dict[str, Any], str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text.strip()

        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                front_matter = "\n".join(lines[1:index])
                body = "\n".join(lines[index + 1:])
                return AgentRegistry._parse_front_matter(front_matter), body.strip()

        return {}, text.strip()

    @staticmethod
    def _parse_front_matter(front_matter: str) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        lines = front_matter.splitlines()
        index = 0

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            key, value = AgentRegistry._parse_front_matter_line(line)
            if value in {">", "|"}:
                block_lines, index = AgentRegistry._consume_indented_block(lines, index + 1)
                parsed[key] = AgentRegistry._format_block_value(block_lines, value)
                continue
            if value == "":
                parsed[key], index = AgentRegistry._consume_list_items(lines, index + 1)
                continue
            parsed[key] = AgentRegistry._coerce_scalar(value)
            index += 1

        return parsed

    @staticmethod
    def _parse_front_matter_line(line: str) -> tuple[str, str]:
        if ":" not in line:
            raise ValueError(f"invalid front matter line: {line!r}")
        key, raw_value = line.split(":", 1)
        return key.strip(), raw_value.strip()

    @staticmethod
    def _consume_indented_block(lines: List[str], index: int) -> tuple[List[str], int]:
        block_lines: List[str] = []
        while index < len(lines):
            block_line = lines[index]
            if block_line.startswith("  ") or block_line.startswith("\t"):
                block_lines.append(block_line.strip())
                index += 1
                continue
            if not block_line.strip():
                block_lines.append("")
                index += 1
                continue
            break
        return block_lines, index

    @staticmethod
    def _format_block_value(block_lines: List[str], operator: str) -> str:
        if operator == ">":
            return " ".join(part for part in block_lines if part).strip()
        return "\n".join(block_lines).strip()

    @staticmethod
    def _consume_list_items(lines: List[str], index: int) -> tuple[List[str], int]:
        list_items: List[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if not stripped.startswith("- "):
                break
            list_items.append(stripped[2:].strip())
            index += 1
        return list_items, index

    @staticmethod
    def _coerce_scalar(value: str) -> Any:
        stripped = value.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            stripped = stripped[1:-1]
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        return stripped

    @staticmethod
    def _coerce_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []