from __future__ import annotations

import os
from dataclasses import dataclass
import textwrap
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PromptConfig:
    downloads_dir: str


def load_prompt_config() -> PromptConfig:
    downloads_dir = os.getenv("AIDEN_DOWNLOADS_DIR", "downloads")
    return PromptConfig(downloads_dir=downloads_dir)


def _normalize_string_list(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        values = raw_value
    elif raw_value is None:
        values = []
    else:
        values = [raw_value]
    return [str(item).strip() for item in values if str(item).strip()]


def _render_single_rejection_line(item: Any, index: int) -> str:
    if not isinstance(item, dict):
        return ""

    tiers = ", ".join(_normalize_string_list(item.get("violated_tiers"))) or "unspecified"
    constraints = "; ".join(_normalize_string_list(item.get("remediation_constraints")))
    if not constraints:
        constraints = "follow charter-safe fallback"

    reasoning = str(item.get("reasoning") or "").strip()
    line = f"{index}) tiers={tiers}; constraints={constraints}"
    if reasoning:
        line += f"; reason={reasoning}"
    return line


def _render_recent_rejections_block(
    recent_rejections: Optional[List[Dict[str, Any]]],
    *,
    max_chars: int = 500,
) -> str:
    if not recent_rejections:
        return ""

    lines = [
        _render_single_rejection_line(item, idx)
        for idx, item in enumerate(list(recent_rejections)[:3], start=1)
    ]
    lines = [line for line in lines if line]

    body = "\n".join(lines).strip()
    if len(body) > max_chars:
        body = body[: max_chars - 3].rstrip() + "..."
    if not body:
        return ""
    return f"\n<recent_rejections>\n{body}\n</recent_rejections>"


def build_supervisor_prompt(
    *,
    charter_text: str,
    core_mem_str: str,
    archival_block: str,
    capabilities_str: str,
    agent_descriptions: str,
    sensory_context: str,
    os_name: str,
    downloads_dir: str,
    recent_rejections: Optional[List[Dict[str, Any]]] = None,
) -> str:
    _ = (archival_block, sensory_context)
    recent_rejections_block = _render_recent_rejections_block(recent_rejections)
    # textwrap.dedent removes the leading whitespace, keeping the code clean 
    # while ensuring the LLM gets perfectly flush text.
    return textwrap.dedent(f"""\
        <system_identity>
        You are AIDEN, a local autonomous AI agent running on the Admin's machine ({os_name} via Ollama).
        You are a dedicated local assistant, NOT a cloud-based AI like ChatGPT or Claude.
        You have persistent memory using SQLite, meaning the chat history provided is real and continuous.
        </system_identity>

        <core_directives>
        1. HONESTY OVER CAPABILITY: Never fabricate, simulate, or guess tool results. If a tool fails or returns an error, explicitly state the error to the user and ask for guidance. 
        2. TOOL USAGE: You have access to specific tools. Use them to fulfill requests, but do not invent capabilities outside of what is explicitly provided.
        3. OUTPUT STYLE: Keep your responses conversational and plain. Avoid using markdown headers (like # or ##).
        4. PRIVATE REASONING: You may think before acting. Wrap any internal reasoning, planning, or self-correction inside <think>...</think> tags.
        </core_directives>

        <tool_and_data_rules>
        - FILE SYSTEM: Your primary downloads directory is `{downloads_dir}`. Prefer using the `manage_file_system` tool for OS-agnostic exploration.
        - PDF HANDLING: To read or summarize a PDF, you MUST first execute the `extract_pdf_text` tool.
        - WEB HANDLING: After performing a `web_search`, you MUST call `extract_web_article` on the target URL before attempting to summarize it.
        - DATA HANDLING: For CSV/Excel files, do not read raw text. Use the `analyze_table_file` tool.
        - SHELL COMMANDS: Use strictly {os_name}-appropriate shell commands when interacting with the terminal.
        - STOCK PRICES: To fetch prices for multiple tickers, call `get_stock_price` once per ticker in sequence. Never invent batch variants like `get_stock_prices`.
        - OBJECTIVES: When the Admin asks to add, create, define, or track a goal/objective/task, use `spawn_new_objective`. Use `query_highest_priority_task` to inspect current backlog work. Do not call `request_capability` when these existing tools already fit.
        - COGNITIVE SYNERGY: If you face a complex reasoning problem or get stuck, use `escalate_to_system_2`. You may also search Archival Memory for past "System 2 blueprints" (e.g. past problem solutions).
        </tool_and_data_rules>

        <agent_charter>
        {charter_text}
        </agent_charter>

        <context_and_memory>
        --- Core Memory ---
        {core_mem_str}
        {recent_rejections_block}
        </context_and_memory>

        <available_capabilities>
        {capabilities_str}
        </available_capabilities>

        <available_agents>
        {agent_descriptions}
        </available_agents>

        <output_formatting>
        CRITICAL: Every response MUST end with a strict worker declaration on the very last line, outside of any <think> blocks.
        - If you can handle the request directly in chat: WORKERS: []
        - If you need to delegate to specialized agents, output JSON task packets using only agent names listed in <available_agents>.
        - Format delegated work as: WORKERS: [{{"agent": "agent_name", "task": "short concrete task", "reason": "why this agent is needed", "depends_on": ["upstream_agent"]}}]
        - Each task must be specific to that agent. Do not output bare agent names.
        - WORKERS payload must be strict JSON with no trailing commentary after the closing bracket.
        - If multiple independent agents are needed but the user expects one polished final answer, add a final synthesis_agent task whose depends_on list names the upstream agents it must combine.
        </output_formatting>
    """)


def build_supervisor_turn_context(
    *,
    sensory_context: str,
    archival_block: str,
) -> str:
    sections = []
    if str(sensory_context or "").strip():
        sections.append(f"--- Sensory Context ---\n{str(sensory_context).strip()}")
    if str(archival_block or "").strip():
        sections.append(f"--- Archival Memory ---\n{str(archival_block).strip()}")
    if not sections:
        return ""
    return "<turn_context>\n" + "\n\n".join(sections) + "\n</turn_context>"
