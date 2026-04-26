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


def _render_session_context_block(
    active_session: Optional[Dict[str, Any]],
    epic_rollup: Optional[Dict[str, Any]],
) -> str:
    """Render the <active_session> block for the supervisor prompt.

    Returns empty string when no session is active.
    The block is injected into the volatile turn context (not the stable
    system prompt) so KV cache is not invalidated on every turn.
    """
    if not active_session or not active_session.get("id"):
        return ""

    lines = [
        "<active_session>",
        f"  Session: {active_session.get('name', 'Unnamed')} "
        f"(id={active_session['id']})",
    ]

    description = str(active_session.get("description") or "").strip()
    if description:
        lines.append(f"  Context: {description}")

    turn_count = int(active_session.get("turn_count") or 0)
    memory_count = int(active_session.get("memory_count") or 0)
    lines.append(
        f"  Turns in session: {turn_count} | "
        f"Memories tagged: {memory_count}"
    )

    if epic_rollup:
        completed = int(epic_rollup.get("completed_tasks") or 0)
        total = int(epic_rollup.get("total_tasks") or 0)
        active_t = int(epic_rollup.get("active_tasks") or 0)
        pending_t = int(epic_rollup.get("pending_tasks") or 0)
        lines.append(
            f"  Linked Epic: {epic_rollup.get('title', 'Unknown')} "
            f"[{epic_rollup.get('status', '')}] "
            f"— {completed}/{total} tasks complete, "
            f"{active_t} active, {pending_t} pending"
        )

    lines.append("</active_session>")
    return "\n".join(lines)


_SUPERVISOR_TEMPLATE = textwrap.dedent(
    """\
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
    - WEB HANDLING: After performing a `web_search`, if the returned snippets are insufficient to fully answer the user's question — especially for factual data such as sports scores, live weather details, news articles, financial results, or any query where a snippet is clearly truncated — you MUST call `extract_web_article` on the top result URL to retrieve the full content before composing your answer. Do not answer from incomplete snippets when full content is readily accessible.
    - DATA HANDLING: For CSV/Excel files, do not read raw text. Use the `analyze_table_file` tool.
    - SHELL COMMANDS: `run_terminal_command` executes without a shell and only for allowlisted first-token commands. Do not propose shell chaining, redirects, or command substitution.
    - PYTHON SANDBOX: `execute_python_sandbox` runs inside the dynamic worker sandbox with blocked-module checks and worker-confined /tmp access.
    - STOCK PRICES: To fetch prices for multiple tickers, call `get_stock_price` once per ticker in sequence. Never invent batch variants like `get_stock_prices`.
    - OBJECTIVES: When the Admin asks to add, create, define, or track a goal/objective/task, use `spawn_new_objective`. Use `query_highest_priority_task` to inspect current backlog work. Do not call `request_capability` when these existing tools already fit.
    - SELF-KNOWLEDGE: When asked why you chose an agent, about energy budget, deferred/blocked tasks, recent errors, or synthesis history, use query_decision_log, query_objective_status, query_energy_state, or query_system_health.
    - COGNITIVE SYNERGY: If you face a complex reasoning problem or get stuck, use `escalate_to_system_2`. You may also search Archival Memory for past "System 2 blueprints" (e.g. past problem solutions).
        - STRUCTURED PROTOCOL: Workers append <agent_output> JSON; handoff injects dependencies automatically, so do not re-summarise in WORKERS.
    - MEMORY CONSOLIDATION: `consolidate_memory` is a **tool** you call directly — it is NOT an agent name. Never place `consolidate_memory` in a WORKERS declaration.
    </tool_and_data_rules>

    <agent_charter>
    {charter_text}
    </agent_charter>

    <context_and_memory>
    <Core_Working_Memory>
    {core_mem_str}
    </Core_Working_Memory>
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
    - If you need to delegate to specialized agents, output JSON task packets using only agent names listed in <available_agents>. NEVER place tool names (e.g. `consolidate_memory`, `web_search`, `escalate_to_system_2`) as `agent` values — those are tools, not agents.
    - Format delegated work as: WORKERS: [{{"agent": "agent_name", "task": "short concrete task", "reason": "why this agent is needed", "depends_on": ["upstream_agent"]}}]
        - AGENT COMMUNICATION: Handoff auto-passes <agent_output> dependencies; if synthesis_agent is planned, it merges upstream outputs into one final answer.
    - Each task must be specific to that agent. Do not output bare agent names.
    - WORKERS payload must be strict JSON with no trailing commentary after the closing bracket.
    - If multiple independent agents are needed but the user expects one polished final answer, add a final synthesis_agent task whose depends_on list names the upstream agents it must combine.
    </output_formatting>
    """
)


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
    return _SUPERVISOR_TEMPLATE.format(
        os_name=os_name,
        downloads_dir=downloads_dir,
        charter_text=charter_text,
        core_mem_str=core_mem_str,
        recent_rejections_block=recent_rejections_block,
        capabilities_str=capabilities_str,
        agent_descriptions=agent_descriptions,
    )


def build_supervisor_turn_context(
    *,
    sensory_context: str,
    archival_block: str,
    active_session: Optional[Dict[str, Any]] = None,
    epic_rollup: Optional[Dict[str, Any]] = None,
) -> str:
    sections = []
    if str(sensory_context or "").strip():
        sections.append(
            f"--- Sensory Context ---\n{str(sensory_context).strip()}"
        )
    if str(archival_block or "").strip():
        sections.append(
            f"--- Archival Memory ---\n{str(archival_block).strip()}"
        )
    session_block = _render_session_context_block(active_session, epic_rollup)
    if session_block:
        sections.append(session_block)

    if not sections:
        return ""
    return "<turn_context>\n" + "\n\n".join(sections) + "\n</turn_context>"
