import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.agent_definition import AgentDefinition
from src.core.agent_registry import AgentRegistry
from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator
from src.core.state_model import AgentState
from src.core.prompt_config import build_supervisor_prompt


def test_agent_registry_loads_agent_markdown(tmp_path: Path):
    agent_file = tmp_path / "agents" / "planner_agent" / "AGENT.md"
    agent_file.parent.mkdir(parents=True)
    agent_file.write_text(
        """---
name: planner_agent
description: >
  Use this agent when the task needs planning
  before execution.
allowed_tools:
  - query_highest_priority_task
preferred_model: system_1
max_tool_calls: 3
energy_cost: 12
depends_on:
  - research_agent
---

You are the Planner Agent.
""",
        encoding="utf-8",
    )

    registry = AgentRegistry(agents_dir=tmp_path / "agents")
    planner = registry.get("planner_agent")

    assert planner is not None
    assert planner.description == "Use this agent when the task needs planning before execution."
    assert planner.allowed_tools == ["query_highest_priority_task"]
    assert planner.depends_on == ["research_agent"]
    assert planner.max_tool_calls == 3
    assert planner.energy_cost == 12


def test_agent_registry_falls_back_to_builtin_agents(tmp_path: Path):
    registry = AgentRegistry(agents_dir=tmp_path / "agents")

    assert registry.get("research_agent") is not None
    assert registry.get("coder_agent") is not None
    assert registry.get("synthesis_agent") is not None


def test_build_supervisor_prompt_includes_registered_agents():
    prompt = build_supervisor_prompt(
        charter_text="Charter",
        core_mem_str="Core memory",
        archival_block="Archival memory",
        capabilities_str="Capabilities",
        agent_descriptions="- research_agent: searches\n- coder_agent: implements",
        sensory_context="Sensory context",
        os_name="Windows",
        downloads_dir="downloads",
    )

    assert "<available_agents>" in prompt
    assert "research_agent: searches" in prompt
    assert '"agent": "agent_name"' in prompt
    assert '"depends_on": ["upstream_agent"]' in prompt


@pytest.mark.asyncio
async def test_execute_workers_node_uses_registry_for_dynamic_agent():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))
    custom_agent = AgentDefinition(
        name="planner_agent",
        description="Plans tasks",
        system_prompt="You are the Planner Agent.",
    )
    orchestrator.agent_registry.register(custom_agent)

    called = []

    async def fake_run_agent(agent_def, state, **_kwargs):
        await asyncio.sleep(0)
        called.append(agent_def.name)
        state["worker_outputs"][agent_def.name] = "ok"
        return state

    orchestrator._run_agent = fake_run_agent

    state = AgentState.new("user-1", "plan this").to_dict()
    state["current_plan"] = ["planner_agent"]

    result = await Orchestrator.execute_workers_node(orchestrator, state)

    assert called == ["planner_agent"]
    assert result["worker_outputs"]["planner_agent"] == "ok"


@pytest.mark.asyncio
async def test_execute_workers_node_runs_dependencies_before_requested_agent():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))

    called = []

    async def fake_run_agent(agent_def, state, **_kwargs):
        await asyncio.sleep(0)
        called.append(agent_def.name)
        state["worker_outputs"][agent_def.name] = f"done:{agent_def.name}"
        return state

    orchestrator._run_agent = fake_run_agent

    state = AgentState.new("user-1", "implement this").to_dict()
    state["current_plan"] = ["coder_agent"]

    result = await Orchestrator.execute_workers_node(orchestrator, state)

    assert called == ["research_agent", "coder_agent"]
    assert result["worker_outputs"]["research_agent"] == "done:research_agent"
    assert result["worker_outputs"]["coder_agent"] == "done:coder_agent"


def test_build_execution_plan_deduplicates_dependencies():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))

    plan, _ = Orchestrator._build_execution_batches(
        orchestrator,
        ["research_agent", "coder_agent", "research_agent"],
    )

    flattened = [agent.name for batch in plan for agent, _step in batch]
    assert flattened == ["research_agent", "coder_agent"]


def test_build_execution_plan_skips_unknown_dependency_and_continues():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))
    orchestrator.agent_registry.register(
        AgentDefinition(
            name="planner_agent",
            description="Plans tasks",
            system_prompt="You are the Planner Agent.",
            depends_on=["missing_agent"],
        )
    )

    plan, _ = Orchestrator._build_execution_batches(orchestrator, ["planner_agent"])

    flattened = [agent.name for batch in plan for agent, _step in batch]
    assert flattened == ["planner_agent"]


def test_parse_supervisor_response_normalizes_task_packets():
    orchestrator = Orchestrator.__new__(Orchestrator)
    state = AgentState.new("user-1", "investigate this").to_dict()

    result = Orchestrator._parse_supervisor_response(
        orchestrator,
        'Delegating now.\nWORKERS: [{"agent": "research_agent", "task": "Find the root cause", "reason": "Need evidence first"}, {"agent": "synthesis_agent", "task": "Combine the findings into one final answer", "reason": "User needs one response", "depends_on": ["research_agent"]}]',
        state,
    )

    assert result["current_plan"] == [
        {"agent": "research_agent", "task": "Find the root cause", "reason": "Need evidence first", "depends_on": [], "preferred_model": ""},
        {"agent": "synthesis_agent", "task": "Combine the findings into one final answer", "reason": "User needs one response", "depends_on": ["research_agent"], "preferred_model": ""},
    ]
    assert result["worker_outputs"]["supervisor_context"] == "Delegating now."


def test_build_execution_plan_respects_task_packet_dependencies():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))
    orchestrator.agent_registry.register(
        AgentDefinition(
            name="planner_agent",
            description="Plans tasks",
            system_prompt="You are the Planner Agent.",
        )
    )

    plan, _ = Orchestrator._build_execution_batches(
        orchestrator,
        [
            {"agent": "research_agent", "task": "Gather facts", "reason": "Need evidence"},
            {"agent": "planner_agent", "task": "Outline the work", "reason": "Need a plan"},
            {
                "agent": "synthesis_agent",
                "task": "Merge the research and plan into one final reply",
                "reason": "The user needs one polished answer",
                "depends_on": ["research_agent", "planner_agent"],
            },
        ],
    )

    flattened = [[agent.name for agent, _step in batch] for batch in plan]
    assert flattened == [["research_agent", "planner_agent"], ["synthesis_agent"]]


def test_get_output_to_evaluate_prefers_final_synthesis_output():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))
    orchestrator.agent_registry.register(
        AgentDefinition(
            name="planner_agent",
            description="Plans tasks",
            system_prompt="You are the Planner Agent.",
        )
    )

    state = AgentState.new("user-1", "research and plan this").to_dict()
    state["current_plan"] = [
        {"agent": "research_agent", "task": "Gather facts", "reason": "Need evidence"},
        {"agent": "planner_agent", "task": "Outline the work", "reason": "Need a plan"},
        {
            "agent": "synthesis_agent",
            "task": "Merge the research and plan into one final reply",
            "reason": "The user needs one polished answer",
            "depends_on": ["research_agent", "planner_agent"],
        },
    ]
    state["worker_outputs"] = {
        "research_agent": "Research findings.",
        "planner_agent": "Implementation plan.",
        "synthesis_agent": "Final combined answer.",
    }

    assert Orchestrator._get_output_to_evaluate(orchestrator, state) == "Final combined answer."


@pytest.mark.asyncio
async def test_execute_workers_node_parallelizes_independent_agents():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))
    orchestrator.agent_registry.register(
        AgentDefinition(
            name="planner_agent",
            description="Plans tasks",
            system_prompt="You are the Planner Agent.",
        )
    )

    active = 0
    peak_active = 0
    both_started = asyncio.Event()

    async def fake_run_agent(agent_def, state, **_kwargs):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        if active >= 2:
            both_started.set()
        try:
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            state["worker_outputs"][agent_def.name] = f"done:{agent_def.name}"
            return state
        finally:
            active -= 1

    orchestrator._run_agent = fake_run_agent

    state = AgentState.new("user-1", "research and plan this").to_dict()
    state["current_plan"] = [
        {"agent": "research_agent", "task": "Gather background", "reason": "Need facts"},
        {"agent": "planner_agent", "task": "Outline the implementation", "reason": "Need a plan"},
    ]

    result = await Orchestrator.execute_workers_node(orchestrator, state)

    assert peak_active == 2
    assert result["worker_outputs"]["research_agent"] == "done:research_agent"
    assert result["worker_outputs"]["planner_agent"] == "done:planner_agent"


@pytest.mark.asyncio
async def test_route_agent_request_prefers_system_2_for_agent_preference():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.get_system_2_available.return_value = True
    orchestrator.cognitive_router.route_to_system_2 = AsyncMock(
        return_value=RouterResult(status="ok", content="Combined answer")
    )
    orchestrator._route_to_system_1 = AsyncMock(
        return_value=RouterResult(status="ok", content="Local answer")
    )

    agent_def = AgentDefinition(
        name="synthesis_agent",
        description="Combines outputs",
        system_prompt="You are the Synthesis Agent.",
        preferred_model="system_2",
    )

    router_result, attempted_system_2 = await Orchestrator._route_agent_request(
        orchestrator,
        [{"role": "user", "content": "Combine the prior outputs"}],
        agent_def,
    )

    assert attempted_system_2 is True
    assert router_result is not None
    assert router_result.content == "Combined answer"
    orchestrator.cognitive_router.route_to_system_2.assert_awaited_once()
    orchestrator._route_to_system_1.assert_not_called()


def test_build_agent_handoff_uses_declared_dependencies():
    orchestrator = Orchestrator.__new__(Orchestrator)
    agent_def = AgentDefinition(
        name="coder_agent",
        description="Implements",
        system_prompt="You are the Coder Agent.",
        depends_on=["research_agent"],
    )
    state = AgentState.new("user-1", "fix the bug").to_dict()
    state["worker_outputs"] = {
        "supervisor_context": "Focus on the failing test first.",
        "research_agent": "Found the failing import path.",
    }

    handoff = Orchestrator._build_agent_handoff(
        orchestrator,
        agent_def,
        state,
        task_packet={"agent": "coder_agent", "task": "Patch the broken import and rerun the test", "reason": "Research already isolated the defect"},
    )

    assert "Focus on the failing test first." in handoff
    assert "research_agent: Found the failing import path." in handoff
    assert "fix the bug" in handoff
    assert "Patch the broken import and rerun the test" in handoff
    assert "Research already isolated the defect" in handoff