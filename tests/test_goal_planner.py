import json
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.goal_planner import GoalPlanner
from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator


class _LedgerStub:
    def __init__(self, start_id: int = 100):
        self._next_id = int(start_id)
        self.add_calls = []

    async def add_objective(self, **kwargs):
        await asyncio.sleep(0)
        self.add_calls.append(dict(kwargs))
        current_id = self._next_id
        self._next_id += 1
        return current_id


@pytest.mark.asyncio
async def test_plan_goal_parses_hierarchy_and_resolves_dependency_refs():
    planner = GoalPlanner()
    ledger = _LedgerStub(start_id=200)
    captured = {}

    llm_response = {
        "epic": {
            "title": "Launch market intelligence pipeline",
            "acceptance_criteria": "Stories and tasks exist with clear dependencies.",
            "priority": 3,
            "estimated_energy": 70,
        },
        "stories": [
            {
                "story_id": "story_data",
                "title": "Prepare ingest + normalization",
                "acceptance_criteria": "All source feeds are normalized.",
                "priority": 4,
                "estimated_energy": 30,
                "tasks": [
                    {
                        "task_id": "task_a",
                        "title": "Ingest source feeds",
                        "acceptance_criteria": "Feeds are persisted in staging.",
                        "priority": 4,
                        "estimated_energy": 10,
                        "depends_on_ids": [],
                    },
                    {
                        "task_id": "task_b",
                        "title": "Normalize schema",
                        "acceptance_criteria": "Schema is normalized and validated.",
                        "priority": 5,
                        "estimated_energy": 12,
                        "depends_on_ids": ["task_a"],
                    },
                ],
            }
        ],
    }

    async def fake_route_to_system_2(messages):
        await asyncio.sleep(0)
        captured["messages"] = messages
        return json.dumps(llm_response)

    result = await planner.plan_goal(
        "Build a production-ready market intelligence pipeline.",
        context="The user needs a structured decomposition with dependencies.",
        route_to_system_2=fake_route_to_system_2,
        ledger_memory=ledger,
        origin="User:test-user",
    )

    assert result.epic_id == 200
    assert result.story_ids == [201]
    assert result.task_ids == [202, 203]

    assert len(ledger.add_calls) == 4
    assert ledger.add_calls[0]["tier"] == "Epic"
    assert ledger.add_calls[1]["tier"] == "Story"
    assert ledger.add_calls[2]["tier"] == "Task"
    assert ledger.add_calls[3]["tier"] == "Task"

    assert ledger.add_calls[2]["depends_on_ids"] == []
    assert ledger.add_calls[3]["depends_on_ids"] == [202]

    payload = json.loads(captured["messages"][1]["content"])
    assert payload["task"] == "decompose_goal_hierarchy"
    assert payload["objective"].startswith("Build a production-ready")


@pytest.mark.asyncio
async def test_plan_goal_uses_phase2_redaction_boundary_for_planning_payload():
    planner = GoalPlanner()
    ledger = _LedgerStub(start_id=300)
    captured = {}

    async def fake_route_to_system_2(messages):
        await asyncio.sleep(0)
        captured["messages"] = messages
        return {
            "epic": {"title": "Redacted epic"},
            "stories": [
                {
                    "story_id": "story_1",
                    "title": "Story",
                    "tasks": [
                        {"task_id": "task_1", "title": "Task", "depends_on_ids": []}
                    ],
                }
            ],
        }

    await planner.plan_goal(
        "Epic: Handle my account ilir@example.com and C:\\Users\\iboci\\secret.txt",
        context=(
            "<Core_Working_Memory>name=Ilir Boci</Core_Working_Memory> "
            "Please include private profile details."
        ),
        route_to_system_2=fake_route_to_system_2,
        ledger_memory=ledger,
        redactor=Orchestrator._redact_text_for_cloud,
        origin="User:test-user",
    )

    payload = json.loads(captured["messages"][1]["content"])
    serialized = json.dumps(payload)

    assert "ilir@example.com" not in serialized
    assert "C:\\Users\\iboci\\secret.txt" not in serialized
    assert "Ilir Boci" not in serialized
    assert "REDACTED" in serialized


@pytest.mark.asyncio
async def test_plan_goal_raises_when_dependency_cannot_be_resolved():
    planner = GoalPlanner()
    ledger = _LedgerStub(start_id=400)

    async def fake_route_to_system_2(_messages):
        await asyncio.sleep(0)
        return {
            "epic": {"title": "Broken dependency epic"},
            "stories": [
                {
                    "story_id": "story_1",
                    "title": "Story",
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "title": "Dependent task",
                            "depends_on_ids": ["task_missing"],
                        }
                    ],
                }
            ],
        }

    with pytest.raises(ValueError, match="could not resolve task dependencies"):
        await planner.plan_goal(
            "Create a dependency graph",
            context="",
            route_to_system_2=fake_route_to_system_2,
            ledger_memory=ledger,
            origin="User:test-user",
        )


@pytest.mark.asyncio
async def test_orchestrator_invokes_goal_planner_for_high_level_request():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.goal_planner = GoalPlanner()
    orchestrator.ledger_memory = _LedgerStub(start_id=500)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.get_system_2_available.return_value = True
    orchestrator._route_to_system_2_redacted = AsyncMock(
        return_value=RouterResult(
            status="ok",
            content={
                "epic": {"title": "Launch autonomous planning"},
                "stories": [
                    {
                        "story_id": "story_1",
                        "title": "Design planning graph",
                        "tasks": [
                            {"task_id": "task_a", "title": "Model decomposition schema", "depends_on_ids": []},
                            {"task_id": "task_b", "title": "Wire planner entrypoint", "depends_on_ids": ["task_a"]},
                        ],
                    }
                ],
            },
        )
    )

    state = {
        "user_id": "user-1",
        "user_input": "Epic: Build a multi-step roadmap and architecture plan for an autonomous trading workflow.",
        "chat_history": [
            {"role": "user", "content": "I need a phased roadmap."},
            {"role": "assistant", "content": "Understood."},
        ],
    }

    reply = await Orchestrator._try_goal_planning_response(orchestrator, state)

    assert reply is not None
    assert "Planning only complete" in reply
    assert len(orchestrator.ledger_memory.add_calls) == 4
    orchestrator._route_to_system_2_redacted.assert_awaited_once()
