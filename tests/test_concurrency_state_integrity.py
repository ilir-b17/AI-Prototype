import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.agent_definition import AgentDefinition
from src.core.agent_registry import AgentRegistry
from src.core.llm_router import RequiresHITLError, RouterResult
from src.core.orchestrator import Orchestrator, _BLOCKED_KEY
from src.core.state_model import AgentState


def _build_lock_test_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._ready = asyncio.Event()
    orchestrator._ready.set()
    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}
    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = 100
    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_resume_tool_approval = AsyncMock(return_value=None)
    orchestrator._remember_user_profile = AsyncMock(return_value=False)
    orchestrator._remember_assistant_identity = AsyncMock(return_value=False)
    orchestrator._try_goal_planning_response = AsyncMock(return_value=None)
    orchestrator._try_fast_path_response = AsyncMock(return_value=None)
    orchestrator._load_state = AsyncMock(
        side_effect=lambda user_id, user_message, **_kwargs: AgentState.new(user_id, user_message).to_dict()
    )
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)
    return orchestrator


def _build_parallel_test_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.agent_registry = AgentRegistry(agents_dir=Path("does-not-matter"))
    orchestrator.agent_registry.register(
        AgentDefinition(
            name="alpha_agent",
            description="Independent alpha worker",
            system_prompt="Alpha",
            energy_cost=15,
        )
    )
    orchestrator.agent_registry.register(
        AgentDefinition(
            name="beta_agent",
            description="Independent beta worker",
            system_prompt="Beta",
            energy_cost=15,
        )
    )
    return orchestrator


@pytest.mark.asyncio
async def test_user_lock_eviction_never_allows_same_user_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_LOCKS_MAX_SIZE", "5")
    orchestrator = _build_lock_test_orchestrator()

    active_by_user: DefaultDict[str, int] = defaultdict(int)
    violations: List[str] = []
    guard = asyncio.Lock()

    async def run_graph_loop(_state: Dict[str, object], user_id: str, user_message: str) -> str:
        async with guard:
            active_by_user[user_id] += 1
            if active_by_user[user_id] > 1:
                violations.append(user_id)
        try:
            await asyncio.sleep(0.01)
            return f"ok:{user_id}:{user_message}"
        finally:
            async with guard:
                active_by_user[user_id] -= 1

    orchestrator._run_graph_loop = run_graph_loop

    tasks = [
        asyncio.create_task(
            Orchestrator.process_message(orchestrator, f"message-{index}", f"user-{index % 20}")
        )
        for index in range(50)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 50
    assert violations == []


@pytest.mark.asyncio
async def test_parallel_batch_reraises_hitl_before_merging_successful_outputs() -> None:
    orchestrator = _build_parallel_test_orchestrator()

    async def run_agent(agent_def: AgentDefinition, state: Dict[str, object], **_kwargs: object) -> Dict[str, object]:
        await asyncio.sleep(0)
        if agent_def.name == "alpha_agent":
            raise RequiresHITLError("Need admin guidance")
        state["worker_outputs"] = {agent_def.name: "valid sibling output"}
        return state

    orchestrator._run_agent = run_agent
    state = AgentState.new("user-1", "run two agents").to_dict()
    state["current_plan"] = [
        {"agent": "alpha_agent", "task": "May need guidance", "reason": "Test HITL propagation"},
        {"agent": "beta_agent", "task": "Return output", "reason": "Successful sibling"},
    ]

    with pytest.raises(RequiresHITLError, match="Need admin guidance"):
        await Orchestrator.execute_workers_node(orchestrator, state)

    assert state["worker_outputs"] == {}


@pytest.mark.asyncio
async def test_hitl_resume_preserves_critic_rejection_count_and_abandons() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._ready = asyncio.Event()
    orchestrator._ready.set()
    orchestrator.pending_mfa = {}
    orchestrator.pending_tool_approval = {}
    orchestrator.pending_hitl_state = {
        "user-1": {
            **AgentState.new("user-1", "unsafe task").to_dict(),
            "iteration_count": 3,
            "critic_feedback": "FAIL: third rejection",
            "_hitl_created_at": time.time(),
        }
    }
    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = 100
    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_resume_tool_approval = AsyncMock(return_value=None)
    orchestrator._run_graph_loop = AsyncMock(return_value="should not run")
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.clear_hitl_state = AsyncMock()
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)

    response = await Orchestrator.process_message(orchestrator, "admin guidance", "user-1")

    assert "abandoned" in response.lower()
    orchestrator._run_graph_loop.assert_not_awaited()
    orchestrator.ledger_memory.clear_hitl_state.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_parallel_blocked_batch_refunds_unused_sibling_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "100")
    orchestrator = _build_parallel_test_orchestrator()
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = 70

    async def run_agent(agent_def: AgentDefinition, state: Dict[str, object], **_kwargs: object) -> Dict[str, object]:
        await asyncio.sleep(0)
        state = dict(state)
        state["worker_outputs"] = dict(state.get("worker_outputs", {}) or {})
        if agent_def.name == "alpha_agent":
            state[_BLOCKED_KEY] = RouterResult(status="hitl_required", hitl_message="pause")
            return state
        state["worker_outputs"][agent_def.name] = "unused sibling output"
        return state

    orchestrator._run_agent = run_agent
    state = AgentState.new("user-1", "run two agents").to_dict()
    state["_energy_gate_cleared"] = True
    state["current_plan"] = [
        {"agent": "alpha_agent", "task": "Block", "reason": "Triggers HITL"},
        {"agent": "beta_agent", "task": "Succeed", "reason": "Unused sibling"},
    ]

    result = await Orchestrator.execute_workers_node(orchestrator, state)

    assert result[_BLOCKED_KEY].status == "hitl_required"
    assert "beta_agent" not in result["worker_outputs"]
    assert result["energy_remaining"] == 85
    assert orchestrator._predictive_energy_budget_remaining == 85
