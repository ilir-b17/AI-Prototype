import pytest
import inspect

from src.core.state_model import AgentState, normalize_state
from src.core.workflow_graph import LANGGRAPH_AVAILABLE, build_orchestrator_graph


class _StubOrchestrator:
    async def supervisor_node(self, state):
        state = dict(state)
        state["current_plan"] = []
        state["final_response"] = state.get("final_response") or "ok"
        return state

    async def execute_workers_node(self, state):
        return state

    async def critic_node(self, state):
        state = dict(state)
        state["critic_feedback"] = "PASS"
        return state


def test_agent_state_defaults_include_critic_instructions():
    state = AgentState.new("u", "hello").to_dict()
    assert state["critic_instructions"] == ""


def test_normalize_state_preserves_extra_runtime_keys():
    raw = {"user_id": "u", "user_input": "x", "_blocked_result": "sentinel"}
    normalized = normalize_state(raw)
    assert normalized["_blocked_result"] == "sentinel"
    assert normalized["energy_remaining"] == 100


@pytest.mark.asyncio
async def test_langgraph_workflow_builder_compiles_and_runs_when_available():
    graph = build_orchestrator_graph(_StubOrchestrator())
    if not LANGGRAPH_AVAILABLE:
        assert graph is None
        return

    assert graph is not None
    result = graph.ainvoke({"user_id": "u", "user_input": "hello", "current_plan": []})
    if not inspect.isawaitable(result):
        pytest.skip("LangGraph runtime is mocked in this test environment.")
    out = await result
    assert out["critic_feedback"] == "PASS"
