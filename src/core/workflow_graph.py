from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    END = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


def build_orchestrator_graph(orchestrator) -> Optional[Any]:
    """Build a LangGraph workflow for one supervisor->workers->critic pass."""
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(dict)
    graph.add_node("supervisor", orchestrator.supervisor_node)
    graph.add_node("execute_workers", orchestrator.execute_workers_node)
    graph.add_node("critic", orchestrator.critic_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "execute_workers")
    graph.add_edge("execute_workers", "critic")
    graph.add_edge("critic", END)

    return graph.compile()
