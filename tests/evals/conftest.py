"""
Shared pytest fixtures for eval suites.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark eval as slow (requires Ollama, skipped in fast mode)"
    )
    config.addinivalue_line(
        "markers",
        "tier1: deterministic eval, no IO"
    )
    config.addinivalue_line(
        "markers",
        "tier2: heuristic eval, SkillRegistry only"
    )
    config.addinivalue_line(
        "markers",
        "tier3: LLM-graded eval, requires Ollama"
    )


@pytest.fixture(scope="session")
def skill_registry():
    """Real SkillRegistry loaded from src/skills/. Session-scoped for speed."""
    from src.core.skill_manager import SkillRegistry
    return SkillRegistry()


@pytest.fixture(scope="session")
def routing_orchestrator(skill_registry):
    """Minimal Orchestrator with only routing methods available.

    Does NOT connect to SQLite, ChromaDB, or Ollama.
    Uses a stub CognitiveRouter that exposes only the registry.
    """
    import types
    from src.core.orchestrator import Orchestrator

    # Build a stub that satisfies _assess_request_route's dependencies
    # without touching any database or network resource.
    stub_router = types.SimpleNamespace(
        registry=skill_registry,
    )

    orch = object.__new__(Orchestrator)
    orch.cognitive_router = stub_router
    # Populate constants needed by routing methods
    orch._intent_classification_cache = {}
    return orch
