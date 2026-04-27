import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.runtime_context import set_runtime_context
from src.skills.spawn_new_objective import spawn_new_objective


@pytest.mark.asyncio
async def test_spawn_new_objective_passes_agent_domain_to_ledger() -> None:
    ledger = MagicMock()
    ledger.add_objective = AsyncMock(return_value=42)
    set_runtime_context(ledger=ledger, core=None, vector=None, orchestrator=None)
    try:
        raw = await spawn_new_objective(
            tier="Task",
            title="Process inbox request",
            estimated_energy=12,
            agent_domain="google",
        )
        payload = json.loads(raw)
        assert payload["status"] == "success"
        assert payload["data"]["id"] == 42
        assert payload["data"]["agent_domain"] == "google"
        assert ledger.add_objective.await_args.kwargs["agent_domain"] == "google"
    finally:
        set_runtime_context(ledger=None, core=None, vector=None, orchestrator=None)
