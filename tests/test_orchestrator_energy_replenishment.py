import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.energy_roi_engine import ENERGY_MIN_RESERVE
from src.core.orchestrator import Orchestrator


class _CursorWithNoDeferredTasks:
    async def fetchone(self):
        return {"earliest": None}


class _DbWithNoDeferredTasks:
    async def execute(self, _query: str):
        return _CursorWithNoDeferredTasks()


def _build_heartbeat_only_orchestrator(initial_budget: int) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = int(initial_budget)
    orchestrator.ledger_memory = SimpleNamespace(_db=_DbWithNoDeferredTasks())
    return orchestrator


@pytest.mark.asyncio
async def test_heartbeat_replenishment_recovers_predictive_budget_without_user_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "100")
    monkeypatch.setenv("ENERGY_REPLENISH_PER_HEARTBEAT", "2")

    reserve_floor = int(ENERGY_MIN_RESERVE)
    orchestrator = _build_heartbeat_only_orchestrator(initial_budget=max(0, reserve_floor - 8))
    orchestrator._select_executable_heartbeat_tasks = AsyncMock(return_value=[])

    budgets = []
    for _ in range(20):
        await Orchestrator._run_heartbeat_cycle(orchestrator)
        budgets.append(int(orchestrator._predictive_energy_budget_remaining))

    assert budgets[-1] >= reserve_floor

    first_recovered_index = next(index for index, value in enumerate(budgets) if value >= reserve_floor)
    assert all(value >= reserve_floor for value in budgets[first_recovered_index:])


@pytest.mark.asyncio
async def test_heartbeat_and_user_turn_replenishment_do_not_double_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "100")
    monkeypatch.setenv("ENERGY_REPLENISH_PER_HEARTBEAT", "2")

    orchestrator = _build_heartbeat_only_orchestrator(initial_budget=40)
    orchestrator._predictive_energy_budget_last_replenished_at = 1_000.0
    orchestrator._ready = asyncio.Event()
    orchestrator._ready.set()
    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}
    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()
    orchestrator._get_user_lock = AsyncMock(return_value=asyncio.Lock())
    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator._try_resume_tool_approval = AsyncMock(return_value=None)
    orchestrator._run_user_turn_locked = AsyncMock(return_value="ok")
    orchestrator._compute_predictive_energy_replenishment_points_wallclock_locked = MagicMock(return_value=3)

    select_started = asyncio.Event()
    allow_select_to_finish = asyncio.Event()

    async def _delayed_select():
        select_started.set()
        await allow_select_to_finish.wait()
        return []

    orchestrator._select_executable_heartbeat_tasks = AsyncMock(side_effect=_delayed_select)

    heartbeat_cycle = asyncio.create_task(Orchestrator._run_heartbeat_cycle(orchestrator))
    await asyncio.wait_for(select_started.wait(), timeout=1.0)

    user_response = await Orchestrator.process_message(orchestrator, "hello", "user-1")
    assert user_response == "ok"

    allow_select_to_finish.set()
    await heartbeat_cycle

    assert orchestrator._predictive_energy_budget_remaining == 45
