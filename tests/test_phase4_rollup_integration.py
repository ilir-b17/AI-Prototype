import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.energy_judge import EnergyEvaluation
from src.core.energy_roi_engine import EnergyPolicy, EnergyROIEngine
from src.core.orchestrator import Orchestrator
from src.memory.ledger_db import LedgerMemory


async def _get_objective_status(ledger: LedgerMemory, objective_id: int) -> str:
    cursor = await ledger._db.execute(
        "SELECT status FROM objective_backlog WHERE id = ?",
        (objective_id,),
    )
    row = await cursor.fetchone()
    return str(row["status"])


def _build_heartbeat_test_orchestrator(ledger: LedgerMemory, process_side_effect):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = ledger
    orchestrator.outbound_queue = asyncio.Queue()
    orchestrator.process_message = AsyncMock(side_effect=process_side_effect)
    orchestrator._heartbeat_failure_counts = {}
    orchestrator.energy_judge = MagicMock()
    orchestrator.energy_judge.evaluate_with_system1 = AsyncMock(
        return_value=EnergyEvaluation(estimated_effort=1, expected_value=9)
    )
    orchestrator.energy_roi_engine = EnergyROIEngine(
        EnergyPolicy(
            roi_threshold=0.1,
            min_reserve=0,
            effort_multiplier=1,
            defer_cooldown_seconds=30,
        )
    )
    orchestrator._predictive_energy_budget_remaining = 100
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    return orchestrator


@pytest.mark.asyncio
async def test_heartbeat_dependency_execution_and_bottom_up_rollup(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase4_slice3_rollup.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Epic")
        story_id = await ledger.add_objective(tier="Story", title="Story", parent_id=epic_id)
        task_a_id = await ledger.add_objective(
            tier="Task",
            title="Task A",
            parent_id=story_id,
            priority=1,
        )
        task_b_id = await ledger.add_objective(
            tier="Task",
            title="Task B",
            parent_id=story_id,
            priority=1,
            depends_on_ids=[task_a_id],
        )

        orchestrator = _build_heartbeat_test_orchestrator(
            ledger,
            process_side_effect=["Task A completed successfully.", "Task B completed successfully."],
        )

        await Orchestrator._run_heartbeat_cycle(orchestrator)

        assert await _get_objective_status(ledger, task_a_id) == "completed"
        assert await _get_objective_status(ledger, task_b_id) == "pending"
        assert await _get_objective_status(ledger, story_id) == "pending"
        assert await _get_objective_status(ledger, epic_id) == "pending"

        await Orchestrator._run_heartbeat_cycle(orchestrator)

        assert await _get_objective_status(ledger, task_b_id) == "completed"
        assert await _get_objective_status(ledger, story_id) == "completed"
        assert await _get_objective_status(ledger, epic_id) == "completed"
        assert orchestrator.process_message.await_count == 2
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_heartbeat_three_strike_failure_blocks_task_and_notifies_admin(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase4_slice3_blocked.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Epic")
        story_id = await ledger.add_objective(tier="Story", title="Story", parent_id=epic_id)
        task_id = await ledger.add_objective(
            tier="Task",
            title="Failing Task",
            parent_id=story_id,
            priority=1,
        )

        orchestrator = _build_heartbeat_test_orchestrator(
            ledger,
            process_side_effect=[
                "Error: failed attempt 1",
                "Error: failed attempt 2",
                "Error: failed attempt 3",
            ],
        )

        await Orchestrator._run_heartbeat_cycle(orchestrator)
        await Orchestrator._run_heartbeat_cycle(orchestrator)
        await Orchestrator._run_heartbeat_cycle(orchestrator)

        assert await _get_objective_status(ledger, task_id) == "blocked"
        assert await _get_objective_status(ledger, story_id) == "active"
        assert await _get_objective_status(ledger, epic_id) == "active"

        notifications = []
        while not orchestrator.outbound_queue.empty():
            notifications.append(await orchestrator.outbound_queue.get())

        assert any("HITL REQUIRED" in message for message in notifications)
    finally:
        await ledger.close()
