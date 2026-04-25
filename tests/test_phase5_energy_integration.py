import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.energy_judge import EnergyEvaluation
from src.core.energy_roi_engine import EnergyPolicy, EnergyROIEngine
from src.core.orchestrator import Orchestrator
from src.core.runtime_context import set_runtime_context
from src.memory.ledger_db import LedgerMemory
from src.skills.update_objective_status import update_objective_status as update_objective_status_skill


async def _fetch_task_row(ledger: LedgerMemory, task_id: int):
    cursor = await ledger._db.execute(
        "SELECT status, defer_count, next_eligible_at, last_energy_eval_json FROM objective_backlog WHERE id = ?",
        (task_id,),
    )
    return await cursor.fetchone()


@pytest.mark.asyncio
async def test_heartbeat_defers_low_roi_then_executes_high_roi_and_charges_once(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase5_slice4_heartbeat.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Energy Epic")
        story_id = await ledger.add_objective(tier="Story", title="Energy Story", parent_id=epic_id)
        low_roi_task_id = await ledger.add_objective(
            tier="Task",
            title="Low ROI candidate",
            parent_id=story_id,
            priority=1,
        )
        high_roi_task_id = await ledger.add_objective(
            tier="Task",
            title="High ROI candidate",
            parent_id=story_id,
            priority=1,
        )

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger
        orchestrator.outbound_queue = asyncio.Queue()
        orchestrator.process_message = AsyncMock(return_value="Task executed successfully.")
        orchestrator._heartbeat_failure_counts = {}
        orchestrator._predictive_energy_budget_remaining = 20
        orchestrator._predictive_energy_budget_lock = asyncio.Lock()

        # First candidate: low ROI -> deferred. Second candidate: high ROI -> executed.
        orchestrator.energy_judge = SimpleNamespace(
            evaluate_with_system1=AsyncMock(
                side_effect=[
                    EnergyEvaluation(estimated_effort=8, expected_value=4),
                    EnergyEvaluation(estimated_effort=2, expected_value=8),
                ]
            )
        )
        orchestrator.energy_roi_engine = EnergyROIEngine(
            EnergyPolicy(
                roi_threshold=1.2,
                min_reserve=5,
                effort_multiplier=3,
                defer_cooldown_seconds=120,
            )
        )

        await Orchestrator._run_heartbeat_cycle(orchestrator)

        low_row = await _fetch_task_row(ledger, low_roi_task_id)
        high_row = await _fetch_task_row(ledger, high_roi_task_id)

        assert str(low_row["status"]) == "deferred_due_to_energy"
        assert int(low_row["defer_count"]) == 1
        assert low_row["next_eligible_at"] is not None

        assert str(high_row["status"]) == "completed"
        assert orchestrator.process_message.await_count == 1

        heartbeat_prompt = orchestrator.process_message.await_args.kwargs["user_message"]
        assert f"[HEARTBEAT TASK #{high_roi_task_id}]" in heartbeat_prompt

        # Heartbeat replenishes +2 before dispatch, then executed task consumes 6.
        assert orchestrator._predictive_energy_budget_remaining == 16

        # Deferred tasks must not trigger strike/block handling.
        assert low_roi_task_id not in orchestrator._heartbeat_failure_counts
        assert high_roi_task_id not in orchestrator._heartbeat_failure_counts

        notifications = []
        while not orchestrator.outbound_queue.empty():
            notifications.append(await orchestrator.outbound_queue.get())
        assert all("HITL REQUIRED" not in message for message in notifications)
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_heartbeat_fairness_policy_eventually_executes_after_multiple_defers(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase5_slice5_fairness_cycles.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Fairness Epic")
        story_id = await ledger.add_objective(tier="Story", title="Fairness Story", parent_id=epic_id)
        task_id = await ledger.add_objective(
            tier="Task",
            title="Low ROI but necessary task",
            parent_id=story_id,
            priority=1,
        )

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger
        orchestrator.outbound_queue = asyncio.Queue()
        orchestrator.process_message = AsyncMock(return_value="Task executed successfully after fairness aging.")
        orchestrator._heartbeat_failure_counts = {}
        orchestrator._predictive_energy_budget_remaining = 50
        orchestrator._predictive_energy_budget_lock = asyncio.Lock()

        orchestrator.energy_judge = SimpleNamespace(
            evaluate_with_system1=AsyncMock(
                return_value=EnergyEvaluation(estimated_effort=8, expected_value=2)
            )
        )
        orchestrator.energy_roi_engine = EnergyROIEngine(
            EnergyPolicy(
                roi_threshold=3.0,
                min_reserve=5,
                effort_multiplier=1,
                defer_cooldown_seconds=0,
                fairness_boost_multiplier=0.05,
                max_defer_count=3,
            )
        )

        for _ in range(3):
            await Orchestrator._run_heartbeat_cycle(orchestrator)
            assert orchestrator.process_message.await_count == 0

        deferred_row = await _fetch_task_row(ledger, task_id)
        assert str(deferred_row["status"]) == "deferred_due_to_energy"
        assert int(deferred_row["defer_count"]) == 3

        await Orchestrator._run_heartbeat_cycle(orchestrator)

        final_row = await _fetch_task_row(ledger, task_id)
        assert str(final_row["status"]) == "completed"
        assert int(final_row["defer_count"]) == 3
        assert orchestrator.process_message.await_count == 1

        eval_payload = json.loads(str(final_row["last_energy_eval_json"]))
        assert eval_payload["should_execute"] is True
        assert "max defer bypass" in str(eval_payload["reason"]).lower()
        assert float(eval_payload["base_roi"]) < float(eval_payload["roi_threshold"])
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_completion_updates_refund_predictive_budget_after_success_streak(tmp_path: Path, monkeypatch):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase8_completion_refund.db"))
    await ledger.initialize()
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "40")
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Refund Epic")
        story_id = await ledger.add_objective(tier="Story", title="Refund Story", parent_id=epic_id)
        task_ids = [
            await ledger.add_objective(
                tier="Task",
                title=f"Refund Task #{index}",
                parent_id=story_id,
                estimated_energy=25,
            )
            for index in range(1, 5)
        ]

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator._predictive_energy_budget_lock = asyncio.Lock()
        orchestrator._predictive_energy_budget_remaining = 20

        set_runtime_context(ledger, None, None, orchestrator)
        observed_budgets = []
        for task_id in task_ids:
            payload = json.loads(await update_objective_status_skill(task_id, "completed"))
            assert payload["status"] == "success"
            observed_budgets.append(orchestrator._predictive_energy_budget_remaining)

        assert observed_budgets[0] == 25
        assert observed_budgets[-1] == 40
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()


@pytest.mark.asyncio
async def test_completion_update_clears_persisted_heartbeat_failure_strikes(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "phase8_completion_clear_heartbeat.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Clear Strikes Epic")
        story_id = await ledger.add_objective(tier="Story", title="Clear Strikes Story", parent_id=epic_id)
        task_id = await ledger.add_objective(
            tier="Task",
            title="Clear Strikes Task",
            parent_id=story_id,
        )

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger
        orchestrator._heartbeat_failure_counts = {task_id: 2}
        await Orchestrator._persist_heartbeat_failure_counts(orchestrator)

        set_runtime_context(ledger, None, None, orchestrator)
        payload = json.loads(await update_objective_status_skill(task_id, "completed"))

        assert payload["status"] == "success"
        assert task_id not in orchestrator._heartbeat_failure_counts

        persisted = await ledger.get_system_state("heartbeat_task_failure_counts")
        decoded = json.loads(str(persisted or "{}"))
        assert str(task_id) not in decoded
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()
