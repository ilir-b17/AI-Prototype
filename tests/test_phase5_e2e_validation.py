import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core.energy_judge import EnergyJudge
from src.core.energy_roi_engine import EnergyPolicy, EnergyROIEngine
from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator
from src.memory.ledger_db import LedgerMemory


async def _fetch_status(ledger: LedgerMemory, objective_id: int) -> str:
    cursor = await ledger._db.execute(
        "SELECT status FROM objective_backlog WHERE id = ?",
        (objective_id,),
    )
    row = await cursor.fetchone()
    return str(row["status"])


async def _fetch_task_row(ledger: LedgerMemory, task_id: int):
    cursor = await ledger._db.execute(
        "SELECT status, defer_count, next_eligible_at FROM objective_backlog WHERE id = ?",
        (task_id,),
    )
    return await cursor.fetchone()


@pytest.mark.asyncio
async def test_phase5_slice6_e2e_heartbeat_energy_gating_budget_and_rollup(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "phase5_slice6_e2e.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Slice 5 E2E Epic")
        story_id = await ledger.add_objective(tier="Story", title="Slice 5 E2E Story", parent_id=epic_id)

        task_a_title = "Task A low-value high-effort"
        task_b_title = "Task B high-value low-effort"

        task_a_id = await ledger.add_objective(
            tier="Task",
            title=task_a_title,
            parent_id=story_id,
            priority=1,
            estimated_energy=5,
        )
        task_b_id = await ledger.add_objective(
            tier="Task",
            title=task_b_title,
            parent_id=story_id,
            priority=1,
            estimated_energy=5,
        )

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.ledger_memory = ledger
        orchestrator.outbound_queue = asyncio.Queue()
        orchestrator.process_message = AsyncMock(
            side_effect=[
                "Task B executed successfully.",
                "Task A executed successfully after fairness bypass.",
            ]
        )
        orchestrator._heartbeat_failure_counts = {}
        orchestrator._predictive_energy_budget_remaining = 20
        orchestrator._predictive_energy_budget_lock = asyncio.Lock()

        async def _route_energy_judge(messages, **_kwargs):
            await asyncio.sleep(0)
            user_payload = str(messages[-1].get("content") or "")
            if task_a_title in user_payload:
                # Low-value / high-effort profile.
                return RouterResult(
                    status="ok",
                    content='{"estimated_effort": 9, "expected_value": 2}',
                )
            if task_b_title in user_payload:
                # High-value / low-effort profile.
                return RouterResult(
                    status="ok",
                    content='{"estimated_effort": 2, "expected_value": 9}',
                )
            return RouterResult(
                status="ok",
                content='{"estimated_effort": 5, "expected_value": 5}',
            )

        orchestrator._route_to_system_1 = AsyncMock(side_effect=_route_energy_judge)
        orchestrator.energy_judge = EnergyJudge()
        orchestrator.energy_roi_engine = EnergyROIEngine(
            EnergyPolicy(
                roi_threshold=1.25,
                min_reserve=0,
                effort_multiplier=1,
                defer_cooldown_seconds=0,
                fairness_boost_multiplier=0.15,
                max_defer_count=1,
            )
        )

        # Cycle 1: A should defer, B should execute and consume budget.
        await Orchestrator._run_heartbeat_cycle(orchestrator)

        task_a_row = await _fetch_task_row(ledger, task_a_id)
        task_b_row = await _fetch_task_row(ledger, task_b_id)
        assert str(task_a_row["status"]) == "deferred_due_to_energy"
        assert int(task_a_row["defer_count"]) == 1
        assert str(task_b_row["status"]) == "completed"

        assert orchestrator.process_message.await_count == 1
        first_prompt = orchestrator.process_message.await_args_list[0].kwargs["user_message"]
        assert f"[HEARTBEAT TASK #{task_b_id}]" in first_prompt

        # Heartbeat replenishes +2 then task B costs 2 => net unchanged.
        assert orchestrator._predictive_energy_budget_remaining == 20

        # Roll-up should reflect one of two tasks completed.
        rollup_after_cycle_1 = await ledger.get_objective_hierarchy_rollup(epic_id=epic_id)
        story_rollup_1 = next(item for item in rollup_after_cycle_1 if item["id"] == story_id)
        epic_rollup_1 = next(item for item in rollup_after_cycle_1 if item["id"] == epic_id)
        assert story_rollup_1["completed_tasks"] == 1
        assert story_rollup_1["total_tasks"] == 2
        assert story_rollup_1["completion_ratio"] == pytest.approx(0.5)
        assert story_rollup_1["rolled_up_status"] == "pending"
        assert epic_rollup_1["completion_ratio"] == pytest.approx(0.5)
        assert epic_rollup_1["rolled_up_status"] == "pending"

        # Cycle 2: A now hits max-defer bypass and executes.
        await Orchestrator._run_heartbeat_cycle(orchestrator)

        assert await _fetch_status(ledger, task_a_id) == "completed"
        assert await _fetch_status(ledger, story_id) == "completed"
        assert await _fetch_status(ledger, epic_id) == "completed"

        # Second cycle replenishes +2 then task A costs 9.
        assert orchestrator._predictive_energy_budget_remaining == 13

        rollup_after_cycle_2 = await ledger.get_objective_hierarchy_rollup(epic_id=epic_id)
        story_rollup_2 = next(item for item in rollup_after_cycle_2 if item["id"] == story_id)
        epic_rollup_2 = next(item for item in rollup_after_cycle_2 if item["id"] == epic_id)
        assert story_rollup_2["completion_ratio"] == pytest.approx(1.0)
        assert story_rollup_2["rolled_up_status"] == "completed"
        assert epic_rollup_2["completion_ratio"] == pytest.approx(1.0)
        assert epic_rollup_2["rolled_up_status"] == "completed"
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_phase5_slice6_cloud_guard_energy_evaluation_never_calls_system2():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.energy_judge = EnergyJudge()
    orchestrator.energy_roi_engine = EnergyROIEngine(
        EnergyPolicy(
            roi_threshold=1.25,
            min_reserve=0,
            effort_multiplier=1,
            defer_cooldown_seconds=60,
            fairness_boost_multiplier=0.15,
            max_defer_count=5,
        )
    )
    orchestrator._predictive_energy_budget_remaining = 100
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()

    orchestrator._route_to_system_1 = AsyncMock(
        return_value=RouterResult(
            status="ok",
            content='{"estimated_effort": 2, "expected_value": 8}',
        )
    )
    orchestrator._route_to_system_2_redacted = AsyncMock(
        return_value=RouterResult(status="ok", content="unexpected cloud call")
    )
    orchestrator.cognitive_router = SimpleNamespace(
        route_to_system_2=AsyncMock(
            return_value=RouterResult(status="ok", content="unexpected cloud call")
        )
    )

    task = {
        "id": 101,
        "title": "Energy-only local evaluation path",
        "acceptance_criteria": "Return local effort/value JSON.",
        "estimated_energy": 4,
        "status": "pending",
        "depends_on_ids": [],
        "defer_count": 0,
    }

    evaluation, decision, available = await Orchestrator._evaluate_energy_for_context(
        orchestrator,
        task=task,
        story={"id": 11, "title": "Story", "status": "active"},
        epic={"id": 1, "title": "Epic", "status": "active"},
        additional_context="cloud guard test",
    )

    assert evaluation.estimated_effort == 2
    assert evaluation.expected_value == 8
    assert decision.should_execute is True
    assert available == 100

    orchestrator._route_to_system_1.assert_awaited_once()
    orchestrator._route_to_system_2_redacted.assert_not_called()
    orchestrator.cognitive_router.route_to_system_2.assert_not_called()
