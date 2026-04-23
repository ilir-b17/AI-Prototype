import sqlite3
from pathlib import Path

import pytest

from src.memory.ledger_db import LedgerMemory


@pytest.mark.asyncio
async def test_energy_contract_columns_migrate_legacy_schema_idempotently(tmp_path: Path):
    db_path = tmp_path / "energy_legacy.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE objective_backlog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tier TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 5,
                estimated_energy INTEGER NOT NULL DEFAULT 10,
                origin TEXT NOT NULL DEFAULT 'Admin',
                parent_id INTEGER,
                depends_on_ids TEXT NOT NULL DEFAULT '[]',
                acceptance_criteria TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

    ledger = LedgerMemory(db_path=str(db_path))
    await ledger.initialize()
    try:
        await ledger._initialize_tables()

        cursor = await ledger._db.execute("PRAGMA table_info(objective_backlog)")
        rows = await cursor.fetchall()
        columns = {row["name"] for row in rows}

        assert "defer_count" in columns
        assert "last_energy_eval_at" in columns
        assert "next_eligible_at" in columns
        assert "last_energy_eval_json" in columns
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_update_objective_status_accepts_deferred_due_to_energy(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "energy_status.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Energy Epic")
        story_id = await ledger.add_objective(tier="Story", title="Energy Story", parent_id=epic_id)
        task_id = await ledger.add_objective(tier="Task", title="Energy Task", parent_id=story_id)

        await ledger.update_objective_status(task_id, "deferred_due_to_energy")

        cursor = await ledger._db.execute(
            "SELECT status FROM objective_backlog WHERE id = ?",
            (task_id,),
        )
        row = await cursor.fetchone()
        assert str(row["status"]) == "deferred_due_to_energy"
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_energy_candidate_queries_include_parent_context_and_respect_next_eligible(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "energy_candidates.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(
            tier="Epic",
            title="Parent Epic",
            acceptance_criteria="Complete all child stories.",
        )
        story_id = await ledger.add_objective(
            tier="Story",
            title="Parent Story",
            parent_id=epic_id,
            acceptance_criteria="Complete all child tasks.",
        )
        task_ready_id = await ledger.add_objective(
            tier="Task",
            title="Ready Task",
            parent_id=story_id,
            acceptance_criteria="Ready now",
        )
        task_future_id = await ledger.add_objective(
            tier="Task",
            title="Deferred Task",
            parent_id=story_id,
            acceptance_criteria="Not eligible yet",
        )

        await ledger.update_objective_status(task_future_id, "deferred_due_to_energy")
        async with ledger._lock:
            await ledger._db.execute(
                "UPDATE objective_backlog "
                "SET defer_count = 2, "
                "last_energy_eval_json = ?, "
                "next_eligible_at = datetime('now', '+1 hour') "
                "WHERE id = ?",
                ('{"estimated_effort": 9, "expected_value": 3}', task_future_id),
            )
            await ledger._db.commit()

        context = await ledger.get_task_with_parent_context(task_ready_id)
        assert context is not None
        assert context["task"]["id"] == task_ready_id
        assert context["story"]["id"] == story_id
        assert context["epic"]["id"] == epic_id

        candidates_now = await ledger.get_energy_evaluation_candidates()
        candidate_task_ids_now = {entry["task"]["id"] for entry in candidates_now}
        assert task_ready_id in candidate_task_ids_now
        assert task_future_id not in candidate_task_ids_now

        async with ledger._lock:
            await ledger._db.execute(
                "UPDATE objective_backlog SET next_eligible_at = datetime('now', '-1 hour') WHERE id = ?",
                (task_future_id,),
            )
            await ledger._db.commit()

        candidates_later = await ledger.get_energy_evaluation_candidates()
        candidate_task_ids_later = {entry["task"]["id"] for entry in candidates_later}
        assert task_ready_id in candidate_task_ids_later
        assert task_future_id in candidate_task_ids_later
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_highest_priority_task_excludes_future_next_eligible(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "energy_highest_priority_eligibility.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Eligibility Epic")
        story_id = await ledger.add_objective(tier="Story", title="Eligibility Story", parent_id=epic_id)
        blocked_id = await ledger.add_objective(
            tier="Task",
            title="Not yet eligible",
            parent_id=story_id,
            priority=1,
            estimated_energy=1,
        )
        ready_id = await ledger.add_objective(
            tier="Task",
            title="Eligible now",
            parent_id=story_id,
            priority=2,
            estimated_energy=1,
        )

        async with ledger._lock:
            await ledger._db.execute(
                "UPDATE objective_backlog SET next_eligible_at = datetime('now', '+30 minutes') WHERE id = ?",
                (blocked_id,),
            )
            await ledger._db.commit()

        top = await ledger.get_highest_priority_task()
        assert top is not None
        assert int(top["id"]) == ready_id

        async with ledger._lock:
            await ledger._db.execute(
                "UPDATE objective_backlog SET next_eligible_at = datetime('now', '-5 minutes') WHERE id = ?",
                (blocked_id,),
            )
            await ledger._db.commit()

        top_after_expiry = await ledger.get_highest_priority_task()
        assert top_after_expiry is not None
        assert int(top_after_expiry["id"]) == blocked_id
    finally:
        await ledger.close()
