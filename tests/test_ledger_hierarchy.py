import sqlite3
from pathlib import Path

import pytest

from src.memory.ledger_db import LedgerMemory


@pytest.mark.asyncio
async def test_objective_backlog_schema_migrates_legacy_tables(tmp_path: Path):
    db_path = tmp_path / "legacy_ledger.db"

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

    ledger = LedgerMemory(db_path=str(db_path))
    await ledger.initialize()
    try:
        cursor = await ledger._db.execute("PRAGMA table_info(objective_backlog)")
        rows = await cursor.fetchall()
        columns = {row["name"] for row in rows}

        assert "depends_on_ids" in columns
        assert "acceptance_criteria" in columns

        epic_id = await ledger.add_objective(tier="Epic", title="Migration test epic")
        story_id = await ledger.add_objective(
            tier="Story",
            title="Migration test story",
            parent_id=epic_id,
            acceptance_criteria="Story must define measurable acceptance criteria",
        )
        task_id = await ledger.add_objective(
            tier="Task",
            title="Migration test task",
            parent_id=story_id,
            depends_on_ids=[story_id],
            acceptance_criteria="Task should be executable and dependency aware",
        )

        tree = await ledger.get_active_objective_tree(epic_id=epic_id)
        task_row = next(node for node in tree if node["id"] == task_id)

        assert task_row["depends_on_ids"] == [story_id]
        assert "dependency aware" in task_row["acceptance_criteria"]
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_active_tree_returns_only_active_nodes_and_preserves_hierarchy_fields(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "tree_ledger.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Execution Epic")
        story_id = await ledger.add_objective(
            tier="Story",
            title="Execution Story",
            parent_id=epic_id,
            acceptance_criteria="Story accepted when all tasks pass tests",
        )
        task_open_id = await ledger.add_objective(
            tier="Task",
            title="Open task",
            parent_id=story_id,
            acceptance_criteria="Task produces expected result",
        )
        task_done_id = await ledger.add_objective(
            tier="Task",
            title="Completed task",
            parent_id=story_id,
            depends_on_ids=[task_open_id],
            acceptance_criteria="This task is completed",
        )
        await ledger.update_objective_status(task_done_id, "completed")

        tree = await ledger.get_active_objective_tree(epic_id=epic_id)
        ids = {node["id"] for node in tree}

        assert epic_id in ids
        assert story_id in ids
        assert task_open_id in ids
        assert task_done_id not in ids

        open_task = next(node for node in tree if node["id"] == task_open_id)
        assert open_task["tier"] == "Task"
        assert open_task["depends_on_ids"] == []
        assert "expected result" in open_task["acceptance_criteria"]
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_detects_unresolved_dependencies_for_task_nodes(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "deps_ledger.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Dependency Epic")
        story_id = await ledger.add_objective(tier="Story", title="Dependency Story", parent_id=epic_id)
        task_a = await ledger.add_objective(tier="Task", title="Task A", parent_id=story_id)
        task_b = await ledger.add_objective(
            tier="Task",
            title="Task B",
            parent_id=story_id,
            depends_on_ids=[task_a, 9999],
        )

        unresolved_before = await ledger.get_unresolved_depends_on_ids(task_b)
        assert unresolved_before == [task_a, 9999]

        await ledger.update_objective_status(task_a, "completed")
        unresolved_after = await ledger.get_unresolved_depends_on_ids(task_b)
        assert unresolved_after == [9999]

        unresolved_tasks = await ledger.get_tasks_with_unresolved_dependencies()
        unresolved_task_ids = {row["id"] for row in unresolved_tasks}
        assert task_b in unresolved_task_ids
        task_b_row = next(row for row in unresolved_tasks if row["id"] == task_b)
        assert task_b_row["unresolved_depends_on_ids"] == [9999]
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_rollup_status_and_completion_ratio_for_epic_story(tmp_path: Path):
    ledger = LedgerMemory(db_path=str(tmp_path / "rollup_ledger.db"))
    await ledger.initialize()
    try:
        epic_id = await ledger.add_objective(tier="Epic", title="Rollup Epic")
        story_id = await ledger.add_objective(tier="Story", title="Rollup Story", parent_id=epic_id)
        task_1 = await ledger.add_objective(tier="Task", title="Task 1", parent_id=story_id)
        task_2 = await ledger.add_objective(tier="Task", title="Task 2", parent_id=story_id)

        await ledger.update_objective_status(task_1, "completed")

        partial_rollup = await ledger.get_objective_hierarchy_rollup(epic_id=epic_id)
        epic_partial = next(row for row in partial_rollup if row["id"] == epic_id)
        story_partial = next(row for row in partial_rollup if row["id"] == story_id)

        assert story_partial["total_tasks"] == 2
        assert story_partial["completed_tasks"] == 1
        assert story_partial["completion_ratio"] == pytest.approx(0.5)
        assert story_partial["rolled_up_status"] == "pending"

        assert epic_partial["total_tasks"] == 2
        assert epic_partial["completed_tasks"] == 1
        assert epic_partial["completion_ratio"] == pytest.approx(0.5)
        assert epic_partial["rolled_up_status"] == "pending"

        await ledger.update_objective_status(task_2, "completed")

        completed_rollup = await ledger.get_objective_hierarchy_rollup(epic_id=epic_id)
        epic_complete = next(row for row in completed_rollup if row["id"] == epic_id)
        story_complete = next(row for row in completed_rollup if row["id"] == story_id)

        status_cursor = await ledger._db.execute(
            "SELECT id, status FROM objective_backlog WHERE id IN (?, ?)",
            (epic_id, story_id),
        )
        status_rows = await status_cursor.fetchall()
        status_by_id = {int(row["id"]): str(row["status"]) for row in status_rows}

        assert story_complete["completion_ratio"] == pytest.approx(1.0)
        assert story_complete["rolled_up_status"] == "completed"
        assert epic_complete["completion_ratio"] == pytest.approx(1.0)
        assert epic_complete["rolled_up_status"] == "completed"
        assert status_by_id[story_id] == "completed"
        assert status_by_id[epic_id] == "completed"
    finally:
        await ledger.close()
