"""Tests for Outcome Ledger Foundation.

Covers:
- Schema migration (new outcome columns added to objective_backlog)
- write_task_outcome / get_historical_outcome_scores / get_outcome_learnings
- OutcomeJudge.is_trivial, score_system1, record_outcome
- EnergyJudge.blend_with_historical_scores
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.core.energy_judge import EnergyEvaluation, EnergyJudge
from src.core.outcome_judge import OutcomeJudge
from src.memory.ledger_db import LedgerMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_ledger(tmp_path: Path) -> LedgerMemory:
    ledger = LedgerMemory(db_path=str(tmp_path / "test_ledger.db"))
    await ledger.initialize()
    return ledger


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outcome_columns_added_to_fresh_db(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        cursor = await ledger._db.execute("PRAGMA table_info(objective_backlog)")
        rows = await cursor.fetchall()
        columns = {row["name"] for row in rows}
        assert "outcome_score" in columns
        assert "actual_energy_used" in columns
        assert "actual_duration_seconds" in columns
        assert "outcome_recorded_at" in columns
        assert "outcome_notes" in columns
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_outcome_columns_migrated_onto_legacy_schema(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
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
        assert "outcome_score" in columns
        assert "actual_energy_used" in columns
        assert "actual_duration_seconds" in columns
        assert "outcome_recorded_at" in columns
        assert "outcome_notes" in columns
    finally:
        await ledger.close()


# ---------------------------------------------------------------------------
# write_task_outcome
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_task_outcome_persists_all_fields(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        task_id = await ledger.add_objective(tier="Task", title="Write unit tests")
        await ledger.write_task_outcome(
            task_id,
            score=4,
            actual_energy_used=12,
            actual_duration_seconds=300,
            outcome_notes="Tests passed with minor gaps",
        )
        row = await ledger.get_task_row(task_id)
        assert row is not None
        assert row["outcome_score"] == 4
        assert row["actual_energy_used"] == 12
        assert row["actual_duration_seconds"] == 300
        assert row["outcome_notes"] == "Tests passed with minor gaps"
        assert row["outcome_recorded_at"] is not None
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_write_task_outcome_clamps_score_to_1_5(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        task_id = await ledger.add_objective(tier="Task", title="Clamping test task")
        await ledger.write_task_outcome(task_id, score=99)
        row = await ledger.get_task_row(task_id)
        assert row["outcome_score"] == 5

        task_id2 = await ledger.add_objective(tier="Task", title="Clamping test task low")
        await ledger.write_task_outcome(task_id2, score=-10)
        row2 = await ledger.get_task_row(task_id2)
        assert row2["outcome_score"] == 1
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_write_task_outcome_optional_fields_can_be_null(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        task_id = await ledger.add_objective(tier="Task", title="Minimal outcome task")
        await ledger.write_task_outcome(task_id, score=3)
        row = await ledger.get_task_row(task_id)
        assert row["outcome_score"] == 3
        assert row["actual_energy_used"] is None
        assert row["actual_duration_seconds"] is None
        assert row["outcome_notes"] is None
    finally:
        await ledger.close()


# ---------------------------------------------------------------------------
# get_historical_outcome_scores
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_historical_outcome_scores_returns_matching_tier(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        for score in [3, 4, 5]:
            tid = await ledger.add_objective(tier="Task", title="Analyze performance metrics")
            await ledger.write_task_outcome(tid, score=score)

        scores = await ledger.get_historical_outcome_scores("Task", ["analyze", "performance"])
        assert len(scores) == 3
        assert set(scores) == {3, 4, 5}
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_get_historical_outcome_scores_filters_by_token_overlap(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        t1 = await ledger.add_objective(tier="Task", title="Refactor database module")
        await ledger.write_task_outcome(t1, score=4)

        t2 = await ledger.add_objective(tier="Task", title="Write new feature")
        await ledger.write_task_outcome(t2, score=2)

        # 'database' should match t1 but not t2
        scores = await ledger.get_historical_outcome_scores("Task", ["database"])
        assert 4 in scores
        assert 2 not in scores
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_get_historical_outcome_scores_empty_when_no_outcomes(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        await ledger.add_objective(tier="Task", title="Task without outcome")
        scores = await ledger.get_historical_outcome_scores("Task", ["task"])
        assert scores == []
    finally:
        await ledger.close()


# ---------------------------------------------------------------------------
# get_outcome_learnings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_outcome_learnings_top_and_bottom(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        for i, score in enumerate([5, 4, 3, 2, 1, 5, 4]):
            tid = await ledger.add_objective(
                tier="Task", title=f"Learning task variant {i}"
            )
            await ledger.write_task_outcome(tid, score=score, actual_energy_used=10)

        result = await ledger.get_outcome_learnings(days=30)
        assert "top_classes" in result
        assert "bottom_classes" in result
        assert isinstance(result["top_classes"], list)
        assert isinstance(result["bottom_classes"], list)
        assert result["days_window"] == 30
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_get_outcome_learnings_energy_accuracy(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        # Task where estimate is accurate (within 50%)
        t1 = await ledger.add_objective(
            tier="Task", title="Accurate estimate task", estimated_energy=10
        )
        await ledger.write_task_outcome(t1, score=4, actual_energy_used=12)

        # Task where estimate is inaccurate (more than 50% off)
        t2 = await ledger.add_objective(
            tier="Task", title="Inaccurate estimate task", estimated_energy=10
        )
        await ledger.write_task_outcome(t2, score=3, actual_energy_used=30)

        result = await ledger.get_outcome_learnings(days=30)
        assert result["energy_accuracy_sample"] == 2
        # 1 out of 2 tasks is accurate (within the _ENERGY_ACCURACY_THRESHOLD=50%)
        # → 50.0%; this assertion depends on the threshold constant
        assert result["energy_accuracy_pct"] == 50.0
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_get_outcome_learnings_returns_none_accuracy_when_no_data(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        result = await ledger.get_outcome_learnings(days=30)
        assert result["energy_accuracy_pct"] is None
        assert result["energy_accuracy_sample"] == 0
        assert result["top_classes"] == []
        assert result["bottom_classes"] == []
    finally:
        await ledger.close()


# ---------------------------------------------------------------------------
# OutcomeJudge.is_trivial
# ---------------------------------------------------------------------------


def test_outcome_judge_is_trivial_detects_read_only_verbs():
    assert OutcomeJudge.is_trivial({"title": "scan the /src/ directory"}) is True
    assert OutcomeJudge.is_trivial({"title": "Read configuration file"}) is True
    assert OutcomeJudge.is_trivial({"title": "List all active goals"}) is True
    assert OutcomeJudge.is_trivial({"title": "Summarize the chat history"}) is True
    assert OutcomeJudge.is_trivial({"title": "Analyze error logs for patterns"}) is True


def test_outcome_judge_is_trivial_rejects_action_verbs():
    assert OutcomeJudge.is_trivial({"title": "Write a test suite"}) is False
    assert OutcomeJudge.is_trivial({"title": "Deploy the staging environment"}) is False
    assert OutcomeJudge.is_trivial({"title": "Refactor the database module"}) is False
    assert OutcomeJudge.is_trivial({"title": "Fix the authentication bug"}) is False


def test_outcome_judge_is_trivial_on_empty_title():
    assert OutcomeJudge.is_trivial({"title": ""}) is True
    assert OutcomeJudge.is_trivial({}) is True


# ---------------------------------------------------------------------------
# OutcomeJudge.score_system1
# ---------------------------------------------------------------------------


def test_score_system1_returns_5_when_all_criteria_met():
    task = {
        "acceptance_criteria": "deployed running healthy verified",
        "result_json": json.dumps({"output": "service deployed running healthy verified on port 8080"}),
    }
    assert OutcomeJudge.score_system1(task) == 5


def test_score_system1_returns_1_when_no_result():
    task = {
        "acceptance_criteria": "output must contain success message",
        "result_json": None,
    }
    assert OutcomeJudge.score_system1(task) == 1


def test_score_system1_returns_3_when_no_criteria():
    task = {
        "acceptance_criteria": "",
        "result_json": json.dumps({"output": "something"}),
    }
    assert OutcomeJudge.score_system1(task) == 3


def test_score_system1_partial_match_returns_mid_score():
    task = {
        "acceptance_criteria": "alpha beta gamma delta epsilon",
        "result_json": json.dumps({"text": "alpha beta only"}),
    }
    # 2 of 5 tokens match → 40% → score 3
    score = OutcomeJudge.score_system1(task)
    assert 2 <= score <= 3


# ---------------------------------------------------------------------------
# OutcomeJudge.record_outcome (integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_outcome_trivial_task_gets_score_3(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        task_id = await ledger.add_objective(tier="Task", title="Scan log files for errors")
        # Mark completed so get_task_row returns the row
        await ledger.update_objective_status(task_id, "completed")

        await OutcomeJudge.record_outcome(ledger, task_id)

        row = await ledger.get_task_row(task_id)
        assert row["outcome_score"] == 3
        assert "triviality_bypass" in (row["outcome_notes"] or "")
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_record_outcome_action_task_scores_by_criteria(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        task_id = await ledger.add_objective(
            tier="Task",
            title="Deploy the new service",
            acceptance_criteria="deployment must succeed and service must be running",
        )
        await ledger.update_objective_status(task_id, "active")
        # Write a result that satisfies criteria
        await ledger.write_task_result(
            task_id,
            {"output": "deployment succeeded and service is running on port 8080"},
        )

        await OutcomeJudge.record_outcome(ledger, task_id)

        row = await ledger.get_task_row(task_id)
        assert row["outcome_score"] is not None
        assert 1 <= row["outcome_score"] <= 5
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_record_outcome_idempotent_does_not_overwrite(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        task_id = await ledger.add_objective(tier="Task", title="Fix auth bug")
        await ledger.update_objective_status(task_id, "completed")

        await OutcomeJudge.record_outcome(ledger, task_id)
        row_after_first = await ledger.get_task_row(task_id)
        first_score = row_after_first["outcome_score"]

        # Manually set a different score to verify idempotency check
        await ledger.write_task_outcome(task_id, score=5)

        # Second record_outcome call should be skipped (outcome_score already set)
        await OutcomeJudge.record_outcome(ledger, task_id)
        row_after_second = await ledger.get_task_row(task_id)
        assert row_after_second["outcome_score"] == 5  # unchanged (not re-graded)
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_record_outcome_missing_task_does_not_raise(tmp_path: Path):
    ledger = await _make_ledger(tmp_path)
    try:
        # Should silently do nothing, not raise
        await OutcomeJudge.record_outcome(ledger, task_id=99999)
    finally:
        await ledger.close()


# ---------------------------------------------------------------------------
# EnergyJudge.blend_with_historical_scores
# ---------------------------------------------------------------------------


def test_blend_with_historical_scores_no_change_below_5_samples():
    judge = EnergyJudge()
    base = EnergyEvaluation(estimated_effort=4, expected_value=8)
    blended = judge.blend_with_historical_scores(base, [4, 3])
    assert blended.expected_value == 8  # unchanged
    assert blended.estimated_effort == 4


def test_blend_with_historical_scores_applies_70pct_history_at_5_samples():
    judge = EnergyJudge()
    base = EnergyEvaluation(estimated_effort=4, expected_value=10)
    # All 5 historical outcomes are 1/5 → scaled to 1/10
    blended = judge.blend_with_historical_scores(base, [1, 1, 1, 1, 1])
    # historical_avg_scaled = 1.0
    # blended = round(0.7 * 1.0 + 0.3 * 10) = round(3.7) = 4
    assert blended.expected_value == 4
    assert blended.estimated_effort == 4  # never modified


def test_blend_with_historical_scores_preserves_fallback_flag():
    judge = EnergyJudge()
    base = EnergyEvaluation(
        estimated_effort=5,
        expected_value=5,
        used_fallback=True,
        fallback_reason="test_reason",
    )
    blended = judge.blend_with_historical_scores(base, [5, 5, 5, 5, 5])
    assert blended.used_fallback is True
    assert blended.fallback_reason == "test_reason"


def test_blend_with_historical_scores_empty_list():
    judge = EnergyJudge()
    base = EnergyEvaluation(estimated_effort=3, expected_value=7)
    blended = judge.blend_with_historical_scores(base, [])
    assert blended.expected_value == 7  # unchanged


def test_blend_with_historical_scores_clamps_to_1_10():
    judge = EnergyJudge()
    base = EnergyEvaluation(estimated_effort=1, expected_value=1)
    # All 5 outcomes are perfect 5/5 → scaled 10/10
    blended = judge.blend_with_historical_scores(base, [5, 5, 5, 5, 5])
    # historical_avg_scaled = 10
    # blended = round(0.7 * 10 + 0.3 * 1) = round(7.3) = 7
    assert 1 <= blended.expected_value <= 10
