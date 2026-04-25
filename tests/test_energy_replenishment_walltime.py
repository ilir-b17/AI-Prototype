import asyncio

import pytest

from src.core.orchestrator import Orchestrator


def _build_energy_orchestrator() -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._predictive_energy_budget_lock = asyncio.Lock()
    orchestrator._predictive_energy_budget_remaining = 50
    orchestrator._predictive_energy_budget_last_replenished_at = 1_000.0
    return orchestrator


def test_no_replenishment_when_no_time_has_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _build_energy_orchestrator()
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "100")
    monkeypatch.setenv("ENERGY_REPLENISH_PER_HOUR", "30")
    monkeypatch.setattr("src.core.orchestrator.time.time", lambda: 1_000.0)

    gained = Orchestrator._replenish_predictive_energy_budget_wallclock_locked(orchestrator)

    assert gained == 0
    assert orchestrator._predictive_energy_budget_remaining == 50


def test_replenishment_accumulates_correctly_by_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _build_energy_orchestrator()
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "100")
    monkeypatch.setenv("ENERGY_REPLENISH_PER_HOUR", "60")
    monkeypatch.setattr("src.core.orchestrator.time.time", lambda: 1_600.0)  # +600s => +10 points

    gained = Orchestrator._replenish_predictive_energy_budget_wallclock_locked(orchestrator)

    assert gained == 10
    assert orchestrator._predictive_energy_budget_remaining == 60
    assert orchestrator._predictive_energy_budget_last_replenished_at == pytest.approx(1_600.0)


def test_replenishment_respects_budget_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _build_energy_orchestrator()
    orchestrator._predictive_energy_budget_remaining = 99
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "100")
    monkeypatch.setenv("ENERGY_REPLENISH_PER_HOUR", "30")
    monkeypatch.setattr("src.core.orchestrator.time.time", lambda: 4_600.0)  # +3600s => +30 points

    gained = Orchestrator._replenish_predictive_energy_budget_wallclock_locked(orchestrator)

    assert gained == 1
    assert orchestrator._predictive_energy_budget_remaining == 100


def test_subpoint_accumulation_is_preserved_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = _build_energy_orchestrator()
    monkeypatch.setenv("INITIAL_ENERGY_BUDGET", "100")
    monkeypatch.setenv("ENERGY_REPLENISH_PER_HOUR", "60")

    monkeypatch.setattr("src.core.orchestrator.time.time", lambda: 1_030.0)  # +30s => +0.5 points
    first_gain = Orchestrator._replenish_predictive_energy_budget_wallclock_locked(orchestrator)
    assert first_gain == 0
    assert orchestrator._predictive_energy_budget_remaining == 50
    assert orchestrator._predictive_energy_budget_last_replenished_at == pytest.approx(1_000.0)

    monkeypatch.setattr("src.core.orchestrator.time.time", lambda: 1_060.0)  # another +30s => +1 total
    second_gain = Orchestrator._replenish_predictive_energy_budget_wallclock_locked(orchestrator)
    assert second_gain == 1
    assert orchestrator._predictive_energy_budget_remaining == 51
    assert orchestrator._predictive_energy_budget_last_replenished_at == pytest.approx(1_060.0)


def test_deprecated_energy_replenish_per_turn_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("ENERGY_REPLENISH_PER_TURN", "10")
    monkeypatch.setattr(Orchestrator, "_deprecated_energy_replenish_turn_warning_logged", False)

    with caplog.at_level("WARNING", logger="src.core.orchestrator"):
        Orchestrator._warn_deprecated_energy_replenish_turn_env_once()

    assert any(
        "ENERGY_REPLENISH_PER_TURN" in record.message and "ENERGY_REPLENISH_PER_HOUR" in record.message
        for record in caplog.records
    )
