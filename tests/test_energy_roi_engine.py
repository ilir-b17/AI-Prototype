import math
import pytest

from src.core.energy_roi_engine import (
    ENERGY_MIN_RESERVE,
    ENERGY_ROI_THRESHOLD,
    FAIRNESS_BOOST_MULTIPLIER,
    EnergyPolicy,
    EnergyROIEngine,
)


def test_zero_division_guard_uses_max_denominator_one():
    engine = EnergyROIEngine(EnergyPolicy(roi_threshold=1.0, min_reserve=0, effort_multiplier=3, defer_cooldown_seconds=60))

    decision = engine.evaluate(
        estimated_effort=0,
        expected_value=7,
        available_energy=100,
    )

    assert math.isclose(decision.roi, 7.0, rel_tol=1e-12, abs_tol=1e-12)
    assert decision.predicted_cost == 0


def test_predicted_cost_uses_effort_multiplier_mapping():
    engine = EnergyROIEngine(EnergyPolicy(roi_threshold=0.1, min_reserve=0, effort_multiplier=4, defer_cooldown_seconds=60))

    decision = engine.evaluate(
        estimated_effort=6,
        expected_value=6,
        available_energy=100,
    )

    assert decision.predicted_cost == 24


def test_defers_when_roi_is_below_threshold():
    engine = EnergyROIEngine(EnergyPolicy(roi_threshold=2.0, min_reserve=0, effort_multiplier=3, defer_cooldown_seconds=60))

    decision = engine.evaluate(
        estimated_effort=6,
        expected_value=6,
        available_energy=100,
    )

    assert decision.should_execute is False
    assert decision.reason == "ROI too low"


def test_defers_high_roi_when_predicted_cost_violates_min_reserve():
    engine = EnergyROIEngine(EnergyPolicy(roi_threshold=1.0, min_reserve=12, effort_multiplier=3, defer_cooldown_seconds=90))

    # ROI is high (9/2=4.5), but predicted_cost is 6 and leaves reserve 10 (< 12).
    decision = engine.evaluate(
        estimated_effort=2,
        expected_value=9,
        available_energy=16,
    )

    assert decision.should_execute is False
    assert decision.reason == "Insufficient reserve"
    assert math.isclose(decision.roi, 4.5, rel_tol=1e-12, abs_tol=1e-12)
    assert decision.predicted_cost == 6
    assert decision.reserve_after_execution == 10


def test_approves_when_roi_and_reserve_conditions_pass():
    engine = EnergyROIEngine(EnergyPolicy(roi_threshold=1.1, min_reserve=10, effort_multiplier=3, defer_cooldown_seconds=300))

    decision = engine.evaluate(
        estimated_effort=4,
        expected_value=8,
        available_energy=30,
    )

    assert decision.should_execute is True
    assert decision.reason == "Approved"
    assert decision.predicted_cost == 12
    assert decision.reserve_after_execution == 18
    assert decision.defer_cooldown_seconds == 300


def test_fairness_boost_eventually_approves_aging_low_roi_task():
    engine = EnergyROIEngine(
        EnergyPolicy(
            roi_threshold=1.8,
            min_reserve=0,
            effort_multiplier=1,
            defer_cooldown_seconds=60,
            fairness_boost_multiplier=0.25,
            max_defer_count=99,
        )
    )

    decisions = [
        engine.evaluate(
            estimated_effort=8,
            expected_value=8,
            available_energy=100,
            defer_count=defer_count,
        )
        for defer_count in range(0, 8)
    ]

    assert decisions[0].should_execute is False
    assert decisions[0].reason == "ROI too low"
    assert math.isclose(decisions[0].base_roi, 1.0, rel_tol=1e-12, abs_tol=1e-12)

    approved = next((decision for decision in decisions if decision.should_execute), None)
    assert approved is not None
    assert approved.defer_count > 0
    assert approved.effective_roi >= approved.roi_threshold


def test_max_defer_count_bypass_approves_when_reserve_is_sufficient():
    engine = EnergyROIEngine(
        EnergyPolicy(
            roi_threshold=3.0,
            min_reserve=2,
            effort_multiplier=3,
            defer_cooldown_seconds=120,
            fairness_boost_multiplier=0.0,
            max_defer_count=5,
        )
    )

    decision = engine.evaluate(
        estimated_effort=2,
        expected_value=1,
        available_energy=10,
        defer_count=5,
    )

    assert decision.should_execute is True
    assert decision.reason == "Approved (max defer bypass)"
    assert decision.base_roi < decision.roi_threshold
    assert decision.reserve_after_execution >= decision.min_reserve


def test_max_defer_count_bypass_still_respects_reserve_floor():
    engine = EnergyROIEngine(
        EnergyPolicy(
            roi_threshold=3.0,
            min_reserve=2,
            effort_multiplier=3,
            defer_cooldown_seconds=120,
            fairness_boost_multiplier=0.0,
            max_defer_count=5,
        )
    )

    decision = engine.evaluate(
        estimated_effort=2,
        expected_value=1,
        available_energy=7,
        defer_count=5,
    )

    assert decision.should_execute is False
    assert decision.reason == "Insufficient reserve"


def test_default_threshold_constants_are_sane_for_phase5_calibration():
    # Calibration guardrails to prevent accidental drift to unsafe defaults.
    assert 1.0 <= float(ENERGY_ROI_THRESHOLD) <= 2.0
    assert 0 <= int(ENERGY_MIN_RESERVE) <= 25
    assert 0.0 <= float(FAIRNESS_BOOST_MULTIPLIER) <= 0.5

    default_policy = EnergyPolicy()
    assert default_policy.roi_threshold == ENERGY_ROI_THRESHOLD
    assert default_policy.min_reserve == ENERGY_MIN_RESERVE
    assert default_policy.fairness_boost_multiplier == FAIRNESS_BOOST_MULTIPLIER
