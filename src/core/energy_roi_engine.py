from __future__ import annotations

import os
from dataclasses import dataclass


ENERGY_ROI_THRESHOLD = float(os.getenv("ENERGY_ROI_THRESHOLD", "1.25"))
ENERGY_MIN_RESERVE = int(os.getenv("ENERGY_MIN_RESERVE", "10"))
ENERGY_MULTIPLIER = int(os.getenv("ENERGY_MULTIPLIER", "3"))
ENERGY_DEFER_COOLDOWN_SECONDS = int(os.getenv("ENERGY_DEFER_COOLDOWN_SECONDS", "1800"))
FAIRNESS_BOOST_MULTIPLIER = float(os.getenv("FAIRNESS_BOOST_MULTIPLIER", "0.15"))
MAX_DEFER_COUNT = int(os.getenv("MAX_DEFER_COUNT", "5"))


@dataclass(frozen=True)
class EnergyPolicy:
    roi_threshold: float = ENERGY_ROI_THRESHOLD
    min_reserve: int = ENERGY_MIN_RESERVE
    effort_multiplier: int = ENERGY_MULTIPLIER
    defer_cooldown_seconds: int = ENERGY_DEFER_COOLDOWN_SECONDS
    fairness_boost_multiplier: float = FAIRNESS_BOOST_MULTIPLIER
    max_defer_count: int = MAX_DEFER_COUNT


@dataclass(frozen=True)
class EnergyDecision:
    should_execute: bool
    reason: str
    roi: float
    base_roi: float
    effective_roi: float
    predicted_cost: int
    estimated_effort: int
    expected_value: int
    defer_count: int
    available_energy: int
    reserve_after_execution: int
    roi_threshold: float
    min_reserve: int
    fairness_boost_multiplier: float
    max_defer_count: int
    defer_cooldown_seconds: int


class EnergyROIEngine:
    """Deterministic cost-benefit gate computed entirely in Python."""

    def __init__(self, policy: EnergyPolicy | None = None) -> None:
        self.policy = policy or EnergyPolicy()

    @staticmethod
    def _coerce_non_negative_int(value: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, parsed)

    @staticmethod
    def _coerce_non_negative_float(value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, parsed)

    def evaluate(
        self,
        *,
        estimated_effort: int,
        expected_value: int,
        available_energy: int,
        defer_count: int = 0,
    ) -> EnergyDecision:
        effort_score = self._coerce_non_negative_int(estimated_effort)
        value_score = self._coerce_non_negative_int(expected_value)
        available = self._coerce_non_negative_int(available_energy)
        current_defer_count = self._coerce_non_negative_int(defer_count)

        denominator = max(1, effort_score)
        base_roi = float(value_score) / float(denominator)
        fairness_boost_multiplier = self._coerce_non_negative_float(self.policy.fairness_boost_multiplier)
        effective_roi = base_roi * (1.0 + (float(current_defer_count) * fairness_boost_multiplier))
        predicted_cost = effort_score * max(0, int(self.policy.effort_multiplier))
        reserve_after = available - predicted_cost
        roi_threshold = float(self.policy.roi_threshold)
        min_reserve = int(self.policy.min_reserve)
        max_defer_count = max(1, int(self.policy.max_defer_count))
        has_reserve = reserve_after >= min_reserve

        if current_defer_count >= max_defer_count:
            if has_reserve:
                reason = "Approved (max defer bypass)"
                should_execute = True
            else:
                reason = "Insufficient reserve"
                should_execute = False
        elif effective_roi < roi_threshold:
            reason = "ROI too low"
            should_execute = False
        elif not has_reserve:
            reason = "Insufficient reserve"
            should_execute = False
        else:
            reason = "Approved"
            should_execute = True

        return EnergyDecision(
            should_execute=should_execute,
            reason=reason,
            roi=effective_roi,
            base_roi=base_roi,
            effective_roi=effective_roi,
            predicted_cost=predicted_cost,
            estimated_effort=effort_score,
            expected_value=value_score,
            defer_count=current_defer_count,
            available_energy=available,
            reserve_after_execution=reserve_after,
            roi_threshold=roi_threshold,
            min_reserve=min_reserve,
            fairness_boost_multiplier=fairness_boost_multiplier,
            max_defer_count=max_defer_count,
            defer_cooldown_seconds=int(self.policy.defer_cooldown_seconds),
        )
