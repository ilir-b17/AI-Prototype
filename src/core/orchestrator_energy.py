"""Energy management mixin for Orchestrator."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.energy_judge import EnergyEvaluation
from src.core.energy_roi_engine import EnergyDecision
from src.core.llm_router import RequiresHITLError
from src.core.orchestrator_constants import (
    _AD_HOC_COMFORTABLE_BUDGET_FLOOR,
    _HEARTBEAT_REPLENISH_ENV_DEFAULT,
)

logger = logging.getLogger(__name__)


class _EnergyManagerMixin:
    """Energy operation methods. Mixed into Orchestrator."""

    _deprecated_energy_replenish_turn_warning_logged = False

    @classmethod
    def _warn_deprecated_energy_replenish_turn_env_once(cls) -> None:
        """Emit a one-time warning for deprecated turn-based replenish config."""
        if cls._deprecated_energy_replenish_turn_warning_logged:
            return

        raw_deprecated = str(os.getenv("ENERGY_REPLENISH_PER_TURN", "") or "").strip()
        if not raw_deprecated:
            return

        logger.warning(
            "ENERGY_REPLENISH_PER_TURN is deprecated and ignored. "
            "Use ENERGY_REPLENISH_PER_HOUR for wall-clock replenishment instead."
        )
        cls._deprecated_energy_replenish_turn_warning_logged = True

    @staticmethod
    def _resolve_energy_replenish_per_hour() -> float:
        raw_rate = str(os.getenv("ENERGY_REPLENISH_PER_HOUR", "30") or "").strip()
        try:
            rate = float(raw_rate)
        except ValueError:
            logger.warning(
                "Invalid ENERGY_REPLENISH_PER_HOUR=%r. Falling back to 30.",
                raw_rate,
            )
            rate = 30.0
        return max(0.0, rate)

    @staticmethod
    def _resolve_energy_replenish_per_heartbeat() -> int:
        raw_value = str(
            os.getenv("ENERGY_REPLENISH_PER_HEARTBEAT", str(_HEARTBEAT_REPLENISH_ENV_DEFAULT)) or ""
        ).strip()
        try:
            amount = int(raw_value)
        except ValueError:
            logger.warning(
                "Invalid ENERGY_REPLENISH_PER_HEARTBEAT=%r. Falling back to %s.",
                raw_value,
                _HEARTBEAT_REPLENISH_ENV_DEFAULT,
            )
            amount = _HEARTBEAT_REPLENISH_ENV_DEFAULT
        return max(0, amount)

    async def _tick_predictive_energy_budget(self, amount: int, reason: str) -> int:
        if amount <= 0:
            return 0

        before = await self._get_predictive_energy_budget_remaining()
        await self._refund_predictive_energy_budget(int(amount), reason)
        after = await self._get_predictive_energy_budget_remaining()
        applied = max(0, after - before)
        return applied

    async def _get_predictive_energy_budget_remaining(self) -> int:
        async with self._energy_budget_lock:
            return max(0, int(self._energy_budget))

    async def _refund_predictive_energy_budget(self, amount: int, reason: str) -> None:
        if amount <= 0:
            return

        cap = max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
        async with self._energy_budget_lock:
            self._energy_budget = min(cap, self._energy_budget + int(amount))
        logger.info(
            "Energy budget refunded: +%s (%s). Remaining=%s",
            amount, reason, self._energy_budget,
        )

    async def _try_reserve_predictive_energy_budget(
        self,
        *,
        predicted_cost: int,
        min_reserve: int,
        reason: str,
    ) -> bool:
        cost = max(0, int(predicted_cost))
        reserve = max(0, int(min_reserve))
        async with self._energy_budget_lock:
            if self._energy_budget - cost < reserve:
                return False
            self._energy_budget -= cost
        logger.info(
            "Energy budget reserved: -%s (%s). Remaining=%s",
            cost, reason, self._energy_budget,
        )
        return True

    @staticmethod
    def _build_energy_evaluation_record(
        *,
        evaluation: EnergyEvaluation,
        decision: EnergyDecision,
        available_energy: int,
        context: str,
    ) -> Dict[str, Any]:
        return {
            "context": str(context or ""),
            "available_energy": int(available_energy),
            "estimated_effort": int(evaluation.estimated_effort),
            "expected_value": int(evaluation.expected_value),
            "defer_count": int(decision.defer_count),
            "used_fallback": bool(evaluation.used_fallback),
            "fallback_reason": str(evaluation.fallback_reason or ""),
            "should_execute": bool(decision.should_execute),
            "reason": str(decision.reason or ""),
            "roi": float(decision.roi),
            "base_roi": float(decision.base_roi),
            "effective_roi": float(decision.effective_roi),
            "predicted_cost": int(decision.predicted_cost),
            "reserve_after_execution": int(decision.reserve_after_execution),
            "roi_threshold": float(decision.roi_threshold),
            "min_reserve": int(decision.min_reserve),
            "fairness_boost_multiplier": float(decision.fairness_boost_multiplier),
            "max_defer_count": int(decision.max_defer_count),
            "defer_cooldown_seconds": int(decision.defer_cooldown_seconds),
            "evaluated_at": datetime.now().isoformat(),
        }

    async def _evaluate_energy_for_context(
        self,
        *,
        task: Dict[str, Any],
        story: Optional[Dict[str, Any]],
        epic: Optional[Dict[str, Any]],
        additional_context: str,
    ) -> tuple[EnergyEvaluation, EnergyDecision, int]:
        available_energy = await self._get_predictive_energy_budget_remaining()
        defer_count = int(task.get("defer_count") or 0)
        evaluation = await self.energy_judge.evaluate_with_system1(
            task=task,
            story=story,
            epic=epic,
            route_to_system_1=self._route_energy_judge_messages,
            additional_context=additional_context,
        )

        # Blend with historical outcome scores when available (same tier + title overlap)
        ledger = getattr(self, "ledger_memory", None)
        if ledger is not None and str(task.get("tier") or "Task") == "Task":
            try:
                title_tokens = str(task.get("title") or "").split()
                historical_scores = await ledger.get_historical_outcome_scores(
                    "Task",
                    title_tokens,
                    limit=100,
                )
                evaluation = self.energy_judge.blend_with_historical_scores(
                    evaluation,
                    historical_scores,
                )
            except Exception as _blend_exc:
                logger.warning("EnergyJudge: historical blend failed: %s", _blend_exc)

        decision = self.energy_roi_engine.evaluate(
            estimated_effort=evaluation.estimated_effort,
            expected_value=evaluation.expected_value,
            available_energy=available_energy,
            defer_count=defer_count,
        )
        return evaluation, decision, available_energy

    def _build_synthesized_ad_hoc_energy_context(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        user_message = str(state.get("user_input") or "").strip()
        complexity = max(1, self.routing_assessor._estimate_request_complexity(user_message) + 1)
        history_context = self._build_goal_planning_context(state)

        task = {
            "title": user_message,
            "acceptance_criteria": "Provide a correct and concise answer to the current user request.",
            "estimated_energy": min(10, complexity),
            "status": "pending",
            "depends_on_ids": [],
            "defer_count": 0,
        }
        story = {
            "title": "Ad-hoc user response",
            "status": "active",
            "acceptance_criteria": "Answer the user request with useful and policy-compliant output.",
        }
        epic = {
            "title": "Sustain reliable operator assistance",
            "status": "active",
            "acceptance_criteria": "Serve user requests while preserving energy reserves for critical work.",
        }
        return {
            "task": task,
            "story": story,
            "epic": epic,
            "additional_context": history_context,
        }

    @staticmethod
    def _format_ad_hoc_energy_deferral_message(
        *,
        decision: EnergyDecision,
        available_energy: int,
    ) -> str:
        return (
            "Deferred due to energy policy. "
            f"Reason: {decision.reason}. "
            f"ROI={decision.roi:.2f} (base={decision.base_roi:.2f}, defer_count={decision.defer_count}), "
            f"predicted_cost={decision.predicted_cost}, "
            f"available_budget={int(available_energy)}, reserve_floor={decision.min_reserve}."
        )

    async def _try_ad_hoc_dispatch_energy_gate(
        self,
        state: Dict[str, Any],
        *,
        dispatch_context: str,
    ) -> Optional[str]:
        user_id = str(state.get("user_id") or "")
        if user_id == "heartbeat":
            return None

        user_message = str(state.get("user_input") or "").strip()
        if not user_message:
            return None

        context = self._build_synthesized_ad_hoc_energy_context(state)
        evaluation, decision, available_energy = await self._evaluate_energy_for_context(
            task=context["task"] or {},
            story=context["story"],
            epic=context["epic"],
            additional_context=str(context.get("additional_context") or ""),
        )

        if not decision.should_execute:
            if (
                decision.reason == "ROI too low"
                and available_energy > _AD_HOC_COMFORTABLE_BUDGET_FLOOR
            ):
                logger.info(
                    "Ad-hoc dispatch approved by comfort-budget floor (%s): "
                    "ROI=%.2f below threshold but budget=%s > floor=%s",
                    dispatch_context,
                    decision.roi,
                    available_energy,
                    _AD_HOC_COMFORTABLE_BUDGET_FLOOR,
                )
            else:
                message = self._format_ad_hoc_energy_deferral_message(
                    decision=decision,
                    available_energy=available_energy,
                )
                logger.warning(
                    "Ad-hoc dispatch deferred by energy gate (%s): %s",
                    dispatch_context,
                    message,
                )
                return message

        reserved = await self._try_reserve_predictive_energy_budget(
            predicted_cost=decision.predicted_cost,
            min_reserve=decision.min_reserve,
            reason=f"ad_hoc:{dispatch_context}:{user_id}",
        )
        if not reserved:
            current_budget = await self._get_predictive_energy_budget_remaining()
            race_decision = self.energy_roi_engine.evaluate(
                estimated_effort=evaluation.estimated_effort,
                expected_value=evaluation.expected_value,
                available_energy=current_budget,
                defer_count=decision.defer_count,
            )
            message = self._format_ad_hoc_energy_deferral_message(
                decision=race_decision,
                available_energy=current_budget,
            )
            logger.warning(
                "Ad-hoc dispatch deferred after budget reservation race (%s): %s",
                dispatch_context,
                message,
            )
            return message

        logger.info(
            "Ad-hoc dispatch approved by energy gate (%s): roi=%.2f cost=%s remaining=%s",
            dispatch_context,
            decision.roi,
            decision.predicted_cost,
            await self._get_predictive_energy_budget_remaining(),
        )
        return None

    def _deduct_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        state["energy_remaining"] = state.get("energy_remaining", 100) - amount
        logger.debug("Energy -%d (%s). Remaining: %d", amount, reason, state["energy_remaining"])
        if state["energy_remaining"] <= 0:
            raise RequiresHITLError(
                "Energy Budget Exhausted: The system consumed all cognitive energy on this task.\n"
                f"Question: How should I prioritize the remaining work for: '{state.get('user_input', '')}'"
            )
        return state

    def _refund_energy(self, state: Dict[str, Any], amount: int, reason: str) -> Dict[str, Any]:
        """Refund energy deducted by a failed or timed-out operation."""
        state["energy_remaining"] = state.get("energy_remaining", 0) + amount
        logger.debug("Energy +%d refunded (%s). Remaining: %d", amount, reason, state["energy_remaining"])
        return state
