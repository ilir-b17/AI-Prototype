"""Heartbeat management mixin for Orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from src.core.orchestrator_constants import (
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TASK_PREFIX_FMT,
    _HEARTBEAT_FAILURE_STRIKES,
    _HEARTBEAT_FAILURE_STATE_KEY,
    _PENDING_STATE_TTL_SECONDS,
    _CONSOLIDATION_TRIGGER_TURNS,
    _CONSOLIDATION_TURN_COUNT_MAX_USERS,
    _HEARTBEAT_REPLENISH_ENV_DEFAULT,
)
from src.core.energy_roi_engine import EnergyDecision
from src.core.outcome_judge import OutcomeJudge

logger = logging.getLogger(__name__)


class _HeartbeatManagerMixin:
    """Heartbeat loop and pending-state management. Mixed into Orchestrator."""

    @staticmethod
    def _heartbeat_result_indicates_failure(result: str) -> bool:
        lowered = str(result or "").lower()
        return any(
            marker in lowered
            for marker in ("error", "cannot", "unable", "failed", "i cannot", "i am unable")
        )

    @staticmethod
    def _build_heartbeat_execution_prompt(task: Dict[str, Any]) -> str:
        prefix = HEARTBEAT_TASK_PREFIX_FMT.format(task_id=task["id"])
        return (
            f"{prefix}: {task['title']}\n"
            f"You MUST execute this task right now by calling the appropriate tools. "
            f"Do NOT describe what you plan to do — use your tools and report the actual results. "
            f"If the task requires storing data, call update_core_memory or update_ledger explicitly. "
            f"If you cannot complete it, explain exactly why."
        )

    @staticmethod
    def _is_executable_heartbeat_task_candidate(
        node: Dict[str, Any],
        unresolved_task_ids: set[int],
        parent_ids: set[int],
    ) -> bool:
        if str(node.get("tier") or "") != "Task":
            return False
        if str(node.get("status") or "").lower() not in {"pending", "deferred_due_to_energy"}:
            return False

        task_id = int(node.get("id") or 0)
        if task_id <= 0:
            return False
        return task_id not in unresolved_task_ids and task_id not in parent_ids

    async def _select_executable_heartbeat_tasks(self) -> List[Dict[str, Any]]:
        """Return executable Task candidates with Story/Epic context for energy scoring."""
        candidates = await self.ledger_memory.get_energy_evaluation_candidates(
            statuses=["pending", "deferred_due_to_energy"],
        )
        if not candidates:
            return []

        active_nodes = await self.ledger_memory.get_active_objective_tree()
        unresolved_rows = await self.ledger_memory.get_tasks_with_unresolved_dependencies(
            statuses=["pending", "active", "deferred_due_to_energy"],
        )
        unresolved_task_ids = {int(row["id"]) for row in unresolved_rows}
        parent_ids = {
            int(node["parent_id"])
            for node in active_nodes
            if node.get("parent_id") is not None
        }

        executable: List[Dict[str, Any]] = []
        for candidate in candidates:
            task = dict(candidate.get("task") or {})
            if self._is_executable_heartbeat_task_candidate(task, unresolved_task_ids, parent_ids):
                executable.append(candidate)

        executable.sort(
            key=lambda candidate: (
                int((candidate.get("task") or {}).get("priority") or 999),
                int((candidate.get("task") or {}).get("estimated_energy") or 999999),
                int((candidate.get("task") or {}).get("id") or 0),
            )
        )
        return executable

    async def _select_executable_heartbeat_task(self) -> Optional[Dict[str, Any]]:
        """Pick the next executable Task leaf with all dependencies resolved."""
        candidates = await self._select_executable_heartbeat_tasks()
        if not candidates:
            return None
        return dict(candidates[0].get("task") or {})

    async def _handle_heartbeat_task_failure(
        self,
        task: Dict[str, Any],
        *,
        reason: str,
        result_excerpt: str,
    ) -> None:
        task_id = int(task["id"])
        strike_count = await self._increment_heartbeat_failure_count(task_id)

        if strike_count >= _HEARTBEAT_FAILURE_STRIKES:
            await self.ledger_memory.update_objective_status(task_id, "blocked")
            await self.ledger_memory.ensure_parent_chain_active(task_id)
            await self._notify_admin(
                "HITL REQUIRED: Task blocked after repeated heartbeat failures.\n"
                f"  Task #{task_id}: {task['title']}\n"
                f"  Strikes: {strike_count}\n"
                f"  Reason: {reason}\n"
                f"  Latest result: {result_excerpt[:220]}"
            )
            logger.warning(
                "Heartbeat: Task #%s blocked after %s failures (%s)",
                task_id,
                strike_count,
                reason,
            )
            return

        await self.ledger_memory.update_objective_status(task_id, "pending")
        logger.warning(
            "Heartbeat: Task #%s failure strike %s/%s (%s)",
            task_id,
            strike_count,
            _HEARTBEAT_FAILURE_STRIKES,
            reason,
        )

    async def _defer_heartbeat_task_due_to_energy(
        self,
        *,
        task_id: int,
        decision: EnergyDecision,
        evaluation_record: Dict[str, Any],
        available_energy: int,
        race: bool = False,
    ) -> None:
        await self.ledger_memory.defer_task_due_to_energy(
            task_id,
            evaluation_record,
            cooldown_seconds=int(decision.defer_cooldown_seconds),
        )
        logger.info(
            "Heartbeat: Deferred task #%s%s (%s). roi=%.2f cost=%s available=%s reserve_floor=%s",
            task_id,
            " after budget race" if race else "",
            decision.reason,
            decision.roi,
            decision.predicted_cost,
            available_energy,
            decision.min_reserve,
        )

    async def _execute_heartbeat_task(
        self,
        *,
        task: Dict[str, Any],
        decision: EnergyDecision,
    ) -> None:
        task_id = int(task["id"])
        logger.info(
            "Heartbeat: Accepting executable task #%s: %s (roi=%.2f cost=%s budget=%s)",
            task_id,
            str(task.get("title") or "")[:60],
            decision.roi,
            decision.predicted_cost,
            await self._get_predictive_energy_budget_remaining(),
        )
        await self.ledger_memory.update_objective_status(task_id, "active")

        timeout_seconds = HEARTBEAT_INTERVAL * 0.9
        prompt = self._build_heartbeat_execution_prompt(task)

        try:
            result = await asyncio.wait_for(
                self.process_message(user_message=prompt, user_id="heartbeat"),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._handle_heartbeat_task_failure(
                task,
                reason="timeout",
                result_excerpt=f"Timed out after {int(timeout_seconds)}s",
            )
            return
        except Exception as e:
            await self._handle_heartbeat_task_failure(
                task,
                reason="exception",
                result_excerpt=str(e),
            )
            return

        if self._has_pending_heartbeat_hitl(task_id) or self._has_pending_heartbeat_mfa(task_id):
            logger.info("Heartbeat: Task #%s paused awaiting admin guidance.", task_id)
            return

        if self._heartbeat_result_indicates_failure(result):
            await self._handle_heartbeat_task_failure(
                task,
                reason="execution_failure",
                result_excerpt=result,
            )
            return

        await self.ledger_memory.update_objective_status(task_id, "completed")
        await self._clear_heartbeat_failure_count(task_id)

        # Fire-and-forget: score the completed task asynchronously so the
        # heartbeat cycle is not delayed by outcome judgment.
        self._fire_and_forget(
            OutcomeJudge.record_outcome(self.ledger_memory, task_id)
        )

        summary = (
            f"Heartbeat completed task #{task_id}:\n"
            f"  Task: {task.get('title', '')}\n"
            f"  Predicted Energy Cost: {decision.predicted_cost}\n"
            f"  Remaining Predictive Budget: {await self._get_predictive_energy_budget_remaining()}\n"
            f"  Result: {str(result)[:220]}"
        )
        await self._notify_admin(summary)
        logger.info("Heartbeat: Task #%s completed.", task_id)

    async def _run_heartbeat_cycle(self) -> None:
        await self._tick_predictive_energy_budget(
            amount=self._resolve_energy_replenish_per_heartbeat(),
            reason="heartbeat_cycle",
        )

        logger.info("Heartbeat: Querying objective backlog for executable Tasks...")
        candidate_contexts = await self._select_executable_heartbeat_tasks()

        if not candidate_contexts:
            # Try to report when the next deferred task becomes eligible.
            try:
                cursor = await self.ledger_memory._db.execute(
                    "SELECT MIN(next_eligible_at) as earliest "
                    "FROM objective_backlog "
                    "WHERE tier='Task' AND status='deferred_due_to_energy' "
                    "AND next_eligible_at IS NOT NULL"
                )
                row = await cursor.fetchone()
                earliest = row["earliest"] if row else None
            except Exception:
                earliest = None

            if earliest:
                logger.info(
                    "Heartbeat: No executable tasks available. "
                    "Next deferred task eligible at: %s",
                    earliest,
                )
            else:
                logger.info("Heartbeat: No executable dependency-resolved tasks available.")
            return

        for candidate in candidate_contexts:
            task = dict(candidate.get("task") or {})
            story = candidate.get("story")
            epic = candidate.get("epic")

            task_id = int(task.get("id") or 0)
            if task_id <= 0:
                continue

            evaluation, decision, available_energy = await self._evaluate_energy_for_context(
                task=task,
                story=story,
                epic=epic,
                additional_context="heartbeat_dispatch",
            )
            evaluation_record = self._build_energy_evaluation_record(
                evaluation=evaluation,
                decision=decision,
                available_energy=available_energy,
                context="heartbeat",
            )

            if not decision.should_execute:
                await self._defer_heartbeat_task_due_to_energy(
                    task_id=task_id,
                    decision=decision,
                    evaluation_record=evaluation_record,
                    available_energy=available_energy,
                )
                continue

            reserved = await self._try_reserve_predictive_energy_budget(
                predicted_cost=decision.predicted_cost,
                min_reserve=decision.min_reserve,
                reason=f"heartbeat_task:{task_id}",
            )
            if not reserved:
                current_budget = await self._get_predictive_energy_budget_remaining()
                race_decision = self.energy_roi_engine.evaluate(
                    estimated_effort=evaluation.estimated_effort,
                    expected_value=evaluation.expected_value,
                    available_energy=current_budget,
                    defer_count=decision.defer_count,
                )
                race_record = self._build_energy_evaluation_record(
                    evaluation=evaluation,
                    decision=race_decision,
                    available_energy=current_budget,
                    context="heartbeat",
                )
                await self._defer_heartbeat_task_due_to_energy(
                    task_id=task_id,
                    decision=race_decision,
                    evaluation_record=race_record,
                    available_energy=current_budget,
                    race=True,
                )
                continue

            await self.ledger_memory.record_task_energy_evaluation(
                task_id,
                evaluation_record,
                clear_next_eligible=True,
            )
            await self._execute_heartbeat_task(task=task, decision=decision)
            return

        logger.info("Heartbeat: All executable tasks were deferred by energy policy.")

    async def _heartbeat_loop(self) -> None:
        """
        Proactive Heartbeat: wakes every 30 min, selects executable Objective
        leaf tasks (dependency-resolved), executes one, and applies remediation
        logic on repeated failures.

        An asyncio.Lock prevents heartbeat cycles from overlapping: if the
        previous run is still in progress when the next interval fires, the
        new cycle is skipped (ISSUE-004).
        """
        logger.info("Heartbeat loop started.")
        _heartbeat_lock = asyncio.Lock()
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if _heartbeat_lock.locked():
                logger.warning("Heartbeat: Previous run still in progress. Skipping this cycle.")
                continue
            async with _heartbeat_lock:
                try:
                    await self._run_heartbeat_cycle()
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}", exc_info=True)

    @staticmethod
    def _is_pending_entry_expired(
        entry: Any,
        *,
        created_at_key: str,
        ttl_seconds: int,
        now_epoch: float,
    ) -> bool:
        payload = entry if isinstance(entry, dict) else {}
        try:
            created_at = float(payload.get(created_at_key, 0) or 0)
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at <= 0:
            return True
        return (now_epoch - created_at) > float(ttl_seconds)

    def _purge_expired_pending_in_memory(
        self,
        *,
        attr_name: str,
        created_at_key: str,
        ttl_seconds: int,
        now_epoch: float,
    ) -> int:
        pending_store = getattr(self, attr_name, None)
        if not isinstance(pending_store, dict):
            return 0

        expired_users = [
            user_id
            for user_id, payload in pending_store.items()
            if self._is_pending_entry_expired(
                payload,
                created_at_key=created_at_key,
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            )
        ]
        for user_id in expired_users:
            pending_store.pop(user_id, None)
        if expired_users:
            logger.warning(
                "Swept %d expired in-memory pending entries from %s.",
                len(expired_users),
                attr_name,
            )
        return len(expired_users)

    async def _sweep_expired_pending_state(self) -> None:
        ttl_seconds = self._resolve_positive_int_env(
            "PENDING_STATE_TTL_SECONDS",
            _PENDING_STATE_TTL_SECONDS,
        )

        db_deleted: Dict[str, int] = {}
        purge_pending = getattr(self.ledger_memory, "purge_expired_pending", None)
        if callable(purge_pending):
            try:
                db_deleted = await purge_pending(ttl_seconds=ttl_seconds)
            except Exception as e:
                logger.warning("Pending-state DB sweep failed: %s", e)

        now_epoch = time.time()
        mem_deleted = {
            "pending_tool_approval": self._purge_expired_pending_in_memory(
                attr_name="pending_tool_approval",
                created_at_key="_created_at",
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            ),
            "pending_hitl_state": self._purge_expired_pending_in_memory(
                attr_name="pending_hitl_state",
                created_at_key="_hitl_created_at",
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            ),
            "pending_mfa": self._purge_expired_pending_in_memory(
                attr_name="pending_mfa",
                created_at_key="_created_at",
                ttl_seconds=ttl_seconds,
                now_epoch=now_epoch,
            ),
        }

        if any(int(value or 0) > 0 for value in db_deleted.values()) or any(mem_deleted.values()):
            logger.info(
                "Pending-state sweep complete. DB deleted=%s in-memory deleted=%s",
                db_deleted,
                mem_deleted,
            )

    async def _restore_heartbeat_failure_counts(self) -> None:
        """Restore heartbeat task-failure strike counts from persisted system_state."""
        try:
            raw_value = await self.ledger_memory.get_system_state(_HEARTBEAT_FAILURE_STATE_KEY)
            if not raw_value:
                self._heartbeat_failure_counts = {}
                return

            decoded = json.loads(raw_value)
            if not isinstance(decoded, dict):
                self._heartbeat_failure_counts = {}
                return

            restored: Dict[int, int] = {}
            for raw_task_id, raw_count in decoded.items():
                try:
                    task_id = int(raw_task_id)
                    count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                if task_id > 0 and count > 0:
                    restored[task_id] = count
            self._heartbeat_failure_counts = restored
        except Exception as e:
            logger.warning("Failed to restore heartbeat failure counts: %s", e)
            self._heartbeat_failure_counts = {}

    def _coerce_consolidation_turn_counts(self) -> "OrderedDict[str, int]":
        counts = getattr(self, "_consolidation_turn_counts", None)
        if isinstance(counts, OrderedDict):
            return counts
        if isinstance(counts, dict):
            ordered = OrderedDict(counts.items())
        else:
            ordered = OrderedDict()
        self._consolidation_turn_counts = ordered
        self._evict_ordered_mapping_to_size(
            ordered,
            _CONSOLIDATION_TURN_COUNT_MAX_USERS,
            label="Consolidation turn counts",
        )
        return ordered

    async def _restore_consolidation_turn_counts(self) -> None:
        prefix = "consolidation_turn_count:"
        restored: "OrderedDict[str, int]" = OrderedDict()
        if not hasattr(self, "_consolidation_turn_counts"):
            self._consolidation_turn_counts = OrderedDict()

        getter = getattr(self.ledger_memory, "get_system_state_keys_by_prefix", None)
        if not callable(getter):
            self._consolidation_turn_counts = OrderedDict()
            return

        try:
            rows = await getter(prefix)
            for key, value in rows:
                key_text = str(key or "")
                if not key_text.startswith(prefix):
                    continue

                user_id = key_text[len(prefix):].strip()
                if not user_id:
                    continue

                try:
                    count = int(str(value).strip())
                except (TypeError, ValueError):
                    continue

                restored[user_id] = max(0, count)

            self._consolidation_turn_counts = restored
            bounded_counts = self._coerce_consolidation_turn_counts()
            if bounded_counts:
                logger.info("Restored %d consolidation turn counter(s) from DB", len(bounded_counts))
        except Exception as e:
            logger.warning("Failed to restore consolidation turn counts: %s", e)
            self._consolidation_turn_counts = OrderedDict()

    async def _persist_heartbeat_failure_counts(self) -> None:
        payload = {
            str(task_id): int(count)
            for task_id, count in self._heartbeat_failure_counts.items()
            if int(count) > 0
        }
        await self.ledger_memory.set_system_state(
            _HEARTBEAT_FAILURE_STATE_KEY,
            json.dumps(payload, sort_keys=True),
        )

    async def _increment_heartbeat_failure_count(self, task_id: int) -> int:
        current = int(self._heartbeat_failure_counts.get(task_id, 0)) + 1
        self._heartbeat_failure_counts[task_id] = current
        await self._persist_heartbeat_failure_counts()
        return current

    async def _clear_heartbeat_failure_count(self, task_id: int) -> None:
        if task_id in self._heartbeat_failure_counts:
            self._heartbeat_failure_counts.pop(task_id, None)
            await self._persist_heartbeat_failure_counts()

    def _persist_consolidation_turn_count_async(self, user_id: str, count: int) -> None:
        ledger = getattr(self, "ledger_memory", None)
        scheduler = getattr(self, "_fire_and_forget", None)
        if ledger is None or not callable(scheduler):
            return

        setter = getattr(ledger, "set_system_state", None)
        if not callable(setter):
            return

        key = f"consolidation_turn_count:{user_id}"
        try:
            persistence_coro = setter(key, str(max(0, int(count))))
        except Exception as e:
            logger.warning("Failed to schedule consolidation turn count persistence for %s: %s", user_id, e)
            return

        if asyncio.iscoroutine(persistence_coro):
            scheduler(persistence_coro)

    @staticmethod
    def _extract_heartbeat_task_id(user_input: str) -> Optional[int]:
        prefix_pattern = re.escape(HEARTBEAT_TASK_PREFIX_FMT).replace(r"\{task_id\}", r"(\d+)")
        match = re.search(prefix_pattern, str(user_input or ""))
        if not match:
            return None
        try:
            task_id = int(match.group(1))
        except (TypeError, ValueError):
            return None
        return task_id if task_id > 0 else None

    @staticmethod
    def _coerce_positive_task_id(raw_task_id: Any) -> Optional[int]:
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            return None
        return task_id if task_id > 0 else None

    def _get_admin_user_id(self) -> str:
        admin_user_id = str(getattr(self, "_admin_user_id", "") or "").strip()
        if not admin_user_id:
            admin_user_id = str(os.getenv("ADMIN_USER_ID", "") or "").strip()
        if not admin_user_id:
            admin_user_id = "heartbeat"
        self._admin_user_id = admin_user_id
        return admin_user_id

    def _extract_heartbeat_origin_task_id_from_state(self, state: Dict[str, Any]) -> Optional[int]:
        existing_task_id = self._coerce_positive_task_id(state.get("_heartbeat_origin_task_id"))
        if existing_task_id is not None:
            state["_heartbeat_origin_task_id"] = existing_task_id
            return existing_task_id

        if str(state.get("user_id") or "") != "heartbeat":
            return None

        parsed_task_id = self._extract_heartbeat_task_id(str(state.get("user_input") or ""))
        if parsed_task_id is not None:
            state["_heartbeat_origin_task_id"] = parsed_task_id
        return parsed_task_id

    def _pending_owner_for_heartbeat_origin(self, user_id: str, state: Dict[str, Any]) -> str:
        heartbeat_task_id = self._extract_heartbeat_origin_task_id_from_state(state)
        if heartbeat_task_id is None:
            return user_id
        return self._get_admin_user_id()

    def _has_pending_heartbeat_hitl(self, task_id: int) -> bool:
        pending_hitl_state = getattr(self, "pending_hitl_state", {}) or {}
        for pending_state in pending_hitl_state.values():
            if self._coerce_positive_task_id(pending_state.get("_heartbeat_origin_task_id")) == task_id:
                return True
        return False

    def _has_pending_heartbeat_mfa(self, task_id: int) -> bool:
        pending_mfa = getattr(self, "pending_mfa", {}) or {}
        for pending_state in pending_mfa.values():
            if self._coerce_positive_task_id(pending_state.get("_heartbeat_origin_task_id")) == task_id:
                return True
        return False

    async def _finalize_resumed_heartbeat_task(self, task_id: int, response: str) -> None:
        if self._heartbeat_result_indicates_failure(response):
            await self._handle_heartbeat_task_failure(
                {
                    "id": task_id,
                    "title": f"Task #{task_id}",
                },
                reason="post_hitl_resume_failure",
                result_excerpt=response,
            )
            return

        await self.ledger_memory.update_objective_status(task_id, "completed")
        await self._clear_heartbeat_failure_count(task_id)
        logger.info("Heartbeat: Task #%s completed after HITL resume.", task_id)

    async def _suspend_task_for_moral_halt(self, state: Dict[str, Any]) -> Optional[int]:
        task_id = self._extract_heartbeat_task_id(str(state.get("user_input") or ""))
        if task_id is None:
            return None

        ledger = getattr(self, "ledger_memory", None)
        if ledger is None:
            return None

        try:
            await ledger.update_objective_status(task_id, "suspended")
            return task_id
        except Exception as e:
            logger.warning("Could not suspend heartbeat task %s after moral halt: %s", task_id, e)
            return None
