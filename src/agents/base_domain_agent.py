from __future__ import annotations

import abc
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.core.runtime_context import get_ledger
from src.core.skill_manager import SkillRegistry
from src.memory.ledger_db import LedgerMemory


logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Base class for domain execution agents that operate via the blackboard."""

    allowed_tool_names: List[str] = []
    agent_domain: str = ""
    own_energy_budget: int = 1

    def __init__(
        self,
        *,
        agent_name: Optional[str] = None,
        allowed_tool_names: Optional[List[str]] = None,
        agent_domain: Optional[str] = None,
        own_energy_budget: Optional[int] = None,
        poll_interval_seconds: float = 2.0,
        energy_replenish_interval_seconds: float = 30.0,
    ) -> None:
        resolved_domain = str(agent_domain if agent_domain is not None else self.agent_domain).strip()
        if not resolved_domain:
            raise ValueError("agent_domain must be provided")

        resolved_budget = int(
            own_energy_budget if own_energy_budget is not None else self.own_energy_budget
        )
        if resolved_budget <= 0:
            raise ValueError("own_energy_budget must be > 0")

        declared_tools = (
            allowed_tool_names if allowed_tool_names is not None else self.allowed_tool_names
        )
        clean_tool_names: List[str] = []
        for name in declared_tools or []:
            clean_name = str(name).strip()
            if clean_name:
                clean_tool_names.append(clean_name)
        self.allowed_tool_names = list(dict.fromkeys(clean_tool_names))
        self.agent_domain = resolved_domain
        self.agent_name = str(agent_name or f"{self.__class__.__name__}:{self.agent_domain}")

        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.energy_replenish_interval_seconds = max(0.05, float(energy_replenish_interval_seconds))

        self._max_energy_budget = resolved_budget
        self._energy_budget = resolved_budget
        self._last_energy_replenish_at = time.monotonic()

        self._stop_event = asyncio.Event()
        self._ledger: Optional[LedgerMemory] = None

        self.skill_registry = SkillRegistry()
        self._scope_registry_to_allowed_tools()

    @property
    def energy_budget_remaining(self) -> int:
        return int(self._energy_budget)

    def stop(self) -> None:
        self._stop_event.set()

    async def _resolve_ledger(self) -> LedgerMemory:
        if self._ledger is not None:
            return self._ledger
        ledger = get_ledger()
        if ledger is None:
            raise RuntimeError("LedgerMemory is unavailable in runtime_context")
        self._ledger = ledger
        return ledger

    def _scope_registry_to_allowed_tools(self) -> None:
        allowed = set(self.allowed_tool_names)
        available = set(self.skill_registry.get_skill_names())
        self.skill_registry.restrict_to(self.allowed_tool_names)

        missing = sorted(name for name in allowed if name not in available)
        if missing:
            logger.warning(
                "[%s] %s: declared tools not found in SkillRegistry: %s",
                self.agent_domain,
                self.agent_name,
                ", ".join(missing),
            )

    def _replenish_energy_if_due(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_energy_replenish_at
        if elapsed < self.energy_replenish_interval_seconds:
            return

        ticks = int(elapsed // self.energy_replenish_interval_seconds)
        self._energy_budget = min(
            self._max_energy_budget,
            max(0, self._energy_budget) + ticks,
        )
        self._last_energy_replenish_at += ticks * self.energy_replenish_interval_seconds

    def _try_consume_energy(self, amount: int = 1) -> bool:
        self._replenish_energy_if_due()
        if self._energy_budget < amount:
            return False
        self._energy_budget -= amount
        return True

    def _refund_energy(self, amount: int = 1) -> None:
        if amount <= 0:
            return
        self._energy_budget = min(self._max_energy_budget, self._energy_budget + amount)

    async def execute_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Execute an allowed tool through this agent's scoped SkillRegistry."""
        if tool_name not in self.allowed_tool_names:
            return f"Error: Tool '{tool_name}' is not allowed for {self.agent_name}."
        return await self.skill_registry.execute(tool_name, arguments or {})

    async def _write_failed_result(self, ledger: LedgerMemory, task_id: int, payload: Dict[str, Any]) -> None:
        """Persist a task failure payload to the shared blackboard."""
        await ledger.write_task_failure(task_id, payload)

    async def poll_and_execute(self) -> bool:
        """Poll one domain task, claim it, execute it, and persist the result.

        Returns:
            True when a task was claimed/handled, False when no eligible work ran.
        """
        ledger = await self._resolve_ledger()
        pending = await ledger.get_pending_tasks_for_domain(self.agent_domain, limit=1)
        if not pending:
            return False

        if not self._try_consume_energy(1):
            logger.info(
                "[%s] %s: energy depleted (remaining=%s)",
                self.agent_domain,
                self.agent_name,
                self._energy_budget,
            )
            return False

        task = dict(pending[0])
        task_id = int(task.get("id") or 0)
        if task_id <= 0:
            self._refund_energy(1)
            return False

        claimed = await ledger.claim_task(task_id, self.agent_name)
        if not claimed:
            self._refund_energy(1)
            return False

        logger.info("[%s] %s: claimed task #%s", self.agent_domain, self.agent_name, task_id)

        try:
            result = await self.execute_task(task)
            if not isinstance(result, dict):
                result = {"result": result}
            await ledger.write_task_result(task_id, result)
            logger.info("[%s] %s: completed task #%s", self.agent_domain, self.agent_name, task_id)
            return True
        except Exception as exc:
            error_payload = {
                "status": "failed",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }
            try:
                await self._write_failed_result(ledger, task_id, error_payload)
            except Exception:
                logger.exception(
                    "[%s] %s: failed to persist failure payload for task #%s",
                    self.agent_domain,
                    self.agent_name,
                    task_id,
                )
            logger.exception(
                "[%s] %s: task #%s failed during execute_task",
                self.agent_domain,
                self.agent_name,
                task_id,
            )
            return True

    async def run(self) -> None:
        """Run the continuous poll loop until ``stop()`` is called."""
        logger.info("[%s] %s: domain agent loop started", self.agent_domain, self.agent_name)
        while not self._stop_event.is_set():
            try:
                await self.poll_and_execute()
            except Exception:
                logger.exception(
                    "[%s] %s: unexpected poll loop error",
                    self.agent_domain,
                    self.agent_name,
                )

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                continue

        logger.info("[%s] %s: domain agent loop stopped", self.agent_domain, self.agent_name)

    @abc.abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one claimed task and return a JSON-serializable result payload."""
