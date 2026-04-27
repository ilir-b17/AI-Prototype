import asyncio
import json
from pathlib import Path

import pytest

from src.agents.base_domain_agent import BaseAgent
from src.core.runtime_context import set_runtime_context
from src.memory.ledger_db import LedgerMemory


class _SuccessDomainAgent(BaseAgent):
    allowed_tool_names = ["web_search"]
    agent_domain = "google"
    own_energy_budget = 2

    async def execute_task(self, task: dict) -> dict:
        return {"task_id": task["id"], "status": "ok"}


class _FailingDomainAgent(BaseAgent):
    allowed_tool_names = ["web_search"]
    agent_domain = "finance"
    own_energy_budget = 2

    async def execute_task(self, task: dict) -> dict:
        raise RuntimeError(f"unable to execute task #{task['id']}")


async def _seed_domain_task(ledger: LedgerMemory, *, domain: str, title: str) -> int:
    epic_id = await ledger.add_objective(tier="Epic", title=f"{domain} epic")
    story_id = await ledger.add_objective(tier="Story", title=f"{domain} story", parent_id=epic_id)
    task_id = await ledger.add_objective(tier="Task", title=title, parent_id=story_id)
    await ledger._db.execute(
        "UPDATE objective_backlog SET agent_domain = ? WHERE id = ?",
        (domain, task_id),
    )
    await ledger._db.commit()
    return task_id


@pytest.mark.asyncio
async def test_base_agent_scopes_registry_to_allowed_tools() -> None:
    agent = _SuccessDomainAgent()
    assert set(agent.skill_registry.get_skill_names()) == {"web_search"}


@pytest.mark.asyncio
async def test_poll_and_execute_completes_task_and_writes_result(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "domain_agent_success.db"))
    await ledger.initialize()

    try:
        task_id = await _seed_domain_task(ledger, domain="google", title="Collect context")
        set_runtime_context(ledger, None, None, None)

        agent = _SuccessDomainAgent(own_energy_budget=1)
        worked = await agent.poll_and_execute()

        assert worked is True
        row_cursor = await ledger._db.execute(
            "SELECT status, result_json FROM objective_backlog WHERE id = ?",
            (task_id,),
        )
        row = await row_cursor.fetchone()

        assert str(row["status"]) == "completed"
        payload = json.loads(str(row["result_json"]))
        assert payload["task_id"] == task_id
        assert payload["status"] == "ok"
        assert agent.energy_budget_remaining == 0
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()


@pytest.mark.asyncio
async def test_poll_and_execute_marks_failed_and_persists_error_payload(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "domain_agent_failure.db"))
    await ledger.initialize()

    try:
        task_id = await _seed_domain_task(ledger, domain="finance", title="Compute report")
        set_runtime_context(ledger, None, None, None)

        agent = _FailingDomainAgent(own_energy_budget=1)
        worked = await agent.poll_and_execute()

        assert worked is True
        row_cursor = await ledger._db.execute(
            "SELECT status, result_json, result_written_at FROM objective_backlog WHERE id = ?",
            (task_id,),
        )
        row = await row_cursor.fetchone()

        assert str(row["status"]) == "failed"
        assert row["result_written_at"] is not None
        payload = json.loads(str(row["result_json"]))
        assert payload["status"] == "failed"
        assert "unable to execute task" in payload["error"]
        assert payload["error_type"] == "RuntimeError"
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()


@pytest.mark.asyncio
async def test_run_can_start_as_background_task_and_stop(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "domain_agent_run.db"))
    await ledger.initialize()

    try:
        task_id = await _seed_domain_task(ledger, domain="google", title="Background loop task")
        set_runtime_context(ledger, None, None, None)

        agent = _SuccessDomainAgent(
            own_energy_budget=1,
            poll_interval_seconds=0.05,
            energy_replenish_interval_seconds=0.05,
        )
        runner = asyncio.create_task(agent.run())

        for _ in range(20):
            row_cursor = await ledger._db.execute(
                "SELECT status FROM objective_backlog WHERE id = ?",
                (task_id,),
            )
            row = await row_cursor.fetchone()
            if str(row["status"]) == "completed":
                break
            await asyncio.sleep(0.05)

        agent.stop()
        await asyncio.wait_for(runner, timeout=2)

        row_cursor = await ledger._db.execute(
            "SELECT status FROM objective_backlog WHERE id = ?",
            (task_id,),
        )
        row = await row_cursor.fetchone()
        assert str(row["status"]) == "completed"
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()
