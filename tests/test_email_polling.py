import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_run_email_poll_cycle_creates_google_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.add_objective = AsyncMock(side_effect=[101, 102])
    orchestrator.ledger_memory.set_task_agent_domain = AsyncMock()
    orchestrator._email_processed_timestamps = []
    orchestrator._email_poll_last_run_at = None

    payload = {
        "status": "success",
        "emails": [
            {
                "sender": "alice@example.com",
                "subject": "Budget update",
                "body": "Please review attached budget.",
                "attachment_paths": ["/tmp/budget.xlsx"],
                "message_id": "<msg-1>",
                "date": "Mon, 01 Jan 2026 10:00:00 +0000",
            },
            {
                "sender": "bob@example.com",
                "subject": "Follow up",
                "body": "Any update?",
                "attachment_paths": [],
                "message_id": "<msg-2>",
                "date": "Mon, 01 Jan 2026 10:05:00 +0000",
            },
        ],
    }
    monkeypatch.setattr(
        "src.core.orchestrator.read_inbox",
        AsyncMock(return_value=json.dumps(payload)),
    )

    await Orchestrator._run_email_poll_cycle(orchestrator)

    assert orchestrator.ledger_memory.add_objective.await_count == 2
    first_call = orchestrator.ledger_memory.add_objective.await_args_list[0].kwargs
    assert first_call["tier"] == "Task"
    assert first_call["origin"] == "email"
    assert first_call["estimated_energy"] == 20
    assert first_call["title"] == "Process email request: Budget update from alice@example.com"
    acceptance = json.loads(first_call["acceptance_criteria"])
    assert acceptance["body"] == "Please review attached budget."
    assert acceptance["attachment_paths"] == ["/tmp/budget.xlsx"]
    assert acceptance["sender"] == "alice@example.com"
    assert acceptance["subject"] == "Budget update"

    assert orchestrator.ledger_memory.set_task_agent_domain.await_count == 2
    assert orchestrator.ledger_memory.set_task_agent_domain.await_args_list[0].args == (101, "google")
    assert orchestrator.ledger_memory.set_task_agent_domain.await_args_list[1].args == (102, "google")
    assert orchestrator._email_poll_last_run_at is not None
    assert len(orchestrator._email_processed_timestamps) == 2


@pytest.mark.asyncio
async def test_run_email_poll_cycle_warns_and_continues_on_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.add_objective = AsyncMock()
    orchestrator.ledger_memory.set_task_agent_domain = AsyncMock()
    orchestrator._email_processed_timestamps = []
    orchestrator._email_poll_last_run_at = None

    monkeypatch.setattr(
        "src.core.orchestrator.read_inbox",
        AsyncMock(side_effect=RuntimeError("imap unavailable")),
    )

    await Orchestrator._run_email_poll_cycle(orchestrator)

    orchestrator.ledger_memory.add_objective.assert_not_awaited()
    orchestrator.ledger_memory.set_task_agent_domain.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_email_poll_status_reports_24h_and_pending_counts() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.get_pending_tasks_for_domain = AsyncMock(return_value=[{"id": 1}, {"id": 2}, {"id": 3}])
    orchestrator._email_poll_last_run_at = datetime(2026, 4, 27, 12, 30, 0)
    now = time.time()
    orchestrator._email_processed_timestamps = [now - 100.0, now - 90_000.0]

    stats = await Orchestrator.get_email_poll_status(orchestrator)

    assert stats["last_poll_time"] == "2026-04-27 12:30:00"
    assert stats["emails_processed_last_24h"] == 1
    assert stats["pending_google_tasks"] == 3
