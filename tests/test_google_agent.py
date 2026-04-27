import asyncio
import json
from pathlib import Path

import pytest

from src.agents.google_agent import GoogleAgent
from src.core.runtime_context import set_runtime_context
from src.memory.ledger_db import LedgerMemory


async def _seed_google_task(
    ledger: LedgerMemory,
    *,
    title: str,
    acceptance_criteria: str = "",
) -> tuple[int, int]:
    epic_id = await ledger.add_objective(tier="Epic", title="Google Epic")
    story_id = await ledger.add_objective(tier="Story", title="Google Story", parent_id=epic_id)
    task_id = await ledger.add_objective(
        tier="Task",
        title=title,
        parent_id=story_id,
        acceptance_criteria=acceptance_criteria,
    )
    await ledger.set_task_agent_domain(task_id, "google")
    return task_id, story_id


@pytest.mark.asyncio
async def test_google_agent_send_email_task_returns_structured_payload() -> None:
    agent = GoogleAgent()
    calls = []

    async def _fake_execute(tool_name: str, arguments: dict | None = None) -> str:
        calls.append((tool_name, dict(arguments or {})))
        return json.dumps({"status": "success", "message": "ok"})

    agent.execute_tool = _fake_execute  # type: ignore[assignment]

    task = {
        "id": 1,
        "title": "send_email: outbound",
        "acceptance_criteria": json.dumps(
            {
                "recipient": "person@example.com",
                "subject": "Status",
                "body": "Done",
            }
        ),
    }
    result = await agent.execute_task(task)

    assert result["action_taken"] == "send_email"
    assert result["recipient"] == "person@example.com"
    assert result["subject"] == "Status"
    assert result["attachment_paths"] == []
    assert result["error"] == ""
    assert calls and calls[0][0] == "send_email"


@pytest.mark.asyncio
async def test_google_agent_email_poll_creates_google_objectives(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "google_poll.db"))
    await ledger.initialize()
    try:
        set_runtime_context(ledger, None, None, None)
        task_id, _story_id = await _seed_google_task(ledger, title="email_poll: refresh")
        task_row = await ledger.get_task_row(task_id)
        assert task_row is not None

        agent = GoogleAgent()

        async def _fake_execute(tool_name: str, arguments: dict | None = None) -> str:
            assert tool_name == "read_inbox"
            return json.dumps(
                {
                    "status": "success",
                    "emails": [
                        {"sender": "a@example.com", "subject": "Need update", "body": "Please reply"},
                        {"sender": "b@example.com", "subject": "Invoice", "body": "See attached"},
                    ],
                }
            )

        agent.execute_tool = _fake_execute  # type: ignore[assignment]
        result = await agent.execute_task(task_row)

        assert result["action_taken"] == "email_poll"
        assert result["error"] == ""
        assert result["objectives_created"] == 2

        pending_google = await ledger.get_pending_tasks_for_domain("google", limit=10)
        poll_generated = [row for row in pending_google if str(row.get("title", "")).startswith("email_request:")]
        assert len(poll_generated) == 2
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()


@pytest.mark.asyncio
async def test_google_agent_email_request_spawns_aiden_child_and_sends_reply(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "google_request.db"))
    await ledger.initialize()
    try:
        set_runtime_context(ledger, None, None, None)
        task_id, story_id = await _seed_google_task(
            ledger,
            title="email_request: handle inbound",
            acceptance_criteria=json.dumps({"sender": "requester@example.com", "subject": "Need summary"}),
        )
        task_row = await ledger.get_task_row(task_id)
        assert task_row is not None

        agent = GoogleAgent(child_poll_interval_seconds=0.05, child_timeout_seconds=2.0)
        send_email_calls: list[dict] = []

        async def _fake_execute(tool_name: str, arguments: dict | None = None) -> str:
            if tool_name == "read_inbox":
                return json.dumps(
                    {
                        "status": "success",
                        "emails": [
                            {
                                "sender": "requester@example.com",
                                "subject": "Need summary",
                                "body": "Please summarize our notes.",
                                "attachment_paths": [],
                            }
                        ],
                    }
                )
            if tool_name == "send_email":
                send_email_calls.append(dict(arguments or {}))
                return json.dumps({"status": "success"})
            if tool_name == "extract_pdf_text":
                return json.dumps({"status": "success", "text": "doc"})
            raise AssertionError(f"Unexpected tool call: {tool_name}")

        agent.execute_tool = _fake_execute  # type: ignore[assignment]

        async def _complete_child_task() -> None:
            deadline = asyncio.get_running_loop().time() + agent.child_timeout_seconds
            while asyncio.get_running_loop().time() <= deadline:
                cursor = await ledger._db.execute(
                    "SELECT id FROM objective_backlog WHERE tier = 'Task' AND agent_domain = 'aiden' AND parent_id = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (story_id,),
                )
                row = await cursor.fetchone()
                if row:
                    await ledger.write_task_result(int(row["id"]), {"final_answer": "Here is the completed response."})
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("Child task was not created")

        completer = asyncio.create_task(_complete_child_task())
        result = await agent.execute_task(task_row)
        await completer

        assert result["action_taken"] == "email_request"
        assert result["recipient"] == "requester@example.com"
        assert result["subject"] == "Need summary"
        assert result["error"] == ""
        assert send_email_calls
        assert "Here is the completed response." in str(send_email_calls[0].get("body") or "")
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()


@pytest.mark.asyncio
async def test_google_agent_poll_and_execute_marks_blocked_on_unrecoverable_failure(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "google_blocked.db"))
    await ledger.initialize()
    try:
        set_runtime_context(ledger, None, None, None)
        task_id, _story_id = await _seed_google_task(ledger, title="email_request: fail path")

        agent = GoogleAgent()

        async def _failing_execute(tool_name: str, arguments: dict | None = None) -> str:
            _ = arguments
            if tool_name == "read_inbox":
                return json.dumps({"status": "error", "message": "inbox unavailable"})
            raise AssertionError(f"Unexpected tool: {tool_name}")

        agent.execute_tool = _failing_execute  # type: ignore[assignment]
        worked = await agent.poll_and_execute()
        assert worked is True

        row = await ledger.get_task_row(task_id)
        assert row is not None
        assert str(row["status"]) == "blocked"
        payload = json.loads(str(row["result_json"]))
        assert payload["action_taken"] == "failed"
        assert payload["recipient"] == ""
        assert payload["subject"] == ""
        assert payload["attachment_paths"] == []
        assert payload["error"]
    finally:
        set_runtime_context(None, None, None, None)
        await ledger.close()
