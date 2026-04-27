import json
import asyncio
from typing import Any, Dict

import pytest

from src.core.llm_router import RouterResult
from src.core.orchestrator import Orchestrator


class _Router:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> RouterResult:
        await asyncio.sleep(0)
        self.calls.append((tool_name, dict(arguments)))
        return RouterResult(
            status="ok",
            content=json.dumps({"status": "success", "message": "Email sent to ilirb17@gmail.com"}),
        )


@pytest.mark.asyncio
async def test_confirm_send_email_draft_executes_previous_assistant_tool_code() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    router = _Router()
    orchestrator.cognitive_router = router
    chat_history = [
        {
            "role": "assistant",
            "content": (
                "I will now send the email.\n\n"
                "<tool_code>\n"
                "send_email(to=\"ilirb17@gmail.com\", subject=\"Inter result\", "
                "body=\"The final score of the match was a 2-2 draw.\")\n"
                "</tool_code>"
            ),
        }
    ]

    response = await orchestrator._try_confirm_send_email_draft("please do", chat_history)

    assert response == "Email sent to ilirb17@gmail.com"
    assert router.calls == [
        (
            "send_email",
            {
                "recipient": "ilirb17@gmail.com",
                "subject": "Inter result",
                "body": "The final score of the match was a 2-2 draw.",
            },
        )
    ]


@pytest.mark.asyncio
async def test_confirm_send_email_draft_ignores_unrelated_confirmation() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    router = _Router()
    orchestrator.cognitive_router = router

    response = await orchestrator._try_confirm_send_email_draft(
        "please do",
        [{"role": "assistant", "content": "I searched for that already."}],
    )

    assert response is None
    assert router.calls == []