import pytest

from src.core.llm_router import CognitiveRouter
from src.core.orchestrator import Orchestrator


def test_pytest_ast_gate_rejects_malicious_script() -> None:
    malicious_pytest_code = """
import pytest
from unittest.mock import patch
import os


def test_malicious_script():
    assert os.system('echo hacked') == 0
""".strip()

    with pytest.raises(ValueError, match="blocked module 'os'"):
        CognitiveRouter._validate_pytest_code_ast(malicious_pytest_code, "safe_tool")


@pytest.mark.asyncio
async def test_self_test_harness_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHESIS_SELF_TEST_TIMEOUT_SECONDS", "1")

    orchestrator = Orchestrator.__new__(Orchestrator)
    synthesis = {
        "tool_name": "loop_tool",
        "code": (
            "import asyncio\n\n"
            "async def loop_tool() -> str:\n"
            "    while True:\n"
            "        await asyncio.sleep(0)\n"
        ),
        "pytest_code": (
            "import pytest\n"
            "from loop_tool import loop_tool\n\n"
            "@pytest.mark.asyncio\n"
            "async def test_infinite_loop_times_out():\n"
            "    await loop_tool()\n"
        ),
    }

    result = await orchestrator._run_synthesis_self_test(synthesis)

    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert result["exit_code"] is not None
    assert "timed out" in result["error"].lower()
