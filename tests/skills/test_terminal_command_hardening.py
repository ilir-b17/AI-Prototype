import json

import pytest

from src.skills.run_terminal_command import run_terminal_command


class _DummyProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_terminal_command_blocks_semicolon_bypass() -> None:
    result = await run_terminal_command("ls; rm -rf /tmp/foo")
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Command blocked by security policy"


@pytest.mark.asyncio
async def test_terminal_command_blocks_command_substitution_bypass() -> None:
    result = await run_terminal_command("$(rm test.txt)")
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Command blocked by security policy"


@pytest.mark.asyncio
async def test_terminal_command_enforces_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIDEN_TERMINAL_ALLOWLIST", "hostname")

    result = await run_terminal_command("git status")
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Command blocked by security policy"


@pytest.mark.asyncio
async def test_terminal_command_runs_allowlisted_command_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    async def _fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _DummyProcess(stdout=b"host123\n")

    monkeypatch.setenv("AIDEN_TERMINAL_ALLOWLIST", "hostname")
    monkeypatch.setattr(
        "src.skills.run_terminal_command.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
    )

    result = await run_terminal_command("hostname")

    assert "--- STDOUT ---" in result
    assert "host123" in result
    assert captured["args"][0] == "hostname"
