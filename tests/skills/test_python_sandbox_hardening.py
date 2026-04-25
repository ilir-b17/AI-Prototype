import json

import pytest

from src.skills.execute_python_sandbox import execute_python_sandbox


@pytest.mark.asyncio
async def test_python_sandbox_blocks_os_system_escape() -> None:
    result = await execute_python_sandbox("import os\nos.system('echo pwn')")
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Sandbox execution blocked"


@pytest.mark.asyncio
async def test_python_sandbox_blocks_write_outside_worker_tmp(tmp_path) -> None:
    outside_file = tmp_path / "outside.txt"
    code = (
        f"with open({str(outside_file)!r}, 'w', encoding='utf-8') as handle:\n"
        "    handle.write('pwn')\n"
    )

    result = await execute_python_sandbox(code)
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Sandbox execution blocked"
    assert not outside_file.exists()


@pytest.mark.asyncio
async def test_python_sandbox_allows_tmp_writes_inside_worker() -> None:
    code = (
        "with open('/tmp/inside.txt', 'w', encoding='utf-8') as handle:\n"
        "    handle.write('ok')\n"
        "with open('/tmp/inside.txt', 'r', encoding='utf-8') as handle:\n"
        "    print(handle.read())\n"
    )

    result = await execute_python_sandbox(code)

    assert "--- STDOUT ---" in result
    assert "ok" in result
