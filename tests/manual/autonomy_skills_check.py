import asyncio
import json as _json
import os

from src.skills.manage_file_system import manage_file_system
from src.skills.execute_python_sandbox import execute_python_sandbox
from src.skills.run_terminal_command import run_terminal_command

async def test_manage_file_system():
    print("--- Testing manage_file_system ---")
    # Test 'list'
    list_res = await manage_file_system('list', '.')
    assert "src" in list_res, f"Expected 'src' in directory listing, got: {list_res}"
    print("[PASS] manage_file_system 'list'")

    # Test 'read'
    read_res = await manage_file_system('read', 'src/skills/manage_file_system/SKILL.md')
    assert "manage_file_system" in read_res, f"Expected content in read file, got: {read_res}"
    print("[PASS] manage_file_system 'read'")

    # Test 'write' bypass
    write_res = await manage_file_system('write', 'test.txt', 'data')
    assert "SECURITY LOCK:" in write_res, f"Expected security lock on write, got: {write_res}"
    print("[PASS] manage_file_system 'write' lock")

    # Test 'delete' bypass
    delete_res = await manage_file_system('delete', 'test.txt')
    assert "SECURITY LOCK:" in delete_res, f"Expected security lock on delete, got: {delete_res}"
    print("[PASS] manage_file_system 'delete' lock")


async def test_execute_python_sandbox():
    print("\n--- Testing execute_python_sandbox ---")
    # Test valid execution
    code = "print('Hello from sandbox')"
    res = await execute_python_sandbox(code)
    assert "Hello from sandbox" in res, f"Expected output from python code, got: {res}"
    print("[PASS] execute_python_sandbox valid code")

    # Test timeout
    timeout_code = "import time\ntime.sleep(12)\nprint('Done')"
    timeout_res = await execute_python_sandbox(timeout_code)
    assert "timeout" in timeout_res.lower(), f"Expected timeout error, got: {timeout_res}"
    print("[PASS] execute_python_sandbox timeout")


async def test_run_terminal_command():
    print("\n--- Testing run_terminal_command ---")
    # Test valid command
    res = await run_terminal_command("echo 'Hello from terminal'")
    assert "Hello from terminal" in res, f"Expected echo output, got: {res}"
    print("[PASS] run_terminal_command valid command")

    # Test blacklist
    rm_res = await run_terminal_command("rm -rf /tmp/foo")
    rm_data = _json.loads(rm_res)
    assert rm_data.get("status") == "error", f"Expected error status, got: {rm_res}"
    assert "blocked" in rm_data.get("message", "").lower(), f"Expected command to be blocked, got: {rm_res}"
    print("[PASS] run_terminal_command blacklist (rm)")

    sudo_rm_res = await run_terminal_command("sudo rm -rf /tmp/foo")
    sudo_data = _json.loads(sudo_rm_res)
    assert sudo_data.get("status") == "error", f"Expected error status, got: {sudo_rm_res}"
    assert "blocked" in sudo_data.get("message", "").lower(), f"Expected command to be blocked, got: {sudo_rm_res}"
    print("[PASS] run_terminal_command blacklist (sudo rm)")

    # Test timeout — use a cross-platform Python one-liner so it works on both Windows and Linux
    timeout_res = await run_terminal_command('python -c "import time; time.sleep(12)"')
    assert "timeout" in timeout_res.lower(), f"Expected timeout error, got: {timeout_res}"
    print("[PASS] run_terminal_command timeout")

async def main():
    await test_manage_file_system()
    await test_execute_python_sandbox()
    await test_run_terminal_command()
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
