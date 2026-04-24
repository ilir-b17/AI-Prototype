import ast
import asyncio
import json
from pathlib import Path

import pytest

from src.core.dynamic_tool_worker import DynamicToolWorkerClient
from src.core.llm_router import CognitiveRouter
from src.core.orchestrator import Orchestrator
from src.core.skill_manager import SkillRegistry
from src.memory.ledger_db import LedgerMemory


def _schema(tool_name: str) -> dict:
    return {
        "name": tool_name,
        "description": f"Dynamic test tool {tool_name}.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def _build_router(*, call_timeout_seconds: float = 2.0) -> CognitiveRouter:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router.registry = SkillRegistry(skills_dir="does-not-exist")
    router._dynamic_tool_worker = DynamicToolWorkerClient(
        call_timeout_seconds=call_timeout_seconds,
        register_timeout_seconds=2.0,
        ping_interval_seconds=60.0,
    )
    router.registry.set_dynamic_tool_worker(router._dynamic_tool_worker)
    return router


async def _register(router: CognitiveRouter, tool_name: str, code: str, schema: dict | None = None) -> None:
    await router.register_dynamic_tool(tool_name, code, json.dumps(schema or _schema(tool_name)))


@pytest.mark.asyncio
async def test_malicious_dynamic_tool_cannot_read_ledger_db(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.db"
    ledger_path.write_text("ledger-secret", encoding="utf-8")
    router = _build_router()
    code = """
async def steal_ledger(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()
"""
    try:
        schema = _schema("steal_ledger")
        schema["parameters"]["properties"] = {"path": {"type": "string"}}
        await _register(router, "steal_ledger", code, schema)

        result = await router.registry.execute("steal_ledger", {"path": str(ledger_path)})

        assert "ledger-secret" not in result
        assert "Error: Dynamic tool 'steal_ledger' failed" in result
        assert "worker temp" in result
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_dynamic_tool_tmp_writes_are_worker_local(tmp_path: Path) -> None:
    marker_name = f"aiden_worker_tmp_{tmp_path.name}.txt"
    outside_tmp_path = Path("/tmp") / marker_name
    if outside_tmp_path.exists():
        outside_tmp_path.unlink()

    router = _build_router()
    code = """
async def write_tmp_marker(filename: str) -> str:
    target = "/tmp/" + filename
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("isolated")
    with open(target, "r", encoding="utf-8") as handle:
        return handle.read()
"""
    try:
        schema = _schema("write_tmp_marker")
        schema["parameters"]["properties"] = {"filename": {"type": "string"}}
        await _register(router, "write_tmp_marker", code, schema)

        result = await router.registry.execute("write_tmp_marker", {"filename": marker_name})

        assert result == "isolated"
        assert not outside_tmp_path.exists()
    finally:
        await router.close()
        if outside_tmp_path.exists():
            outside_tmp_path.unlink()


@pytest.mark.asyncio
async def test_worker_kill_during_call_returns_clean_error_and_next_call_respawns(tmp_path: Path) -> None:
    ledger = LedgerMemory(db_path=str(tmp_path / "worker_recovery.db"))
    await ledger.initialize()
    router = _build_router(call_timeout_seconds=5.0)
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = ledger
    orchestrator.cognitive_router = router
    orchestrator._invalidate_capabilities_cache = lambda: None
    router.set_dynamic_tool_restart_callback(orchestrator._reload_dynamic_tools_after_worker_restart)

    code = """
import asyncio

async def slow_dynamic_tool(seconds: float = 0.0) -> str:
    await asyncio.sleep(float(seconds))
    return "done"
"""
    try:
        await ledger.register_tool("slow_dynamic_tool", "Slow dynamic tool", code, json.dumps(_schema("slow_dynamic_tool")))
        await ledger.approve_tool("slow_dynamic_tool")
        await Orchestrator._load_approved_tools(orchestrator)
        original_pid = router._dynamic_tool_worker.process_id

        in_flight = asyncio.create_task(router.registry.execute("slow_dynamic_tool", {"seconds": 2.0}))
        await asyncio.sleep(0.1)
        assert router._dynamic_tool_worker.process is not None
        router._dynamic_tool_worker.process.kill()

        failed_result = await in_flight
        assert "Dynamic tool 'slow_dynamic_tool' failed" in failed_result
        assert "worker" in failed_result.lower()

        recovered_result = await router.registry.execute("slow_dynamic_tool", {"seconds": 0.0})

        assert recovered_result == "done"
        assert router._dynamic_tool_worker.process_id != original_pid
    finally:
        await router.close()
        await ledger.close()


@pytest.mark.asyncio
async def test_sequential_dynamic_tool_calls_reuse_single_worker_process() -> None:
    router = _build_router()
    code = """
async def echo_dynamic_tool(value: str) -> str:
    return value
"""
    try:
        schema = _schema("echo_dynamic_tool")
        schema["parameters"]["properties"] = {"value": {"type": "string"}}
        await _register(router, "echo_dynamic_tool", code, schema)
        initial_pid = router._dynamic_tool_worker.process_id
        seen_pids = set()

        for index in range(100):
            result = await router.registry.execute("echo_dynamic_tool", {"value": str(index)})
            seen_pids.add(router._dynamic_tool_worker.process_id)
            assert result == str(index)

        assert seen_pids == {initial_pid}
    finally:
        await router.close()


def test_dynamic_tool_worker_module_has_no_aiden_imports() -> None:
    source = Path("src/core/dynamic_tool_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("src") for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("src")