import asyncio
import importlib
import logging
import logging.handlers
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.core.orchestrator as orchestrator_module
from src.core.nocturnal_consolidation import NocturnalConsolidationSlice1
from src.core.orchestrator import Orchestrator
from src.memory.ledger_db import LedgerMemory


async def _flush_scheduled(tasks: list[asyncio.Task]) -> None:
    if not tasks:
        return
    pending = list(tasks)
    tasks.clear()
    await asyncio.gather(*pending, return_exceptions=True)


def _build_counter_writer(ledger: LedgerMemory) -> tuple[Orchestrator, list[asyncio.Task]]:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = ledger
    orchestrator._consolidation_turn_counts = {}
    orchestrator._consolidate_memory = AsyncMock(return_value=None)

    scheduled: list[asyncio.Task] = []

    def _schedule(coro):
        task = asyncio.create_task(coro)
        scheduled.append(task)
        return task

    orchestrator._fire_and_forget = _schedule
    return orchestrator, scheduled


def _build_restore_orchestrator(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = LedgerMemory(db_path=str(db_path))
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.set_dynamic_tool_restart_callback = MagicMock()
    orchestrator.cognitive_router.start_dynamic_tool_worker = AsyncMock()
    orchestrator.cognitive_router._preload_system_1_on_startup = False
    orchestrator.core_memory = MagicMock()
    orchestrator.core_memory.update = AsyncMock()
    orchestrator.vector_memory = MagicMock()
    orchestrator._restore_persisted_groq_cooldown = AsyncMock()
    orchestrator._load_approved_tools = AsyncMock()
    orchestrator._load_pending_approvals = AsyncMock()
    orchestrator._load_pending_mfa = AsyncMock()
    orchestrator._load_pending_hitl = AsyncMock()
    orchestrator._restore_heartbeat_failure_counts = AsyncMock()
    orchestrator._enforce_charter_policy = MagicMock()
    orchestrator._fire_and_forget = lambda coro: asyncio.create_task(coro)
    orchestrator._consolidation_turn_counts = {}
    orchestrator._ready = asyncio.Event()

    monkeypatch.setattr(orchestrator_module, "build_orchestrator_graph", lambda _self: MagicMock(name="compiled_graph"))
    monkeypatch.setattr(orchestrator_module, "set_runtime_context", lambda *_args, **_kwargs: None)
    return orchestrator


def test_rotating_log_handler_uses_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_token")
    monkeypatch.setenv("ADMIN_USER_ID", "12345")
    monkeypatch.setenv("LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "7")

    existing_module = sys.modules.get("src.interfaces.telegram_bot")
    previous_admin_user_id = getattr(existing_module, "ADMIN_USER_ID", None) if existing_module else None
    previous_bot_token = getattr(existing_module, "TELEGRAM_BOT_TOKEN", None) if existing_module else None
    previous_native_audio = getattr(existing_module, "ENABLE_NATIVE_AUDIO", None) if existing_module else None

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    try:
        telegram_bot = importlib.import_module("src.interfaces.telegram_bot")
        importlib.reload(telegram_bot)

        rotating_handlers = [
            handler
            for handler in root_logger.handlers
            if isinstance(handler, logging.handlers.RotatingFileHandler)
        ]

        assert rotating_handlers, "Expected a RotatingFileHandler on the root logger"
        assert rotating_handlers[0].maxBytes == 4096
        assert rotating_handlers[0].backupCount == 7
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in original_handlers:
            root_logger.addHandler(handler)

        module_after_reload = sys.modules.get("src.interfaces.telegram_bot")
        if module_after_reload is not None and previous_admin_user_id is not None:
            module_after_reload.ADMIN_USER_ID = previous_admin_user_id
        if module_after_reload is not None and previous_bot_token is not None:
            module_after_reload.TELEGRAM_BOT_TOKEN = previous_bot_token
        if module_after_reload is not None and previous_native_audio is not None:
            module_after_reload.ENABLE_NATIVE_AUDIO = previous_native_audio


@pytest.mark.asyncio
async def test_consolidation_counters_persist_across_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "slice7_consolidation_counter.db"

    writer_ledger = LedgerMemory(db_path=str(db_path))
    await writer_ledger.initialize()
    writer, writer_tasks = _build_counter_writer(writer_ledger)
    try:
        await Orchestrator._persist_chat_turns(writer, "user_a", "msg_1", "reply_1")
        await Orchestrator._persist_chat_turns(writer, "user_a", "msg_2", "reply_2")
        await Orchestrator._persist_chat_turns(writer, "user_a", "msg_3", "reply_3")
        await _flush_scheduled(writer_tasks)
    finally:
        await writer_ledger.close()

    restored = _build_restore_orchestrator(db_path, monkeypatch)
    try:
        await Orchestrator.async_init(restored)
        assert restored._consolidation_turn_counts.get("user_a") == 3
    finally:
        await restored.ledger_memory.close()


@pytest.mark.asyncio
async def test_consolidation_counter_reset_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_module, "_CONSOLIDATION_TRIGGER_TURNS", 3)
    db_path = tmp_path / "slice7_consolidation_reset.db"

    writer_ledger = LedgerMemory(db_path=str(db_path))
    await writer_ledger.initialize()
    writer, writer_tasks = _build_counter_writer(writer_ledger)
    try:
        await Orchestrator._persist_chat_turns(writer, "user_a", "msg_1", "reply_1")
        await Orchestrator._persist_chat_turns(writer, "user_a", "msg_2", "reply_2")
        await Orchestrator._persist_chat_turns(writer, "user_a", "msg_3", "reply_3")
        await _flush_scheduled(writer_tasks)
    finally:
        await writer_ledger.close()

    restored = _build_restore_orchestrator(db_path, monkeypatch)
    try:
        await Orchestrator.async_init(restored)
        assert restored._consolidation_turn_counts.get("user_a") == 0
    finally:
        await restored.ledger_memory.close()


def test_voice_placeholder_is_enriched() -> None:
    payload = Orchestrator._coerce_user_prompt_payload(
        {
            "audio_bytes": b"x" * 1024,
            "audio_mime_type": "audio/ogg",
        }
    )

    assert re.match(r"^\[Voice note · 1024 bytes · audio/ogg\]$", payload["text"])


def test_consolidation_extract_candidates_skips_voice_placeholders() -> None:
    engine = NocturnalConsolidationSlice1()
    placeholder = "[Voice note · 1024 bytes · audio/ogg]"

    candidates = engine.extract_candidates(
        user_id="user_a",
        chat_history=[
            {"role": "user", "content": "Normal request about weather conditions."},
            {"role": "user", "content": placeholder},
        ],
    )

    texts = [candidate.text for candidate in candidates]
    assert "Normal request about weather conditions." in texts
    assert placeholder not in texts


def test_summary_skips_voice_placeholders() -> None:
    placeholder = "[Voice note · 1024 bytes · audio/ogg]"
    summary = Orchestrator._summarize_chat_history(
        [
            {"role": "user", "content": placeholder},
            {"role": "assistant", "content": "Assistant turn"},
            {"role": "user", "content": "Can you summarize our weather discussion?"},
        ]
    )

    assert summary is not None
    assert placeholder not in summary
    assert "weather discussion" in summary.lower()
