import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.memory.vector_db as vector_db
from src.core.nocturnal_consolidation import (
    ConsolidationCandidate,
    NocturnalConsolidationSlice1,
    ScoredCandidate,
)
from src.core.orchestrator import Orchestrator
from src.core.prompt_config import build_supervisor_prompt
from src.core.state_model import normalize_state
from src.memory.core_memory import CoreMemory


def _close_fire_and_forget(coro):
    coro.close()


def _build_explicit_memory_orchestrator(*, profile_updated: bool = False) -> Orchestrator:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.pending_mfa = {}
    orchestrator.pending_hitl_state = {}
    orchestrator.pending_tool_approval = {}
    orchestrator._energy_budget = 100
    orchestrator._energy_budget_lock = asyncio.Lock()
    orchestrator._ready = asyncio.Event()
    orchestrator._ready.set()
    orchestrator._user_locks = {}
    orchestrator._user_locks_lock = asyncio.Lock()
    orchestrator.vector_memory = MagicMock()
    orchestrator.vector_memory.add_memory_async = AsyncMock(return_value="mem-1")
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.save_chat_turn = AsyncMock()
    orchestrator._consolidate_memory = AsyncMock()
    orchestrator._save_memory_async = AsyncMock()
    orchestrator._fire_and_forget = _close_fire_and_forget
    orchestrator._try_resume_mfa = AsyncMock(return_value=None)
    orchestrator.synthesis_pipeline = types.SimpleNamespace(
        try_resume_tool_approval=AsyncMock(return_value=None)
    )
    orchestrator._remember_user_profile = AsyncMock(return_value=profile_updated)
    orchestrator._remember_assistant_identity = AsyncMock()
    orchestrator._try_goal_planning_response = AsyncMock(return_value="Memory noted.")
    orchestrator._try_fast_path_response = AsyncMock(return_value=None)
    orchestrator._run_graph_loop = AsyncMock(return_value="Graph fallback should not run")
    orchestrator._load_state = AsyncMock(
        return_value=normalize_state({"user_id": "user-1", "user_input": ""})
    )
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.sanitize_response = MagicMock(side_effect=lambda text: text)
    return orchestrator


@pytest.mark.asyncio
async def test_nocturnal_core_facts_written_by_consolidation_surface_in_supervisor_prompt(tmp_path):
    fact = "User preference: prefers concise execution summaries with metric units."
    memory = CoreMemory(str(tmp_path / "core_memory.json"))
    engine = NocturnalConsolidationSlice1()
    scored = [
        ScoredCandidate(
            candidate_id="cand_1",
            candidate=ConsolidationCandidate(source="chat_user", text=fact, user_id="user-1"),
            factor_scores={"novelty": 5, "actionability": 4, "confidence": 4},
            q_score=4.0,
            passed_threshold=True,
        )
    ]

    result = await engine.write_back_scored_candidates(
        scored,
        core_memory=memory,
        vector_memory=None,
        ledger_memory=None,
        threshold=3.0,
    )
    core_context = await memory.get_context_string()
    supervisor_prompt = build_supervisor_prompt(
        charter_text="Core Directive: preserve user agency.",
        core_mem_str=core_context,
        archival_block="",
        capabilities_str="",
        agent_descriptions="",
        sensory_context="",
        os_name="Windows",
        downloads_dir="downloads",
    )

    assert result["stored_core"] == 1
    assert "<Nocturnal_Core_Facts>" in supervisor_prompt
    assert fact in supervisor_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Remember that my dog is Max",
        "Please remember I prefer metric units.",
    ],
)
async def test_explicit_memory_requests_bypass_conversation_length_threshold(message):
    orchestrator = _build_explicit_memory_orchestrator(profile_updated=False)

    response = await orchestrator.process_message(message, "user-1")

    assert response == "Memory noted."
    orchestrator.vector_memory.add_memory_async.assert_awaited_once_with(
        text=message,
        metadata={"type": "explicit_memory", "source": "user_request"},
    )


@pytest.mark.asyncio
async def test_structured_profile_updates_do_not_duplicate_explicit_vector_memory():
    orchestrator = _build_explicit_memory_orchestrator(profile_updated=True)

    await orchestrator.process_message("Please remember I prefer metric units.", "user-1")

    orchestrator.vector_memory.add_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_error_like_assistant_text_persists_when_turn_not_marked_failed():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.save_chat_turn = AsyncMock()
    orchestrator._consolidate_memory = AsyncMock()
    orchestrator._fire_and_forget = _close_fire_and_forget

    await Orchestrator._persist_chat_turns(
        orchestrator,
        "user-1",
        "What happened to the scheduled job?",
        "Planning timed out yesterday, but the retry completed this morning.",
    )

    assert orchestrator.ledger_memory.save_chat_turn.await_count == 2


@pytest.mark.asyncio
async def test_failed_turn_state_flag_skips_chat_persistence():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.ledger_memory = MagicMock()
    orchestrator.ledger_memory.save_chat_turn = AsyncMock()

    await Orchestrator._persist_chat_turns(
        orchestrator,
        "user-1",
        "Please plan this.",
        "Planning timed out. Please try again.",
        turn_failed=True,
    )

    orchestrator.ledger_memory.save_chat_turn.assert_not_awaited()


def test_vector_memory_init_normalizes_legacy_timestamps(monkeypatch, tmp_path):
    class FakeCollection:
        def __init__(self):
            self.records = {
                "legacy-z": {"timestamp": "2024-01-02T03:04:05Z", "type": "conversation"},
                "legacy-space-offset": {"timestamp": "2024-01-02 04:05:06+02:00"},
                "canonical": {"timestamp": "2024-01-02T03:04:05"},
                "invalid": {"timestamp": "not-a-date"},
            }
            self.update_calls = []

        def get(self, include=None):
            return {
                "ids": list(self.records.keys()),
                "metadatas": [dict(metadata) for metadata in self.records.values()],
            }

        def update(self, ids, metadatas):
            self.update_calls.append((list(ids), [dict(metadata) for metadata in metadatas]))
            for memory_id, metadata in zip(ids, metadatas):
                self.records[memory_id] = dict(metadata)

    class FakeClient:
        def __init__(self, collection):
            self.collection = collection

        def get_or_create_collection(self, name, metadata=None):
            return self.collection

    collection = FakeCollection()
    monkeypatch.setattr(vector_db.chromadb, "Client", lambda _settings: FakeClient(collection))

    memory = vector_db.VectorMemory(persist_dir=str(tmp_path))

    assert collection.records["legacy-z"]["timestamp"] == "2024-01-02T03:04:05"
    assert collection.records["legacy-space-offset"]["timestamp"] == "2024-01-02T02:05:06"
    assert collection.records["canonical"]["timestamp"] == "2024-01-02T03:04:05"
    assert collection.records["invalid"]["timestamp"] == "not-a-date"
    assert len(collection.update_calls) == 1

    collection.update_calls.clear()
    assert memory._normalize_legacy_timestamps() == 0
    assert collection.update_calls == []