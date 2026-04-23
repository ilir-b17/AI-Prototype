import asyncio
import json
import math

import pytest

from src.core.orchestrator import Orchestrator
from src.core.nocturnal_consolidation import (
    ConsolidationCandidate,
    NocturnalConsolidationSlice1,
)


class _VectorMemoryStub:
    def __init__(self, distance_by_query=None, fail=False):
        self.distance_by_query = distance_by_query or {}
        self.fail = fail
        self.calls = []

    async def query_memory_async(self, query_text: str, n_results: int = 1):
        await asyncio.sleep(0)
        self.calls.append((query_text, n_results))
        if self.fail:
            raise RuntimeError("vector backend unavailable")
        distance = self.distance_by_query.get(query_text)
        if distance is None:
            return []
        return [{"id": "mem-1", "document": "existing", "metadata": {}, "distance": distance}]


class _CoreMemoryStub:
    def __init__(self):
        self.state = {
            "user_preferences": "",
            "nocturnal_core_facts": [],
        }
        self.updates = []

    async def get_all(self):
        await asyncio.sleep(0)
        return dict(self.state)

    async def update(self, key, value):
        await asyncio.sleep(0)
        self.state[key] = value
        self.updates.append((key, value))


class _LedgerMemoryStub:
    def __init__(self):
        self.logged = []

    async def log_event(self, log_level, message, context=None):
        await asyncio.sleep(0)
        self.logged.append({
            "level": getattr(log_level, "value", str(log_level)),
            "message": message,
            "context": context or {},
        })
        return len(self.logged)


def test_extract_candidates_collects_all_supported_sources():
    engine = NocturnalConsolidationSlice1(min_chars=10)

    candidates = engine.extract_candidates(
        user_id="user-1",
        chat_history=[
            {"role": "user", "content": "I prefer concise technical summaries with action items."},
            {"role": "assistant", "content": "Noted. I will keep responses concise and technical."},
        ],
        worker_outputs={
            "supervisor_context": "ignored context",
            "research_agent": "Found recurring interest in AMD and market news.",
        },
        critic_feedback="FAIL: The response ignored the user's explicit preference.",
        blueprint_entries=[
            {
                "document": "System 2 Reasoning Blueprint for recurring trend analysis.",
                "metadata": {"type": "system_2_learned_pattern"},
            }
        ],
        ledger_logs=[
            {"log_level": "INFO", "message": "Routine operation"},
            {"log_level": "ERROR", "message": "Tool timeout while fetching market feed."},
        ],
        now_iso="2026-04-23T12:00:00",
    )

    sources = {candidate.source for candidate in candidates}
    assert "chat_user" in sources
    assert "chat_assistant" in sources
    assert "worker_output" in sources
    assert "critic_feedback" in sources
    assert "system_2_blueprint" in sources
    assert "ledger_log" in sources


def test_deterministic_filter_drops_noise_and_exact_duplicates():
    engine = NocturnalConsolidationSlice1(min_chars=25)
    candidates = [
        ConsolidationCandidate(source="chat_user", text="Hi"),
        ConsolidationCandidate(source="chat_user", text="ERROR: network timeout"),
        ConsolidationCandidate(source="ledger_log", text="Routing to System 1 (Local Model)"),
        ConsolidationCandidate(source="chat_user", text="User prefers structured risk summaries in bullet points."),
        ConsolidationCandidate(source="chat_assistant", text="User prefers structured risk summaries in bullet points."),
    ]

    filtered = engine.deterministic_filter(candidates)

    assert len(filtered) == 1
    assert filtered[0].text == "User prefers structured risk summaries in bullet points."


@pytest.mark.asyncio
async def test_semantic_deduplicate_removes_in_batch_paraphrase_duplicates():
    engine = NocturnalConsolidationSlice1(batch_semantic_jaccard_threshold=0.8, min_chars=5)
    candidates = [
        ConsolidationCandidate(
            source="worker_output",
            text="The worker should validate input before writing data to disk.",
        ),
        ConsolidationCandidate(
            source="worker_output",
            text="Before writing data to disk, the worker should validate input.",
        ),
        ConsolidationCandidate(
            source="worker_output",
            text="Market-watch objective should trigger every 30 minutes.",
        ),
    ]

    deduped = await engine.semantic_deduplicate(candidates)

    assert len(deduped) == 2
    texts = {candidate.text for candidate in deduped}
    assert "Market-watch objective should trigger every 30 minutes." in texts


@pytest.mark.asyncio
async def test_semantic_deduplicate_uses_vector_distance_threshold():
    engine = NocturnalConsolidationSlice1(vector_distance_threshold=0.08, min_chars=5)
    vector_stub = _VectorMemoryStub(
        distance_by_query={
            "Recurring preference: user wants concise updates.": 0.03,
            "Critical: worker must retry API 429 with backoff.": 0.34,
        }
    )

    candidates = [
        ConsolidationCandidate(source="chat_user", text="Recurring preference: user wants concise updates."),
        ConsolidationCandidate(source="worker_output", text="Critical: worker must retry API 429 with backoff."),
    ]

    deduped = await engine.semantic_deduplicate(candidates, vector_memory=vector_stub)

    assert len(deduped) == 1
    assert deduped[0].text == "Critical: worker must retry API 429 with backoff."
    assert len(vector_stub.calls) == 2


@pytest.mark.asyncio
async def test_semantic_deduplicate_keeps_candidates_when_vector_query_fails():
    engine = NocturnalConsolidationSlice1(min_chars=5)
    vector_stub = _VectorMemoryStub(fail=True)
    candidates = [
        ConsolidationCandidate(source="chat_user", text="User prefers calendar reminders in the morning."),
        ConsolidationCandidate(source="worker_output", text="Research agent found repeated risk tolerance constraints."),
    ]

    deduped = await engine.semantic_deduplicate(candidates, vector_memory=vector_stub)

    assert len(deduped) == 2


@pytest.mark.asyncio
async def test_extract_and_filter_candidates_runs_end_to_end_pipeline():
    engine = NocturnalConsolidationSlice1(min_chars=25)
    vector_stub = _VectorMemoryStub(
        distance_by_query={
            "research_agent: User repeatedly asks for AMD catalyst updates and timeline checks.": 0.25,
        }
    )

    filtered = await engine.extract_and_filter_candidates(
        user_id="user-1",
        chat_history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ],
        worker_outputs={
            "research_agent": "User repeatedly asks for AMD catalyst updates and timeline checks.",
            "supervisor_context": "ignored",
        },
        critic_feedback="PASS",
        vector_memory=vector_stub,
    )

    assert len(filtered) == 1
    assert filtered[0].source == "worker_output"
    assert "AMD catalyst updates" in filtered[0].text


def test_compute_quality_score_uses_programmatic_formula_exactly():
    engine = NocturnalConsolidationSlice1()
    factors = {
        "novelty": 5,
        "actionability": 4,
        "confidence": 3,
        "recurrence": 4,
        "charter_alignment": 5,
        "contradiction_risk": 2,
        "staleness": 1,
    }

    expected = (
        0.30 * 5
        + 0.25 * 4
        + 0.20 * 3
        + 0.15 * 4
        + 0.10 * 5
        - 0.20 * 2
        - 0.15 * 1
    )
    computed = engine.compute_quality_score(factors)

    assert math.isclose(computed, expected, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.asyncio
async def test_score_candidates_with_system2_applies_threshold_pass_fail():
    engine = NocturnalConsolidationSlice1()
    candidates = [
        ConsolidationCandidate(source="worker_output", text="Candidate one"),
        ConsolidationCandidate(source="worker_output", text="Candidate two"),
    ]

    mocked_json = {
        "scores": {
            "cand_1": {
                "novelty": 5,
                "actionability": 5,
                "confidence": 4,
                "recurrence": 4,
                "charter_alignment": 5,
                "contradiction_risk": 1,
                "staleness": 1,
            },
            "cand_2": {
                "novelty": 2,
                "actionability": 2,
                "confidence": 2,
                "recurrence": 1,
                "charter_alignment": 2,
                "contradiction_risk": 5,
                "staleness": 5,
            },
        }
    }

    async def fake_route_to_system_2(_messages):
        await asyncio.sleep(0)
        return json.dumps(mocked_json)

    scored = await engine.score_candidates_with_system2(
        candidates,
        route_to_system_2=fake_route_to_system_2,
        threshold=3.0,
    )

    assert len(scored) == 2
    assert scored[0].candidate_id == "cand_1"
    assert scored[1].candidate_id == "cand_2"

    expected_first = (
        0.30 * 5
        + 0.25 * 5
        + 0.20 * 4
        + 0.15 * 4
        + 0.10 * 5
        - 0.20 * 1
        - 0.15 * 1
    )
    expected_second = (
        0.30 * 2
        + 0.25 * 2
        + 0.20 * 2
        + 0.15 * 1
        + 0.10 * 2
        - 0.20 * 5
        - 0.15 * 5
    )

    assert math.isclose(scored[0].q_score, expected_first, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(scored[1].q_score, expected_second, rel_tol=1e-12, abs_tol=1e-12)
    assert scored[0].passed_threshold is True
    assert scored[1].passed_threshold is False


@pytest.mark.asyncio
async def test_score_candidates_with_system2_uses_phase2_redaction_boundary():
    engine = NocturnalConsolidationSlice1()
    captured = {}
    candidates = [
        ConsolidationCandidate(
            source="chat_user",
            text=(
                "<Core_Working_Memory>name=Alice, location=Vienna</Core_Working_Memory> "
                "My email is alice@example.com and C:\\Users\\Alice\\Secrets\\todo.txt"
            ),
        )
    ]

    async def fake_route_to_system_2(messages):
        await asyncio.sleep(0)
        captured["messages"] = messages
        return {
            "scores": {
                "cand_1": {
                    "novelty": 3,
                    "actionability": 3,
                    "confidence": 3,
                    "recurrence": 3,
                    "charter_alignment": 3,
                    "contradiction_risk": 2,
                    "staleness": 2,
                }
            }
        }

    scored = await engine.score_candidates_with_system2(
        candidates,
        route_to_system_2=fake_route_to_system_2,
        redactor=Orchestrator._redact_text_for_cloud,
    )

    assert len(scored) == 1
    user_payload = captured["messages"][1]["content"]
    assert "alice@example.com" not in user_payload
    assert "C:\\Users\\Alice\\Secrets" not in user_payload
    assert "name=Alice" not in user_payload
    assert "REDACTED" in user_payload


def test_route_storage_bucket_categorizes_core_vector_ledger():
    engine = NocturnalConsolidationSlice1()

    core_scored = engine.apply_quality_threshold(
        [ConsolidationCandidate(source="chat_user", text="User preference: prefers concise outputs.")],
        {
            "cand_1": {
                "novelty": 4,
                "actionability": 4,
                "confidence": 4,
                "recurrence": 3,
                "charter_alignment": 4,
                "contradiction_risk": 1,
                "staleness": 1,
            }
        },
    )[0]
    ledger_scored = engine.apply_quality_threshold(
        [ConsolidationCandidate(source="critic_feedback", text="Charter compliance audit event")],
        {
            "cand_1": {
                "novelty": 4,
                "actionability": 4,
                "confidence": 4,
                "recurrence": 3,
                "charter_alignment": 4,
                "contradiction_risk": 1,
                "staleness": 1,
            }
        },
    )[0]
    vector_scored = engine.apply_quality_threshold(
        [ConsolidationCandidate(source="system_2_blueprint", text="Generalized episodic insight about market cycles")],
        {
            "cand_1": {
                "novelty": 4,
                "actionability": 4,
                "confidence": 4,
                "recurrence": 3,
                "charter_alignment": 4,
                "contradiction_risk": 1,
                "staleness": 1,
            }
        },
    )[0]

    assert engine.route_storage_bucket(core_scored) == "core"
    assert engine.route_storage_bucket(ledger_scored) == "ledger"
    assert engine.route_storage_bucket(vector_scored) == "vector"


@pytest.mark.asyncio
async def test_write_back_scored_candidates_inserts_into_all_memory_targets():
    engine = NocturnalConsolidationSlice1()
    scored = [
        engine.apply_quality_threshold(
            [ConsolidationCandidate(source="chat_user", text="User preference: prefers technical summaries.")],
            {
                "cand_1": {
                    "novelty": 5,
                    "actionability": 5,
                    "confidence": 4,
                    "recurrence": 4,
                    "charter_alignment": 5,
                    "contradiction_risk": 1,
                    "staleness": 1,
                }
            },
            threshold=3.0,
        )[0],
        engine.apply_quality_threshold(
            [ConsolidationCandidate(source="system_2_blueprint", text="Generalized episodic insight for trend analysis")],
            {
                "cand_1": {
                    "novelty": 5,
                    "actionability": 4,
                    "confidence": 4,
                    "recurrence": 4,
                    "charter_alignment": 4,
                    "contradiction_risk": 1,
                    "staleness": 1,
                }
            },
            threshold=3.0,
        )[0],
        engine.apply_quality_threshold(
            [ConsolidationCandidate(source="critic_feedback", text="Critic charter audit event")],
            {
                "cand_1": {
                    "novelty": 5,
                    "actionability": 4,
                    "confidence": 4,
                    "recurrence": 4,
                    "charter_alignment": 5,
                    "contradiction_risk": 1,
                    "staleness": 1,
                }
            },
            threshold=3.0,
        )[0],
    ]

    core_stub = _CoreMemoryStub()
    vector_stub = _VectorMemoryStub()
    vector_stub.added = []

    async def _add_memory_async(text, metadata):
        await asyncio.sleep(0)
        vector_stub.added.append({"text": text, "metadata": metadata})
        return "mem-1"

    vector_stub.add_memory_async = _add_memory_async
    ledger_stub = _LedgerMemoryStub()

    result = await engine.write_back_scored_candidates(
        scored,
        core_memory=core_stub,
        vector_memory=vector_stub,
        ledger_memory=ledger_stub,
        threshold=3.0,
    )

    assert result["passed"] == 3
    assert result["stored_core"] == 1
    assert result["stored_vector"] == 1
    assert result["stored_ledger"] == 1
    assert len(core_stub.state["nocturnal_core_facts"]) == 1
    assert len(vector_stub.added) == 1
    assert len(ledger_stub.logged) == 1


@pytest.mark.asyncio
async def test_full_pipeline_extract_filter_score_write_back_integration():
    engine = NocturnalConsolidationSlice1(min_chars=20)
    core_stub = _CoreMemoryStub()
    vector_stub = _VectorMemoryStub()
    vector_stub.added = []

    async def _add_memory_async(text, metadata):
        await asyncio.sleep(0)
        vector_stub.added.append({"text": text, "metadata": metadata})
        return "mem-1"

    vector_stub.add_memory_async = _add_memory_async
    ledger_stub = _LedgerMemoryStub()

    filtered = await engine.extract_and_filter_candidates(
        user_id="u1",
        chat_history=[
            {"role": "user", "content": "User preference: prefers concise execution summaries with bullet points."},
            {"role": "assistant", "content": "Acknowledged."},
        ],
        worker_outputs={
            "research_agent": "Generalized episodic insight about repeated market catalyst checks.",
        },
        critic_feedback="FAIL: Charter compliance audit needed for the previous response.",
        ledger_logs=[
            {"log_level": "ERROR", "message": "Critic escalation triggered for charter review."},
        ],
        vector_memory=vector_stub,
    )

    async def fake_route_to_system_2(_messages):
        await asyncio.sleep(0)
        scores = {"scores": {}}
        for idx, _candidate in enumerate(filtered, start=1):
            scores["scores"][f"cand_{idx}"] = {
                "novelty": 4,
                "actionability": 4,
                "confidence": 4,
                "recurrence": 3,
                "charter_alignment": 4,
                "contradiction_risk": 1,
                "staleness": 1,
            }
        return scores

    scored = await engine.score_candidates_with_system2(
        filtered,
        route_to_system_2=fake_route_to_system_2,
        threshold=3.0,
    )
    result = await engine.write_back_scored_candidates(
        scored,
        core_memory=core_stub,
        vector_memory=vector_stub,
        ledger_memory=ledger_stub,
        threshold=3.0,
    )

    assert result["passed"] >= 1
    assert result["stored_total"] >= 1
    assert core_stub.state["nocturnal_core_facts"] or vector_stub.added or ledger_stub.logged
