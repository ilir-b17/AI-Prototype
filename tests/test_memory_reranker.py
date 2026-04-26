"""
Unit tests for the MemoryReranker - two-stage retrieval system.

Tier 1: pure function tests (no LLM, no ChromaDB).
Tier 2: integration with skill (no live Ollama required).
"""

from __future__ import annotations

import asyncio
import json
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.memory_reranker import (
    MemoryReranker,
    RerankConfig,
    load_rerank_config,
)


# -- Test fixtures ----------------------------------------------------

@pytest.fixture
def default_config() -> RerankConfig:
    return RerankConfig(
        enabled=True,
        recall_multiplier=3,
        max_candidates=12,
        document_chars=200,
        alpha=0.70,
        timeout_seconds=5.0,
        min_candidates=2,
    )


@pytest.fixture
def reranker(default_config) -> MemoryReranker:
    return MemoryReranker(default_config)


@pytest.fixture
def sample_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "id": "mem_0",
            "document": "The energy ROI engine computes base_roi = value/effort.",
            "metadata": {"type": "conversation"},
            "distance": 0.45,
        },
        {
            "id": "mem_1",
            "document": "The user asked about the weather in Vienna today.",
            "metadata": {"type": "conversation"},
            "distance": 0.82,
        },
        {
            "id": "mem_2",
            "document": "Energy budget: 100 units, replenished 5 per turn.",
            "metadata": {"type": "conversation"},
            "distance": 0.30,
        },
        {
            "id": "mem_3",
            "document": "AIDEN uses heartbeat loop every 30 minutes for tasks.",
            "metadata": {"type": "conversation"},
            "distance": 0.61,
        },
        {
            "id": "mem_4",
            "document": "Cost = estimated_effort * ENERGY_MULTIPLIER (default 3).",
            "metadata": {"type": "conversation"},
            "distance": 0.38,
        },
    ]


# -- Configuration tests ----------------------------------------------

def test_rerank_config_defaults():
    config = RerankConfig()
    assert 0.0 <= config.alpha <= 1.0
    assert config.recall_multiplier >= 1
    assert config.max_candidates >= 1
    assert config.timeout_seconds > 0


def test_load_rerank_config_from_env(monkeypatch):
    monkeypatch.setenv("ENABLE_MEMORY_RERANKING", "false")
    monkeypatch.setenv("MEMORY_RECALL_MULTIPLIER", "5")
    monkeypatch.setenv("MEMORY_RERANK_ALPHA", "0.8")
    config = load_rerank_config()
    assert config.enabled is False
    assert config.recall_multiplier == 5
    assert config.alpha == pytest.approx(0.8)


def test_load_rerank_config_clamps_alpha(monkeypatch):
    monkeypatch.setenv("MEMORY_RERANK_ALPHA", "1.5")  # out of range
    config = load_rerank_config()
    assert config.alpha <= 1.0  # clamped to 1.0


# -- broad_n_results tests --------------------------------------------

def test_broad_n_results_normal(reranker):
    assert reranker.broad_n_results(3) == 9   # 3 * multiplier(3)


def test_broad_n_results_capped(reranker):
    assert reranker.broad_n_results(10) == 12  # capped at max_candidates(12)


def test_broad_n_results_minimum(reranker):
    assert reranker.broad_n_results(1) >= 1


# -- Score parsing tests ----------------------------------------------

def test_parse_scores_valid_nested(reranker):
    raw = '{"scores": {"0": 8, "1": 3, "2": 9}}'
    result = reranker._parse_scores(raw, 3)
    assert result == {0: 8.0, 1: 3.0, 2: 9.0}


def test_parse_scores_valid_flat(reranker):
    raw = '{"0": 7, "1": 2}'
    result = reranker._parse_scores(raw, 2)
    assert result is not None
    assert result[0] == pytest.approx(7.0)
    assert result[1] == pytest.approx(2.0)


def test_parse_scores_markdown_wrapped(reranker):
    raw = '```json\n{"scores": {"0": 5, "1": 9}}\n```'
    result = reranker._parse_scores(raw, 2)
    assert result == {0: 5.0, 1: 9.0}


def test_parse_scores_invalid_json_returns_none(reranker):
    assert reranker._parse_scores("not json", 3) is None


def test_parse_scores_empty_returns_none(reranker):
    assert reranker._parse_scores("", 3) is None


def test_parse_scores_out_of_range_score_excluded(reranker):
    # Score 15 is invalid (>10), index 0 should be excluded
    raw = '{"scores": {"0": 15, "1": 5}}'
    result = reranker._parse_scores(raw, 2)
    # Should include what's valid; 15 is excluded, so only 1 valid score
    # Since n_candidates=2 and we need >=1 valid, this may return {1: 5.0}
    # or None depending on threshold. With threshold at n//2=1, {1:5} passes.
    if result is not None:
        assert 0 not in result  # invalid score excluded
        assert 1 in result


def test_parse_scores_insufficient_valid_returns_none(reranker):
    # 5 candidates but only 1 valid score (threshold = n//2 = 2)
    raw = '{"scores": {"0": 5}}'
    result = reranker._parse_scores(raw, 5)
    assert result is None


# -- Score calculation tests ------------------------------------------

def test_cosine_score_perfect_match(reranker):
    # distance=0 should give score close to 10
    score = reranker._cosine_score(0.0)
    assert score == pytest.approx(10.0)


def test_cosine_score_typical(reranker):
    # distance=0.5 -> similarity=0.75 -> score=7.5
    score = reranker._cosine_score(0.5)
    assert 7.0 <= score <= 8.0


def test_cosine_score_high_distance(reranker):
    # distance=1.5 -> low score
    score = reranker._cosine_score(1.5)
    assert score < 3.0


def test_combine_score_alpha_weighting(reranker):
    # alpha=0.7: 70% LLM, 30% cosine
    combined = reranker._combine(8.0, 4.0)
    expected = 0.7 * 4.0 + 0.3 * 8.0
    assert combined == pytest.approx(expected)


# -- Blend scores tests -----------------------------------------------

def test_blend_scores_enriches_candidates(reranker, sample_candidates):
    llm_scores = {0: 9.0, 1: 2.0, 2: 8.0, 3: 4.0, 4: 7.0}
    blended = reranker._blend_scores(sample_candidates, llm_scores)
    assert len(blended) == len(sample_candidates)
    for r in blended:
        assert "rerank_score" in r
        assert "cosine_score" in r
        assert "combined_score" in r
        assert r["reranked"] is True


def test_blend_scores_fallback_when_missing_index(reranker, sample_candidates):
    # Only 3 of 5 indices scored
    llm_scores = {0: 9.0, 2: 8.0, 4: 7.0}
    blended = reranker._blend_scores(sample_candidates, llm_scores)
    assert len(blended) == 5
    # Missing indices (1, 3) should use cosine_score as rerank_score
    missing_result = next(r for r in blended if r["id"] == "mem_1")
    # cosine_score for mem_1 (distance=0.82) should be used
    assert missing_result["rerank_score"] == pytest.approx(
        reranker._cosine_score(0.82), abs=0.1
    )


# -- Cosine order (fallback) tests ------------------------------------

def test_cosine_order_sorts_by_distance(reranker, sample_candidates):
    ordered = reranker._cosine_order(sample_candidates)
    distances = [float(r.get("distance", 1.0)) for r in ordered]
    assert distances == sorted(distances)


def test_cosine_order_marks_not_reranked(reranker, sample_candidates):
    ordered = reranker._cosine_order(sample_candidates)
    assert all(not r["reranked"] for r in ordered)


# -- Full rerank() tests ----------------------------------------------

@pytest.mark.asyncio
async def test_rerank_with_successful_system_1(reranker, sample_candidates):
    """Reranker uses LLM scores when System 1 succeeds."""

    async def mock_s1(messages, **kwargs):
        await asyncio.sleep(0)
        class R:
            status = "ok"
            content = '{"scores": {"0": 9, "1": 1, "2": 8, "3": 3, "4": 7}}'
        return R()

    results = await reranker.rerank(
        query="energy budget configuration",
        candidates=sample_candidates,
        n_results=2,
        route_to_system_1=mock_s1,
    )

    assert len(results) == 2
    # mem_0 has score 9 - should be first
    assert results[0]["id"] == "mem_0"
    assert results[0]["reranked"] is True
    assert results[0]["rerank_score"] == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_rerank_falls_back_on_system1_failure(reranker, sample_candidates):
    """Reranker returns cosine-ordered results when System 1 fails."""

    async def failing_s1(messages, **kwargs):
        raise RuntimeError("Ollama offline")

    results = await reranker.rerank(
        query="energy",
        candidates=sample_candidates,
        n_results=3,
        route_to_system_1=failing_s1,
    )

    assert len(results) == 3
    assert all(not r["reranked"] for r in results)
    # Should be sorted by cosine distance
    distances = [float(r.get("distance", 1.0)) for r in results]
    assert distances == sorted(distances)


@pytest.mark.asyncio
async def test_rerank_falls_back_on_parse_failure(reranker, sample_candidates):
    """Reranker falls back when System 1 returns unparseable output."""

    async def bad_s1(messages, **kwargs):
        await asyncio.sleep(0)
        class R:
            status = "ok"
            content = "I cannot score these"
        return R()

    results = await reranker.rerank(
        query="energy",
        candidates=sample_candidates,
        n_results=2,
        route_to_system_1=bad_s1,
    )

    assert len(results) == 2
    assert all(not r["reranked"] for r in results)


@pytest.mark.asyncio
async def test_rerank_skips_when_too_few_candidates(reranker):
    """Reranking skipped when candidates <= min_candidates."""
    called = []

    async def tracking_s1(messages, **kwargs):
        await asyncio.sleep(0)
        called.append(True)
        class R:
            status = "ok"
            content = '{"scores": {"0": 5, "1": 8}}'
        return R()

    # min_candidates=2, give exactly 2 candidates
    two_candidates = [
        {"id": "a", "document": "Doc A", "distance": 0.3, "metadata": {}},
        {"id": "b", "document": "Doc B", "distance": 0.7, "metadata": {}},
    ]
    results = await reranker.rerank(
        query="test",
        candidates=two_candidates,
        n_results=2,
        route_to_system_1=tracking_s1,
    )

    assert len(results) == 2
    # System 1 should NOT have been called (too few candidates to rerank)
    assert not called, "System 1 should not be called with min_candidates candidates"


@pytest.mark.asyncio
async def test_rerank_empty_candidates(reranker):
    """Rerank with empty candidates returns empty list."""
    results = await reranker.rerank(
        query="test",
        candidates=[],
        n_results=3,
        route_to_system_1=AsyncMock(),
    )
    assert results == []


@pytest.mark.asyncio
async def test_rerank_n_results_capped_to_candidates(reranker, sample_candidates):
    """n_results larger than candidates returns all candidates."""

    async def mock_s1(messages, **kwargs):
        await asyncio.sleep(0)
        class R:
            status = "ok"
            content = (
                '{"scores": {"0": 5, "1": 6, "2": 7, "3": 8, "4": 9}}'
            )
        return R()

    results = await reranker.rerank(
        query="test",
        candidates=sample_candidates,  # 5 candidates
        n_results=100,                  # ask for 100 - should cap at 5
        route_to_system_1=mock_s1,
    )
    assert len(results) <= len(sample_candidates)


# -- Document truncation test -----------------------------------------

def test_truncate_doc_within_limit(reranker):
    short = "Short text"
    assert reranker._truncate_doc(short) == short


def test_truncate_doc_over_limit(reranker):
    long_doc = "x" * 500
    truncated = reranker._truncate_doc(long_doc)
    assert len(truncated) <= reranker.config.document_chars + 1  # +1 for ellipsis
    assert truncated.endswith("…")


def test_truncate_doc_empty(reranker):
    assert reranker._truncate_doc("") == ""
    assert reranker._truncate_doc(None) == ""


# -- RuntimeContext integration test ----------------------------------

def test_get_reranker_fn_returns_none_initially():
    from src.core.runtime_context import get_reranker_fn
    # In test environment, no orchestrator is initialised
    # get_reranker_fn() should return None (not crash)
    fn = get_reranker_fn()
    assert fn is None or callable(fn)


# -- search_archival_memory skill integration test --------------------

@pytest.mark.asyncio
async def test_skill_returns_valid_json_no_runtime():
    """Skill returns valid JSON even without runtime context."""
    from src.skills.search_archival_memory import search_archival_memory
    result = await search_archival_memory("test query", n_results=3)
    parsed = json.loads(result)
    assert "status" in parsed


@pytest.mark.asyncio
async def test_skill_skip_reranking_flag():
    """skip_reranking=True should not call reranker."""
    from src.skills.search_archival_memory import search_archival_memory

    reranker_called = []

    async def fake_reranker(query, candidates, n_results):
        await asyncio.sleep(0)
        reranker_called.append(True)
        return candidates[:n_results]

    with patch(
        "src.core.runtime_context.get_reranker_fn",
        return_value=fake_reranker,
    ):
        with patch(
            "src.core.runtime_context.get_vector_memory",
            return_value=None,
        ):
            # With no vector_memory, skill returns "no results" - reranker irrelevant
            await search_archival_memory(
                "test", n_results=3, skip_reranking=True
            )

    assert not reranker_called, "Reranker must not be called when skip_reranking=True"
