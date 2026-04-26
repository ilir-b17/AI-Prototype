"""
pytest integration for Tier 3 LLM response quality evals.

These tests are marked @pytest.mark.slow and require Ollama running.
They skip gracefully when Ollama is offline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile

import pytest

from tests.evals.framework.grader import is_ollama_available
from tests.evals.framework.snapshot import check_regression, save_snapshot

logger = logging.getLogger(__name__)

LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")


def _require_ollama() -> None:
    """Skip test if Ollama is not available."""
    if not is_ollama_available(LOCAL_MODEL):
        pytest.skip(f"Ollama not running or model {LOCAL_MODEL!r} not available")


def _require_mfa_passphrase() -> None:
    """Skip test if orchestrator MFA prerequisite is not configured."""
    if not str(os.getenv("MFA_PASSPHRASE", "")).strip():
        pytest.skip("MFA_PASSPHRASE is not configured; skipping Tier 3 orchestrator evals")


@pytest.fixture(scope="module")
def full_orchestrator():
    """Fully initialized Orchestrator for quality evals.

    This fixture is module-scoped to avoid re-initializing for every test.
    It connects to real databases using temporary paths.
    """
    _require_ollama()
    _require_mfa_passphrase()

    tmp = tempfile.mkdtemp(prefix="aiden_eval_")
    vector_path = os.path.join(tmp, "chroma")
    ledger_path = os.path.join(tmp, "ledger.db")
    core_path = os.path.join(tmp, "core_memory.json")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    from src.core.orchestrator import Orchestrator

    orch = Orchestrator(
        vector_db_path=vector_path,
        ledger_db_path=ledger_path,
        core_memory_path=core_path,
    )
    orch._eval_loop = loop

    loop.run_until_complete(orch.async_init())

    try:
        yield orch
    finally:
        try:
            if hasattr(orch, "cognitive_router") and orch.cognitive_router:
                loop.run_until_complete(orch.cognitive_router.close())
            if hasattr(orch, "ledger_memory") and orch.ledger_memory:
                loop.run_until_complete(orch.ledger_memory.close())
            orch.close()
        except Exception:
            logger.exception("Failed to cleanly close quality eval orchestrator")
        finally:
            try:
                if not loop.is_closed():
                    loop.close()
            except Exception:
                logger.exception("Failed to close quality eval event loop")
            asyncio.set_event_loop(None)
            shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.tier3
def test_response_quality_pass_rate(full_orchestrator):
    """Response quality eval suite must achieve >= 70% pass rate."""
    from tests.evals.suites.quality_evals import ResponseQualityEvalSuite

    suite = ResponseQualityEvalSuite(full_orchestrator, local_model=LOCAL_MODEL)
    result = suite.run()
    result.save()

    regression = check_regression("response_quality", result.pass_rate, tolerance=0.10)
    if regression:
        pytest.fail(regression)

    if result.pass_rate < 0.70:
        failures = [r for r in result.results if not r.passed]
        detail = "\n".join(
            f"  [{r.case_id}] {', '.join(r.failures[:2])}"
            for r in failures
        )
        pytest.fail(
            f"Response quality pass rate {result.pass_rate:.1%} < 70%.\n"
            f"Failures:\n{detail}"
        )


@pytest.mark.slow
@pytest.mark.tier3
def test_no_internal_markup_in_responses(full_orchestrator):
    """No response must contain internal markup (WORKERS, tool_call, etc.)."""
    from tests.evals.suites.quality_evals import ResponseQualityEvalSuite

    suite = ResponseQualityEvalSuite(full_orchestrator, local_model=LOCAL_MODEL)
    result = suite.run()

    markup_leaks = [
        r for r in result.results
        if any(
            forbidden in r.actual.get("response_preview", "")
            for forbidden in ["WORKERS:", "tool_call", "function_call"]
        )
    ]
    assert not markup_leaks, (
        f"Internal markup leaked to user in {len(markup_leaks)} responses:\n"
        + "\n".join(f"  [{r.case_id}]" for r in markup_leaks)
    )


@pytest.mark.slow
@pytest.mark.tier3
def test_save_quality_baseline(full_orchestrator):
    """Utility: capture current quality pass rate as baseline snapshot.

    Run manually after confirming quality level is acceptable:
      pytest tests/evals/test_quality.py::test_save_quality_baseline -s -m slow
    """
    from tests.evals.suites.quality_evals import ResponseQualityEvalSuite

    suite = ResponseQualityEvalSuite(full_orchestrator, local_model=LOCAL_MODEL)
    result = suite.run()
    path = save_snapshot("response_quality", result.pass_rate)
    print(f"\nBaseline saved: {path} (pass_rate={result.pass_rate:.1%})")
