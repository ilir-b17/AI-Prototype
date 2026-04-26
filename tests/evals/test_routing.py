"""
pytest integration for routing eval suites.
These are Tier 2 (SkillRegistry loaded from disk, no LLM).
"""

import pytest

from tests.evals.framework.snapshot import check_regression, save_snapshot
from tests.evals.suites.routing_evals import (
    ComplexityEvalSuite,
    RoutingEvalSuite,
    TickerExtractionEvalSuite,
)


@pytest.mark.tier2
def test_routing_pass_rate(routing_orchestrator):
    """Routing eval suite must maintain >= 80% pass rate."""
    suite = RoutingEvalSuite(routing_orchestrator)
    result = suite.run()
    path = result.save()

    regression = check_regression("routing", result.pass_rate, tolerance=0.05)
    if regression:
        pytest.fail(regression)

    failures = [r for r in result.results if not r.passed]
    if failures:
        detail = "\n".join(
            f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures
        )
        # Fail if below threshold, warn otherwise
        if result.pass_rate < 0.80:
            pytest.fail(
                f"Routing eval pass rate {result.pass_rate:.1%} < 80%.\n"
                f"Failures:\n{detail}\nResults saved to: {path}"
            )
        else:
            print(f"\nWarning: {len(failures)} routing failures:\n{detail}")


@pytest.mark.tier2
def test_routing_fast_path_accuracy(routing_orchestrator):
    """Fast path cases (direct + single_tool) must achieve 90% accuracy."""
    suite = RoutingEvalSuite(routing_orchestrator)
    fast_cases = ["direct", "single_tool"]
    result = suite.run(filter_tags=fast_cases)
    failures = [r for r in result.results if not r.passed]
    assert result.pass_rate >= 0.90, (
        f"Fast path accuracy {result.pass_rate:.1%} < 90%.\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )


@pytest.mark.tier2
def test_weather_not_routed_to_web_search(routing_orchestrator):
    """Weather queries must never select web_search when weather_current exists."""
    suite = RoutingEvalSuite(routing_orchestrator)
    result = suite.run(filter_tags=["weather"])
    web_search_hits = [
        r for r in result.results if r.actual.get("tool") == "web_search"
    ]
    assert not web_search_hits, (
        "weather_current queries incorrectly routed to web_search: "
        f"{[r.case_id for r in web_search_hits]}"
    )


@pytest.mark.tier1
def test_complexity_ordering(routing_orchestrator):
    """Simple messages must have lower complexity than complex ones."""
    suite = ComplexityEvalSuite(routing_orchestrator)
    result = suite.run()
    assert result.pass_rate >= 0.9999, (
        "Complexity ordering failures: "
        + "\n".join(
            f"  [{r.case_id}] {', '.join(r.failures)}"
            for r in result.results
            if not r.passed
        )
    )


@pytest.mark.tier1
def test_ticker_extraction(routing_orchestrator):
    """Ticker extraction must correctly identify multi-ticker messages."""
    suite = TickerExtractionEvalSuite(routing_orchestrator)
    result = suite.run()
    assert result.pass_rate >= 0.9999, (
        "Ticker extraction failures:\n"
        + "\n".join(
            f"  [{r.case_id}] {', '.join(r.failures)}"
            for r in result.results
            if not r.passed
        )
    )


@pytest.mark.tier2
def test_save_routing_baseline(routing_orchestrator):
    """Utility: capture current pass rate as the baseline snapshot.

    Run this manually after confirming routing behaviour is correct:
        pytest tests/evals/test_routing.py::test_save_routing_baseline -s
    """
    suite = RoutingEvalSuite(routing_orchestrator)
    result = suite.run()
    path = save_snapshot("routing", result.pass_rate)
    print(f"\nBaseline saved: {path} (pass_rate={result.pass_rate:.1%})")
