"""pytest integration for moral audit eval suites."""

import pytest

from tests.evals.suites.moral_evals import (
    MoralPayloadEvalSuite,
    MoralSeverityEvalSuite,
    TrivialityBypassEvalSuite,
)


@pytest.mark.tier1
def test_moral_payload_parsing():
    """All moral decision payloads must be parsed correctly."""
    suite = MoralPayloadEvalSuite()
    result = suite.run()
    result.save()
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "Moral payload parsing failures:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )


@pytest.mark.tier1
def test_invalid_payloads_rejected():
    """All invalid moral payloads must produce validation_error != ''."""
    suite = MoralPayloadEvalSuite()
    result = suite.run(filter_tags=["invalid"])
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "Invalid payloads not rejected:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )


@pytest.mark.tier2
def test_triviality_bypass_decisions(routing_orchestrator):
    """Triviality bypass must fire exactly when expected."""
    suite = TrivialityBypassEvalSuite(routing_orchestrator)
    result = suite.run()
    result.save()
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "Triviality bypass failures:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )


@pytest.mark.tier2
def test_no_false_bypasses(routing_orchestrator):
    """Outputs that must NOT bypass the critic must not be bypassed."""
    suite = TrivialityBypassEvalSuite(routing_orchestrator)
    result = suite.run(filter_tags=["no_bypass"])
    false_bypasses = [
        r for r in result.results
        if r.actual.get("bypassed") is True
    ]
    assert not false_bypasses, (
        "False positives - outputs incorrectly bypassing moral audit:\n"
        + "\n".join(f"  [{r.case_id}]" for r in false_bypasses)
    )


@pytest.mark.tier1
def test_moral_severity_classification(routing_orchestrator):
    """Moral rejection severity must match expected tier."""
    suite = MoralSeverityEvalSuite(routing_orchestrator)
    result = suite.run()
    result.save()
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "Moral severity classification failures:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )
