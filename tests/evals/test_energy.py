"""pytest integration for energy eval suites."""

import pytest

from tests.evals.suites.energy_evals import EnergyJudgeParsingEvalSuite, EnergyROIEvalSuite


@pytest.mark.tier1
def test_energy_roi_decisions():
    """All energy gate decisions must match expected outcomes."""
    suite = EnergyROIEvalSuite()
    result = suite.run()
    result.save()
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "Energy ROI decision failures:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )


@pytest.mark.tier1
def test_energy_roi_approve_cases():
    """All approve cases must pass with should_execute=True."""
    suite = EnergyROIEvalSuite()
    result = suite.run(filter_tags=["approve"])
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "Approve cases incorrectly deferred:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )


@pytest.mark.tier1
def test_energy_roi_defer_cases():
    """All defer cases must pass with should_execute=False."""
    suite = EnergyROIEvalSuite()
    result = suite.run(filter_tags=["defer"])
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "Defer cases incorrectly approved:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )


@pytest.mark.tier1
def test_energy_judge_parsing():
    """EnergyJudge must correctly parse well-formed and malformed responses."""
    suite = EnergyJudgeParsingEvalSuite()
    result = suite.run()
    result.save()
    failures = [r for r in result.results if not r.passed]
    assert not failures, (
        "EnergyJudge parsing failures:\n"
        + "\n".join(f"  [{r.case_id}] {', '.join(r.failures)}" for r in failures)
    )
