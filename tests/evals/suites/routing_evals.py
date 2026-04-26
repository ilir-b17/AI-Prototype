"""
Routing eval suite - tests _assess_request_route and supporting methods.

Tier 1: pure deterministic routing decisions
Tier 2: tool scoring with real SkillRegistry schemas
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tests.evals.framework.base import EvalCase, EvalSuite, EvalTier, SuiteResult
from tests.evals.fixtures.routing_cases import ALL_ROUTING_CASES


class RoutingEvalSuite(EvalSuite):
    """Evaluates AIDEN's fast-path routing decisions."""

    name = "routing"
    tier = EvalTier.HEURISTIC  # needs SkillRegistry

    def __init__(self, orchestrator_stub):
        """
        orchestrator_stub: object with cognitive_router.registry populated
        (see conftest.routing_orchestrator fixture).
        """
        self._orch = orchestrator_stub

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c["id"],
                description=c["message"][:80],
                tier=EvalTier.HEURISTIC,
                tags=c["tags"],
                input={"message": c["message"]},
                expected={
                    "mode": c["expected_mode"],
                    "tool": c.get("expected_tool"),
                    "tickers": c.get("expected_tickers", []),
                    "excluded_tool": c.get("excluded_tool"),
                },
            )
            for c in ALL_ROUTING_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        message = case.input["message"]
        expected_mode = case.expected["mode"]
        expected_tool = case.expected.get("tool")
        expected_tickers = case.expected.get("tickers", [])
        excluded_tool = case.expected.get("excluded_tool")

        assessment = self._orch._assess_request_route(message)
        actual_mode = assessment.get("mode", "unknown")
        actual_tool = assessment.get("tool_name")
        actual_tickers = assessment.get("tickers", [])

        failures: List[str] = []

        # Mode check
        if actual_mode != expected_mode:
            failures.append(
                f"mode: expected {expected_mode!r}, got {actual_mode!r}"
            )

        # Tool check (only for single_tool mode)
        if expected_mode == "single_tool" and expected_tool is not None:
            if actual_tool != expected_tool:
                failures.append(
                    f"tool: expected {expected_tool!r}, got {actual_tool!r}"
                )

        # Excluded tool check
        if excluded_tool is not None and actual_tool == excluded_tool:
            failures.append(
                f"excluded_tool: {excluded_tool!r} must NOT be selected"
            )

        # Ticker check (only for multi_ticker mode)
        if expected_mode == "multi_ticker" and expected_tickers:
            for ticker in expected_tickers:
                if ticker not in actual_tickers:
                    failures.append(
                        f"tickers: expected {ticker!r} in {actual_tickers!r}"
                    )

        actual = {
            "mode": actual_mode,
            "tool": actual_tool,
            "tickers": actual_tickers,
            "complexity": assessment.get("complexity"),
        }
        return actual, failures


class ComplexityEvalSuite(EvalSuite):
    """Evaluates complexity scoring.

    Complexity must be monotonically higher for harder requests.
    No single expected value - we check ordering relationships.
    """

    name = "complexity"
    tier = EvalTier.DETERMINISTIC

    _ORDERING_CASES = [
        # (id, lower_complexity_msg, higher_complexity_msg, description)
        (
            "cx_001",
            "Hi",
            "Analyse my CSV, find outliers, and write a report",
            "greeting vs complex multi-step",
        ),
        (
            "cx_002",
            "What time is it?",
            "Build a stock screening pipeline with email alerts and tax loss harvesting",
            "simple query vs complex build",
        ),
        (
            "cx_003",
            "TSLA price",
            "Compare TSLA and AAPL over the last quarter and suggest a rebalancing strategy",
            "ticker lookup vs analysis",
        ),
        (
            "cx_004",
            "What is Python?",
            "Refactor the orchestrator to extract bounded contexts, add tests, and commit",
            "definition vs multi-step code task",
        ),
    ]

    def __init__(self, orchestrator_stub):
        self._orch = orchestrator_stub

    def _estimate(self, message: str) -> int:
        # Compatibility for refactored routing code where complexity lives on
        # routing_assessor instead of Orchestrator.
        if hasattr(self._orch, "_estimate_request_complexity"):
            return int(self._orch._estimate_request_complexity(message))
        assessor = getattr(self._orch, "routing_assessor", None)
        if assessor is None and hasattr(self._orch, "cognitive_router"):
            from src.core.routing_assessor import RoutingAssessor

            assessor = RoutingAssessor(self._orch.cognitive_router.registry)
            self._orch.routing_assessor = assessor
        return int(assessor._estimate_request_complexity(message))

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c[0],
                description=c[3],
                tier=EvalTier.DETERMINISTIC,
                tags=["complexity", "ordering"],
                input={"low": c[1], "high": c[2]},
                expected={"low_lt_high": True},
            )
            for c in self._ORDERING_CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        low_msg = case.input["low"]
        high_msg = case.input["high"]
        low_score = self._estimate(low_msg)
        high_score = self._estimate(high_msg)
        failures = []
        if low_score >= high_score:
            failures.append(
                f"Expected complexity({low_msg!r}) < complexity({high_msg!r}), "
                f"got {low_score} >= {high_score}"
            )
        return {"low": low_score, "high": high_score}, failures


class TickerExtractionEvalSuite(EvalSuite):
    """Evaluates ticker extraction - no SkillRegistry needed."""

    name = "ticker_extraction"
    tier = EvalTier.DETERMINISTIC

    _CASES = [
        ("tk_001", "What are AAPL and MSFT prices?", ["AAPL", "MSFT"], "two tickers"),
        ("tk_002", "Compare TSLA NVDA AMZN", ["TSLA", "NVDA", "AMZN"], "three tickers"),
        ("tk_003", "AAPL price", [], "single ticker - not multi"),
        ("tk_004", "What is the weather today?", [], "no tickers"),
        ("tk_005", "I am a trader looking at A and B", [], "stopwords only"),
        ("tk_006", "Stock quotes for GOOG META AMZN AAPL", ["GOOG", "META", "AMZN", "AAPL"], "four tickers"),
        ("tk_007", "Show me TSLA and tesla prices", [], "mixed case - has lowercase duplicate"),
    ]

    def __init__(self, orchestrator_stub):
        self._orch = orchestrator_stub

    def cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id=c[0],
                description=c[3],
                tier=EvalTier.DETERMINISTIC,
                tags=["ticker_extraction"],
                input={"message": c[1]},
                expected={"tickers": c[2]},
            )
            for c in self._CASES
        ]

    def run_case(self, case: EvalCase) -> Tuple[Dict[str, Any], List[str]]:
        message = case.input["message"]
        expected_tickers = case.expected["tickers"]
        actual_tickers = self._orch._extract_multiple_tickers(message)
        failures = []
        if set(actual_tickers) != set(expected_tickers):
            failures.append(
                f"tickers: expected {sorted(expected_tickers)}, "
                f"got {sorted(actual_tickers)}"
            )
        return {"tickers": actual_tickers}, failures


def run_all_routing_suites(orchestrator_stub) -> List[SuiteResult]:
    """Run all routing-related eval suites and return results."""
    return [
        RoutingEvalSuite(orchestrator_stub).run(),
        ComplexityEvalSuite(orchestrator_stub).run(),
        TickerExtractionEvalSuite(orchestrator_stub).run(),
    ]
