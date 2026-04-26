#!/usr/bin/env python3
"""
AIDEN Eval Runner - CLI entry point for the evaluation framework.

Usage:
  python scripts/run_evals.py                    # run all fast evals
  python scripts/run_evals.py --suite routing    # run one suite
  python scripts/run_evals.py --all              # include slow/Tier 3
  python scripts/run_evals.py --baseline         # capture baseline snapshots
  python scripts/run_evals.py --tags weather     # filter by tag
  python scripts/run_evals.py --check-regression # fail on regression
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FAST_PASS_THRESHOLD = 0.80
QUALITY_PASS_THRESHOLD = 0.70
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

FAST_SUITE_NAMES = {
    "routing",
    "complexity",
    "ticker_extraction",
    "energy_roi",
    "energy_judge_parsing",
    "moral_payload",
    "triviality_bypass",
    "moral_severity",
}
QUALITY_SUITE_NAMES = {"response_quality", "quality"}


def _normalize_suite_name(raw_name: Optional[str]) -> Optional[str]:
    if not raw_name:
        return None
    normalized = str(raw_name).strip().lower()
    if normalized == "quality":
        return "response_quality"
    return normalized


def _build_routing_orchestrator():
    """Build a minimal orchestrator stub for Tier 1/2 eval suites."""
    from src.core.orchestrator import Orchestrator
    from src.core.routing_assessor import RoutingAssessor
    from src.core.skill_manager import SkillRegistry

    registry = SkillRegistry()
    assessor = RoutingAssessor(registry)

    stub_router = types.SimpleNamespace(registry=registry)
    orch = object.__new__(Orchestrator)
    orch.cognitive_router = stub_router
    orch.routing_assessor = assessor
    orch._intent_classification_cache = {}

    # Keep parity with eval suite expectations that call legacy helpers directly.
    orch._assess_request_route = lambda message, chat_history=None: assessor.assess_request_route(
        message,
        chat_history,
    )
    orch._estimate_request_complexity = assessor._estimate_request_complexity
    orch._extract_multiple_tickers = assessor._extract_multiple_tickers
    return orch


def _get_fast_suites(suite_filter: Optional[str] = None):
    from tests.evals.suites.energy_evals import (
        EnergyJudgeParsingEvalSuite,
        EnergyROIEvalSuite,
    )
    from tests.evals.suites.moral_evals import (
        MoralPayloadEvalSuite,
        MoralSeverityEvalSuite,
        TrivialityBypassEvalSuite,
    )
    from tests.evals.suites.routing_evals import (
        ComplexityEvalSuite,
        RoutingEvalSuite,
        TickerExtractionEvalSuite,
    )

    selected = set(FAST_SUITE_NAMES)
    if suite_filter:
        selected = {suite_filter}

    need_orch = bool(
        selected
        & {
            "routing",
            "complexity",
            "ticker_extraction",
            "triviality_bypass",
            "moral_severity",
        }
    )
    orch = _build_routing_orchestrator() if need_orch else None

    suites = []
    if "routing" in selected:
        suites.append(RoutingEvalSuite(orch))
    if "complexity" in selected:
        suites.append(ComplexityEvalSuite(orch))
    if "ticker_extraction" in selected:
        suites.append(TickerExtractionEvalSuite(orch))
    if "energy_roi" in selected:
        suites.append(EnergyROIEvalSuite())
    if "energy_judge_parsing" in selected:
        suites.append(EnergyJudgeParsingEvalSuite())
    if "moral_payload" in selected:
        suites.append(MoralPayloadEvalSuite())
    if "triviality_bypass" in selected:
        suites.append(TrivialityBypassEvalSuite(orch))
    if "moral_severity" in selected:
        suites.append(MoralSeverityEvalSuite(orch))

    return suites


def _print_suite_result_line(result: Any, threshold: float) -> bool:
    if result.total == 0:
        print(
            f"  [SKIP] {result.suite_name:<25} "
            f"no matching cases"
        )
        return True

    passed = result.pass_rate >= threshold
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] {result.suite_name:<25} "
        f"{result.passed}/{result.total} passed "
        f"({result.pass_rate:.0%}) "
        f"in {result.duration_ms}ms"
    )

    failures = [r for r in result.results if not r.passed]
    for failure in failures[:5]:
        detail = ", ".join(failure.failures[:2])
        print(f"         - [{failure.case_id}] {detail}")

    return passed


def run_fast_suites(
    *,
    suite_filter: Optional[str] = None,
    filter_tags: Optional[list[str]] = None,
    check_regression: bool = False,
) -> tuple[list[Any], bool]:
    """Run Tier 1 + Tier 2 suites."""
    from tests.evals.framework.snapshot import check_regression as check_reg

    suites = _get_fast_suites(suite_filter=suite_filter)
    all_results = []
    all_passed = True

    for suite in suites:
        result = suite.run(filter_tags=filter_tags)
        result.save()
        all_results.append(result)

        suite_passed = _print_suite_result_line(result, FAST_PASS_THRESHOLD)
        all_passed = all_passed and suite_passed

        if check_regression and result.total > 0:
            regression = check_reg(result.suite_name, result.pass_rate)
            if regression:
                print(f"         REGRESSION: {regression}")
                all_passed = False

    return all_results, all_passed


def capture_baselines(
    *,
    suite_filter: Optional[str] = None,
    filter_tags: Optional[list[str]] = None,
) -> bool:
    """Run fast suites and save pass-rate baselines."""
    from tests.evals.framework.snapshot import save_snapshot

    print("Capturing eval baselines...")
    results, passed = run_fast_suites(
        suite_filter=suite_filter,
        filter_tags=filter_tags,
        check_regression=False,
    )

    for result in results:
        if result.total == 0:
            continue
        path = save_snapshot(result.suite_name, result.pass_rate)
        print(f"  Saved: {path.name} ({result.pass_rate:.1%})")

    print("Baselines captured.")
    return passed


def _quality_prerequisites_met(model: str) -> tuple[bool, str]:
    from tests.evals.framework.grader import is_ollama_available

    if not is_ollama_available(model):
        return False, f"Ollama not running or model {model!r} unavailable"

    if not str(os.getenv("MFA_PASSPHRASE", "")).strip():
        return False, "MFA_PASSPHRASE is not configured"

    return True, ""


def _close_quality_orchestrator(loop: asyncio.AbstractEventLoop, orchestrator: Any) -> None:
    try:
        if hasattr(orchestrator, "cognitive_router") and orchestrator.cognitive_router:
            loop.run_until_complete(orchestrator.cognitive_router.close())
    except Exception:
        logger.exception("Failed to close quality eval cognitive router")

    try:
        if hasattr(orchestrator, "ledger_memory") and orchestrator.ledger_memory:
            loop.run_until_complete(orchestrator.ledger_memory.close())
    except Exception:
        logger.exception("Failed to close quality eval ledger")

    try:
        orchestrator.close()
    except Exception:
        logger.exception("Failed to close quality eval orchestrator")


def run_quality_suite(
    *,
    filter_tags: Optional[list[str]] = None,
    check_regression: bool = False,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> tuple[Optional[Any], bool]:
    from tests.evals.framework.snapshot import check_regression as check_reg
    from tests.evals.suites.quality_evals import ResponseQualityEvalSuite

    prereq_ok, reason = _quality_prerequisites_met(model)
    if not prereq_ok:
        print(f"  [SKIP] response_quality           {reason}")
        return None, True

    from src.core.orchestrator import Orchestrator

    tmp_dir = tempfile.mkdtemp(prefix="aiden_eval_")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    orchestrator = None
    try:
        orchestrator = Orchestrator(
            vector_db_path=str(Path(tmp_dir) / "chroma"),
            ledger_db_path=str(Path(tmp_dir) / "ledger.db"),
            core_memory_path=str(Path(tmp_dir) / "core_memory.json"),
        )
        orchestrator._eval_loop = loop
        loop.run_until_complete(orchestrator.async_init())

        suite = ResponseQualityEvalSuite(orchestrator, local_model=model)
        result = suite.run(filter_tags=filter_tags)
        result.save()

        passed = _print_suite_result_line(result, QUALITY_PASS_THRESHOLD)

        if check_regression and result.total > 0:
            regression = check_reg(result.suite_name, result.pass_rate, tolerance=0.10)
            if regression:
                print(f"         REGRESSION: {regression}")
                passed = False

        return result, passed
    finally:
        if orchestrator is not None:
            _close_quality_orchestrator(loop, orchestrator)

        try:
            if not loop.is_closed():
                loop.close()
        except Exception:
            logger.exception("Failed to close quality eval event loop")
        asyncio.set_event_loop(None)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _validate_suite_filter(parser: argparse.ArgumentParser, suite_filter: Optional[str]) -> None:
    if suite_filter is None:
        return

    known = FAST_SUITE_NAMES | {"response_quality"}
    if suite_filter not in known:
        options = ", ".join(sorted(known))
        parser.error(f"Unknown suite {suite_filter!r}. Choose one of: {options}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIDEN Eval Runner")
    parser.add_argument("--suite", help="Run only this suite name")
    parser.add_argument("--all", action="store_true", help="Include Tier 3 slow evals")
    parser.add_argument("--baseline", action="store_true", help="Capture baseline snapshots")
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Fail if pass rate dropped below baseline",
    )
    parser.add_argument("--tags", nargs="*", help="Filter cases by tag")
    return parser


def _run_fast_section(args: argparse.Namespace, suite_filter: Optional[str]) -> bool:
    print("Fast suites (Tier 1 + Tier 2):")
    _, fast_passed = run_fast_suites(
        suite_filter=suite_filter,
        filter_tags=args.tags,
        check_regression=args.check_regression,
    )
    return fast_passed


def _run_quality_section(args: argparse.Namespace) -> bool:
    print("Slow suites (Tier 3 - requires Ollama):")
    _, quality_passed = run_quality_suite(
        filter_tags=args.tags,
        check_regression=args.check_regression,
        model=DEFAULT_OLLAMA_MODEL,
    )
    return quality_passed


def _execute_from_args(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    suite_filter: Optional[str],
) -> int:
    print(f"\nAIDEN Eval Runner - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    overall_passed = True

    run_fast = suite_filter is None or suite_filter in FAST_SUITE_NAMES
    if run_fast:
        overall_passed = overall_passed and _run_fast_section(args, suite_filter)

    run_quality = args.all or suite_filter == "response_quality"
    if run_quality:
        if run_fast:
            print("")
        overall_passed = overall_passed and _run_quality_section(args)

    if not run_fast and not run_quality:
        parser.error("No suites selected to run")

    print("=" * 60)
    print(f"Overall: {'PASS' if overall_passed else 'FAIL'}")
    return 0 if overall_passed else 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    suite_filter = _normalize_suite_name(args.suite)
    _validate_suite_filter(parser, suite_filter)

    if args.baseline and suite_filter == "response_quality":
        parser.error("--baseline only supports fast suites (Tier 1 + Tier 2)")

    if args.baseline:
        passed = capture_baselines(
            suite_filter=suite_filter,
            filter_tags=args.tags,
        )
        sys.exit(0 if passed else 1)

    exit_code = _execute_from_args(
        parser=parser,
        args=args,
        suite_filter=suite_filter,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
