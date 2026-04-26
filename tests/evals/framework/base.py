"""
AIDEN Evaluation Framework - Base types and runner.

Three eval tiers:
  Tier 1: deterministic (no LLM, no IO)
  Tier 2: heuristic integration (SkillRegistry OK, no LLM)
  Tier 3: LLM-graded quality (requires Ollama)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EvalTier(Enum):
    DETERMINISTIC = 1    # pure functions, no IO
    HEURISTIC = 2        # SkillRegistry OK, no LLM
    LLM_GRADED = 3       # requires Ollama


class EvalStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    description: str
    tier: EvalTier
    tags: List[str] = field(default_factory=list)
    # Input to the function under test (free-form)
    input: Dict[str, Any] = field(default_factory=dict)
    # What we expect (free-form, suite-specific)
    expected: Dict[str, Any] = field(default_factory=dict)
    # Optional: context needed to set up the test
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of running one EvalCase."""
    case_id: str
    suite: str
    status: EvalStatus
    score: float           # 0.0 (total fail) to 1.0 (perfect)
    actual: Dict[str, Any]
    expected: Dict[str, Any]
    failures: List[str]    # human-readable failure reasons
    duration_ms: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        d["tier"] = None  # filled by suite
        return d

    @property
    def passed(self) -> bool:
        return self.status == EvalStatus.PASS


@dataclass
class SuiteResult:
    """Aggregated results for one eval suite."""
    suite_name: str
    tier: EvalTier
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_ms: int
    results: List[EvalResult]
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    @property
    def pass_rate(self) -> float:
        eligible = self.total - self.skipped
        return self.passed / eligible if eligible else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "tier": self.tier.name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "pass_rate": round(self.pass_rate, 4),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, name_suffix: str = "") -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{name_suffix}" if name_suffix else ""
        filename = f"{self.suite_name}{suffix}_{ts}.json"
        path = RESULTS_DIR / filename
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return path


# Type alias for eval check functions
CheckFn = Callable[
    [EvalCase, Any],      # (case, actual_output) -> list of failure reasons
    List[str],
]


class EvalSuite:
    """Base class for all eval suites.

    Subclasses implement:
      - cases() -> List[EvalCase]
      - run_case(case) -> (actual_output, failures)
    """

    name: str = "base"
    tier: EvalTier = EvalTier.DETERMINISTIC

    def cases(self) -> List[EvalCase]:
        raise NotImplementedError

    def run_case(self, case: EvalCase) -> tuple[Any, List[str]]:
        """Run one case. Returns (actual_output, failure_reasons).
        failure_reasons is empty on pass."""
        raise NotImplementedError

    def run(
        self,
        filter_tags: Optional[List[str]] = None,
        filter_ids: Optional[List[str]] = None,
    ) -> SuiteResult:
        """Run all cases and return aggregated SuiteResult."""
        cases = self.cases()
        if filter_tags:
            cases = [c for c in cases if any(t in c.tags for t in filter_tags)]
        if filter_ids:
            cases = [c for c in cases if c.id in filter_ids]

        results: List[EvalResult] = []
        suite_start = time.perf_counter()

        for case in cases:
            start = time.perf_counter()
            try:
                actual, failures = self.run_case(case)
                duration_ms = int((time.perf_counter() - start) * 1000)
                if failures:
                    status = EvalStatus.FAIL
                    score = max(0.0, 1.0 - (len(failures) / max(1, len(case.expected))))
                else:
                    status = EvalStatus.PASS
                    score = 1.0
                result = EvalResult(
                    case_id=case.id,
                    suite=self.name,
                    status=status,
                    score=score,
                    actual=actual if isinstance(actual, dict) else {"value": actual},
                    expected=case.expected,
                    failures=failures,
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                duration_ms = int((time.perf_counter() - start) * 1000)
                logger.error("Eval case %s raised: %s", case.id, exc, exc_info=True)
                result = EvalResult(
                    case_id=case.id,
                    suite=self.name,
                    status=EvalStatus.ERROR,
                    score=0.0,
                    actual={},
                    expected=case.expected,
                    failures=[f"Exception: {exc}"],
                    duration_ms=duration_ms,
                    error=str(exc),
                )
            results.append(result)

        suite_duration = int((time.perf_counter() - suite_start) * 1000)
        return SuiteResult(
            suite_name=self.name,
            tier=self.tier,
            total=len(results),
            passed=sum(1 for r in results if r.status == EvalStatus.PASS),
            failed=sum(1 for r in results if r.status == EvalStatus.FAIL),
            skipped=sum(1 for r in results if r.status == EvalStatus.SKIP),
            errors=sum(1 for r in results if r.status == EvalStatus.ERROR),
            duration_ms=suite_duration,
            results=results,
        )

    def _check_exact(self, key: str, actual: Any, expected: Any) -> Optional[str]:
        """Return failure message if actual[key] != expected[key], else None."""
        if actual != expected:
            return f"{key}: expected {expected!r}, got {actual!r}"
        return None

    def _check_in(self, label: str, item: Any, collection: Any) -> Optional[str]:
        if item not in collection:
            return f"{label}: {item!r} not in {collection!r}"
        return None

    def _check_not_in(self, label: str, item: Any, collection: Any) -> Optional[str]:
        if item in collection:
            return f"{label}: {item!r} unexpectedly found in {collection!r}"
        return None
