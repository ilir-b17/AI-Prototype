"""
Baseline snapshot management for regression detection.

A snapshot is a JSON file capturing SuiteResult pass rates at a known
good point. Running evals against a snapshot detects regressions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

SNAPSHOTS_DIR = Path(__file__).resolve().parents[1] / "results" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_snapshot(suite_name: str, pass_rate: float) -> Path:
    """Save current pass rate as the baseline for a suite."""
    data = {
        "suite_name": suite_name,
        "pass_rate": pass_rate,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = SNAPSHOTS_DIR / f"{suite_name}_baseline.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_snapshot(suite_name: str) -> Optional[Dict]:
    """Load the baseline snapshot for a suite, or None if absent."""
    path = SNAPSHOTS_DIR / f"{suite_name}_baseline.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_regression(
    suite_name: str,
    current_pass_rate: float,
    tolerance: float = 0.05,
) -> Optional[str]:
    """Return a regression message if pass rate dropped below baseline
    minus tolerance, else None."""
    baseline = load_snapshot(suite_name)
    if baseline is None:
        return None
    baseline_rate = float(baseline.get("pass_rate", 0.0))
    threshold = baseline_rate - tolerance
    if current_pass_rate < threshold:
        return (
            f"REGRESSION in {suite_name}: "
            f"pass rate dropped from {baseline_rate:.1%} to "
            f"{current_pass_rate:.1%} "
            f"(threshold {threshold:.1%})"
        )
    return None
