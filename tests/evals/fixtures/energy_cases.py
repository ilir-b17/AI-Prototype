"""
Ground-truth test cases for the energy gate system.

Cases cover:
  - Normal approve/defer decisions
  - Fairness boost behavior (high defer_count)
  - Force-execute bypass (max defer count)
  - Reserve floor enforcement
  - ROI threshold boundary conditions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_REASON_APPROVED = "Approved"
_REASON_ROI_LOW = "ROI too low"
_REASON_RESERVE = "reserve"
_REASON_BYPASS = "bypass"


def _ecase(
    id: str,
    effort: int,
    value: int,
    available: int,
    defer_count: int,
    expected_execute: bool,
    expected_reason_contains: str,
    description: str,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": id,
        "effort": effort,
        "value": value,
        "available": available,
        "defer_count": defer_count,
        "expected_execute": expected_execute,
        "expected_reason_contains": expected_reason_contains,
        "description": description,
        "tags": tags or [],
    }


# Default policy from env: ROI_THRESHOLD=1.25, MIN_RESERVE=10,
# EFFORT_MULTIPLIER=3, FAIRNESS_BOOST=0.15, MAX_DEFER=5

ENERGY_CASES = [
    # -- Clear approve cases ----------------------------------------------------
    _ecase(
        "energy_001",
        effort=3,
        value=9,
        available=100,
        defer_count=0,
        expected_execute=True,
        expected_reason_contains=_REASON_APPROVED,
        description="High value, low effort, full budget",
        tags=["approve"],
    ),
    _ecase(
        "energy_002",
        effort=5,
        value=7,
        available=80,
        defer_count=0,
        expected_execute=True,
        expected_reason_contains=_REASON_APPROVED,
        description="Moderate ROI = 1.4 > 1.25 threshold",
        tags=["approve"],
    ),
    _ecase(
        "energy_003",
        effort=1,
        value=2,
        available=50,
        defer_count=0,
        expected_execute=True,
        expected_reason_contains=_REASON_APPROVED,
        description="ROI = 2.0, well above threshold",
        tags=["approve"],
    ),

    # -- Clear defer cases ------------------------------------------------------
    _ecase(
        "energy_004",
        effort=5,
        value=5,
        available=100,
        defer_count=0,
        expected_execute=False,
        expected_reason_contains=_REASON_ROI_LOW,
        description="ROI = 1.0 < 1.25 threshold",
        tags=["defer", "roi"],
    ),
    _ecase(
        "energy_005",
        effort=8,
        value=9,
        available=100,
        defer_count=0,
        expected_execute=False,
        expected_reason_contains=_REASON_ROI_LOW,
        description="ROI = 1.125 < 1.25 threshold",
        tags=["defer", "roi"],
    ),
    _ecase(
        "energy_006",
        effort=1,
        value=1,
        available=100,
        defer_count=0,
        expected_execute=False,
        expected_reason_contains=_REASON_ROI_LOW,
        description="ROI = 1.0, minimum possible",
        tags=["defer", "roi"],
    ),

    # -- Reserve floor cases ----------------------------------------------------
    _ecase(
        "energy_007",
        effort=10,
        value=10,
        available=25,
        defer_count=0,
        expected_execute=False,
        expected_reason_contains=_REASON_ROI_LOW,
        description="Both ROI and reserve fail; engine reports ROI first by policy order",
        tags=["defer", "roi", "reserve"],
    ),
    _ecase(
        "energy_008",
        effort=3,
        value=9,
        available=15,
        defer_count=0,
        expected_execute=False,
        expected_reason_contains=_REASON_RESERVE,
        description="Budget 15, cost=9, reserve_after=6 < 10 floor",
        tags=["defer", "reserve"],
    ),
    _ecase(
        "energy_009",
        effort=2,
        value=9,
        available=20,
        defer_count=0,
        expected_execute=True,
        expected_reason_contains=_REASON_APPROVED,
        description="Budget 20, cost=6, reserve_after=14 >= 10 floor",
        tags=["approve", "reserve"],
    ),

    # -- Fairness boost cases ---------------------------------------------------
    _ecase(
        "energy_010",
        effort=5,
        value=5,
        available=100,
        defer_count=2,
        expected_execute=True,
        expected_reason_contains=_REASON_APPROVED,
        description="ROI=1.0 with defer boost 2*0.15 gives effective ROI=1.30 > 1.25",
        tags=["approve", "fairness"],
    ),
    _ecase(
        "energy_011",
        effort=6,
        value=6,
        available=100,
        defer_count=1,
        expected_execute=False,
        expected_reason_contains=_REASON_ROI_LOW,
        description="ROI=1.0 with defer boost 1*0.15 gives effective ROI=1.15 < 1.25",
        tags=["defer", "fairness"],
    ),
    _ecase(
        "energy_012",
        effort=8,
        value=9,
        available=100,
        defer_count=3,
        expected_execute=True,
        expected_reason_contains=_REASON_APPROVED,
        description="ROI=1.125 * (1 + 3*0.15) = 1.631 > 1.25",
        tags=["approve", "fairness"],
    ),

    # -- Force-execute bypass (max defer count) ---------------------------------
    _ecase(
        "energy_013",
        effort=5,
        value=5,
        available=100,
        defer_count=5,
        expected_execute=True,
        expected_reason_contains=_REASON_BYPASS,
        description="defer_count=MAX_DEFER=5 triggers force execute with sufficient reserve",
        tags=["approve", "force_execute"],
    ),
    _ecase(
        "energy_014",
        effort=5,
        value=5,
        available=12,
        defer_count=5,
        expected_execute=False,
        expected_reason_contains=_REASON_RESERVE,
        description="Force execute evaluated but insufficient reserve (12-15=-3 < 10)",
        tags=["defer", "force_execute", "reserve"],
    ),

    # -- Edge: zero effort ------------------------------------------------------
    _ecase(
        "energy_015",
        effort=0,
        value=5,
        available=100,
        defer_count=0,
        expected_execute=True,
        expected_reason_contains=_REASON_APPROVED,
        description="Zero effort: denominator coerced to 1, ROI=5.0",
        tags=["edge", "approve"],
    ),

    # -- Edge: zero available budget -------------------------------------------
    _ecase(
        "energy_016",
        effort=1,
        value=10,
        available=0,
        defer_count=0,
        expected_execute=False,
        expected_reason_contains=_REASON_RESERVE,
        description="Zero budget: predicted_cost=3, reserve=-3 < 10",
        tags=["edge", "defer", "reserve"],
    ),
]
