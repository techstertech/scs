"""Describe the evidence actually assessed by a bounded investigation."""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict, cast


class CoverageStatus(StrEnum):
    ASSESSED = "assessed"
    EMPTY = "empty"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    NOT_ASSESSED = "not_assessed"


class AspectCoverage(TypedDict):
    status: CoverageStatus
    evidence_count: int
    stages: list[str]
    reason: str


ASPECTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "implementation": (("symbols", "files"), ("search", "inspect", "inventory", "traversal", "impact")),
    "dependents": (("symbols",), ("impact",)),
    "tests": (("test_targets",), ("impact",)),
    "references": (("references",), ("references",)),
    "contracts": (("contracts",), ("contracts",)),
}
REASONS = {
    CoverageStatus.ASSESSED: "The executed stages returned static evidence; correctness is not established.",
    CoverageStatus.EMPTY: "The executed stages returned no evidence within their configured scope.",
    CoverageStatus.PARTIAL: "Evidence is limited by a failed, degraded, or truncated stage.",
    CoverageStatus.UNAVAILABLE: "The required stages could not provide usable evidence.",
    CoverageStatus.NOT_ASSESSED: "No stage assessed this aspect; a separate investigation is needed.",
}


def investigation_coverage(
    evidence: dict[str, list[dict[str, object]]], trace: list[dict[str, object]],
) -> dict[str, AspectCoverage]:
    """Keep empty results distinct from missing, partial, and failed execution.

    Stage identity is essential: finding a test filename during discovery does
    not mean dependency-based test selection ran. This report does not infer
    requested behavior from prose or equate static evidence with runtime coverage.
    """

    report: dict[str, AspectCoverage] = {}
    for aspect, (categories, stage_names) in ASPECTS.items():
        executed = [item for item in trace if item.get("stage") in stage_names]
        count = sum(
            1 for category in categories for item in evidence.get(category, [])
            if item.get("kind") != "unresolved_range"
        )
        if aspect == "dependents":
            count = 0
            for item in executed:
                counts = item.get("counts")
                if isinstance(counts, dict):
                    value = cast(dict[str, object], counts).get("dependents", 0)
                    if type(value) is int:
                        count += value
        usable = [item for item in executed if item.get("status") == "ok"]
        limited = any(item.get("truncated") is True or item.get("status") != "ok" for item in executed)
        if not executed:
            status = CoverageStatus.NOT_ASSESSED
        elif not usable and not count:
            status = CoverageStatus.UNAVAILABLE
        elif limited:
            status = CoverageStatus.PARTIAL
        elif count:
            status = CoverageStatus.ASSESSED
        else:
            status = CoverageStatus.EMPTY
        report[aspect] = {
            "status": status, "evidence_count": count,
            "stages": list(dict.fromkeys(str(item["stage"]) for item in executed)),
            "reason": REASONS[status],
        }
    return report
