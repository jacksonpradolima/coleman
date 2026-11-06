"""Typed finite weighted coverage selection, exact certificates and native artifact adapters.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.selection.artifacts import coverage_record, publish_coverage_result
    from coleman.selection.contracts import (
        CoverageCandidate,
        CoverageConstraints,
        CoverageOutcome,
        CoverageProblem,
        Quantity,
        UnitSemantics,
    )
    from coleman.selection.resources import SearchLimits
    from coleman.selection.results import CoverageCheckpoint, CoverageResult
    from coleman.selection.solver import solve_coverage

_EXPORTS = {
    "CoverageCandidate": ("coleman.selection.contracts", "CoverageCandidate"),
    "CoverageCheckpoint": ("coleman.selection.results", "CoverageCheckpoint"),
    "CoverageConstraints": ("coleman.selection.contracts", "CoverageConstraints"),
    "CoverageOutcome": ("coleman.selection.contracts", "CoverageOutcome"),
    "CoverageProblem": ("coleman.selection.contracts", "CoverageProblem"),
    "CoverageResult": ("coleman.selection.results", "CoverageResult"),
    "Quantity": ("coleman.selection.contracts", "Quantity"),
    "SearchLimits": ("coleman.selection.resources", "SearchLimits"),
    "UnitSemantics": ("coleman.selection.contracts", "UnitSemantics"),
    "coverage_record": ("coleman.selection.artifacts", "coverage_record"),
    "publish_coverage_result": ("coleman.selection.artifacts", "publish_coverage_result"),
    "solve_coverage": ("coleman.selection.solver", "solve_coverage"),
}

__all__ = [
    "CoverageCandidate",
    "CoverageCheckpoint",
    "CoverageConstraints",
    "CoverageOutcome",
    "CoverageProblem",
    "CoverageResult",
    "Quantity",
    "SearchLimits",
    "UnitSemantics",
    "coverage_record",
    "publish_coverage_result",
    "solve_coverage",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
