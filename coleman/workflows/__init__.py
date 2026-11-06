"""Typed reproducible CI planning, independent of command and model transports.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.workflows.contracts import (
        DatasetManifest,
        ExperimentPlan,
        PlanningRequest,
        ScheduleBudget,
        WorkflowLimits,
    )
    from coleman.workflows.planning import make_plan

_EXPORTS = {
    "DatasetManifest": ("coleman.workflows.contracts", "DatasetManifest"),
    "ExperimentPlan": ("coleman.workflows.contracts", "ExperimentPlan"),
    "PlanningRequest": ("coleman.workflows.contracts", "PlanningRequest"),
    "ScheduleBudget": ("coleman.workflows.contracts", "ScheduleBudget"),
    "WorkflowLimits": ("coleman.workflows.contracts", "WorkflowLimits"),
    "make_plan": ("coleman.workflows.planning", "make_plan"),
}

__all__ = ["DatasetManifest", "ExperimentPlan", "PlanningRequest", "ScheduleBudget", "WorkflowLimits", "make_plan"]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
