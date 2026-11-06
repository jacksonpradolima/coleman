"""Finite joint-scenario resource allocation with signed payoffs and conservative evidence handling.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.allocation.artifacts import allocation_record, publish_allocation_result
    from coleman.allocation.contracts import (
        AllocationProblem,
        CurrentAllocation,
        DiscreteAllocation,
        HoldingDefinition,
        JointScenario,
        NamedAmount,
        ProbabilityEvidence,
        ResourceCapacity,
        RiskConstraints,
        ScenarioPayoff,
        UtilitySpec,
    )
    from coleman.allocation.forecasts import ActionEvaluation, ActionForecast, evaluate_allocation_action
    from coleman.allocation.quantities import (
        ExactPayoff,
        NonnegativeMoment,
        SignedPayoff,
        penalty_semantics,
        variance_semantics,
    )
    from coleman.allocation.resources import AllocationLimits
    from coleman.allocation.results import AllocationCheckpoint, AllocationResult
    from coleman.allocation.solver import select_allocation

_EXPORTS = {
    "ActionEvaluation": ("coleman.allocation.forecasts", "ActionEvaluation"),
    "ActionForecast": ("coleman.allocation.forecasts", "ActionForecast"),
    "AllocationCheckpoint": ("coleman.allocation.results", "AllocationCheckpoint"),
    "AllocationLimits": ("coleman.allocation.resources", "AllocationLimits"),
    "AllocationProblem": ("coleman.allocation.contracts", "AllocationProblem"),
    "AllocationResult": ("coleman.allocation.results", "AllocationResult"),
    "CurrentAllocation": ("coleman.allocation.contracts", "CurrentAllocation"),
    "DiscreteAllocation": ("coleman.allocation.contracts", "DiscreteAllocation"),
    "ExactPayoff": ("coleman.allocation.quantities", "ExactPayoff"),
    "HoldingDefinition": ("coleman.allocation.contracts", "HoldingDefinition"),
    "JointScenario": ("coleman.allocation.contracts", "JointScenario"),
    "NamedAmount": ("coleman.allocation.contracts", "NamedAmount"),
    "NonnegativeMoment": ("coleman.allocation.quantities", "NonnegativeMoment"),
    "ProbabilityEvidence": ("coleman.allocation.contracts", "ProbabilityEvidence"),
    "ResourceCapacity": ("coleman.allocation.contracts", "ResourceCapacity"),
    "RiskConstraints": ("coleman.allocation.contracts", "RiskConstraints"),
    "ScenarioPayoff": ("coleman.allocation.contracts", "ScenarioPayoff"),
    "SignedPayoff": ("coleman.allocation.quantities", "SignedPayoff"),
    "UtilitySpec": ("coleman.allocation.contracts", "UtilitySpec"),
    "allocation_record": ("coleman.allocation.artifacts", "allocation_record"),
    "evaluate_allocation_action": ("coleman.allocation.forecasts", "evaluate_allocation_action"),
    "penalty_semantics": ("coleman.allocation.quantities", "penalty_semantics"),
    "publish_allocation_result": ("coleman.allocation.artifacts", "publish_allocation_result"),
    "select_allocation": ("coleman.allocation.solver", "select_allocation"),
    "variance_semantics": ("coleman.allocation.quantities", "variance_semantics"),
}

__all__ = [
    "ActionEvaluation",
    "ActionForecast",
    "AllocationCheckpoint",
    "AllocationLimits",
    "AllocationProblem",
    "AllocationResult",
    "CurrentAllocation",
    "DiscreteAllocation",
    "ExactPayoff",
    "HoldingDefinition",
    "JointScenario",
    "NamedAmount",
    "NonnegativeMoment",
    "ProbabilityEvidence",
    "ResourceCapacity",
    "RiskConstraints",
    "ScenarioPayoff",
    "SignedPayoff",
    "UtilitySpec",
    "allocation_record",
    "evaluate_allocation_action",
    "penalty_semantics",
    "publish_allocation_result",
    "select_allocation",
    "variance_semantics",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
