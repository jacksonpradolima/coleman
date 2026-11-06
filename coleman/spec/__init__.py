"""coleman.spec - Typed run specification models.

Pydantic v2 models that represent the full resolved run definition,
replacing ad-hoc TOML dict access with self-validating schemas.

Classes
-------
ExecutionSpec
    Execution-level settings (parallelism, repetitions).
ExperimentSpec
    Experiment-level settings (datasets, policies, rewards).
AlgorithmSpec
    Per-algorithm hyper-parameter block.
HCSConfigurationSpec
    HCS-specific flags.
ContextualInformationSpec
    Contextual feature configuration.
ResultsSpec
    Results sink settings.
CheckpointSpec
    Checkpoint persistence settings.
TelemetrySpec
    OpenTelemetry export settings.
HooksSpec
    Lifecycle hook plugin settings.
RunSpec
    Top-level experiment specification (composes all sub-specs).
SweepAxis
    A single sweep dimension (grid or zip).
SweepSpec
    Multi-axis sweep definition.

Functions
---------
compute_run_id
    Deterministic run identifier from a resolved ``RunSpec``.
load_spec
    Load a ``RunSpec`` from a YAML file with optional pack resolution.
save_resolved
    Persist a resolved ``RunSpec`` as canonical JSON.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.budget import BudgetMode
    from coleman.spec.io import load_spec, save_resolved
    from coleman.spec.models import (
        AlgorithmSpec,
        BudgetSpec,
        CheckpointSpec,
        ContextualInformationSpec,
        ExecutionSpec,
        ExperimentSpec,
        HCSConfigurationSpec,
        HooksSpec,
        ResultsSpec,
        RunSpec,
        TelemetrySpec,
    )
    from coleman.spec.packs import resolve_packs
    from coleman.spec.run_id import compute_run_id
    from coleman.spec.selection import resolve_requested_names
    from coleman.spec.sweep import SweepAxis, SweepSpec, expand_sweep

_EXPORTS = {
    "AlgorithmSpec": ("coleman.spec.models", "AlgorithmSpec"),
    "BudgetMode": ("coleman.budget", "BudgetMode"),
    "BudgetSpec": ("coleman.spec.models", "BudgetSpec"),
    "CheckpointSpec": ("coleman.spec.models", "CheckpointSpec"),
    "ContextualInformationSpec": ("coleman.spec.models", "ContextualInformationSpec"),
    "ExecutionSpec": ("coleman.spec.models", "ExecutionSpec"),
    "ExperimentSpec": ("coleman.spec.models", "ExperimentSpec"),
    "HCSConfigurationSpec": ("coleman.spec.models", "HCSConfigurationSpec"),
    "HooksSpec": ("coleman.spec.models", "HooksSpec"),
    "ResultsSpec": ("coleman.spec.models", "ResultsSpec"),
    "RunSpec": ("coleman.spec.models", "RunSpec"),
    "SweepAxis": ("coleman.spec.sweep", "SweepAxis"),
    "SweepSpec": ("coleman.spec.sweep", "SweepSpec"),
    "TelemetrySpec": ("coleman.spec.models", "TelemetrySpec"),
    "compute_run_id": ("coleman.spec.run_id", "compute_run_id"),
    "expand_sweep": ("coleman.spec.sweep", "expand_sweep"),
    "load_spec": ("coleman.spec.io", "load_spec"),
    "resolve_requested_names": ("coleman.spec.selection", "resolve_requested_names"),
    "resolve_packs": ("coleman.spec.packs", "resolve_packs"),
    "save_resolved": ("coleman.spec.io", "save_resolved"),
}

__all__ = [
    "AlgorithmSpec",
    "BudgetMode",
    "BudgetSpec",
    "CheckpointSpec",
    "ContextualInformationSpec",
    "ExecutionSpec",
    "ExperimentSpec",
    "HCSConfigurationSpec",
    "HooksSpec",
    "ResultsSpec",
    "RunSpec",
    "SweepAxis",
    "SweepSpec",
    "TelemetrySpec",
    "compute_run_id",
    "expand_sweep",
    "load_spec",
    "resolve_requested_names",
    "resolve_packs",
    "save_resolved",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
