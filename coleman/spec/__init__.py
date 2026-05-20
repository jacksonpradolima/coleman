"""
coleman.spec - Typed run specification models.

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
"""

from coleman.spec.io import load_spec, save_resolved
from coleman.spec.models import (
    AlgorithmSpec,
    CheckpointSpec,
    ContextualInformationSpec,
    ExecutionSpec,
    ExperimentSpec,
    HCSConfigurationSpec,
    ResultsSpec,
    RunSpec,
    TelemetrySpec,
)
from coleman.spec.packs import resolve_packs
from coleman.spec.run_id import compute_run_id
from coleman.spec.sweep import SweepAxis, SweepSpec, expand_sweep

__all__ = [
    "AlgorithmSpec",
    "CheckpointSpec",
    "ContextualInformationSpec",
    "ExecutionSpec",
    "ExperimentSpec",
    "HCSConfigurationSpec",
    "ResultsSpec",
    "RunSpec",
    "SweepAxis",
    "SweepSpec",
    "TelemetrySpec",
    "compute_run_id",
    "expand_sweep",
    "load_spec",
    "resolve_packs",
    "save_resolved",
]
