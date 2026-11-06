"""Library-first protected temporal evaluation and versioned paired reports.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.evaluation.protected.audit import AccessReceipt, EvaluationAudit
    from coleman.evaluation.protected.bounded_inference import BoundedMeanProtocol
    from coleman.evaluation.protected.comparison import FamilyReport, evaluate
    from coleman.evaluation.protected.contracts import (
        ComparisonHypothesis,
        EvaluationProtocol,
        FrozenCandidate,
        FrozenEvaluation,
        freeze,
        named_seed,
    )
    from coleman.evaluation.protected.inference import holm_adjust
    from coleman.evaluation.protected.observations import ObservationReference, PairedObservation
    from coleman.evaluation.protected.temporal import SplitDecision, TemporalObservation, TemporalPlan

_EXPORTS = {
    "AccessReceipt": ("coleman.evaluation.protected.audit", "AccessReceipt"),
    "BoundedMeanProtocol": ("coleman.evaluation.protected.bounded_inference", "BoundedMeanProtocol"),
    "ComparisonHypothesis": ("coleman.evaluation.protected.contracts", "ComparisonHypothesis"),
    "EvaluationAudit": ("coleman.evaluation.protected.audit", "EvaluationAudit"),
    "EvaluationProtocol": ("coleman.evaluation.protected.contracts", "EvaluationProtocol"),
    "FamilyReport": ("coleman.evaluation.protected.comparison", "FamilyReport"),
    "FrozenCandidate": ("coleman.evaluation.protected.contracts", "FrozenCandidate"),
    "FrozenEvaluation": ("coleman.evaluation.protected.contracts", "FrozenEvaluation"),
    "ObservationReference": ("coleman.evaluation.protected.observations", "ObservationReference"),
    "PairedObservation": ("coleman.evaluation.protected.observations", "PairedObservation"),
    "SplitDecision": ("coleman.evaluation.protected.temporal", "SplitDecision"),
    "TemporalObservation": ("coleman.evaluation.protected.temporal", "TemporalObservation"),
    "TemporalPlan": ("coleman.evaluation.protected.temporal", "TemporalPlan"),
    "evaluate": ("coleman.evaluation.protected.comparison", "evaluate"),
    "freeze": ("coleman.evaluation.protected.contracts", "freeze"),
    "holm_adjust": ("coleman.evaluation.protected.inference", "holm_adjust"),
    "named_seed": ("coleman.evaluation.protected.contracts", "named_seed"),
}

__all__ = [
    "AccessReceipt",
    "BoundedMeanProtocol",
    "ComparisonHypothesis",
    "EvaluationAudit",
    "EvaluationProtocol",
    "FamilyReport",
    "FrozenCandidate",
    "FrozenEvaluation",
    "ObservationReference",
    "PairedObservation",
    "SplitDecision",
    "TemporalObservation",
    "TemporalPlan",
    "evaluate",
    "freeze",
    "holm_adjust",
    "named_seed",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
