"""Point-in-time observations and versioned signals, demonstrated by offline CI/TCPCI adapters.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.observations.artifacts import publish_model, publish_signal
    from coleman.observations.contracts import (
        MAXIMUM_REVISIONS,
        DecisionSignal,
        ObservationHistory,
        ObservationRevision,
        ObservationSnapshot,
        SignalProvenance,
        SignalRequest,
    )
    from coleman.observations.mean import (
        EVENT_AGE_SECONDS,
        MeanEstimate,
        QuantityMeanModel,
        TrainingSet,
        fit_quantity_mean,
        prepare_training,
    )
    from coleman.observations.providers import ObservationProvider, OfflineObservationProvider

_EXPORTS = {
    "EVENT_AGE_SECONDS": ("coleman.observations.mean", "EVENT_AGE_SECONDS"),
    "MAXIMUM_REVISIONS": ("coleman.observations.contracts", "MAXIMUM_REVISIONS"),
    "DecisionSignal": ("coleman.observations.contracts", "DecisionSignal"),
    "MeanEstimate": ("coleman.observations.mean", "MeanEstimate"),
    "ObservationHistory": ("coleman.observations.contracts", "ObservationHistory"),
    "ObservationProvider": ("coleman.observations.providers", "ObservationProvider"),
    "ObservationRevision": ("coleman.observations.contracts", "ObservationRevision"),
    "ObservationSnapshot": ("coleman.observations.contracts", "ObservationSnapshot"),
    "OfflineObservationProvider": ("coleman.observations.providers", "OfflineObservationProvider"),
    "QuantityMeanModel": ("coleman.observations.mean", "QuantityMeanModel"),
    "SignalProvenance": ("coleman.observations.contracts", "SignalProvenance"),
    "SignalRequest": ("coleman.observations.contracts", "SignalRequest"),
    "TrainingSet": ("coleman.observations.mean", "TrainingSet"),
    "fit_quantity_mean": ("coleman.observations.mean", "fit_quantity_mean"),
    "prepare_training": ("coleman.observations.mean", "prepare_training"),
    "publish_model": ("coleman.observations.artifacts", "publish_model"),
    "publish_signal": ("coleman.observations.artifacts", "publish_signal"),
}

__all__ = [
    "EVENT_AGE_SECONDS",
    "MAXIMUM_REVISIONS",
    "DecisionSignal",
    "MeanEstimate",
    "ObservationHistory",
    "ObservationProvider",
    "ObservationRevision",
    "ObservationSnapshot",
    "OfflineObservationProvider",
    "QuantityMeanModel",
    "SignalProvenance",
    "SignalRequest",
    "TrainingSet",
    "fit_quantity_mean",
    "prepare_training",
    "publish_model",
    "publish_signal",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
