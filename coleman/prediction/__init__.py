"""Independent versioned online prediction, without an optimizer or transport."""

from coleman.prediction.contracts import (
    RidgeConfig,
    RidgeFailure,
    RidgeInput,
    RidgeLimits,
    RidgePayload,
    RidgePrediction,
    RidgeSample,
    RidgeUpdate,
)
from coleman.prediction.ridge import AppendOnlyRidge

__all__ = [
    "AppendOnlyRidge",
    "RidgeConfig",
    "RidgeFailure",
    "RidgeInput",
    "RidgeLimits",
    "RidgePayload",
    "RidgePrediction",
    "RidgeSample",
    "RidgeUpdate",
]
