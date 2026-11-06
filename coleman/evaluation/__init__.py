"""coleman.evaluation - Evaluation Metrics for Coleman.

This module provides classes and methods to evaluate the performance of the Coleman framework
in the context of test case prioritization. Various metrics such as NAPFD (Normalized Average
Percentage of Faults Detected) based on errors or verdicts can be utilized to measure the
effectiveness.

Classes
-------
EvaluationMetric
    Base class for all evaluation metrics.
    Defines basic attributes and methods used across all metrics.
NAPFDMetric
    Implements the NAPFD metric based on error counts.
NAPFDVerdictMetric
    Implements the NAPFD metric based on test verdicts (e.g., pass/fail).

Notes
-----
The ``evaluate`` method in ``EvaluationMetric`` is abstract and should be overridden in
child classes. Ensure that the ``reset`` method is called at the beginning of each
evaluation to reset metric values.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from .apfdc import APFDcMetric
    from .base import EvaluationMetric
    from .napfd import NAPFDMetric, NAPFDVerdictMetric
    from .topk import (
        AveragePrecisionAtKMetric,
        NDCGAtKMetric,
        PrecisionAtKMetric,
        RecallAtKMetric,
        ReciprocalRankAtKMetric,
    )

_EXPORTS = {
    "EvaluationMetric": ("coleman.evaluation.base", "EvaluationMetric"),
    "NAPFDMetric": ("coleman.evaluation.napfd", "NAPFDMetric"),
    "NAPFDVerdictMetric": ("coleman.evaluation.napfd", "NAPFDVerdictMetric"),
    "APFDcMetric": ("coleman.evaluation.apfdc", "APFDcMetric"),
    "PrecisionAtKMetric": ("coleman.evaluation.topk", "PrecisionAtKMetric"),
    "RecallAtKMetric": ("coleman.evaluation.topk", "RecallAtKMetric"),
    "AveragePrecisionAtKMetric": ("coleman.evaluation.topk", "AveragePrecisionAtKMetric"),
    "ReciprocalRankAtKMetric": ("coleman.evaluation.topk", "ReciprocalRankAtKMetric"),
    "NDCGAtKMetric": ("coleman.evaluation.topk", "NDCGAtKMetric"),
}

__all__ = [
    "EvaluationMetric",
    "NAPFDMetric",
    "NAPFDVerdictMetric",
    "APFDcMetric",
    "PrecisionAtKMetric",
    "RecallAtKMetric",
    "AveragePrecisionAtKMetric",
    "ReciprocalRankAtKMetric",
    "NDCGAtKMetric",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
