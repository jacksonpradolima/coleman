"""Optional causal comparisons, reference conformance and conservative dispositions.

Pure contracts remain independent of transports and durable execution. Explicit
outer composition reuses the existing replay, owned state and protected audit.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from .comparison import ComparisonDisposition, ComparisonPlan, assess_comparison
    from .conformance import (
        MetricTolerance,
        ReferenceConformancePlan,
        ReferenceConformanceReport,
        check_reference_conformance,
    )
    from .contracts import (
        ComponentObservation,
        ConformanceLimits,
        ConformanceTrace,
        DomainExecutionReceipt,
        ExecutionKey,
        StepEvidence,
    )
    from .diagnostics import DiagnosticAudit
    from .evidence import DiagnosticEvidence, DiagnosticLocator, DiagnosticPanel, DiagnosticView
    from .execution import (
        DomainExecutionContext,
        DomainExecutor,
        DomainStepPlan,
        PrequentialStep,
        execute_prequential_step,
    )

_EXPORTS = {
    "ComparisonDisposition": ("coleman.evaluation.prequential.comparison", "ComparisonDisposition"),
    "ComparisonPlan": ("coleman.evaluation.prequential.comparison", "ComparisonPlan"),
    "assess_comparison": ("coleman.evaluation.prequential.comparison", "assess_comparison"),
    "MetricTolerance": ("coleman.evaluation.prequential.conformance", "MetricTolerance"),
    "ReferenceConformancePlan": ("coleman.evaluation.prequential.conformance", "ReferenceConformancePlan"),
    "ReferenceConformanceReport": ("coleman.evaluation.prequential.conformance", "ReferenceConformanceReport"),
    "check_reference_conformance": ("coleman.evaluation.prequential.conformance", "check_reference_conformance"),
    "ComponentObservation": ("coleman.evaluation.prequential.contracts", "ComponentObservation"),
    "ConformanceLimits": ("coleman.evaluation.prequential.contracts", "ConformanceLimits"),
    "ConformanceTrace": ("coleman.evaluation.prequential.contracts", "ConformanceTrace"),
    "DomainExecutionReceipt": ("coleman.evaluation.prequential.contracts", "DomainExecutionReceipt"),
    "ExecutionKey": ("coleman.evaluation.prequential.contracts", "ExecutionKey"),
    "StepEvidence": ("coleman.evaluation.prequential.contracts", "StepEvidence"),
    "DiagnosticAudit": ("coleman.evaluation.prequential.diagnostics", "DiagnosticAudit"),
    "DiagnosticEvidence": ("coleman.evaluation.prequential.evidence", "DiagnosticEvidence"),
    "DiagnosticLocator": ("coleman.evaluation.prequential.evidence", "DiagnosticLocator"),
    "DiagnosticPanel": ("coleman.evaluation.prequential.evidence", "DiagnosticPanel"),
    "DiagnosticView": ("coleman.evaluation.prequential.evidence", "DiagnosticView"),
    "DomainExecutionContext": ("coleman.evaluation.prequential.execution", "DomainExecutionContext"),
    "DomainExecutor": ("coleman.evaluation.prequential.execution", "DomainExecutor"),
    "DomainStepPlan": ("coleman.evaluation.prequential.execution", "DomainStepPlan"),
    "PrequentialStep": ("coleman.evaluation.prequential.execution", "PrequentialStep"),
    "execute_prequential_step": ("coleman.evaluation.prequential.execution", "execute_prequential_step"),
}

__all__ = [
    "ComparisonDisposition",
    "ComparisonPlan",
    "assess_comparison",
    "MetricTolerance",
    "ReferenceConformancePlan",
    "ReferenceConformanceReport",
    "check_reference_conformance",
    "ComponentObservation",
    "ConformanceLimits",
    "ConformanceTrace",
    "DomainExecutionReceipt",
    "ExecutionKey",
    "StepEvidence",
    "DiagnosticAudit",
    "DiagnosticEvidence",
    "DiagnosticLocator",
    "DiagnosticPanel",
    "DiagnosticView",
    "DomainExecutionContext",
    "DomainExecutor",
    "DomainStepPlan",
    "PrequentialStep",
    "execute_prequential_step",
]


def __getattr__(name: str) -> Any:
    """Load only the selected contract or explicit optional composition boundary."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List declared names without importing replay, storage or inference providers."""
    return sorted(set(globals()) | set(__all__))
