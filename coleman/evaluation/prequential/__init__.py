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

_COMPARISON_MODULE = "coleman.evaluation.prequential.comparison"
_CONFORMANCE_MODULE = "coleman.evaluation.prequential.conformance"
_CONTRACTS_MODULE = "coleman.evaluation.prequential.contracts"
_EVIDENCE_MODULE = "coleman.evaluation.prequential.evidence"
_EXECUTION_MODULE = "coleman.evaluation.prequential.execution"

_EXPORTS = {
    "ComparisonDisposition": (_COMPARISON_MODULE, "ComparisonDisposition"),
    "ComparisonPlan": (_COMPARISON_MODULE, "ComparisonPlan"),
    "assess_comparison": (_COMPARISON_MODULE, "assess_comparison"),
    "MetricTolerance": (_CONFORMANCE_MODULE, "MetricTolerance"),
    "ReferenceConformancePlan": (_CONFORMANCE_MODULE, "ReferenceConformancePlan"),
    "ReferenceConformanceReport": (_CONFORMANCE_MODULE, "ReferenceConformanceReport"),
    "check_reference_conformance": (_CONFORMANCE_MODULE, "check_reference_conformance"),
    "ComponentObservation": (_CONTRACTS_MODULE, "ComponentObservation"),
    "ConformanceLimits": (_CONTRACTS_MODULE, "ConformanceLimits"),
    "ConformanceTrace": (_CONTRACTS_MODULE, "ConformanceTrace"),
    "DomainExecutionReceipt": (_CONTRACTS_MODULE, "DomainExecutionReceipt"),
    "ExecutionKey": (_CONTRACTS_MODULE, "ExecutionKey"),
    "StepEvidence": (_CONTRACTS_MODULE, "StepEvidence"),
    "DiagnosticAudit": ("coleman.evaluation.prequential.diagnostics", "DiagnosticAudit"),
    "DiagnosticEvidence": (_EVIDENCE_MODULE, "DiagnosticEvidence"),
    "DiagnosticLocator": (_EVIDENCE_MODULE, "DiagnosticLocator"),
    "DiagnosticPanel": (_EVIDENCE_MODULE, "DiagnosticPanel"),
    "DiagnosticView": (_EVIDENCE_MODULE, "DiagnosticView"),
    "DomainExecutionContext": (_EXECUTION_MODULE, "DomainExecutionContext"),
    "DomainExecutor": (_EXECUTION_MODULE, "DomainExecutor"),
    "DomainStepPlan": (_EXECUTION_MODULE, "DomainStepPlan"),
    "PrequentialStep": (_EXECUTION_MODULE, "PrequentialStep"),
    "execute_prequential_step": (_EXECUTION_MODULE, "execute_prequential_step"),
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
