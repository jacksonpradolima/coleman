"""Frozen scoped comparison eligibility, separate from fallback and adoption."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from coleman.core.primitives import Fingerprint, Identifier
from coleman.core.reading import bounded_json_bytes
from coleman.evaluation.design._wire import guard_graph, model_storage
from coleman.evaluation.design.contracts import ExperimentDesign
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.selection import CellIdentity, PhysicalAttempt, SelectionManifest
from coleman.evaluation.prequential.conformance import ReferenceConformancePlan, ReferenceConformanceReport
from coleman.evaluation.prequential.contracts import (
    ComponentObservation,
    ExecutionKey,
    PrequentialContract,
    StepEvidence,
)
from coleman.evaluation.prequential.evidence import DiagnosticEvidence, DiagnosticPanel
from coleman.evaluation.protected.contracts import FrozenCandidate
from coleman.evaluation.protected.observations import ComparisonResult
from coleman.objectives.bindings import SemanticDefinition

if TYPE_CHECKING:
    from coleman.evaluation.design.panel import RawPanel
    from coleman.evaluation.design.reports import ProtocolReport

ComparisonStatus = Literal[
    "qualified_comparison",
    "insufficient_evidence",
    "descriptive_only",
    "unsupported_inference",
    "failed_conformance",
    "unknown_conformance",
]


def _candidate_ingress(value: Any) -> Any:
    """Bound legacy embedded configuration JSON before its own parser runs."""
    fields = model_storage(value) if isinstance(value, BaseModel) else value
    if isinstance(fields, dict):
        configuration = fields.get("configuration_json")
        if isinstance(configuration, str):
            bounded_json_bytes(configuration, 16 * 1024)
    return value


class ComparisonPlan(PrequentialContract, SemanticDefinition):
    """Freeze comparison/fallback rules before selection and protected access.

    Attributes
    ----------
    reference, lineage : Reference, DefinitionLineage
        Versioned rule identity and explicit supplied declaration lineage.
    design_fingerprint : str
        Exact earlier complete design. No later selection fingerprint is present:
        SelectionManifest.rule binds this plan, then the existing freeze binds it.
    conformance_plan : ReferenceConformancePlan
        Earlier recipe/tolerances; actual protected reference outputs arrive later.
    baseline : FrozenCandidate
        Exact deployable BASE configuration, distinct from projected candidates.
    required_hypotheses : tuple of str
        Predeclared baseline contrasts used by the all-required-benefit rule. The
        complete design family, including other contrasts, still receives Holm.
    cell_result_opportunities : tuple of ExecutionKey
        Exactly one baseline terminal opportunity per protected member/seed cell.
        Its original primary observation is that cell's estimand, never an
        implicitly inferred sum/mean over the earlier decision trajectory.
    fallback : {'baseline', 'abstain'}, optional
        Frozen conservative disposition, retained even when candidates qualify.
    rule : {'all-required-benefit-v1'}, optional
        All declared contrasts for a candidate must show existing family-wise
        benefit. There is no score maximization or automatic production switch.
    """

    design_fingerprint: Fingerprint
    conformance_plan: ReferenceConformancePlan
    baseline: FrozenCandidate
    required_hypotheses: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=64)]
    cell_result_opportunities: Annotated[tuple[ExecutionKey, ...], Field(min_length=1, max_length=4096)]
    fallback: Literal["baseline", "abstain"] = "baseline"
    rule: Literal["all-required-benefit-v1"] = "all-required-benefit-v1"

    @field_validator("baseline", mode="before")
    @classmethod
    def _baseline_ingress(cls, value: Any) -> Any:
        """Preserve finite configuration bytes/depth before the old candidate reader."""
        return _candidate_ingress(value)

    @model_validator(mode="after")
    def _plan(self) -> Self:
        """Reject ambiguous baseline, hypothesis and terminal opportunity bindings."""
        conformance = self.conformance_plan
        if (self.baseline.candidate_id, self.baseline.configuration_fingerprint) != (
            conformance.baseline_candidate_id,
            conformance.base_configuration_fingerprint,
        ):
            raise ValueError("Comparison baseline differs from its conformance recipe")
        if len(set(self.required_hypotheses)) != len(self.required_hypotheses):
            raise ValueError("Comparison required hypotheses must be unique")
        cells = [key.cell.model_dump_json() for key in self.cell_result_opportunities]
        known = {key.logical_identity for key in conformance.opportunities}
        if len(cells) != len(set(cells)) or any(
            key.logical_identity not in known for key in self.cell_result_opportunities
        ):
            raise ValueError("Comparison terminal cells require unique declared conformance opportunities")
        object.__setattr__(self, "required_hypotheses", tuple(sorted(self.required_hypotheses)))
        object.__setattr__(
            self,
            "cell_result_opportunities",
            tuple(
                sorted(
                    self.cell_result_opportunities,
                    key=lambda key: key.logical_identity,
                )
            ),
        )
        return self

    def validate_for(self, design: ExperimentDesign, selection: SelectionManifest) -> None:
        """Preflight this complete rule before persistence or protected source access.

        Parameters
        ----------
        design : ExperimentDesign
            Actual complete earlier design, including its exact BASE baseline,
            full hypothesis family, primary quantity and protected cell catalogue.
        selection : SelectionManifest
            Actual selection whose rule binds this plan. The plan deliberately
            contains no later selection fingerprint, preserving the acyclic freeze.

        Raises
        ------
        ValueError, TypeError
            Invalid or oversized actual input, changed design/rule, undeclared
            contrast, incompatible baseline/source, incomplete terminal catalogue
            or mismatched primary metric. All three inputs are revalidated first.

        Notes
        -----
        Performs no persistence, reservation, outcome loading or adoption. Call
        before an audit callback; later assessment reuses the same checked-plan
        predicates on its already revalidated evidence without repeating this
        complete ingress. No source authenticity or empirical admission is implied.
        """
        guard_graph((self, design, selection), maximum_bytes=16 * 1024**2, maximum_items=2_000_000)
        plan = ComparisonPlan.model_validate(self)
        checked_design = ExperimentDesign.model_validate(design)
        checked_selection = SelectionManifest.model_validate(selection)
        _require_plan(plan, checked_design, checked_selection)


class ComparisonDisposition(PrequentialContract):
    """Retain scoped eligibility and the unchanged predeclared fallback.

    Attributes
    ----------
    plan_fingerprint, conformance_fingerprint, evidence_fingerprint : str
        Complete original inputs; consumers must rejoin them with validate_for.
    design_fingerprint, selection_fingerprint : str
        Exact earlier design and actual rule-bound selection.
    baseline, fallback : FrozenCandidate, str
        Exact frozen BASE fallback or explicit abstention, not an executed action.
    status, reasons : str, tuple of str
        Qualified, insufficient, descriptive, unsupported or conformance outcome.
        Reasons are conservative blockers, never inferred missing observations.
    results, eligible_candidates : tuple
        The complete existing family conclusions and exact eligible BASE members.
        A qualified comparison may have zero or several eligible candidates.
    evidence_basis, source_admission_fingerprint : str, str or None
        Original caller declarations, never a verified empirical-admission flag.

    Notes
    -----
    This artifact cannot deploy a candidate. Canonical identity and internal
    consistency do not authenticate observations or prove scientific independence.
    """

    plan_fingerprint: Fingerprint
    conformance_fingerprint: Fingerprint
    evidence_fingerprint: Fingerprint
    design_fingerprint: Fingerprint
    selection_fingerprint: Fingerprint
    baseline: FrozenCandidate
    fallback: Literal["baseline", "abstain"]
    status: ComparisonStatus
    reasons: Annotated[tuple[Identifier, ...], Field(max_length=64)]
    results: Annotated[tuple[ComparisonResult, ...], Field(max_length=64)]
    eligible_candidates: Annotated[tuple[FrozenCandidate, ...], Field(max_length=64)]
    evidence_basis: Literal["constructed_mechanism", "empirical_protected", "unqualified"]
    source_admission_fingerprint: Fingerprint | None

    @field_validator("baseline", mode="before")
    @classmethod
    def _baseline_ingress(cls, value: Any) -> Any:
        """Bound the actual legacy baseline before parsing its configuration."""
        return _candidate_ingress(value)

    @field_validator("eligible_candidates", mode="before")
    @classmethod
    def _candidate_ingress(cls, values: Any) -> Any:
        """Check actual count and every embedded configuration before item parsing."""
        if not isinstance(values, tuple | list) or len(values) > 64:
            raise ValueError("Comparison candidate catalogue exceeds its finite capacity")
        for value in values:
            _candidate_ingress(value)
        return values

    @model_validator(mode="after")
    def _disposition(self) -> Self:
        """Reject duplicate or impossible standalone claims before context joins."""
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("Comparison disposition reasons must be unique")
        names = [item.candidate_id for item in self.eligible_candidates]
        if len(names) != len(set(names)) or self.baseline.candidate_id in names:
            raise ValueError("Comparison eligible candidates must be distinct nonbaseline members")
        if self.status != "qualified_comparison" and self.eligible_candidates:
            raise ValueError("Only a qualified comparison may identify eligible candidates")
        return self

    def validate_for(
        self,
        plan: ComparisonPlan,
        conformance: ReferenceConformanceReport,
        evidence: ProtocolReport | DiagnosticEvidence,
    ) -> None:
        """Re-derive this exact persisted disposition against its complete inputs.

        Parameters
        ----------
        plan, conformance, evidence : typed contracts
            Actual frozen plan, complete reference check and original diagnostic
            or protocol report. Isolated booleans or result rows are insufficient.

        Raises
        ------
        ValueError
            Any canonical field, binding, family, baseline or eligibility differs.
            No policy is executed or source admission inferred.
        """
        expected = assess_comparison(plan, conformance, evidence)
        if type(self).model_validate(self).to_json() != expected.to_json():
            raise ValueError("Comparison disposition differs from its complete rederived inputs")


def _require_plan(plan: ComparisonPlan, design: ExperimentDesign, selection: SelectionManifest) -> None:
    """Join the acyclic rule, complete baseline family and terminal cell catalogue."""
    selection.require_design(design)
    if plan.design_fingerprint != design.fingerprint or selection.rule != plan.binding:
        raise ValueError("Comparison plan differs from its pre-access selected rule")
    _require_baseline(plan, design)
    _require_family(plan, design)
    _require_terminals(plan, design)


def _require_baseline(plan: ComparisonPlan, design: ExperimentDesign) -> None:
    """Join deployable BASE content and the interpreted source exactly."""
    candidate = next(
        (item for item in design.candidates if item.candidate.candidate_id == plan.baseline.candidate_id), None
    )
    if (
        candidate is None
        or candidate.candidate.model_dump_json() != plan.baseline.model_dump_json()
        or candidate.role
        not in {
            "fixed_baseline",
            "equal_weight_baseline",
            "periodic_refit_baseline",
        }
    ):
        raise ValueError("Comparison requires its exact deployable BASE baseline")
    if plan.conformance_plan.data_fingerprint != design.snapshot.data_fingerprint:
        raise ValueError("Comparison conformance uses another interpreted source")


def _require_family(plan: ComparisonPlan, design: ExperimentDesign) -> None:
    """Keep the frozen eligibility subset inside the complete adjusted family."""
    hypotheses = {item.hypothesis_id: item for item in design.hypotheses}
    if any(
        name not in hypotheses or hypotheses[name].baseline_id != plan.baseline.candidate_id
        for name in plan.required_hypotheses
    ):
        raise ValueError("Comparison required contrasts differ from the complete frozen baseline family")


def _require_terminals(plan: ComparisonPlan, design: ExperimentDesign) -> None:
    """Require the complete terminal-cell catalogue and exact primary quantity."""
    expected = {(member.member_id, seed) for member in design.protected_members for seed in design.seeds}
    actual = {(key.cell.member_id, key.cell.seed) for key in plan.cell_result_opportunities}
    if actual != expected:
        raise ValueError("Comparison terminal opportunities omit or add required baseline cells")
    metrics = {metric.name: metric.model_dump_json() for metric in plan.conformance_plan.metrics}
    primary = design.metric(design.primary_metric)
    if metrics.get(primary.name) != primary.model_dump_json():
        raise ValueError("Comparison conformance omits the exact declared primary metric")


def _require_terminal_originals(
    plan: ComparisonPlan,
    conformance: ReferenceConformanceReport,
    panel: RawPanel | DiagnosticPanel,
) -> None:
    """Join each predeclared terminal estimand to its original complete raw cell."""
    steps = {step.receipt.key.logical_identity: step for step in conformance.actual_trace.steps}
    rows = {(row.member_id, row.seed): row for row in panel.rows if row.candidate_id == plan.baseline.candidate_id}
    for key in plan.cell_result_opportunities:
        step = steps.get(key.logical_identity)
        if step is None:
            continue  # The complete conformance report retains the missing opportunity failure.
        components = {item.metric.name: item for item in step.components}
        primary = components.get(panel.design.primary_metric)
        if primary is None:
            continue  # Missing required components remain unknown, never qualified.
        row = rows[key.cell.member_id, key.cell.seed]
        if primary.original.model_dump_json() != row.evidence.model_dump_json():
            raise ValueError("Terminal conformance primary differs from the original baseline raw cell")
        if primary.metric.model_dump_json() != panel.design.metric(panel.design.primary_metric).model_dump_json():
            raise ValueError("Terminal primary metric differs from its declared cell estimand")
        _require_named_primary(primary, row.evidence, panel.design.primary_metric)
    _require_physical_costs(plan, conformance, panel)


def _require_named_primary(primary: ComponentObservation, original: OriginalRecordEvidence, name: str) -> None:
    """Bind the selected terminal quantity to the raw measurement used by inference."""
    measurement = original.record.measurement(name)
    actual = primary.measurement_name, primary.value.value, primary.value.status
    expected = name, measurement.value, measurement.status
    if actual != expected:
        raise ValueError("Terminal conformance primary differs from the named raw primary measurement")


def _require_physical_costs(
    plan: ComparisonPlan,
    conformance: ReferenceConformanceReport,
    panel: RawPanel | DiagnosticPanel,
) -> None:
    """Verify realized group/cost references once against the actual complete ledger."""
    retained = _baseline_costs(plan, panel)
    seen = set()
    for step, component in _realized_costs(conformance):
        key = component.physical_group_id, component.original.record_key
        found = retained.get(key)
        if (
            found is None
            or found[0].model_dump_json() != component.original.model_dump_json()
            or step.receipt.key.cell not in found[1]
        ):
            raise ValueError("Realized conformance cost differs from its original physical group ledger")
        seen.add(key)
    if set(retained) != seen and conformance.status == "pass":
        raise ValueError("Baseline physical charges are omitted from the complete conformance trace")


def _baseline_costs(
    plan: ComparisonPlan,
    panel: RawPanel | DiagnosticPanel,
) -> dict[tuple[str | None, tuple[str, str, str, str]], tuple[OriginalRecordEvidence, tuple[CellIdentity, ...]]]:
    """Join each original charge once to its same-candidate group's output union."""
    attempts = tuple(attempt for attempt in panel.ledger.attempts if attempt.candidate_id == plan.baseline.candidate_id)
    outputs = _group_outputs(attempts)
    return {
        (attempt.group_id, cost.record_key): (cost, tuple(outputs[attempt.group_id].values()))
        for attempt in attempts
        for cost in attempt.costs
    }


def _group_outputs(attempts: tuple[PhysicalAttempt, ...]) -> dict[str, dict[tuple[str, str, int], CellIdentity]]:
    """Collect exact output coordinates across already same-candidate retries."""
    outputs: dict[str, dict[tuple[str, str, int], CellIdentity]] = {}
    for attempt in attempts:
        outputs.setdefault(attempt.group_id, {}).update(
            ((cell.candidate_id, cell.member_id, cell.seed), cell) for cell in attempt.outputs
        )
    return outputs


def _realized_costs(
    conformance: ReferenceConformanceReport,
) -> Iterator[tuple[StepEvidence, ComponentObservation]]:
    """Visit declared realized-cost references while retaining original step keys."""
    for step in conformance.actual_trace.steps:
        for component in step.components:
            if component.metric.role == "physical_cost" and component.value.origin == "realized":
                yield step, component


def _assessment_status(
    conformance: ReferenceConformanceReport,
    evidence: ProtocolReport | DiagnosticEvidence,
    results: tuple[ComparisonResult, ...],
) -> tuple[ComparisonStatus, tuple[str, ...]]:
    """Retain distinct known failure, unsupported, source and sufficient-evidence outcomes."""
    if isinstance(evidence, DiagnosticEvidence):
        status, reasons = (
            ("unsupported_inference", evidence.qualification.reasons)
            if (evidence.qualification.status == "unsupported")
            else ("descriptive_only", ("diagnostic_evidence_only",))
        )
    else:
        status, reasons = _protocol_status(evidence, results)
    if conformance.status == "fail":
        return "failed_conformance", (*reasons, "reference_conformance_failed")
    if conformance.status == "unknown":
        return "unknown_conformance", (*reasons, "reference_conformance_unknown")
    return status, reasons


def _protocol_status(
    evidence: ProtocolReport,
    results: tuple[ComparisonResult, ...],
) -> tuple[ComparisonStatus, tuple[str, ...]]:
    """Keep actual source eligibility separate from outcomes and declared scope."""
    reasons = []
    exposed = not evidence.confirmatory_eligible
    unqualified = evidence.panel.design.evidence_basis == "unqualified"
    insufficient = any(result.conclusion == "insufficient_evidence" for result in results)
    if exposed:
        reasons.append("source_or_protected_access_not_confirmatory")
    if unqualified:
        reasons.append("unqualified_evidence_basis")
    if insufficient:
        reasons.append("incomplete_or_insufficient_independent_units")
    if exposed or unqualified:
        return "descriptive_only", tuple(reasons)
    return ("insufficient_evidence", tuple(reasons)) if insufficient else ("qualified_comparison", ())


def _eligible(
    plan: ComparisonPlan,
    design: ExperimentDesign,
    results: tuple[ComparisonResult, ...],
) -> tuple[FrozenCandidate, ...]:
    """Apply only the frozen all-required-benefit rule within the complete family."""
    results_by_id = {item.hypothesis_id: item for item in results}
    required = set(plan.required_hypotheses)
    grouped: dict[str, list[str]] = {}
    for hypothesis in design.hypotheses:
        if hypothesis.hypothesis_id in required:
            grouped.setdefault(hypothesis.candidate_id, []).append(hypothesis.hypothesis_id)
    names = {
        name
        for name, hypotheses in grouped.items()
        if all(results_by_id[key].conclusion == "evidence_of_benefit" for key in hypotheses)
    }
    return tuple(item.candidate for item in design.candidates if item.candidate.candidate_id in names)


def assess_comparison(
    plan: ComparisonPlan,
    conformance: ReferenceConformanceReport,
    evidence: ProtocolReport | DiagnosticEvidence,
) -> ComparisonDisposition:
    """Assess exact scoped comparison support without executing a replacement.

    Parameters
    ----------
    plan : ComparisonPlan
        Predeclared acyclic rule, complete baseline and terminal estimand mapping.
    conformance : ReferenceConformanceReport
        Full actual/reference traces and rederived comparisons under that plan.
    evidence : ProtocolReport or DiagnosticEvidence
        Complete existing protected result or explicitly descriptive original grid.

    Returns
    -------
    ComparisonDisposition
        Complete family conclusions, scoped eligible candidates and unchanged
        conservative fallback. Source admission remains an external assertion.

    Raises
    ------
    ValueError, TypeError
        Invalid/capacity inputs or mismatched plan/selection/baseline/original
        terminal quantity/actual cost/family. No I/O or source access occurs.
    """
    guard_graph((plan, conformance, evidence), maximum_bytes=16 * 1024**2, maximum_items=2_000_000)
    plan = ComparisonPlan.model_validate(plan)
    conformance = ReferenceConformanceReport.model_validate(conformance)
    conformance.validate_for(plan.conformance_plan)
    results: tuple[ComparisonResult, ...]
    if isinstance(evidence, DiagnosticEvidence):
        evidence = DiagnosticEvidence.model_validate(evidence)
        results = ()
    else:
        # Keep optional storage/report dependencies out of pure declaration imports.
        from coleman.evaluation.design.reports import ProtocolReport  # pylint: disable=import-outside-toplevel

        if not isinstance(evidence, ProtocolReport):
            raise TypeError("Comparison requires complete diagnostic or protocol evidence")
        evidence = ProtocolReport.model_validate(evidence)
        results = evidence.family.results
    design, selection = evidence.panel.design, evidence.panel.selection
    _require_plan(plan, design, selection)
    _require_terminal_originals(plan, conformance, evidence.panel)
    status, reasons = _assessment_status(conformance, evidence, results)
    eligible = _eligible(plan, design, results) if status == "qualified_comparison" else ()
    return ComparisonDisposition(
        schema_version=1,
        plan_fingerprint=plan.fingerprint,
        conformance_fingerprint=conformance.fingerprint,
        evidence_fingerprint=evidence.fingerprint,
        design_fingerprint=design.fingerprint,
        selection_fingerprint=selection.fingerprint,
        baseline=plan.baseline,
        fallback=plan.fallback,
        status=status,
        reasons=tuple(sorted(reasons)),
        results=results,
        eligible_candidates=eligible,
        evidence_basis=design.evidence_basis,
        source_admission_fingerprint=design.source_admission_fingerprint,
    )
