"""Recompute complete reference conformance without inference or execution."""

from __future__ import annotations

from datetime import datetime
from fractions import Fraction
from operator import itemgetter
from typing import Annotated, Literal, NamedTuple, Self, cast

from pydantic import BaseModel, Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, Identifier, fingerprint
from coleman.evaluation.design._wire import guard_graph, wire_size
from coleman.evaluation.design.contracts import MetricPlan
from coleman.evaluation.prequential.contracts import (
    ComponentObservation,
    ConformanceLimits,
    ConformanceTrace,
    ExecutionKey,
    PrequentialContract,
    StepEvidence,
    require_trace_capacity,
)
from coleman.objectives.bindings import DefinitionBinding, SemanticDefinition

ConformanceStatus = Literal["pass", "fail", "unknown"]
CheckField = Literal["catalogue", "binding", "proposal", "execution", "intervention", "component", "clock"]


class MetricTolerance(CoreImmutable):
    """Freeze an absolute and reference-relative tolerance for one quantity.

    Attributes
    ----------
    metric_name : str
        Exact named metric in the plan; its complete units/meaning are retained.
    absolute : float
        Nonnegative finite allowance in that metric's units, default zero.
    relative : float
        Nonnegative finite dimensionless factor times abs(reference), default zero.

    Notes
    -----
    Stored binary64 values are compared as exact rational numbers. This numerical
    tolerance is neither scientific equivalence nor a unit conversion.
    """

    metric_name: Identifier
    absolute: Annotated[Binary64, Field(ge=0)] = 0.0
    relative: Annotated[Binary64, Field(ge=0)] = 0.0


class ReferenceConformancePlan(PrequentialContract, SemanticDefinition):
    """Freeze a bounded reference recipe before inspecting protected outputs.

    Attributes
    ----------
    reference, lineage : Reference, DefinitionLineage
        Versioned definition identity and explicit supplied lineage, not proof.
    adapter, reference_recipe : DefinitionBinding
        Exact distinct-domain adapter and reference generation definitions.
    baseline_candidate_id, base_configuration_fingerprint : str
        Actual compared workflow and BASE configuration, not a promoted method.
    data_fingerprint : str
        Interpreted source input manifest, not newly read protected outcomes.
    opportunities : tuple of ExecutionKey
        Complete expected catalogue, with no dropped failed/harmful opportunity.
    metrics, tolerances : tuple
        Complete required quantities and optional named numerical allowances.
        Unspecified tolerance is exactly zero, independently of metric bounds.
    clock_tolerance_us : int
        Nonnegative absolute difference in integer microseconds, default zero.
    limits : ConformanceLimits
        Finite counts and actual complete-envelope bytes, checked jointly.

    Notes
    -----
    The later report binds actual reference-output bytes. Freezing this plan must
    not itself inspect hidden protected outputs. No source admission, statistics,
    adapter execution or I/O occurs. Inherited binding covers this full plan.
    """

    adapter: DefinitionBinding
    reference_recipe: DefinitionBinding
    baseline_candidate_id: Identifier
    base_configuration_fingerprint: Fingerprint
    data_fingerprint: Fingerprint
    opportunities: Annotated[tuple[ExecutionKey, ...], Field(min_length=1, max_length=4096)]
    metrics: Annotated[tuple[MetricPlan, ...], Field(min_length=1, max_length=32)]
    tolerances: Annotated[tuple[MetricTolerance, ...], Field(max_length=32)] = ()
    clock_tolerance_us: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)] = 0
    limits: ConformanceLimits = ConformanceLimits()

    @model_validator(mode="after")
    def _catalogue(self) -> Self:
        """Canonicalize complete sets and reject duplicate or unbounded declarations."""
        keys = [item.logical_identity for item in self.opportunities]
        names = {item.name for item in self.metrics}
        tolerance_names = [item.metric_name for item in self.tolerances]
        if len(keys) != len(set(keys)) or len(names) != len(self.metrics):
            raise ValueError("Conformance plan contains duplicate opportunities or metric names")
        if any(item.cell.candidate_id != self.baseline_candidate_id for item in self.opportunities):
            raise ValueError("Reference opportunities must bind the actual baseline candidate")
        if len(tolerance_names) != len(set(tolerance_names)) or not set(tolerance_names) <= names:
            raise ValueError("Conformance tolerances require unique declared metric names")
        if len(keys) > self.limits.max_steps or len(names) > self.limits.max_components:
            raise ValueError("Conformance plan exceeds selected catalogue capacities")
        ordered = sorted(zip(keys, self.opportunities, strict=True), key=itemgetter(0))
        object.__setattr__(self, "opportunities", tuple(map(itemgetter(1), ordered)))
        object.__setattr__(self, "metrics", tuple(sorted(self.metrics, key=lambda item: item.name)))
        object.__setattr__(self, "tolerances", tuple(sorted(self.tolerances, key=lambda item: item.metric_name)))
        if wire_size(self) + 1 > self.limits.max_bytes:
            raise ValueError("Conformance plan exceeds its selected encoded-byte capacity")
        return self


class ConformanceCheck(CoreImmutable):
    """Retain one reproducible comparison and exact source evidence references.

    Attributes
    ----------
    key : ExecutionKey
        Opportunity, including missing or extra catalogue entries.
    field, name : str
        Stable comparison category and exact component/clock/detail name.
    status : {'pass', 'fail', 'unknown'}
        Actual comparison; unknown is never treated as a passing match.
    expected, actual : str or None
        Digests of embedded source values, or null for absent evidence.
    reason : str
        Stable discrepancy/support code; full values remain in the report traces.
    """

    key: ExecutionKey
    field: CheckField
    name: Identifier
    status: ConformanceStatus
    expected: Fingerprint | None
    actual: Fingerprint | None
    reason: Identifier


class ReferenceConformanceReport(PrequentialContract):
    """Embed full inputs and reject invented statuses or omitted discrepancies.

    Attributes
    ----------
    plan : ReferenceConformancePlan
        Predeclared complete recipe/tolerances, independently of later outcomes.
    reference_trace, actual_trace : ConformanceTrace
        Actual retained reference and replay outputs; neither is source admission.
    checks, status : tuple of ConformanceCheck, str
        Complete rederived checks and fail-dominant overall disposition.

    Notes
    -----
    Pass means declared computational conformance only. It is not independent
    sampling, empirical admission, protected eligibility or a production switch.
    Unknown availability remains explicit even when final values agree.
    """

    plan: ReferenceConformancePlan
    reference_trace: ConformanceTrace
    actual_trace: ConformanceTrace
    checks: Annotated[tuple[ConformanceCheck, ...], Field(max_length=65536)]
    status: ConformanceStatus

    @model_validator(mode="after")
    def _recomputed(self) -> Self:
        """Recompute every predicate and selected complete-report capacity."""
        guard_graph(self, maximum_bytes=self.plan.limits.max_bytes, maximum_items=2_000_000)
        expected = _compute_checks(self.plan, self.reference_trace, self.actual_trace)
        if self.checks != expected or self.status != _status(expected):
            raise ValueError("Conformance report differs from its complete recomputed evidence")
        if wire_size(self) + 1 > self.plan.limits.max_bytes:
            raise ValueError("Complete conformance report exceeds its selected byte capacity")
        return self

    @property
    def plan_fingerprint(self) -> str:
        """Return the exact declared plan identity without admitting its source.

        Returns
        -------
        str
            SHA-256 of the complete validated recipe, catalogue and tolerances.
        """
        return self.plan.fingerprint

    def validate_for(self, plan: ReferenceConformancePlan) -> None:
        """Revalidate this full report against an explicitly supplied frozen plan.

        Parameters
        ----------
        plan : ReferenceConformancePlan
            Actual expected definition, not an unresolved fingerprint.

        Raises
        ------
        ValueError
            Any derived field, source input or complete plan binding differs.
            Performs no lookup, scientific inference or I/O.
        """
        checked = type(self).model_validate(self)
        if checked.plan.fingerprint != plan.fingerprint:
            raise ValueError("Conformance report belongs to another declared plan")


def _digest(value: object) -> str:
    """Identify an already typed comparison value, never authenticate it."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return fingerprint(value)


class _Comparison(NamedTuple):
    """Carry a private comparison result to its bounded evidence accumulator."""

    status: ConformanceStatus
    expected: str | None
    actual: str | None
    reason: str


class _Checks:
    """Accumulate bounded comparisons without discarding unknown or failed fields."""

    def __init__(self, plan: ReferenceConformancePlan) -> None:
        """Initialize a local accumulator charged against the selected plan."""
        self.plan = plan
        self.rows: list[ConformanceCheck] = []

    def add(self, key: ExecutionKey, field: CheckField, name: str, expected: object, actual: object) -> None:
        """Compare exact typed content and retain both value fingerprints."""
        expected_digest, actual_digest = _digest(expected), _digest(actual)
        self.record(
            key,
            field,
            name,
            _Comparison(
                "pass" if expected_digest == actual_digest else "fail",
                expected_digest,
                actual_digest,
                "equal" if expected_digest == actual_digest else "different",
            ),
        )

    def record(self, key: ExecutionKey, field: CheckField, name: str, result: _Comparison) -> None:
        """Charge every check before retaining its complete typed row."""
        if len(self.rows) >= self.plan.limits.max_checks:
            raise ValueError("Conformance checks exceed the selected finite capacity")
        self.rows.append(
            ConformanceCheck(
                key=key,
                field=field,
                name=name,
                status=result.status,
                expected=result.expected,
                actual=result.actual,
                reason=result.reason,
            )
        )


def _status(checks: tuple[ConformanceCheck, ...]) -> ConformanceStatus:
    """Known discrepancies dominate unknown support; only complete passing checks pass."""
    if any(row.status == "fail" for row in checks):
        return "fail"
    return "unknown" if any(row.status == "unknown" for row in checks) else "pass"


def _clock(checks: _Checks, key: ExecutionKey, name: str, expected: datetime | None, actual: datetime | None) -> None:
    """Compare exact integer microseconds while retaining absent history as unknown."""
    if expected is None or actual is None:
        checks.record(
            key,
            "clock",
            name,
            _Comparison(
                "unknown",
                None if expected is None else _digest(expected.isoformat()),
                None if actual is None else _digest(actual.isoformat()),
                "unknown_clock",
            ),
        )
        return
    delta = abs(actual - expected)
    microseconds = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    status = "pass" if microseconds <= checks.plan.clock_tolerance_us else "fail"
    checks.record(
        key,
        "clock",
        name,
        _Comparison(
            status,
            _digest(expected.isoformat()),
            _digest(actual.isoformat()),
            "within_clock_tolerance" if status == "pass" else "clock_drift",
        ),
    )


def _component(
    checks: _Checks,
    key: ExecutionKey,
    metric: MetricPlan,
    reference: ComponentObservation | None,
    actual: ComponentObservation | None,
) -> None:
    """Compare one declared component with independent units, origin and clocks."""
    name = metric.name
    if reference is None or actual is None:
        checks.record(
            key,
            "component",
            name,
            _Comparison(
                "unknown",
                None if reference is None else _digest(reference),
                None if actual is None else _digest(actual),
                "missing_component",
            ),
        )
        return
    checks.add(key, "component", name + ".reference_schema", metric, reference.metric)
    checks.add(key, "component", name + ".actual_schema", metric, actual.metric)
    checks.add(key, "component", name + ".origin", reference.value.origin, actual.value.origin)
    checks.add(key, "component", name + ".transform", reference.value.transform, actual.value.transform)
    _component_value(checks, key, name, reference, actual)
    for clock in ("event_at", "available_at", "mature_at", "received_at"):
        _clock(checks, key, name + "." + clock, getattr(reference.timing, clock), getattr(actual.timing, clock))
    _clock(checks, key, name + ".applied_at", reference.applied_at, actual.applied_at)


def _component_value(
    checks: _Checks, key: ExecutionKey, name: str, reference: ComponentObservation, actual: ComponentObservation
) -> None:
    """Evaluate declared finite tolerance without float overflow or missing imputation."""
    left, right = reference.value, actual.value
    if left.status != right.status:
        status, reason = "fail", "different_status"
    elif left.status in {"unknown", "unavailable"}:
        status, reason = "unknown", "unknown_value"
    elif left.status == "not_applicable":
        status, reason = "pass", "not_applicable"
    else:
        tolerance = next((row for row in checks.plan.tolerances if row.metric_name == name), None)
        absolute = Fraction(tolerance.absolute) if tolerance else Fraction()
        relative = Fraction(tolerance.relative) if tolerance else Fraction()
        difference = abs(Fraction(cast(float, right.value)) - Fraction(cast(float, left.value)))
        allowance = absolute + relative * abs(Fraction(cast(float, left.value)))
        status = "pass" if difference <= allowance else "fail"
        reason = "within_metric_tolerance" if status == "pass" else "value_drift"
    checks.record(key, "component", name + ".value", _Comparison(status, _digest(left), _digest(right), reason))


def _step(checks: _Checks, reference: StepEvidence, actual: StepEvidence) -> None:
    """Compare the actual domain boundary, not merely its final aggregate score."""
    key, plan = reference.receipt.key, checks.plan
    left, right = reference.receipt, actual.receipt
    for side, receipt in (("reference", left), ("actual", right)):
        checks.add(key, "binding", side + ".adapter", plan.adapter, receipt.adapter)
        checks.add(
            key,
            "binding",
            side + ".configuration",
            plan.base_configuration_fingerprint,
            receipt.base_configuration_fingerprint,
        )
        checks.add(key, "binding", side + ".data", plan.data_fingerprint, receipt.data_fingerprint)
    checks.add(key, "proposal", "action", left.executed.proposal.action, right.executed.proposal.action)
    checks.add(key, "execution", "action", left.executed.action, right.executed.action)
    checks.add(key, "execution", "mapping", left.executed.mapping, right.executed.mapping)
    checks.add(
        key,
        "execution",
        "sampling_disposition",
        left.executed.sampling_disposition,
        right.executed.sampling_disposition,
    )
    checks.add(key, "execution", "sampling", left.executed.sampling, right.executed.sampling)
    checks.add(key, "intervention", "law", left.intervention, right.intervention)
    checks.add(key, "intervention", "feasibility", left.feasibility, right.feasibility)
    _request_clocks(checks, key, reference, actual)
    reference_components = {row.metric.name: row for row in reference.components}
    actual_components = {row.metric.name: row for row in actual.components}
    names = {metric.name for metric in plan.metrics}
    for name in sorted((reference_components.keys() | actual_components.keys()) - names):
        checks.record(key, "component", name, _Comparison("fail", None, None, "undeclared_component"))
    for metric in plan.metrics:
        _component(checks, key, metric, reference_components.get(metric.name), actual_components.get(metric.name))


def _request_clocks(checks: _Checks, key: ExecutionKey, reference: StepEvidence, actual: StepEvidence) -> None:
    """Compare source/target mapping separately from actual execution clocks."""
    left, right = reference.receipt.executed.proposal.request, actual.receipt.executed.proposal.request
    left_timing, right_timing = left.timing, right.timing
    if left_timing is None or right_timing is None:
        raise ValueError("Validated finite domain executions require both original request clocks")
    checks.add(
        key,
        "proposal",
        "request_inputs",
        left.model_dump(mode="json", exclude={"timing"}),
        right.model_dump(mode="json", exclude={"timing"}),
    )
    for name in ("information_cutoff", "decision_at", "target_start", "target_end"):
        _clock(checks, key, name, getattr(left_timing, name), getattr(right_timing, name))
    checks.add(key, "proposal", "time_mode", left_timing.mode, right_timing.mode)
    checks.add(
        key,
        "proposal",
        "eligible_input",
        left_timing.eligible_input_fingerprint,
        right_timing.eligible_input_fingerprint,
    )
    _clock(checks, key, "executed_at", reference.receipt.executed_at, actual.receipt.executed_at)


def _compute_checks(
    plan: ReferenceConformancePlan, reference: ConformanceTrace, actual: ConformanceTrace
) -> tuple[ConformanceCheck, ...]:
    """Validate selected capacities, then enumerate every declared or extra step."""
    require_trace_capacity(reference, plan.limits)
    require_trace_capacity(actual, plan.limits)
    checks = _Checks(plan)
    first = plan.opportunities[0]
    checks.add(first, "binding", "reference_recipe", plan.reference_recipe, reference.recipe)
    checks.add(first, "binding", "actual_recipe", plan.adapter, actual.recipe)
    expected = {key.logical_identity: key for key in plan.opportunities}
    left = {row.receipt.key.logical_identity: row for row in reference.steps}
    right = {row.receipt.key.logical_identity: row for row in actual.steps}
    for identity in sorted(left.keys() | right.keys() | expected.keys()):
        key = expected.get(identity) or (left.get(identity) or right[identity]).receipt.key
        if identity not in expected:
            checks.record(
                key, "catalogue", "opportunity", _Comparison("fail", None, identity, "undeclared_opportunity")
            )
        elif identity not in left or identity not in right:
            checks.record(
                key,
                "catalogue",
                "opportunity",
                _Comparison(
                    "fail",
                    identity if identity in left else None,
                    identity if identity in right else None,
                    "missing_opportunity",
                ),
            )
        else:
            checks.add(key, "catalogue", "opportunity", key, key)
            _step(checks, left[identity], right[identity])
    return tuple(checks.rows)


def check_reference_conformance(
    plan: ReferenceConformancePlan, reference_trace: ConformanceTrace, actual_trace: ConformanceTrace
) -> ReferenceConformanceReport:
    """Recompute bounded reference agreement from complete supplied typed inputs.

    Parameters
    ----------
    plan : ReferenceConformancePlan
        Frozen actual recipe/tolerances, not inferred from the compared outcomes.
    reference_trace, actual_trace : ConformanceTrace
        Full retained traces. Missing opportunities fail and missing values or
        receipt/application clocks remain unknown, rather than passing as nulls.

    Returns
    -------
    ReferenceConformanceReport
        Strict versioned complete evidence with independently rechecked status.

    Raises
    ------
    ValueError, TypeError
        Invalid contracts, duplicate identities or selected count/byte limits.

    Notes
    -----
    Performs no source reads, adapter calls, admission, inference or adoption.
    """
    plan = ReferenceConformancePlan.model_validate(plan)
    guard_graph((plan, reference_trace, actual_trace), maximum_bytes=plan.limits.max_bytes, maximum_items=2_000_000)
    reference_trace = ConformanceTrace.model_validate(reference_trace)
    actual_trace = ConformanceTrace.model_validate(actual_trace)
    checks = _compute_checks(plan, reference_trace, actual_trace)
    return ReferenceConformanceReport(
        schema_version=1,
        plan=plan,
        reference_trace=reference_trace,
        actual_trace=actual_trace,
        checks=checks,
        status=_status(checks),
    )
