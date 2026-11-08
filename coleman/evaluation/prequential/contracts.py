"""Bounded evidence for a distinct offline domain action and its later outcomes."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import Field, model_validator

from coleman.core.actions import AbstainAction, ChoiceAction, ChoiceSpace
from coleman.core.base import CoreImmutable
from coleman.core.decisions import ExecutedDecision
from coleman.core.feedback import OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar
from coleman.core.primitives import Fingerprint, Identifier, canonical_json, fingerprint
from coleman.core.temporal import Timestamp
from coleman.core.timing import FeedbackTiming
from coleman.evaluation.design._wire import DesignContract, wire_size
from coleman.evaluation.design.contracts import MetricPlan
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.selection import CellIdentity
from coleman.objectives.bindings import DefinitionBinding, require_quantity
from coleman.objectives.feasibility import FiniteFeasibilityResult

MIB = 1024**2


class PrequentialContract(DesignContract):
    """Reuse guarded strict v1 codecs under a 16-MiB hard envelope ceiling.

    Notes
    -----
    Depth 64, duplicate keys, finite numeric ingress, unknown fields and future
    versions follow the existing bounded design reader. Concrete plans impose
    smaller selected budgets. Reading never executes an adapter or performs I/O.
    """

    _maximum_bytes: ClassVar[int] = 16 * MIB

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Bound mutable input before the inherited guarded reader copies it.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete UTF-8 content. Bytearray length is checked before copying.
        **kwargs : object
            Existing strict guarded validation options; relaxation still rejects.

        Returns
        -------
        Self
            Strict bounded v1 model, without migration, source access or I/O.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid size, depth, JSON, version, fields or concrete invariants.
        """
        if isinstance(json_data, bytearray) and len(json_data) > cls._maximum_bytes:
            raise ValueError("Prequential bytearray exceeds its hard byte capacity before copying")
        return super().model_validate_json(json_data, **kwargs)


class ConformanceLimits(CoreImmutable):
    """Declare coupled finite trace and complete report capacities.

    Attributes
    ----------
    max_steps : int
        Opportunities in each trace, default 128 and hard ceiling 4096.
    max_components : int
        Distinct quantities per opportunity, default 8 and hard ceiling 32.
    max_bytes : int
        Complete plan, both traces and all checks together, including framing;
        default 2 MiB and hard ceiling 16 MiB. Count maxima need not jointly fit.
    max_checks : int
        Maximum retained comparisons, default 8192 and hard ceiling 65536.

    Notes
    -----
    These are evidence limits, not CPU, RAM or callback sandbox guarantees.
    """

    max_steps: Annotated[int, Field(strict=True, ge=1, le=4096)] = 128
    max_components: Annotated[int, Field(strict=True, ge=1, le=32)] = 8
    max_bytes: Annotated[int, Field(strict=True, ge=1024, le=16 * MIB)] = 2 * MIB
    max_checks: Annotated[int, Field(strict=True, ge=1, le=65536)] = 8192


class ExecutionKey(CoreImmutable):
    """Identify one domain opportunity independently of a physical invocation.

    Attributes
    ----------
    cell : CellIdentity
        Exact candidate/source-member/seed coordinate. Seeds are repetitions.
    task : Reference
        Distinct domain task and semantic version, separate from coefficient space.
    decision_id : str
        Stable domain opportunity ID, retained on retry and resume.
    """

    cell: CellIdentity
    task: Reference
    decision_id: Identifier

    @property
    def logical_identity(self) -> str:
        """Return a stable content key without authentication or registration.

        Returns
        -------
        str
            Digest over cell, exact task version and logical domain opportunity.

        Raises
        ------
        ValueError
            Unchecked fields cannot satisfy the declared immutable contract.
        """
        checked = type(self).model_validate(self)
        return fingerprint(checked.model_dump(mode="json"))


class DomainExecutionReceipt(PrequentialContract):
    """Bind an earlier coefficient decision to a distinct finite domain action.

    Attributes
    ----------
    profile : {'finite-choice-offline-v1'}
        Deterministic simulated domain execution, never physical delivery proof.
    key, adapter : ExecutionKey, DefinitionBinding
        Logical opportunity and exact fixed adapter definition.
    base_configuration_fingerprint, data_fingerprint : str
        Actual BASE candidate and interpreted input source content.
    coefficient_record_fingerprint, coefficient_execution_fingerprint : str
        Earlier immutable replay record and its unchanged coefficient execution.
        The optional composition must verify these against the retained originals.
    prior_domain_state_fingerprint : str
        Complete domain owner state before execution; never this receipt's after-state.
    executed : ExecutedDecision
        Separate domain proposal and actual choice or explicit abstention. Mapping
        and sampling use core semantics; coefficient replay v1 is not extended.
    intervention, feasibility : DefinitionBinding, FiniteFeasibilityResult
        Exact intervention law and retained finite constraint calculations.
    executed_at : datetime
        Declared simulation execution instant, not an OS clock capture or external
        acknowledgement. It cannot precede the domain proposal's decision clock.

    Raises
    ------
    ValueError
        Untimed/mismatched opportunity, unsupported action shape, sampled execution,
        incompatible finite universe, infeasible chosen member or invalid clocks.

    Notes
    -----
    Construction performs no execution, lookup or I/O. Outcomes arriving later
    belong to ComponentObservation. Reference digests do not prove authenticity.
    """

    profile: Literal["finite-choice-offline-v1"] = "finite-choice-offline-v1"
    key: ExecutionKey
    adapter: DefinitionBinding
    base_configuration_fingerprint: Fingerprint
    data_fingerprint: Fingerprint
    coefficient_record_fingerprint: Fingerprint
    coefficient_execution_fingerprint: Fingerprint
    prior_domain_state_fingerprint: Fingerprint
    executed: ExecutedDecision
    intervention: DefinitionBinding
    feasibility: FiniteFeasibilityResult
    executed_at: Timestamp

    @model_validator(mode="after")
    def _receipt(self) -> Self:
        """Verify finite choice, explicit sampling and producing opportunity joins."""
        request = self.executed.proposal.request
        timing = request.timing
        if request.task != self.key.task or request.decision_id != self.key.decision_id:
            raise ValueError("Domain execution differs from its exact logical opportunity")
        if timing is None or self.executed_at < timing.decision_at:
            raise ValueError("Domain execution requires timing and cannot predate its proposal")
        _finite_execution(self.executed, self.feasibility)
        return self


def _finite_execution(executed: ExecutedDecision, feasibility: FiniteFeasibilityResult) -> None:
    """Check the explicitly qualified deterministic finite-choice profile."""
    if not isinstance(executed.proposal.action, ChoiceAction):
        raise ValueError("Offline domain proposal must be a finite choice")
    if not isinstance(executed.action, ChoiceAction | AbstainAction):
        raise ValueError("Offline domain execution must be a finite choice or explicit abstention")
    if not isinstance(executed.action.space, ChoiceSpace):
        raise ValueError("Offline domain execution must retain a finite choice space")
    if executed.action.space != executed.proposal.action.space:
        raise ValueError("Offline finite execution cannot change its declared choice universe")
    if executed.action.space.members != feasibility.request.universe:
        raise ValueError("Finite feasibility universe differs from the domain execution")
    if executed.proposal.sampling.mechanism != "deterministic" or executed.sampling.mechanism != "deterministic":
        raise ValueError("Offline finite profile requires deterministic sampling declarations")
    if isinstance(executed.action, ChoiceAction) and executed.action.chosen not in feasibility.feasible_members:
        raise ValueError("Executed domain choice lacks retained feasible evidence")


class ComponentObservation(PrequentialContract):
    """Retain one later quantity without inserting future values into a decision.

    Attributes
    ----------
    execution_fingerprint : str
        Exact producing DomainExecutionReceipt, verified by its enclosing step.
    metric, value : MetricPlan, OutcomeValue
        Complete quantity/horizon/estimand and explicit origin/status/value. Raw,
        surrogate, executed objective and cost roles are never interchangeable.
    original, measurement_name : OriginalRecordEvidence, str
        Original producer bytes/digest and exact selected measurement name.
        A different generic metric name may explicitly project a native producer.
    timing, applied_at : FeedbackTiming, datetime or None
        Event/availability/maturity/first receipt and separately supplied actual
        application. Unknown receipt cannot authorize an application.
    physical_group_id : str or None
        Required for realized physical-cost evidence, otherwise null. The outer
        attempt ledger verifies ownership and charges the group only once.

    Raises
    ------
    ValueError
        Quantity or original measurement mismatch, unavailable application clock,
        nonfinite values or a missing/incorrect physical group declaration.

    Notes
    -----
    Modeled costs remain modeled. No fee is derived from lambda or coefficient
    movement, and no observation, update, cost charging or I/O occurs here.
    """

    execution_fingerprint: Fingerprint
    metric: MetricPlan
    value: OutcomeValue
    original: OriginalRecordEvidence
    measurement_name: Identifier
    timing: FeedbackTiming
    applied_at: Timestamp | None
    physical_group_id: Identifier | None

    @model_validator(mode="after")
    def _observation(self) -> Self:
        """Bind exact original quantity, availability and independently known application."""
        _require_original_component(self)
        if self.applied_at is not None and (self.timing.usable_at is None or self.applied_at < self.timing.usable_at):
            raise ValueError("Component application requires all known eligible observation clocks")
        physical = self.metric.role == "physical_cost"
        if physical and self.value.origin != "realized":
            raise ValueError("Modeled or synthetic charges cannot be labeled realized physical costs")
        if physical != (self.physical_group_id is not None):
            raise ValueError("Only realized physical costs require an explicit physical group")
        return self


def _require_original_component(item: ComponentObservation) -> None:
    """Match exact quantity, value/status and availability to the original producer."""
    scalar = Scalar(spec=item.value.spec, value=item.value.value, status=item.value.status)
    require_quantity(scalar, item.metric.quantity)
    rows = [row for row in item.original.record.measurements if row.name == item.measurement_name]
    if len(rows) != 1:
        raise ValueError("Component observation requires its exact original measurement")
    measurement = rows[0]
    semantics = item.metric.quantity.spec.semantics
    if (measurement.unit, measurement.meaning, measurement.direction) != (
        semantics.unit,
        semantics.meaning,
        item.metric.direction,
    ):
        raise ValueError("Original component units, meaning or direction differ")
    measured = canonical_json([measurement.value, measurement.status])
    if measured != canonical_json([item.value.value, item.value.status]):
        raise ValueError("Component value or missingness differs from the original producer")
    if item.original.record.available_at != item.timing.available_at:
        raise ValueError("Component availability differs from its original producer")


class StepEvidence(PrequentialContract):
    """Join a produced action to its separately arriving bounded observations.

    Attributes
    ----------
    receipt : DomainExecutionReceipt
        Decision-time evidence, complete independently of later outcomes.
    components : tuple of ComponentObservation
        Up to 32 unique metric names; missing declared metrics remain detectable
        by conformance. Original source/configuration joins are checked here.

    Notes
    -----
    This completed evidence view is not the earlier decision-time snapshot.
    No counterfactual full expert panel is inferred from its executed scalar.
    """

    receipt: DomainExecutionReceipt
    components: Annotated[tuple[ComponentObservation, ...], Field(max_length=32)] = ()

    @model_validator(mode="after")
    def _step(self) -> Self:
        """Retain unique exact producing joins without collapsing repeated costs."""
        if len({item.metric.name for item in self.components}) != len(self.components):
            raise ValueError("Step contains duplicate component metric names")
        receipt = self.receipt
        for item in self.components:
            record = item.original.record
            if item.execution_fingerprint != receipt.fingerprint:
                raise ValueError("Component differs from its producing domain execution")
            if record.data_fingerprint != receipt.data_fingerprint:
                raise ValueError("Component source differs from its domain execution")
            if record.provenance.spec_fingerprint != receipt.base_configuration_fingerprint:
                raise ValueError("Component differs from its actual BASE configuration")
            if item.timing.event_at < receipt.executed_at:
                raise ValueError("Component event cannot predate its domain execution")
        object.__setattr__(self, "components", tuple(sorted(self.components, key=lambda item: item.metric.name)))
        return self


class ConformanceTrace(PrequentialContract):
    """Retain finite actual/reference evidence without claiming catalogue completeness.

    Attributes
    ----------
    recipe : DefinitionBinding
        Exact supplied producing recipe. Content binding is not execution proof.
    steps : tuple of StepEvidence
        At most 4096 unique opportunities; plan conformance checks missing/extra
        steps rather than assuming an incomplete trace is a complete experiment.

    Notes
    -----
    Canonical key ordering preserves each embedded action's meaningful ordering.
    Complete trace bytes obey the hard reader cap; selected budgets are checked
    with the plan before conformance. No independent-unit claim follows.
    """

    recipe: DefinitionBinding
    steps: Annotated[tuple[StepEvidence, ...], Field(max_length=4096)]

    @model_validator(mode="after")
    def _trace(self) -> Self:
        """Reject duplicate logical steps instead of dropping repeated executions."""
        keys = [item.receipt.key.logical_identity for item in self.steps]
        if len(keys) != len(set(keys)):
            raise ValueError("Conformance trace contains duplicate logical opportunities")
        object.__setattr__(self, "steps", tuple(sorted(self.steps, key=lambda item: item.receipt.key.logical_identity)))
        return self


def require_trace_capacity(trace: ConformanceTrace, limits: ConformanceLimits) -> None:
    """Check a validated trace under selected count and actual UTF-8 budgets.

    Parameters
    ----------
    trace : ConformanceTrace
        Complete supplied trace, without reading or executing any component.
    limits : ConformanceLimits
        Selected capacities, independently bounded by the hard profile.

    Raises
    ------
    ValueError
        Selected step/component count or actual serialized-byte budget is exceeded.
    """
    if len(trace.steps) > limits.max_steps or any(len(item.components) > limits.max_components for item in trace.steps):
        raise ValueError("Conformance trace exceeds selected step/component capacities")
    if wire_size(trace) + 1 > limits.max_bytes:
        raise ValueError("Conformance trace exceeds its selected encoded-byte capacity")
