"""Versioned controlled-mixture configuration, receipts and complete state.

All amounts describe expert coefficients or bounded mathematical losses.
Neither these records nor the movement penalty represent paid fees or trades.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictBool, model_validator

from coleman.core.feedback import FeedbackRow, FullPanel
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec
from coleman.core.primitives import Fingerprint, canonical_json
from coleman.core.temporal import Timestamp
from coleman.learning._control_values import _weight_bytes
from coleman.learning.contracts import (
    LearningFailure,
    LearningLimits,
    _BoundedContract,
    _BoundedImmutable,
    validate_loss_spec,
)
from coleman.objectives.bindings import DefinitionBinding, SemanticDefinition
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExactPathUsage,
    L1MovementPenalty,
    MixtureControls,
    MovementUsage,
    ProjectionOutput,
    ReleaseInterval,
    RoundCadence,
)
from coleman.objectives.preferences import CoefficientWeights, InitialPrior, coefficient_weights

ControlProfile = Literal["controlled-hedge-projection", "controlled-hedge-l1-movement"]
ControlDisposition = Literal["held", "released", "empty", "infeasible", "numerical_failure"]
Count = Annotated[int, Field(strict=True, ge=0, le=1048576)]


def _raw_field(value: Any, field: str) -> object:
    """Read only ordinary dictionaries or the selected immutable DTOs."""
    if isinstance(value, dict):
        return value.get(field)
    known = (
        ControlledConfig,
        CoefficientOrigin,
        InitialPrior,
        AcceptedPanel,
        FeedbackRow,
        FullPanel,
        ControlOutcome,
        ClosedInterval,
    )
    if isinstance(value, known):
        return getattr(value, field, None)
    return None


def _raw_count(value: object, maximum: int, name: str) -> int:
    """Reject oversized flat input before nested model validation."""
    if not isinstance(value, (tuple, list)):
        raise ValueError(name + " requires a finite tuple or list")
    if len(value) > maximum:
        raise ValueError(name + " exceeds the selected count capacity")
    return len(value)


class _ControllerContract(_BoundedContract):
    """Share hard byte/depth guards across both new controller JSON readers."""


class ControlledLimits(LearningLimits):
    """Bound selected pending panels and retained host-clock declarations.

    Attributes
    ----------
    maximum_pending : int
        One through 4096 complete panels retained before a successful release.
    maximum_intervals : int
        One through 4096 retained cadence or movement-period declarations.

    Notes
    -----
    Inherited member/update/byte ceilings also apply. Their Cartesian product
    is not a promise that every largest configuration fits one checkpoint.
    """

    maximum_pending: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096
    maximum_intervals: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096


class ControlledConfig(SemanticDefinition, _ControllerContract):
    """Bind one immutable constant-rate exponential/share/output law.

    Attributes
    ----------
    profile : {'controlled-hedge-projection', 'controlled-hedge-l1-movement'}
        Separate version-one algorithm identities. The output discriminator
        must agree; missing penalty data never selects a penalty implicitly.
    controls : MixtureControls
        Complete initial coefficients, target, boxes, movement and one cadence.
        Its L1 origin, when present, is the initial emitted vector only.
    loss_spec : ScalarSpec
        Exact dimensionless bounded loss interpretation, not raw signed P&L.
    transform : DefinitionBinding
        Exact declared transform; fitting and source authentication are external.
    eta : float
        Finite nonnegative constant rate on the sum of released panel losses.
    period_clock : Reference or None
        Required host-clock identity for per-interval movement only. Periods
        are explicit UTC half-open intervals, independent from release cadence.
    limits : ControlledLimits
        Coupled finite state/admission bounds.
    numerical_profile : {'exp-share-joint-emitted-v1'}
        Exponential update, one target share and one joint constrained output.

    Notes
    -----
    Parameters are fixed within this state lineage. A target/profile/strength
    change requires an explicit compatible future lineage, retaining the old
    pending-feedback receiver. The moving origin is complete state under this
    fixed law; changing coefficients does not rewrite the law's binding.
    No reference Hedge/Fixed Share regret or downstream-risk claim is made.
    """

    profile: ControlProfile
    controls: MixtureControls
    loss_spec: ScalarSpec
    transform: DefinitionBinding
    eta: Annotated[Binary64, Field(ge=0)]
    period_clock: Reference | None = None
    limits: ControlledLimits = ControlledLimits()
    numerical_profile: Literal["exp-share-joint-emitted-v1"] = "exp-share-joint-emitted-v1"

    @model_validator(mode="before")
    @classmethod
    def _member_capacity(cls, data: Any) -> Any:
        """Reject selected prior counts before traversing nested controls."""
        if isinstance(data, dict):
            limits = ControlledLimits.model_validate(data.get("limits", ControlledLimits()))
            controls = data.get("controls")
            prior = (
                controls.initial_prior
                if isinstance(controls, MixtureControls)
                else _raw_field(controls, "initial_prior")
            )
            _raw_count(_raw_field(prior, "weights"), limits.maximum_members, "Controlled members")
        return data

    @model_validator(mode="after")
    def _configuration(self) -> Self:
        """Validate selected controls without projecting the initial vector."""
        validate_loss_spec(self.loss_spec)
        penalized = isinstance(self.controls.output, L1MovementPenalty)
        if penalized != (self.profile == "controlled-hedge-l1-movement"):
            raise ValueError("Controlled profile and explicit output discriminator disagree")
        weights = self.controls.initial_prior.weights
        if len(weights) > self.limits.maximum_members:
            raise ValueError("Controlled members exceed the selected limit")
        for weight, bound in zip(weights, self.controls.bounds.bounds, strict=True):
            if not bound.lower <= weight.value <= bound.upper:
                raise ValueError("Initial emitted coefficients must satisfy every original static box")
        self._initial_origin()
        movement = self.controls.movement
        period_required = movement is not None and movement.scope == "per_interval"
        if period_required != (self.period_clock is not None):
            raise ValueError("Exactly per-interval movement requires an explicit period clock")
        if (
            isinstance(self.controls.cadence, RoundCadence)
            and self.controls.cadence.panels > self.limits.maximum_pending
        ):
            raise ValueError("The selected pending limit cannot admit one complete cadence block")
        if len(canonical_json(self.model_dump(mode="json")).encode("utf-8")) > self.limits.maximum_bytes:
            raise ValueError("Controlled law exceeds the selected complete-byte capacity")
        return self

    def _initial_origin(self) -> None:
        """Require an L1 declaration's origin to retain the exact initial bytes."""
        output = self.controls.output
        if isinstance(output, L1MovementPenalty):
            old = canonical_json([item.model_dump(mode="json") for item in output.origin.weights])
            initial = canonical_json([item.model_dump(mode="json") for item in self.controls.initial_prior.weights])
            if old != initial:
                raise ValueError("The initial L1 origin must equal the declared initial emitted vector")


class MovementPeriod(_ControllerContract):
    """Supply an explicit UTC half-open movement-budget period.

    Attributes
    ----------
    calendar, interval : Reference
        Exact host-clock and logical period identities/versions.
    start, end : datetime
        Known normalized UTC boundaries with start strictly before end. A
        release is charged here exactly when start <= apply_at < end.

    Notes
    -----
    No market calendar or ambient clock is inferred. This period is separate
    from a cadence interval and cannot flush a partial logical panel block.
    """

    calendar: Reference
    interval: Reference
    start: Timestamp
    end: Timestamp

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Reject an empty/reversed period without choosing a calendar."""
        if self.start >= self.end:
            raise ValueError("Movement periods require start before end")
        return self


class AcceptedPanel(_ControllerContract):
    """Retain one fully observed panel before its eventual block consumption.

    Attributes
    ----------
    sequence : int
        Positive accepted-panel clock, not a decision or a release counter.
    reference : Reference
        Stable owner/configuration/sequence identity supplied by the learner.
    panel : FullPanel
        Complete original bounded values, units, transform and origin labels.
    eligible_at, apply_at : datetime or None
        Explicit supplied eligibility/application inputs, or untimed nulls.
        Eligibility cannot follow a known actual application instant.
    """

    sequence: Annotated[int, Field(strict=True, ge=1, le=1048576)]
    reference: Reference
    panel: FullPanel
    eligible_at: Timestamp | None
    apply_at: Timestamp | None

    @model_validator(mode="before")
    @classmethod
    def _panel_shape(cls, data: Any) -> Any:
        """Admit bounded row/value counts before recursive feedback decoding."""
        if isinstance(data, dict):
            rows = _raw_field(data.get("panel"), "rows")
            _raw_count(rows, 4096, "Accepted panel rows")
            if not isinstance(rows, (tuple, list)):
                raise ValueError("Accepted panel rows require a finite tuple or list")
            for row in rows:
                values = _raw_field(row, "values")
                if _raw_count(values, 1, "Accepted scalar values") != 1:
                    raise ValueError("Accepted rows require exactly one loss value")
        return data

    @model_validator(mode="after")
    def _clocks(self) -> Self:
        """Keep unknown clocks unknown instead of manufacturing a receipt."""
        if self.eligible_at is not None and self.apply_at is not None and self.eligible_at > self.apply_at:
            raise ValueError("Panel eligibility cannot follow its actual application")
        return self


class AggregateLoss(_BoundedImmutable):
    """Store a derived release-block loss sum, never another observation.

    Attributes
    ----------
    member : Reference
        Exact expert identity/version.
    value : float
        Finite nonnegative sum of accepted bounded losses in the configured
        loss units. It can exceed one; panel_count identifies its denominator.
    """

    member: Reference
    value: Annotated[Binary64, Field(ge=0)]


class ControlOutcome(_ControllerContract):
    """Record a held panel, closed empty interval or explicit release attempt.

    Attributes
    ----------
    disposition : str
        Held/empty are not exponential/share/penalty steps. Released includes
        nonempty optimal holds; infeasible/numerical failure consumes no block.
    law : DefinitionBinding
        Complete fixed configuration binding, including the selected strength.
    update_count, release_count, consumed_panels : int
        Accepted panels U, successful nonempty releases R and consumed panels.
    weights : tuple of Weight
        Actual current emitted coefficients, not a hidden latent proposal.
    origin : CoefficientOrigin
        Complete pre-attempt movement/penalty origin and its version.
    emitted_origin : CoefficientOrigin or None
        Newly committed origin exactly for a successful nonempty release.
    release : Reference or None
        Logical attempt identity, nullable when no solve was attempted.
    panels : tuple of Reference
        Exact ordered block membership; held receipts can name the pending block.
    summed_losses : tuple of AggregateLoss
        Derived block sums. Empty for held/empty outcomes, never fake zero panels.
    proposal : tuple of Weight or None
        Original proposal after target share, including on solver infeasibility;
        null when a preceding numerical failure prevented its construction.
    proposal_status : {'observed', 'unavailable', 'not_applicable'}
        Explicit proposal availability. Held/empty outcomes perform no proposal;
        a failed earlier numerical stage can make an attempted one unavailable.
    output : ProjectionOutput or L1MovementPenalty or None
        Complete attempted output, including the actual current L1 origin.
    movement_usage : MovementUsage or None
        Complete pre-attempt net-period context, retained for historical solves.
    exact_path_before : ExactPathUsage or None
        Complete pre-attempt accumulated-period ledger, independent from the
        current ledger after later releases or an admitted period reset.
    input_fingerprint : str or None
        Full attempt identity over law, block, proposal, output and context;
        distinct from the geometry-only solver fingerprint below.
    solver_profile, solver_inputs : str or None
        Selected mathematical profile and original exact-geometry fingerprint.
    failure : LearningFailure or None
        Explicit negative disposition, never a paid charge or observed zero.

    Notes
    -----
    Exact rational KKT diagnostics remain in the independently consumable
    in-process solve result. This receipt plus the complete fixed ControlledConfig
    retains the inputs needed to reproduce them, including historical pre-attempt
    period usage. An opaque law binding alone is insufficient. No rational
    transport codec is invented.
    """

    disposition: ControlDisposition
    law: DefinitionBinding
    update_count: Count
    release_count: Count
    consumed_panels: Count
    weights: CoefficientWeights
    origin: CoefficientOrigin
    emitted_origin: CoefficientOrigin | None
    release: Reference | None
    panels: Annotated[tuple[Reference, ...], Field(max_length=4096)]
    summed_losses: Annotated[tuple[AggregateLoss, ...], Field(max_length=4096)]
    proposal: CoefficientWeights | None
    proposal_status: Literal["observed", "unavailable", "not_applicable"]
    output: ProjectionOutput | L1MovementPenalty | None
    movement_usage: MovementUsage | None
    exact_path_before: ExactPathUsage | None
    input_fingerprint: Fingerprint | None
    solver_profile: Literal["binary64-grid-breakpoint-kkt-v1"] | None
    solver_inputs: Fingerprint | None
    failure: LearningFailure | None

    @model_validator(mode="before")
    @classmethod
    def _flat_counts(cls, data: Any) -> Any:
        """Admit bounded flat receipt vectors before their nested models."""
        if isinstance(data, dict):
            for field in ("weights", "panels", "summed_losses"):
                _raw_count(data.get(field), 4096, "Control outcome " + field)
            if data.get("proposal") is not None:
                _raw_count(data["proposal"], 4096, "Control outcome proposal")
        return data

    @model_validator(mode="after")
    def _outcome(self) -> Self:
        """Validate finite emitted state and distinct success/failure meanings."""
        object.__setattr__(self, "weights", coefficient_weights(self.weights, "binary64-simplex-v1"))
        self._disposition()
        self._proposal()
        self._idle_inputs()
        self._attempt_inputs()
        self._attempt_vectors()
        return self

    def _disposition(self) -> None:
        """Keep successful, failed and non-attempted receipts distinct."""
        failed = self.disposition in {"infeasible", "numerical_failure"}
        if failed != (self.failure is not None):
            raise ValueError("Exactly failed control outcomes require an explicit failure")
        successful = self.disposition == "released"
        if successful != (self.emitted_origin is not None):
            raise ValueError("Exactly successful releases require the committed emitted origin")
        if successful and (self.release is None or not self.panels or not self.summed_losses):
            raise ValueError("A successful release requires its nonempty complete panel block")
        if self.disposition in {"held", "empty"} and (self.release is not None or self.summed_losses):
            raise ValueError("Held/empty outcomes cannot perform a hidden mathematical release")
        if self.failure is not None and (self.failure.code == "infeasible") != (self.disposition == "infeasible"):
            raise ValueError("The negative disposition and its failure code disagree")
        if len(set(self.panels)) != len(self.panels):
            raise ValueError("A release receipt cannot repeat a panel identity")

    def _proposal(self) -> None:
        """Retain actual proposal availability without fabricating observations."""
        if self.proposal is not None:
            object.__setattr__(self, "proposal", coefficient_weights(self.proposal, "binary64-simplex-v1"))
            if tuple(item.member for item in self.proposal) != tuple(item.member for item in self.weights):
                raise ValueError("The proposal must retain every exact coefficient member")
        if (self.proposal_status == "observed") != (self.proposal is not None):
            raise ValueError("A proposal value and its explicit availability must agree")
        absent = self.disposition in {"held", "empty"}
        if absent != (self.proposal_status == "not_applicable"):
            raise ValueError("Held/empty outcomes cannot invent an unavailable proposal attempt")

    def _idle_inputs(self) -> None:
        """Keep held/empty receipts free of hidden solve or budget steps."""
        absent = self.disposition in {"held", "empty"}
        if absent and (self.output is not None or self.input_fingerprint is not None):
            raise ValueError("Held/empty outcomes cannot apply a hidden output objective")
        if absent and (self.solver_profile is not None or self.solver_inputs is not None):
            raise ValueError("Held/empty outcomes cannot retain a hidden solver attempt")
        if absent and (self.movement_usage is not None or self.exact_path_before is not None):
            raise ValueError("Held/empty outcomes cannot retain hidden pre-attempt movement context")

    def _attempt_inputs(self) -> None:
        """Join complete attempts, original origins and finite block sums."""
        attempted = self.disposition not in {"held", "empty"}
        missing = self.release is None or not self.panels or self.output is None or self.input_fingerprint is None
        if attempted and missing:
            raise ValueError("A mathematical attempt requires its release, block, output and complete input binding")
        if (self.solver_profile is None) != (self.solver_inputs is None):
            raise ValueError("Solver profile and exact input identity must be present together")
        if self.disposition in {"released", "infeasible"} and (self.proposal is None or self.solver_profile is None):
            raise ValueError("A committed or infeasible release requires an observed proposal and solver receipt")

    def _attempt_vectors(self) -> None:
        """Join exact block coordinates and the actually committed origin vector."""
        if self.disposition not in {"held", "empty"}:
            members = tuple(item.member for item in self.weights)
            if tuple(item.member for item in self.summed_losses) != members:
                raise ValueError("Block sums must retain every exact coefficient member")
            if any(item.value > len(self.panels) for item in self.summed_losses):
                raise ValueError("A bounded block sum cannot exceed its panel count")
        if isinstance(self.output, L1MovementPenalty) and self.output.origin.to_json() != self.origin.to_json():
            raise ValueError("The attempted penalty must use the retained actual origin")
        committed = self.origin if self.emitted_origin is None else self.emitted_origin
        if _weight_bytes(committed.weights) != _weight_bytes(self.weights):
            raise ValueError("Release receipt coefficients differ from its committed origin")


class ClosedInterval(_BoundedImmutable):
    """Freeze one declared release cutoff and its original disposition.

    Attributes
    ----------
    interval : ReleaseInterval
        Complete supplied cutoff and single release point.
    outcome : ControlOutcome
        Original result, returned for an exact retry without another step.

    Notes
    -----
    Failed points retain their pending panels for a later eligible interval.
    Their closed cutoff does not permit adding late panels to the old attempt.
    """

    interval: ReleaseInterval
    outcome: ControlOutcome


class ControlledPayload(_ControllerContract):
    """Own the complete fixed-law controlled-mixture continuation.

    Attributes
    ----------
    config : ControlledConfig
        Immutable mathematical law and supplied selection/source lineage.
    origin : CoefficientOrigin
        Actual last committed emitted coefficients and version.
    support : tuple of bool
        Required structural support in canonical member order, true exactly
        for a strictly positive actual emitted coefficient.
    update_count, release_count, consumed_panels : int
        Separate U/R and consumed-observation counters.
    pending : tuple of AcceptedPanel
        Complete unconsumed accepted panels, including a failed block. Their
        original order is retained when interval cutoffs select only a subset.
    intervals : tuple of ClosedInterval
        Retained exact cadence cutoffs and retry results, including empty points.
    periods : tuple of MovementPeriod
        Monotone explicit budget periods; old identities cannot reset a ledger.
    period_origin : CoefficientOrigin or None
        Emitted coefficients when the current movement period was admitted.
    exact_path : ExactPathUsage or None
        Authoritative accumulated-absolute variation; absent for net/per-release.
    last_apply_at : datetime or None
        Last explicitly supplied processing clock, without an ambient-time guess.
    last_outcome : ControlOutcome or None
        Latest held/released/empty/negative result, retained with pending inputs.
    unresolved_failure : ControlOutcome or None
        Required explicit negative release frontier. Later held/empty outcomes
        and new periods retain it; a successful nonempty release clears it.

    Notes
    -----
    Counts are complete sufficient state, not authenticated reconstruction of
    discarded experiments. The owner validates binding/profile, selected bytes,
    original joins and release frontiers before replacing any live state.
    """

    config: ControlledConfig
    origin: CoefficientOrigin
    support: Annotated[tuple[StrictBool, ...], Field(min_length=1, max_length=4096)]
    update_count: Count = 0
    release_count: Count = 0
    consumed_panels: Count = 0
    pending: Annotated[tuple[AcceptedPanel, ...], Field(max_length=4096)] = ()
    intervals: Annotated[tuple[ClosedInterval, ...], Field(max_length=4096)] = ()
    periods: Annotated[tuple[MovementPeriod, ...], Field(max_length=4096)] = ()
    period_origin: CoefficientOrigin | None = None
    exact_path: ExactPathUsage | None = None
    last_apply_at: Timestamp | None = None
    last_outcome: ControlOutcome | None = None
    unresolved_failure: ControlOutcome | None

    @model_validator(mode="before")
    @classmethod
    def _bounded_input(cls, data: Any) -> Any:
        """Admit selected flat counts and panel work before recursive decoding."""
        if not isinstance(data, dict):
            return data
        config = ControlledConfig.model_validate(data.get("config"))
        limits = config.limits
        count = _raw_count(data.get("pending", ()), limits.maximum_pending, "Controlled pending panels")
        intervals = _raw_count(data.get("intervals", ()), limits.maximum_intervals, "Controlled intervals")
        _raw_count(data.get("periods", ()), limits.maximum_intervals, "Controlled periods")
        _raw_count(data.get("support"), limits.maximum_members, "Structural support")
        members = config.controls.initial_prior.weights
        schema = len(config.loss_spec.model_dump_json().encode("utf-8"))
        schema += len(config.transform.reference.model_dump_json().encode("utf-8"))
        coordinates = sum(len(item.member.model_dump_json().encode("utf-8")) for item in members)
        minimum = len(members) * schema + coordinates
        if count * minimum + intervals * coordinates > limits.maximum_bytes:
            raise ValueError("Retained schemas and coordinates exceed the selected complete-byte capacity")
        return data

    @model_validator(mode="after")
    def _frontiers(self) -> Self:
        """Validate complete mathematical continuation before owner assignment."""
        validate_payload(self)
        return self


def validate_payload(payload: ControlledPayload) -> None:
    """Validate complete controlled continuation, including retained input joins.

    Parameters
    ----------
    payload : ControlledPayload
        Finite fixed-law payload. Pending observations retain exact members,
        loss units and transform; structural support is explicit.

    Returns
    -------
    None
        Original counter, history, origin and geometry joins are valid. A
        well-formed mathematically infeasible exact set remains valid state.

    Raises
    ------
    ValueError, TypeError
        Malformed joins, forged receipts, selected capacities or bytes.

    Notes
    -----
    This validates mathematical state, not authenticated event history.
    Complete outer AlgorithmState bytes must additionally fit the selected cap.
    """
    # Defer the private validator until the public DTO definitions are complete.
    # pylint: disable-next=import-outside-toplevel
    from coleman.learning._control_validation import validate_payload as _validate_payload

    _validate_payload(payload)
