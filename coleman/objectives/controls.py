"""Typed coefficient controls and pure feasibility checks, without a learner."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, SimplexProfile
from coleman.core.primitives import MAX_MEMBERS, canonical_json, fingerprint
from coleman.core.reading import bounded_json_bytes
from coleman.core.temporal import Timestamp
from coleman.core.units import UnitSemantics
from coleman.objectives.arithmetic import PROFILE, checked_fraction
from coleman.objectives.bindings import DefinitionBinding, SemanticDefinition
from coleman.objectives.movement import (
    ExactCoefficientVariation,
    ExactExpertInterval,
    ExactMixtureIntervals,
    _binary64_atoms,
    _exact_intersection,
    _exact_path_remaining,
    _intersection_reason,
    _same_members,
)
from coleman.objectives.preferences import (
    COEFFICIENT_SEMANTICS,
    CoefficientWeights,
    DeclaredTargetShare,
    InitialPrior,
    coefficient_weights,
)


class ExpertBound(CoreImmutable):
    """Declare one well-formed hard coefficient box.

    Attributes
    ----------
    member : Reference
        Exact expert identity and version.
    lower, upper : float
        Inclusive finite coefficient fractions with 0 <= lower <= upper <= 1.

    Notes
    -----
    A collection may have an empty simplex intersection despite every box being
    well formed. That is an infeasible problem, not malformed input.
    """

    member: Reference
    lower: Annotated[Binary64, Field(ge=0, le=1)]
    upper: Annotated[Binary64, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _interval(self) -> Self:
        """Reject a reversed individual interval, without repairing bounds."""
        if self.lower > self.upper:
            raise ValueError("An individual expert lower bound exceeds its upper bound")
        return self


class ExpertBounds(SemanticDefinition):
    """Version a complete collection of static coefficient boxes.

    Attributes
    ----------
    bounds : tuple of ExpertBound
        One through 4096 unique expert boxes, canonicalized by ID.
    semantics : UnitSemantics
        Required canonical coefficient-fraction meaning, never asset exposure.

    Notes
    -----
    Construction validates declarations only. It does not claim simplex
    feasibility, projection, financial risk control or external execution.
    """

    bounds: Annotated[tuple[ExpertBound, ...], Field(min_length=1, max_length=MAX_MEMBERS)]
    semantics: UnitSemantics

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Require unique exact experts and canonical fraction semantics."""
        _fraction_semantics(self.semantics)
        ordered_references(tuple(bound.member for bound in self.bounds))
        object.__setattr__(self, "bounds", tuple(sorted(self.bounds, key=lambda bound: bound.member.id)))
        return self


class MovementLimit(CoreImmutable):
    """Attach one nonnegative fraction allowance or usage to an expert.

    Attributes
    ----------
    member : Reference
        Exact expert identity/version.
    maximum : float
        Finite nonnegative coefficient fractions. A redundant allowance above
        one is permitted; this is not a probability or share parameter.
    """

    member: Reference
    maximum: Annotated[Binary64, Field(ge=0)]


class MovementAllowance(SemanticDefinition):
    """Declare hard movement limits separately from a soft movement penalty.

    Attributes
    ----------
    limits : tuple of MovementLimit
        Complete unique per-expert allowances, canonicalized by ID.
    scope : {'per_release', 'per_interval'}
        Whether limits restart for each accepted release or one declared period.
    measure : {'net_displacement', 'accumulated_absolute'}
        Net displacement from period origin or total absolute path variation.
        They coincide for a single per-release step.
    semantics : UnitSemantics
        Canonical coefficient-fraction meaning.

    Notes
    -----
    Period usage is supplied explicitly. This declaration cannot reset budgets,
    read a calendar, consume feedback or establish durable idempotency.
    """

    limits: Annotated[tuple[MovementLimit, ...], Field(min_length=1, max_length=MAX_MEMBERS)]
    scope: Literal["per_release", "per_interval"]
    measure: Literal["net_displacement", "accumulated_absolute"]
    semantics: UnitSemantics

    @model_validator(mode="after")
    def _limits(self) -> Self:
        """Validate exact experts and explicit coefficient units."""
        _fraction_semantics(self.semantics)
        ordered_references(tuple(limit.member for limit in self.limits))
        object.__setattr__(self, "limits", tuple(sorted(self.limits, key=lambda limit: limit.member.id)))
        return self


class CoefficientOrigin(SemanticDefinition):
    """Bind a declared emitted coefficient snapshot and its exact version.

    Attributes
    ----------
    weights : tuple of Weight
        Complete simplex including structural zeros, canonicalized by ID.
    profile : {'binary64-simplex-v1'}
        Existing numerical coefficient convention.
    semantics : UnitSemantics
        Canonical coefficient fractions, distinct from holdings/probabilities.

    Notes
    -----
    Reference and lineage identify the supplied snapshot; they do not prove it
    was committed. A runtime owns that frontier. No hidden latent proposal,
    executed holdings, storage, clock or learner state is consulted.
    """

    weights: CoefficientWeights
    profile: SimplexProfile
    semantics: UnitSemantics

    @model_validator(mode="after")
    def _weights(self) -> Self:
        """Retain a canonical finite simplex with exact expert versions."""
        _fraction_semantics(self.semantics)
        object.__setattr__(self, "weights", coefficient_weights(self.weights, self.profile))
        return self


class MovementUsage(CoreImmutable):
    """Supply a period identity, origin and already consumed movement budget.

    Attributes
    ----------
    interval : Reference
        Explicit host period identity/version; never inferred from wall time.
    origin : CoefficientOrigin
        Coefficients at the beginning of this period, distinct from the last
        emitted vector used by an L1 penalty.
    used : tuple of MovementLimit
        Complete per-coordinate usage in coefficient fractions. For accumulated
        variation this is a caller-supplied ledger attestation; net displacement
        is derived from the supplied period/current origins by the checker.

    Notes
    -----
    This immutable snapshot consumes nothing and authenticates no ledger.
    Its interval and complete content must survive retries/checkpoint recovery.
    """

    interval: Reference
    origin: CoefficientOrigin
    used: Annotated[tuple[MovementLimit, ...], Field(min_length=1, max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _usage(self) -> Self:
        """Require complete exact coordinates without inventing zero usage."""
        ordered_references(tuple(limit.member for limit in self.used))
        ordered = tuple(sorted(self.used, key=lambda limit: limit.member.id))
        if tuple(limit.member for limit in ordered) != tuple(weight.member for weight in self.origin.weights):
            raise ValueError("Movement usage requires every exact period-origin expert")
        object.__setattr__(self, "used", ordered)
        return self


class RoundCadence(CoreImmutable):
    """Declare release after a fixed number of accepted complete panels.

    Attributes
    ----------
    kind : {'round_block'}
        Logical feedback-block profile, independent from calendar days.
    panels : int
        Strict positive accepted-panel count. Pending data creates no zero panel.

    Notes
    -----
    The consuming learner aggregates sum losses and applies share/output once
    per nonempty release. This value performs no scheduling or state update.
    """

    kind: Literal["round_block"] = "round_block"
    panels: Annotated[int, Field(strict=True, ge=1)]


class IntervalCadence(CoreImmutable):
    """Declare at most one accepted release per host-defined interval.

    Attributes
    ----------
    kind : {'interval_clock'}
        Explicit host-clock profile, with no embedded market calendar.
    calendar : Reference
        Required identity/version of the externally supplied interval sequence.

    Notes
    -----
    An optimal hold consumes a nonempty release opportunity just like a change.
    Empty intervals perform no exponential, share or penalty step. Runtime
    ordering, atomic release consumption and durable deduplication are external.
    """

    kind: Literal["interval_clock"] = "interval_clock"
    calendar: Reference


class ReleaseInterval(CoreImmutable):
    """Describe one host-supplied interval and its single planned release point.

    Attributes
    ----------
    interval : Reference
        Exact period identity/version.
    start, end : datetime
        Zoned UTC-normalized interval boundaries with start strictly before end.
    cutoff, release_at : datetime
        Explicit inclusion cutoff and release point, satisfying
        start <= cutoff <= release_at <= end. A boundary release at end is
        assigned to this interval by its identity, not inferred membership.

    Notes
    -----
    This declaration reads no clock and proves neither eligibility nor release.
    Older retained panels may be consumed at its cutoff; no zero feedback is
    manufactured for dates or intervals without observations.
    """

    interval: Reference
    start: Timestamp
    end: Timestamp
    cutoff: Timestamp
    release_at: Timestamp

    @model_validator(mode="after")
    def _interval(self) -> Self:
        """Validate explicit boundaries without ambient clock or epoch guesses."""
        if self.start >= self.end or not self.start <= self.cutoff <= self.release_at <= self.end:
            raise ValueError("Release interval requires start <= cutoff <= release <= end and start < end")
        return self


class ProjectionOutput(CoreImmutable):
    """Select the unpenalized constrained squared-Euclidean output contract.

    Attributes
    ----------
    kind : {'projection'}
        Explicit original output profile; no missing penalty field is inferred.

    Notes
    -----
    Actual projection and numerical qualification belong to the method owner.
    """

    kind: Literal["projection"] = "projection"


class L1MovementPenalty(SemanticDefinition):
    """Declare the joint L1 movement-output objective without solving it.

    Attributes
    ----------
    kind : {'l1_movement'}
        Required full-L1 penalty discriminator, distinct from squared distance.
    strength : float
        Finite nonnegative lambda in coefficient-fraction units under the fixed
        objective 0.5*||p-r||_2**2 + lambda*||p-p_old||_1; not limited to [0, 1].
    origin : CoefficientOrigin
        Exact previously emitted/committed vector and its version/lineage.
    semantics : UnitSemantics
        Canonical coefficient fractions; percentages/probabilities reject.
    objective_convention : {'half_squared_plus_full_l1_v1'}
        Fixed half-squared proximity and full L1, never half-L1 turnover.
    arithmetic_profile : {'binary64-rational-v1'}
        Declared arithmetic convention, not a solver optimality certificate.

    Notes
    -----
    A linear coordinate-unit rescaling requires the same linear rescaling of
    lambda. Development-only selection is retained in supplied definition
    lineage; this layer does not authenticate tuning evidence. Future changes
    require explicit outer change validation. A penalty is neither paid fee nor
    realized turnover. No solve, initialization, update, calendar read or I/O
    occurs; the consumer applies it once per accepted nonempty release.
    """

    kind: Literal["l1_movement"] = "l1_movement"
    strength: Annotated[Binary64, Field(ge=0)]
    origin: CoefficientOrigin
    semantics: UnitSemantics
    objective_convention: Literal["half_squared_plus_full_l1_v1"] = "half_squared_plus_full_l1_v1"
    arithmetic_profile: Literal["binary64-rational-v1"] = PROFILE

    @model_validator(mode="after")
    def _penalty(self) -> Self:
        """Keep explicit units and the supported moving-origin convention."""
        _fraction_semantics(self.semantics)
        return self


Cadence = Annotated[RoundCadence | IntervalCadence, Field(discriminator="kind")]
OutputProfile = Annotated[ProjectionOutput | L1MovementPenalty, Field(discriminator="kind")]


class MixtureControls(SemanticDefinition):
    """Bind distinct explicit controls without constructing a mixture learner.

    Attributes
    ----------
    initial_prior : InitialPrior
        Explicit initialization, including permitted structural zeros.
    target : DeclaredTargetShare or None
        Persistent share target, or required null when disabled.
    bounds : ExpertBounds
        Complete static boxes, which may have an empty feasible intersection.
    movement : MovementAllowance or None
        Hard movement limits, or required null when disabled.
    cadence : RoundCadence or IntervalCadence
        One selected release gate; implicit AND-composition is unsupported.
    output : ProjectionOutput or L1MovementPenalty
        Explicit output profile; missing field cannot silently mean projection.

    Raises
    ------
    ValueError
        Conflicting expert identities/versions or incompatible declarations.

    Notes
    -----
    The method owner validates initialization feasibility and implements the
    actual share/output recurrence, release counters and checkpoint commit.
    Construction neither repairs a soft target nor rejects it for lying outside
    hard bounds. It performs no learning, inference, durable deduplication or I/O.
    """

    initial_prior: InitialPrior
    target: DeclaredTargetShare | None
    bounds: ExpertBounds
    movement: MovementAllowance | None
    cadence: Cadence
    output: OutputProfile

    @model_validator(mode="after")
    def _members(self) -> Self:
        """Require one exact catalogue across independently meaningful controls."""
        expected = tuple(bound.member for bound in self.bounds.bounds)
        _same_members(expected, tuple(weight.member for weight in self.initial_prior.weights))
        if self.target is not None:
            _same_members(expected, tuple(weight.member for weight in self.target.weights))
        if self.movement is not None:
            _same_members(expected, tuple(limit.member for limit in self.movement.limits))
        if isinstance(self.output, L1MovementPenalty):
            _same_members(expected, tuple(weight.member for weight in self.output.origin.weights))
        return self


class EffectiveExpertBound(CoreImmutable):
    """Retain computed limits, including a well-formed empty intersection.

    Attributes
    ----------
    member : Reference
        Exact coordinate identity/version.
    lower, upper : float
        Effective fraction limits. Lower greater than upper represents an empty
        intersection, never a malformed original individual bound.
    """

    member: Reference
    lower: Binary64
    upper: Binary64


class MixtureIntersection(CoreContract):
    """Return finite feasibility evidence without projecting coefficients.

    Attributes
    ----------
    bounds, origin : DefinitionBinding
        Exact static-bound and current emitted-origin definitions.
    bounds_definition : ExpertBounds
        Complete static declaration used to reproduce effective bounds.
    origin_definition : CoefficientOrigin
        Complete current origin used to reproduce movement constraints.
    movement_definition : MovementAllowance or None
        Complete hard movement declaration, or required null when disabled.
    movement : DefinitionBinding or None
        Hard movement definition, independently absent from any soft penalty.
    usage : MovementUsage or None
        Explicit period attestation, retained verbatim when supplied.
    effective : tuple of EffectiveExpertBound
        Canonical computed boxes, retaining conflicting limits when infeasible.
    status : {'feasible', 'infeasible'}
        Numerical admission under the existing named simplex residual bound.
    exact_feasible : bool
        Whether original rational effective bounds also admit exact sum one
        without a violated budget. False with feasible status is numerical admission,
        never a proof that the exact-real simplex intersection is nonempty.
    numerical_profile : {'binary64-simplex-v1'}
        Exact rational sum-residual tolerance 2**-49; no coordinate relaxation.
    arithmetic_profile : {'binary64-rational-v1'}
        Effective endpoints are guarded binary64 projections of rational math.
    reason : {'nonempty', 'empty_coordinate', 'simplex_sum', 'period_budget'}
        Deterministic evidence category; no financial or global-optimum claim.

    Notes
    -----
    Exact/numerical feasibility is recomputed from the complete supplied inputs.
    Display endpoints are guarded binary64 projections; they cannot hide an
    original reversed coordinate or tiny overspent budget. Sum-only residuals
    may be admitted by the named profile. No KKT solution is claimed and nothing
    is normalized or repaired. Input attestations are not authenticated.
    No solver, repaired output, event consumption, state mutation or I/O occurs.
    """

    bounds: DefinitionBinding
    origin: DefinitionBinding
    movement: DefinitionBinding | None
    bounds_definition: ExpertBounds
    origin_definition: CoefficientOrigin
    movement_definition: MovementAllowance | None
    usage: MovementUsage | None
    effective: Annotated[tuple[EffectiveExpertBound, ...], Field(min_length=1, max_length=MAX_MEMBERS)]
    status: Literal["feasible", "infeasible"]
    exact_feasible: Annotated[bool, Field(strict=True)]
    numerical_profile: SimplexProfile = "binary64-simplex-v1"
    arithmetic_profile: Literal["binary64-rational-v1"] = PROFILE
    reason: Literal["nonempty", "empty_coordinate", "simplex_sum", "period_budget"]

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Reproduce all facts, including budget facts lost in endpoint rounding."""
        ordered_references(tuple(bound.member for bound in self.effective))
        ordered = tuple(sorted(self.effective, key=lambda bound: bound.member.id))
        object.__setattr__(self, "effective", ordered)
        effective, reason, exact = _mixture_facts(
            self.bounds_definition, self.origin_definition, self.movement_definition, self.usage
        )
        expected_movement = None if self.movement_definition is None else self.movement_definition.binding
        expected_bindings = (self.bounds_definition.binding, self.origin_definition.binding, expected_movement)
        actual = [self.reason, self.exact_feasible, self.status, [bound.model_dump(mode="json") for bound in ordered]]
        expected = [
            reason,
            exact,
            "feasible" if reason == "nonempty" else "infeasible",
            [bound.model_dump(mode="json") for bound in effective],
        ]
        if (self.bounds, self.origin, self.movement) != expected_bindings or canonical_json(actual) != canonical_json(
            expected
        ):
            raise ValueError("Mixture intersection does not reproduce its full definitions and effective boxes")
        return self


def _fraction_semantics(semantics: UnitSemantics) -> None:
    """Reject ambiguous probability, percent and other coefficient meanings."""
    if semantics != COEFFICIENT_SEMANTICS:
        raise ValueError("Controls require canonical coefficient-fraction semantics")


def _movement_limits(
    origin: CoefficientOrigin, movement: MovementAllowance, usage: MovementUsage | None
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Derive per-coordinate movement intervals without consuming a budget."""
    if movement.scope == "per_release":
        if usage is not None:
            raise ValueError("Per-release movement cannot silently use period usage")
        centers = tuple(Fraction(weight.value) for weight in origin.weights)
        remaining = tuple(Fraction(limit.maximum) for limit in movement.limits)
    else:
        if usage is None:
            raise ValueError("Per-interval movement requires explicit period origin and usage")
        _same_members(
            tuple(weight.member for weight in origin.weights), tuple(weight.member for weight in usage.origin.weights)
        )
        centers, remaining = _period_limits(origin, movement, usage)
    return tuple((center - allowance, center + allowance) for center, allowance in zip(centers, remaining, strict=True))


def _period_limits(
    origin: CoefficientOrigin, movement: MovementAllowance, usage: MovementUsage
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Keep net displacement and accumulated absolute variation distinct."""
    if movement.measure == "net_displacement":
        centers = tuple(Fraction(weight.value) for weight in usage.origin.weights)
        for current, start, used in zip(origin.weights, usage.origin.weights, usage.used, strict=True):
            expected = checked_fraction(abs(Fraction(current.value) - Fraction(start.value)))
            if used.maximum != expected:
                raise ValueError("Net-displacement usage disagrees with the declared origins")
        remaining = tuple(Fraction(limit.maximum) for limit in movement.limits)
    else:
        centers = tuple(Fraction(weight.value) for weight in origin.weights)
        for current, start, used in zip(origin.weights, usage.origin.weights, usage.used, strict=True):
            minimum = checked_fraction(abs(Fraction(current.value) - Fraction(start.value)))
            if used.maximum < minimum:
                raise ValueError("Accumulated usage cannot be below known net displacement")
        remaining = tuple(
            Fraction(limit.maximum) - Fraction(used.maximum)
            for limit, used in zip(movement.limits, usage.used, strict=True)
        )
    return centers, remaining


def check_mixture_intersection(
    bounds: ExpertBounds,
    origin: CoefficientOrigin,
    *,
    movement: MovementAllowance | None = None,
    usage: MovementUsage | None = None,
) -> MixtureIntersection:
    """Check declared boxes, movement and simplex without solving for weights.

    Parameters
    ----------
    bounds : ExpertBounds
        Well-formed static coefficient boxes, possibly jointly infeasible.
    origin : CoefficientOrigin
        Exact previously emitted coefficient snapshot.
    movement : MovementAllowance or None, optional
        Hard per-release or explicitly recorded per-interval allowance.
    usage : MovementUsage or None, optional
        Required period origin/usage for per-interval movement only.

    Returns
    -------
    MixtureIntersection
        Pure exact/numerical feasibility evidence under binary64-simplex-v1;
        it never emits a replacement vector or claims an exact KKT solution.

    Raises
    ------
    ValueError
        Invalid/incompatible declarations, missing period usage, inconsistent
        net usage or nonrepresentable operation results.

    Notes
    -----
    A negative remaining accumulated budget is typed infeasible. Net limits use
    the interval origin; accumulated limits use the last emitted origin with the
    remaining allowance. No retries, interval resets, projection, penalty solve,
    hidden clock, learner mutation or external effect occurs.
    """
    declared = ExpertBounds.model_validate(bounds)
    current = CoefficientOrigin.model_validate(origin)
    checked_movement = None if movement is None else MovementAllowance.model_validate(movement)
    checked_usage = None if usage is None else MovementUsage.model_validate(usage)
    effective, reason, exact = _mixture_facts(declared, current, checked_movement, checked_usage)
    return MixtureIntersection(
        schema_version=1,
        bounds=declared.binding,
        origin=current.binding,
        movement=None if checked_movement is None else checked_movement.binding,
        bounds_definition=declared,
        origin_definition=current,
        movement_definition=checked_movement,
        usage=checked_usage,
        effective=effective,
        status="feasible" if reason == "nonempty" else "infeasible",
        exact_feasible=exact,
        reason=reason,
    )


def _mixture_facts(
    bounds: ExpertBounds,
    origin: CoefficientOrigin,
    movement: MovementAllowance | None,
    usage: MovementUsage | None,
) -> tuple[
    tuple[EffectiveExpertBound, ...], Literal["nonempty", "empty_coordinate", "simplex_sum", "period_budget"], bool
]:
    """Derive original rational feasibility before projecting display endpoints."""
    intervals, limits = _effective_intervals(bounds, origin, movement, usage)
    effective = tuple(
        EffectiveExpertBound(
            member=bound.member,
            lower=checked_fraction(interval[0]),
            upper=checked_fraction(interval[1]),
        )
        for bound, interval in zip(bounds.bounds, intervals, strict=True)
    )
    return effective, _intersection_reason(intervals, limits), _exact_intersection(intervals)


def _effective_intervals(
    bounds: ExpertBounds,
    origin: CoefficientOrigin,
    movement: MovementAllowance | None,
    usage: MovementUsage | None,
    exact_path: ExactPathUsage | None = None,
) -> tuple[tuple[tuple[Fraction, Fraction], ...], tuple[tuple[Fraction, Fraction], ...]]:
    """Share original geometry while preserving the legacy validation sequence."""
    expected = tuple(bound.member for bound in bounds.bounds)
    _same_members(expected, tuple(weight.member for weight in origin.weights))
    if exact_path is not None:
        if movement is None or usage is not None:
            raise ValueError("Exact path usage requires movement and excludes legacy usage")
        limits = _path_intervals(origin, movement, exact_path)
    elif movement is None:
        if usage is not None:
            raise ValueError("Period usage requires a declared movement allowance")
        limits = tuple((Fraction(0), Fraction(1)) for _ in bounds.bounds)
    else:
        _same_members(expected, tuple(limit.member for limit in movement.limits))
        limits = _movement_limits(origin, movement, usage)
    intervals = tuple(
        (max(Fraction(bound.lower), interval[0]), min(Fraction(bound.upper), interval[1]))
        for bound, interval in zip(bounds.bounds, limits, strict=True)
    )
    return intervals, limits


EXACT_PATH_PROFILE = "emitted-binary64-absolute-variation-grid-v1"
EXACT_INTERVAL_PROFILE = "stored-binary64-effective-intervals-v1"
MAX_CONTROL_BYTES = 16 * 1024 * 1024


class ExactPathUsage(CoreContract):
    """Bind exact per-period variation to its supplied emission frontier.

    Attributes
    ----------
    profile : {'emitted-binary64-absolute-variation-grid-v1'}
        Fixed quantum and full absolute emitted-coordinate variation.
    semantics : UnitSemantics
        Canonical coefficient-fraction units.
    interval : Reference
        Explicit period identity, never inferred from time or a restart.
    movement : MovementAllowance
        Complete per-interval accumulated-absolute allowance definition.
    period_origin : CoefficientOrigin
        Complete actual emitted origin at the start of this period.
    through_origin : DefinitionBinding
        Exact current emitted snapshot, including original signed-zero bytes.
    through_release : Reference or None
        Last successful nonempty release, or null for the initial zero ledger.
    used : tuple of ExactCoefficientVariation
        Complete canonical member counts, each no greater than its allowance.

    Notes
    -----
    Imports attest to history; the owner preserves monotone period/releases.
    Content is capped at 16 MiB before parsing counts; 4096 members times 632
    digits need not fit a smaller state envelope. MovementUsage v1 is unchanged.
    """

    profile: Literal["emitted-binary64-absolute-variation-grid-v1"] = EXACT_PATH_PROFILE
    semantics: UnitSemantics
    interval: Reference
    movement: MovementAllowance
    period_origin: CoefficientOrigin
    through_origin: DefinitionBinding
    through_release: Reference | None
    used: Annotated[tuple[ExactCoefficientVariation, ...], Field(min_length=1, max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _path(self) -> Self:
        """Bound exact counts and bind the complete declared accounting profile."""
        _fraction_semantics(self.semantics)
        if (self.movement.scope, self.movement.measure) != ("per_interval", "accumulated_absolute"):
            raise ValueError("Exact path usage requires per-interval accumulated absolute movement")
        ordered_references(tuple(item.member for item in self.used))
        ordered = tuple(sorted(self.used, key=lambda item: item.member.id))
        expected = tuple(weight.member for weight in self.period_origin.weights)
        _same_members(expected, tuple(item.member for item in ordered))
        _same_members(expected, tuple(item.member for item in self.movement.limits))
        object.__setattr__(self, "used", ordered)
        if len(canonical_json(self.model_dump(mode="json")).encode("utf-8")) > MAX_CONTROL_BYTES:
            raise ValueError("Exact movement content exceeds the 16 MiB byte ceiling")
        for item, limit in zip(ordered, self.movement.limits, strict=True):
            if int(item.atoms_decimal) > _binary64_atoms(limit.maximum):
                raise ValueError("Exact accumulated movement exceeds its declared allowance")
        if self.through_release is None and (
            self.through_origin != self.period_origin.binding or any(int(item.atoms_decimal) for item in ordered)
        ):
            raise ValueError("An initial exact ledger requires its period origin and zero usage")
        return self

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Read bounded versioned exact movement content without migration.

        Parameters
        ----------
        payload : str or bytes
            Complete duplicate-key guarded JSON, at most 16 MiB and depth 64.

        Returns
        -------
        Self
            Revalidated exact accounting attestation.

        Raises
        ------
        ValueError, TypeError
            Invalid shape, version, units, identities, counts or capacity.
        """
        return super().from_json(bounded_json_bytes(payload, MAX_CONTROL_BYTES))

    @classmethod
    def model_validate_json(  # pylint: disable=too-many-arguments
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Bound alternate JSON ingress before decoding or count conversion.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete UTF-8 content at most 16 MiB/depth 64; bytearrays copy only
            after byte admission.
        strict : {True} or None, optional
            Existing core strict mode; False cannot relax this contract.
        extra : {'forbid'} or None, optional
            Existing unknown-field policy; relaxation rejects.
        context : object or None, optional
            Explicit validation context, never persisted.
        by_alias, by_name : bool or None, optional
            Standard core field-name selection.

        Returns
        -------
        Self
            Revalidated versioned exact accounting attestation.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Bounds, UTF-8, duplicate keys, versions or typed invariants fail.
        """
        if type(json_data) is bytearray:  # pylint: disable=unidiomatic-typecheck
            if len(json_data) > MAX_CONTROL_BYTES:
                raise ValueError("Exact movement JSON exceeds the 16 MiB byte ceiling")
            json_data = bytes(json_data)
        if not isinstance(json_data, str | bytes):
            raise TypeError("Exact movement JSON requires built-in str, bytes or bytearray")
        encoded = bounded_json_bytes(json_data, MAX_CONTROL_BYTES)
        return super().model_validate_json(
            encoded, strict=strict, extra=extra, context=context, by_alias=by_alias, by_name=by_name
        )


def _path_remaining(
    origin: CoefficientOrigin, movement: MovementAllowance, usage: ExactPathUsage
) -> tuple[Fraction, ...]:
    """Join complete definitions before using the shared exact path arithmetic."""
    if usage.movement.binding != movement.binding or usage.through_origin != origin.binding:
        raise ValueError("Exact path usage does not bind this movement law and current origin")
    expected = tuple(weight.member for weight in origin.weights)
    _same_members(expected, tuple(weight.member for weight in usage.period_origin.weights))
    _same_members(expected, tuple(limit.member for limit in movement.limits))
    return _exact_path_remaining(
        tuple(weight.value for weight in origin.weights),
        tuple(weight.value for weight in usage.period_origin.weights),
        tuple(item.atoms_decimal for item in usage.used),
        tuple(limit.maximum for limit in movement.limits),
    )


def _path_intervals(
    origin: CoefficientOrigin, movement: MovementAllowance, usage: ExactPathUsage
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Center authoritative path allowances on the actual current emission."""
    return tuple(
        (Fraction(weight.value) - allowance, Fraction(weight.value) + allowance)
        for weight, allowance in zip(origin.weights, _path_remaining(origin, movement, usage), strict=True)
    )


def initial_exact_path_usage(
    *, interval: Reference, movement: MovementAllowance, origin: CoefficientOrigin
) -> ExactPathUsage:
    """Create an explicit zero ledger for a newly admitted period.

    Parameters
    ----------
    interval : Reference
        Host-admitted new period identity and version.
    movement : MovementAllowance
        Complete per-interval accumulated-absolute law.
    origin : CoefficientOrigin
        Actual emitted coefficients at the period boundary.

    Returns
    -------
    ExactPathUsage
        Zero counts with the complete period origin and null release frontier.

    Raises
    ------
    ValueError
        Incompatible declarations, units, members or capacity.

    Notes
    -----
    The owner admits new periods and rejects resets, retries and rewinds.
    """
    current = CoefficientOrigin.model_validate(origin)
    declared = MovementAllowance.model_validate(movement)
    return ExactPathUsage(
        schema_version=1,
        semantics=COEFFICIENT_SEMANTICS,
        interval=Reference.model_validate(interval),
        movement=declared,
        period_origin=current,
        through_origin=current.binding,
        through_release=None,
        used=tuple(ExactCoefficientVariation(member=weight.member, atoms_decimal="0") for weight in current.weights),
    )


def advance_exact_path_usage(
    *, usage: ExactPathUsage, origin: CoefficientOrigin, emitted: CoefficientOrigin, release: Reference
) -> ExactPathUsage:
    """Charge actual emitted binary64 deltas once to a supplied exact ledger.

    Parameters
    ----------
    usage : ExactPathUsage
        Complete previously retained period ledger.
    origin : CoefficientOrigin
        Exact current origin bound by the ledger.
    emitted : CoefficientOrigin
        Candidate next committed emitted vector, including its version.
    release : Reference
        New successful nonempty release identity; immediate repeats reject.

    Returns
    -------
    ExactPathUsage
        New immutable counts and frontier, including zero charges for holds.

    Raises
    ------
    ValueError
        Wrong joins, repeated release, underspecified history or overspending.

    Notes
    -----
    The owner atomically publishes coefficients, ledger and consumed releases,
    and rejects nonadjacent old releases/period resets. Failed proposals, held
    deliveries and empty intervals do not call this operation.
    """
    previous = ExactPathUsage.model_validate(usage)
    current = CoefficientOrigin.model_validate(origin)
    following = CoefficientOrigin.model_validate(emitted)
    next_release = Reference.model_validate(release)
    _path_remaining(current, previous.movement, previous)
    _same_members(
        tuple(weight.member for weight in current.weights), tuple(weight.member for weight in following.weights)
    )
    if next_release == previous.through_release:
        raise ValueError("Exact movement cannot charge the same release twice")
    used = tuple(
        ExactCoefficientVariation(
            member=old.member,
            atoms_decimal=str(int(item.atoms_decimal) + abs(_binary64_atoms(new.value) - _binary64_atoms(old.value))),
        )
        for old, new, item in zip(current.weights, following.weights, previous.used, strict=True)
    )
    return ExactPathUsage(
        schema_version=1,
        semantics=previous.semantics,
        interval=previous.interval,
        movement=previous.movement,
        period_origin=previous.period_origin,
        through_origin=following.binding,
        through_release=next_release,
        used=used,
    )


def exact_mixture_intervals(
    bounds: ExpertBounds,
    origin: CoefficientOrigin,
    *,
    movement: MovementAllowance | None = None,
    usage: MovementUsage | None = None,
    exact_path: ExactPathUsage | None = None,
) -> ExactMixtureIntervals[MixtureIntersection]:
    """Derive exact shared boxes for projection and joint L1 output.

    Parameters
    ----------
    bounds : ExpertBounds
        Complete well-formed static boxes.
    origin : CoefficientOrigin
        Actual current emitted vector and exact version.
    movement : MovementAllowance or None, optional
        Explicit hard movement law, independently absent from soft penalties.
    usage : MovementUsage or None, optional
        Existing legacy attestation with unchanged v1 validation semantics.
    exact_path : ExactPathUsage or None, optional
        Authoritative exact emitted-path usage, mutually exclusive with usage.

    Returns
    -------
    ExactMixtureIntervals
        Exact-S disposition and original rational endpoints. A legacy feasible
        status with exact_feasible false remains separately visible.

    Raises
    ------
    ValueError
        Malformed or incompatible declarations, joins or bounded content.

    Notes
    -----
    This operation emits no coefficients and consumes no release. Exact path
    intervals and the ledger updater use the same unrounded atom accounting.
    """
    declared = ExpertBounds.model_validate(bounds)
    current = CoefficientOrigin.model_validate(origin)
    checked_movement = None if movement is None else MovementAllowance.model_validate(movement)
    checked_usage = None if usage is None else MovementUsage.model_validate(usage)
    checked_path = None if exact_path is None else ExactPathUsage.model_validate(exact_path)
    payload = {
        "profile": EXACT_INTERVAL_PROFILE,
        "bounds": declared.model_dump(mode="json"),
        "origin": current.model_dump(mode="json"),
        "movement": None if checked_movement is None else checked_movement.model_dump(mode="json"),
        "usage": None if checked_usage is None else checked_usage.model_dump(mode="json"),
        "exact_path": None if checked_path is None else checked_path.model_dump(mode="json"),
    }
    if len(canonical_json(payload).encode("utf-8")) > MAX_CONTROL_BYTES:
        raise ValueError("Exact geometry exceeds the 16 MiB complete-input byte ceiling")
    intervals, limits = _effective_intervals(declared, current, checked_movement, checked_usage, checked_path)
    reason = _intersection_reason(intervals, limits)
    if reason == "nonempty" and not _exact_intersection(intervals):
        reason = "simplex_sum"
    legacy = None
    if checked_path is None:
        legacy = check_mixture_intersection(declared, current, movement=checked_movement, usage=checked_usage)
    return ExactMixtureIntervals(
        profile=EXACT_INTERVAL_PROFILE,
        inputs_fingerprint=fingerprint(payload),
        intervals=tuple(
            ExactExpertInterval(bound.member, lower, upper)
            for bound, (lower, upper) in zip(declared.bounds, intervals, strict=True)
        ),
        status="feasible" if reason == "nonempty" else "infeasible",
        reason=reason,
        legacy_intersection=legacy,
    )
