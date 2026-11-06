"""Causal frozen sample scales and signed-outcome losses without a runtime."""

from __future__ import annotations

import struct
from collections import Counter
from collections.abc import Iterable
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.feedback import OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, Scalar, ScalarSpec
from coleman.core.primitives import MAX_MEMBERS, Fingerprint, canonical_json
from coleman.core.temporal import Timestamp
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.objectives.arithmetic import checked_fraction
from coleman.objectives.bindings import (
    DefinitionBinding,
    QuantityMeaning,
    SemanticDefinition,
    _project_loss_outcome,
    require_quantity,
)

SD_PROFILE = "binary64-rational-sample-sd-v1"
ScaleStatus = Literal["available", "zero_scale", "insufficient_support"]
Clipping = Literal["none", "lower", "upper", "not_evaluated"]
LossReason = Literal[
    "computed",
    "input_unknown",
    "input_unavailable",
    "input_not_applicable",
    "zero_scale",
    "insufficient_support",
    "stale_scale",
]


def _float_at(bits: int) -> float:
    """Decode monotonically ordered nonnegative finite binary64 bits."""
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _root_floor_bits(value: Fraction) -> int:
    """Find the greatest finite nonnegative float whose exact square fits."""
    low, high = 0, 0x7FEFFFFFFFFFFFFF
    while low < high:
        middle = (low + high + 1) // 2
        if Fraction.from_float(_float_at(middle)) ** 2 <= value:
            low = middle
        else:
            high = middle - 1
    return low


def _root_above_maximum(value: Fraction, lower: Fraction) -> float:
    """Distinguish rounding to the finite maximum from overflow to infinity."""
    previous = Fraction.from_float(_float_at(0x7FEFFFFFFFFFFFFE))
    half_spacing = (lower - previous) / 2
    if value >= (lower + half_spacing) ** 2:
        raise ValueError("Sample standard deviation overflows binary64")
    return float(lower)


def _rounded_sqrt(value: Fraction) -> float:
    """Round an exact nonnegative rational root with midpoint-square comparisons."""
    if value == 0:
        return 0.0
    low = _root_floor_bits(value)
    lower = Fraction.from_float(_float_at(low))
    if lower * lower == value:
        return float(lower)
    if low == 0x7FEFFFFFFFFFFFFF:
        return _root_above_maximum(value, lower)
    upper = Fraction.from_float(_float_at(low + 1))
    midpoint_square = ((lower + upper) / 2) ** 2
    choose_lower = value < midpoint_square or (value == midpoint_square and low % 2 == 0)
    result = _float_at(low if choose_lower else low + 1)
    if result == 0:
        raise ValueError("Nonzero sample standard deviation underflows binary64")
    return result


class FitSample(CoreImmutable):
    """Supply one historical observation revision and its actual visibility clocks.

    Attributes
    ----------
    observation : Reference
        Stable logical observation ID and exact revision/version.
    value : Scalar
        Signed finite value or an explicit null state, never an imputed zero.
    timing : FeedbackTiming
        Event, availability, maturity and independently known receipt clocks.

    Notes
    -----
    Source identities and clocks are supplied attestations, not authentication.
    Revision selection belongs to the caller; a fit rejects duplicate logical IDs.
    No source access, delivery acceptance or observation mutation occurs.
    """

    observation: Reference
    value: Scalar
    timing: FeedbackTiming


class FitSnapshot(CoreContract):
    """Retain a bounded historical view eligible at one explicit training cutoff.

    Attributes
    ----------
    source : Reference
        Exact declared source/schema identity; not a fingerprint of future data.
    quantity : QuantityMeaning
        Shared scalar schema, interpretation and measurement horizon.
    member : Reference or None
        Exact expert for an indexed scale, or null for a shared scale.
    training_cutoff : datetime
        UTC information boundary; every row has known receipt and is usable by it.
    samples : tuple of FitSample
        At most 4096 supplied revisions, canonical by logical observation ID.

    Raises
    ------
    ValueError
        Duplicate IDs, conflicting schemas, unknown receipt or post-cutoff facts.

    Notes
    -----
    A selected view attests supplied completeness only. Future history is not an
    input; append it outside this immutable snapshot. No replay engine or source
    correction selection is implemented. Missing rows remain explicit.
    """

    source: Reference
    quantity: QuantityMeaning
    member: Reference | None
    training_cutoff: Timestamp
    samples: Annotated[tuple[FitSample, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _eligible(self) -> Self:
        """Reject duplicate logical rows, incompatible values and unavailable fits."""
        ids = [sample.observation.id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Fit snapshot requires one revision per logical observation")
        for sample in self.samples:
            require_quantity(sample.value, self.quantity)
            usable = sample.timing.usable_at
            if usable is None or usable > self.training_cutoff:
                raise ValueError("Fit sample is not received, mature and available by the cutoff")
        object.__setattr__(self, "samples", tuple(sorted(self.samples, key=lambda sample: sample.observation.id)))
        return self


class RollingScaleSpec(SemanticDefinition):
    """Define a full-window past-only sample standard deviation with ddof one.

    Attributes
    ----------
    quantity : QuantityMeaning
        Raw signed outcome or monetary quantity and its explicit horizon.
    scale_spec : ScalarSpec
        Separate diagnostic dimension/meaning, same physical unit, lower bound 0.
    window : int
        Two through 4096 observed historical rows required for a complete window.
    scope : {'shared', 'per_member'}
        Shared series or explicitly indexed expert series; never inferred.
    maximum_age_seconds : float or None
        Optional nonnegative event-age limit at the producing information cutoff.
        Null declares no freshness guarantee, rather than infinity or zero.
    profile : str, optional
        Fixed rational-variance and binary64-root arithmetic identity.

    Raises
    ------
    ValueError
        Invalid window, non-outcome interpretation or incompatible scale units.

    Notes
    -----
    Select the latest ``window`` observed rows by event time then observation ID;
    exclude nulls without imputation and require the full observed support. Compute
    the centered sample variance with ddof=1; the eventual outcome is not demeaned.
    Support count is not statistical independence or calibrated risk evidence.
    """

    quantity: QuantityMeaning
    scale_spec: ScalarSpec
    window: Annotated[int, Field(strict=True, ge=2, le=MAX_MEMBERS)]
    scope: Literal["shared", "per_member"]
    maximum_age_seconds: Annotated[Binary64, Field(ge=0)] | None
    profile: Literal["binary64-rational-sample-sd-v1"] = SD_PROFILE

    @model_validator(mode="after")
    def _schema(self) -> Self:
        """Require a distinct nonnegative diagnostic scale in original outcome units."""
        if self.quantity.interpretation not in {"raw_outcome", "monetary_amount"}:
            raise ValueError("Rolling outcome scale requires raw or monetary outcome meaning")
        if (
            self.scale_spec.role != "diagnostic"
            or self.scale_spec.minimum != 0
            or self.scale_spec.semantics.unit != self.quantity.spec.semantics.unit
            or self.scale_spec.dimension.id == self.quantity.spec.dimension.id
        ):
            raise ValueError("Scale requires a distinct diagnostic schema with minimum zero and identical units")
        return self


def _selected(definition: RollingScaleSpec, snapshot: FitSnapshot) -> tuple[FitSample, ...]:
    """Select the declared full observed window deterministically without repair."""
    if snapshot.quantity.model_dump_json() != definition.quantity.model_dump_json():
        raise ValueError("Scale input quantity/horizon differs from its fit snapshot")
    if (definition.scope == "shared") != (snapshot.member is None):
        raise ValueError("Scale scope and fit member disagree")
    observed = (sample for sample in snapshot.samples if sample.value.status == "observed")
    ordered = sorted(observed, key=lambda sample: (sample.timing.event_at, sample.observation.id))
    return tuple(ordered[-definition.window :])


def _scale_result(
    definition: RollingScaleSpec, snapshot: FitSnapshot
) -> tuple[tuple[Reference, ...], Scalar, ScaleStatus]:
    """Compute exact centered variance and one guarded rounded square root."""
    rows = _selected(definition, snapshot)
    refs = tuple(sample.observation for sample in rows)
    if len(rows) != definition.window:
        return refs, Scalar(spec=definition.scale_spec, value=None, status="unavailable"), "insufficient_support"
    values = tuple(Fraction.from_float(sample.value.value) for sample in rows if sample.value.value is not None)
    mean = sum(values, Fraction()) / len(values)
    variance = sum(((value - mean) ** 2 for value in values), Fraction()) / (len(values) - 1)
    scale = _rounded_sqrt(variance)
    result = Scalar(spec=definition.scale_spec, value=scale, status="observed")
    return refs, result, "zero_scale" if scale == 0 else "available"


class FrozenScale(CoreContract):
    """Retain complete immutable scale inputs and verify derived facts on reading.

    Attributes
    ----------
    definition : RollingScaleSpec
        Exact method, units, horizon, window, scope and numerical profile.
    snapshot : FitSnapshot
        Bounded eligible historical input; never the future producing outcome.
    frozen_at : datetime
        UTC freeze instant at or after the supplied training cutoff.
    selected : tuple of Reference
        Selected observed rows in event/ID order, including partial support.
    scale : Scalar
        Observed nonnegative same-unit SD, or unavailable/null for insufficient
        support. An observed zero is retained separately from null.
    status : {'available', 'zero_scale', 'insufficient_support'}
        Computation state; zero scale cannot authorize division.

    Raises
    ------
    ValueError
        Invalid chronology, altered selected rows/scale/status or arithmetic failure.

    Notes
    -----
    Guarded readers and serialization recompute the window, sample variance and
    output, preventing an unchecked serialized scale from claiming different facts.
    ``binary64-rational-sample-sd-v1`` uses exact rational variance on stored inputs;
    its root compares exact squared adjacent-float midpoints and rounds ties even.
    Internal variance is not prematurely projected, so tiny representable SDs
    survive. No durable store, mutable rolling buffer, clocks or I/O are used.
    """

    definition: RollingScaleSpec
    snapshot: FitSnapshot
    frozen_at: Timestamp
    selected: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]
    scale: Scalar
    status: ScaleStatus

    @model_validator(mode="after")
    def _facts(self) -> Self:
        """Recompute all selected and derived scale facts before accepting a snapshot."""
        if self.snapshot.training_cutoff > self.frozen_at:
            raise ValueError("Scale freeze precedes its training cutoff")
        declared_at = self.definition.lineage.declared_at
        if declared_at is not None and declared_at > self.frozen_at:
            raise ValueError("Scale definition was declared after its claimed freeze")
        selected, scale, status = _scale_result(self.definition, self.snapshot)
        if (
            self.selected != selected
            or self.scale.model_dump_json() != scale.model_dump_json()
            or self.status != status
        ):
            raise ValueError("Frozen scale facts differ from the supplied historical inputs")
        return self


def fit_rolling_scale(definition: RollingScaleSpec, snapshot: FitSnapshot, *, frozen_at: Timestamp) -> FrozenScale:
    """Fit a complete rolling sample SD and retain its immutable continuation input.

    Parameters
    ----------
    definition : RollingScaleSpec
        Explicit full-window/ddof-one method and same-unit output declaration.
    snapshot : FitSnapshot
        Eligible same-schema history, with known availability/maturity/receipt.
    frozen_at : datetime
        Explicit UTC-compatible freeze time, no earlier than training cutoff.

    Returns
    -------
    FrozenScale
        Complete validated facts, including zero/insufficient-support status.

    Raises
    ------
    ValueError
        Invalid schemas/clocks or nonfinite/overflow/nonzero-underflow output.

    Notes
    -----
    This pure call fits supplied evidence only. Restoring its guarded JSON gives
    identical facts; extending history requires a new snapshot and fingerprint.
    No fitted mean is subtracted from a later outcome and no learner is updated.
    """
    checked = RollingScaleSpec.model_validate(definition)
    history = FitSnapshot.model_validate(snapshot)
    selected, scale, status = _scale_result(checked, history)
    return FrozenScale(
        schema_version=1,
        definition=checked,
        snapshot=history,
        frozen_at=frozen_at,
        selected=selected,
        scale=scale,
        status=status,
    )


class SignedVolatilitySpec(SemanticDefinition):
    """Declare signed same-unit outcome scaling and an explicit clipped loss.

    Attributes
    ----------
    kind : {'signed_volatility'}, optional
        Fixed discriminant, distinct from identity and affine loss definitions.
    input : QuantityMeaning
        Raw signed outcome or monetary quantity, with original bounds and horizon.
    output : QuantityMeaning
        Separate normalized-loss dimension with exact unit ``1`` and bounds [0,1].
    scale_definition : DefinitionBinding
        Exact rolling-scale definition, including its scope/window/units/profile.
    c : float
        Strictly positive dimensionless development-selected clipping strength.

    Raises
    ------
    ValueError
        Invalid interpretations, output bounds, horizons or strength.

    Notes
    -----
    For x and positive s in identical units, z=x/s and loss=clip((1-z/c)/2,0,1).
    Positive outcomes lower this declared surrogate; observed zero maps to 0.5.
    No raw-return bound, calibration, accounting rewrite or efficacy is implied.
    """

    kind: Literal["signed_volatility"] = "signed_volatility"
    input: QuantityMeaning
    output: QuantityMeaning
    scale_definition: DefinitionBinding
    c: Annotated[Binary64, Field(gt=0)]

    @model_validator(mode="after")
    def _schema(self) -> Self:
        """Keep outcome/loss meanings and the original measurement horizon separate."""
        if self.input.interpretation not in {"raw_outcome", "monetary_amount"}:
            raise ValueError("Signed scaling requires raw or monetary outcome interpretation")
        if (
            self.output.interpretation != "normalized_loss"
            or self.output.spec.minimum != 0
            or self.output.spec.maximum != 1
            or self.input.horizon != self.output.horizon
            or self.input.spec.dimension.id == self.output.spec.dimension.id
        ):
            raise ValueError("Signed scaling requires a separate full [0,1] loss at the same horizon")
        return self


class ScaleSelection(CoreContract):
    """Freeze the exact scale/transform for one producing decision and outcome.

    Attributes
    ----------
    definition : SignedVolatilitySpec
        Exact loss definition and selected scale-protocol binding.
    scale : FrozenScale
        Immutable previously fitted scale with full supplied historical evidence.
    decision_fingerprint : str
        Producing request fingerprint, or caller-declared direct boundary digest.
    timing : DecisionTiming
        Producing decision clocks; later feedback application is not a fit cutoff.
    outcome : Reference
        Expected logical outcome/revision; its logical ID cannot enter this fit.
    member : Reference or None
        Expert identity for indexed scales; shared scales may serve declared members.

    Raises
    ------
    ValueError
        Post-cutoff fit/freeze, self-outcome fitting or incompatible definition/scope.

    Notes
    -----
    Outer semantic requests bind this content before execution and check the
    actual producing request fingerprint/timing. Bare calls attest supplied
    consistency only; these fields cannot authenticate when a host created them.
    No current/application timestamp can authorize a later replacement scale.
    """

    definition: SignedVolatilitySpec
    scale: FrozenScale
    decision_fingerprint: Fingerprint
    timing: DecisionTiming
    outcome: Reference
    member: Reference | None

    @model_validator(mode="after")
    def _causal(self) -> Self:
        """Bind the selected scale to the original producing information boundary."""
        self._definitions()
        self._producing_boundary()
        if any(sample.observation.id == self.outcome.id for sample in self.scale.snapshot.samples):
            raise ValueError("Producing outcome or its corrections cannot enter its scale fit")
        if self.scale.definition.scope == "per_member" and self.member != self.scale.snapshot.member:
            raise ValueError("Indexed scale belongs to another expert/version")
        return self

    def _definitions(self) -> None:
        """Require exact scale-protocol, quantity and horizon content."""
        if self.scale.definition.binding != self.definition.scale_definition:
            raise ValueError("Selected scale definition binding differs from the transform")
        if self.scale.definition.quantity.model_dump_json() != self.definition.input.model_dump_json():
            raise ValueError("Scale quantity/units/horizon differ from the signed outcome")

    def _producing_boundary(self) -> None:
        """Compare fixed fit/definition facts only with the producing information cutoff."""
        if self.scale.frozen_at > self.timing.information_cutoff:
            raise ValueError("Scale was not frozen by the producing decision cutoff")
        declarations = (self.definition.lineage.declared_at, self.scale.definition.lineage.declared_at)
        if any(instant is not None and instant > self.timing.information_cutoff for instant in declarations):
            raise ValueError("Scale/transform definition was declared after the producing cutoff")


def select_scale(  # pylint: disable=too-many-arguments
    definition: SignedVolatilitySpec,
    scale: FrozenScale,
    *,
    decision_fingerprint: str,
    timing: DecisionTiming,
    outcome: Reference,
    member: Reference | None = None,
) -> ScaleSelection:
    """Bind one immutable scale to its producing decision before delayed feedback.

    Parameters
    ----------
    definition : SignedVolatilitySpec
        Typed loss transform and exact scale-definition binding.
    scale : FrozenScale
        Historical fitted snapshot, verified again before use.
    decision_fingerprint : str
        Producing request or declared direct-boundary SHA-256 identity.
    timing : DecisionTiming
        Original producing information/decision/target clocks.
    outcome : Reference
        Stable expected outcome ID/version, excluded from historical fitting.
    member : Reference or None, optional
        Explicit expert ID/version; required by per-member scale protocols.

    Returns
    -------
    ScaleSelection
        Content-bound choice; the caller retains it for eventual application.

    Raises
    ------
    ValueError
        Invalid binding, unknown version or future/self-contaminated fit.

    Notes
    -----
    No request/runtime/store is required. Full contextual validation checks the
    declared digest against the actual request; this function performs no I/O.
    """
    return ScaleSelection(
        schema_version=1,
        definition=definition,
        scale=scale,
        decision_fingerprint=decision_fingerprint,
        timing=timing,
        outcome=outcome,
        member=member,
    )


def _stale(selection: ScaleSelection) -> bool:
    """Compare supplied event age with the fixed producing cutoff, never wall time."""
    maximum = selection.scale.definition.maximum_age_seconds
    if maximum is None or not selection.scale.selected:
        return False
    selected_ids = {reference.id for reference in selection.scale.selected}
    newest = max(
        sample.timing.event_at for sample in selection.scale.snapshot.samples if sample.observation.id in selected_ids
    )
    elapsed = selection.timing.information_cutoff - newest
    seconds = Fraction(elapsed.days * 86400 + elapsed.seconds) + Fraction(elapsed.microseconds, 1_000_000)
    return seconds > Fraction.from_float(maximum)


def _loss_facts(
    selection: ScaleSelection, raw: Scalar
) -> tuple[Scalar, float | None, float | None, Clipping, LossReason]:
    """Derive missingness or projected intermediates with no unchecked arithmetic."""
    definition = selection.definition
    require_quantity(raw, definition.input)
    if raw.status != "observed":
        reasons: dict[str, LossReason] = {
            "unknown": "input_unknown",
            "unavailable": "input_unavailable",
            "not_applicable": "input_not_applicable",
        }
        return (
            Scalar(spec=definition.output.spec, value=None, status=raw.status),
            None,
            None,
            "not_evaluated",
            reasons[raw.status],
        )
    unavailable = _scale_unavailable(selection)
    if unavailable is not None:
        return (
            Scalar(spec=definition.output.spec, value=None, status="unavailable"),
            None,
            None,
            "not_evaluated",
            unavailable,
        )
    if raw.value is None or selection.scale.scale.value is None:
        raise ValueError("Observed outcome and available scale require values")
    ratio = Fraction.from_float(raw.value) / Fraction.from_float(selection.scale.scale.value)
    z = checked_fraction(ratio)
    unbounded = (1 - ratio / Fraction.from_float(definition.c)) / 2
    preclip = checked_fraction(unbounded)
    clipping: Clipping = "none"
    if unbounded < 0:
        clipping = "lower"
    elif unbounded > 1:
        clipping = "upper"
    loss = checked_fraction(min(Fraction(1), max(Fraction(0), unbounded)))
    return Scalar(spec=definition.output.spec, value=loss, status="observed"), z, preclip, clipping, "computed"


def _scale_unavailable(selection: ScaleSelection) -> LossReason | None:
    """Retain zero, insufficient and stale scale as separate noncomputable causes."""
    if selection.scale.status == "zero_scale":
        return "zero_scale"
    if selection.scale.status == "insufficient_support":
        return "insufficient_support"
    return "stale_scale" if _stale(selection) else None


class VolatilityApplication(CoreContract):
    """Preserve raw outcome, frozen selection, unclipped values and loss status.

    Attributes
    ----------
    selection : ScaleSelection
        Original pre-decision scale choice and definition/source/fit lineage.
    raw : Scalar
        Original signed outcome/null with original units and bytes untouched.
    output : Scalar
        Dimensionless loss or explicit inherited/noncomputable null state.
    z, preclip_loss : float or None
        Guarded projections of exact x/s and (1-(x/s)/c)/2 respectively. Both
        derive from stored input ratios, not a hidden chain of rounded results.
    clipping : {'none', 'lower', 'upper', 'not_evaluated'}
        Strict clipping direction; exact thresholds are ``none``.
    reason : str
        Computed, missing-input, zero/insufficient-scale or stale-scale outcome.

    Raises
    ------
    ValueError
        Altered output/receipt facts, incompatible schema or arithmetic failure.

    Notes
    -----
    Recomputes output on guarded reads. It updates no learner or accounting book.
    Clipping never claims bounds on the original outcome or repairs bad scales.
    A receipt is deterministic supplied-input evidence, not source authentication.
    """

    selection: ScaleSelection
    raw: Scalar
    output: Scalar
    z: Binary64 | None
    preclip_loss: Binary64 | None
    clipping: Clipping
    reason: LossReason

    @model_validator(mode="after")
    def _facts(self) -> Self:
        """Reject serialized receipt changes by reproducing every operation fact."""
        output, z, preclip, clipping, reason = _loss_facts(self.selection, self.raw)
        actual = [self.output.model_dump(mode="json"), self.z, self.preclip_loss, self.clipping, self.reason]
        expected = [output.model_dump(mode="json"), z, preclip, clipping, reason]
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError("Signed-volatility receipt differs from its supplied inputs")
        return self


def apply_signed_volatility(selection: ScaleSelection, raw: Scalar) -> VolatilityApplication:
    """Compute signed volatility-scaled loss using only the frozen original choice.

    Parameters
    ----------
    selection : ScaleSelection
        Pre-decision selection; later feedback cannot supply a new fit cutoff.
    raw : Scalar
        Signed original outcome or explicit unknown/unavailable/not-applicable.

    Returns
    -------
    VolatilityApplication
        Retained raw/scale/definition inputs, intermediate projections and loss.

    Raises
    ------
    ValueError
        Incompatible schema, future scale or nonfinite/overflow/nonzero-underflow
        of a projected operation boundary. Missing/zero/stale scale is a typed
        noncomputable receipt, not a division, synthetic zero or implicit epsilon.

    Notes
    -----
    Exact rational intermediates use stored binary64 ratios; z, preclip loss and
    clipped loss are independently projected through ``binary64-rational-v1``.
    Pure computation performs no update, filesystem access or current-clock read.
    """
    checked = ScaleSelection.model_validate(selection)
    original = Scalar.model_validate(raw)
    output, z, preclip, clipping, reason = _loss_facts(checked, original)
    return VolatilityApplication(
        schema_version=1,
        selection=checked,
        raw=original,
        output=output,
        z=z,
        preclip_loss=preclip,
        clipping=clipping,
        reason=reason,
    )


def volatility_outcome(application: VolatilityApplication, raw: OutcomeValue) -> OutcomeValue:
    """Attach a verified signed loss without changing the supplied outcome origin.

    Parameters
    ----------
    application : VolatilityApplication
        Validated operation retaining the exact original scalar content.
    raw : OutcomeValue
        Original realized, modeled-counterfactual or synthetic observation.

    Returns
    -------
    OutcomeValue
        Derived loss with the original origin and selected transform reference.
        Input null states propagate; a noncomputable scale remains unavailable
        with explicit null instead of an invented learner observation.

    Raises
    ------
    ValueError
        Raw scalar content differs, including signed zero, or receipt validation
        fails. Units, horizon and causal fit validation retain their existing rules.

    Notes
    -----
    This does not qualify source authenticity, feedback coverage, timing of
    application or learner consumption. It changes no observation or batch and
    performs no I/O. The origin supplied by the caller is preserved, never inferred.
    """
    checked = VolatilityApplication.model_validate(application)
    return _project_loss_outcome(raw, checked.raw, checked.output, checked.selection.definition.reference)


class ClippingStats(CoreContract):
    """Summarize unique applications for one transform/member/evaluation split.

    Attributes
    ----------
    definition : DefinitionBinding
        Exact transform content; differing scale protocols cannot be pooled here.
    member : Reference or None
        Exact expert grouping, with null retained for a shared/ungrouped series.
    split : Reference
        Explicit evaluation split, never inferred from the outcome values.
    applications : tuple of str
        Canonically sorted unique receipt digests; no authenticity is implied.
    observed, computed, noncomputable : int
        Raw observed count; evaluated loss count; observed but invalid-scale count.
    unknown, unavailable, not_applicable : int
        Distinct raw null-state counts, excluded from the clipping denominator.
    lower, upper : int
        Strict lower/upper clipping counts among computed losses.
    lower_rate, upper_rate : float or None
        Dimensionless rates with denominator ``computed``; null if it is zero.

    Notes
    -----
    Counts have observation units, rates unit ``1``. Missing and noncomputable
    cases are separate; observed raw zero is included. The summary checks internal
    arithmetic but its digests alone cannot authenticate absent source receipts.
    It makes no inference about saturation efficacy or independent observations.
    """

    definition: DefinitionBinding
    member: Reference | None
    split: Reference
    applications: Annotated[tuple[Fingerprint, ...], Field(min_length=1, max_length=MAX_MEMBERS)]
    observed: Annotated[int, Field(strict=True, ge=0)]
    computed: Annotated[int, Field(strict=True, ge=0)]
    noncomputable: Annotated[int, Field(strict=True, ge=0)]
    unknown: Annotated[int, Field(strict=True, ge=0)]
    unavailable: Annotated[int, Field(strict=True, ge=0)]
    not_applicable: Annotated[int, Field(strict=True, ge=0)]
    lower: Annotated[int, Field(strict=True, ge=0)]
    upper: Annotated[int, Field(strict=True, ge=0)]
    lower_rate: Binary64 | None
    upper_rate: Binary64 | None

    @model_validator(mode="after")
    def _counts(self) -> Self:
        """Check count conservation, explicit denominators and canonical identities."""
        total = self.observed + self.unknown + self.unavailable + self.not_applicable
        if len(set(self.applications)) != len(self.applications) or total != len(self.applications):
            raise ValueError("Clipping summary requires unique applications and conserved raw states")
        if self.computed + self.noncomputable != self.observed or self.lower + self.upper > self.computed:
            raise ValueError("Clipping summary counts violate the computed-loss denominator")
        rates = (_rate(self.lower, self.computed), _rate(self.upper, self.computed))
        if canonical_json([self.lower_rate, self.upper_rate]) != canonical_json(list(rates)):
            raise ValueError("Clipping rates require the explicit computed-loss denominator")
        object.__setattr__(self, "applications", tuple(sorted(self.applications)))
        return self


def _rate(count: int, denominator: int) -> float | None:
    """Retain an unavailable rate rather than inventing zero over zero."""
    return checked_fraction(Fraction(count, denominator)) if denominator else None


def _unique_applications(applications: Iterable[VolatilityApplication]) -> tuple[VolatilityApplication, ...]:
    """Check bounded local retries without claiming durable session acceptance."""
    unique: dict[tuple[str, str, str | None], VolatilityApplication] = {}
    for index, application in enumerate(applications):
        if index >= MAX_MEMBERS:
            raise ValueError("Clipping input exceeds 4096 attempted applications")
        checked = VolatilityApplication.model_validate(application)
        selection = checked.selection
        key = selection.decision_fingerprint, selection.outcome.id, selection.member.id if selection.member else None
        previous = unique.setdefault(key, checked)
        if previous.to_json() != checked.to_json():
            raise ValueError("Conflicting application at one producing decision/outcome/member key")
    return tuple(unique.values())


def clipping_statistics(applications: Iterable[VolatilityApplication], *, split: Reference) -> ClippingStats:
    """Count unique typed applications with explicit raw and computable denominators.

    Parameters
    ----------
    applications : iterable of VolatilityApplication
        One through 4096 attempts for one transform/member. Identical canonical
        retries count once; changed content at a logical key is a conflict.
    split : Reference
        Caller-declared development/protected split identity and version.

    Returns
    -------
    ClippingStats
        Separate null/noncomputable counts and rates over evaluated losses only.

    Raises
    ------
    ValueError
        Empty/oversized input, conflicting retry or mixed transform/member groups.

    Notes
    -----
    Consumes at most 4097 attempts. This pure local idempotence check neither
    persists delivery acceptance nor authorizes changes based on protected rates.
    """
    rows = _unique_applications(applications)
    if not rows:
        raise ValueError("Clipping summary requires at least one application")
    definition, member = rows[0].selection.definition.binding, rows[0].selection.member
    if any(row.selection.definition.binding != definition or row.selection.member != member for row in rows):
        raise ValueError("Clipping summary requires one exact transform and member")
    states = Counter(row.raw.status for row in rows)
    outcomes = Counter(row.reason for row in rows)
    clips = Counter(row.clipping for row in rows)
    observed, computed = states["observed"], outcomes["computed"]
    lower, upper = clips["lower"], clips["upper"]
    return ClippingStats(
        schema_version=1,
        definition=definition,
        member=member,
        split=split,
        applications=tuple(row.fingerprint for row in rows),
        observed=observed,
        computed=computed,
        noncomputable=observed - computed,
        unknown=states["unknown"],
        unavailable=states["unavailable"],
        not_applicable=states["not_applicable"],
        lower=lower,
        upper=upper,
        lower_rate=_rate(lower, computed),
        upper_rate=_rate(upper, computed),
    )
