"""Explicit identity and affine learner-loss surrogates with typed evidence."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract
from coleman.core.feedback import OutcomeValue
from coleman.core.numeric import Binary64, Scalar
from coleman.objectives.arithmetic import PROFILE, checked_fraction
from coleman.objectives.bindings import QuantityMeaning, SemanticDefinition, _project_loss_outcome, require_quantity

Clipping = Literal["none", "lower", "upper", "not_evaluated"]
ApplicationStatus = Literal["applied", "missing", "rejected"]


class IdentityLoss(SemanticDefinition):
    """Declare identity only for an already interpreted normalized loss.

    Attributes
    ----------
    kind : {'identity'}, optional
        Explicit operation discriminator; never inferred from a unit symbol.
    quantity : QuantityMeaning
        Loss-role dimensionless quantity with explicit bounds inside [0,1].

    Raises
    ------
    ValueError
        The input is not an explicitly normalized loss.

    Notes
    -----
    Scores, probabilities, rewards and raw outcomes need their own operation.
    This definition performs no conversion, execution or I/O.
    """

    kind: Literal["identity"] = "identity"
    quantity: QuantityMeaning

    @model_validator(mode="after")
    def _loss(self) -> Self:
        """Reject an identity label on a differently interpreted quantity."""
        if self.quantity.interpretation != "normalized_loss":
            raise ValueError("Identity loss requires an already normalized loss")
        return self


class AffineLoss(SemanticDefinition):
    """Declare a fixed bounded affine surrogate with explicit outside policy.

    Attributes
    ----------
    kind : {'affine'}, optional
        Operation discriminator. This version does not fit bounds.
    input, output : QuantityMeaning
        Raw schema and distinct normalized-loss schema with the same horizon.
    lower, upper : float
        Finite ordered strict range in input units, not claimed raw observations.
    direction : {'minimize', 'maximize'}
        Minimize maps lower to zero; maximize maps upper to zero.
    outside : {'reject', 'clip'}
        Explicit treatment of out-of-range observations.
    bound_basis : {'modeling_assumption', 'external_bound'}
        Assumed range or an externally sourced declaration. External bounds
        require a source identity and digest; their truth is not authenticated.
    arithmetic_profile : {'binary64-rational-v1'}, optional
        Rational calculation on stored numbers followed by guarded projection.

    Raises
    ------
    ValueError
        Reversed/constant bounds, incompatible output, unsupported parameters
        or missing source facts for an external-bound claim.

    Notes
    -----
    Clipping defines a surrogate and never bounds the retained raw population
    or establishes calibration, independence or inferential validity. No fit,
    implicit mean adjustment, accounting rewrite or I/O occurs.
    """

    kind: Literal["affine"] = "affine"
    input: QuantityMeaning
    output: QuantityMeaning
    lower: Binary64
    upper: Binary64
    direction: Literal["minimize", "maximize"]
    outside: Literal["reject", "clip"]
    bound_basis: Literal["modeling_assumption", "external_bound"]
    arithmetic_profile: Literal["binary64-rational-v1"] = PROFILE

    @model_validator(mode="after")
    def _definition(self) -> Self:
        """Check the declared range and interpretation without fitting data."""
        if self.lower >= self.upper:
            raise ValueError("Affine loss requires lower < upper")
        if self.output.interpretation != "normalized_loss" or (
            self.output.spec.minimum != 0 or self.output.spec.maximum != 1
        ):
            raise ValueError("Affine output must explicitly declare the full normalized-loss interval")
        if self.input.horizon != self.output.horizon:
            raise ValueError("Affine transform cannot change the measurement horizon")
        if self.bound_basis == "external_bound" and (
            self.lineage.source is None or self.lineage.source_fingerprint is None
        ):
            raise ValueError("External bound requires explicit source identity and content digest")
        return self


LossDefinition = Annotated[IdentityLoss | AffineLoss, Field(discriminator="kind")]


def _derive(
    definition: IdentityLoss | AffineLoss, raw: Scalar
) -> tuple[Scalar | None, ApplicationStatus, Clipping, float | None]:
    """Calculate the explicit operation without constructing a recursive receipt."""
    quantity = definition.quantity if isinstance(definition, IdentityLoss) else definition.input
    output = definition.quantity if isinstance(definition, IdentityLoss) else definition.output
    require_quantity(raw, quantity)
    if raw.value is None:
        return Scalar(spec=output.spec, value=None, status=raw.status), "missing", "not_evaluated", None
    if isinstance(definition, IdentityLoss):
        return raw, "applied", "none", raw.value
    lower, upper = Fraction.from_float(definition.lower), Fraction.from_float(definition.upper)
    value = Fraction.from_float(raw.value)
    numerator = value - lower if definition.direction == "minimize" else upper - value
    exact = numerator / (upper - lower)
    before = checked_fraction(exact)
    clipping: Clipping = "none"
    if exact < 0:
        clipping = "lower"
    elif exact > 1:
        clipping = "upper"
    if clipping != "none" and definition.outside == "reject":
        return None, "rejected", "not_evaluated", before
    loss = checked_fraction(max(Fraction(), min(Fraction(1), exact)))
    return Scalar(spec=output.spec, value=loss, status="observed"), "applied", clipping, before


class LossApplication(CoreContract):
    """Retain the full loss definition, signed raw data and verified operation.

    Attributes
    ----------
    definition : IdentityLoss or AffineLoss
        Exact immutable definition and source/assumption lineage.
    raw : Scalar
        Original signed/raw value and missingness, without ledger changes.
    output : Scalar or None
        Explicit normalized loss, matching null status, or null for rejection.
    status : {'applied', 'missing', 'rejected'}
        Operation outcome; rejected input is never a completed observed loss.
    clipping : {'none', 'lower', 'upper', 'not_evaluated'}
        Side of the pre-clipped loss changed by clipping. Equality is not clipping.
    preclip_loss : float or None
        Finite surrogate before clipping/rejection, or null for missing data.

    Raises
    ------
    ValueError
        Unchecked input, arithmetic failure or an inconsistent claimed result.

    Notes
    -----
    A read recomputes arithmetic; a digest alone does not establish eligibility.
    Pure calls require no decision/runtime/artifact. There is no imputation or I/O.
    """

    definition: LossDefinition
    raw: Scalar
    output: Scalar | None
    status: ApplicationStatus
    clipping: Clipping
    preclip_loss: Binary64 | None

    @model_validator(mode="after")
    def _application(self) -> Self:
        """Reject forged receipts and retain signed-zero identity behavior."""
        output, status, clipping, before = _derive(self.definition, self.raw)
        expected = None if output is None else output.model_dump_json()
        actual = None if self.output is None else self.output.model_dump_json()
        before_bits = None if before is None else float(before).hex()
        actual_bits = None if self.preclip_loss is None else float(self.preclip_loss).hex()
        if (expected, status, clipping, before_bits) != (actual, self.status, self.clipping, actual_bits):
            raise ValueError("Loss application does not reproduce its declared operation")
        return self


def apply_loss(definition: IdentityLoss | AffineLoss, raw: Scalar) -> LossApplication:
    """Apply a declared loss operation with explicit raw-preserving evidence.

    Parameters
    ----------
    definition : IdentityLoss or AffineLoss
        Complete operation, semantics, range/direction and outside policy.
    raw : Scalar
        Finite observed value or one of the explicit null states.

    Returns
    -------
    LossApplication
        Validated result or typed rejection, retaining the full raw quantity.

    Raises
    ------
    ValueError
        Definition/schema mismatch, nonfinite arithmetic or nonzero underflow.
    TypeError
        The definition is not a supported typed operation.

    Notes
    -----
    Deterministic and side-effect free. Does not fit, read a clock, update a
    learner, authenticate data or qualify inference on clipped raw outcomes.
    """
    if not isinstance(definition, IdentityLoss | AffineLoss):
        raise TypeError("A typed identity or affine loss definition is required")
    checked = type(definition).model_validate(definition)
    value = Scalar.model_validate(raw)
    output, status, clipping, before = _derive(checked, value)
    return LossApplication(
        schema_version=1,
        definition=checked,
        raw=value,
        output=output,
        status=status,
        clipping=clipping,
        preclip_loss=before,
    )


def loss_outcome(application: LossApplication, raw: OutcomeValue) -> OutcomeValue:
    """Attach the verified loss to an existing outcome without changing its origin.

    Parameters
    ----------
    application : LossApplication
        Applied or missing result bound to the exact raw scalar content.
    raw : OutcomeValue
        Original realized, counterfactual or synthetic observation.

    Returns
    -------
    OutcomeValue
        Loss and transform identity with unchanged origin and missingness.

    Raises
    ------
    ValueError
        Raw content differs or the operation rejected rather than produced output.

    Notes
    -----
    Does not reveal unobserved members, validate feedback timing, mutate a batch
    or qualify actual learner consumption. It performs no I/O.
    """
    checked = LossApplication.model_validate(application)
    if checked.output is None:
        raise ValueError("Outcome does not match an application producing a loss")
    return _project_loss_outcome(raw, checked.raw, checked.output, checked.definition.reference)
