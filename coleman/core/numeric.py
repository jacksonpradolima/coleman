"""Explicit binary64 mathematical values and bounded numerical simplex checks."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Literal, Self, cast

from pydantic import BeforeValidator, Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import MAX_MEMBERS
from coleman.core.units import UnitSemantics

SIMPLEX_PROFILE = "binary64-simplex-v1"
SimplexProfile = Literal["binary64-simplex-v1"]
SIMPLEX_TOLERANCE = Fraction(1, 2**49)


def binary64(value: object) -> float:
    """Validate an explicitly projected finite binary64 mathematical value.

    Parameters
    ----------
    value : float or int or decimal.Decimal
        Built-in finite float, exactly representable built-in integer, or finite
        Decimal projected with binary64 rounding. Guarded JSON preserves Decimal
        tokens until validation. Signed zero remains signed.

    Returns
    -------
    float
        Stored binary64 value, not exact decimal accounting. Source lexical
        spelling is not retained in mathematical data.

    Raises
    ------
    ValueError
        Nonfinite/overflow, inexact integer promotion, nonzero Decimal underflow,
        or unsupported kind including bool, strings, Fraction, subclasses and
        arbitrary callback-bearing objects.

    Notes
    -----
    Executes no user conversion callbacks, clipping, unit conversion, fixed-scale
    rounding or I/O. Retained legacy Finite coercion remains unchanged.
    """
    # Exact built-in types prevent arbitrary subclass conversion callbacks.
    # pylint: disable=unidiomatic-typecheck
    if type(value) not in {float, int, Decimal}:
        raise ValueError("Binary64 accepts only built-in float, exact integer or Decimal")
    if type(value) is Decimal and not value.is_finite():
        raise ValueError("Binary64 requires a finite Decimal")
    try:
        result = float(cast(float | int | Decimal, value))
    except (OverflowError, ValueError) as exc:
        raise ValueError("Binary64 conversion overflowed or is invalid") from exc
    if not math.isfinite(result):
        raise ValueError("Binary64 requires a finite value")
    if type(value) is int and int(result) != value:
        raise ValueError("Integer is not exactly representable as binary64")
    if type(value) is Decimal and value != 0 and result == 0:
        raise ValueError("Nonzero Decimal underflows binary64")
    return result


Binary64 = Annotated[float, BeforeValidator(binary64), Field(strict=True, allow_inf_nan=False)]
ScalarRole = Literal["objective", "resource", "loss", "reward", "feature", "prediction", "diagnostic", "action"]
ScalarStatus = Literal["observed", "unknown", "unavailable", "not_applicable"]


class ScalarSpec(CoreImmutable):
    """Define one mathematical dimension, its role, units and admitted bounds.

    Attributes
    ----------
    dimension : Reference
        Versioned dimension identity, independent from action coverage.
    semantics : UnitSemantics
        Explicit unit and meaning, without implicit conversion.
    role : str
        Objective, resource, loss, reward, feature, prediction, diagnostic or
        action. Equal shape and unit do not make different roles equivalent.
    minimum, maximum : float or None
        Optional inclusive finite bounds in declared units. None means the bound
        is not declared, never zero or an unavailable amount.

    Raises
    ------
    ValueError
        Invalid input, nonfinite bounds or minimum greater than maximum.

    Notes
    -----
    Bounds are declarations, not scientific evidence. No loss transform,
    preference, physical-cost model or feasibility oracle is executed.
    """

    dimension: Reference
    semantics: UnitSemantics
    role: ScalarRole
    minimum: Binary64 | None = None
    maximum: Binary64 | None = None

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Reject reversed bounds without clipping or approximate comparison."""
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Scalar minimum exceeds maximum")
        return self


class Scalar(CoreImmutable):
    """Retain one mathematical value or its distinct explicit missingness state.

    Attributes
    ----------
    spec : ScalarSpec
        Exact dimension, role, units, meaning and inclusive bounds.
    value : float or None
        Required finite value for observed status, explicit null otherwise.
        Zero and permitted negatives are valid; omission is invalid.
    status : {'observed', 'unknown', 'unavailable', 'not_applicable'}
        Value state, independent from the number of observed actions.

    Raises
    ------
    ValueError
        Invalid fields, incompatible status/value or a violated declared bound.

    Notes
    -----
    Null retains its units and semantics. No imputation, reward transformation,
    comparison or domain record/provenance construction occurs.
    """

    spec: ScalarSpec
    value: Binary64 | None
    status: ScalarStatus

    @model_validator(mode="after")
    def _value(self) -> Self:
        """Keep explicit null states and check observed values against bounds."""
        if (self.status == "observed") != (self.value is not None):
            raise ValueError("Observed scalar requires a value; other statuses require null")
        if self.value is not None:
            if self.spec.minimum is not None and self.value < self.spec.minimum:
                raise ValueError("Scalar is below its declared minimum")
            if self.spec.maximum is not None and self.value > self.spec.maximum:
                raise ValueError("Scalar is above its declared maximum")
        return self


def validate_simplex(weights: tuple[float, ...]) -> None:
    """Check a bounded stored-binary64 simplex without normalization or repair.

    Parameters
    ----------
    weights : tuple of float
        One through 4096 built-in finite weights in [0, 1]. Zero components and
        signed zero are permitted; order is irrelevant.

    Raises
    ------
    ValueError
        Invalid shape/count/type, nonfinite/out-of-range entry, or exact stored
        rational residual from one greater than 2**-49 (eight ulps at one).

    Notes
    -----
    ``binary64-simplex-v1`` is an absolute rational residual bound independent of
    reduction order and Decimal context. Inputs are unchanged. Numerical
    feasibility is not exact-real arithmetic, sampler support or a theorem.
    Only bounded arithmetic is performed.
    """
    # Weights are already validated binary64, never callback-bearing subclasses.
    # pylint: disable=unidiomatic-typecheck
    if not isinstance(weights, tuple) or not 1 <= len(weights) <= MAX_MEMBERS:
        raise ValueError("Simplex requires a tuple of one through 4096 weights")
    invalid = (type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1 for value in weights)
    if any(invalid):
        raise ValueError("Simplex requires finite binary64 weights in [0, 1]")
    total = sum((Fraction.from_float(value) for value in weights), Fraction())
    if abs(total - 1) > SIMPLEX_TOLERANCE:
        raise ValueError("Simplex exceeds binary64-simplex-v1 absolute residual")
