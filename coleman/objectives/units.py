"""Explicit positive physical-unit conversions with raw-preserving receipts."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import model_validator

from coleman.core.base import CoreContract
from coleman.core.numeric import Scalar
from coleman.objectives.arithmetic import PROFILE, checked_fraction
from coleman.objectives.bindings import QuantityMeaning, SemanticDefinition, require_quantity

_CATALOG = {
    "us": ("time", Fraction(1, 1_000_000)),
    "ms": ("time", Fraction(1, 1000)),
    "s": ("time", Fraction(1)),
    "min": ("time", Fraction(60)),
    "byte": ("storage", Fraction(1)),
    "KiB": ("storage", Fraction(1024)),
    "MiB": ("storage", Fraction(1024**2)),
}


def conversion_factor(source: str, target: str) -> Fraction:
    """Return the exact factor from the explicit v1 physical conversion catalog.

    Parameters
    ----------
    source, target : str
        Case-sensitive units. Identical opaque units admit identity only.

    Returns
    -------
    Fraction
        Positive multiplier, with time and storage families kept separate.

    Raises
    ------
    ValueError
        Unknown differing units or incompatible physical families.

    Notes
    -----
    No symbol inference, offsets, currency conversion, registry or I/O occurs.
    The catalog is limited to us/ms/s/min and byte/KiB/MiB.
    """
    # pylint: disable-next=unidiomatic-typecheck
    if type(source) is not str or type(target) is not str or not source.strip() or not target.strip():
        raise ValueError("Conversion units require explicit nonempty string declarations")
    if source == target:
        return Fraction(1)
    if source not in _CATALOG or target not in _CATALOG:
        raise ValueError("Unknown conversion unit; declare a supported explicit pair")
    family, source_scale = _CATALOG[source]
    target_family, target_scale = _CATALOG[target]
    if family != target_family:
        raise ValueError("Conversion units belong to different physical families")
    return source_scale / target_scale


class ConversionSpec(SemanticDefinition):
    """Declare a supported scale conversion without changing an old raw schema.

    Attributes
    ----------
    input, output : QuantityMeaning
        Explicit source/target schemas. Role, interpretation, meaning, horizon
        and risk definition agree; output bounds follow the positive factor.
    arithmetic_profile : {'binary64-rational-v1'}, optional
        Exact rational factor followed by guarded binary64 projection.

    Raises
    ------
    ValueError
        Incompatible semantics, unit families, bounds or arithmetic profile.

    Notes
    -----
    This pure definition does not convert input data or rewrite any ledger.
    Exact Decimal accounting retains its existing independent boundary.
    """

    input: QuantityMeaning
    output: QuantityMeaning
    arithmetic_profile: Literal["binary64-rational-v1"] = PROFILE

    @model_validator(mode="after")
    def _compatible(self) -> Self:
        """Require explicit matching interpretations and unit-scaled bounds."""
        first, last = self.input, self.output
        if (
            first.interpretation != last.interpretation
            or first.horizon != last.horizon
            or first.risk_definition != last.risk_definition
            or first.spec.role != last.spec.role
            or first.spec.semantics.meaning != last.spec.semantics.meaning
        ):
            raise ValueError("Conversion cannot change role, meaning, interpretation or horizon")
        factor = conversion_factor(first.spec.semantics.unit, last.spec.semantics.unit)
        for lower, upper in [(first.spec.minimum, last.spec.minimum), (first.spec.maximum, last.spec.maximum)]:
            expected = None if lower is None else checked_fraction(Fraction.from_float(lower) * factor)
            if expected != upper:
                raise ValueError("Conversion output bounds do not follow the declared factor")
        return self


def _converted(definition: ConversionSpec, raw: Scalar) -> Scalar:
    """Apply a validated exact factor while retaining null and signed zero."""
    require_quantity(raw, definition.input)
    value = raw.value
    if value is not None and value != 0:
        factor = conversion_factor(definition.input.spec.semantics.unit, definition.output.spec.semantics.unit)
        value = checked_fraction(Fraction.from_float(value) * factor)
    return Scalar(spec=definition.output.spec, value=value, status=raw.status)


class ConversionReceipt(CoreContract):
    """Retain source, definition and independently checkable converted value.

    Attributes
    ----------
    definition : ConversionSpec
        Full immutable definition, including exact unit catalog/profile binding.
    raw, converted : Scalar
        Original quantity and converted result with unchanged value status.

    Raises
    ------
    ValueError
        Input schema, arithmetic or claimed converted output is invalid.

    Notes
    -----
    Reading rechecks arithmetic rather than trusting a claimed receipt. No
    source authentication, ledger mutation, durable publication or I/O occurs.
    """

    definition: ConversionSpec
    raw: Scalar
    converted: Scalar

    @model_validator(mode="after")
    def _result(self) -> Self:
        """Reject forged results, including loss of retained signed zero."""
        expected = _converted(self.definition, self.raw)
        if expected.model_dump_json() != self.converted.model_dump_json():
            raise ValueError("Conversion receipt does not reproduce its definition")
        return self


def convert_quantity(definition: ConversionSpec, raw: Scalar) -> ConversionReceipt:
    """Apply an explicitly declared conversion and return immutable raw evidence.

    Parameters
    ----------
    definition : ConversionSpec
        Exact source/target semantics, bounded catalog and arithmetic profile.
    raw : Scalar
        Original finite signed value or explicit missingness state.

    Returns
    -------
    ConversionReceipt
        Raw input plus checked output, definition content and lineage.

    Raises
    ------
    ValueError
        Invalid definition/schema, unknown units, overflow or nonzero underflow.

    Notes
    -----
    Pure and deterministic. Does not mutate the input, perform accounting,
    access a runtime or silently convert percentages, currencies or probabilities.
    """
    checked = ConversionSpec.model_validate(definition)
    value = Scalar.model_validate(raw)
    return ConversionReceipt(schema_version=1, definition=checked, raw=value, converted=_converted(checked, value))
