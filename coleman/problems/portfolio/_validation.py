"""Pure exact public receipt validation, independent of transition constructors."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from coleman.core.primitives import fingerprint
from coleman.problems.portfolio._settlement_validation import _accepted, _refusal

if TYPE_CHECKING:
    from coleman.problems.portfolio.contracts import Execution, Valuation


def _validate_valuation(value: Valuation) -> None:
    """Check shape, missingness and exact weights without claiming authentication."""
    size = len(value.prices.quotes)
    if len(value.asset_values) != size or len(value.drifted_weights) != size:
        raise ValueError("Valuation requires one asset value and drifted weight per quote")
    if any(amount is not None and amount.numerator < 0 for amount in value.asset_values):
        raise ValueError("Funded asset valuation cannot be negative")
    _valuation_status(value)
    total = None if value.value is None else value.value.fraction
    if total is not None and total < sum(
        (amount.fraction for amount in value.asset_values if amount is not None), Fraction()
    ):
        raise ValueError("Funded valuation cannot imply negative cash")
    _valuation_weights(value, total)


def _valuation_status(value: Valuation) -> None:
    """Verify known/stale/unknown status against all retained asset marks."""
    unknown = any(amount is None for amount in value.asset_values)
    stale = any(
        (quote.status == "stale" or quote.event_at != value.prices.at) and amount is not None and amount.numerator != 0
        for quote, amount in zip(value.prices.quotes, value.asset_values, strict=True)
    )
    expected = "unknown" if unknown else "stale" if stale else "complete"
    if value.status != expected or (value.value is None) != unknown:
        raise ValueError("Valuation status and exact total missingness disagree")


def _valuation_weights(value: Valuation, total: Fraction | None) -> None:
    """Check each retained quote/mark/weight without treating null as zero."""
    for quote, amount, weight in zip(value.prices.quotes, value.asset_values, value.drifted_weights, strict=True):
        if quote.value is None and amount is not None and amount.numerator != 0:
            raise ValueError("Missing quote cannot supply a nonzero asset mark")
        expected_weight = None if total is None or total == 0 or amount is None else amount.fraction / total
        if (None if weight is None else weight.fraction) != expected_weight:
            raise ValueError("Drifted weight differs from the exact retained marked value")


def _input_identity(value: Execution) -> str:
    """Recompute declared complete input identity; content is not authenticity."""
    return fingerprint(
        {
            "book": value.before.fingerprint,
            "prices": value.valuation.prices.fingerprint,
            "target": None if value.target is None else value.target.fingerprint,
            "fees": value.fees.fingerprint,
            "request": value.request.model_dump(mode="json"),
        }
    )


def _validate_execution(value: Execution) -> None:
    """Check exact original mark, settlement conservation and target interpretation."""
    assets = tuple(position.asset for position in value.before.positions)
    prices = value.valuation.prices
    if (
        tuple(quote.asset for quote in prices.quotes) != assets
        or value.valuation.book_fingerprint != value.before.fingerprint
    ):
        raise ValueError("Settlement valuation belongs to another exact funded book")
    if value.target is not None and tuple(weight.asset for weight in value.target.weights) != assets:
        raise ValueError("Settlement target has another exact asset catalogue")
    if value.input_fingerprint != _input_identity(value):
        raise ValueError("Settlement input identity differs from its complete original inputs")
    amounts, gross = _original_mark(value)
    if value.status == "refused":
        _refusal(value, gross, amounts)
        return
    _accepted(value, assets)


def _original_mark(value: Execution) -> tuple[tuple[Fraction | None, ...], Fraction | None]:
    """Check the original held-unit mark, preserving missing total value."""
    prices = value.valuation.prices
    amounts_list = []
    for position, quote in zip(value.before.positions, prices.quotes, strict=True):
        if not position.units.numerator:
            amounts_list.append(Fraction())
        elif quote.value is None:
            amounts_list.append(None)
        else:
            amounts_list.append(position.units.fraction * quote.value.fraction)
    amounts = tuple(amounts_list)
    if tuple(None if amount is None else amount.fraction for amount in value.valuation.asset_values) != amounts:
        raise ValueError("Settlement mark differs from its original held asset units")
    gross = (
        None
        if any(amount is None for amount in amounts)
        else value.before.cash.fraction + sum((amount for amount in amounts if amount is not None), Fraction())
    )
    if (None if value.gross_value is None else value.gross_value.fraction) != gross or (
        None if value.valuation.value is None else value.valuation.value.fraction
    ) != gross:
        raise ValueError("Settlement gross value differs from its original funded mark")
    return amounts, gross
