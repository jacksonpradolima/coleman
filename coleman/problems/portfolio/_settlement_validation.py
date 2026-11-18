"""Validate exact funded settlement separately from original mark interpretation."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coleman.problems.portfolio.contracts import Execution


def _refusal(value: Execution, gross: Fraction | None, amounts: tuple[Fraction | None, ...]) -> None:
    """Verify negative outcomes against their original declared availability/funding."""
    expected = _refusal_reason(value)
    if value.reason != expected:
        raise ValueError("Refusal reason differs from original quote/fee availability")
    if expected == "insufficient_funding":
        assert gross is not None and value.fees.recurring is not None and value.fees.proportional is not None
        available = gross - value.fees.recurring.fraction
        can_fund = (
            value.before.cash.fraction >= value.fees.recurring.fraction
            if value.target is None
            else available
            >= value.fees.proportional.fraction * sum((amount for amount in amounts if amount is not None), Fraction())
        )
        if can_fund:
            raise ValueError("Refused funding is feasible under its original declared exact law")


def _refusal_reason(value: Execution) -> str:
    """Check actual quote/schedule eligibility at the original execution cutoff."""
    quotes = value.valuation.prices.quotes
    return (
        "missing_price"
        if any(quote.status in {"unknown", "unavailable"} for quote in quotes)
        else "stale_price"
        if any(quote.status == "stale" or quote.event_at != value.valuation.prices.at for quote in quotes)
        else "missing_fee"
        if value.fees.status == "unavailable"
        else "ineligible_fee"
        if value.fees.declared_at > value.valuation.prices.at
        else "insufficient_funding"
    )


def _accepted(value: Execution, assets: tuple) -> None:
    """Validate a complete accepted receipt without calling the production solver."""
    prices, fees = value.valuation.prices, value.fees
    if (
        fees.status != "observed"
        or fees.declared_at > prices.at
        or any(quote.status != "observed" or quote.event_at != prices.at for quote in prices.quotes)
    ):
        raise ValueError("Execution requires actual observed prices and both charges")
    if (
        tuple(trade.asset for trade in value.trades) != assets
        or tuple(position.asset for position in value.after.positions) != assets
    ):
        raise ValueError("Executed trades and after holdings require the complete original catalogue")
    if (
        value.after.reference != value.before.reference
        or value.after.sequence != value.before.sequence + 1
        or value.after.charged_periods != (*value.before.charged_periods, fees.period)
    ):
        raise ValueError("Execution changed original funded owner or fee/period frontier")
    _positive_settlement(value)
    _trade_join(value)
    _funding_join(value)


def _positive_settlement(value: Execution) -> None:
    """Require observed nonnegative amounts before exact arithmetic."""
    fees = value.fees
    assert fees.proportional is not None and fees.recurring is not None
    assert (
        value.net_value is not None
        and value.turnover is not None
        and value.proportional_paid is not None
        and value.recurring_paid is not None
        and value.gross_value is not None
    )
    if any(
        amount.numerator < 0
        for amount in (value.net_value, value.turnover, value.proportional_paid, value.recurring_paid)
    ):
        raise ValueError("Executed paid amounts, turnover and funded value cannot be negative")


def _trade_join(value: Execution) -> None:
    """Reconcile actual changes in asset units and signed traded credits."""
    prices = value.valuation.prices
    for before, after, trade, quote in zip(
        value.before.positions, value.after.positions, value.trades, prices.quotes, strict=True
    ):
        assert quote.value is not None
        if (
            after.units.fraction != before.units.fraction + trade.units.fraction
            or trade.notional.fraction != trade.units.fraction * quote.value.fraction
        ):
            raise ValueError("Executed holdings and signed traded notional do not reconcile")


def _funding_join(value: Execution) -> None:
    """Check actual netted fees, funded wealth conservation and target amounts."""
    prices, fees = value.valuation.prices, value.fees
    assert fees.proportional is not None and fees.recurring is not None
    assert value.turnover is not None and value.proportional_paid is not None and value.recurring_paid is not None
    assert value.net_value is not None and value.gross_value is not None
    turnover = sum((abs(trade.notional.fraction) for trade in value.trades), Fraction())
    paid, recurring = turnover * fees.proportional.fraction, fees.recurring.fraction
    if (value.turnover.fraction, value.proportional_paid.fraction, value.recurring_paid.fraction) != (
        turnover,
        paid,
        recurring,
    ):
        raise ValueError("Executed fees differ from netted actual trades and recurring charge")
    actual = tuple(
        position.units.fraction * quote.value.fraction
        for position, quote in zip(value.after.positions, prices.quotes, strict=True)
        if quote.value is not None
    )
    net = value.after.cash.fraction + sum(actual, Fraction())
    if value.net_value.fraction != net or net + paid + recurring != value.gross_value.fraction:
        raise ValueError("Executed funded values violate exact self-financing conservation")
    if value.target is None:
        if any(trade.units.numerator != 0 for trade in value.trades):
            raise ValueError("Holding units cannot silently rebalance a funded book")
    elif actual != tuple(weight.value.fraction * net for weight in value.target.weights):
        raise ValueError("Executed holdings differ from declared post-charge target fractions")
