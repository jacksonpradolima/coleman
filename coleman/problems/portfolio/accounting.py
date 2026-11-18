"""Pure exact self-financing accounting for authored synthetic long-only books."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, TypedDict

from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.problems.portfolio.contracts import (
    Book,
    Exact,
    Execution,
    Fees,
    Position,
    PricePanel,
    Target,
    Trade,
    Valuation,
)


def _catalogue(book: Book, prices: PricePanel, target: Target | None) -> None:
    """Require exact member versions, not merely matching display labels."""
    assets = tuple(item.asset for item in book.positions)
    if tuple(item.asset for item in prices.quotes) != assets:
        raise ValueError("Price catalogue differs from the funded book")
    if target is not None and tuple(item.asset for item in target.weights) != assets:
        raise ValueError("Target catalogue differs from the funded book")


def mark(book: Book, prices: PricePanel) -> Valuation:
    """Mark actual held units, retaining price drift and missingness.

    Parameters
    ----------
    book : Book
        Exact nonnegative funded cash and asset units.
    prices : PricePanel
        Same exact catalogue and explicit cutoff. Carried stale prices retain
        their original event/availability timestamps and permit valuation only.

    Returns
    -------
    Valuation
        Exact synthetic-credit total and asset weights when known. Missing a
        positive holding's price leaves total and weights null. A zero holding
        contributes known zero even when its quote is unavailable.

    Raises
    ------
    ValueError
        Catalogue/version mismatch or rational capacity overflow.

    Notes
    -----
    Performs no mutation, I/O, execution or carry-forward price invention.
    """
    book, prices = Book.model_validate(book), PricePanel.model_validate(prices)
    _catalogue(book, prices, None)
    values = _asset_values(book, prices)
    known = all(value is not None for value in values)
    total = book.cash.fraction + sum((value for value in values if value is not None), Fraction()) if known else None
    stale = any(
        (quote.status == "stale" or quote.event_at != prices.at) and position.units.numerator != 0
        for position, quote in zip(book.positions, prices.quotes, strict=True)
    )
    return Valuation(
        schema_version=1,
        book_fingerprint=book.fingerprint,
        prices=prices,
        value=None if total is None else Exact.of(total),
        status="unknown" if not known else "stale" if stale else "complete",
        asset_values=tuple(None if value is None else Exact.of(value) for value in values),
        drifted_weights=tuple(
            None if total is None or total == 0 or value is None else Exact.of(value / total) for value in values
        ),
    )


def _asset_values(book: Book, prices: PricePanel) -> tuple[Fraction | None, ...]:
    """Mark held units only, with zero holdings distinct from missing prices."""
    values = tuple(
        Fraction()
        if position.units.numerator == 0
        else None
        if quote.value is None
        else position.units.fraction * quote.value.fraction
        for position, quote in zip(book.positions, prices.quotes, strict=True)
    )
    return values


def _post_charge_value(
    gross: Fraction, amounts: tuple[Fraction, ...], weights: tuple[Fraction, ...], rate: Fraction, recurring: Fraction
) -> Fraction | None:
    """Solve the exact monotone piecewise-linear funding identity over[0,V]."""
    available = gross - recurring
    if available < 0 or rate * sum(amounts) > available:
        return None
    knots = {Fraction(), available}
    knots.update(
        amount / weight
        for amount, weight in zip(amounts, weights, strict=True)
        if weight and 0 < amount / weight < available
    )
    ordered = sorted(knots)
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        probe = (lower + upper) / 2
        signs = tuple(1 if weight * probe >= amount else -1 for amount, weight in zip(amounts, weights, strict=True))
        intercept = rate * sum((sign * amount for sign, amount in zip(signs, amounts, strict=True)), Fraction())
        slope = 1 + rate * sum((sign * weight for sign, weight in zip(signs, weights, strict=True)), Fraction())
        root = (available + intercept) / slope
        if lower <= root <= upper:
            return root
    return Fraction() if available == 0 else None


def _availability(
    prices: PricePanel, fees: Fees
) -> Literal["missing_price", "stale_price", "missing_fee", "ineligible_fee"] | None:
    """Choose explicit refusal priority without inventing missing evidence."""
    return (
        "missing_price"
        if any(item.status in {"unknown", "unavailable"} for item in prices.quotes)
        else "stale_price"
        if any(item.status == "stale" or item.event_at != prices.at for item in prices.quotes)
        else "missing_fee"
        if fees.status != "observed"
        else "ineligible_fee"
        if fees.declared_at > prices.at
        else None
    )


def execute(book: Book, prices: PricePanel, target: Target | None, fees: Fees, *, request: Reference) -> Execution:
    """Execute one netted funded target or hold actual units for one period.

    Parameters
    ----------
    book : Book
        Independently funded starting path. Existing charged periods cannot be
        submitted again; delivery deduplication belongs to the outer session.
    prices : PricePanel
        Mark-to-market occurs first at these supplied exogenous prices.
    target : Target or None
        Post-charge long-only target fractions. Null holds units; it does not
        rebalance drifted weights. Opposing recipe targets must be combined
        before this call, so only each asset's actual net trade is charged.
    fees : Fees
        Explicit proportional rate and recurring charge for an uncharged period.
    request : Reference
        Stable original producer/delivery identity, retained in the receipt.

    Returns
    -------
    Execution
        Exact receipt. The target solve satisfies W+c*sum(abs(w_i*W-h_i))+F=V.
        Its unique root is found by finite rational breakpoint enumeration.
        A hold charges only recurring cash; insufficient cash refuses the hold.

    Raises
    ------
    ValueError
        Wrong catalogue, duplicate charged period, selected128-period capacity
        or exact arithmetic beyond4096 bits. Unknown prices/fees and well-formed
        insufficient funding produce an unchanged-book refused receipt.

    Notes
    -----
    Pure math has no delivery idempotency, locks or durable publication. No
    contribution, withdrawal, short, leverage, interest, borrow, margin, spread,
    lot-size or market-impact model is supplied. It is not an execution service.
    """
    book, prices = Book.model_validate(book), PricePanel.model_validate(prices)
    fees, request = Fees.model_validate(fees), Reference.model_validate(request)
    target = None if target is None else Target.model_validate(target)
    _catalogue(book, prices, target)
    if book.sequence >= 128 or any(period.id == fees.period.id for period in book.charged_periods):
        raise ValueError("Book capacity or recurring-fee period already consumed")
    valuation = mark(book, prices)
    identity = fingerprint(
        {
            "book": book.fingerprint,
            "prices": prices.fingerprint,
            "target": None if target is None else target.fingerprint,
            "fees": fees.fingerprint,
            "request": request.model_dump(mode="json"),
        }
    )
    base: _ReceiptInputs = {
        "schema_version": 1,
        "request": request,
        "input_fingerprint": identity,
        "before": book,
        "valuation": valuation,
        "target": target,
        "fees": fees,
        "gross_value": valuation.value,
    }
    reason: Literal["missing_price", "stale_price", "missing_fee", "ineligible_fee", "insufficient_funding"] | None = (
        _availability(prices, fees)
    )
    if reason is None:
        settled = _funded_receipt(base)
        if settled is not None:
            return settled
        reason = "insufficient_funding"
    return Execution(
        **base,
        after=book,
        status="refused",
        reason=reason,
        trades=(),
        net_value=None,
        turnover=None,
        proportional_paid=None,
        recurring_paid=None,
    )


class _ReceiptInputs(TypedDict):
    """Type the finite private constructor arguments; no untyped boundary."""

    schema_version: int
    request: Reference
    input_fingerprint: str
    before: Book
    valuation: Valuation
    target: Target | None
    fees: Fees
    gross_value: Exact | None


def _funded_receipt(base: _ReceiptInputs) -> Execution | None:
    """Solve and settle one fully observed funded target, or return infeasible."""
    book, prices = base["before"], base["valuation"].prices
    target, fees, valuation = base["target"], base["fees"], base["valuation"]
    assert fees.proportional is not None and fees.recurring is not None and valuation.value is not None
    amount = tuple(item.fraction for item in valuation.asset_values if item is not None)
    recurring = fees.recurring.fraction
    net = valuation.value.fraction - recurring if target is None and book.cash.fraction >= recurring else None
    if target is not None:
        net = _post_charge_value(
            valuation.value.fraction,
            amount,
            tuple(item.value.fraction for item in target.weights),
            fees.proportional.fraction,
            recurring,
        )
    if net is not None:
        desired = amount if target is None else tuple(item.value.fraction * net for item in target.weights)
        trades = tuple(
            Trade(
                asset=position.asset,
                units=Exact.of((value - old) / quote.value.fraction),
                notional=Exact.of(value - old),
            )
            for position, quote, value, old in zip(book.positions, prices.quotes, desired, amount, strict=True)
            if quote.value is not None
        )
        turnover = sum((abs(trade.notional.fraction) for trade in trades), Fraction())
        paid = turnover * fees.proportional.fraction
        if net + paid + recurring != valuation.value.fraction:
            raise ValueError("Exact funded settlement failed its conservation identity")
        positions = tuple(
            Position(asset=old.asset, units=Exact.of(old.units.fraction + trade.units.fraction))
            for old, trade in zip(book.positions, trades, strict=True)
        )
        after = Book(
            schema_version=1,
            reference=book.reference,
            cash=Exact.of(net - sum(desired)),
            positions=positions,
            sequence=book.sequence + 1,
            charged_periods=(*book.charged_periods, fees.period),
        )
        return Execution(
            **base,
            after=after,
            status="executed",
            reason="settled",
            trades=trades,
            net_value=Exact.of(net),
            turnover=Exact.of(turnover),
            proportional_paid=Exact.of(paid),
            recurring_paid=Exact.of(recurring),
        )
    return None
