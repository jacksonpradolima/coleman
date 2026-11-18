"""Exact, versioned public synthetic funded-book boundaries.

These records model synthetic credits and divisible asset units. They provide
neither a broker integration nor financial funding, margin or risk approval.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core._wire import BoundedContract
from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp
from coleman.problems.portfolio._validation import _validate_execution, _validate_valuation


class Exact(CoreImmutable):
    """Store a reduced rational without floating-point or currency rounding.

    Attributes
    ----------
    numerator, denominator : int
        Coprime integers with positive denominator, each at most4096 bits.
        Zero is represented only as0/1. Units belong to the enclosing field.

    Notes
    -----
    Input is exact and finite. Noncanonical fractions reject rather than being
    silently rewritten. This arithmetic profile has no lot size or tick size.
    """

    numerator: Annotated[int, Field(strict=True)]
    denominator: Annotated[int, Field(strict=True, gt=0)] = 1

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        """Bound rational work and reject duplicate representations."""
        if max(abs(self.numerator).bit_length(), self.denominator.bit_length()) > 4096:
            raise ValueError("Synthetic exact amount exceeds4096-bit capacity")
        if gcd(self.numerator, self.denominator) != 1:
            raise ValueError("Synthetic exact amount must be reduced")
        return self

    @classmethod
    def of(cls, value: Fraction | int | str) -> Exact:
        """Construct a canonical exact value from an explicit rational token.

        Parameters
        ----------
        value : Fraction or int or str
            Finite rational; floats are deliberately rejected.

        Returns
        -------
        Exact
            Reduced finite value, with no I/O or rounding.

        Raises
        ------
        ValueError, TypeError
            Invalid token, floating-point input or exceeded bit capacity.
        """
        if type(value) not in {Fraction, int, str}:
            raise TypeError("Exact values require Fraction, int or rational text")
        rational = Fraction(value)
        return cls(numerator=rational.numerator, denominator=rational.denominator)

    @property
    def fraction(self) -> Fraction:
        """Return the exact quantity in its enclosing field's declared units.

        Returns
        -------
        Fraction
            Reduced rational, without mutation or side effects.
        """
        return Fraction(self.numerator, self.denominator)


class PortfolioContract(BoundedContract):
    """Retain strict v1 known-field JSON within16MiB and depth64.

    Notes
    -----
    Unknown versions reject. No old TCPCI artifact is reinterpreted or migrated;
    this new namespace and its exact synthetic units are a separate contract.
    """

    _maximum_bytes = 16 * 1024**2


class Position(CoreImmutable):
    """Declare nonnegative divisible asset units, never mixture coefficients."""

    asset: Reference
    units: Exact

    @model_validator(mode="after")
    def _long_only(self) -> Self:
        """Reject short positions without inventing borrow or collateral."""
        if self.units.numerator < 0:
            raise ValueError("Funded-long-only-v1 forbids negative asset units")
        return self


class PriceQuote(CoreImmutable):
    """Retain price availability and source clocks in credit per asset unit.

    Attributes
    ----------
    asset : Reference
        Exact synthetic instrument identity.
    value : Exact or None
        Strictly positive credit/unit when observed or carried stale; otherwise
        null. Stale marks may value a book but cannot authorize trading.
    status : str
        Observed, carried stale, unknown or unavailable; zero is never missing.
    event_at, available_at : datetime
        UTC mark event and availability clocks, retained even for carried marks.
    """

    asset: Reference
    value: Exact | None
    status: Literal["observed", "stale", "unknown", "unavailable"]
    event_at: Timestamp
    available_at: Timestamp
    unit: Literal["synthetic-credit/asset-unit"] = "synthetic-credit/asset-unit"

    @model_validator(mode="after")
    def _availability(self) -> Self:
        """Preserve null states, positive prices and actual source ordering."""
        if (self.status in {"observed", "stale"}) != (self.value is not None):
            raise ValueError("Quote value and explicit availability disagree")
        if self.value is not None and self.value.numerator <= 0:
            raise ValueError("Observed synthetic prices must be strictly positive")
        if self.event_at > self.available_at:
            raise ValueError("A quote cannot be available before its event")
        return self


class PricePanel(PortfolioContract):
    """Bind a complete bounded authored price panel to an explicit cutoff."""

    source: Reference
    source_fingerprint: Fingerprint
    at: Timestamp
    quotes: Annotated[tuple[PriceQuote, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def _catalogue(self) -> Self:
        """Require unique ordered assets and reject future quote availability."""
        identities = tuple(item.asset.id for item in self.quotes)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("Price assets must be unique and sorted")
        if any(item.available_at > self.at for item in self.quotes):
            raise ValueError("Price panel cannot expose a future quote")
        return self


class AssetWeight(CoreImmutable):
    """Declare an exact nonnegative fraction of the post-charge funded value."""

    asset: Reference
    value: Exact

    @model_validator(mode="after")
    def _fraction(self) -> Self:
        """Reject shorts and leverage in the minimal funded profile."""
        if not 0 <= self.value.fraction <= 1:
            raise ValueError("Funded asset fractions require [0,1]")
        return self


class Target(PortfolioContract):
    """Set long-only post-charge asset fractions; the remainder stays in cash."""

    reference: Reference
    weights: Annotated[tuple[AssetWeight, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def _funded(self) -> Self:
        """Preserve asset order and require an explicit nonnegative cash fraction."""
        identities = tuple(item.asset.id for item in self.weights)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("Target assets must be unique and sorted")
        if sum((item.value.fraction for item in self.weights), Fraction()) > 1:
            raise ValueError("Funded-long-only-v1 forbids leverage or negative cash")
        return self


class Fees(PortfolioContract):
    """Declare proportional executed notional and per-period recurring charges.

    Attributes
    ----------
    proportional : Exact or None
        Credit per absolute traded credit; finite rate in[0,1). Each netted
        asset trade is charged once. No fee is inferred from coefficient changes.
    recurring : Exact or None
        Nonnegative synthetic credits charged once per supplied period, even
        when trading is zero. Initial execution may select an explicit zero.
    status : {'observed', 'unavailable'}
        Missing fees refuse execution. They never imply free trading.
    declared_at : datetime
        Explicit UTC availability of this caller-declared schedule. A future
        declaration cannot authorize current execution, even if rates are known
        to a retrospective caller. This is not source authentication.
    """

    reference: Reference
    period: Reference
    declared_at: Timestamp
    proportional: Exact | None
    recurring: Exact | None
    status: Literal["observed", "unavailable"]

    @model_validator(mode="after")
    def _rates(self) -> Self:
        """Reject unavailable/partial schedules and unqualified funding rates."""
        available = self.proportional is not None and self.recurring is not None
        if (self.status == "observed") != available:
            raise ValueError("Fee availability must match both declared charges")
        if self.status == "unavailable" and (self.proportional is not None or self.recurring is not None):
            raise ValueError("Unavailable fee schedules require explicit null charges")
        if available:
            assert self.proportional is not None and self.recurring is not None
            if not 0 <= self.proportional.fraction < 1 or self.recurring.numerator < 0:
                raise ValueError("Invalid proportional or recurring synthetic charge")
        return self


class Book(PortfolioContract):
    """Retain funded holdings and cash, independent from any decision transport.

    Attributes
    ----------
    reference : Reference
        Independent funded path identity, not a recipe or learner identity.
    cash : Exact
        Nonnegative synthetic credits. No borrowing, deposits or withdrawals.
    positions : tuple of Position
        Sorted complete instrument catalogue, including explicit zero holdings.
    sequence : int
        Accepted executed periods, not observations, solver calls or deliveries.
    charged_periods : tuple of Reference
        Recurring-fee periods already settled in this bounded path.
    """

    reference: Reference
    cash: Exact
    positions: Annotated[tuple[Position, ...], Field(min_length=1, max_length=8)]
    sequence: Annotated[int, Field(strict=True, ge=0, le=128)] = 0
    charged_periods: Annotated[tuple[Reference, ...], Field(max_length=128)] = ()
    profile: Literal["funded-long-only-exact-v1"] = "funded-long-only-exact-v1"
    currency: Literal["synthetic-credit"] = "synthetic-credit"

    @model_validator(mode="after")
    def _funding(self) -> Self:
        """Validate complete catalogue, nonnegative cash and distinct fee periods."""
        if self.cash.numerator < 0:
            raise ValueError("Funded cash cannot be negative")
        identities = tuple(item.asset.id for item in self.positions)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("Book positions must be unique and sorted")
        if len({item.id for item in self.charged_periods}) != len(self.charged_periods):
            raise ValueError("A recurring-fee period cannot be charged twice")
        if self.sequence != len(self.charged_periods):
            raise ValueError("Every accepted funded period requires its settled recurring-fee frontier")
        return self


class Valuation(PortfolioContract):
    """Keep complete, carried-stale or unknown mark-to-market value explicit.

    Attributes
    ----------
    value : Exact or None
        Total synthetic credits, including cash. Observed zero is a known empty
        value; null means a positive holding lacks an available price.
    asset_values : tuple of Exact or None
        Marked synthetic credits in the exact price catalogue's order. A zero
        holding contributes known zero even with an unavailable price.
    drifted_weights : tuple of Exact or None
        Dimensionless marked asset value divided by total value. Cash is the
        residual. Unknown or zero total leaves every ratio null, never zero.
    """

    book_fingerprint: Fingerprint
    prices: PricePanel
    value: Exact | None
    status: Literal["complete", "stale", "unknown"]
    asset_values: tuple[Exact | None, ...]
    drifted_weights: tuple[Exact | None, ...]

    @model_validator(mode="after")
    def _values(self) -> Self:
        """Validate exact mark/weight structure without authenticating its source."""
        _validate_valuation(self)
        return self


class Trade(CoreImmutable):
    """Retain one netted signed trade; positive units buy, negative units sell.

    Attributes
    ----------
    units : Exact
        Signed divisible asset units added to the original holding.
    notional : Exact
        Signed synthetic credits, exactly units times the original execution
        price. Full absolute notional, rather than signed net cash, incurs fees.
    """

    asset: Reference
    units: Exact
    notional: Exact


class Execution(PortfolioContract):
    """Record accepted funded execution or an explicit unchanged-book refusal.

    Attributes
    ----------
    gross_value, net_value : Exact or None
        Pre-charge and post-charge synthetic credits at the same price panel.
        Refused settlement retains any known gross mark but has null net value.
    turnover : Exact or None
        Sum of absolute netted executed synthetic-credit notionals, without a
        half factor. Zero is an observed hold; null means execution was refused.
    proportional_paid, recurring_paid : Exact or None
        Separate actual modeled synthetic-credit charges. Explicit observed zero
        differs from null on a refused execution. Neither is an objective penalty.

    Notes
    -----
    Gross value is pre-charge marked wealth; net value is post-charge wealth
    at the same prices. Turnover is full absolute executed notional, without a
    half factor. This is not expert-coefficient movement or an optimizer penalty.
    """

    request: Reference
    input_fingerprint: Fingerprint
    before: Book
    after: Book
    valuation: Valuation
    target: Target | None
    fees: Fees
    status: Literal["executed", "refused"]
    reason: Literal["settled", "missing_price", "stale_price", "missing_fee", "ineligible_fee", "insufficient_funding"]
    trades: tuple[Trade, ...]
    gross_value: Exact | None
    net_value: Exact | None
    turnover: Exact | None
    proportional_paid: Exact | None
    recurring_paid: Exact | None

    @model_validator(mode="after")
    def _settlement(self) -> Self:
        """Reject false committed, fully valued or free-execution claims."""
        charges = (self.net_value, self.turnover, self.proportional_paid, self.recurring_paid)
        if self.status == "refused":
            if self.before.to_json() != self.after.to_json() or self.trades or any(v is not None for v in charges):
                raise ValueError("Refused execution must retain book and unknown settlement quantities")
        elif self.reason != "settled" or any(v is None for v in charges) or self.gross_value is None:
            raise ValueError("Executed settlement requires complete exact quantities")
        _validate_execution(self)
        return self
