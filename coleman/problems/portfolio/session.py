"""Bounded in-memory delivery ownership around the pure funded-book function."""

from __future__ import annotations

from threading import RLock
from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.core.identity import Reference
from coleman.core.primitives import canonical_json
from coleman.core.reading import bounded_json_bytes
from coleman.problems.portfolio.accounting import execute
from coleman.problems.portfolio.contracts import Book, Execution, Fees, PortfolioContract, PricePanel, Target


class BookSnapshot(PortfolioContract):
    """Retain the original book and complete accepted/refused delivery receipts.

    Notes
    -----
    Validation independently reapplies the declared exact pure transition and
    checks every successor, request and input digest. This is a supplied history,
    not authenticated evidence that an external trade occurred. V1 continuation
    requires these original units, fees, prices and exact arithmetic profile.
    """

    initial: Book
    receipts: Annotated[tuple[Execution, ...], Field(max_length=128)] = ()

    @model_validator(mode="after")
    def _chain(self) -> Self:
        """Check complete immutable successors without replaying physical effects."""
        book = self.initial
        requests: set[str] = set()
        last_at = None
        for receipt in self.receipts:
            if receipt.request.id in requests or receipt.before.to_json() != book.to_json():
                raise ValueError("Book delivery history has duplicate identity or broken lineage")
            if last_at is not None and receipt.valuation.prices.at < last_at:
                raise ValueError("Book delivery clocks cannot rewind")
            expected = execute(book, receipt.valuation.prices, receipt.target, receipt.fees, request=receipt.request)
            if expected.to_json() != receipt.to_json():
                raise ValueError("Book receipt differs from its exact original transition")
            book = receipt.after
            requests.add(receipt.request.id)
            last_at = receipt.valuation.prices.at
        return self

    @property
    def book(self) -> Book:
        """Return the latest accepted book, including a final unchanged refusal.

        Returns
        -------
        Book
            Exact held units and cash; not a valuation or external execution ACK.
        """
        return self.receipts[-1].after if self.receipts else self.initial


class BookSession:
    """Serialize local delivery and deduplicate exact original producer requests.

    Parameters
    ----------
    initial : Book
        Independently funded starting book, retained in the complete snapshot.

    Notes
    -----
    The reentrant lock covers a single in-process owner. Snapshots do not supply
    cross-process durability; the existing COLEMAN state store owns publication,
    compare-and-swap and acknowledgement when this component is wrapped there.
    This session never submits an external trade or claims exactly-once effects.
    """

    def __init__(self, initial: Book) -> None:
        """Create a local owner without I/O or source discovery."""
        self._state = BookSnapshot(schema_version=1, initial=initial)
        self._lock = RLock()

    @property
    def book(self) -> Book:
        """Return the current immutable funded state.

        Returns
        -------
        Book
            Latest accepted local mathematical state.
        """
        with self._lock:
            return self._state.book

    def deliver(self, prices: PricePanel, target: Target | None, fees: Fees, *, request: Reference) -> Execution:
        """Apply one logical request, returning an exact retry's original receipt.

        Parameters
        ----------
        prices : PricePanel
            Original causal exogenous marks, in nondecreasing processing order.
        target : Target or None
            Original proposed post-charge asset fractions, or a hold of units.
        fees : Fees
            Original observed/unavailable schedule; never inferred from a retry.
        request : Reference
            Exact producer identity/version. Same ID with different content or
            revision conflicts, including retries of a refused request.

        Returns
        -------
        Execution
            Original retained receipt or a newly committed local successor.

        Raises
        ------
        ValueError
            Conflicting delivery, backwards clock, capacity or typed transition
            error. A failed validation leaves the complete state unchanged.

        Notes
        -----
        Holds the local owner lock through validation and state replacement.
        No filesystem or external effects occur; refused receipts remain visible.
        The complete initial book and receipt history must also fit the selected
        16MiB/depth64 snapshot profile. A byte-capacity refusal can precede the
        128-receipt limit; it accepts no new state or deduplication entry.
        """
        request = Reference.model_validate(request)
        with self._lock:
            for receipt in self._state.receipts:
                if receipt.request.id == request.id:
                    proposed = execute(receipt.before, prices, target, fees, request=request)
                    if proposed.to_json() != receipt.to_json():
                        raise ValueError("Conflicting original book delivery")
                    return receipt
            receipt = execute(self.book, prices, target, fees, request=request)
            if len(self._state.receipts) >= 128:
                raise ValueError("Book delivery capacity exhausted")
            if self._state.receipts and prices.at < self._state.receipts[-1].valuation.prices.at:
                raise ValueError("Book delivery clocks cannot rewind")
            # Every historical receipt is already validated and immutable. The
            # new receipt came from the exact pure function; replaying all past
            # transitions per append adds work but no new validation boundary.
            candidate = BookSnapshot.model_construct(
                schema_version=1, initial=self._state.initial, receipts=(*self._state.receipts, receipt)
            )
            # A valid next receipt can exceed the aggregate snapshot capacity
            # before the receipt-count cap. Check the complete selected wire
            # profile, reserving its artifact LF, before accepting any change.
            bounded_json_bytes(
                canonical_json(candidate.model_dump(mode="json")),
                BookSnapshot._maximum_bytes - 1,  # pylint: disable=protected-access
            )
            self._state = candidate
            return receipt

    def fork(self) -> BookSession:
        """Create an independent lock/owner over the same immutable valid history.

        Returns
        -------
        BookSession
            Fresh mutable owner and lock. Only immutable validated records are
            shared; later delivery replaces its own state and cannot affect this
            owner. No deserialization, I/O or history replay occurs.
        """
        with self._lock:
            owner = BookSession(self._state.initial)
            # Same-class fork shares only the already validated immutable value.
            owner._state = self._state  # pylint: disable=protected-access
            return owner

    def export_state(self) -> BookSnapshot:
        """Capture the entire immutable local receipt and funded-state frontier.

        Returns
        -------
        BookSnapshot
            Known-version complete continuation, with no I/O or ACK claim.
        """
        with self._lock:
            return self._state

    def restore_state(self, state: BookSnapshot, /) -> None:
        """Restore a validated complete same-initial-book continuation atomically.

        Parameters
        ----------
        state : BookSnapshot
            Original book and every receipt, within the selected128-delivery cap.

        Raises
        ------
        ValueError
            Unknown version, malformed transition or a different original owner.
            Live state remains unchanged on refusal. No migration is implicit.
        """
        checked = BookSnapshot.model_validate(state)
        with self._lock:
            if checked.initial.to_json() != self._state.initial.to_json():
                raise ValueError("Book continuation belongs to another original owner")
            self._state = checked
