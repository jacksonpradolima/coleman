"""Explicit information and observation clocks without consulting wall time."""

from typing import Literal, Self

from pydantic import model_validator

from coleman.core.base import CoreImmutable
from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp


class DecisionTiming(CoreImmutable):
    """Describe the information boundary and target interval of a decision.

    Attributes
    ----------
    decision_at, information_cutoff : datetime
        Explicit zoned instants normalized to UTC; cutoff cannot follow decision.
    target_start, target_end : datetime
        Ordered target interval. Future mode requires decision <= start;
        ongoing mode requires start <= decision <= end.
    eligible_input_fingerprint : str
        Digest of the caller's eligible input snapshot, not proof of honesty.
    mode : {'future', 'ongoing'}
        Explicit target interpretation; no universal future-target assumption.

    Notes
    -----
    Performs no clock reads, filtering, storage or leakage detection. Untimed
    problems omit this entire object instead of inventing timestamps.
    """

    decision_at: Timestamp
    information_cutoff: Timestamp
    target_start: Timestamp
    target_end: Timestamp
    eligible_input_fingerprint: Fingerprint
    mode: Literal["future", "ongoing"]

    @model_validator(mode="after")
    def _clock_order(self) -> Self:
        """Check cutoff and mode-specific target ordering without reading time."""
        if self.information_cutoff > self.decision_at or self.target_start > self.target_end:
            raise ValueError("Cutoff/decision or target interval is reversed")
        if self.mode == "future" and self.decision_at > self.target_start:
            raise ValueError("Future target starts before the decision")
        if self.mode == "ongoing" and not self.target_start <= self.decision_at <= self.target_end:
            raise ValueError("Ongoing target must contain the decision instant")
        return self


class FeedbackTiming(CoreImmutable):
    """Retain event, availability, maturity and separately known receipt times.

    Attributes
    ----------
    event_at : datetime
        Zoned event instant, normalized to UTC.
    available_at, mature_at : datetime
        Explicit availability and maturity, each no earlier than the event.
    received_at : datetime or None
        Required nullable host receipt. None means unknown, never immediate.

    Notes
    -----
    Receipt must not predate availability. Maturity may follow receipt. No
    scheduling, implicit current time or provenance authentication occurs.
    """

    event_at: Timestamp
    available_at: Timestamp
    mature_at: Timestamp
    received_at: Timestamp | None

    @model_validator(mode="after")
    def _clock_order(self) -> Self:
        """Check event, maturity and known receipt without filling missing facts."""
        if min(self.available_at, self.mature_at) < self.event_at:
            raise ValueError("Availability and maturity cannot predate the event")
        if self.received_at is not None and self.received_at < self.available_at:
            raise ValueError("Receipt cannot predate availability")
        return self

    @property
    def usable_at(self) -> Timestamp | None:
        """Return the latest required clock, or None when receipt is unknown.

        Returns
        -------
        datetime or None
            Latest feedback-local clock, or None when receipt is unknown.
            The originating decision's clock is checked separately by feedback
            validation; this property alone does not authorize an update.

        Notes
        -----
        No clock is read and no feedback application is performed.
        """
        if self.received_at is None:
            return None
        return max(self.event_at, self.available_at, self.mature_at, self.received_at)
