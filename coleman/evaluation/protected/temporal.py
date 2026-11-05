"""Point-in-time temporal eligibility, grouping and interval exclusion contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from coleman.core.temporal import Timestamp as Timestamp
from coleman.results.domain import Identifier

Partition = Literal["train", "validation", "test"]


class TemporalObservation(BaseModel):  # pylint: disable=too-few-public-methods
    """Metadata for one revision of an observation; no outcomes are inferred.

    All times are explicitly zoned and normalized to UTC. Outcome availability
    cannot precede the event. Target intervals are closed, start at ``event_at``
    and optionally end at ``target_end`` (otherwise the event itself). A revision
    retains its event/group identity but has its own availability and revision ID.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    observation_id: Identifier
    revision_id: Identifier = "original"
    group_id: Identifier
    event_at: Timestamp
    available_at: Timestamp
    target_end: Timestamp | None = None

    @model_validator(mode="after")
    def _chronology(self) -> Self:
        """Reject impossible availability and reversed target intervals."""
        if self.available_at < self.event_at:
            raise ValueError("Outcome availability cannot precede the event")
        if self.target_end is not None and self.target_end < self.event_at:
            raise ValueError("Target interval ends before its event")
        return self


class SplitDecision(BaseModel):  # pylint: disable=too-few-public-methods
    """Auditable inclusion/exclusion; excluded rows retain their nominal partition."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    observation: TemporalObservation
    partition: Partition | None
    eligible: bool
    reason: Literal["eligible", "outside_window", "outcome_not_available", "overlapping_target"]


class TemporalPlan(BaseModel):
    """Chronological closed-right splits with conservative point-in-time purging.

    Train is (-infinity, train_end], validation is (train_end, validation_end],
    test is (validation_end, test_end]. Event time fixes the nominal partition;
    late availability never moves a training row into validation/test. Availability
    and the entire target interval must be within that partition's cutoff. Exact
    endpoints are inclusive. Missing metadata fails; ineligible rows are excluded
    with reasons. Groups spanning any two nominal partitions fail even when one
    of their observations would otherwise be excluded for late availability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    train_end: Timestamp
    validation_end: Timestamp
    test_end: Timestamp
    event_field: Identifier = "event_at"
    availability_field: Identifier = "available_at"
    group_field: Identifier = "group_id"
    observation_field: Identifier = "observation_id"
    revision_field: Identifier = "revision_id"
    target_end_field: Identifier = "target_end"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require increasing cutoffs and distinct mapped metadata columns."""
        if not self.train_end < self.validation_end < self.test_end:
            raise ValueError("Cutoffs must satisfy train_end < validation_end < test_end")
        fields = (
            self.event_field,
            self.availability_field,
            self.group_field,
            self.observation_field,
            self.revision_field,
            self.target_end_field,
        )
        if len(set(fields)) != len(fields):
            raise ValueError("Metadata field mappings must be distinct")
        return self

    def observation(self, values: Mapping[str, Any]) -> TemporalObservation:
        """Convert a dataset mapping once at the boundary; missing required keys fail.

        Unused dataset columns remain outside the typed metadata contract. Optional
        revision/interval columns may be omitted; explicit null revision is invalid.
        No fallback timestamp, group or observation identity is invented.
        """
        return TemporalObservation.model_validate(
            {
                "observation_id": values[self.observation_field],
                "group_id": values[self.group_field],
                "event_at": values[self.event_field],
                "available_at": values[self.availability_field],
                "revision_id": values.get(self.revision_field, "original"),
                "target_end": values.get(self.target_end_field),
            }
        )

    def nominal_partition(self, observation: TemporalObservation) -> Partition | None:
        """Assign event time only, using closed-right cutoff inclusivity."""
        if observation.event_at <= self.train_end:
            return "train"
        if observation.event_at <= self.validation_end:
            return "validation"
        if observation.event_at <= self.test_end:
            return "test"
        return None

    def decide(self, observation: TemporalObservation) -> SplitDecision:
        """Validate one metadata object and record eligibility without reading values."""
        observation = TemporalObservation.model_validate(observation)
        partition = self.nominal_partition(observation)
        if partition is None:
            reason = "outside_window"
        else:
            cutoff = {"train": self.train_end, "validation": self.validation_end, "test": self.test_end}[partition]
            if (observation.target_end or observation.event_at) > cutoff:
                reason = "overlapping_target"
            elif observation.available_at > cutoff:
                reason = "outcome_not_available"
            else:
                reason = "eligible"
        return SplitDecision(observation=observation, partition=partition, eligible=reason == "eligible", reason=reason)

    def training_eligible(self, observation: TemporalObservation) -> bool:
        """Return train eligibility for typed metadata; no automatic fold reassignment."""
        decision = self.decide(observation)
        return decision.partition == "train" and decision.eligible

    def split(self, observations: tuple[TemporalObservation, ...]) -> tuple[SplitDecision, ...]:
        """Validate revision identity/group separation, returning deterministic decisions.

        Duplicate (observation_id, revision_id) or an observation ID changing its
        event/group identity fails. This excludes post-cutoff revisions without
        overwriting legitimate historical observations. No source data is mutated.
        """
        identities: set[tuple[str, str]] = set()
        events: dict[str, tuple[str, datetime]] = {}
        groups: dict[str, Partition] = {}
        decisions = []
        for raw in observations:
            observation = TemporalObservation.model_validate(raw)
            identity = observation.observation_id, observation.revision_id
            if identity in identities:
                raise ValueError("Duplicate observation revision")
            identities.add(identity)
            event = observation.group_id, observation.event_at
            previous = events.setdefault(observation.observation_id, event)
            if previous != event:
                raise ValueError("Revision changed observation event/group identity")
            partition = self.nominal_partition(observation)
            if partition is not None:
                previous_partition = groups.setdefault(observation.group_id, partition)
                if previous_partition != partition:
                    raise ValueError("A group spans temporal partitions")
            decisions.append(self.decide(observation))
        return tuple(
            sorted(decisions, key=lambda item: (item.observation.observation_id, item.observation.revision_id))
        )
