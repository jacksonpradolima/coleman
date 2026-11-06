"""Transport-independent point-in-time observations and explicitly interpreted decision signals.

Provider clock/identity declarations are attestations. This module validates their
consistency; it cannot prove that an upstream provider supplied every observation.
No provider, clock, filesystem, native runner, CLI or MCP is invoked here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import Immutable
from coleman.core.quantities import Quantity
from coleman.core.splits import TemporalObservation
from coleman.core.temporal import Timestamp
from coleman.core.units import UnitSemantics
from coleman.results.domain import DomainRecord, Fingerprint, Identifier, Measurement, ResultProvenance, fingerprint
from coleman.results.versioned import VersionedContract

MAXIMUM_REVISIONS = 4096
EVENT_AGE_SECONDS = UnitSemantics(unit="s", meaning="source-event-age-at-decision-v1")


class ObservationRevision(VersionedContract):
    """One immutable measured revision with source availability and first local receipt.

    ``metadata`` reuses protected-evaluation time/group/revision validation.
    ``received_at`` is the first receipt, retained across duplicate deliveries, and
    cannot precede source availability. Effective availability is the later of
    those instants. Clock disagreement fails rather than inferring an adjustment.
    A correction names its direct predecessor; null names a lineage root.
    """

    metadata: TemporalObservation
    entity_id: Identifier
    received_at: Timestamp
    supersedes: Identifier | None
    measurement: Measurement

    @model_validator(mode="after")
    def _receipt(self) -> Self:
        """Reject impossible receipt chronology and self-referential corrections."""
        if self.received_at < self.metadata.available_at:
            raise ValueError("First receipt cannot precede source availability")
        if self.supersedes == self.metadata.revision_id:
            raise ValueError("A revision cannot supersede itself")
        return self

    @property
    def effective_metadata(self) -> TemporalObservation:
        """Return reused temporal metadata with the actual information-availability cutoff."""
        return TemporalObservation.model_validate(
            {**self.metadata.model_dump(), "available_at": max(self.received_at, self.metadata.available_at)}
        )


class ObservationSnapshot(VersionedContract):
    """Latest eligible revisions declared visible at one cutoff, canonical by logical observation.

    This bounded selected view excludes future values from its fingerprint.
    Source completeness/latest-revision selection is established by
    ``ObservationHistory.as_of`` for the supplied history, not by an isolated
    serialized view's metadata. No independent-observation claim follows.
    """

    source_id: Identifier
    source_version: Identifier
    information_cutoff: Timestamp
    observations: Annotated[tuple[ObservationRevision, ...], Field(max_length=MAXIMUM_REVISIONS)]

    @model_validator(mode="after")
    def _eligible(self) -> Self:
        """Reject unavailable/duplicate logical observations and normalize unordered input."""
        ids = [row.metadata.observation_id for row in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Snapshot must contain one revision per logical observation")
        if any(row.effective_metadata.available_at > self.information_cutoff for row in self.observations):
            raise ValueError("Snapshot contains information unavailable at its cutoff")
        object.__setattr__(
            self, "observations", tuple(sorted(self.observations, key=lambda row: row.metadata.observation_id))
        )
        return self


class ObservationHistory(VersionedContract):
    """Bounded source history with complete, unambiguous revision lineages.

    Source ID/version describe the appendable provider/schema identity, not a hash
    of future history. Whole-history fingerprints intentionally change on append;
    selected-view fingerprints do not include those future values. Identical
    duplicate revisions require identical canonical JSON; changed values, signed-zero
    representations or first receipts conflict.
    Missing parents, forks, cycles and changed logical/measurement semantics fail.
    Out-of-order delivery is supported; explicit lineage orders equal timestamps.
    """

    source_id: Identifier
    source_version: Identifier
    revisions: Annotated[tuple[ObservationRevision, ...], Field(max_length=MAXIMUM_REVISIONS)]

    @model_validator(mode="after")
    def _history(self) -> Self:
        """Deduplicate exact retries, validate all source lineages and canonicalize identity."""
        unique: dict[tuple[str, str], ObservationRevision] = {}
        for row in self.revisions:
            key = row.metadata.observation_id, row.metadata.revision_id
            previous = unique.setdefault(key, row)
            if previous is not row and previous.to_json() != row.to_json():
                raise ValueError("Conflicting observation revision or first receipt")
        rows = tuple(unique[key] for key in sorted(unique))
        _chains(rows)
        object.__setattr__(self, "revisions", rows)
        return self

    def as_of(self, information_cutoff: Timestamp) -> ObservationSnapshot:
        """Select the latest eligible lineage member using receipt and source availability.

        Inclusive cutoff and explicit predecessor order handle ties without lexical
        revision inference. A future correction cannot overwrite earlier knowledge.
        Selection is a bounded scan, without indexes, I/O or shared mutable state.
        Unknown/unavailable measurements remain typed states, not invented labels.
        """
        history = ObservationHistory.model_validate(self)
        empty = ObservationSnapshot(
            schema_version=1,
            source_id=history.source_id,
            source_version=history.source_version,
            information_cutoff=information_cutoff,
            observations=(),
        )
        selected = []
        for chain in _chains(history.revisions):
            eligible = next(
                (row for row in reversed(chain) if row.effective_metadata.available_at <= empty.information_cutoff),
                None,
            )
            if eligible is not None:
                selected.append(eligible)
        return ObservationSnapshot.model_validate({**empty.model_dump(), "observations": tuple(selected)})


def _chains(rows: tuple[ObservationRevision, ...]) -> tuple[tuple[ObservationRevision, ...], ...]:
    """Validate and order bounded lineages iteratively, without recursive depth dependence."""
    groups: dict[str, list[ObservationRevision]] = defaultdict(list)
    for row in rows:
        groups[row.metadata.observation_id].append(row)
    return tuple(_chain(tuple(groups[name])) for name in sorted(groups))


def _logical_semantics(row: ObservationRevision) -> tuple:
    """Retain event/entity/group/target and measurement definition across revisions."""
    meta, measure = row.metadata, row.measurement
    return (
        row.entity_id,
        meta.group_id,
        meta.event_at,
        meta.target_end,
        measure.name,
        measure.unit,
        measure.meaning,
        measure.direction,
        measure.minimum,
    )


def _chain(rows: tuple[ObservationRevision, ...]) -> tuple[ObservationRevision, ...]:
    """Reject ambiguous branches, unavailable parents and backward source revision clocks."""
    by_id = {row.metadata.revision_id: row for row in rows}
    roots = [row for row in rows if row.supersedes is None]
    if len(roots) != 1 or any(_logical_semantics(row) != _logical_semantics(roots[0]) for row in rows):
        raise ValueError("Revision lineage requires one root and stable logical/measurement semantics")
    children = _successors(rows, by_id)
    ordered = [roots[0]]
    # Each unique node has one parent and the root has none. Any cycle is
    # disconnected from the root and is rejected by the count check below.
    while ordered[-1].metadata.revision_id in children:
        ordered.append(children[ordered[-1].metadata.revision_id])
    if len(ordered) != len(rows):
        raise ValueError("Disconnected or cyclic revision lineage")
    return tuple(ordered)


def _successors(
    rows: tuple[ObservationRevision, ...], by_id: dict[str, ObservationRevision]
) -> dict[str, ObservationRevision]:
    """Validate correction edges without requiring monotonic receipt/delivery order."""
    children: dict[str, ObservationRevision] = {}
    for row in rows:
        if row.supersedes is None:
            continue
        parent = by_id.get(row.supersedes)
        if parent is None or row.supersedes in children:
            raise ValueError("Revision lineage has a missing parent or ambiguous fork")
        if row.metadata.available_at < parent.metadata.available_at:
            raise ValueError("Source correction availability precedes its predecessor")
        children[row.supersedes] = row
    return children


class SignalRequest(VersionedContract):
    """One logical future decision target and its explicit information cutoff/horizon.

    All instants use the existing aware UTC temporal contract. The information
    cutoff cannot exceed decision time; the closed target interval starts no
    earlier than the decision. Calendar/clock synchronization is a source assumption.
    """

    decision_id: Identifier
    entity_id: Identifier
    decided_at: Timestamp
    information_cutoff: Timestamp
    target_start: Timestamp
    target_end: Timestamp

    @model_validator(mode="after")
    def _chronology(self) -> Self:
        """Reject future information and target intervals preceding their decision."""
        if not self.information_cutoff <= self.decided_at <= self.target_start <= self.target_end:
            raise ValueError("Signal requires cutoff <= decision <= target start <= target end")
        return self


class SignalProvenance(Immutable):
    """Explicit producer/model/source identity and actual historical fit/freeze cutoffs.

    Fingerprints identify the eligible input view and frozen model, not the full
    source history containing future corrections. They attest reproducible inputs,
    not statistical independence, calibrated probability or causal effectiveness.
    """

    source_id: Identifier
    source_version: Identifier
    source_fingerprint: Fingerprint
    model_fingerprint: Fingerprint
    producer: Identifier
    producer_version: Identifier
    model_version: Identifier
    training_cutoff: Timestamp
    frozen_at: Timestamp

    @model_validator(mode="after")
    def _freeze(self) -> Self:
        """Prevent a fitted model claiming a freeze before its training cutoff."""
        if self.training_cutoff > self.frozen_at:
            raise ValueError("Model freeze precedes its training cutoff")
        return self


class DecisionSignal(VersionedContract):
    """Versioned quantity estimate or ranking score, explicitly separate from realized outcomes.

    Measurement availability denotes availability of the signal value, not an
    observed future outcome. Signed scores are valid when their declared lower
    bound permits them; zero is real. Unknown, stale/unavailable and inapplicable
    signals retain distinct null states. No calibrated probability is inferred.
    ``maximum_event_age`` retains the declared freshness rule when used; null
    means unspecified by this producer, not a guarantee of unlimited freshness.
    """

    request: SignalRequest
    interpretation: Literal["quantity_estimate", "priority_score"]
    measurement: Measurement
    availability: Literal["available", "missing", "stale", "unavailable", "not_applicable"]
    provenance: SignalProvenance
    maximum_event_age: Quantity | None = None

    @model_validator(mode="after")
    def _availability(self) -> Self:
        """Check signal missingness and prevent using a future-frozen model at an earlier decision."""
        allowed = {
            "observed": {"available"},
            "unknown": {"missing"},
            "unavailable": {"stale", "unavailable"},
            "not_applicable": {"not_applicable"},
        }
        if self.availability not in allowed[self.measurement.status]:
            raise ValueError("Signal value/status and availability disagree")
        if self.provenance.frozen_at > self.request.information_cutoff:
            raise ValueError("Signal model was not frozen by its information cutoff")
        if self.maximum_event_age is not None and self.maximum_event_age.semantics != EVENT_AGE_SECONDS:
            raise ValueError("Incompatible source-event age units or meaning")
        return self

    @property
    def logical_id(self) -> str:
        """Hash stable decision/policy dimensions, excluding values and runtime retry metadata."""
        return fingerprint(
            {
                "decision_id": self.request.decision_id,
                "entity_id": self.request.entity_id,
                "measurement": self.measurement.name,
                "interpretation": self.interpretation,
                "source_id": self.provenance.source_id,
                "source_version": self.provenance.source_version,
                "producer": self.provenance.producer,
                "model_version": self.provenance.model_version,
                "maximum_event_age": (
                    self.maximum_event_age.model_dump(mode="json") if self.maximum_event_age is not None else None
                ),
            }
        )

    @property
    def specification_fingerprint(self) -> str:
        """Bind resolved cutoff/horizon, frozen model, freshness and measure definition, excluding values."""
        return fingerprint(
            {
                "request": self.request.model_dump(mode="json"),
                "producer": self.provenance.producer,
                "producer_version": self.provenance.producer_version,
                "model_version": self.provenance.model_version,
                "model_fingerprint": self.provenance.model_fingerprint,
                "interpretation": self.interpretation,
                "measurement": self.measurement.model_dump(exclude={"value", "status"}),
                "maximum_event_age": (
                    self.maximum_event_age.model_dump(mode="json") if self.maximum_event_age is not None else None
                ),
            }
        )

    def to_domain_record(self, *, run_id: str, execution_id: str, evaluation_fingerprint: str) -> DomainRecord:
        """Project a prediction into existing named domain sidecars, without a learning reward.

        Availability is the declared decision instant, not ingestion or realized
        outcome time. The complete signal artifact retains cutoff/horizon/model
        evidence; its measurement explicitly names an estimate/score. Callers own
        a stable logical execution identity and the shared comparison protocol.
        Legacy native runtime IDs may contain attempt suffixes; callers must not
        use those as retry-stable domain identities.
        """
        signal = DecisionSignal.model_validate(self)
        provenance = signal.provenance
        return DomainRecord(
            schema_version=1,
            namespace="decision.signals",
            run_id=run_id,
            execution_id=execution_id,
            decision_id=signal.logical_id,
            data_fingerprint=provenance.source_fingerprint,
            available_at=signal.request.decided_at,
            provenance=ResultProvenance(
                producer=provenance.producer,
                producer_version=provenance.producer_version,
                spec_fingerprint=signal.specification_fingerprint,
                evaluation_fingerprint=evaluation_fingerprint,
                source_revision=provenance.source_version,
            ),
            measurements=(signal.measurement,),
        )
