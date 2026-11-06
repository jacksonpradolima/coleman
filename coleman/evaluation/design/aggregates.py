"""Pure aggregate construction and conservative pre-outcome output capacity checks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from coleman.core.numeric import Scalar
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.splits import TemporalObservation
from coleman.core.temporal import Timestamp
from coleman.evaluation.design._wire import wire_size
from coleman.evaluation.design.contracts import ExperimentDesign
from coleman.evaluation.design.projection import aggregate_meaning, project_frozen
from coleman.evaluation.design.selection import SelectionManifest
from coleman.evaluation.design.source_contracts import SourceMember
from coleman.evaluation.protected.contracts import FrozenEvaluation
from coleman.evaluation.protected.observations import ObservationReference, PairedObservation
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance


@dataclass(frozen=True)
class AggregateProjection:
    """Fixed pure output context shared by admission and actual aggregation.

    Parameters
    ----------
    design : ExperimentDesign
        Complete validated quantity, units, member and capacity declarations.
    frozen : FrozenEvaluation
        Actual projection previously derived from this design and selection.
    parent_fingerprint : str
        Complete immutable raw-panel fingerprint; preflight uses an equal-width
        placeholder without claiming that unobserved evidence already exists.
    available_at : datetime
        Supplied common actual assembly; preflight uses the fixed deadline only
        to size its wire representation and does not capture a clock.
    """

    design: ExperimentDesign
    frozen: FrozenEvaluation
    parent_fingerprint: str
    available_at: Timestamp
    _design_fingerprint: str = field(init=False, repr=False)
    _frozen_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute immutable common digests once, without external side effects."""
        object.__setattr__(self, "_design_fingerprint", self.design.fingerprint)
        object.__setattr__(self, "_frozen_fingerprint", self.frozen.fingerprint)

    def paired(
        self,
        key: tuple[str, str],
        value: float | None,
        members: tuple[SourceMember, ...],
    ) -> PairedObservation:
        """Construct a new record and enforce its actual body/header wire limits.

        Parameters
        ----------
        key : tuple of str
            Candidate and declared independent-unit identifiers.
        value : float or None
            Already computed finite binary64 mean, or explicit unavailable mean.
        members : tuple of SourceMember
            Every original member in this unit, used for truthful event clocks.

        Returns
        -------
        PairedObservation
            New aggregate with compact whole-panel provenance; input originals
            are neither rewritten nor mutated.

        Raises
        ------
        ValueError
            Incompatible contract or encoded record/header exceeds its limit.
        """
        candidate, unit = key
        frozen, context = self.frozen, self.parent_fingerprint
        primary = self.design.metric(self.design.primary_metric)
        Scalar(spec=primary.quantity.spec, value=value, status="observed" if value is not None else "unavailable")
        identity = fingerprint([self._design_fingerprint, unit])
        record = DomainRecord(
            schema_version=1,
            namespace="evaluation.protocol.unit_means",
            run_id=self._frozen_fingerprint,
            execution_id=candidate,
            decision_id=identity,
            data_fingerprint=self.design.snapshot.data_fingerprint,
            available_at=self.available_at,
            provenance=ResultProvenance(
                producer="coleman.evaluation.equal-seed-member-mean",
                producer_version="1",
                spec_fingerprint=frozen.candidate(candidate).configuration_fingerprint,
                evaluation_fingerprint=self._frozen_fingerprint,
                source_revision=context,
            ),
            measurements=(
                Measurement(
                    name=primary.name,
                    value=value,
                    unit=primary.quantity.spec.semantics.unit,
                    meaning=aggregate_meaning(self.design),
                    direction=primary.direction,
                    status="observed" if value is not None else "unavailable",
                    minimum=primary.quantity.spec.minimum,
                ),
            ),
        )
        event = max(member.metadata.event_at for member in members)
        targets = [member.metadata.target_end for member in members if member.metadata.target_end is not None]
        metadata = TemporalObservation(
            observation_id=identity,
            group_id=unit,
            event_at=event,
            available_at=self.available_at,
            target_end=max(event, *targets) if targets else None,
        )
        reference = ObservationReference(
            candidate_id=candidate,
            metadata=metadata,
            record_key=record.logical_key,
            record_digest=record.digest,
            record=record,
        )
        if len(record.to_json().encode("utf-8")) + 1 > self.design.limits.max_aggregate_record_bytes:
            raise ValueError("Actual derived record exceeds its declared complete-wire capacity")
        if wire_size(reference.model_dump(mode="json", exclude={"record"})) > self.design.limits.max_input_header_bytes:
            raise ValueError("Actual derived-input header exceeds its declared capacity")
        return PairedObservation(candidate_id=candidate, metadata=metadata, record=record)


def require_projection_capacity(design: ExperimentDesign, selection: SelectionManifest) -> None:
    """Reject infeasible known output bodies/headers before protected loading.

    Parameters
    ----------
    design : ExperimentDesign
        Valid finite counts, exact units, member identities and wire allowances.
    selection : SelectionManifest
        Complete final selection bound to this design.

    Raises
    ------
    ValueError
        Unsupported projection or its fixed output plus conservative variable
        spelling exceeds the declared body/header allowances.

    Notes
    -----
    This pure preflight constructs no observation evidence. It sizes the same
    constructor used by actual aggregation with equal-width digest placeholders,
    then reserves a 24-byte finite binary64 spelling and up to seven additional
    fractional-second bytes. A declared bound supplies the sizing value, or zero
    when both bounds are absent. The unavailable spelling fits that reservation.
    Source events, target clocks, identifiers, units and meaning are fixed and
    counted exactly, including UTF-8 and JSON escaping. Actual producer bodies,
    complete panel metadata and final report bytes still require runtime checks.
    No outcome, callback, clock, filesystem or optional analytics is accessed.
    """
    frozen = project_frozen(design, selection)
    projection = AggregateProjection(design, frozen, "sha256:" + "0" * 64, design.comparison_plan.test_end)
    units: dict[str, list[SourceMember]] = defaultdict(list)
    for member in design.protected_members:
        if member.unit_id is not None:
            units[member.unit_id].append(member)
    quantity = design.metric(design.primary_metric).quantity.spec
    placeholder = quantity.minimum if quantity.minimum is not None else quantity.maximum
    if placeholder is None:
        placeholder = 0.0
    scalar_padding = 24 - len(canonical_json(placeholder))
    for candidate in design.candidates:
        for unit, members in units.items():
            row = projection.paired((candidate.candidate.candidate_id, unit), placeholder, tuple(members))
            _require_spelling_capacity(projection, row, scalar_padding)


def _require_spelling_capacity(context: AggregateProjection, row: PairedObservation, scalar_padding: int) -> None:
    """Reserve variable number/time spellings beyond the exact fixed skeleton."""
    time_padding = 7 if context.available_at.microsecond == 0 else 0
    record_bytes = len(row.record.to_json().encode("utf-8")) + 1
    if record_bytes + scalar_padding + time_padding > context.design.limits.max_aggregate_record_bytes:
        raise ValueError("Declared aggregate capacity cannot contain every supported output spelling")
    reference = ObservationReference(
        candidate_id=row.candidate_id,
        metadata=row.metadata,
        record_key=row.record.logical_key,
        record_digest=row.record.digest,
        record=row.record,
    )
    header_bytes = wire_size(reference.model_dump(mode="json", exclude={"record"}))
    if header_bytes + time_padding > context.design.limits.max_input_header_bytes:
        raise ValueError("Declared input-header capacity cannot contain every supported output spelling")
