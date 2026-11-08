"""Pure complete-panel evidence and projection to paired independent units."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Any, Literal, Self, cast

from pydantic import Field, field_serializer, model_validator

from coleman.core.temporal import Timestamp
from coleman.evaluation._repetition import complete_mean
from coleman.evaluation.design._raw_evidence import RawEvidence
from coleman.evaluation.design._wire import DesignContract, wire_size
from coleman.evaluation.design.aggregates import AggregateProjection
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.contracts import ExperimentDesign
from coleman.evaluation.design.projection import project_frozen
from coleman.evaluation.design.selection import AttemptLedger, SelectionManifest
from coleman.evaluation.protected.contracts import FrozenEvaluation
from coleman.evaluation.protected.observations import PairedObservation


class RawPanel(DesignContract):
    """Retain every required raw cell before constructing any independent mean.

    Attributes
    ----------
    design, selection : ExperimentDesign, SelectionManifest
        Exact frozen source/family/law and development-only selection. Actual
        source reservation and persistence order belong to the optional adapter.
    rows : tuple of RawCell
        Complete candidate/protected-member/seed grid, including unavailable
        outcomes. Original logical keys/digests and omission semantics survive.
    ledger : AttemptLedger
        Separate protected physical groups and measured costs. One group may
        emit many cells; retries/failures never add independent units. Physical
        attempt IDs and cost logical keys cannot overlap development work;
        distinct attempts may retain a common consistent work-group identity.
    mode : {'produced', 'imported'}
        Produced records bind the current projected evaluation; imported ones
        retain historical evaluation identity. The adapter must declare import
        exposure before loading and cannot grant fresh confirmation to imports.
        Every known produced start, completion and cost-availability time must
        be at or after actual selection; an unknown start never waives a known finish.
        The optional adapter also enforces its actual reservation lower bound.
    assembled_at : datetime
        Supplied actual common assembly, after consumed raw availability and
        known physical completion, within the frozen comparison deadline.

    Raises
    ------
    ValueError
        Incomplete/duplicate/extra grid, conflicting raw identity or provenance,
        invalid measurement/chronology/physical grouping, or exceeded capacity.

    Notes
    -----
    This is pure evidence validation, not a runner, importer, durable artifact,
    source admission or proof of physical work. Every consumed value keeps its
    units, horizon binding and status. Missingness invalidates the whole required
    panel estimate; no cell is dropped or imputed. Old corpus/readers stay intact.
    """

    design: ExperimentDesign
    selection: SelectionManifest
    rows: Annotated[tuple[RawCell, ...], Field(max_length=8192)]
    ledger: AttemptLedger
    mode: Literal["produced", "imported"]
    assembled_at: Timestamp

    @field_serializer("rows")
    def _original_cells(self, rows: tuple[RawCell, ...]) -> list[dict[str, Any]]:
        """Preserve supplied reasons and each original record's unset fields."""
        return [row.model_dump(mode="json", exclude_unset=True) for row in rows]

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Require the complete bounded grid before any mean or claim exists."""
        self.selection.require_design(self.design)
        frozen = self.frozen
        evidence = RawEvidence(self.design, self.selection, self.rows, self.ledger, self.mode, self.assembled_at)
        rows = evidence.require_rows(frozen.fingerprint)
        self._require_wire(frozen)
        object.__setattr__(self, "rows", rows)
        return self

    def _require_wire(self, frozen: FrozenEvaluation) -> None:
        """Admit actual repeated metadata and complete-envelope bytes explicitly."""
        limits = self.design.limits
        if self.metadata_bytes + wire_size(frozen) + 1 > limits.max_metadata_bytes:
            raise ValueError("Actual repeated panel/freeze metadata exceeds its declared allowance")
        if wire_size(self) + 1 > limits.max_envelope_bytes:
            raise ValueError("Actual complete raw panel exceeds its declared envelope capacity")

    @property
    def frozen(self) -> FrozenEvaluation:
        """Return the exact deterministic supported projection, without persistence."""
        return project_frozen(self.design, self.selection)

    @property
    def metadata_bytes(self) -> int:
        """Count every actual non-row metadata occurrence, including grouped costs."""
        return wire_size(self.model_dump(mode="json", exclude={"rows"}))

    def paired_observations(self) -> tuple[PairedObservation, ...]:
        """Project complete equal-seed/member means into the existing paired API.

        Returns
        -------
        tuple of PairedObservation
            One NEW record per candidate/declared unit, with the primary target's
            units/horizon binding, truthful common assembly and the full panel's
            fingerprint as compact provenance. Any nonobserved required cell
            makes its unit mean unavailable; all raw statuses stay in this panel.

        Raises
        ------
        ValueError
            Revalidation, output/header capacity or existing paired-contract
            invariants fail. No partial result is returned.

        Notes
        -----
        Uses the unchanged rational corpus arithmetic on binary64 inputs, not a
        new estimator. No inference, source access, actual cost aggregation or
        I/O occurs. Physical attempts/costs and learner surrogates do not become
        extra units or replace the explicitly selected primary target.
        """
        panel = RawPanel.model_validate(self)
        members = {member.member_id: member for member in panel.design.protected_members}
        grouped: dict[tuple[str, str], dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
        for row in panel.rows:
            unit = cast(str, members[row.member_id].unit_id)
            grouped[row.candidate_id, unit][row.member_id].append(
                row.record.measurement(panel.design.primary_metric).value
            )
        projection = AggregateProjection(panel.design, panel.frozen, panel.fingerprint, panel.assembled_at)
        result = []
        for (candidate, unit), member_values in sorted(grouped.items()):
            mean = complete_mean(tuple(tuple(values) for _, values in sorted(member_values.items())))
            unit_members = tuple(members[member] for member in member_values)
            result.append(projection.paired((candidate, unit), mean, unit_members))
        return tuple(result)
