"""Pure complete-panel evidence and projection to paired independent units."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Any, Literal, Self, cast

from pydantic import Field, field_serializer, model_validator

from coleman.core.numeric import Scalar
from coleman.core.temporal import Timestamp
from coleman.evaluation._repetition import complete_mean
from coleman.evaluation.design._wire import DesignContract, wire_size
from coleman.evaluation.design.aggregates import AggregateProjection
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.contracts import ExperimentDesign, MetricPlan
from coleman.evaluation.design.projection import project_frozen
from coleman.evaluation.design.selection import AttemptLedger, ConfigurationTrial, PhysicalAttempt, SelectionManifest
from coleman.evaluation.design.source_contracts import SourceMember
from coleman.evaluation.protected.contracts import FrozenEvaluation
from coleman.evaluation.protected.observations import PairedObservation
from coleman.results.domain import DomainRecord


def _require_measurement(record: DomainRecord, metric: MetricPlan) -> None:
    """Check a declared raw quantity without converting units or replacing data."""
    value = record.measurement(metric.name)
    semantics = metric.quantity.spec.semantics
    if (value.unit, value.meaning, value.direction) != (semantics.unit, semantics.meaning, metric.direction):
        raise ValueError("Raw measurement differs from its declared unit, meaning or direction")
    Scalar(spec=metric.quantity.spec, value=value.value, status=value.status)


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
        members = {member.member_id: member for member in self.design.protected_members}
        expected = {
            (candidate.candidate.candidate_id, member, seed)
            for candidate in self.design.candidates
            for member in members
            for seed in self.design.seeds
        }
        seen: set[tuple[str, str, int]] = set()
        records: set[tuple[str, str, str, str]] = set()
        configurations = {
            item.candidate.candidate_id: item.candidate.configuration_fingerprint for item in self.design.candidates
        }
        frozen_digest = frozen.fingerprint
        for row in self.rows:
            key = row.candidate_id, row.member_id, row.seed
            if key not in expected or key in seen:
                raise ValueError("Raw panel contains an unexpected or duplicate repetition cell")
            if row.record.logical_key in records:
                raise ValueError("One original raw record cannot supply several required cells")
            seen.add(key)
            records.add(row.record.logical_key)
            self._require_row(row, members[row.member_id], configurations[row.candidate_id], frozen_digest)
        if seen != expected:
            raise ValueError("Every required panel cell needs explicit observed or unavailable evidence")
        self._require_chronology()
        self._require_attempts(seen)
        self._require_disjoint_work()
        self._require_wire(frozen)
        object.__setattr__(
            self, "rows", tuple(sorted(self.rows, key=lambda item: (item.candidate_id, item.member_id, item.seed)))
        )
        return self

    def _require_row(self, row: RawCell, member: SourceMember, configuration: str, frozen_digest: str) -> None:
        """Bind original data/configuration, source clocks and declared quantities."""
        record = row.record
        if record.data_fingerprint != self.design.snapshot.data_fingerprint:
            raise ValueError("Raw result source snapshot differs from its fixed design")
        if record.provenance.spec_fingerprint != configuration:
            raise ValueError("Raw result specification differs from its BASE candidate configuration")
        if self.mode == "produced" and record.provenance.evaluation_fingerprint != frozen_digest:
            raise ValueError("Produced raw result does not bind the actual projected evaluation")
        if not member.metadata.available_at <= record.available_at <= self.assembled_at:
            raise ValueError("Raw availability must follow original source labels and precede assembly")
        if self.mode == "produced" and record.available_at < self.selection.actual_selected_at:
            raise ValueError("Produced raw outcome predates its actual selected configuration")
        primary = self.design.metric(self.design.primary_metric)
        _require_measurement(record, primary)
        present = {item.name for item in record.measurements}
        for metric in self.design.metrics:
            if metric.name != primary.name and metric.name in present:
                _require_measurement(record, metric)
        if row.state != "completed" and record.measurement(primary.name).status != "unavailable":
            raise ValueError("Incomplete cell execution requires an explicit unavailable primary outcome")
        if row.evidence.record_bytes + 1 > self.design.limits.max_raw_record_bytes:
            raise ValueError("Original raw record exceeds its declared complete-wire capacity")
        if row.header_bytes > self.design.limits.max_raw_header_bytes:
            raise ValueError("Actual encoded raw-cell header exceeds its declared capacity")

    def _require_chronology(self) -> None:
        """Keep historical eligibility separate from actual common assembly."""
        if not self.selection.actual_selected_at <= self.assembled_at <= self.design.comparison_plan.test_end:
            raise ValueError("Actual panel assembly lies outside selection and the frozen analysis deadline")
        if any(
            member.metadata.target_end is not None and member.metadata.target_end > self.assembled_at
            for member in self.design.protected_members
        ):
            raise ValueError("Panel assembly predates a required source target interval")
        captured = self.design.snapshot.captured_at
        if captured is not None and captured > self.assembled_at:
            raise ValueError("Panel assembly predates actual selected-source capture/import")

    def _require_attempts(self, cells: set[tuple[str, str, int]]) -> None:
        """Check real grouped-work declarations without assuming one job per cell."""
        if self.ledger.stage != "protected":
            raise ValueError("Raw panel requires its separate protected physical-attempt ledger")
        budget = self.design.budget
        development = self.selection.ledger
        if len(development.attempts) + len(self.ledger.attempts) > budget.maximum_physical_attempts:
            raise ValueError("Complete development/protected work exceeds its declared attempt budget")
        if development.output_references + self.ledger.output_references > budget.maximum_output_references:
            raise ValueError("Complete development/protected output references exceed their declared budget")
        known = {item.candidate.candidate_id for item in self.design.candidates}
        trials = {trial.trial_id: trial for trial in self.selection.ledger.trials}
        groups: dict[str, str] = {}
        covered: set[tuple[str, str, int]] = set()
        for attempt in self.ledger.attempts:
            if attempt.candidate_id not in known:
                raise ValueError("Protected attempt references an unfrozen candidate")
            if groups.setdefault(attempt.group_id, attempt.candidate_id) != attempt.candidate_id:
                raise ValueError("One physical work group cannot change its candidate between retries")
            self._require_attempt_trial(attempt, trials)
            outputs = {(item.candidate_id, item.member_id, item.seed) for item in attempt.outputs}
            if not outputs <= cells:
                raise ValueError("Protected physical group claims an undeclared logical output")
            covered.update(outputs)
            self._require_attempt_time(attempt)
        if self.mode == "produced" and any(
            row.state != "skipped" and (row.candidate_id, row.member_id, row.seed) not in covered for row in self.rows
        ):
            raise ValueError("Produced non-skipped cells require supplied protected physical-group references")

    def _require_attempt_trial(self, attempt: PhysicalAttempt, trials: dict[str, ConfigurationTrial]) -> None:
        """Check retained trial identity after the attempt's candidate/group checks."""
        if attempt.trial_id is not None and (
            attempt.trial_id not in trials
            or trials[attempt.trial_id].candidate != self.design.candidate(attempt.candidate_id)
        ):
            raise ValueError("Protected attempt does not match its retained development trial")

    def _require_attempt_time(self, attempt: PhysicalAttempt) -> None:
        """Reject contradictory supplied actual times without inventing unknowns."""
        moments = (attempt.started_at, attempt.completed_at, *(cost.record.available_at for cost in attempt.costs))
        if any(moment is not None and moment > self.assembled_at for moment in moments):
            raise ValueError("Panel assembly predates a supplied actual physical attempt time")
        if self.mode == "produced" and (
            (attempt.started_at is not None and attempt.started_at < self.selection.actual_selected_at)
            or any(cost.record.available_at < self.selection.actual_selected_at for cost in attempt.costs)
        ):
            raise ValueError("Protected physical work predates actual selected configuration")

    def _require_disjoint_work(self) -> None:
        """Keep stage attribution unique without prohibiting legitimate grouped retries."""
        development = self.selection.ledger.attempts
        protected = self.ledger.attempts
        if {item.attempt_id for item in development} & {item.attempt_id for item in protected}:
            raise ValueError("One physical attempt cannot be attributed to both development and protected work")
        known_costs = {cost.record_key for item in development for cost in item.costs}
        if any(cost.record_key in known_costs for item in protected for cost in item.costs):
            raise ValueError("One physical cost cannot be attributed to both development and protected work")

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
