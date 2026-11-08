"""Shared pure original-cell validation; protected projection stays with RawPanel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from coleman.core.numeric import Scalar
from coleman.core.temporal import Timestamp
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.contracts import ExperimentDesign, MetricPlan
from coleman.evaluation.design.selection import AttemptLedger, ConfigurationTrial, PhysicalAttempt, SelectionManifest
from coleman.evaluation.design.source_contracts import SourceMember
from coleman.results.domain import DomainRecord


def _require_measurement(record: DomainRecord, metric: MetricPlan) -> None:
    """Check a declared raw quantity without converting units or replacing data."""
    value = record.measurement(metric.name)
    semantics = metric.quantity.spec.semantics
    if (value.unit, value.meaning, value.direction) != (semantics.unit, semantics.meaning, metric.direction):
        raise ValueError("Raw measurement differs from its declared unit, meaning or direction")
    Scalar(spec=metric.quantity.spec, value=value.value, status=value.status)


@dataclass(frozen=True)
class RawEvidence:
    """Borrow complete typed evidence without projection, I/O or mutable ownership.

    Attributes
    ----------
    design, selection : ExperimentDesign, SelectionManifest
        Existing complete declarations; callers validate their exact binding first.
    rows, ledger : tuple of RawCell, AttemptLedger
        Original bounded cell grid and separate supplied actual physical work.
    mode : {'produced', 'imported'}
        Explicit provenance/chronology profile; import never invents work.
    assembled_at : datetime
        Supplied common assembly after every known consumed outcome/work time.
    """

    design: ExperimentDesign
    selection: SelectionManifest
    rows: tuple[RawCell, ...]
    ledger: AttemptLedger
    mode: Literal["produced", "imported"]
    assembled_at: Timestamp

    def require_rows(self, frozen_digest: str | None) -> tuple[RawCell, ...]:
        """Validate original evidence under an explicit projected or descriptive binding.

        Parameters
        ----------
        frozen_digest : str or None
            Actual supported projected identity for RawPanel. Explicit null keeps
            original evaluation provenance descriptive, never a protected freeze.

        Returns
        -------
        tuple of RawCell
            Canonical complete grid with original records and omissions intact.

        Raises
        ------
        ValueError
            A grid, source, quantity, chronology or physical-work join fails.
        """
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
        return tuple(sorted(self.rows, key=lambda item: (item.candidate_id, item.member_id, item.seed)))

    def _require_row(self, row: RawCell, member: SourceMember, configuration: str, frozen_digest: str | None) -> None:
        """Bind original data/configuration, source clocks and declared quantities."""
        record = row.record
        if record.data_fingerprint != self.design.snapshot.data_fingerprint:
            raise ValueError("Raw result source snapshot differs from its fixed design")
        if record.provenance.spec_fingerprint != configuration:
            raise ValueError("Raw result specification differs from its BASE candidate configuration")
        if (
            self.mode == "produced"
            and frozen_digest is not None
            and record.provenance.evaluation_fingerprint != frozen_digest
        ):
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
        """Bound every known work/cost instant without imputing unknown clocks."""
        moments = (attempt.started_at, attempt.completed_at, *(cost.record.available_at for cost in attempt.costs))
        if any(moment is not None and moment > self.assembled_at for moment in moments):
            raise ValueError("Panel assembly predates a supplied actual physical attempt time")
        if self.mode == "produced" and any(
            moment is not None and moment < self.selection.actual_selected_at for moment in moments
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
