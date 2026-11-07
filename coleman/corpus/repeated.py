"""Pure complete-grid corpus aggregation into the existing protected paired API."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.corpus.repeated_contracts import MEAN_SUFFIX, CorpusRepetition, RepetitionDesign
from coleman.evaluation._repetition import complete_mean as _complete_mean
from coleman.evaluation.protected import FrozenEvaluation, PairedObservation, TemporalObservation
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, fingerprint
from coleman.results.versioned import VersionedContract


class RepeatedCorpusReport(VersionedContract):
    """Embedded raw evidence for one predeclared fold/budget with explicit missingness.

    Validation checks the full candidate/build/seed grid, original logical keys,
    frozen provenance, metric semantics and label/assembly clocks. It performs no
    inference, source access, policy execution or I/O. Reading validates the same
    invariants as construction; only explicit schema version 1 is accepted.
    Distinct unit labels remain reviewed scientific assumptions, not a certificate
    of independent repositories. Source admission is referenced, never granted.
    """

    design: RepetitionDesign
    frozen: FrozenEvaluation
    rows: Annotated[tuple[CorpusRepetition, ...], Field(min_length=1, max_length=65536)]
    assembled_at: Timestamp

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Reject missing/extra/reused cells and future, incompatible or mislabeled results."""
        self.design.require_frozen(self.frozen)
        builds = {build.member_id: build for build in self.design.builds}
        candidates = {
            candidate.candidate_id: candidate.configuration_fingerprint for candidate in self.frozen.candidates
        }
        frozen_digest = self.frozen.fingerprint
        expected = {
            (candidate, member, seed) for candidate in candidates for member in builds for seed in self.design.seeds
        }
        seen = set()
        record_keys = set()
        for row in self.rows:
            key = row.candidate_id, row.member_id, row.seed
            if key not in expected or key in seen:
                raise ValueError("Unexpected or duplicate corpus repetition cell")
            seen.add(key)
            if row.record.logical_key in record_keys:
                raise ValueError("A raw result identity cannot be reused across corpus cells")
            record_keys.add(row.record.logical_key)
            self._check_row(row, builds[row.member_id].metadata, candidates[row.candidate_id], frozen_digest)
        if seen != expected:
            raise ValueError("Every expected corpus cell requires explicit completed or unavailable evidence")
        if self.assembled_at > self.design.plan.test_end:
            raise ValueError("Unit means are unavailable before assembly; freeze a cutoff including assembly")
        object.__setattr__(
            self, "rows", tuple(sorted(self.rows, key=lambda row: (row.candidate_id, row.member_id, row.seed)))
        )
        return self

    def _check_row(
        self, row: CorpusRepetition, metadata: TemporalObservation, configuration: str, frozen_digest: str
    ) -> None:
        """Validate raw binary64 observations without changing their source quantities or semantics."""
        record = row.record
        if (
            record.data_fingerprint != self.frozen.data_fingerprint
            or record.provenance.evaluation_fingerprint != frozen_digest
            or record.provenance.spec_fingerprint != configuration
        ):
            raise ValueError("Corpus raw result differs from frozen data/configuration/evaluation")
        if not metadata.available_at <= record.available_at <= self.assembled_at:
            raise ValueError("Raw result availability must follow source labels and precede mean assembly")
        if metadata.target_end is not None and metadata.target_end > self.assembled_at:
            raise ValueError("Unit mean assembly cannot precede the source target interval")
        metric = record.measurement(self.design.raw_metric)
        if (
            metric.unit != self.design.raw_semantics.unit
            or metric.meaning != self.design.raw_semantics.meaning
            or metric.direction != self.frozen.protocol.direction
        ):
            raise ValueError("Corpus raw metric unit/meaning/direction differs from its declared contract")
        if row.state != "completed" and (metric.status != "unavailable" or metric.value is not None):
            raise ValueError("Incomplete execution requires an explicit unavailable metric")

    def paired_observations(self) -> tuple[PairedObservation, ...]:
        """Produce one equally weighted build/seed mean per candidate and independent unit.

        Any null input makes its whole unit mean unavailable; no complete-case
        selection occurs. Signed raw values and observed zero remain valid.
        Exact rational arithmetic over binary64 inputs avoids sum overflow and
        order dependence; one final binary64 rounding remains explicit. This is
        not an exact measurement of original decimal/physical quantities. Assembly
        time is retained as derived availability, never backdated to label time.
        The existing protected evaluator owns access history, assumptions, paired
        inference, multiplicity, uncertainty and conservative conclusions.
        """
        report = RepeatedCorpusReport.model_validate(self)
        members = {build.member_id: build for build in report.design.builds}
        grouped: dict[tuple[str, str], list[CorpusRepetition]] = defaultdict(list)
        for row in report.rows:
            grouped[row.candidate_id, members[row.member_id].unit_id].append(row)
        result = []
        report_digest = report.fingerprint
        design_digest = report.design.fingerprint
        frozen_digest = report.frozen.fingerprint
        configurations = {item.candidate_id: item.configuration_fingerprint for item in report.frozen.candidates}
        for (candidate, unit), rows in sorted(grouped.items()):
            by_build: dict[str, list[float | None]] = defaultdict(list)
            for row in rows:
                by_build[row.member_id].append(row.record.measurement(report.design.raw_metric).value)
            mean = _complete_mean(tuple(tuple(values) for _, values in sorted(by_build.items())))
            identity = fingerprint([design_digest, unit])
            event_at = max(members[row.member_id].metadata.event_at for row in rows)
            record = DomainRecord(
                schema_version=1,
                namespace="ci.corpus.unit_means",
                run_id=frozen_digest,
                execution_id=candidate,
                decision_id=identity,
                data_fingerprint=report.frozen.data_fingerprint,
                available_at=report.assembled_at,
                provenance=ResultProvenance(
                    producer="coleman.corpus.equal-build-seed-mean",
                    producer_version="1",
                    spec_fingerprint=configurations[candidate],
                    evaluation_fingerprint=frozen_digest,
                    source_revision=report_digest,
                ),
                measurements=(
                    Measurement(
                        name=report.design.raw_metric,
                        unit=report.design.raw_semantics.unit,
                        meaning=report.design.raw_semantics.meaning + MEAN_SUFFIX,
                        direction=report.frozen.protocol.direction,
                        value=mean,
                        status="observed" if mean is not None else "unavailable",
                    ),
                ),
            )
            result.append(
                PairedObservation(
                    candidate_id=candidate,
                    metadata=TemporalObservation(
                        observation_id=identity,
                        group_id=unit,
                        event_at=event_at,
                        available_at=record.available_at,
                    ),
                    record=record,
                )
            )
        return tuple(result)


def publish_repeated_corpus(report: RepeatedCorpusReport, writer: ArtifactWriter) -> Path:
    """Atomically publish validated embedded evidence with existing immutable local artifacts.

    Identical concurrent retries succeed; conflicting content and I/O failures
    propagate. Process interruption cannot expose partial completed JSON. Local
    hard-link/fsync and power-loss limits are those of ArtifactWriter. Persist the
    design in frozen configurations before outcomes and publish this raw evidence
    before saving a derived comparison; publication does not reserve holdout access.
    """
    report = RepeatedCorpusReport.model_validate(report)
    path = writer.run_root / "corpus-repetitions" / (report.fingerprint.removeprefix("sha256:") + ".json")
    writer.write_immutable(path, report.to_json() + "\n")
    return path
