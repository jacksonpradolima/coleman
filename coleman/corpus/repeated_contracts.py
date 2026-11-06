"""Predeclared corpus repetitions; source clocks and independent units stay distinct."""

from __future__ import annotations

import json
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.evaluation._repetition import repetition_execution_id
from coleman.evaluation.protected import FrozenEvaluation, TemporalObservation, TemporalPlan
from coleman.evaluation.protected.temporal import Timestamp
from coleman.results.domain import DomainRecord, Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, Quantity, UnitSemantics

Seed = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
MEAN_SUFFIX = ":equal-build-seed-mean-binary64-v1"
BUDGET_RATIO = UnitSemantics(unit="1", meaning="fraction-of-full-suite-native-work-v1")


class CorpusBuild(Immutable):
    """One preselected logical build, retaining its source revision and source clocks.

    ``member_id`` identifies a full logical build, including its reviewed handling
    of sibling jobs/variants/retries. Such copies must not be separate members.
    ``unit_id`` names the reviewed independent repository/family; distinct strings
    alone cannot establish independence. Metadata groups identify source build
    leakage groups, separately from inference units. No identity is inferred from
    a test name, worker, seed, variant or job finish label.
    """

    member_id: Identifier
    project_id: Identifier
    unit_id: Identifier
    metadata: TemporalObservation


def _require_membership(builds: tuple[CorpusBuild, ...]) -> None:
    """Reject duplicate builds/revisions/siblings and projects split into false independent units."""
    ids = [build.member_id for build in builds]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate logical corpus member")
    observations = [build.metadata.observation_id for build in builds]
    if len(observations) != len(set(observations)):
        raise ValueError("Choose one point-in-time revision per logical source build")
    groups: dict[str, str] = {}
    projects: dict[str, str] = {}
    for build in builds:
        previous_unit = projects.setdefault(build.project_id, build.unit_id)
        if previous_unit != build.unit_id:
            raise ValueError("One project cannot create multiple independent units")
        previous = groups.setdefault(build.metadata.group_id, build.member_id)
        if previous != build.member_id:
            raise ValueError("Sibling jobs/variants share one logical corpus member")


class RepetitionDesign(VersionedContract):
    """Freeze one fold, one native budget and a complete repeated-measures grid.

    Equal seed means within each build, then equal build means within each unit
    define the estimand. Reviewed families receive one paired observation each.
    All seeds and builds must be retained; unavailable measurements invalidate the whole
    unit mean rather than silently selecting complete cases. Warmup ends no later
    than training; train-only preprocessing and point-in-time revision selection
    remain producer obligations. No fitter, fold tuner or normality detector runs.
    V1 accepts only explicit version 1; there is no migration from native rows.
    """

    fold_id: Identifier
    data_fingerprint: Fingerprint
    evidence_basis: Literal["constructed_mechanism", "empirical_protected"]
    source_admission_fingerprint: Fingerprint | None
    plan: TemporalPlan
    warmup_end: Timestamp
    budget: Quantity
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1, max_length=64)]
    builds: Annotated[tuple[CorpusBuild, ...], Field(min_length=1, max_length=4096)]
    raw_metric: Identifier
    raw_semantics: UnitSemantics
    independent_unit: Literal["reviewed_repository_family"] = "reviewed_repository_family"
    cost_projection: Literal["exact_integer_seconds_v1", "binary64_seconds_v1", "binary32_carried_binary64_seconds_v1"]
    aggregation: Literal["equal_seed_then_equal_build_v1"] = "equal_seed_then_equal_build_v1"
    preprocessing: Literal["identity_or_fit_on_train_only_v1"] = "identity_or_fit_on_train_only_v1"
    revision_selection: Literal["predeclared_point_in_time_revision_v1"] = "predeclared_point_in_time_revision_v1"

    @model_validator(mode="after")
    def _grid(self) -> Self:
        """Reject ambiguous membership, leakage, invalid units and unbounded cross products."""
        if self.warmup_end > self.plan.train_end:
            raise ValueError("Warmup must end within training")
        if self.evidence_basis == "empirical_protected" and self.source_admission_fingerprint is None:
            raise ValueError("Empirical plans must reference their independently reviewed source admission")
        if self.budget.semantics != BUDGET_RATIO or not 0 < self.budget.value <= 1:
            raise ValueError("Corpus design requires an explicit native-work budget ratio in (0, 1]")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Duplicate repetition seed")
        if len(self.seeds) * len(self.builds) > 16384:
            raise ValueError("Repetition design exceeds 16,384 build/seed cells per candidate")
        _require_membership(self.builds)
        decisions = self.plan.split(tuple(build.metadata for build in self.builds))
        if any(not decision.eligible or decision.partition != "test" for decision in decisions):
            raise ValueError("Corpus members require eligible protected-test source metadata")
        object.__setattr__(self, "seeds", tuple(sorted(self.seeds)))
        object.__setattr__(self, "builds", tuple(sorted(self.builds, key=lambda item: item.member_id)))
        return self

    def require_frozen(self, frozen: FrozenEvaluation) -> None:
        """Require the whole design in every frozen configuration, before outcome loading.

        Fingerprints record declarations, not proof that sources are independent,
        that timing is authentic or that an external fitter respected the holdout.
        Legacy frozen evaluations remain readable but cannot be silently rebound.
        """
        if frozen.plan != self.plan or frozen.data_fingerprint != self.data_fingerprint:
            raise ValueError("Frozen evaluation differs from corpus data or temporal plan")
        protocol = frozen.protocol
        if (
            protocol.metric_name != self.raw_metric
            or protocol.unit != self.raw_semantics.unit
            or protocol.meaning != self.raw_semantics.meaning + MEAN_SUFFIX
            or protocol.independent_unit != self.independent_unit
        ):
            raise ValueError("Frozen protocol must declare the unit-mean metric semantics")
        if len(frozen.candidates) * len(self.seeds) * len(self.builds) > 65536:
            raise ValueError("Corpus comparison exceeds 65,536 raw cells")
        design_digest = self.fingerprint
        for candidate in frozen.candidates:
            if json.loads(candidate.configuration_json).get("corpus_repetition_design") != design_digest:
                raise ValueError("Every candidate must freeze the full corpus repetition design fingerprint")


class CorpusRepetition(Immutable):
    """One candidate/build/seed cell with its original immutable result record.

    Failed, aborted and skipped attempts require a null unavailable measurement,
    never zero or a successful completion. A completed test run may contain failed
    tests and negative metrics: ``completed`` describes execution, not its verdict.
    Native rows require an explicit producer adapter with reviewed metric meaning.
    Its immutable decision_id equals member_id and its execution_id equals
    repetition_execution_id(candidate_id, seed). Other original identities require
    a new explicit projection retaining original source provenance; they are not
    silently relabeled. Swapping unchanged records across cells raises ValueError.
    """

    candidate_id: Identifier
    member_id: Identifier
    seed: Seed
    state: Literal["completed", "failed", "aborted", "skipped"]
    record: DomainRecord

    @model_validator(mode="after")
    def _cell_identity(self) -> Self:
        """Bind declared candidate/member/seed membership to the immutable raw result key."""
        if self.record.decision_id != self.member_id or self.record.execution_id != repetition_execution_id(
            self.candidate_id, self.seed
        ):
            raise ValueError("Raw result identity does not bind the declared repetition cell")
        return self
