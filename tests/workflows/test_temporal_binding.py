"""Temporal identity ordering must never reorder the native learning/evaluation sequence."""

import itertools
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from coleman.application.workflow_evaluation import EvaluationHandoff
from coleman.application.workflow_exports import EvidenceBundle
from coleman.application.workflow_registry import owned_fixture, snapshot_history
from coleman.application.workflows import WorkflowService
from coleman.corpus.repeated_contracts import BUDGET_RATIO, CorpusBuild, RepetitionDesign
from coleman.evaluation.protected import EvaluationAudit, PairedObservation, TemporalObservation, evaluate
from coleman.results.domain import fingerprint
from coleman.selection.contracts import Quantity, UnitSemantics
from coleman.workflows.contracts import BuildMetadata, DatasetManifest, PlanningRequest, ScheduleBudget, WorkflowLimits
from coleman.workflows.lifecycle import ExecutePlan
from coleman.workflows.planning import make_plan, protected_repetition_design
from tests.workflows.test_execution import wait
from tests.workflows.test_planning import recorded_runtime


def test_build_decisions_preserve_distinct_revisions_in_native_order(tmp_path):
    original = owned_fixture(tmp_path)
    train, validation, test = original.metadata
    first = train.observation.model_copy(update={"observation_id": "z-training", "revision_id": "rev-b"})
    second = first.model_copy(
        update={"revision_id": "rev-a", "available_at": first.available_at + timedelta(minutes=1)}
    )
    metadata = (
        train.model_copy(update={"build_id": 10, "observation": first}),
        train.model_copy(update={"build_id": 20, "observation": second}),
        validation.model_copy(update={"build_id": 30}),
        test.model_copy(update={"build_id": 40}),
    )
    dataset = DatasetManifest.model_validate(
        {**original.model_dump(), "build_ids": (10, 20, 30, 40), "rows": 12, "metadata": metadata}
    )
    decisions = dataset.split_by_build()
    assert tuple(item.observation for item in decisions) == tuple(item.observation for item in metadata)
    assert tuple(item.partition for item in decisions) == ("train", "train", "validation", "test")


@pytest.mark.parametrize("mode,unit,value", [("fixed_time", "s", 0.5), ("subset_size", "count", 1.0)])
def test_protected_design_cannot_relabel_native_budget_units(tmp_path, mode, unit, value):
    dataset = owned_fixture(tmp_path)
    request = PlanningRequest(
        schema_version=1,
        dataset_id=dataset.dataset_id,
        evaluation="protected",
        budget=ScheduleBudget(mode=mode, unit=unit, values=(value,)),
    )
    with pytest.raises(ValueError, match="ratio"):
        protected_repetition_design(request, dataset)


def test_build_decisions_do_not_invent_missing_temporal_metadata(tmp_path):
    original = owned_fixture(tmp_path)
    dataset = DatasetManifest.model_validate({**original.model_dump(), "metadata": (), "temporal_plan": None})
    with pytest.raises(ValueError, match="metadata"):
        dataset.split_by_build()


def test_analysis_window_preserves_source_partitions_and_field_mappings(tmp_path):
    dataset = owned_fixture(tmp_path)
    request = PlanningRequest(schema_version=1, dataset_id=dataset.dataset_id, evaluation="protected")
    source = dataset.temporal_plan
    assert source is not None
    extended = source.model_copy(update={"test_end": source.test_end + timedelta(days=1)})
    design = protected_repetition_design(request, dataset, analysis_plan=extended)
    assert design.plan == extended
    assert tuple(build.member_id for build in design.builds) == ("build-3",)
    changes = (
        {"train_end": source.train_end + timedelta(seconds=1)},
        {"validation_end": source.validation_end + timedelta(seconds=1)},
        {"test_end": source.test_end - timedelta(seconds=1)},
        {"group_field": "another_group"},
    )
    for change in changes:
        altered = source.model_copy(update=change)
        with pytest.raises(ValueError, match="retain source partitions"):
            protected_repetition_design(request, dataset, analysis_plan=altered)


@pytest.mark.parametrize("ambiguity", ["revision", "sibling_group"])
def test_protected_preflight_reuses_the_core_membership_guards(tmp_path, monkeypatch, ambiguity):
    original = repetition_manifest(tmp_path, 2)
    metadata = list(original.metadata)
    first = metadata[2].observation
    other = metadata[3].observation
    if ambiguity == "revision":
        other = first.model_copy(update={"revision_id": "another-revision"})
    else:
        other = other.model_copy(update={"group_id": first.group_id})
    metadata[3] = metadata[3].model_copy(update={"observation": other})
    dataset = DatasetManifest.model_validate({**original.model_dump(), "metadata": tuple(metadata)})
    request = PlanningRequest(schema_version=1, dataset_id=dataset.dataset_id, evaluation="protected")
    runtime = recorded_runtime()

    def forbidden(*args, **kwargs):
        raise RuntimeError("An ambiguous protected membership reached sweep expansion")

    monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
    with pytest.raises(ValueError, match="revision|Sibling"):
        make_plan(request, dataset, runtime)


def test_two_family_native_means_preserve_one_assembly_and_source_events(tmp_path):
    service = WorkflowService(tmp_path / "workspace")
    staging = tmp_path / "source"
    original = owned_fixture(staging)
    frame = pl.read_parquet(staging / "datasets" / f"{original.sha256[7:]}.parquet")
    frame = pl.concat(
        [
            frame,
            frame.filter(pl.col("BuildId") == 3).with_columns(
                pl.lit(4, dtype=frame.schema["BuildId"]).alias("BuildId")
            ),
        ]
    )
    source = staging / "two-families.parquet"
    frame.write_parquet(source)
    third = original.metadata[2]
    fourth = third.model_copy(
        update={
            "build_id": 4,
            "unit_id": "another-authored-family",
            "observation": third.observation.model_copy(
                update={
                    "observation_id": "heldout-four",
                    "group_id": "owned-build-four",
                    "event_at": third.observation.event_at + timedelta(hours=2),
                    "available_at": third.observation.available_at + timedelta(hours=2),
                }
            ),
        }
    )
    dataset = snapshot_history(
        staging,
        source,
        dataset_id="authored-two-family-ci",
        source_fingerprint=fingerprint(["authored-two-family-ci", frame.to_dicts()]),
        projection_fingerprint=fingerprint(["authored-native-seconds-two-families"]),
        evidence="constructed_mechanism",
        assumptions=("Two authored families, not empirical source evidence.",),
        metadata=(*original.metadata, fourth),
        temporal_plan=original.temporal_plan,
    )
    service.register(dataset, snapshot_root=staging)
    preview = service.plan(
        PlanningRequest(
            schema_version=1, dataset_id=dataset.dataset_id, evaluation="protected", ucb_exploration=(1.0,), seeds=(42,)
        )
    )
    job = service.execute(ExecutePlan(schema_version=1, plan_id=preview.plan_id, request_id="two-family-source-events"))
    wait(service, job.job_id, "awaiting_evaluation")
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    bundle = service.export(job.job_id)
    comparison = bundle.report.comparison
    assert comparison is not None
    assert comparison.results[0].observed_pairs == 2
    assert len(comparison.inputs) == 4
    assert {item.metadata.event_at for item in comparison.inputs} == {
        third.observation.event_at,
        fourth.observation.event_at,
    }
    assert len({item.record.available_at for item in comparison.inputs}) == 1
    assert service.import_evidence(bundle.to_json()).read_only
    altered = []
    for reference in comparison.inputs:
        metadata, record = reference.metadata, reference.record
        if metadata.group_id == fourth.unit_id:
            timestamp = record.available_at - timedelta(seconds=1)
            metadata = metadata.model_copy(update={"available_at": timestamp})
            record = record.model_copy(update={"available_at": timestamp})
        altered.append(PairedObservation(candidate_id=reference.candidate_id, metadata=metadata, record=record))
    audit = EvaluationAudit(tmp_path / "altered-audit")
    audit.save_freeze(comparison.frozen)
    rebuilt = evaluate(comparison.frozen, audit, lambda: tuple(altered), actor="assembly-regression")
    payload = bundle.model_dump(mode="json")
    payload["report"]["comparison"] = rebuilt.model_dump(mode="json")
    with pytest.raises(ValueError, match="assembly timestamp"):
        EvidenceBundle.model_validate(payload)


@pytest.mark.parametrize("partitions", list(itertools.permutations(("train", "validation", "test"))))
def test_native_partition_contiguity_uses_build_order_not_observation_sort(tmp_path, monkeypatch, partitions):
    original = owned_fixture(tmp_path)
    positions = {"train": (1, "a-training"), "validation": (2, "m-validation"), "test": (3, "z-holdout")}
    metadata = []
    for item, partition in zip(original.metadata, partitions, strict=True):
        day, identity = positions[partition]
        event = datetime(2026, 1, day, tzinfo=UTC)
        observation = TemporalObservation(
            observation_id=identity,
            group_id=identity,
            event_at=event,
            available_at=event + timedelta(hours=1),
        )
        metadata.append(item.model_copy(update={"observation": observation}))
    dataset = DatasetManifest.model_validate({**original.model_dump(), "metadata": tuple(metadata)})
    request = PlanningRequest(schema_version=1, dataset_id=dataset.dataset_id, evaluation="protected")
    runtime = recorded_runtime()
    if partitions == ("train", "validation", "test"):
        assert len(make_plan(request, dataset, runtime).trials) == 8
    else:

        def forbidden(*args, **kwargs):
            raise RuntimeError("A noncontiguous protected native sequence reached sweep expansion")

        monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
        with pytest.raises(ValueError, match="chronological contiguous"):
            make_plan(request, dataset, runtime)


def test_actual_protected_continuation_joins_observation_identity_to_native_build(tmp_path):
    service = WorkflowService(tmp_path)
    original = owned_fixture(tmp_path)
    names = ("m-training", "z-validation", "a-holdout")
    metadata = tuple(
        item.model_copy(update={"observation": item.observation.model_copy(update={"observation_id": name})})
        for item, name in zip(original.metadata, names, strict=True)
    )
    dataset = DatasetManifest.model_validate({**original.model_dump(), "metadata": metadata})
    service.register(dataset)
    preview = service.plan(
        PlanningRequest(
            schema_version=1,
            dataset_id=dataset.dataset_id,
            evaluation="protected",
            ucb_exploration=(1.0,),
            limits=WorkflowLimits(max_attempts=2),
        )
    )
    job = service.execute(ExecutePlan(schema_version=1, plan_id=preview.plan_id, request_id="identity-bound-split"))
    wait(service, job.job_id, "awaiting_evaluation")
    prefix = service.inspect(job.job_id, limit=8)
    assert len(prefix.trials) == 4
    assert all(tuple(row.source_build_id for row in trial.rows) == (1, 2) for trial in prefix.trials)
    service.evaluate(job.job_id)
    wait(service, job.job_id)
    report = service.report(job.job_id)
    assert report.comparison is not None
    assert report.comparison.access.prior_test_accesses == 0
    assert report.interpretation == "insufficient_evidence"
    assert all(tuple(row.source_build_id for row in trial.rows) == (1, 2, 3) for trial in report.trials)
    handoff = (tmp_path / "jobs" / job.job_id / "evaluation-handoff.json").read_text()
    saved = EvaluationHandoff.from_json(handoff)
    assert tuple(build.member_id for build in saved.design.builds) == ("build-3",)
    assert saved.design.builds[0].metadata.observation_id == "a-holdout"
    assert service.import_evidence(service.export(job.job_id).to_json()).read_only


def repetition_manifest(root, test_builds):
    """Declare a bounded authored grid, without allocating or executing native repetitions."""
    original = owned_fixture(root)
    metadata = []
    for build in range(1, test_builds + 3):
        event = datetime(2026, 1, min(build, 3), tzinfo=UTC) + timedelta(seconds=build)
        identity = f"ci-build-{build:04d}"
        metadata.append(
            BuildMetadata(
                build_id=build,
                observation=TemporalObservation(
                    observation_id=identity, group_id=identity, event_at=event, available_at=event + timedelta(hours=1)
                ),
                unit_id="authored-family",
                costs_known_at_decision=True,
            )
        )
    return DatasetManifest.model_validate(
        {
            **original.model_dump(),
            "build_ids": tuple(item.build_id for item in metadata),
            "rows": len(metadata) * 3,
            "metadata": tuple(metadata),
        }
    )


@pytest.mark.parametrize("test_builds", [256, 257])
def test_protected_repetition_cell_limit_is_checked_before_expansion(tmp_path, monkeypatch, test_builds):
    dataset = repetition_manifest(tmp_path, test_builds)
    request = PlanningRequest(
        schema_version=1,
        dataset_id=dataset.dataset_id,
        evaluation="protected",
        ucb_exploration=(1.0,),
        seeds=tuple(range(64)),
        limits=WorkflowLimits(max_configurations=2, max_trials=128, max_attempts=2),
    )
    runtime = recorded_runtime()
    if test_builds == 257:

        def forbidden(*args, **kwargs):
            raise RuntimeError("An oversized protected repetition grid reached sweep expansion")

        monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
        with pytest.raises(ValueError, match="16,384"):
            make_plan(request, dataset, runtime)
        return
    # The corpus design's exact cell boundary remains valid, while this many
    # checkpointed native trials exceeds the stricter worker entry reservation.
    assert len(protected_repetition_design(request, dataset).builds) * len(request.seeds) == 16384
    with pytest.raises(ValueError, match="artifact.*entries"):
        make_plan(request, dataset, runtime)
    assert dataset.temporal_plan is not None
    design = RepetitionDesign(
        schema_version=1,
        fold_id="authored-boundary",
        data_fingerprint=dataset.fingerprint,
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        plan=dataset.temporal_plan,
        warmup_end=dataset.temporal_plan.train_end,
        budget=Quantity(value=Decimal("0.5"), semantics=BUDGET_RATIO),
        seeds=request.seeds,
        builds=tuple(
            CorpusBuild(
                member_id=f"build-{item.build_id}",
                project_id=item.unit_id,
                unit_id=item.unit_id,
                metadata=item.observation,
            )
            for item in dataset.metadata[2:]
        ),
        raw_metric="native_napfd",
        raw_semantics=UnitSemantics(unit="1", meaning="native_napfd_v1"),
        cost_projection="binary32_carried_binary64_seconds_v1",
    )
    assert len(design.seeds) * len(design.builds) == 16384
