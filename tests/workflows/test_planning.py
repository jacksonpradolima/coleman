"""Bounded native planning, version compatibility and metamorphic identities."""

import itertools
import json
from pathlib import Path

import pytest

from coleman.application.workflow_registry import owned_fixture
from coleman.results.domain import fingerprint
from coleman.spec.runtime_identity import DistributionIdentity, InstalledRuntime
from coleman.workflows import (
    DatasetManifest,
    ExperimentPlan,
    PlanningRequest,
    ScheduleBudget,
    WorkflowLimits,
    make_plan,
)


def recorded_runtime():
    return InstalledRuntime(
        schema_version=1,
        package_fingerprint="sha256:" + "a" * 64,
        python_implementation="CPython",
        python_version="3.14.0",
        operating_system="Linux",
        machine="x86_64",
        distributions=(DistributionIdentity(name="coleman", version="0.0.0"),),
    )


def manifest():
    return DatasetManifest(
        schema_version=1,
        dataset_id="owned-ci",
        sha256="sha256:" + "b" * 64,
        size_bytes=1024,
        build_ids=(1, 2, 3),
        rows=9,
        maximum_tests=3,
        source_fingerprint="sha256:" + "c" * 64,
        projection_fingerprint="sha256:" + "d" * 64,
        evidence="constructed_mechanism",
        assumptions=("Authored deterministic CI fixture.",),
    )


def test_fixture_and_golden_roundtrip(tmp_path):
    data = owned_fixture(tmp_path)
    plan = make_plan(PlanningRequest(schema_version=1, dataset_id=data.dataset_id), data, recorded_runtime())
    assert len(plan.trials) == 8
    assert len({t.configuration_id for t in plan.trials}) == 4
    assert sum(t.policy == "Random" for t in plan.trials) == 2
    assert ExperimentPlan.from_json(plan.to_json()) == plan
    request = PlanningRequest(schema_version=1, dataset_id="owned-ci")
    golden = make_plan(request, manifest(), recorded_runtime())
    expected = Path(__file__).with_name("fixtures").joinpath("experiment-plan-v1.json").read_text().strip()
    assert golden.to_json() == expected
    assert ExperimentPlan.from_json(expected) == golden


def test_grid_order_metamorphism_and_relevant_changes():
    data = manifest()
    reference = None
    for values in itertools.permutations((0.5, 1.0, 2.0)):
        request = PlanningRequest(
            schema_version=1,
            dataset_id=data.dataset_id,
            policies=("UCB", "Random"),
            seeds=(43, 42),
            ucb_exploration=values,
        )
        plan = make_plan(request, data, recorded_runtime())
        reference = reference or plan
        assert plan == reference
        assert plan.fingerprint == reference.fingerprint
    assert reference is not None
    changed = data.model_copy(update={"projection_fingerprint": fingerprint(["different-projection"])})
    other = make_plan(reference.request, changed, recorded_runtime())
    assert other.fingerprint != reference.fingerprint
    assert set(t.trial_id for t in other.trials).isdisjoint(t.trial_id for t in reference.trials)


def test_zip_pairs_and_baseline_not_multiplied():
    request = PlanningRequest(
        schema_version=1, dataset_id="owned-ci", mode="zip", budget=ScheduleBudget(values=(0.2, 0.5, 0.9))
    )
    plan = make_plan(request, manifest(), recorded_runtime())
    assert len(plan.trials) == 12
    assert {(t.exploration, t.budget_value) for t in plan.trials if t.policy == "UCB"} == {
        (0.5, 0.2),
        (1.0, 0.5),
        (2.0, 0.9),
    }
    assert sum(t.policy == "Random" for t in plan.trials) == 6


@pytest.mark.parametrize("values", [(0,), (-1,), (float("nan"),), (float("inf"),), (True,), (1, 1)])
def test_invalid_ucb_values(values):
    with pytest.raises(ValueError):
        PlanningRequest(schema_version=1, dataset_id="owned-ci", ucb_exploration=values)


@pytest.mark.parametrize(
    "mode,unit,values",
    [
        ("ratio", "s", (0.5,)),
        ("ratio", "1", (0,)),
        ("fixed_time", "s", (-1,)),
        ("subset_size", "count", (0.5,)),
        ("ratio", "1", (None,)),
    ],
)
def test_native_units_and_values_fail_closed(mode, unit, values):
    with pytest.raises(ValueError):
        ScheduleBudget(mode=mode, unit=unit, values=values)


def test_preflight_before_expansion(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized grid was expanded")

    monkeypatch.setattr("coleman.workflows.planning.expand_sweep", forbidden)
    request = PlanningRequest(schema_version=1, dataset_id="owned-ci", limits=WorkflowLimits(max_trials=7))
    source_manifest = manifest()
    installed_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="before expansion"):
        make_plan(request, source_manifest, installed_runtime)


@pytest.mark.parametrize("version", [0, 2, True, None])
def test_unknown_versions(version):
    payload = {"schema_version": version, "dataset_id": "owned-ci"}
    future_request_json = json.dumps(payload)
    with pytest.raises(ValueError):
        PlanningRequest.from_json(future_request_json)


def test_duplicate_unknown_keys_and_parameter_missingness():
    with pytest.raises(ValueError):
        PlanningRequest.from_json('{"schema_version":1,"dataset_id":"a","dataset_id":"b"}')
    with pytest.raises(ValueError):
        PlanningRequest.model_validate({"schema_version": 1, "dataset_id": "owned-ci", "arbitrary_path": "/tmp"})
    plan = make_plan(
        PlanningRequest(schema_version=1, dataset_id="owned-ci", seeds=(0,)), manifest(), recorded_runtime()
    )
    assert any(t.exploration is None for t in plan.trials)
    assert all(t.seed == 0 for t in plan.trials)


def test_protected_requires_actual_clocks_and_rejects_late_labels(tmp_path):
    protected_request = PlanningRequest(schema_version=1, dataset_id="owned-ci", evaluation="protected")
    missing_clock_manifest = manifest()
    missing_clock_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="temporal"):
        make_plan(protected_request, missing_clock_manifest, missing_clock_runtime)
    data = owned_fixture(tmp_path)
    request = PlanningRequest(schema_version=1, dataset_id=data.dataset_id, evaluation="protected")
    assert len(make_plan(request, data, recorded_runtime()).trials) == 8
    metadata = list(data.metadata)
    metadata[0] = metadata[0].model_copy(update={"costs_known_at_decision": False})
    changed = data.model_copy(update={"metadata": tuple(metadata)})
    changed_runtime = recorded_runtime()
    with pytest.raises(ValueError, match="predecision"):
        make_plan(request, changed, changed_runtime)
