"""Actual shipped Hedge and local ACKs through the external fixture source.

These source tests select fixed public fixture paths explicitly. Installed
distribution capture/entry-point loading is qualified separately by the host.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from coleman.core.primitives import fingerprint
from coleman.learning.reference import Hedge
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import TaskPlan
from coleman.tasks.protocols import AttemptContext, PreparationContext, encode_value

ROOT = Path(__file__).parents[2]
PROVIDER = ROOT / "examples/typed_executor/provider/src"
SUPPORT = ROOT / "examples/typed_executor/support/src"


@pytest.fixture
def source(monkeypatch):
    monkeypatch.syspath_prepend(str(SUPPORT))
    monkeypatch.syspath_prepend(str(PROVIDER))
    return SimpleNamespace(
        package=importlib.import_module("coleman_typed_hedge"),
        contracts=importlib.import_module("coleman_typed_hedge.contracts"),
        recipe=importlib.import_module("coleman_typed_hedge.recipe"),
        sample=importlib.import_module("coleman_typed_hedge.sample"),
        executor=importlib.import_module("coleman_typed_hedge.executor"),
        state=importlib.import_module("coleman_typed_hedge._state"),
        host=importlib.import_module("examples.typed_executor.ci"),
    )


def case(source, tmp_path, config=None):
    config = source.sample.configuration() if config is None else config
    inputs = source.sample.inputs(config)
    request = source.sample.planning_request()
    fixture = TaskPlan.from_json((ROOT / "tests/fixtures/tasks/v1/plan.json").read_bytes())
    implementation = fixture.implementation.binding
    context = PreparationContext(
        fingerprint({"configuration": config.fingerprint, "source": inputs.fingerprint}),
        inputs.binding,
        implementation,
        ExecutionCompatibility(
            schema_version=1, runtime=fixture.implementation.runtime, owner_implementations=(implementation,)
        ),
        tmp_path,
    )
    from coleman.tasks.protocols import CoordinateContext

    envelope = type(request.coordinates[0]).model_validate(
        {
            **request.coordinates[0].model_dump(),
            "body": encode_value(source.recipe.COORDINATE, inputs.trajectories[0]),
        }
    )
    coordinate = CoordinateContext(
        context,
        envelope.fingerprint,
        inputs.trajectories[0].reference,
        request.coordinates[0].target,
        tmp_path / "coordinate",
    )
    prepared = source.package.bootstrap().prepare(config, inputs, context)
    return SimpleNamespace(
        config=config,
        inputs=inputs,
        request=request,
        context=context,
        coordinate=coordinate,
        prepared=prepared,
        trajectory=inputs.trajectories[0],
    )


def attempt(item, guard=lambda: None):
    return AttemptContext(item.coordinate, "actual-source-attempt", 1, 1e20, guard)


def direct(item):
    learner = Hedge(item.config.reference, owner=item.trajectory.reference)
    for panel in item.trajectory.panels:
        assert learner.observe(panel).status == "accepted"
    return learner.export_state()


@pytest.mark.parametrize("mode", ["recipe", "direct"])
def test_recipe_is_static_complete_and_cold_direct_math_is_registry_free(source, mode):
    recipe = source.package.recipe()
    assert {item.role for item in recipe.codecs} == {
        "configuration",
        "inputs",
        "coordinate",
        "coordinate_result",
        "final_result",
        "state",
        "progress_proof",
    }
    assert recipe.effects == "local_state"
    library_paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    program = f"""
import sys
sys.path.extend({library_paths!r})
sys.path.insert(0, {str(ROOT)!r})
blocked=('sqlite3','coleman.state.local','coleman.application','numpy','polars','pyarrow')
if {mode!r} == 'direct': blocked += ('coleman.tasks',)
assert not any(name == root or name.startswith(root+'.') for name in sys.modules for root in blocked)
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name+'.') for name in blocked):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from coleman.learning.reference import Hedge
if {mode!r} == 'recipe':
    sys.path.insert(0, {str(PROVIDER)!r})
    from coleman_typed_hedge import recipe
    assert recipe().effects == 'local_state'
else:
    from coleman.learning.contracts import ReferenceConfig
    from coleman.core.feedback import FullPanel
    from coleman.core.identity import Reference
    config = ReferenceConfig.from_json({source.sample.configuration().reference.to_json()!r})
    learner = Hedge(config, owner=Reference(id='cold-direct', version='1'))
    panel = FullPanel.model_validate_json({source.sample.inputs().trajectories[0].panels[0].model_dump_json()!r})
    assert learner.observe(panel).status == 'accepted'
    assert learner.export_state().payload.update_count == 1
assert not any(name == root or name.startswith(root+'.') for name in sys.modules for root in blocked)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program], check=False, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_real_three_panels_batch_two_full_direct_state_and_generation_are_distinct(source, tmp_path):
    item = case(source, tmp_path)
    progress = []
    result = item.prepared.advance(item.trajectory, attempt(item), progress.append)
    assert result.checkpoint.learner.to_json() == direct(item).to_json()
    assert result.proof.frontier.position == 3
    assert result.proof.observed_head.generation == 2
    assert tuple(value.frontier.position for value in progress) == (2, 3)
    assert result.proof.receipt.status == "retained"
    for codec, value in (
        (source.recipe.CONFIGURATION, item.config),
        (source.recipe.INPUTS, item.inputs),
        (source.recipe.COORDINATE, item.trajectory),
        (source.recipe.STATE, result.checkpoint),
        (source.recipe.COORDINATE_RESULT, result),
        (source.recipe.PROGRESS_PROOF, result.proof),
    ):
        assert codec.decode(codec.encode(value)).to_json() == value.to_json()
    item.prepared.publish_coordinate(item.trajectory, result, item.coordinate)
    assert item.prepared.read_completed(item.trajectory, item.coordinate).to_json() == result.to_json()
    calls = []
    report = item.prepared.boundary.produce(lambda: calls.append("load") or (result,))
    assert calls == ["load"]
    assert report.interpretation == "constructed_mechanism_only"
    assert source.recipe.FINAL_RESULT.decode(source.recipe.FINAL_RESULT.encode(report)).to_json() == report.to_json()


def test_post_ack_callback_loss_reconciles_real_receipt_and_never_reobserves_accepted_panels(source, tmp_path):
    item = case(source, tmp_path)

    def lose_response(progress):
        assert progress.frontier.position == 2
        raise RuntimeError("response lost after actual ACK")

    with pytest.raises(RuntimeError, match="response lost"):
        item.prepared.advance(item.trajectory, attempt(item), lose_response)
    recovered = item.prepared.reconcile(item.coordinate)
    assert recovered.frontier.position == 2
    assert recovered.proof.receipt.status == "retained"
    item.prepared.validate(recovered, item.coordinate)
    fresh = source.package.bootstrap().prepare(item.config, item.inputs, item.context)
    callbacks = []
    result = fresh.advance(item.trajectory, attempt(item), callbacks.append)
    assert [entry.frontier.position for entry in callbacks] == [3]
    assert result.checkpoint.learner.to_json() == direct(item).to_json()
    assert result.proof.observed_head.generation == 2


def test_retained_original_command_runs_zero_factory_restore_or_callback(source, tmp_path, monkeypatch):
    item = case(source, tmp_path)
    item.prepared.advance(item.trajectory, attempt(item), lambda _: None)

    def forbidden(*_args):
        raise AssertionError("retained lookup reached a real owner factory")

    monkeypatch.setattr(source.state.HedgeOwner, "__init__", forbidden)
    first = item.prepared.reconcile(item.coordinate)
    second = item.prepared.reconcile(item.coordinate)
    assert first.proof.to_json() == second.proof.to_json()
    assert second.frontier.position == 3


def test_prepare_and_static_recipe_create_zero_owned_factories_and_properties_are_readonly(
    source, tmp_path, monkeypatch
):
    calls = []
    original = source.state.HedgeOwner.__init__

    def record(owner, initial):
        calls.append(initial.learner.payload.update_count)
        original(owner, initial)

    monkeypatch.setattr(source.state.HedgeOwner, "__init__", record)
    item = case(source, tmp_path)
    executor = source.package.bootstrap()
    source.package.recipe()
    assert calls == []
    assert item.prepared.reconcile(item.coordinate) is None
    assert calls == []
    with pytest.raises(AttributeError):
        executor.configuration_codec = source.recipe.CONFIGURATION
    item.prepared.advance(item.trajectory, attempt(item), lambda _: None)
    assert calls == [0, 0, 0]  # Initialization and two accepted transitions; each starts with a fresh U0 owner.


def test_numeric_refusal_does_not_ack_a_partial_batch_or_mutate_complete_math(source, tmp_path):
    base = source.sample.configuration()
    config = source.contracts.HedgeConfiguration.model_validate(
        {
            **base.model_dump(),
            "reference": {**base.reference.model_dump(), "eta": 1e308},
        }
    )
    item = case(source, tmp_path, config)
    callbacks = []
    with pytest.raises(ValueError, match="numerically"):
        item.prepared.advance(item.trajectory, attempt(item), callbacks.append)
    assert callbacks == []
    progress = item.prepared.reconcile(item.coordinate)
    assert progress.frontier.position == 0
    assert progress.proof.observed_head.generation == 0
    layout = source.state.Layout(item.config, item.trajectory, item.coordinate)
    assert layout.checkpoint(layout.open()).learner.to_json() == item.trajectory.initial_state.to_json()


def test_published_result_corruption_cannot_replace_actual_acknowledged_math(source, tmp_path):
    item = case(source, tmp_path)
    result = item.prepared.advance(item.trajectory, attempt(item), lambda _: None)
    item.prepared.publish_coordinate(item.trajectory, result, item.coordinate)
    raw = result.model_dump()
    raw["checkpoint"]["learner"]["payload"]["cumulative_losses"] = (1.5, 1.75, 1.5)
    forged = source.contracts.HedgeCoordinateResult.model_validate(raw)
    (item.coordinate.artifact_directory / "coordinate-result.json").write_text(
        source.recipe.COORDINATE_RESULT.encode(forged),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete actual current checkpoint"):
        item.prepared.read_completed(item.trajectory, item.coordinate)
    actual = item.prepared.reconcile(item.coordinate)
    assert actual.proof.observed_head == result.proof.observed_head
    layout = source.state.Layout(item.config, item.trajectory, item.coordinate)
    assert layout.checkpoint(layout.open()).learner.to_json() == direct(item).to_json()


def test_static_resource_matches_pure_recipe_and_all_required_declared_codecs(source):
    resource = PROVIDER / "coleman_typed_hedge/recipe.json"
    assert resource.read_text() == source.package.recipe().to_json() + "\n"
    executor = source.package.bootstrap()
    for role, attribute in (
        ("configuration", "configuration_codec"),
        ("inputs", "input_codec"),
        ("coordinate", "coordinate_codec"),
        ("coordinate_result", "coordinate_result_codec"),
        ("final_result", "final_result_codec"),
        ("state", "state_codec"),
        ("progress_proof", "progress_proof_codec"),
    ):
        assert getattr(executor, attribute).declaration == source.package.recipe().codec_for(role)


def test_concrete_reader_rejects_oversized_mutable_input_before_copying(source):
    body = bytearray(b" " * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="before copying"):
        source.contracts.HedgeInputs.model_validate_json(body)
    assert len(body) == 2 * 1024**2 + 1


def test_declared_trajectory_refuses_exhausted_math_capacity_before_any_state_effect(source):
    base = source.sample.configuration()
    config = source.contracts.HedgeConfiguration.model_validate(
        {
            **base.model_dump(),
            "reference": {
                **base.reference.model_dump(),
                "limits": {
                    **base.reference.limits.model_dump(),
                    "maximum_updates": 2,
                },
            },
        }
    )
    with pytest.raises(ValueError, match="accepted-panel capacity"):
        source.sample.inputs(config)


@pytest.mark.parametrize("reader", ["operator", "state"])
@pytest.mark.parametrize("capacity", [-2, -1, 0, True, 1.0, 2 * 1024**2 + 1])
def test_invalid_acquisition_capacity_refuses_before_open(source, tmp_path, monkeypatch, reader, capacity):
    selected = source.host.read_artifact if reader == "operator" else source.state.read_text
    path = tmp_path / "never-opened"
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("opened")
        raise AssertionError("Invalid byte capacity reached file acquisition")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="strict positive capacity"):
        selected(path, capacity)
    assert calls == []


@pytest.mark.parametrize("kind", ["source", "configuration", "trajectory", "target", "profile"])
def test_exact_preflight_bindings_refuse_before_guard_or_state_creation(source, tmp_path, kind):
    item = case(source, tmp_path)
    calls = []
    if kind == "source":
        context = dataclasses.replace(item.context, source=source.recipe.binding("foreign-source", {"x": 1}))
        with pytest.raises(ValueError, match="source"):
            source.package.bootstrap().prepare(item.config, item.inputs, context)
    elif kind == "configuration":
        config = source.contracts.HedgeConfiguration.model_validate(
            {
                **item.config.model_dump(),
                "reference": {
                    **item.config.reference.model_dump(),
                    "eta": 0.0,
                },
            }
        )
        with pytest.raises(ValueError, match="configurations"):
            source.package.bootstrap().prepare(config, item.inputs, item.context)
    else:
        coordinate, trajectory = item.coordinate, item.trajectory
        if kind == "trajectory":
            trajectory = source.contracts.HedgeTrajectory.model_validate(
                {
                    **trajectory.model_dump(),
                    "panels": trajectory.panels[:2],
                }
            )
        elif kind == "target":
            coordinate = dataclasses.replace(coordinate, target=coordinate.target.model_copy(update={"position": 2}))
        else:
            target = coordinate.target.model_copy(update={"profile": source.recipe.binding("wrong", {})})
            coordinate = dataclasses.replace(coordinate, target=target)
        context = dataclasses.replace(attempt(item, lambda: calls.append("guard")), coordinate=coordinate)
        with pytest.raises(ValueError, match="coordinate"):
            item.prepared.advance(trajectory, context, calls.append)
    assert calls == []
    assert not item.coordinate.artifact_directory.exists()


def test_wrong_progress_and_missing_previous_store_refuse_without_fresh_fallback(source, tmp_path):
    item = case(source, tmp_path)
    result = item.prepared.advance(item.trajectory, attempt(item), lambda _: None)
    actual = item.prepared.reconcile(item.coordinate)
    wrong = dataclasses.replace(actual, plan_fingerprint=fingerprint({"other": "plan"}))
    old = result.checkpoint.to_json()
    with pytest.raises(ValueError, match="actual current ACK"):
        item.prepared.validate(wrong, item.coordinate)
    assert (
        source.state.Layout(item.config, item.trajectory, item.coordinate)
        .checkpoint(source.state.Layout(item.config, item.trajectory, item.coordinate).open())
        .to_json()
        == old
    )
    shutil.rmtree(item.coordinate.artifact_directory / "state")
    with pytest.raises(ValueError, match="missing"):
        item.prepared.reconcile(item.coordinate)
    with pytest.raises(ValueError, match="fresh fallback"):
        item.prepared.advance(item.trajectory, attempt(item), lambda _: None)
    assert not (item.coordinate.artifact_directory / "state").exists()


def test_batch_failure_keeps_previous_complete_state_and_zero_ack_callbacks(source, tmp_path):
    item = case(source, tmp_path)
    checks, callbacks = [], []

    def guard():
        checks.append("checked")
        if len(checks) == 3:
            raise RuntimeError("second original panel interrupted")

    with pytest.raises(RuntimeError, match="second original"):
        item.prepared.advance(item.trajectory, attempt(item, guard), callbacks.append)
    assert callbacks == []
    progress = item.prepared.reconcile(item.coordinate)
    assert progress.frontier.position == 0
    layout = source.state.Layout(item.config, item.trajectory, item.coordinate)
    assert layout.checkpoint(layout.open()).learner.to_json() == item.trajectory.initial_state.to_json()
    result = item.prepared.advance(item.trajectory, attempt(item), callbacks.append)
    assert result.checkpoint.learner.to_json() == direct(item).to_json()


@pytest.mark.parametrize("edit", ["future", "unknown", "nonfinite", "unavailable", "noncanonical"])
def test_selected_concrete_codec_refuses_malformed_original_data(source, edit):
    trajectory = source.sample.inputs().trajectories[0]
    data = json.loads(trajectory.to_json())
    if edit == "future":
        data["schema_version"] = 2
    elif edit == "unknown":
        data["surprise"] = 1
    elif edit == "nonfinite":
        data["panels"][0]["rows"][0]["values"][0]["value"] = float("nan")
    elif edit == "unavailable":
        data["panels"][0]["rows"][0]["values"][0].update(value=None, status="unavailable")
    if edit == "noncanonical":
        body = json.dumps(data, indent=2)
    else:
        body = json.dumps(data, separators=(",", ":"), sort_keys=True)
    with pytest.raises(ValueError):
        source.recipe.COORDINATE.decode(body)
