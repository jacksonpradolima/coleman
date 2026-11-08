"""Actual shipped-mixture ownership, composed recovery and installed processes."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coleman.registry.ownership import DeclaredRootGuard, RootInventory
from tests.learning.test_installed import installed_learning as installed_learning

ROOT = Path(__file__).parents[2]
PROCESS_OBSERVATION_SECONDS = 600


@pytest.fixture(scope="module")
def collection_example():
    name = "authored_learning_collection_test"
    path = ROOT / "examples/learning/collection.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def collection_driver():
    name = "authored_learning_collection_source_driver"
    path = ROOT / "tests/learning/installed_collection_driver.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def test_source_real_reference_moving_and_holding_roots_are_complete(collection_example):
    example = collection_example
    layout = example.collection_layout((0, 3, 4))
    owners, values = example.pending_values(layout)
    reports = tuple(item.inspect(owner) for item, owner in zip(layout.inspections, owners, strict=True))
    inventory = RootInventory(owners[-1].recipe_bindings, tuple(root for report in reports for root in report.roots))
    DeclaredRootGuard(layout.ownership).check(inventory)
    summary = example.summary(layout, owners)
    assert summary["D"] == 12
    assert summary["U"] == summary["pending"] == 6
    assert summary["moving_exact_usage_positive"] is True
    assert summary["optimal_holds_spend_zero"] is True
    assert len(inventory.roots) == 18
    assert len({id(item.root) for item in inventory.roots}) == 18
    assert all(item.codec.encode(item.value).content for item in values)
    first, later = owners[0].session.state.decisions[0], owners[0].session.state.decisions[2]
    assert first.executed.action.weights != later.executed.action.weights


def test_source_full32_shipped_profiles_complete_the_same_exact_movement_period(
    collection_example, tmp_path, record_property
):
    example = collection_example
    source = Path(example.__file__)
    before_source = hashlib.sha256(source.read_bytes()).hexdigest()
    layout = example.collection_layout()
    owners, initial = example.pending_values(layout)
    reports = tuple(item.inspect(owner) for item, owner in zip(layout.inspections, owners, strict=True))
    roots = tuple(root for report in reports for root in report.roots)
    DeclaredRootGuard(layout.ownership).check(RootInventory(owners[-1].recipe_bindings, roots))
    assert len(roots) == len({id(item.root) for item in roots}) == 173
    before = example.summary(layout, owners)
    assert (before["children"], before["components"], before["D"], before["U"], before["pending"]) == (
        32,
        33,
        128,
        64,
        64,
    )
    assert before["R_controlled"] == 18
    assert before["origins"] == 128
    assert before["moving_children"] == before["holding_children"] == 6
    example.complete_pending(layout, owners)
    after = example.summary(layout, owners)
    assert (after["D"], after["U"], after["pending"], after["R_controlled"], after["origins"]) == (128, 128, 0, 36, 128)
    assert after["moving_exact_usage_positive"] is after["optimal_holds_spend_zero"] is True
    for owner in owners[:-1]:
        if isinstance(owner.method, example.ControlledHedge):
            payload = owner.method.export_state().payload
            recipe = next(item for item in layout.ownership.recipes if item.child == owner.specification.identity)
            assert recipe.controls == owner.method.config.binding
            assert payload.release_count >= 2
            assert len(payload.periods) == 1
            assert payload.exact_path is not None
            assert payload.exact_path.through_origin == payload.origin.binding
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_source
    proof = {"before": before, "after": after, "source_sha256": before_source, "initial_components": len(initial)}
    artifact = tmp_path / "source32-summary.json"
    artifact.write_text(json.dumps(proof, sort_keys=True))
    record_property("actual_source32_summary", str(artifact))


def test_source_actual_store_failure_preserves_all_complete_states_and_exact_retry(collection_driver, tmp_path):
    example = collection_driver.load_example()
    layout = example.collection_layout((0, 3, 4))
    result = collection_driver.failed_then_retry(tmp_path / "source-store", layout=layout)
    assert result["generation"] == 1
    assert result["status"] == "new"
    assert result["retry"] == "retained"
    assert result["failure_kept_original_head_and_all_states"] is True
    assert result["retained_retry_callbacks"] == 0
    assert result["callbacks"].count("transition") == 2
    assert result["observation"]["U"] == 12
    assert result["observation"]["pending"] == 0


@pytest.fixture(scope="module")
def installed_collection(installed_learning):
    base = installed_learning
    outside = base["outside"] / "collection"
    outside.mkdir()
    sources = {
        "learning_ci.py": ROOT / "examples/learning/ci.py",
        "collection.py": ROOT / "examples/learning/collection.py",
        "collection_driver.py": ROOT / "tests/learning/installed_collection_driver.py",
    }
    source_hashes = {}
    for name, path in sources.items():
        content = path.read_bytes()
        (outside / name).write_bytes(content)
        source_hashes[name] = hashlib.sha256(content).hexdigest()
    assert not outside.is_relative_to(ROOT)
    result = {**base, "outside": outside, "driver": outside / "collection_driver.py", "sources": source_hashes}
    (base["owned"] / "collection-installation.json").write_text(
        json.dumps(
            {
                "wheel": str(base["wheel"]),
                "wheel_sha256": hashlib.sha256(base["wheel"].read_bytes()).hexdigest(),
                "python": str(base["python"]),
                "outside": str(outside),
                "sources": source_hashes,
            },
            sort_keys=True,
        )
    )
    return result


def invoke(installed, *arguments):
    completed = subprocess.run(
        [str(installed["python"]), "-I", "-B", str(installed["driver"]), *map(str, arguments)],
        cwd=installed["outside"],
        env=installed["environment"],
        capture_output=True,
        text=True,
        timeout=PROCESS_OBSERVATION_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def phase_values(outside, artifact):
    payload = (outside / artifact["name"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == artifact["file_sha256"]
    states = json.loads(payload)
    assert len(states) == 33
    hashes = [
        hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        for state in states
    ]
    assert hashes == artifact["component_sha256"]
    return hashes


@pytest.fixture(scope="module")
def direct_qualification(installed_collection):
    installed = installed_collection
    root = installed["outside"] / "direct"
    aggregate = root / "direct-qualification.json"
    assert not aggregate.exists()
    common = []
    identity = None
    all_phases = []
    for order in ("forward", "reverse"):
        phases = {}
        for phase in ("reference", "interleaved", "restored", "completed"):
            result = invoke(installed, "direct", root, order, phase)
            assert result["order"] == order
            assert result["phase"] == phase
            assert json.loads((root / f"{order}-{phase}.json").read_text()) == result
            if identity is None:
                identity = result["identity"]
            assert result["identity"] == identity
            phases[phase] = result
            assert not aggregate.exists(), "No aggregate may precede all real bounded processes"
        initial = phase_values(root, phases["reference"]["initial"])
        final = phase_values(root, phases["reference"]["final"])
        assert phase_values(root, phases["interleaved"]["initial"]) == initial
        assert phase_values(root, phases["restored"]["initial"]) == initial
        assert phase_values(root, phases["completed"]["final"]) == final
        assert phases["restored"]["actual_live_root_checks"] is True
        assert phases["restored"]["repeated_complete_restores"] == 2
        assert phases["completed"]["other_children_reset_unchanged"] is True
        common.append({"initial_sha256": initial, "final_sha256": final})
        all_phases.append(phases)
    assert common[0] == common[1]
    for name, digest in installed["sources"].items():
        assert hashlib.sha256((installed["outside"] / name).read_bytes()).hexdigest() == digest
    result = {
        "children": 32,
        "orders": 2,
        "components": 33,
        "declared_roots": 173,
        "identity": identity,
        "isolated_reference": common[0],
        "completed_bounded_processes": 8,
        "per_process_observation_seconds": 600,
    }
    temporary = aggregate.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True))
    temporary.replace(aggregate)
    return result, all_phases


def test_installed_actual32_forward_reverse_and_repeated_live_restores_are_complete(direct_qualification):
    result, phases = direct_qualification
    assert result["completed_bounded_processes"] == 8
    for group in phases:
        before, after = group["reference"]["before"], group["completed"]["after"]
        assert (before["D"], before["U"], before["pending"], before["R_controlled"]) == (128, 64, 64, 18)
        assert (after["D"], after["U"], after["pending"], after["R_controlled"]) == (128, 128, 0, 36)
        assert after["moving_exact_usage_positive"] is after["optimal_holds_spend_zero"] is True


def test_installed_actual33_owner_failure_then_retry_matches_uninterrupted_full_state(
    installed_collection, installed_pending_store, direct_qualification
):
    installed = installed_collection
    case = installed["outside"] / "failure"
    case.mkdir()
    root = case / "store"
    shutil.copytree(installed_pending_store, root)
    shutil.copyfile(installed_pending_store.parent / "initial-head.json", case / "initial-head.json")
    failure = invoke(installed, "failed-transition", root)
    failed_bytes = (case / "failed-transition.json").read_bytes()
    assert json.loads(failed_bytes) == failure
    assert failure["initial_sha256"] == direct_qualification[0]["isolated_reference"]["initial_sha256"]
    assert failure["callbacks"].count("transition") == 1
    result = invoke(installed, "retry-transition", root)
    assert result["failed_transition_sha256"] == hashlib.sha256(failed_bytes).hexdigest()
    assert result["generation"] == 1
    assert result["status"] == "new"
    assert result["retry"] == "retained"
    assert result["failure_kept_original_head_and_all_states"] is True
    assert result["retained_retry_callbacks"] == 0
    assert result["callbacks"].count("transition") == 2
    assert result["final_sha256"] == direct_qualification[0]["isolated_reference"]["final_sha256"]
    observed = result["observation"]
    assert (observed["children"], observed["components"], observed["D"], observed["U"], observed["pending"]) == (
        32,
        33,
        128,
        128,
        0,
    )
    assert observed["R_controlled"] == 36


@pytest.fixture(scope="module")
def installed_pending_store(installed_collection):
    root = installed_collection["outside"] / "initial" / "store"
    root.parent.mkdir()
    result = invoke(installed_collection, "initialize", root)
    assert (
        result["generation"],
        result["children"],
        result["components"],
        result["D"],
        result["U"],
        result["pending"],
    ) == (0, 32, 33, 128, 64, 64)
    return root


def wait_for(path, process):
    deadline = time.monotonic() + PROCESS_OBSERVATION_SECONDS
    while not path.exists():
        if process.poll() is not None:
            output, error = process.communicate()
            raise AssertionError(output + error)
        if time.monotonic() >= deadline:
            raise AssertionError("The real writer did not reach its declared process barrier")
        time.sleep(0.01)


@pytest.mark.parametrize("divergent", [False, True])
def test_installed_actual_two_writers_have_one_callback_winner_and_zero_callback_loser(
    installed_collection, installed_pending_store, direct_qualification, divergent
):
    installed = installed_collection
    case = installed["outside"] / ("writers-different" if divergent else "writers-identical")
    case.mkdir()
    root = case / "store"
    shutil.copytree(installed_pending_store, root)
    shutil.copyfile(installed_pending_store.parent / "initial-head.json", case / "initial-head.json")
    base = [str(installed["python"]), "-I", "-B", str(installed["driver"]), "writer", str(root)]
    children = []
    try:
        loser = subprocess.Popen(
            base + ["loser", "different" if divergent else "same"],
            cwd=installed["outside"],
            env=installed["environment"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(loser)
        wait_for(case / "loser-ready.json", loser)
        winner = subprocess.Popen(
            base + ["winner", "same"],
            cwd=installed["outside"],
            env=installed["environment"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(winner)
        wait_for(case / "lock-held.json", winner)
        (case / "start-contention").write_text("attempt under actual held SQLite ownership")
        wait_for(case / "loser-busy.json", loser)
        busy = json.loads((case / "loser-busy.json").read_text())
        assert busy["callbacks"] == []
        assert busy["lock_wait_seconds"] == 30.0
        assert busy["elapsed_seconds"] > 0.0
        assert winner.poll() is None
        (case / "release-lock").write_text("publish the actual completed winner")
        output, error = winner.communicate(timeout=PROCESS_OBSERVATION_SECONDS)
        assert winner.returncode == 0, output + error
        winning = json.loads(output)
        assert winning["status"] == "new"
        assert winning["generation"] == 1
        assert winning["callbacks"].count("transition") == 1
        assert sum(item.startswith("factory:") for item in winning["callbacks"]) == 33
        (case / "retry-loser").write_text("reconcile the original predecessor and intent")
        output, error = loser.communicate(timeout=PROCESS_OBSERVATION_SECONDS)
        assert loser.returncode == 0, output + error
        losing = json.loads(output)
        assert losing["status"] == ("rejected" if divergent else "retained")
        assert losing["callbacks"] == []
        observed = invoke(installed, "inspect", root)
        assert (
            observed["generation"],
            observed["D"],
            observed["U"],
            observed["pending"],
            observed["R_controlled"],
        ) == (1, 128, 128, 0, 36)
        assert observed["component_sha256"] == direct_qualification[0]["isolated_reference"]["final_sha256"]
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
