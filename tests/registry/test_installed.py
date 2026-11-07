"""Real noneditable base-wheel32-child registry ownership and recovery."""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]
PROCESS_OBSERVATION_SECONDS = 600


@pytest.fixture(scope="module")
def installed_registry(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-registry-state")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    virtual = owned / "installed"
    wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    outside = owned / "outside"
    outside.mkdir()
    shutil.copyfile(ROOT / "examples/state/ci.py", outside / "owned_ci.py")
    shutil.copyfile(ROOT / "examples/registry/collection.py", outside / "collection.py")
    shutil.copyfile(ROOT / "tests/registry/installed_driver.py", outside / "driver.py")
    assert not outside.is_relative_to(ROOT)
    (owned / "installation.json").write_text(
        json.dumps(
            {
                "wheel": str(wheel),
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "python": str(virtual / "bin/python"),
                "outside": str(outside),
            }
        )
    )
    return virtual / "bin/python", outside, environment


def invoke(installed, arguments, *, expected=0, timeout=PROCESS_OBSERVATION_SECONDS):
    python, outside, environment = installed
    result = subprocess.run(
        [str(python), "-I", "-B", *arguments],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def pending_store(installed_registry):
    outside = installed_registry[1]
    root = outside / "initial32" / "store"
    result = json.loads(invoke(installed_registry, [str(outside / "driver.py"), "initialize32", str(root)]))
    assert result["D"] == 95
    assert result["U"] == 48
    assert result["pending"] == 47
    assert result["loss"] == 16
    assert result["cadence"] == 42
    assert result["virtual_seconds"] == 389
    assert result["actual_seconds"] == 351
    assert result["components"] == 34
    assert result["declared_roots"] == 98
    assert result["registry_origins"] == 95
    assert result["registry_facts"] == 32
    assert result["registry_bytes"] < 2 * 1024**2
    (root.parent / "initial-summary.json").write_text(json.dumps(result, sort_keys=True))
    return root


def test_pure_registry_declarations_and_actual_roots_use_only_the_installed_base(installed_registry):
    script = r"""
import importlib.abc,importlib.metadata,random,sys
from pathlib import Path
blocked=('sqlite3','numpy','scipy','polars','pyarrow','duckdb','mcp','openai','anthropic',
         'coleman.state.local','coleman.state.components','coleman.artifacts','coleman.spec',
         'coleman.cli','coleman.runner','coleman.application','coleman.native','coleman.agent',
         'coleman.policy','coleman.telemetry')
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
   raise AssertionError('Pure registry dependency crossed boundary: '+fullname)
sys.meta_path.insert(0,Boundary())
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.contracts import StateBinding
from coleman.registry.definitions import RecipeBinding
import coleman.registry.lifecycle
import coleman.registry.envelopes
import coleman.registry.composition
import coleman.registry.ownership as ownership
assert Path(ownership.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(ownership.__file__).resolve().is_relative_to(Path.cwd())
installed={item.metadata['Name'].lower() for item in importlib.metadata.distributions()}
assert not {'numpy','scipy','polars','pyarrow','duckdb','mcp'} & installed
r=lambda name:Reference(id=name,version='v1')
d=lambda name:DefinitionBinding(reference=r(name),definition_fingerprint=fingerprint(name))
owner=StateBinding(owner=r('owner'),implementation=d('toy'),state_codec=r('state'),configuration_fingerprint=fingerprint(1))
recipe=RecipeBinding(schema_version=1,child=r('child'),task=r('task'),expert=d('expert'),state_epoch=r('epoch'),
 owners=(owner,),objective=r('objective'),target=None,loss_transform=None,controls=None)
claims=tuple(ownership.ResourceClaim(resource=r(role),owner=owner,child=recipe.child,role=role,access='owned')
 for role in ('owner','weights','rng'))
declared=ownership.OwnershipDeclaration(schema_version=1,recipes=(recipe,),resources=claims)
inventory=ownership.RootInventory((recipe,),tuple(ownership.RootBinding(claim,root)
 for claim,root in zip(claims,(object(),[1,2],random.Random(4)),strict=True)))
ownership.DeclaredRootGuard(ownership.OwnershipDeclaration.from_json(declared.to_json())).check(inventory)
assert not any(name==item or name.startswith(item+'.') for name in sys.modules for item in blocked)
print('declared roots/bindings checked')
"""
    assert invoke(installed_registry, ["-c", script]).strip() == "declared roots/bindings checked"


def test_actual32_isolated_interleaved_reordered_and_restored_child_traces(installed_registry):
    outside = installed_registry[1]
    result = json.loads(
        invoke(installed_registry, [str(outside / "driver.py"), "direct-traces", str(outside / "direct" / "unused")])
    )
    assert result == {
        "children": 32,
        "orders": 2,
        "origins": 95,
        "preserved_complete_child_traces": True,
        "restored_pending": 47,
    }


def test_actual32_wrong_child_expert_epoch_controls_catalogue_and_aliases_reject(installed_registry):
    outside = installed_registry[1]
    result = json.loads(
        invoke(installed_registry, [str(outside / "driver.py"), "rejections", str(outside / "rejections" / "unused")])
    )
    assert set(result["rejected"]) == {
        "child",
        "expert",
        "controls",
        "state_epoch",
        "catalogue",
        "feedback-child",
        "live-host",
        "live-replay",
        "future-registry",
        "inactive-before-emission",
        "unavailable-before-emission",
        "retired-before-emission",
        "policy-signed-zero",
    }
    assert result["unchanged_live_owners"] == 34


@pytest.mark.parametrize(
    ("fault", "code", "acknowledged"),
    [
        # The unchanged store's eight publication boundaries remain exercised
        # by tests/state/test_installed.py. These cases qualify the added registry
        # component's actual typed stream/blob and complete head/acknowledgement.
        ("registry-mid-stream", 88, False),
        ("after_blob:33", 87, False),
        ("after_head_update", 87, False),
        ("after_commit", 87, True),
    ],
)
def test_actual_process_death_recovers_only_acknowledged_complete34_owner_collection(
    installed_registry, pending_store, fault, code, acknowledged
):
    outside = installed_registry[1]
    case = outside / ("fault-" + fault.replace(":", "-"))
    case.mkdir()
    root = case / "store"
    shutil.copytree(pending_store, root)
    shutil.copyfile(pending_store.parent / "initial-head.json", case / "initial-head.json")
    invoke(installed_registry, [str(outside / "driver.py"), "transition", str(root), fault], expected=code)
    assert (case / "fault-barrier.json").is_file()
    if fault == "registry-mid-stream":
        assert json.loads((case / "fault-barrier.json").read_text()) == {
            "phase": "registry-mid-stream",
            "component": 33,
        }
    before_retry = json.loads(invoke(installed_registry, [str(outside / "driver.py"), "inspect", str(root)]))
    assert before_retry["generation"] == int(acknowledged)
    assert before_retry["U"] == (95 if acknowledged else 48)
    assert before_retry["pending"] == (0 if acknowledged else 47)
    assert before_retry["registry_origins"] == 95
    assert before_retry["registry_facts"] == (33 if acknowledged else 32)
    initial_summary = json.loads((pending_store.parent / "initial-summary.json").read_text())
    assert (before_retry["registry_fingerprint"] != initial_summary["registry_fingerprint"]) == acknowledged
    if fault == "registry-mid-stream":
        assert list(root.rglob("*.tmp")), "Abrupt real immutable writer death leaves an unacknowledged temporary"
    recovered = json.loads(invoke(installed_registry, [str(outside / "driver.py"), "transition", str(root)]))
    assert recovered["generation"] == 1
    assert recovered["D"] == 95
    assert recovered["U"] == 95
    assert recovered["pending"] == 0
    assert recovered["loss"] == 35.5
    assert recovered["cadence"] == 42
    assert recovered["actual_seconds"] == 351
    assert recovered["virtual_seconds"] == 389
    assert recovered["registry_origins"] == 95
    assert recovered["registry_facts"] == 33
    assert recovered["registry_fingerprint"] != initial_summary["registry_fingerprint"]
    assert recovered["status"] == ("retained" if acknowledged else "new")
    diagnostic = (case / "attempt-calls.log").read_text().splitlines()
    assert diagnostic.count("actual-transition") == (1 if acknowledged else 2)
    baseline = json.loads((pending_store.parent / "observed-states.json").read_text())
    resumed = json.loads((case / "observed-states.json").read_text())
    for initial, final in zip(baseline[:-2], resumed[:-2], strict=True):
        assert final["host_state"]["rng"] == initial["host_state"]["rng"]
        assert final["host_state"]["book"] == initial["host_state"]["book"]
        assert final["host_state"]["cadence_count"] == initial["host_state"]["cadence_count"]
    assert resumed[-2] == baseline[-2]
    assert resumed[-1]["executions"] == baseline[-1]["executions"]
    assert sum(len(item["facts"]) for item in resumed[-1]["registries"]) == 33


@pytest.mark.parametrize("damage", ["missing", "corrupt", "future"])
def test_missing_corrupt_or_future_registry_rejects_before_any_factory_or_inspector(
    installed_registry, pending_store, damage
):
    outside = installed_registry[1]
    case = outside / ("registry-" + damage)
    case.mkdir()
    root = case / "store"
    shutil.copytree(pending_store, root)
    shutil.copyfile(pending_store.parent / "initial-head.json", case / "initial-head.json")
    initial = json.loads((pending_store.parent / "initial-summary.json").read_text())
    blob = root / "blobs" / (initial["registry_blob"].removeprefix("sha256:") + ".json")
    if damage == "missing":
        blob.unlink()
    elif damage == "corrupt":
        blob.write_bytes(b"corrupt-registry")
    else:
        # Publish self-consistent future bytes rather than letting a stale raw
        # digest/count reject before the selected registry component reader.
        publication = json.loads(
            invoke(installed_registry, [str(outside / "driver.py"), "publish-future-registry", str(root)])
        )
        assert publication["schema_version"] == 2
        assert publication["metadata_opened"] is True
        assert publication["generation"] == 0
        blob = root / "blobs" / (publication["registry_blob"][7:] + ".json")
        content = blob.read_bytes()
        assert "sha256:" + hashlib.sha256(content).hexdigest() == publication["registry_blob"]
        assert len(content) == publication["registry_bytes"]
        assert json.loads(content)["schema_version"] == 2
        manifest_path = root / "manifests" / (publication["manifest_fingerprint"][7:] + ".json")
        manifest_bytes = manifest_path.read_bytes()
        assert "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() == publication["manifest_fingerprint"]
        manifest = json.loads(manifest_bytes)
        receipt = next(item for item in manifest["components"] if item["fingerprint"] == publication["registry_blob"])
        assert receipt["byte_count"] == len(content)
        head = json.loads((case / "initial-head.json").read_text())
        assert head["manifest_fingerprint"] == publication["manifest_fingerprint"]
    before = {path: path.read_bytes() for path in (root / "blobs").iterdir()}
    result = json.loads(invoke(installed_registry, [str(outside / "driver.py"), "transition", str(root)]))
    assert "rejected" in result
    (case / "rejection.json").write_text(json.dumps(result, sort_keys=True))
    if damage == "future":
        assert result["rejection_stage"] == "commit"
        assert result["error_type"] == "ValidationError"
        assert result["validation_model"] == "RegistryCheckpoint"
        assert result["validation_errors"] == [
            {
                "type": "less_than_equal",
                "loc": ["schema_version"],
                "msg": "Input should be less than or equal to 1",
                "input": 2,
            }
        ]
    else:
        assert result["rejection_stage"] == "open"
        assert result["error_type"] == "ValueError"
        assert (
            result["rejected"]
            == {
                "missing": "Acknowledged state file is missing or is not a regular owned file",
                "corrupt": "Acknowledged state artifact digest mismatch",
            }[damage]
        )
    assert result["callbacks"] == []
    assert not (case / "attempt-calls.log").exists()
    assert {path: path.read_bytes() for path in (root / "blobs").iterdir()} == before


def wait_for(path, process, *, timeout=PROCESS_OBSERVATION_SECONDS):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            output, error = process.communicate()
            raise AssertionError(output + error)
        if time.monotonic() >= deadline:
            raise AssertionError("Explicit process ownership barrier was not reached")
        time.sleep(0.01)


@pytest.mark.parametrize("divergent", [False, True])
def test_actual_two_writer_processes_have_one_callback_winner_and_zero_callback_loser(
    installed_registry, pending_store, divergent
):
    python, outside, environment = installed_registry
    case = outside / ("writers-divergent" if divergent else "writers-identical")
    case.mkdir()
    root = case / "store"
    shutil.copytree(pending_store, root)
    shutil.copyfile(pending_store.parent / "initial-head.json", case / "initial-head.json")
    base = [str(python), "-I", "-B", str(outside / "driver.py"), "transition", str(root)]
    winner = None
    loser = None
    try:
        # Open both clients before contention; the loser's first actual commit
        # must exhaust the persisted finite wait while the winner is held.
        loser = subprocess.Popen(
            base + ["contend", "loser", "divergent" if divergent else "identical"],
            cwd=outside,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for(case / "loser-ready", loser)
        winner = subprocess.Popen(
            base + ["hold-lock", "winner"],
            cwd=outside,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for(case / "lock-held", winner)
        (case / "start-contention").write_text("attempt the original intent under actual writer contention")
        wait_for(case / "loser-busy.json", loser)
        contention = json.loads((case / "loser-busy.json").read_text())
        assert contention["outcome"] in {"sqlite_busy", "ownership_timeout"}
        if contention["outcome"] == "sqlite_busy":
            assert contention["sqlite_errorcode"] == sqlite3.SQLITE_BUSY
        else:
            assert contention["sqlite_errorcode"] is None
        assert contention["lock_wait_seconds"] == 30.0
        assert contention["elapsed_seconds"] > 0.0
        assert contention["callbacks"] == []
        assert not (case / "loser-calls.log").exists()
        assert winner.poll() is None
        (case / "release-lock").write_text("release the actual SQLite owner")
        winner_output, winner_error = winner.communicate(timeout=PROCESS_OBSERVATION_SECONDS)
        assert winner.returncode == 0, winner_output + winner_error
        (case / "retry-loser").write_text("retry the same intent after acknowledged winner completion")
        loser_output, loser_error = loser.communicate(timeout=PROCESS_OBSERVATION_SECONDS)
        assert loser.returncode == 0, loser_output + loser_error
        assert json.loads(winner_output)["status"] == "new"
        loser_result = json.loads(loser_output)
        if divergent:
            assert "rejected" in loser_result
            assert "conflicts" in loser_result["rejected"]
        else:
            assert loser_result["status"] == "retained"
        assert not (case / "loser-calls.log").exists()
        assert (case / "winner-calls.log").read_text().splitlines().count("actual-transition") == 1
        current = json.loads(invoke(installed_registry, [str(outside / "driver.py"), "inspect", str(root)]))
        assert current["generation"] == 1
        assert current["U"] == 95
    finally:
        for process in (winner, loser):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
