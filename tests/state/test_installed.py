"""Real base-wheel state/replay/store recovery, process death and concurrency."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]
# Bound process observation on slower CI hosts separately from SQLite's 30s wait.
PROCESS_OBSERVATION_SECONDS = 600


@pytest.fixture(scope="module")
def installed_state(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-owned-state")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    virtual = owned / "installed"
    wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    outside = owned / "outside"
    outside.mkdir()
    shutil.copyfile(ROOT / "examples/state/ci.py", outside / "ci.py")
    shutil.copyfile(ROOT / "tests/state/installed_driver.py", outside / "driver.py")
    shutil.copyfile(ROOT / "tests/fixtures/state/v1/ci-32-oracle.json", outside / "oracle.json")
    shutil.copytree(ROOT / "tests/fixtures/state", outside / "state-goldens")
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/state-recovery.md").read_text(), re.DOTALL)
    assert len(blocks) == 2, "Qualify both complete literal Python guide blocks"
    for number, block in enumerate(blocks):
        (outside / f"guide-{number}.py").write_text(block + "\n")
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
def pending_store(installed_state):
    outside = installed_state[1]
    root = outside / "initial32" / "store"
    result = json.loads(invoke(installed_state, [str(outside / "driver.py"), "initialize32", str(root)]))
    assert result["D"] == 95
    assert result["U"] == 48
    assert result["pending"] == 47
    assert result["loss"] == 16
    assert result["cadence"] == 42
    assert result["virtual_seconds"] == 389
    assert result["actual_seconds"] == 351
    return root


def test_base_installed_direct_state_and_owned_rng_do_not_import_durable_or_optional_stacks(installed_state):
    script = """
import importlib.abc,importlib.metadata,random,runpy,sys
from pathlib import Path
blocked=('sqlite3','numpy','scipy','polars','pyarrow','duckdb','mcp','openai','anthropic',
         'coleman.state.local','coleman.artifacts','coleman.spec','coleman.cli','coleman.runner',
         'coleman.application','coleman.native','coleman.agent','coleman.policy','coleman.telemetry')
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
   raise AssertionError('Pure owned-state dependency crossed boundary: '+fullname)
sys.meta_path.insert(0,Boundary())
from coleman.state.random_state import capture_random,restore_random
import coleman.state.codec as codec
assert Path(codec.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(codec.__file__).resolve().is_relative_to(Path.cwd())
installed={item.metadata['Name'].lower() for item in importlib.metadata.distributions()}
assert not {'numpy','scipy','polars','pyarrow','duckdb','mcp'} & installed
source=random.Random(100);state=capture_random(source);target=random.Random(7);restore_random(target,state)
assert target.getrandbits(32)==625644691
runpy.run_path('ci.py',run_name='__main__')
assert not any(name==item or name.startswith(item+'.') for name in sys.modules for item in blocked)
"""
    receipt = json.loads(invoke(installed_state, ["-c", script]))
    assert receipt["decisions"] == 3
    assert receipt["updates"] == 3
    assert receipt["virtual_seconds"] == 13


@pytest.mark.parametrize("block", [0, 1])
def test_both_literal_human_guide_blocks_use_the_real_installed_base(installed_state, block):
    launcher = (
        "import runpy,sys;from pathlib import Path;sys.path.insert(0,str(Path.cwd()));"
        f"runpy.run_path('guide-{block}.py')"
    )
    output = invoke(installed_state, ["-c", launcher])
    if block == 0:
        assert output.strip() == "ci-plan-b 3 2"
    else:
        result = json.loads(output)
        assert result["D"] == 1
        assert result["U"] == 1
        assert result["actual_seconds"] == 3
        assert result["generation"] == 1
        assert result["retry"] == "retained"


@pytest.mark.parametrize(
    ("fault", "code", "acknowledged"),
    [
        ("mid-stream", 88, False),
        ("before_blobs", 87, False),
        ("after_blob:0", 87, False),
        ("after_blob:15", 87, False),
        ("after_blob:31", 87, False),
        ("after_manifest", 87, False),
        ("after_head_update", 87, False),
        ("after_commit", 87, True),
    ],
)
def test_real_process_death_recovers_only_the_sqlite_acknowledged_complete32_collection(
    installed_state, pending_store, fault, code, acknowledged
):
    outside = installed_state[1]
    case = outside / ("fault-" + fault.replace(":", "-"))
    case.mkdir()
    root = case / "store"
    shutil.copytree(pending_store, root)
    shutil.copyfile(pending_store.parent / "initial-head.json", case / "initial-head.json")
    invoke(installed_state, [str(outside / "driver.py"), "transition", str(root), fault], expected=code)
    assert (case / "fault-barrier.json").is_file()
    before_retry = json.loads(invoke(installed_state, [str(outside / "driver.py"), "inspect", str(root)]))
    assert before_retry["generation"] == int(acknowledged)
    assert before_retry["U"] == (95 if acknowledged else 48)
    assert before_retry["pending"] == (0 if acknowledged else 47)
    if fault == "mid-stream":
        assert list(root.rglob("*.tmp")), "Actual abrupt writer death retains an unacknowledged temporary"
    recovered = json.loads(invoke(installed_state, [str(outside / "driver.py"), "transition", str(root)]))
    assert recovered["generation"] == 1
    assert recovered["D"] == 95
    assert recovered["U"] == 95
    assert recovered["pending"] == 0
    assert recovered["loss"] == 35.5
    assert recovered["cadence"] == 42
    assert recovered["actual_seconds"] == 351
    assert recovered["virtual_seconds"] == 389
    assert recovered["status"] == ("retained" if acknowledged else "new")
    diagnostic = (case / "attempt-calls.log").read_text().splitlines()
    assert diagnostic.count("actual-transition") == (1 if acknowledged else 2)
    baseline = json.loads((pending_store.parent / "observed-states.json").read_text())
    resumed = json.loads((case / "observed-states.json").read_text())
    for initial, final in zip(baseline[:-1], resumed[:-1], strict=True):
        assert final["host_state"]["rng"] == initial["host_state"]["rng"]
        assert final["host_state"]["book"] == initial["host_state"]["book"]
        assert final["host_state"]["cadence_count"] == initial["host_state"]["cadence_count"]


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
    installed_state, pending_store, divergent
):
    python, outside, environment = installed_state
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
        current = json.loads(invoke(installed_state, [str(outside / "driver.py"), "inspect", str(root)]))
        assert current["generation"] == 1
        assert current["U"] == 95
    finally:
        for process in (winner, loser):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def test_one_actual_installed_consumer_crosses4096_with_bounded_state_and_explicit_later_store_handoff(installed_state):
    outside = installed_state[1]
    root = outside / "long-one-child" / "store"
    results = []
    for milestone in (2048, 4096, 4352):
        result = json.loads(
            invoke(
                installed_state,
                [str(outside / "driver.py"), "long-run", str(root), str(milestone), "1"],
                timeout=1200,
            )
        )
        results.append(result)
        assert (root.parent / f"direct-{milestone}.json").is_file()
        if milestone < 4352:
            assert not root.exists(), "Direct compaction is not a SQLite acknowledgement"
            assert result["D"] == milestone
            assert result["U"] == milestone
            assert result["maximum_tail_records"] <= 5
            assert result["state_bytes"] < 32768
    final = results[-1]
    direct = final["direct"]
    assert direct["D"] == 4352
    assert direct["U"] == 4352
    assert direct["pending"] == 0
    assert direct["loss"] == 1632
    assert direct["cadence"] == 2176
    assert direct["count_b"] == 2270
    assert direct["switches"] == 2178
    assert direct["spent_seconds"] == 21952
    assert direct["backend_refills"] == 136
    assert direct["remaining_bits"] == 3
    assert direct["next_bits"] == [0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
    assert direct["maximum_tail_records"] <= 5
    assert direct["state_bytes"] < 32768
    assert final["generation"] == 2
    assert final["D"] == 4354
    assert final["U"] == 4354
    assert final["loss"] == 1632.25
    assert final["pending"] == 0
    assert final["owned_bytes"] < 16 * 1024**2


def test_complete_versioned_state_goldens_round_trip_using_only_the_real_installed_base(installed_state):
    script = r"""
import hashlib,json,runpy
from pathlib import Path
from typing import Literal
from coleman.core.base import CoreContract
from coleman.state.random_state import PythonRandomState
from coleman.replay.retained_codec import read_retained_continuation
ci=runpy.run_path('ci.py')
HostState=ci['HostState']
class KnownWrapper(CoreContract):
 layout:Literal['authored-complete-host-wrapper-v2']
 payload:HostState
root=Path('state-goldens')
manifest=json.loads((root/'manifest.json').read_bytes())
for item in manifest['fixtures']:
 name=item['path'];raw=(root/name).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==item['file_sha256']
 if name.endswith('python-random-cached.json'):
  state=PythonRandomState.from_json(raw)
 elif name.endswith('host-two-pending.json'):
  state=HostState.from_json(raw)
 else:
  state=read_retained_continuation(raw,state_type=KnownWrapper if 'migration' in name else HostState)
 assert state.to_json().encode()+b'\n'==raw
 assert state.fingerprint==item['model_fingerprint']
print(len(manifest['fixtures']))
"""
    assert invoke(installed_state, ["-c", script]).strip() == "5"
