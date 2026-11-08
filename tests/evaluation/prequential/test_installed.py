"""Qualify one real nine-mode CI flow and joint-state failure boundaries outside source.

The complete 252-acknowledgement example runs once per module, including an
actual process exit during result publication. Every test reuses that same
retained evidence. Individual failure/concurrency cases copy only an initialized
two-owner store, then use the original source tape to recover identical states.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[3]
# This NEW composed fixture performs 36 initialize + 216 transition ACKs.
# Its upfront bound includes slower CI durable writes; original tests retain
# their own 600s/30s budgets and the repository's unchanged whole-job limit.
FULL_COMPARISON_SECONDS = 1800
PROCESS_SECONDS = 600
EXAMPLE_FILES = ("__init__.py", "ci.py", "model.py", "profile.py", "reference.py", "session.py")


def _invoke(installed, *arguments, expected=0, timeout=PROCESS_SECONDS):
    result = subprocess.run(
        [str(installed["python"]), "-I", "-B", str(installed["outside"] / "driver.py"), *map(str, arguments)],
        cwd=installed["outside"],
        env=installed["environment"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return json.loads(result.stdout) if expected == 0 else {}


@pytest.fixture(scope="module")
def installed_prequential(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-prequential-comparison")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "COVERAGE_PROCESS_START", "COVERAGE_FILE"}
        and not key.startswith("COV_CORE_")
    }
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy", OMP_NUM_THREADS="1")
    virtual, outside = owned / "base", owned / "outside"
    wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    shutil.copyfile(ROOT / "tests/evaluation/prequential/installed_driver.py", outside / "driver.py")
    assert {path.name for path in (ROOT / "examples/prequential").glob("*.py")} == set(EXAMPLE_FILES)
    for name in EXAMPLE_FILES:
        shutil.copyfile(ROOT / "examples/prequential" / name, outside / name)
    installed = {
        "python": virtual / "bin/python",
        "outside": outside,
        "environment": environment,
        "root": outside / "comparison",
    }
    started = time.monotonic()
    _invoke(installed, "full-fault", installed["root"], expected=73, timeout=FULL_COMPARISON_SECONDS)
    assert (outside / "publication-boundary.json").is_file()
    assert (installed["root"] / "completed-evidence.json").is_file()
    assert not (installed["root"] / "completed-locator.json").exists()
    proof = _invoke(installed, "full", installed["root"])
    installed["proof"] = proof
    _save = {
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "python": str(installed["python"]),
        "outside": str(outside),
        "elapsed_seconds": time.monotonic() - started,
        "full_comparison_bound_seconds": FULL_COMPARISON_SECONDS,
        "proof": proof,
    }
    (owned / "installation.json").write_text(json.dumps(_save, sort_keys=True))
    return installed


def test_installed_actual_nine_mode_comparison_has_explicit_five_components_and_conservative_result(
    installed_prequential,
):
    proof = installed_prequential["proof"]
    summary = proof["summary"]
    assert len(summary["candidate_modes"]) == 9
    assert summary["raw_cells"] == 36 and summary["decisions_per_world"] == 3
    assert summary["conformance"] == "pass" and summary["comparison_status"] == "descriptive_only"
    assert summary["retained_fallback"] == "fixed" and summary["automatic_adoption"] is False
    assert summary["confirmatory_eligible"] is False
    assert proof["scalar_oracle_decisions"] == 108 and proof["installed_modules"] > 0
    assert len(proof["snapshots"]) == 36 and all(len(states) == 2 for states in proof["snapshots"].values())
    assert proof["reference_drift_diagnostics"] == {"action": "fail", "cost": "fail", "time": "fail"}


def test_interrupted_result_publication_recovers_original_report_and_cached_reads_leave_all_heads(
    installed_prequential,
):
    assert installed_prequential["proof"]["publication_recovered_original"]
    assert installed_prequential["proof"]["cached_all36_heads_unchanged"]
    root = installed_prequential["root"]
    assert (root / "completed-locator.json").read_bytes() == (root / "prepared-result-locator.json").read_bytes()


@pytest.fixture(scope="module")
def initial_store(installed_prequential):
    case = installed_prequential["outside"] / "initialized-cell"
    case.mkdir()
    result = _invoke(installed_prequential, "prepare", installed_prequential["root"], case)
    assert result["generation"] == 0
    return case


def _copy_initial(initial_store, case):
    case.mkdir()
    shutil.copytree(initial_store / "store", case / "store")
    shutil.copyfile(initial_store / "emit-request.json", case / "emit-request.json")
    shutil.copyfile(initial_store / "initial-head.json", case / "initial-head.json")


@pytest.mark.parametrize(
    "fault,code,acknowledged", [("coefficient", 81, False), ("domain", 82, False), ("ack", 83, True)]
)
def test_real_process_exit_at_each_composed_boundary_recovers_same_complete_original_states(
    installed_prequential, initial_store, fault, code, acknowledged
):
    installed = installed_prequential
    case = installed["outside"] / ("fault-" + fault)
    _copy_initial(initial_store, case)
    _invoke(installed, "emit", installed["root"], case, fault, expected=code)
    assert (case / "fault-barrier").is_file()
    before = _invoke(installed, "inspect", installed["root"], case)
    assert before["head"]["generation"] == int(acknowledged)
    assert before["D"] == before["domain_receipts"] == int(acknowledged)
    assert before["U"] == before["domain_observations"] == 0
    _invoke(installed, "emit", installed["root"], case, "none")
    accepted = _invoke(installed, "inspect", installed["root"], case)
    assert accepted["head"]["generation"] == 1
    assert accepted["D"] == accepted["domain_receipts"] == 1
    calls = (case / "single-calls.log").read_text().splitlines()
    assert calls.count("completed-domain") == (2 if fault == "domain" else 1)
    recovered = _invoke(installed, "finish", installed["root"], case)
    assert recovered == {"complete_two_owner_bytes_equal": True, "D": 3, "U": 3}


def _wait(path, process):
    deadline = time.monotonic() + PROCESS_SECONDS
    while not path.exists():
        if process.poll() is not None:
            output, error = process.communicate()
            raise AssertionError(output + error)
        if time.monotonic() >= deadline:
            raise AssertionError("The actual writer process did not reach its declared boundary")
        time.sleep(0.01)


@pytest.mark.parametrize("divergent", [False, True])
def test_actual_two_process_writers_have_one_domain_execution_and_retained_or_stale_loser(
    installed_prequential, initial_store, divergent
):
    installed = installed_prequential
    case = installed["outside"] / ("writers-divergent" if divergent else "writers-identical")
    _copy_initial(initial_store, case)
    base = [
        str(installed["python"]),
        "-I",
        "-B",
        str(installed["outside"] / "driver.py"),
        "emit",
        str(installed["root"]),
        str(case),
    ]
    processes = []
    try:
        loser = subprocess.Popen(
            [*base, "none", "loser", "divergent" if divergent else "identical"],
            cwd=installed["outside"],
            env=installed["environment"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(loser)
        _wait(case / "loser-ready", loser)
        winner = subprocess.Popen(
            [*base, "hold", "winner"],
            cwd=installed["outside"],
            env=installed["environment"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(winner)
        _wait(case / "writer-held", winner)
        (case / "start-loser").write_text("attempt the exact original expected head under writer ownership")
        _wait(case / "loser-attempting", loser)
        (case / "release-writer").write_text("allow the actual winner to acknowledge both owners")
        winner_output, winner_error = winner.communicate(timeout=PROCESS_SECONDS)
        loser_output, loser_error = loser.communicate(timeout=PROCESS_SECONDS)
        assert winner.returncode == 0, winner_output + winner_error
        assert loser.returncode == 0, loser_output + loser_error
        winner_result, loser_result = json.loads(winner_output), json.loads(loser_output)
        if divergent:
            assert loser_result["callbacks"] == 0 and "rejected" in loser_result
        else:
            assert loser_result == winner_result
        assert not (case / "loser-calls.log").exists()
        assert (case / "winner-calls.log").read_text().splitlines().count("completed-domain") == 1
        accepted = _invoke(installed, "inspect", installed["root"], case)
        assert accepted["head"]["generation"] == accepted["D"] == accepted["domain_receipts"] == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def test_installed_pure_contracts_stay_outside_transports_stores_and_heavy_extras(installed_prequential):
    installed = installed_prequential
    script = r"""
import importlib,importlib.abc,importlib.metadata,json,sys
from pathlib import Path
sys.path[:0]=sys.argv[1:]
blocked=('sqlite3','coleman.state.local','coleman.cli','coleman.transports','coleman.application',
         'coleman.evaluation.design.audit','mcp','numpy','pandas','polars','pyarrow','scipy')
assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules for prefix in blocked)
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==prefix or fullname.startswith(prefix+'.') for prefix in blocked):
   raise AssertionError('Undeclared dependency: '+fullname)
sys.meta_path.insert(0,Boundary())
for leaf in ('contracts','conformance','evidence','comparison','execution'):
 module=importlib.import_module('coleman.evaluation.prequential.'+leaf)
 assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not {'numpy','polars','pyarrow','scipy','mcp'} & {
 item.metadata['Name'].lower() for item in importlib.metadata.distributions()}
print('installed pure boundary passed')
"""
    # -S disables .pth startup hooks without concealing a preloaded forbidden
    # module. Mount ONLY the selected installed environment's package roots.
    query = (
        "import json,sysconfig;print(json.dumps(sorted({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')})))"
    )
    sites = subprocess.run(
        [str(installed["python"]), "-I", "-B", "-c", query],
        cwd=installed["outside"],
        env=installed["environment"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(installed["python"]), "-I", "-S", "-B", "-c", script, *json.loads(sites.stdout)],
        cwd=installed["outside"],
        env=installed["environment"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "installed pure boundary passed"
