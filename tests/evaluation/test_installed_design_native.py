"""Exercise all public protocol examples through one noneditable installed wheel."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from coleman.evaluation.design.reports import ProtocolReport
from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def test_installed_protocol_examples_and_real_native_ci_outside_checkout(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "COVERAGE_PROCESS_START", "COVERAGE_FILE"}
        and not key.startswith("COV_CORE_")
    }
    env.update(
        UV_OFFLINE="1",
        UV_NO_INDEX="1",
        UV_NO_PROGRESS="1",
        UV_LINK_MODE="copy",
        POLARS_MAX_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, tmp_path / "wheels", virtual, env, extras=("ci",))
    python = virtual / "bin" / "python"
    external = tmp_path / "external"
    external.mkdir()
    for name in ("ci.py", "finance.py", "native_ci.py"):
        shutil.copy(ROOT / "examples/evaluation-protocol" / name, external / name)

    def execute(name):
        launcher = (
            "import pathlib,runpy,sys; "
            "root=pathlib.Path(sys.argv[1]); "
            "name=sys.argv[2]; output=sys.argv[3]; "
            "sys.path.insert(0,str(root)); "
            "sys.argv=[str(root/name),output]; "
            "runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                launcher,
                str(external),
                name,
                str(external / (name.removesuffix(".py") + "-output")),
            ],
            cwd=external,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        return json.loads(result.stdout.splitlines()[-1])

    ci = execute("ci.py")
    assert ci["raw_cells"] == 8
    assert ci["paired_units"] == 2
    assert ci["raw_mean_difference"] == 0.25
    assert ci["conclusion"] == "insufficient_evidence"
    assert ci["evidence_basis"] == "constructed_mechanism"
    assert ci["same_original_report"] is True
    assert ci["new_confirmation"] is False

    finance = execute("finance.py")
    assert finance["raw_cells"] == 8
    assert finance["paired_units"] == 4
    assert finance["raw_mean_difference"] == -0.125
    assert finance["surrogate_loss_improvement"] == 0.0625
    assert finance["conclusion"] == "insufficient_evidence"
    assert finance["evidence_basis"] == "constructed_mechanism"
    assert finance["new_confirmation"] is False

    native = execute("native_ci.py")
    assert native["raw_cells"] == 16
    assert native["paired_units"] == 1
    assert native["native_run_invocations"] == 4
    assert native["local_driver_processes"] == 1
    assert native["external_ci_jobs_claimed"] is False
    assert native["subject_test_commands_claimed"] is False
    assert native["source_dates_are_authored_fiction"] is True
    assert native["evidence_basis"] == "constructed_mechanism"
    assert native["conclusion"] == "insufficient_evidence"
    assert native["original_confirmatory_eligible"] is False
    assert native["original_producer_wire_preserved"] is True
    assert native["new_confirmation"] is False
    assert native["completed"] is True
    report_file = Path(native["report_file"])
    report = ProtocolReport.from_json(report_file.read_bytes())
    assert report_file.read_bytes() == (report.to_json() + "\n").encode()
    panel = report.panel
    assert panel.mode == "produced"
    assert panel.design.scope.prehistory == "known_exposed"
    assert panel.design.snapshot.captured_at is not None
    assert panel.design.snapshot.captured_at <= panel.selection.actual_selected_at
    assert panel.selection.actual_selected_at <= report.source_receipt.requested_at
    assert panel.design.source_plan.test_end.year == 2019
    assert panel.design.comparison_plan.test_end > report.completed_at
    assert all(member.metadata.event_at.year == 2019 for member in panel.design.members)
    assert all(row.record.available_at >= report.source_receipt.requested_at for row in panel.rows)
    assert len(panel.ledger.attempts) == 4
    assert sum(len(attempt.outputs) for attempt in panel.ledger.attempts) == 16
    assert sum(len(attempt.costs) for attempt in panel.ledger.attempts) == 4
    for attempt in panel.ledger.attempts:
        assert attempt.started_at is not None and attempt.completed_at is not None
        assert report.source_receipt.requested_at <= attempt.started_at <= attempt.completed_at <= panel.assembled_at
        cost = attempt.costs[0].record
        assert {metric.name for metric in cost.measurements} == {"local_wall_seconds", "self_cpu_seconds"}
        assert all(
            metric.unit == "s" and metric.value is not None and metric.value >= 0 for metric in cost.measurements
        )
    assert all(row.record.measurement("verdict_napfd").unit == "1" for row in panel.rows)
    session = Path(native["session"])
    executions = json.loads((session / "actual-executions.json").read_bytes())
    assert len(executions) == 4
    assert len({item["driver_pid"] for item in executions}) == 1
    assert {(item["candidate"], item["seed"]) for item in executions} == {
        (policy, seed) for policy in ("Random", "UCB") for seed in (11, 42)
    }
    assert len(tuple((session / "native").rglob("*.pkl"))) == 8
    assert len(tuple((session / "native").rglob("*.parquet"))) == 8
    assert len(tuple((session / "producer-records").glob("*.json"))) == 16

    guard = """
import importlib.abc, pathlib, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('coleman.cli', 'coleman.transports', 'mcp')):
            raise AssertionError('Domain attempted transport import: ' + fullname)
sys.meta_path.insert(0, Guard())
import coleman.evaluation.design.contracts as domain
import coleman.evaluation.design.panel
import coleman.evaluation.design.projection
assert pathlib.Path(domain.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
assert not any(name.startswith(('coleman.cli', 'coleman.transports', 'mcp')) for name in sys.modules)
storage = ('coleman.evaluation.design.audit', 'coleman.evaluation.design.reports')
assert not any(name.startswith(storage) for name in sys.modules)
"""
    subprocess.run(
        [str(python), "-I", "-c", guard],
        cwd=external,
        env=env,
        capture_output=True,
        check=True,
        timeout=30,
    )
