"""Installed native CI/repetition/bounded-inference workflow and source-independent import boundary."""

import json
import shutil
import subprocess
from pathlib import Path

from coleman.evaluation.protected import BoundedMeanProtocol, FamilyReport
from tests.corpus.installed_source_support import installed_source_environment
from tests.corpus.test_installed_controlled import ROOT, command


def test_installed_bounded_native_ci_and_existing_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_OFFLINE", "1")
    python, outside, env = installed_source_environment(tmp_path, extras=("ci",))
    guard = """
import importlib.abc,importlib.util,sys
from pathlib import Path
assert importlib.util.find_spec('mcp') is None
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='coleman.cli' or fullname.startswith(('coleman.transports','coleman.application','mcp')):
            raise AssertionError('Protected domain imported a transport/application: '+fullname)
sys.meta_path.insert(0,Boundary())
from coleman.evaluation.protected import BoundedMeanProtocol,FamilyReport
import coleman.evaluation.protected.bounded_inference as arithmetic
assert Path(arithmetic.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print('installed-bounded-domain')
"""
    assert command([str(python), "-I", "-c", guard], outside, env).stdout.strip() == "installed-bounded-domain"
    script = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/corpus-evaluation/ci.py", script)
    args = [str(python), "-I", str(script), "output", "--bounded"]
    first = json.loads(command(args, outside, env).stdout.splitlines()[-1])
    assert first["raw_cells"] == 16
    assert first["independent_units"] == 1
    assert first["conclusion"] == "insufficient_evidence"
    assert first["retry_confirmatory"] is False
    report = FamilyReport.from_json(Path(first["comparison"]).read_bytes())
    assert isinstance(report.frozen.protocol, BoundedMeanProtocol)
    assert report.frozen.protocol.outcome_lower == 0
    assert report.frozen.protocol.outcome_upper == 1
    assert report.results[0].p_value is None
    native = outside / "output/native"
    before = {str(path.relative_to(native)): path.read_bytes() for path in native.rglob("*") if path.is_file()}
    assert before
    second = json.loads(command(args, outside, env).stdout.splitlines()[-1])
    assert second["raw_cells"] == 16
    assert second["independent_units"] == 1
    assert second["conclusion"] == "insufficient_evidence"
    assert not FamilyReport.from_json(Path(second["comparison"]).read_bytes()).confirmatory_eligible
    assert before == {str(path.relative_to(native)): path.read_bytes() for path in native.rglob("*") if path.is_file()}
    checkpoints = list(native.rglob("progress_ex1.json"))
    assert len(checkpoints) == 8
    assert all(json.loads(path.read_bytes())["step_committed"] == 2 for path in checkpoints)
    unknown = json.loads(report.frozen.to_json())
    unknown["protocol"]["schema_version"] = 2
    path = outside / "future.json"
    path.write_text(json.dumps(unknown))
    rejected = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "from pathlib import Path;from coleman.evaluation.protected import FrozenEvaluation;"
            "FrozenEvaluation.from_json(Path('future.json').read_bytes())",
        ],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "ValidationError" in rejected.stderr
