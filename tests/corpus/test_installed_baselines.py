"""Installed CI-capability wheel matrix outside checkout, no MCP, and transport-free domain import boundary."""

import os
import shutil
import subprocess
from pathlib import Path

from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def test_installed_comparable_matrix_outside_checkout(tmp_path):
    """Build/install offline, execute every candidate through real workers, then verify exact idempotent retry."""
    wheels = tmp_path / "wheels"
    environment = tmp_path / "installed"
    build_install_offline_wheel("uv", wheels, environment, os.environ, extras=("ci",))
    python = environment / "bin/python"
    outside = tmp_path / "outside"
    outside.mkdir()
    script = outside / "ci.py"
    shutil.copy(ROOT / "examples/comparable_baselines.py", script)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", POLARS_MAX_THREADS="2")
    boundary = """
import importlib.abc, importlib.util, sys
assert importlib.util.find_spec("mcp") is None
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("coleman.cli", "coleman.transports", "coleman.application", "mcp")):
            raise AssertionError("Forbidden domain import: " + fullname)
sys.meta_path.insert(0, Guard())
import coleman.corpus.baseline_contracts
import coleman.corpus.baseline_eligibility
import coleman.corpus.baseline_heuristics
import coleman.corpus.baseline_reference
import coleman.corpus.baseline_results
"""
    subprocess.run(
        [str(python), "-I", "-c", boundary],
        cwd=outside,
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )
    output = outside / "evidence"
    command = [str(python), "-I", str(script), str(output)]
    first = subprocess.run(command, cwd=outside, env=env, capture_output=True, text=True, timeout=120, check=True)
    assert first.stdout.count(" completed ") == 12
    assert "external_dante_prioritization incompatible" in first.stdout
    assert "development_only_insufficient_evidence" in first.stdout
    protected = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    second = subprocess.run(command, cwd=outside, env=env, capture_output=True, text=True, timeout=120, check=True)
    assert second.stdout.splitlines() == first.stdout.splitlines()[-14:]
    assert protected == {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    verify = """
import sys
from pathlib import Path
from coleman.application.baselines import read_baseline_matrix
root = next(Path(sys.argv[1]).glob("baseline-matrices/*/*/complete.json")).parent
report = read_baseline_matrix(root)
assert len(report.cells) == 13
assert all(len(cell.observations) == 3 for cell in report.cells if cell.state == "completed")
print(report.mathematical_fingerprint)
"""
    subprocess.run(
        [str(python), "-I", "-c", verify, str(output)],
        cwd=outside,
        env=env,
        check=True,
        capture_output=True,
        timeout=30,
    )
