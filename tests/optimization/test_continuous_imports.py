"""Cold continuous mathematics must remain independent of optional execution."""

import subprocess
import sys
import sysconfig
from pathlib import Path


def test_cold_continuous_solver_uses_only_core_objectives_stdlib_and_pydantic():
    """Run the real hand problem while heavy/runtime imports actively refuse."""
    root = Path(__file__).parents[2]
    script = """
import importlib.abc
import pathlib
import sys
sys.path[:0]=[sys.argv[1],sys.argv[2],sys.argv[3]]
blocked=("numpy","scipy","polars","pyarrow","duckdb","coleman.learning",
         "coleman.replay","coleman.state","coleman.registry","coleman.evaluation",
         "coleman.application","coleman.transports","coleman.allocation")
assert not any(name==prefix or name.startswith(prefix+".") for name in sys.modules for prefix in blocked)
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname==prefix or fullname.startswith(prefix+".") for prefix in blocked):
            raise AssertionError("forbidden continuous dependency: "+fullname)
sys.meta_path.insert(0,Refuse())
from coleman.optimization.continuous_contracts import ContinuousProblem
from coleman.optimization.continuous import solve_continuous
problem=ContinuousProblem.from_json(pathlib.Path(sys.argv[4]).read_bytes())
result=solve_continuous(problem)
assert result.status=="optimal"
assert result.executed==(0.75,0.25)
assert not any(name==prefix or name.startswith(prefix+".") for name in sys.modules for prefix in blocked)
print("cold continuous hand problem passed")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(root),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
            str(root / "tests/optimization/fixtures/continuous/v1/problem.json"),
        ],
        cwd=root.parent,
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "cold continuous hand problem passed"
