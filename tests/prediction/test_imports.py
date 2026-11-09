"""A fresh base-only import blocks numerical, optimization and runtime engines."""

import subprocess
import sys
import sysconfig
from pathlib import Path


def test_prediction_is_independent_from_optional_backends_optimizers_and_transports():
    """Block real imports before loading every public predictor entry point."""
    root = Path(__file__).parents[2]
    code = r"""
import importlib.abc
import sys
sys.path[:0] = list(dict.fromkeys(sys.argv[1:]))
blocked = ('numpy','scipy','polars','pyarrow','mcp','coleman.application',
 'coleman.selection','coleman.allocation','coleman.replay','coleman.transports',
 'coleman.cli','coleman.state.local','coleman.runner','coleman.agent','coleman.policy')
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if any(name==prefix or name.startswith(prefix+'.') for prefix in blocked):
   raise AssertionError('Forbidden prediction import: '+name)
sys.meta_path.insert(0,Guard())
from coleman.prediction import AppendOnlyRidge,RidgeConfig,RidgeInput,RidgePayload
assert all(callable(value) for value in (AppendOnlyRidge,RidgeConfig,RidgeInput,RidgePayload))
assert not any(any(name==prefix or name.startswith(prefix+'.') for prefix in blocked) for name in sys.modules)
print('independent prediction boundary passed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(root),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "independent prediction boundary passed"
