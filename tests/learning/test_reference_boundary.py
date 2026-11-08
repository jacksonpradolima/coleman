"""A fresh process imports real pure learners without optional integration code."""

import json
import subprocess
import sys
import sysconfig
from pathlib import Path


def test_actual_reference_import_is_registry_replay_storage_and_provider_free(tmp_path):
    root = Path(__file__).parents[2]
    script = """
import importlib.abc,json,sys
sys.path[:0]=json.loads(sys.argv[1])
blocked=('sqlite3','numpy','scipy','polars','pyarrow','duckdb','openai','anthropic','mcp',
 'coleman.registry','coleman.replay','coleman.state.local','coleman.state.components',
 'coleman.artifacts','coleman.spec','coleman.cli','coleman.runner','coleman.application',
 'coleman.agent','coleman.native','coleman.policy')
preloaded=[name for name in sys.modules if any(name==p or name.startswith(p+'.') for p in blocked)]
assert not preloaded,('Unexpected startup dependencies',preloaded)
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
   raise AssertionError('Pure learning crossed dependency boundary: '+fullname)
sys.meta_path.insert(0,Boundary())
from coleman.learning.reference import Hedge,AnytimeHedge,FixedShare
from coleman.learning.contracts import ReferenceConfig,LearningLimits,read_loss_panel
loaded=[name for name in sys.modules if any(name==p or name.startswith(p+'.') for p in blocked)]
assert not loaded,('Forbidden modules after pure SDK import',loaded)
assert all(item.__module__=='coleman.learning.reference' for item in (Hedge,AnytimeHedge,FixedShare))
print('actual pure reference modules')
"""
    sites = sorted({sysconfig.get_paths()[name] for name in ("purelib", "platlib")})
    paths = [str(root), *sites]
    # -S excludes ambient coverage/provider .pth startup. Mount the selected
    # interpreter's exact dependency paths; never remove entries from sys.modules.
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, json.dumps(paths)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "actual pure reference modules"
