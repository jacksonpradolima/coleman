"""Qualify exact learning dependency envelopes without ambient site startup."""

import ast
import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BLOCKED = (
    "sqlite3",
    "numpy",
    "scipy",
    "polars",
    "pyarrow",
    "duckdb",
    "mcp",
    "openai",
    "anthropic",
    "opentelemetry",
    "clickhouse_connect",
    "coleman.cli",
    "coleman.transports",
    "coleman.application",
    "coleman.workflows",
    "coleman.runner",
    "coleman.native",
    "coleman.agent",
    "coleman.policy",
    "coleman.bandit",
    "coleman.reward",
    "coleman.telemetry",
    "coleman.results",
    "coleman.corpus",
    "coleman.evaluation",
    "coleman.registry",
    "coleman.artifacts",
    "coleman.spec",
    "coleman.state.local",
    "coleman.state.components",
    "coleman.state.storage",
)
PURE_ALLOWED = (
    "coleman.core",
    "coleman.objectives",
    "coleman.optimization",
    "coleman.learning",
    "coleman.state.contracts",
    "coleman.state.codec",
)

# -S is intentional: this tests the SDK's dependency closure, independently from
# a parent pytest/coverage site's startup imports. The caller explicitly mounts
# its selected site-packages; no sys.modules entries are removed or concealed.
COLD_SCRIPT = r"""
import importlib,importlib.abc,json,sys
from pathlib import Path
sys.path[:0]=json.loads(sys.argv[1])
modules=json.loads(sys.argv[2])
expected=Path(sys.argv[3]).resolve()
blocked=tuple(json.loads(sys.argv[4]))
preloaded=[name for name in sys.modules if any(name==p or name.startswith(p+'.') for p in blocked)]
assert not preloaded,('Unexpected startup dependency',preloaded)
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(fullname==p or fullname.startswith(p+'.') for p in blocked):
   raise AssertionError('Learning crossed its dependency envelope: '+fullname)
sys.meta_path.insert(0,Boundary())
paths={name:str(Path(importlib.import_module(name).__file__).resolve()) for name in modules}
assert all(Path(path).is_relative_to(expected) for path in paths.values()),paths
loaded=[name for name in sys.modules if any(name==p or name.startswith(p+'.') for p in blocked)]
assert not loaded,loaded
print(json.dumps({'modules':paths,'forbidden':loaded},sort_keys=True))
"""


def _matches(name, prefixes):
    return any(name == item or name.startswith(item + ".") for item in prefixes)


@pytest.mark.parametrize("source", sorted((ROOT / "coleman/learning").glob("*.py")), ids=lambda path: path.name)
def test_learning_static_imports_stay_within_explicit_pure_or_optional_replay_boundary(source):
    optional = source.stem == "replay"
    allowed = PURE_ALLOWED + (("coleman.replay",) if optional else ())
    blocked = BLOCKED + (() if optional else ("coleman.replay",))
    for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
        if isinstance(node, ast.Import):
            imported = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "Make learning dependency boundaries explicit"
            imported = (node.module or "",)
        else:
            continue
        assert not [name for name in imported if _matches(name, blocked)]
        assert not [name for name in imported if name.startswith("coleman") and not _matches(name, allowed)]


@pytest.mark.parametrize("optional_replay", [False, True], ids=["pure", "optional-replay"])
def test_learning_real_cold_imports_do_not_require_runtime_registry_storage_or_heavy_providers(
    tmp_path, optional_replay
):
    modules = [
        "coleman.learning." + path.stem
        for path in sorted((ROOT / "coleman/learning").glob("*.py"))
        if path.stem != "__init__" and (optional_replay or path.stem != "replay")
    ]
    sites = sorted({sysconfig.get_paths()[name] for name in ("purelib", "platlib")})
    paths = [str(ROOT), *sites]
    # Existing optional replay retains generic ObservationHistory/DomainRecord
    # contracts. Pure learners do not acquire that broader dependency envelope.
    blocked = tuple(name for name in BLOCKED if not optional_replay or name != "coleman.results")
    blocked += () if optional_replay else ("coleman.replay",)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            COLD_SCRIPT,
            json.dumps(paths),
            json.dumps(modules),
            str(ROOT),
            json.dumps(blocked),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["forbidden"] == []
    assert set(observed["modules"]) == set(modules)
