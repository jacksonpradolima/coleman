"""Keep objective mathematics independent of experiments and optional services."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from . import ALLOWED, DEPENDENCIES, FORBIDDEN

ROOT = Path(__file__).parents[2]


def _imports(source):
    for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
        if isinstance(node, ast.Import):
            yield from ((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not node.level, f"Use auditable absolute imports: {source}:{node.lineno}"
            yield node.module or "", node.lineno


def test_objective_ast_dependencies_remain_pure_and_acyclic():
    files = sorted((ROOT / "coleman/objectives").glob("*.py"))
    assert files
    graph = {source.stem: set() for source in files}
    for source in files:
        assert source.stem in DEPENDENCIES, f"Qualify the explicit import envelope for {source}"
        for name, line in _imports(source):
            assert name.startswith(("coleman.core.", "coleman.objectives.")) or name.split(".")[0] in ALLOWED, (
                f"Outward dependency {name} at {source}:{line}"
            )
            assert not any(name == block or name.startswith(block + ".") for block in FORBIDDEN)
            if name.startswith("coleman.objectives."):
                dependency = name.split(".")[2]
                assert dependency in DEPENDENCIES[source.stem], f"Upward definition import at {source}:{line}"
                graph[source.stem].add(dependency)
    remaining = dict(graph)
    while remaining:
        leaves = {module for module, dependencies in remaining.items() if not dependencies.intersection(remaining)}
        assert leaves, f"Cyclic objective dependencies: {remaining}"
        remaining = {module: dependencies for module, dependencies in remaining.items() if module not in leaves}


def test_existing_core_does_not_depend_on_objectives():
    for source in sorted((ROOT / "coleman/core").glob("*.py")):
        for name, line in _imports(source):
            assert not name.startswith("coleman.objectives"), f"Upward dependency at {source}:{line}"


def test_cold_objective_import_cannot_initialize_runtime_or_scientific_backends(tmp_path):
    modules = [
        "coleman.objectives." + source.stem
        for source in sorted((ROOT / "coleman/objectives").glob("*.py"))
        if source.stem != "__init__"
    ]
    script = """
import importlib, importlib.abc, json, sys
forbidden = json.loads(sys.argv[1])
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('Forbidden objective dependency: ' + fullname)
sys.meta_path.insert(0, Boundary())
for name in json.loads(sys.argv[2]):
    importlib.import_module(name)
assert not any(any(name == block or name.startswith(block + '.') for block in forbidden) for name in sys.modules)
print(json.dumps({'modules': json.loads(sys.argv[2]), 'forbidden_loaded': False}))
"""
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, json.dumps(FORBIDDEN), json.dumps(modules)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert json.loads(completed.stdout) == {"modules": modules, "forbidden_loaded": False}
