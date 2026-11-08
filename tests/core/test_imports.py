"""Enforce the new namespace's dependency direction statically and on cold import."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
FORBIDDEN = (
    "coleman.agent",
    "coleman.environment",
    "coleman.bandit",
    "coleman.scenarios",
    "coleman.policy",
    "coleman.reward",
    "coleman.evaluation",
    "coleman.observations",
    "coleman.results",
    "coleman.selection",
    "coleman.allocation",
    "coleman.runner",
    "coleman.api",
    "coleman.spec",
    "coleman.application",
    "coleman.workflows",
    "coleman.cli",
    "coleman.transports",
    "coleman.telemetry",
    "coleman.corpus",
    "mcp",
    "duckdb",
    "pyarrow",
    "polars",
    "opentelemetry",
    "clickhouse_connect",
    "numpy",
    "scipy",
    "openai",
    "anthropic",
    "boto3",
)


def test_core_ast_imports_point_only_to_leaves_or_small_required_dependencies():
    allowed = {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "operator",
        "re",
        "types",
        "typing",
        "typing_extensions",
        "zoneinfo",
        "pydantic",
    }
    assert "zoneinfo" in sys.stdlib_module_names
    for source in sorted((ROOT / "coleman/core").glob("*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert not node.level, f"Use auditable absolute leaf imports: {source}:{node.lineno}"
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.startswith("coleman.core.") or name.split(".")[0] in allowed, (
                    f"Outward dependency {name} at {source}:{node.lineno}"
                )
                assert not any(name == item or name.startswith(item + ".") for item in FORBIDDEN)


def test_cold_core_import_cannot_initialize_a_domain_runtime_or_optional_backend(tmp_path):
    modules = [
        "coleman.core." + path.stem for path in sorted((ROOT / "coleman/core").glob("*.py")) if path.stem != "__init__"
    ]
    script = """
import importlib, importlib.abc, json, sys
forbidden = json.loads(sys.argv[1])
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('Forbidden contract dependency: ' + fullname)
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
    assert json.loads(completed.stdout)["forbidden_loaded"] is False
