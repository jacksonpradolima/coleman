"""Resolve real noneditable wheels with independent capability closures."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCIENTIFIC = {"duckdb", "numpy", "polars", "pyarrow", "scipy", "scikit-posthocs", "pyyaml"}
BASE = {
    "coleman",
    "pydantic",
    "pydantic-core",
    "packaging",
    "annotated-types",
    "typing-extensions",
    "typing-inspection",
}


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    root = tmp_path_factory.mktemp("profile-wheel")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
    # Consumer resolution is intentional here. A frozen development sync alone
    # cannot prove the dependency closure advertised by a wheel to an embedder.
    environment.pop("UV_OFFLINE", None)
    result = subprocess.run(
        [uv, "build", "--wheel", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(root)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(root.glob("*.whl"))
    assert len(wheels) == 1
    return uv, wheels[0], environment


@pytest.mark.parametrize(
    ("extras", "present", "absent"),
    [
        ((), set(), SCIENTIFIC | {"mcp", "opentelemetry-sdk", "clickhouse-connect"}),
        (("ci",), {"duckdb", "numpy", "polars", "pyarrow", "pyyaml"}, {"scipy", "scikit-posthocs", "mcp"}),
        (("analytics",), SCIENTIFIC - {"pyyaml"}, {"pyyaml", "mcp"}),
        (("mcp",), {"mcp"}, SCIENTIFIC),
        (("telemetry",), {"opentelemetry-sdk"}, SCIENTIFIC | {"mcp"}),
        (("ci", "mcp", "telemetry"), {"mcp", "opentelemetry-sdk", "numpy", "pyarrow"}, {"scipy", "scikit-posthocs"}),
        (("ci", "analytics"), SCIENTIFIC, {"mcp", "opentelemetry-sdk"}),
    ],
    ids=("minimal", "ci", "analytics", "mcp-only", "telemetry-only", "ci-tools", "ci-analytics"),
)
def test_real_consumer_resolution_and_cold_imports(built_wheel, tmp_path, extras, present, absent):
    uv, wheel, environment = built_wheel
    outside = tmp_path / "outside"
    outside.mkdir()

    def command(arguments):
        result = subprocess.run(
            arguments, cwd=outside, env=environment, capture_output=True, text=True, timeout=240, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable])
    python = virtual / "bin/python"
    requirement = str(wheel) + ("[" + ",".join(extras) + "]" if extras else "")
    command([uv, "pip", "install", "--python", str(python), requirement])
    command([uv, "pip", "check", "--python", str(python)])
    script = """
import importlib, importlib.metadata as metadata, json, sys
from pathlib import Path
import coleman.core.actions as core
assert Path(core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(core.__file__).resolve().is_relative_to(Path.cwd())
for package in ('coleman.core','coleman.objectives'):
    module=importlib.import_module(package)
    for file in Path(module.__file__).parent.glob('*.py'):
        importlib.import_module(package+'.'+file.stem)
from coleman.selection import solve_coverage
from coleman.allocation import select_allocation
from coleman.observations import ObservationHistory
from coleman.evaluation.protected import TemporalPlan,BoundedMeanProtocol,holm_adjust
from coleman.capabilities import require_capability
extras=json.loads(sys.argv[1])
for extra in extras: require_capability(extra)
forbidden=('numpy','polars','pyarrow','duckdb','scipy','mcp','opentelemetry','coleman.cli','coleman.runner')
assert not any(any(n==f or n.startswith(f+'.') for f in forbidden) for n in sys.modules)
names={d.metadata['Name'].lower().replace('_','-') for d in metadata.distributions()}
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import (
    SelectedInstalledRuntime,read_runtime,capture_native_runtime,require_native_runtime)
runtime=SelectedInstalledRuntime.capture(extras=tuple(extras))
assert {row.name for row in runtime.distributions}==names
assert read_runtime(runtime.to_json())==runtime
base={row.name for row in InstalledRuntime.capture().distributions}
assert base=={
    'coleman','pydantic','pydantic-core','packaging','annotated-types','typing-extensions','typing-inspection'}
if 'ci' in extras:
    native=capture_native_runtime()
    assert native.selected_extras==('ci',)
    assert 'numpy' in {row.name for row in native.distributions}
    assert 'mcp' not in {row.name for row in native.distributions}
    require_native_runtime(native)
assert not any(any(n==f or n.startswith(f+'.') for f in forbidden) for n in sys.modules)
requires=metadata.requires('coleman')
print(json.dumps({'names':sorted(names),'requires':requires}))
"""
    inventory = json.loads(command([str(python), "-I", "-B", "-c", script, json.dumps(extras)]))
    names = set(inventory["names"])
    assert present <= names
    assert not absent & names
    if not extras:
        assert names == BASE
        command([str(python), "-I", "-m", "coleman.cli", "--help"])
        command([str(python), "-I", "-m", "coleman.transports.mcp", "--help"])
        failure = subprocess.run(
            [str(python), "-I", "-m", "coleman.cli", "run", "--config", "does-not-exist.yaml"],
            cwd=outside,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert failure.returncode != 0
        assert "[ci]" in failure.stderr
        assert "does-not-exist.yaml" not in failure.stderr
        assert not (tmp_path / "results").exists()
        # A supported host may impose a compatible transitive constraint. Exact
        # direct pins remain deliberate until a wider range is qualified.
        command([uv, "pip", "install", "--python", str(python), requirement, "typing-extensions==4.15.0"])
        command([uv, "pip", "check", "--python", str(python)])
        command([str(python), "-I", "-B", "-c", script, "[]"])
        contradiction = subprocess.run(
            [uv, "pip", "install", "--dry-run", "--python", str(python), requirement, "pydantic<2"],
            cwd=outside,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert contradiction.returncode != 0
        assert "pydantic" in contradiction.stderr
        command([uv, "pip", "check", "--python", str(python)])
