"""CI-capability wheel installation outside the checkout, real native checkpoint reuse and runnable docs."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tests.installed_support import seed_locked_dependencies

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=240)


def mathematical_artifacts(root):
    paths = [
        path
        for family in ("decision-signals", "observation-models", "domain-records")
        for path in root.rglob(f"{family}/*.json")
    ]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_installed_base_wheel_real_native_fresh_stale_resume_and_documentation(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    distributions = tmp_path / "dist"
    command(
        [uv, "build", "--offline", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(distributions)],
        ROOT,
        env,
    )
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    wheel = next(distributions.glob("*.whl"))
    seed_locked_dependencies(uv, virtual, env, extras=("ci",))
    command(
        [uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", str(python), str(wheel) + "[ci]"],
        tmp_path,
        env,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    origin = command(
        [str(python), "-c", "import coleman.observations; print(coleman.observations.__file__)"], outside, env
    )
    assert Path(origin.stdout.strip()).resolve().is_relative_to(virtual.resolve())
    command(
        [
            str(python),
            "-c",
            "import importlib.util,sys,coleman.observations; "
            "assert importlib.util.find_spec('mcp') is None; assert 'coleman.cli' not in sys.modules; "
            "assert 'coleman.transports.mcp' not in sys.modules",
        ],
        outside,
        env,
    )
    example = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/point-in-time/ci.py", example)
    run = command([str(python), str(example), "owned-native"], outside, env)
    results = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{"case"')]
    assert {result["case"] for result in results} == {"fresh", "stale"}
    snapshots = {}
    for result in results:
        assert result["signals"] == 6
        assert result["evidence"] == "synthetic_functional_only"
        root = Path(result["artifacts_dir"])
        snapshots[result["case"]] = mathematical_artifacts(root)
        assert len(snapshots[result["case"]]) == 13
        manifest = json.loads((root / "manifest.json").read_bytes())
        assert sum(item["type"] == "decision_signal" for item in manifest["artifacts"]) == 6
        assert sum(item["type"] == "observation_model" for item in manifest["artifacts"]) == 1
    again = command([str(python), str(example), "owned-native"], outside, env)
    repeated = [json.loads(line) for line in again.stdout.splitlines() if line.startswith('{"case"')]
    assert [row["run_id"] for row in repeated] == [row["run_id"] for row in results]
    for result in repeated:
        assert mathematical_artifacts(Path(result["artifacts_dir"])) == snapshots[result["case"]]
    documentation = (ROOT / "docs/point-in-time-observations.md").read_text()
    snippet = re.search(r"```python\n(.*?)```", documentation, flags=re.S)
    assert snippet is not None
    command([str(python), "-c", snippet.group(1)], outside, env)
