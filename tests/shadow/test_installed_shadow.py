"""Installed CI-capability wheel outside the checkout: real native shadow and complete API example."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from coleman.shadow import ShadowReport
from tests.installed_support import seed_locked_dependencies

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=240)


def test_installed_native_shadow_resume_and_documented_contract(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    dist = tmp_path / "dist"
    command(
        [uv, "build", "--offline", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(dist)],
        ROOT,
        env,
    )
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    seed_locked_dependencies(uv, virtual, env, extras=("ci",))
    command(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-index",
            "--no-deps",
            "--python",
            str(python),
            str(next(dist.glob("*.whl"))) + "[ci]",
        ],
        tmp_path,
        env,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    script = outside / "ci.py"
    shutil.copyfile(ROOT / "examples/prospective-shadow/ci.py", script)
    identity = command([str(python), "-c", "import coleman; print(coleman.__file__)"], outside, env)
    assert Path(identity.stdout.strip()).is_relative_to(virtual)
    assert not outside.is_relative_to(ROOT)
    code = (
        "import importlib.util,sys; assert importlib.util.find_spec('mcp') is None; "
        "import coleman.shadow; assert 'coleman.cli' not in sys.modules; "
        "assert 'coleman.transports.mcp' not in sys.modules"
    )
    command([str(python), "-c", code], outside, env)
    reports = []
    for _ in range(2):
        execution = command([str(python), str(script), "shadow-output"], outside, env)
        result = json.loads(execution.stdout.strip().splitlines()[-1])
        assert result["research"] == "inconclusive"
        assert result["forward_builds"] == 0
        reports.append(ShadowReport.from_json(Path(result["report"]).read_bytes()))
    assert reports[0].to_json() == reports[1].to_json()
    evidence = reports[0].evidence()
    assert evidence.received_logical_builds == evidence.scored_logical_builds == 2
    assert evidence.distinct_decision_days == 1
    assert evidence.actual_known_required_runner.value >= 14000
    assert evidence.actual_accounting_complete
    assert len(list((outside / "shadow-output").rglob("domain-records/*.json"))) == 2
    documentation = (ROOT / "docs/prospective-shadow.md").read_text()
    snippet = re.search(r"```python\n(.*?)```", documentation, flags=re.S)
    assert snippet is not None
    command([str(python), "-c", snippet.group(1)], outside, env)
