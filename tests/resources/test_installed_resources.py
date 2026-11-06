"""Installed CI-capability wheel, real CI resource replay, native checkpoints and executable documentation."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from coleman.resources import ResourceReplayReport

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=240)


def test_installed_native_resource_replay_and_retry(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    distributions = tmp_path / "dist"
    command([uv, "build", "--out-dir", str(distributions)], ROOT, env)
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    wheel = next(distributions.glob("*.whl"))
    command([uv, "pip", "install", "--offline", "--python", str(python), str(wheel) + "[ci]"], tmp_path, env)
    outside = tmp_path / "outside"
    outside.mkdir()
    example = outside / "run.py"
    shutil.copyfile(ROOT / "examples/resource-replay/ci.py", example)
    identity = command([str(python), "-c", "import coleman; print(coleman.__file__)"], outside, env)
    assert Path(identity.stdout.strip()).is_relative_to(virtual)
    assert not outside.is_relative_to(ROOT)
    code = (
        "import importlib.util,sys; assert importlib.util.find_spec('mcp') is None; "
        "import coleman.resources; assert 'coleman.cli' not in sys.modules; "
        "assert 'coleman.transports.mcp' not in sys.modules"
    )
    command([str(python), "-c", code], outside, env)
    results = []
    for _ in range(2):
        execution = command([str(python), str(example), "resource-output"], outside, env)
        result = json.loads(execution.stdout.strip().splitlines()[-1])
        assert result["status"] == "completed"
        results.append(ResourceReplayReport.from_json(Path(result["report"]).read_bytes()))
    assert results[0] == results[1]
    report = results[0]
    assert len(report.runs) == 4
    assert report.missing == ()
    assert all(finding.adoption == "insufficient_evidence" for finding in report.comparisons())
    artifacts = outside / "resource-output"
    assert (artifacts / "report.md").is_file()
    assert len(list(artifacts.rglob("domain-records/**/*.json"))) or len(list(artifacts.rglob("*domain*.parquet")))

    documentation = (ROOT / "docs/resource-replay.md").read_text()
    snippet = re.search(r"```python\n(.*?)```", documentation, flags=re.S)
    assert snippet is not None
    command([str(python), "-c", snippet.group(1)], outside, env)
