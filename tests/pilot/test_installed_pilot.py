"""Actual CI-capability wheel native TCPCI pilot from outside the checkout, including retry."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from coleman.pilot import PilotReport
from tests.installed_support import build_install_offline_wheel

ROOT = Path(__file__).parents[2]


def command(args, cwd, env):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, check=True, timeout=240)


def test_installed_native_ci_pilot_and_retry(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update({"UV_LINK_MODE": "copy", "UV_NO_PROGRESS": "1"})
    distributions = tmp_path / "dist"
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, distributions, virtual, env, extras=("ci",))
    python = virtual / "bin/python"
    outside = tmp_path / "outside"
    outside.mkdir()
    example = outside / "run.py"
    shutil.copyfile(ROOT / "examples/operational-pilot/run.py", example)
    identity = command([str(python), "-c", "import coleman; print(coleman.__file__)"], outside, env)
    assert Path(identity.stdout.strip()).is_relative_to(virtual)
    assert not outside.is_relative_to(ROOT)
    code = "import importlib.util; assert importlib.util.find_spec('mcp') is None; import coleman.pilot"
    command([str(python), "-c", code], outside, env)
    results = []
    for _ in range(2):
        execution = command([str(python), str(example), "pilot-output"], outside, env)
        result = json.loads(execution.stdout.strip().splitlines()[-1])
        assert result["status"] == "completed"
        results.append(PilotReport.from_json(Path(result["artifact"]).read_bytes()))
    assert results[0] == results[1]
    report = results[0]
    assert len(report.runs) == 8
    assert report.missing == ()
    assert all(finding.adoption == "insufficient_evidence" for finding in report.findings())
    artifacts = outside / "pilot-output"
    assert (artifacts / "report.md").is_file()
    assert len(list(artifacts.rglob("domain-records/**/*.json"))) or len(list(artifacts.rglob("*domain*.parquet")))

    documentation = (ROOT / "docs/operational-pilot.md").read_text()
    snippet = re.search(r"```python\n(.*?)```", documentation, flags=re.S)
    assert snippet is not None
    command([str(python), "-c", snippet.group(1)], outside, env)
