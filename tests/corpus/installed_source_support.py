"""Real installed-wheel setup, offline source replay and native CI/checkpoint verification."""

import json
import os
import shutil
import subprocess
import sys

from tests.corpus.test_installed_controlled import ROOT, command


def installed_source_environment(tmp_path, *, extras=()):
    """Build and install the selected wheel capabilities offline outside the checkout."""
    uv = shutil.which("uv")
    assert uv is not None
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    dist = tmp_path / "dist"
    command([uv, "build", "--wheel", "--out-dir", str(dist)], ROOT, env)
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable], tmp_path, env)
    python = virtual / "bin/python"
    wheel = str(next(dist.glob("*.whl")))
    requirement = wheel + ("[" + ",".join(extras) + "]" if extras else "")
    command([uv, "pip", "install", "--offline", "--python", str(python), requirement], tmp_path, env)
    outside = tmp_path / "outside"
    outside.mkdir()
    return python, outside, env


def verify_offline_source_cli(python, outside, env, language, cache, writer, recipe):
    """Require a completed producer replay to succeed with no executable runtime available on PATH."""
    no_docker = {**env, "PATH": ""}
    cli = subprocess.run(
        [
            str(python),
            "-m",
            "coleman.transports." + language,
            "--recipe-stdin",
            "--cache",
            str(cache.root),
            "--output",
            str(writer.run_root),
            "--trial",
            "owned-installed",
        ],
        input=recipe.to_json(),
        cwd=outside,
        env=no_docker,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(cli.stdout)["status"] == "collected"


def verify_native_checkpoint(python, outside, env, tmp_path):
    """Run the actual owned native CI application twice and require identical checkpoint continuation."""
    shutil.copyfile(ROOT / "examples/controlled_faults/ci.py", outside / "ci.py")
    native = [
        json.loads(
            command([str(python), "ci.py", "native-owned", str(tmp_path / "native-cache")], outside, env)
            .stdout.strip()
            .splitlines()[-1]
        )
        for _ in range(2)
    ]
    assert native[0] == native[1]
    assert native[0]["native_rows"] == 1
    return native[0]
