"""Noneditable base replay, literal human guide and actual abrupt process death."""

import hashlib
import json
import os
import re
import shutil
import subprocess

import pytest

from tests.installed_support import build_install_offline_wheel
from tests.replay.support import ROOT


@pytest.fixture(scope="module")
def installed_replay(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-causal-replay")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1")
    virtual = owned / "installed"
    wheel = build_install_offline_wheel(uv, owned / "dist", virtual, environment)
    outside = owned / "outside"
    outside.mkdir()
    shutil.copyfile(ROOT / "examples/replay/ci.py", outside / "ci.py")
    shutil.copyfile(ROOT / "tests/replay/installed_driver.py", outside / "driver.py")
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/causal-replay.md").read_text(), re.DOTALL)
    assert len(blocks) == 1, "Qualify the entire unique human guide Python block"
    (outside / "guide.py").write_text(blocks[0] + "\n", encoding="utf-8")
    assert not outside.is_relative_to(ROOT)
    (owned / "installation.json").write_text(
        json.dumps({"wheel": str(wheel), "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    return virtual / "bin/python", outside, environment


def command(installed, arguments, *, expected=0):
    python, outside, environment = installed
    result = subprocess.run(
        [str(python), "-I", "-B", *arguments],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result.stdout


def test_base_installed_replay_and_copied_host_remain_cold_of_optional_stacks(installed_replay):
    script = """
import importlib.abc, importlib.metadata, json, runpy, sys
from pathlib import Path
blocked = ('numpy', 'scipy', 'polars', 'pyarrow', 'duckdb', 'mcp', 'openai',
           'anthropic', 'opentelemetry', 'boto3', 'clickhouse_connect',
           'coleman.cli', 'coleman.runner', 'coleman.application',
           'coleman.workflows', 'coleman.transports', 'coleman.artifacts',
           'coleman.telemetry', 'coleman.native', 'coleman.agent', 'coleman.policy')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Optional dependency crossed replay boundary: ' + fullname)
sys.meta_path.insert(0, Boundary())
import coleman.replay.engine as engine
from coleman.replay.codec import read_continuation
assert Path(engine.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(engine.__file__).resolve().is_relative_to(Path.cwd())
assert not {'numpy','scipy','polars','pyarrow','duckdb','mcp'} & {
    item.metadata['Name'].lower() for item in importlib.metadata.distributions()}
runpy.run_path('ci.py', run_name='__main__')
assert not any(name == item or name.startswith(item + '.') for name in sys.modules for item in blocked)
"""
    receipt = json.loads(command(installed_replay, ["-c", script]))
    assert receipt["choices"] == ["suite-a", "suite-a", "suite-b"]
    assert receipt["accepted_rounds"] == 1
    assert receipt["counts"] == [1, 0, 0]


def test_complete_literal_human_guide_runs_from_installed_base_outside_checkout(installed_replay):
    # -I deliberately excludes cwd from imports. The trusted copied script is
    # added by this host launcher, without exposing the repository source root.
    launch = (
        "import runpy,sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); runpy.run_path('guide.py')"
    )
    lines = command(installed_replay, ["-c", launch]).splitlines()
    assert lines[0] == "['suite-a', 'suite-a', 'suite-b']"
    assert lines[1].startswith("sha256:")
    document = json.loads(lines[2])
    assert document["host_state"]["decision_count"] == 3
    assert document["host_state"]["update_count"] == 1
    assert len(document["host_state"]["pending"]) == 2
    assert document["state"]["decisions"][0]["executed"]["action"]["chosen"]["id"] == "suite-a"


@pytest.mark.parametrize("fault,code,published", [("crash-before", 81, False), ("crash-after", 82, True)])
def test_actual_process_death_around_single_aggregate_publication_resumes_like_uninterrupted(
    installed_replay, fault, code, published
):
    outside = installed_replay[1]
    interrupted = outside / fault
    command(installed_replay, [str(outside / "driver.py"), fault, str(interrupted)], expected=code)
    assert (interrupted / "initial.json").is_file()
    assert (interrupted / "middle.json").exists() is published
    if not published:
        assert list(interrupted.glob("*.tmp")), "An abrupt death leaves only an uncommitted temporary"
    retained = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in interrupted.glob("*.json")}
    resumed = json.loads(command(installed_replay, [str(outside / "driver.py"), "resume", str(interrupted)]))
    reference = outside / (fault + "-reference")
    uninterrupted = json.loads(command(installed_replay, [str(outside / "driver.py"), "reference", str(reference)]))
    assert resumed == uninterrupted
    assert resumed["choices"] == ["suite-a", "suite-a", "suite-b", "suite-b", "suite-c"]
    assert resumed["decisions"] == 5
    assert resumed["updates"] == 3
    assert resumed["counts"] == [2, 1, 0]
    assert resumed["sums"] == [1.0, 0.0, 0.0]
    assert (interrupted / "final.json").read_bytes() == (reference / "final.json").read_bytes()
    assert retained == {name: hashlib.sha256((interrupted / name).read_bytes()).hexdigest() for name in retained}
