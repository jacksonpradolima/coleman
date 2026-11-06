"""Actual installed 37-route Environment matrix and explicit trusted checkpoint composition."""

import hashlib
import json
import os
import shutil
import subprocess

import pytest

from tests.installed_support import build_install_offline_wheel
from tests.native_semantics.support import FIXTURES, ROOT


@pytest.fixture(scope="module")
def installed_native(tmp_path_factory):
    owned = tmp_path_factory.mktemp("installed-native-semantics")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(
        UV_OFFLINE="1",
        UV_NO_INDEX="1",
        UV_LINK_MODE="copy",
        POLARS_MAX_THREADS="1",
        COLEMAN_POLICY_SEED="20261106",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    virtual = owned / "installed"
    build_install_offline_wheel(uv, owned / "dist", virtual, environment, extras=("ci",))
    outside = owned / "outside"
    outside.mkdir()
    shutil.copytree(FIXTURES, outside / "fixtures")
    shutil.copyfile(ROOT / "tests/native_semantics/installed_driver.py", outside / "driver.py")
    assert not outside.is_relative_to(ROOT)
    return virtual / "bin/python", outside, environment


def _run(installed, mode, owned, *arguments, expected=0):
    python, outside, environment = installed
    owned.mkdir(exist_ok=True)
    result = subprocess.run(
        [str(python), "-I", "-B", str(outside / "driver.py"), mode, str(outside / "fixtures"), str(owned), *arguments],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return json.loads((owned / "receipt.json").read_text()) if expected == 0 else None


def test_noneditable_ci_wheel_runs_all_37_real_native_environment_routes(installed_native):
    result = _run(installed_native, "matrix", installed_native[1] / "matrix")
    assert result is not None
    assert result["completed"] == 34
    assert result["failed"] == 3
    assert result["native_rows"] == 144
    assert len(result["records"]) == 37
    assert {row["name"] for row in result["records"] if row["single_trace_reproducibility"] == "not_qualified"} == {
        "FRRMAB"
    }


def test_installed_catalog_and_record_reader_keep_optional_native_algorithms_unimported(installed_native):
    python, outside, environment = installed_native
    source = """
import importlib.abc
import sys
from pathlib import Path

forbidden = ('numpy', 'polars', 'pyarrow', 'duckdb', 'scipy', 'coleman.policy',
             'coleman.agent', 'coleman.runner', 'coleman.environment', 'coleman.cli',
             'coleman.transports', 'mcp')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('Optional native import crossed pure boundary: ' + fullname)
sys.meta_path.insert(0, Boundary())
import coleman.native.decisions
import coleman.native.publication
from coleman.native.semantics import native_catalog
catalog = native_catalog()
assert len(catalog.policies) == 37
assert len(catalog.rewards) == 6
assert Path(coleman.native.decisions.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not any(name == part or name.startswith(part + '.') for name in sys.modules for part in forbidden)
"""
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", source],
        cwd=outside,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "fault,code,resume", [("crash-publication", 81, "resume"), ("crash-checkpoint", 82, "resume-checkpoint")]
)
def test_actual_capture_publication_and_trusted_checkpoint_resume_matches_uninterrupted_on_the_same_host(
    installed_native, fault, code, resume
):
    outside = installed_native[1]
    interrupted = outside / fault
    _run(installed_native, "checkpoint", interrupted, fault, expected=code)
    decision_paths = list((interrupted / "records").rglob("*.json"))
    assert len(decision_paths) == 2
    protected = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in decision_paths}
    marker = json.loads(next((interrupted / "checkpoints").rglob("progress_ex1.json")).read_text())
    assert marker["step_committed"] == (1 if fault == "crash-publication" else 2)
    resumed = _run(installed_native, "checkpoint", interrupted, resume)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in decision_paths} == protected
    reference = _run(installed_native, "checkpoint", outside / (fault + "-reference"), "fresh")
    assert resumed == reference
    assert resumed is not None
    assert resumed["counter"] == 4
    assert len(resumed["decisions"]) == 4
