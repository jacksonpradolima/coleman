import os
import shutil
import subprocess

from tests.installed_support import build_install_offline_wheel


def test_noneditable_base_only_wheel_inspects_native_semantics_without_a_ci_backend(tmp_path, record):
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_LINK_MODE="copy")
    virtual = tmp_path / "installed"
    build_install_offline_wheel(uv, tmp_path / "dist", virtual, environment)
    outside = tmp_path / "outside"
    outside.mkdir()
    wire = outside / "retained.json"
    wire.write_text(record.to_json(), encoding="utf-8")
    child = """
import importlib.util, sys
from pathlib import Path
import coleman
from coleman.capabilities import CapabilityUnavailableError
from coleman.native.capture import choose_and_freeze_native
from coleman.native.decisions import native_binding
from coleman.native.publication import read_native_decision
from coleman.native.semantics import native_catalog
assert Path(coleman.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(coleman.__file__).resolve().is_relative_to(Path.cwd())
providers=('numpy','polars','pyarrow','duckdb','scipy','yaml','mcp')
assert all(importlib.util.find_spec(name) is None for name in providers)
catalog=native_catalog()
record=read_native_decision(Path('retained.json').read_bytes())
assert len(catalog.policies)==37 and len(catalog.rewards)==6
assert record.context.binding.catalog_fingerprint==catalog.fingerprint
assert record.context.binding==native_binding('EXP3','RNFail')
before=set(Path.cwd().iterdir())
try:
    choose_and_freeze_native(object(),context=record.context)
except CapabilityUnavailableError as error:
    assert '[ci]' in str(error)
else:
    raise AssertionError('missing native capability was not refused')
assert set(Path.cwd().iterdir())==before
backends=('numpy','polars','coleman.policy','coleman.agent','coleman.runner','coleman.cli','mcp')
assert not any(name in sys.modules for name in backends)
print(record.logical_identity)
"""
    result = subprocess.run(
        [str(virtual / "bin/python"), "-I", "-B", "-c", child],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == record.logical_identity
