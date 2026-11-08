"""Shared resource observations preserve native worker hooks and refusal boundaries."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from coleman.application import resource_usage, worker


def test_neutral_resource_import_has_no_domain_worker_transport_or_provider_side_effects(tmp_path):
    script = """
import importlib.abc, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
blocked = ('coleman.application.worker', 'coleman.application.workflow_worker',
           'coleman.application.contracts', 'coleman.application.registry',
           'coleman.runner', 'coleman.cli', 'coleman.transports', 'coleman.telemetry',
           'numpy', 'polars', 'pyarrow', 'mcp')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Resource import crossed its boundary: ' + fullname)
sys.meta_path.insert(0, Boundary())
from coleman.application import resource_usage
assert not tuple(Path.cwd().iterdir())
assert not any(any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)
               for name in sys.modules)
print(json.dumps({'module': str(Path(resource_usage.__file__).resolve()), 'written_files': 0}))
"""
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(root)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    observation = json.loads(result.stdout)
    assert observation["module"] == str(root / "coleman/application/resource_usage.py")
    assert observation["written_files"] == 0


@pytest.mark.parametrize("metadata,expected", [("VmHWM:\t0 kB\n", 0.0), ("VmHWM:\tunknown kB\n", None)])
def test_shared_linux_sampler_keeps_observed_zero_distinct_from_unavailable(tmp_path, metadata, expected):
    source = tmp_path / "status"
    source.write_text(metadata, encoding="ascii")

    def status_path(name):
        assert name == "/proc/self/status"
        return source

    assert resource_usage.peak_rss_mib(_path_factory=status_path, _platform="linux") == expected


def test_other_platforms_reuse_telemetry_without_opening_linux_metadata(monkeypatch):
    def forbidden_path(_name):
        raise AssertionError("Non-Linux sampling must not open Linux kernel metadata")

    monkeypatch.setattr("coleman.telemetry.resources._get_peak_rss_mib", lambda: 124.0)
    assert resource_usage.peak_rss_mib(_path_factory=forbidden_path, _platform="darwin") == 124.0


@pytest.mark.parametrize("probe", [resource_usage.output_exceeds, worker._output_exceeds], ids=["shared", "legacy"])
def test_output_exact_byte_and_entry_limits_preserve_relative_worker_defaults(tmp_path, monkeypatch, probe):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "run"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "first").write_bytes(b"1234")
    (root / "second").write_bytes(b"123456")
    assert probe(10) is False
    assert probe(9) is True
    assert probe(10, root=root, maximum_entries=3) is False
    assert probe(10, root=root, maximum_entries=2) is True


@pytest.mark.parametrize("probe", [resource_usage.output_exceeds, worker._output_exceeds], ids=["shared", "legacy"])
def test_default_128_entry_limit_counts_zero_byte_files(tmp_path, probe):
    for ordinal in range(128):
        (tmp_path / str(ordinal)).touch()
    assert probe(0, root=tmp_path) is False
    (tmp_path / "one-more").touch()
    assert probe(0, root=tmp_path) is True


@pytest.mark.parametrize("directory", [False, True], ids=["file", "directory"])
def test_output_refuses_actual_symlinks_without_following_their_targets(tmp_path, directory):
    root = tmp_path / "run"
    root.mkdir()
    target = tmp_path / "outside"
    if directory:
        target.mkdir()
        (target / "not-worker-output").write_bytes(b"data")
    else:
        target.write_bytes(b"data")
    (root / "link").symlink_to(target, target_is_directory=directory)
    assert resource_usage.output_exceeds(1024, root=root) is True
    assert worker._output_exceeds(1024, root=root) is True


def test_disappearing_temporary_entry_is_skipped_after_its_slot_is_counted(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.write_bytes(b"data")
    original_lstat = Path.lstat
    inspected = []

    def disappear(path, *args, **kwargs):
        inspected.append(path)
        if path == temporary:
            temporary.unlink()
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", disappear)
    assert resource_usage.output_exceeds(0, root=tmp_path, maximum_entries=0) is True
    assert inspected == []
    assert resource_usage.output_exceeds(0, root=tmp_path, maximum_entries=1) is False
    assert inspected == [temporary]


def test_metadata_errors_propagate_to_the_unchanged_worker_fail_closed_guard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    target = root / "data"
    target.write_bytes(b"data")
    original_lstat = Path.lstat

    def denied(path, *args, **kwargs):
        if path.absolute() == target:
            raise PermissionError("Metadata observation refused")
        return original_lstat(path, *args, **kwargs)

    def stopped(code):
        raise SystemExit(code)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError, match="Metadata observation refused"):
        resource_usage.output_exceeds(1024, root=root)
    monkeypatch.setattr(worker, "_worker_peak_rss_mib", lambda: 1.0)
    monkeypatch.setattr(worker.os, "_exit", stopped)
    limits = worker.Limits()
    with pytest.raises(SystemExit, match="74"):
        worker._check_resources(limits)


def test_legacy_mode_provider_is_resolved_at_call_time_without_replacing_the_shared_walker(tmp_path, monkeypatch):
    (tmp_path / "data").write_bytes(b"data")
    modes = ModuleType("host-owned-mode-provider")
    monkeypatch.setattr(modes, "S_ISLNK", lambda _mode: False, raising=False)
    monkeypatch.setattr(modes, "S_ISREG", lambda _mode: False, raising=False)
    monkeypatch.setattr(worker, "stat", modes)
    assert worker._output_exceeds(0, root=tmp_path) is False
    assert resource_usage.output_exceeds(0, root=tmp_path) is True
