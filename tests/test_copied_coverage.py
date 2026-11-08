"""Actual collector events are credited only to byte-identical copied sources."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import pytest
from coverage import Coverage, CoverageData

from tests.copied_coverage import (
    CopiedCoverage,
    finish_copied_coverage,
    prepare_copied_coverage,
    source_manifest,
)

_CASE: ContextVar[str] = ContextVar("copied_coverage_selftest_case")
_ISOLATED = "COLEMAN_COPIED_COVERAGE_SELFTEST_CHILD"


@pytest.fixture(autouse=True)
def _collector_case(request):
    """Bind the exact parametrized case; a child must have no startup collector."""
    if os.environ.get(_ISOLATED) == "1" and Coverage.current() is not None:
        raise RuntimeError("Collector self-test child must start without an inherited coverage collector")
    token = _CASE.set(request.node.nodeid)
    try:
        yield
    finally:
        _CASE.reset(token)


def _run_collector_case(node_id: str) -> None:
    """Run one real case in a bounded child, without stopping the caller's tracer.

    The 90-second outer fence includes pytest startup and the unchanged
    30-second inner child fences. A nonzero exit or skipped/non-singleton
    report fails the calling case. No real COLEMAN SDK code is measured here.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COVERAGE_", "COV_CORE_")) and key != "PYTEST_ADDOPTS"
    }
    environment[_ISOLATED] = "1"
    with tempfile.TemporaryDirectory(prefix="coleman-copied-selftest-") as temporary:
        report = Path(temporary) / "junit.xml"
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "pytest",
                node_id,
                "--no-cov",
                "-q",
                "-o",
                "addopts=",
                "-o",
                "junit_family=legacy",
                f"--junitxml={report}",
            ],
            cwd=Path(__file__).parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-24000:]
        cases = list(ET.parse(report).getroot().iter("testcase"))
        assert len(cases) == 1
        assert cases[0].attrib["classname"] == "tests.test_copied_coverage"
        assert cases[0].attrib["name"] == node_id.split("::", 1)[1]
        assert all(cases[0].find(status) is None for status in ("failure", "error", "skipped"))


def _isolated_collector[**P](function: Callable[P, None]) -> Callable[P, None]:
    """Keep collector-owning tests from pausing an existing pytest collector.

    On Python 3.14/Coverage 7.13.4, nested SysMonitor stop/resume can lose
    already-warmed code events. Scratch data basenames do not isolate that
    interpreter state. Preserve the real test body, fixtures and parametrized
    identity in a fresh process whenever a caller is already collecting.
    """

    @wraps(function)
    def isolated(*args: P.args, **kwargs: P.kwargs) -> None:
        if Coverage.current() is None:
            function(*args, **kwargs)
        else:
            _run_collector_case(_CASE.get())

    return isolated


def _project(tmp_path: Path) -> tuple[Path, Path]:
    """Author a small independent source tree, including a real source resource."""
    original = tmp_path / "project" / "coleman"
    original.mkdir(parents=True)
    (original / "__init__.py").write_text("", encoding="utf-8")
    (original / "choice.py").write_text(
        "def choose(value):\n    if value:\n        return 7\n    return -3\n", encoding="utf-8"
    )
    (original / "ColemanPytestCapture.py.source").write_text("def probe():\n    return 11\n", encoding="utf-8")
    (original / "py.typed").write_text("", encoding="utf-8")
    (original.parent / "pyproject.toml").write_text(
        '[tool.coverage.run]\nbranch = true\nsource = ["coleman"]\nparallel = true\n', encoding="utf-8"
    )
    copied = tmp_path / "isolated" / "coleman"
    shutil.copytree(original, copied)
    return original, copied


@contextmanager
def _active_parent(original: Path, *, branch: bool = True) -> Iterator[Coverage]:
    """Use a real collector with a scratch data basename, never global data."""
    parent = Coverage(
        config_file=original.parent / "pyproject.toml",
        source=[str(original)],
        data_file=original.parent / "parent-data",
        branch=branch,
    )
    parent.start()
    try:
        yield parent
    finally:
        parent.stop()


def _prepare(original: Path, copied: Path, *, branch: bool = True) -> CopiedCoverage:
    """Retain a session prepared against the actual public active collector."""
    with _active_parent(original, branch=branch):
        session = prepare_copied_coverage(original, copied, raw_parent=copied.parent)
    assert session is not None
    return session


def _raw_data(session: CopiedCoverage, filename: Path, *, branch: bool = True) -> Path:
    """Author raw event data only for refusal/merge unit cases, not SDK evidence."""
    data = CoverageData(basename=str(session.raw_basename) + ".unit")
    data.set_context("authored-unit-events")
    if branch:
        data.add_arcs({str(filename): [(-1, 1), (1, 2), (2, -1)]})
    else:
        data.add_lines({str(filename): [1, 2]})
    data.write()
    return Path(data.data_filename())


@_isolated_collector
def test_manifest_keeps_every_source_resource_and_ignores_only_copy_recipe_caches(tmp_path):
    """A non-.py resource remains bound, while existing runtime caches do not alias."""
    original, copied = _project(tmp_path)
    expected = {"__init__.py", "choice.py", "ColemanPytestCapture.py.source", "py.typed"}
    (copied / "__pycache__").mkdir()
    (copied / "__pycache__" / "choice.cpython-314.pyc").write_bytes(b"cache bytes")
    (copied / "choice.pyo").write_bytes(b"old cache bytes")
    actual = source_manifest(copied)
    assert {pin.relative for pin in actual} == expected
    assert actual == source_manifest(original)


@pytest.mark.parametrize("damage", ("changed", "missing", "additional", "symlink"))
@_isolated_collector
def test_before_execution_refuses_any_source_byte_or_inventory_difference(tmp_path, damage):
    """An identical filename alone cannot authorize coverage from different code."""
    original, copied = _project(tmp_path)
    target = copied / "choice.py"
    if damage == "changed":
        target.write_text("def choose(value):\n    return 99\n", encoding="utf-8")
    elif damage == "missing":
        target.unlink()
    elif damage == "additional":
        (copied / "unexpected.bin").write_bytes(b"new unbound resource")
    else:
        target.unlink()
        target.symlink_to(original / "choice.py")
    with _active_parent(original), pytest.raises(ValueError, match="source"):
        prepare_copied_coverage(original, copied, raw_parent=copied.parent)


@pytest.mark.parametrize("tree", ("original", "copied"))
@pytest.mark.parametrize("damage", ("changed", "missing", "additional"))
@_isolated_collector
def test_after_execution_rechecks_complete_original_and_copy_bytes(tmp_path, tree, damage):
    """Neither source tree may change after its actual measurement was prepared."""
    original, copied = _project(tmp_path)
    session = _prepare(original, copied)
    _raw_data(session, copied / "choice.py")
    directory = original if tree == "original" else copied
    target = directory / "choice.py"
    if damage == "changed":
        target.write_text("def choose(value):\n    return 99\n", encoding="utf-8")
    elif damage == "missing":
        target.unlink()
    else:
        (directory / "unexpected.bin").write_bytes(b"new unbound resource")
    with pytest.raises(ValueError, match="inventory changed"):
        finish_copied_coverage(session)
    assert not list(session.parent_basename.parent.glob(session.parent_basename.name + ".*"))


@_isolated_collector
def test_after_execution_refuses_changed_root_configuration(tmp_path):
    """A different effective project configuration cannot silently share the data."""
    original, copied = _project(tmp_path)
    session = _prepare(original, copied)
    _raw_data(session, copied / "choice.py")
    configuration = original.parent / "pyproject.toml"
    configuration.write_text(configuration.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        finish_copied_coverage(session)


@_isolated_collector
def test_unverified_foreign_measured_path_is_rejected_before_destination_creation(tmp_path):
    """A broad site/coleman alias must never credit an unrelated SDK version."""
    original, copied = _project(tmp_path)
    session = _prepare(original, copied)
    foreign = tmp_path / "foreign" / "site" / "coleman" / "choice.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text((copied / "choice.py").read_text(encoding="utf-8"), encoding="utf-8")
    raw_path = _raw_data(session, foreign)
    with pytest.raises(ValueError, match="unexpected measured source"):
        finish_copied_coverage(session)
    assert raw_path.is_file()
    assert not list(session.parent_basename.parent.glob(session.parent_basename.name + ".*"))


@_isolated_collector
def test_raw_branch_mode_must_match_actual_parent(tmp_path):
    """Statement-only data cannot be credited to a branch-mode qualification."""
    original, copied = _project(tmp_path)
    session = _prepare(original, copied)
    _raw_data(session, copied / "choice.py", branch=False)
    with pytest.raises(ValueError, match="branch mode"):
        finish_copied_coverage(session)


@_isolated_collector
def test_absent_child_data_is_a_failure(tmp_path):
    """Preparing a measurement never qualifies an unexecuted child."""
    original, copied = _project(tmp_path)
    session = _prepare(original, copied)
    with pytest.raises(ValueError, match="exactly one actual child"):
        finish_copied_coverage(session)


def _execute_actual_child(session: CopiedCoverage, selected: Path) -> None:
    """Execute both independently authored branches in the real isolated child."""
    paths = [str(selected.parent), str(Path(__file__).parents[1]), sysconfig.get_path("purelib")]
    child = f"""
import sys
from pathlib import Path
sys.path[:0] = {paths!r}
from coleman.choice import choose
assert Path(choose.__code__.co_filename) == Path({str(selected / "choice.py")!r})
from coverage import Coverage
collector = Coverage.current()
assert collector is not None
collector.switch_context("positive")
assert choose(True) == 7
collector.switch_context("negative")
assert choose(False) == -3
resource = Path({str(selected / "ColemanPytestCapture.py.source")!r})
namespace = {{}}
exec(compile(resource.read_text(), str(resource), "exec"), namespace)
assert namespace["probe"]() == 11
print("original authored outcomes preserved")
"""
    instrumented = session.instrument(child, before="from coleman.choice import choose")
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", instrumented],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "original authored outcomes preserved"


def _raw_contexts(raw: CoverageData, names: list[str]) -> dict[int, set[str]]:
    """Independently union every original raw context attached to each line."""
    contexts: dict[int, set[str]] = {}
    for name in names:
        for line, values in raw.contexts_by_lineno(name).items():
            contexts.setdefault(line, set()).update(values)
    return contexts


def _assert_mapped_events(raw: CoverageData, mapped: CoverageData, original: Path, copied: Path) -> None:
    """Compare exact event unions, including the collector's untouched entries."""
    expected: dict[str, list[str]] = {}
    for name in raw.measured_files():
        path = Path(name)
        assert path.is_relative_to(original) or path.is_relative_to(copied)
        root = original if path.is_relative_to(original) else copied
        target = str(original / path.relative_to(root))
        expected.setdefault(target, []).append(name)
    assert mapped.measured_files() == set(expected)
    for target, names in expected.items():
        assert set(mapped.lines(target) or ()) == {line for name in names for line in raw.lines(name) or ()}
        assert {line: set(values) for line, values in mapped.contexts_by_lineno(target).items()} == _raw_contexts(
            raw, names
        )
        if raw.has_arcs():
            assert set(mapped.arcs(target) or ()) == {arc for name in names for arc in raw.arcs(name) or ()}


@pytest.mark.parametrize("original_source", (False, True))
@pytest.mark.parametrize("branch", (False, True))
@_isolated_collector
def test_actual_isolated_child_preserves_exact_events_and_contexts(tmp_path, original_source, branch):
    """Real execution of unchanged files retains every line, arc and context."""
    original, copied = _project(tmp_path)
    session = _prepare(original, copied, branch=branch)
    selected = original if original_source else copied
    _execute_actual_child(session, selected)
    destination = finish_copied_coverage(session)
    raw_paths = list(session.raw_basename.parent.glob("raw-child.*"))
    assert len(raw_paths) == 1
    raw = CoverageData(basename=raw_paths[0])
    raw.read()
    mapped = CoverageData(basename=destination)
    mapped.read()
    assert mapped.has_arcs() == branch
    assert mapped.measured_contexts() == raw.measured_contexts()
    assert "positive" in mapped.measured_contexts()
    assert "negative" in mapped.measured_contexts()
    assert all(not raw.lines(name) for name in raw.measured_files() if not Path(name).is_relative_to(selected))
    _assert_mapped_events(raw, mapped, original, copied)
    assert str(original / "ColemanPytestCapture.py.source") in mapped.measured_files()
    assert destination.parent == session.parent_basename.parent
    assert destination.name.startswith(session.parent_basename.name + ".")
    metadata = json.loads((session.raw_basename.parent / "measurement.json").read_text(encoding="utf-8"))
    assert metadata["source_files"] == 4


@_isolated_collector
def test_without_active_parent_the_helper_does_no_manifest_or_database_work(tmp_path):
    """An isolated uninstrumented child keeps the original no-cov behavior."""
    paths = [str(Path(__file__).parents[1]), sysconfig.get_path("purelib")]
    absent = tmp_path / "not-created"
    child = f"""
import sys
from pathlib import Path
sys.path[:0] = {paths!r}
from tests.copied_coverage import prepare_copied_coverage
absent = Path({str(absent)!r})
assert prepare_copied_coverage(absent, absent, raw_parent=absent) is None
assert not absent.exists()
print("no measurement work")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", child],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no measurement work"
    assert not absent.exists()


@pytest.mark.parametrize("boundary", ("absent", "repeat"))
@_isolated_collector
def test_instrumentation_requires_one_exact_pre_product_boundary(tmp_path, boundary):
    """Startup cannot silently move after an unknown product import boundary."""
    original, copied = _project(tmp_path)
    session = _prepare(original, copied)
    source = "print('no product')" if boundary == "absent" else "import product\nimport product\n"
    with pytest.raises(ValueError, match="one exact child import boundary"):
        session.instrument(source, before="import product")


def _execute_single_choice(session: CopiedCoverage, value: bool) -> None:
    """Execute one actual branch, retaining a distinct child context and payload."""
    paths = [str(session.copied.parent), str(Path(__file__).parents[1]), sysconfig.get_path("purelib")]
    child = f"""
import sys
from pathlib import Path
sys.path[:0] = {paths!r}
from coleman.choice import choose
assert Path(choose.__code__.co_filename) == Path({str(session.copied / "choice.py")!r})
from coverage import Coverage
collector = Coverage.current()
assert collector is not None
collector.switch_context({"true-child" if value else "false-child"!r})
assert choose({value!r}) == {7 if value else -3}
print("one actual branch preserved")
"""
    instrumented = session.instrument(child, before="from coleman.choice import choose")
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", instrumented],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "one actual branch preserved"


def _assert_combined_events(inputs: list[CoverageData], combined: CoverageData) -> None:
    """Compare all actual combined events with an independent exact set union."""
    names = {name for data in inputs for name in data.measured_files()}
    assert combined.measured_files() == names
    assert combined.measured_contexts() == {context for data in inputs for context in data.measured_contexts()}
    assert combined.has_arcs() is True
    for name in names:
        assert set(combined.lines(name) or ()) == {line for data in inputs for line in data.lines(name) or ()}
        assert set(combined.arcs(name) or ()) == {arc for data in inputs for arc in data.arcs(name) or ()}
        contexts: dict[int, set[str]] = {}
        for data in inputs:
            for line, values in data.contexts_by_lineno(name).items():
                contexts.setdefault(line, set()).update(values)
        assert {line: set(values) for line, values in combined.contexts_by_lineno(name).items()} == contexts


@_isolated_collector
def test_two_actual_children_survive_public_combination(tmp_path):
    """Public combination must retain both distinct actual copied-child traces."""
    original, copied = _project(tmp_path)
    sessions = (_prepare(original, copied), _prepare(original, copied))
    inputs = []
    for session, value in zip(sessions, (True, False), strict=True):
        _execute_single_choice(session, value)
        destination = finish_copied_coverage(session)
        raw_paths = list(session.raw_basename.parent.glob("raw-child.*"))
        assert len(raw_paths) == 1
        raw = CoverageData(basename=raw_paths[0])
        raw.read()
        mapped = CoverageData(basename=destination)
        mapped.read()
        _assert_mapped_events(raw, mapped, original, copied)
        inputs.append(mapped)
    choice = str(original / "choice.py")
    assert inputs[0].arcs(choice) != inputs[1].arcs(choice)
    assert inputs[0].measured_contexts() != inputs[1].measured_contexts()
    combined = Coverage(
        config_file=original.parent / "pyproject.toml", data_file=sessions[0].parent_basename, branch=True
    )
    combined.combine(strict=True, keep=True)
    combined.save()
    _assert_combined_events(inputs, combined.get_data())


@_isolated_collector
def test_selftest_child_preserves_warmed_parent_branch_events(tmp_path):
    """A real collector-owning child cannot erase the caller's later branch."""
    source = tmp_path / "outer" / "choice.py"
    source.parent.mkdir()
    source.write_text("def choose(value):\n    if value:\n        return 7\n    return -3\n", encoding="utf-8")
    specification = importlib.util.spec_from_file_location("copied_coverage_outer_probe", source)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    choose = module.choose
    control = Coverage(config_file=False, source=[str(source.parent)], data_file=tmp_path / "control-data", branch=True)
    control.start()
    try:
        assert choose(True) == 7
        assert choose(False) == -3
    finally:
        control.stop()
    expected = control.get_data()
    assert set(expected.lines(str(source)) or ()) == {2, 3, 4}
    assert {(2, 3), (2, 4)} <= set(expected.arcs(str(source)) or ())
    collector = Coverage(config_file=False, source=[str(source.parent)], data_file=tmp_path / "outer-data", branch=True)
    collector.start()
    try:
        assert choose(True) == 7
        _run_collector_case("tests/test_copied_coverage.py::test_two_actual_children_survive_public_combination")
        assert choose(False) == -3
    finally:
        collector.stop()
    data = collector.get_data()
    assert set(data.lines(str(source)) or ()) == {2, 3, 4}
    assert set(data.arcs(str(source)) or ()) == set(expected.arcs(str(source)) or ())
