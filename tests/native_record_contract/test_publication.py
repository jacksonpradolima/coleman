import os
import subprocess
import sys
import time

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.native.decisions import FrozenNativeDecision
from coleman.native.publication import NativeDecisionConflictError, publish_native_decision, read_native_decision

CHILD = """
import os, sys, time
from pathlib import Path
from coleman.artifacts import ArtifactWriter
from coleman.native.publication import publish_native_decision, read_native_decision
record=read_native_decision(Path(sys.argv[1]).read_bytes())
mode=sys.argv[3]
original=os.link
def link(source,destination,*args,**kwargs):
    if mode=='before':
        os._exit(91)
    result=original(source,destination,*args,**kwargs)
    if mode=='after':
        os._exit(92)
    return result
os.link=link
if len(sys.argv)>4:
    ready=Path(sys.argv[4])
    ready.write_text('ready')
    deadline=time.monotonic()+20
    while not ready.with_name('go').exists():
        if time.monotonic()>deadline:
            raise TimeoutError('publication barrier timed out')
        time.sleep(.01)
try:
    result=publish_native_decision(record,ArtifactWriter(sys.argv[2]))
except ValueError:
    sys.exit(23)
print(result)
"""


def _environment():
    return {key: value for key, value in os.environ.items() if key not in {"PYTHONHOME", "PYTHONPATH"}}


def _command(tmp_path, record, *, label, mode="normal", barrier=False):
    source = tmp_path / f"{label}-retained.json"
    source.write_text(record.to_json(), encoding="utf-8")
    result = [sys.executable, "-I", "-c", CHILD, str(source), str(tmp_path / "artifacts"), mode]
    if barrier:
        result.append(str(tmp_path / f"{label}-ready"))
    return result


def _path(tmp_path, record):
    return (
        tmp_path / "artifacts" / "native-decisions" / "v1" / (record.logical_identity.removeprefix("sha256:") + ".json")
    )


def test_same_canonical_record_republication_is_idempotent_and_names_never_become_paths(tmp_path, record):
    labels = ("../CI/A", " α / test ", "")
    changed = FrozenNativeDecision.model_validate(record.model_copy(update={"candidates": labels, "ranking": labels}))
    writer = ArtifactWriter(tmp_path / "artifacts")
    path = publish_native_decision(changed, writer)
    assert path == _path(tmp_path, changed)
    assert read_native_decision(path.read_bytes()) == changed
    assert publish_native_decision(changed, writer) == path
    assert path.read_bytes() == changed.to_json().encode("utf-8")
    assert path.resolve().is_relative_to(writer.run_root.resolve())
    assert len(list(writer.run_root.rglob("*.json"))) == 1
    assert not list(writer.run_root.rglob("*.tmp"))


def test_same_logical_key_changed_complete_ranking_conflicts_without_a_second_file(tmp_path, record):
    other = FrozenNativeDecision.model_validate(record.model_copy(update={"ranking": record.ranking[::-1]}))
    writer = ArtifactWriter(tmp_path / "artifacts")
    path = publish_native_decision(record, writer)
    with pytest.raises(NativeDecisionConflictError):
        publish_native_decision(other, writer)
    assert read_native_decision(path.read_bytes()) == record
    assert len(list(writer.run_root.rglob("*.json"))) == 1
    assert not list(writer.run_root.rglob("*.tmp"))


def test_byte_limit_is_checked_before_any_artifact_side_effect(tmp_path, record):
    destination = tmp_path / "not-created"
    with pytest.raises(ValueError, match="maximum_bytes"):
        publish_native_decision(record, ArtifactWriter(destination), maximum_bytes=1)
    assert not destination.exists()


@pytest.mark.parametrize("mode,code,completed", [("before", 91, False), ("after", 92, True)])
def test_genuine_process_exit_never_exposes_partial_completed_evidence(tmp_path, record, mode, code, completed):
    command = _command(tmp_path, record, label="producer", mode=mode)
    result = subprocess.run(command, cwd=tmp_path, env=_environment(), capture_output=True, text=True, timeout=30)
    assert result.returncode == code, result.stderr
    path = _path(tmp_path, record)
    assert path.exists() is completed
    if completed:
        assert read_native_decision(path.read_bytes()) == record
    # Only retained pure evidence is republished; no choose, observe or pickle is invoked.
    assert publish_native_decision(record, ArtifactWriter(tmp_path / "artifacts")) == path
    assert read_native_decision(path.read_bytes()) == record
    assert len(list((tmp_path / "artifacts").rglob("*.json"))) == 1


@pytest.mark.parametrize("conflicting", [False, True])
def test_real_concurrent_publications_have_one_complete_immutable_winner(tmp_path, record, conflicting):
    other = FrozenNativeDecision.model_validate(record.model_copy(update={"ranking": record.ranking[::-1]}))
    commands = [
        _command(tmp_path, record, label="first", barrier=True),
        _command(tmp_path, other if conflicting else record, label="second", barrier=True),
    ]
    processes = [
        subprocess.Popen(
            command, cwd=tmp_path, env=_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for command in commands
    ]
    try:
        deadline = time.monotonic() + 20
        while not all((tmp_path / f"{label}-ready").exists() for label in ("first", "second")):
            assert time.monotonic() < deadline, "native publisher children failed to reach barrier"
            assert all(process.poll() is None for process in processes), "native publisher exited before barrier"
            time.sleep(0.01)
        (tmp_path / "go").write_text("release", encoding="utf-8")
        outputs = [process.communicate(timeout=30) for process in processes]
        assert sorted(process.returncode for process in processes) == ([0, 23] if conflicting else [0, 0]), outputs
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    files = list((tmp_path / "artifacts").rglob("*.json"))
    assert files == [_path(tmp_path, record)]
    actual = read_native_decision(files[0].read_bytes())
    assert actual in (record, other)
    assert not list((tmp_path / "artifacts").rglob("*.tmp"))


def test_new_native_evidence_does_not_rewrite_existing_unverified_csv(tmp_path, record):
    # A real legacy export retains its original unversioned result shape.
    writer = ArtifactWriter(tmp_path / "artifacts")
    legacy = writer.run_root / "legacy.csv"
    writer.write_csv_atomic(legacy, [{"step": 1, "policy": "EXP3", "reward": 0.0}])
    old_bytes = legacy.read_bytes()
    path = publish_native_decision(record, writer)
    assert legacy.read_bytes() == old_bytes
    assert path != legacy
    assert "semantic_ref" not in legacy.read_text()
    assert read_native_decision(path.read_bytes()).context.binding.policy.name == "EXP3"
