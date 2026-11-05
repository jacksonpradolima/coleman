"""Bounded immutable artifact downloads and complete-last publication recovery."""

import base64
import hashlib
import json
import subprocess
import sys

import pytest

from coleman.application.baseline_lock import baseline_lock
from coleman.application.workflow_downloads import ArtifactChunk, artifact_page, read_artifact, report_overview
from coleman.application.workflow_store import JobRecord
from coleman.application.workflows import WorkflowService
from coleman.workflows.lifecycle import ExecutePlan, JobView
from tests.workflows.test_execution import setup_workflow, wait


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    service, preview = setup_workflow(tmp_path_factory.mktemp("download-evidence"))
    job = service.execute(ExecutePlan(schema_version=1, request_id="download", plan_id=preview.plan_id))
    wait(service, job.job_id)
    return service, job


def test_bounded_file_handles_reassemble_exact_plan_report_and_native_bytes(completed):
    service, job = completed
    overview = report_overview(service, job.job_id)
    assert overview.trial_count == 8
    assert overview.native_build_rows == 24
    first = artifact_page(service, job.job_id, limit=2)
    assert len(first.artifacts) == 2
    assert first.next_offset == 2
    descriptors = list(first.artifacts)
    offset = first.next_offset
    while offset is not None:
        page = artifact_page(service, job.job_id, offset=offset, limit=3)
        descriptors.extend(page.artifacts)
        offset = page.next_offset
    assert len(descriptors) == first.total_artifacts == 26
    assert len({item.artifact_id for item in descriptors}) == len(descriptors)
    assert not any("checkpoint" in item.name or item.name.endswith(".pkl") for item in descriptors)
    for descriptor in (*descriptors[:2], descriptors[-1]):
        offset, parts = 0, []
        while offset is not None:
            chunk = read_artifact(service, job.job_id, descriptor.artifact_id, offset=offset, maximum_bytes=65536)
            assert ArtifactChunk.from_json(chunk.to_json()) == chunk
            part = base64.b64decode(chunk.content_base64, validate=True)
            assert chunk.chunk_sha256 == "sha256:" + hashlib.sha256(part).hexdigest()
            parts.append(part)
            offset = chunk.next_offset
        payload = b"".join(parts)
        assert len(payload) == descriptor.size_bytes
        assert "sha256:" + hashlib.sha256(payload).hexdigest() == descriptor.sha256
        if descriptor.kind == "plan":
            assert json.loads(payload)["dataset"]["rows"] == 9
        elif descriptor.kind == "report":
            assert len(json.loads(payload)["physical_attempts"]) == 8
        else:
            assert payload[:4] == b"PAR1"
    assert service.status(job.job_id).attempts == 8


def test_chunk_ranges_verify_membership_once_per_service_and_preserve_expiry(completed, monkeypatch):
    import coleman.application.workflow_downloads as downloads

    original_service, job = completed
    service = WorkflowService(original_service.store.root)
    calls = []
    original = downloads._membership

    def counted(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(downloads, "_membership", counted)
    descriptor = artifact_page(service, job.job_id, limit=2).artifacts[1]
    for offset in (0, 32, 64, 96):
        chunk = read_artifact(service, job.job_id, descriptor.artifact_id, offset=offset, maximum_bytes=32)
        assert len(base64.b64decode(chunk.content_base64)) == 32
    assert calls == [job.job_id]
    with service.store.transaction() as connection:
        record = service._job(connection, job.job_id)
        expired = JobRecord.model_validate({**record.model_dump(), "state": "expired"})
        service.store.save_job(connection, expired)
    try:
        with pytest.raises(ValueError, match="completed"):
            read_artifact(service, job.job_id, descriptor.artifact_id)
    finally:
        with service.store.transaction() as connection:
            service.store.save_job(connection, record)


def test_cached_manifest_still_verifies_requested_native_bytes(completed):
    original, job = completed
    service = WorkflowService(original.store.root)
    descriptor = next(a for a in artifact_page(service, job.job_id, limit=32).artifacts if a.kind == "native_parquet")
    path = service.store.root / "jobs" / job.job_id / descriptor.name
    before = path.read_bytes()
    path.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
    try:
        with pytest.raises(ValueError, match="changed"):
            read_artifact(service, job.job_id, descriptor.artifact_id)
    finally:
        path.write_bytes(before)
    assert base64.b64decode(read_artifact(service, job.job_id, descriptor.artifact_id).content_base64) == before
    service.close()
    assert service._artifact_manifest is None


@pytest.mark.parametrize("offset,limit", [(-1, 1), (True, 1), (0, 0), (0, 65)])
def test_invalid_manifest_page_rejects_before_read(completed, offset, limit):
    service, job = completed
    with pytest.raises(ValueError):
        artifact_page(service, job.job_id, offset=offset, limit=limit)


def test_unknown_foreign_and_corrupt_chunks_reject(completed):
    service, job = completed
    first = artifact_page(service, job.job_id, limit=1).artifacts[0]
    with pytest.raises(KeyError):
        read_artifact(service, job.job_id, "missing")
    foreign_service = WorkflowService(service.store.root, owner="foreign")
    with pytest.raises(KeyError):
        read_artifact(foreign_service, job.job_id, first.artifact_id)
    with pytest.raises(ValueError):
        read_artifact(service, job.job_id, first.artifact_id, offset=first.size_bytes + 1)
    with pytest.raises(ValueError):
        read_artifact(service, job.job_id, first.artifact_id, maximum_bytes=65537)
    chunk = read_artifact(service, job.job_id, first.artifact_id, maximum_bytes=10)
    for changes in (
        {"chunk_sha256": "sha256:" + "0" * 64},
        {"next_offset": None},
        {"content_base64": ""},
        {"offset": first.size_bytes + 1},
        {"schema_version": 2},
    ):
        chunk_payload = {**chunk.model_dump(), **changes}
        with pytest.raises(ValueError):
            ArtifactChunk.model_validate(chunk_payload)


def test_versioned_lifecycle_golden():
    from pathlib import Path

    path = Path(__file__).with_name("fixtures") / "job-view-v1.json"
    view = JobView.from_json(path.read_bytes())
    assert view.to_json() + "\n" == path.read_text()
    assert view.attempts == 2
    assert view.invocations == 1
    assert view.eta_seconds.status == "unavailable"
    assert view.eta_seconds.value is None


_PUBLICATION_FAULT = """
import os,sys
from pathlib import Path
from coleman.application.workflow_worker import run_job
from coleman.artifacts import ArtifactWriter
root,job_id,mode=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
original=ArtifactWriter.write_immutable
def write(self,path,content):
    final=path==root/"jobs"/job_id/"complete.json"
    if final and mode=="before":raise OSError("Controlled local publication failure")
    result=original(self,path,content)
    if final and mode=="after":os._exit(88)
    return result
ArtifactWriter.write_immutable=write
run_job(root,job_id)
"""


@pytest.mark.parametrize("mode", ["before", "after"])
def test_interrupted_report_publication_never_issues_a_false_completed_handle(tmp_path, monkeypatch, mode):
    service, preview = setup_workflow(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(service, "_start", lambda: None)
        job = service.execute(ExecutePlan(schema_version=1, request_id="publication-fault", plan_id=preview.plan_id))
    with baseline_lock(tmp_path / "locks" / f"{job.job_id}.lock", handoff=True) as descriptor:
        with service.store.transaction() as connection:
            record = service._job(connection, job.job_id)
            service.store.save_job(
                connection, JobRecord.model_validate({**record.model_dump(), "state": "running", "invocations": 1})
            )
        process = subprocess.Popen(
            [sys.executable, "-c", _PUBLICATION_FAULT, str(tmp_path), job.job_id, mode],
            pass_fds=(descriptor,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    _, error = process.communicate(timeout=40)
    assert process.returncode == (88 if mode == "after" else 0), error.decode()
    view = service.status(job.job_id)
    assert view.state == ("interrupted" if mode == "after" else "failed")
    assert view.result_digest is None
    assert view.completed_trials == 8
    assert view.attempts == 8
    with pytest.raises(ValueError, match="completed report"):
        service.report(job.job_id)
    root = tmp_path / "jobs" / job.job_id
    before = {str(path): path.read_bytes() for path in (root / "evidence").rglob("*") if path.is_file()}
    published = (root / "complete.json").read_bytes() if mode == "after" else None
    service.resume(job.job_id)
    final = wait(service, job.job_id)
    assert final.attempts == 8
    assert final.invocations == 2
    assert all(path.read_bytes() == before[str(path)] for path in (root / "evidence").rglob("*") if path.is_file())
    if published is not None:
        assert (root / "complete.json").read_bytes() == published
    assert len(service.report(job.job_id).physical_attempts) == 8
