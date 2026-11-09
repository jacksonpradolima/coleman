"""Adversarial complete-last boundaries with real SQL, files and accepted state.

The reused source fixture explicitly selects a declared image, records detached
startup and supplies a successful 1 MiB unit RSS sample. No installed capture or
physical-cost claim follows. Concrete codecs, Hedge, ACKs, guards and publication
transactions remain real. The separate guard child actually exits on corruption.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from coleman.application import execution_driver as driver
from coleman.application import execution_lifecycle as lifecycle
from coleman.application import execution_views as views
from coleman.application.baseline_lock import baseline_lock
from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.learning.reference import Hedge
from coleman.objectives.bindings import DefinitionBinding
from coleman.tasks.identity import CodecDeclaration
from coleman.tasks.protocols import CoordinateContext
from coleman.workflows.contracts import WorkflowError

from .test_runtime_boundaries import compile_program
from .test_service_flow import admit, changed, records, run_claimed
from .test_service_flow import service_case as service_case
from .test_service_flow import source as source
from .test_shared_driver import _publication_boundary, _PublicationValue


class _LargeOutput(CoreContract):
    """A valid test-owned output; the common host independently bounds bytes."""

    text: str


@dataclass(frozen=True)
class _LargeCodec:
    """A concrete roundtripping codec, with observed operations for refusal order."""

    calls: list[str]

    @property
    def declaration(self) -> CodecDeclaration:
        """Declare the real selected schema and maximum reader capacity."""
        schema = fingerprint(_LargeOutput.model_json_schema())
        return CodecDeclaration(
            binding=DefinitionBinding(
                reference=Reference(id="publication-large-value", version="1"), definition_fingerprint=schema
            ),
            schema_fingerprint=schema,
            maximum_bytes=16777216,
        )

    def validate(self, value, /):
        """Revalidate the actual selected concrete type without slicing its body."""
        self.calls.append("validate")
        return _LargeOutput.model_validate(value)

    def encode(self, value, /):
        """Produce the complete actual UTF-8 JSON, never a mocked byte count."""
        self.calls.append("encode")
        return value.to_json()

    def decode(self, body, /):
        """Read the actual complete concrete value when the host admits its bytes."""
        self.calls.append("decode")
        return _LargeOutput.from_json(body)


@pytest.mark.parametrize("character", ["x", "é"])
def test_complete_codec_output_above_reader_utf8_limit_refuses_before_publication(tmp_path, character):
    """Bound encoded bytes, including multibyte output shorter in characters."""
    worker, program, _ = _publication_boundary(tmp_path)
    calls = []
    codec = _LargeCodec(calls)
    value = _LargeOutput(schema_version=1, text=character * (16777216 // len(character.encode("utf-8"))))
    program = replace(program, output_codec=codec, output_fingerprint=lambda item: item.fingerprint)
    original = driver.read_job(worker.store, worker.job.job_id).to_json()
    with pytest.raises(WorkflowError, match="16 MiB reader limit"):
        worker.publish(program, value)
    assert calls == ["validate", "encode"]
    assert not program.output_path.exists()
    assert driver.read_job(worker.store, worker.job.job_id).to_json() == original


def test_cancellation_after_last_running_observation_wins_inside_real_final_transaction(tmp_path):
    """A stale successful check cannot overwrite a committed cancellation."""
    worker, program, observed = _publication_boundary(tmp_path)
    reads = []

    def read_then_cancel(store, identity):
        actual = driver.read_job(store, identity)
        reads.append(actual)
        with store.transaction() as connection:
            store.save_record(connection, views.changed(actual, state="cancelled"))
        return actual

    worker = replace(worker, hooks=replace(worker.hooks, read_job=read_then_cancel))
    program = replace(program, output_fingerprint=lambda item: item.fingerprint)
    value = _PublicationValue(schema_version=1, count=2)
    with pytest.raises(InterruptedError, match="Cancellation won before report publication"):
        worker.publish(program, value)
    assert len(reads) == 1
    assert reads[0].state == "running"
    retained = driver.read_job(worker.store, worker.job.job_id)
    assert retained.state == "cancelled"
    assert retained.result_digest is None
    assert not program.output_path.exists()
    assert set(observed) == {"validate", "encode", "decode"}


@pytest.mark.parametrize("state", ["queued", "cancelled", "expired"])
def test_public_worker_entry_does_not_load_or_publish_unclaimed_or_terminal_work(tmp_path, state):
    """The real public worker returns before a provider/owner on these states."""
    worker, program, _ = _publication_boundary(tmp_path)
    retained = views.changed(worker.job, state=state)
    with worker.store.transaction() as connection:
        worker.store.save_record(connection, retained)
    driver.run_job(worker.store.root, retained.job_id)
    assert driver.read_job(worker.store, retained.job_id).to_json() == retained.to_json()
    assert not program.output_path.exists()
    assert not (worker.store.root / "jobs").exists()


def test_reserved_but_never_acknowledged_source_coordinate_reconciles_as_virgin(service_case):
    """A real physical reservation is not a learner ACK or a nonzero minimum."""
    item = service_case
    _, queued = admit(item)
    identity = queued.job.job_id
    with baseline_lock(item.service.store.root / "locks" / f"{identity}.lock"):
        with item.service.store.transaction() as connection:
            job = lifecycle.owned_record(item.service.store, connection, identity, item.service.owner)
            job = changed(job, state="running", invocations=1)
            item.service.store.save_record(connection, job)
        program = compile_program(item, job)
        worker = driver._Worker(
            item.service.store,
            job,
            item.service.saved_plan(job.plan_id),
            time.monotonic() + 30,
            driver.WorkerHooks(peak_rss_mib=lambda: 1.0),
            None,
        )
        work = program.coordinates[0]
        worker.reserve(work)
        context = CoordinateContext(
            program.context, work.coordinate_fingerprint, work.reference, work.target, work.artifact_directory
        )
        before = records(item.service)
        worker._reconcile(program, context)
    assert records(item.service) == before
    actual = item.service.status(identity).job
    assert actual.attempts == 1
    assert actual.frontier is None
    assert not (work.artifact_directory / "state").exists()
    assert "advance" not in item.calls


def _claimed_run_with_native_hook(item, identity, hook):
    """Claim the actual source job and supply a forbidden native-only host hook."""
    with baseline_lock(item.service.store.root / "locks" / f"{identity}.lock"):
        with item.service.store.transaction() as connection:
            current = lifecycle.owned_record(item.service.store, connection, identity, item.service.owner)
            item.service.store.save_record(connection, changed(current, state="running", invocations=1))
        driver.run_with_hooks(
            item.service.store.root,
            identity,
            progress_hook=hook,
            bootstrap_profile=item.profile,
            hooks=driver.WorkerHooks(peak_rss_mib=lambda: 1.0),
        )


def test_native_build_hook_is_rejected_for_generic_work_before_attempt_or_learner_state(service_case):
    """Native build ordinals cannot be substituted for generic task frontiers."""
    item = service_case
    _, queued = admit(item)
    notifications = []
    _claimed_run_with_native_hook(item, queued.job.job_id, lambda *values: notifications.append(values))
    failed = item.service.status(queued.job.job_id).job
    assert failed.state == "failed"
    assert failed.error is not None
    assert "native build hook" in failed.error
    assert failed.attempts == 0
    assert failed.frontier is None
    assert not notifications
    assert "advance" not in item.calls
    directory = item.service.store.root / "jobs" / failed.job_id
    assert not list(directory.rglob("state"))
    assert not list(directory.rglob("attempts/*.json"))
    assert not (directory / "complete.json").exists()


def test_public_resume_missing_real_accepted_state_refuses_before_new_attempt_or_panel(service_case, monkeypatch):
    """Keep real U2 ACK/outer metadata after loss; no fresh U0 may impersonate it."""
    item = service_case
    _, queued = admit(item)
    original = driver._Worker.record_progress

    def stop_after_actual_progress(worker, program, context, progress, **kwargs):
        original(worker, program, context, progress, **kwargs)
        if progress.frontier.position == 2:
            raise InterruptedError("stop after real accepted U2")

    with monkeypatch.context() as patch:
        patch.setattr(driver._Worker, "record_progress", stop_after_actual_progress)
        run_claimed(item, queued.job.job_id)
    interrupted = item.service.status(queued.job.job_id).job
    assert interrupted.frontier is not None
    assert interrupted.frontier.position == 2
    root = item.service.store.root / "jobs" / interrupted.job_id
    coordinate = root / "evidence" / item.service.saved_plan(interrupted.plan_id).coordinates[0].fingerprint[7:]
    marker = (coordinate / "owned-definition.json").read_bytes()
    shutil.rmtree(coordinate / "state")
    assert item.service.resume(interrupted.job_id).job.state == "queued"
    panels = []
    actual_observe = Hedge.observe

    def observe_actual(owner, panel):
        panels.append(panel)
        return actual_observe(owner, panel)

    with monkeypatch.context() as patch:
        patch.setattr(Hedge, "observe", observe_actual)
        run_claimed(item, interrupted.job_id)
    failed = item.service.status(interrupted.job_id).job
    assert failed.state == "failed"
    assert failed.frontier == interrupted.frontier
    assert failed.attempts == interrupted.attempts
    assert failed.invocations == 2
    assert not panels
    assert (coordinate / "owned-definition.json").read_bytes() == marker
    assert not (coordinate / "state").exists()
    assert not (root / "complete.json").exists()


def test_real_guard_process_exits_fail_closed_on_actual_metadata_corruption(tmp_path):
    """The actual OS exit path refuses an unreadable queue instead of continuing."""
    roots = [str(Path(__file__).parents[2]), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    program = """
import sys
sys.path[:0]=ROOTS
from pathlib import Path
from threading import Event
from tests.tasks.test_shared_driver import _publication_boundary
worker, output, _ = _publication_boundary(Path(sys.argv[1]))
with worker.store.transaction() as connection:
    connection.execute('DROP TABLE jobs')
Path(sys.argv[1], 'guard-entered').write_text('actual metadata corruption')
worker.guard(Event())
raise AssertionError('guard returned after actual metadata corruption')
""".replace("ROOTS", repr(roots))
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 74, completed.stderr
    assert (tmp_path / "guard-entered").read_text(encoding="utf-8") == "actual metadata corruption"
    assert not list(tmp_path.rglob("complete.json"))
