"""Independent authored report boundaries, never synthetic timing evidence."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from coleman.spec.runtime_identity import InstalledRuntime
from coleman.tasks.contracts import TaskJobRecord, TaskLimits
from coleman.tasks.identity import ExecutorImplementationIdentity, HostBootstrapProfile
from coleman.tasks.requests import TaskExecuteRequest, TaskJobView

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("damage", ("partial", "populated", "changed_cache", "changed_runtime", "future", "nonfinite"))
def test_bootstrap_reader_requires_both_actual_cache_profiles_and_same_runtime(bootstrap_measurement, damage):
    values = json.loads((Path(__file__).with_name("fixtures") / "bootstrap-v1.json").read_text())
    if damage == "partial":
        values["cells"] = values["cells"][:1]
    elif damage == "populated":
        values["cells"][0]["cache_before_files"] = 1
    elif damage == "changed_cache":
        values["cells"][1]["cache_before_bytes"] += 1
    elif damage == "changed_runtime":
        values["cells"][1]["runtime"]["python_version"] = "3.14.1"
    elif damage == "future":
        values["schema_version"] = 2
    else:
        values["cells"][0]["elapsed_seconds"] = float("inf")
    with pytest.raises(ValueError):
        bootstrap_measurement.BootstrapReport.model_validate(values)


def test_independent_bootstrap_golden_round_trip_keeps_zero_null_and_unknown_versions(bootstrap_measurement):
    source = (Path(__file__).with_name("fixtures") / "bootstrap-v1.json").read_bytes()
    report = bootstrap_measurement.BootstrapReport.from_json(source)
    assert bootstrap_measurement.BootstrapReport.from_json(report.to_json()).to_json() == report.to_json()
    assert report.cells[0].host_cpu_seconds == 0.0
    assert report.cells[0].host_peak_rss_mib is None
    assert report.cells[0].runtime.to_json() == (ROOT / "tests/spec/fixtures/runtime-v1.json").read_text().strip()


def hand_task_report(module):
    """Compose independent existing image/job fixtures and literal finite work."""
    fixture = ROOT / "tests/fixtures/tasks/v1"
    image = ExecutorImplementationIdentity.from_json((fixture / "implementation.json").read_bytes())
    profile = HostBootstrapProfile.from_json((fixture / "bootstrap.json").read_bytes())
    runtime = InstalledRuntime.model_validate(
        {
            "schema_version": 1,
            **{
                key: getattr(image.runtime, key)
                for key in (
                    "package_fingerprint",
                    "python_implementation",
                    "python_version",
                    "operating_system",
                    "machine",
                    "distributions",
                )
            },
        }
    )
    original_job = json.loads((fixture / "job.json").read_text())
    cells = []
    for cell_index, (batch, size) in enumerate(((1, "small"), (3, "small"), (1, "larger"), (3, "larger"))):
        coordinates, updates = (1, 1) if size == "small" else (4, 32)
        jobs = []
        for index in range(batch):
            job = TaskJobRecord.model_validate(
                {
                    **original_job,
                    "job_id": f"unit-case-{cell_index}-job-{index}",
                    "request_id": f"unit-request-{cell_index}-{index}",
                    "request_digest": "sha256:" + f"{cell_index * 3 + index:064x}",
                    "plan_id": ("2" if size == "small" else "3") * 64,
                    "state": "completed",
                    "completed_coordinates": coordinates,
                    "total_coordinates": coordinates,
                    "attempts": coordinates,
                    "invocations": 1,
                    "result_digest": "sha256:" + "1" * 64,
                }
            )
            before = TaskJobView(
                schema_version=1,
                contract="coleman-tasks-v1",
                artifact_type="job_view",
                job=job,
                worker_active=False,
                elapsed_seconds=0.0,
            )
            jobs.append(
                module.JobObservation(
                    schema_version=1,
                    job=job,
                    before_cleanup=before,
                    worker_active=False,
                    planned_updates=updates,
                    accepted_updates=updates,
                    acknowledged_checkpoint_bytes=50,
                    committed_state_digest="sha256:" + "4" * 64,
                    complete_state_status="verified",
                    mathematical_digest="sha256:" + ("2" if size == "small" else "3") * 64,
                    age_seconds=0.0,
                    result_bytes=100,
                    final_checkpoint_representation_bytes=50,
                    artifact_files=2,
                    artifact_bytes=200,
                )
            )
        cells.append(
            module.TaskCell(
                schema_version=1,
                complete=True,
                disposition="completed",
                observation_status="observed",
                batch=batch,
                size=size,
                planned_updates=batch * updates,
                input_bytes=100,
                max_queued_views_per_sweep=1,
                worker_owner_observed=False,
                elapsed_seconds=0.0,
                observation_elapsed_seconds=0.0,
                observation_limit_seconds=60.0,
                host_cpu_seconds=0.0,
                host_peak_rss_mib=None,
                worker_cpu_seconds=None,
                worker_peak_rss_mib=None,
                worker_resource_status="unavailable",
                task_limits=TaskLimits(
                    max_coordinates=coordinates,
                    max_work_units=updates * 2,
                    max_attempts=2,
                    max_invocations=3,
                    attempt_seconds=60.0,
                    max_output_bytes=4 * 1024**2,
                ),
                jobs=tuple(jobs),
            )
        )
    return module.TaskLoadReport(
        schema_version=1,
        complete=True,
        outcome="completed",
        planned_updates=132,
        outer_limit_seconds=180.0,
        entry_reserve_seconds=30.0,
        acknowledged_update_lower_bound=132,
        runtime=runtime,
        bootstrap=profile,
        implementation=image,
        hardware="authored unit fixture, not measured performance",
        native_threads=(),
        host_policy={"schema_version": 1, "max_running": 1, "max_queued": 3},
        driver_sha256="0" * 64,
        example_sha256="1" * 64,
        helper_sources=(("task_inputs.py", "2" * 64), ("task_records.py", "3" * 64)),
        queue_evidence="observed",
        cells=tuple(cells),
    )


def limited_task_report(module):
    """Declare one completed, one partially ACKed and one queued/cancelled job."""
    values = hand_task_report(module).model_dump(mode="json")
    values["complete"], values["outcome"] = False, "limited"
    last = values["cells"][-1]
    last["complete"], last["disposition"] = False, "observation_deadline"
    last["observation_elapsed_seconds"] = 60.0
    for index, completed in ((1, 2), (2, 0)):
        row = last["jobs"][index]
        row["job"].update(
            state="cancelled",
            result_digest=None,
            completed_coordinates=completed,
            attempts=completed,
            invocations=1 if completed else 0,
        )
        row["before_cleanup"]["job"] = {**row["job"], "state": "running" if completed else "queued"}
        row["before_cleanup"]["worker_active"] = bool(completed)
        row.update(
            accepted_updates=18 if completed else 0,
            acknowledged_checkpoint_bytes=50 if completed else 0,
            committed_state_digest="sha256:" + "5" * 64 if completed else None,
            complete_state_status="not_applicable",
            mathematical_digest=None,
            result_bytes=None,
            final_checkpoint_representation_bytes=None,
            artifact_files=2 if completed else 0,
            artifact_bytes=200 if completed else 0,
        )
    values["acknowledged_update_lower_bound"] = 86  # 4small +32large +32completed +18partial.
    return module.TaskLoadReport.model_validate(values)


@pytest.mark.parametrize(
    "damage", ("missing", "duplicate", "changed_math", "false_queue", "worker_resource", "work", "future")
)
def test_task_load_reader_rejects_false_completion_or_resource_interpretation(task_measurement, damage):
    values = hand_task_report(task_measurement).model_dump(mode="json")
    if damage == "missing":
        values["cells"].pop()
    elif damage == "duplicate":
        values["cells"][1]["jobs"][1] = values["cells"][1]["jobs"][0]
    elif damage == "changed_math":
        values["cells"][1]["jobs"][0]["mathematical_digest"] = "sha256:" + "f" * 64
    elif damage == "false_queue":
        values["cells"][1]["max_queued_views_per_sweep"] = 0
    elif damage == "worker_resource":
        values["cells"][0]["worker_peak_rss_mib"] = 0.0
    elif damage == "work":
        values["cells"][0]["jobs"][0]["accepted_updates"] = 2
    else:
        values["schema_version"] = 2
    with pytest.raises(ValueError):
        task_measurement.TaskLoadReport.from_json(json.dumps(values))


def test_task_report_round_trip_and_honest_insufficient_queue_outcome(task_measurement):
    report = hand_task_report(task_measurement)
    assert task_measurement.TaskLoadReport.from_json(report.to_json()).to_json() == report.to_json()
    values = report.model_dump(mode="json")
    values["cells"][1]["max_queued_views_per_sweep"] = 0
    values["queue_evidence"] = "insufficient"
    insufficient = task_measurement.TaskLoadReport.from_json(json.dumps(values))
    assert insufficient.complete is True  # Every task completed; queue evidence remains inconclusive.
    assert insufficient.queue_evidence == "insufficient"


def test_complete_state_digest_uses_json_array_and_retains_owner_order(task_measurement):
    states = ("independent-state-a", "independent-state-b")
    expected = "sha256:" + hashlib.sha256(b'["independent-state-a","independent-state-b"]').hexdigest()
    assert task_measurement._state_digest(states) == expected
    assert task_measurement._state_digest(tuple(states)) == expected
    assert task_measurement._state_digest(states[::-1]) != expected


def test_limited_reader_keeps_actual_partial_ack_work_and_nondelivery_distinct(task_measurement):
    report = limited_task_report(task_measurement)
    assert report.complete is False
    assert report.outcome == "limited"
    assert report.planned_updates == 132
    assert report.acknowledged_update_lower_bound == 86
    assert task_measurement.TaskLoadReport.from_json(report.to_json()).to_json() == report.to_json()
    partial, unstarted = report.cells[-1].jobs[1:]
    assert partial.accepted_updates == 18  # ActualACK work can exceed completed coordinate prefixes.
    assert partial.before_cleanup.job.state == "running"
    assert partial.job.state == "cancelled"
    assert partial.mathematical_digest is None
    assert partial.complete_state_status == "not_applicable"
    assert unstarted.before_cleanup.job.state == "queued"
    assert unstarted.job.invocations == unstarted.accepted_updates == unstarted.artifact_bytes == 0
    assert unstarted.result_bytes is None  # N/A complete result is not an observed zero-byte result.
    assert all(not row.worker_active for cell in report.cells for row in cell.jobs)


@pytest.mark.parametrize(
    "damage", ("false_complete", "missing_job", "active_owner", "false_work", "fake_result", "bug")
)
def test_limited_reader_refuses_false_completion_or_software_error_relabeling(task_measurement, damage):
    values = limited_task_report(task_measurement).model_dump(mode="json")
    row = values["cells"][-1]["jobs"][1]
    if damage == "false_complete":
        values["complete"], values["outcome"] = True, "completed"
    elif damage == "missing_job":
        values["cells"][-1]["jobs"].pop()
    elif damage == "active_owner":
        row["worker_active"] = True
    elif damage == "false_work":
        values["acknowledged_update_lower_bound"] += 1
    elif damage == "fake_result":
        row["mathematical_digest"] = "sha256:" + "8" * 64
    else:
        row["job"].update(state="failed", error="task_execution_error:TypeError")
    with pytest.raises(ValueError):
        task_measurement.TaskLoadReport.from_json(json.dumps(values))


def test_notstarted_retains_planned_grid_and_explicitly_unavailable_measurements(task_measurement):
    unavailable = task_measurement._not_started((3, "larger"), 29.0)
    assert unavailable.planned_updates == 96
    assert unavailable.jobs == ()
    assert unavailable.observation_status == "unavailable"
    assert unavailable.input_bytes is unavailable.elapsed_seconds is unavailable.host_cpu_seconds is None
    values = hand_task_report(task_measurement).model_dump(mode="json")
    values.update(complete=False, outcome="limited", acknowledged_update_lower_bound=36, queue_evidence="insufficient")
    values["cells"][-1] = unavailable.model_dump(mode="json")
    report = task_measurement.TaskLoadReport.from_json(json.dumps(values))
    assert report.planned_updates == 132
    assert report.complete is False
    assert report.acknowledged_update_lower_bound == 36
    damaged = unavailable.model_dump(mode="json")
    damaged["host_cpu_seconds"] = 0.0
    with pytest.raises(ValueError):
        task_measurement.NotStartedCell.model_validate(damaged)
    with pytest.raises(ValueError):
        task_measurement._not_started((3, "larger"), 90.0)


@pytest.mark.parametrize("reverse", (False, True))
def test_only_actual_known_resource_failures_can_be_classified_as_limited(task_measurement, reverse):
    template = limited_task_report(task_measurement).cells[-1].jobs[1].before_cleanup.model_dump(mode="json")
    template["job"].update(state="failed", error="sampled_resource_limit")
    resource = TaskJobView.model_validate(template)
    assert task_measurement._disposition((resource,)) == "resource_limit"
    template["job"]["error"] = "task_execution_error:TypeError"
    unexpected = TaskJobView.model_validate(template)
    pair = (unexpected, resource) if reverse else (resource, unexpected)
    with pytest.raises(RuntimeError):
        task_measurement._disposition(pair)


def test_real_limited_publication_is_idempotent_and_never_creates_complete(task_measurement, tmp_path):
    report = limited_task_report(task_measurement)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: task_measurement._publish(report, tmp_path), range(4)))
    assert len(set(paths)) == 1
    assert paths[0].name == "limited.json"
    assert not (tmp_path / "complete.json").exists()
    assert task_measurement.TaskLoadReport.from_json(paths[0].read_bytes()).complete is False
    forged = report.model_copy(update={"complete": True, "outcome": "completed"})
    with pytest.raises(ValueError):
        task_measurement._publish(forged, tmp_path)
    assert not (tmp_path / "complete.json").exists()


def test_limited_write_interruption_cannot_publish_a_false_completed_artifact(task_measurement, tmp_path, monkeypatch):
    report = limited_task_report(task_measurement)

    def interruption(*_args, **_kwargs):
        raise InterruptedError("controlled incomplete measurement publication")

    with monkeypatch.context() as patch:
        patch.setattr("coleman.artifacts.os.link", interruption)
        with pytest.raises(InterruptedError):
            task_measurement._publish(report, tmp_path)
    assert not (tmp_path / "complete.json").exists()
    assert not (tmp_path / "limited.json").exists()
    assert task_measurement._publish(report, tmp_path).name == "limited.json"


def test_completed_during_cleanup_cannot_retroactively_pass_the_observation_window(task_measurement, tmp_path):
    values = limited_task_report(task_measurement).model_dump(mode="json")
    completed = hand_task_report(task_measurement).model_dump(mode="json")
    for index in (1, 2):
        before = values["cells"][-1]["jobs"][index]["before_cleanup"]
        values["cells"][-1]["jobs"][index] = completed["cells"][-1]["jobs"][index]
        values["cells"][-1]["jobs"][index]["before_cleanup"] = before
    values["acknowledged_update_lower_bound"] = 132
    report = task_measurement.TaskLoadReport.model_validate(values)
    assert all(row.job.state == "completed" for cell in report.cells for row in cell.jobs)
    assert report.complete is False
    assert task_measurement._publish(report, tmp_path).name == "limited.json"
    assert not (tmp_path / "complete.json").exists()
    values["cells"][-1].update(complete=True, disposition="completed")
    values.update(complete=True, outcome="completed")
    with pytest.raises(ValueError, match="observation"):
        task_measurement.TaskLoadReport.model_validate(values)


def test_status_read_crossing_deadline_keeps_last_eligible_view(task_measurement, monkeypatch):
    """Unit clock/read boundary only; the installed flow uses real TaskService."""
    completed = hand_task_report(task_measurement).cells[0].jobs[0].before_cleanup
    values = completed.model_dump(mode="json")
    values["job"].update(state="running", completed_coordinates=0, result_digest=None)
    values["worker_active"] = True
    running = TaskJobView.model_validate(values)
    known = {running.job.job_id: running}
    clock = iter((0.0, 0.0, 60.0, 60.0, 60.0))
    reads = []

    def late_status(*_args):
        reads.append("actual-unit-read-boundary")
        return (completed,)

    with monkeypatch.context() as patch:
        patch.setattr(task_measurement, "_views", late_status)
        patch.setattr(task_measurement.time, "monotonic", lambda: next(clock))
        window = task_measurement._window(
            SimpleNamespace(service=object(), deadline=180.0), [running.job.job_id], known, (1, True)
        )
    assert window.disposition == "observation_deadline"
    assert reads == ["actual-unit-read-boundary"]
    assert window.seconds == 60.0
    assert window.before == (running,)
    assert known == {running.job.job_id: running}


@pytest.mark.parametrize("damage", ("missing", "duplicate", "order", "non_hex"))
def test_task_report_binds_every_copied_helper_identity(task_measurement, damage):
    values = hand_task_report(task_measurement).model_dump(mode="json")
    sources = values["helper_sources"]
    if damage == "missing":
        sources.pop()
    elif damage == "duplicate":
        sources[1] = sources[0]
    elif damage == "order":
        sources.reverse()
    else:
        sources[0][1] = "x" * 64
    with pytest.raises(ValueError):
        task_measurement.TaskLoadReport.model_validate(values)


def contention(code):
    """Create only a unit metadata exception, never fake physical-task evidence."""
    error = sqlite3.OperationalError("unit controlled metadata contention")
    error.sqlite_errorcode = code
    return error


@pytest.mark.parametrize("code", (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_BUSY | (1 << 8)))
def test_metadata_retry_accepts_only_primary_or_extended_contention(task_measurement, monkeypatch, code):
    calls = []
    marker = object()

    def same_operation():
        calls.append("same-retained-operation")
        if len(calls) == 1:
            raise contention(code)
        return marker

    with monkeypatch.context() as patch:
        patch.setattr(task_measurement.time, "monotonic", lambda: 1.0)
        patch.setattr(task_measurement.time, "sleep", lambda _seconds: None)
        assert task_measurement._retry_metadata(SimpleNamespace(deadline=180.0), same_operation) is marker
    assert calls == ["same-retained-operation"] * 2


@pytest.mark.parametrize(
    "error", (contention(sqlite3.SQLITE_ERROR), sqlite3.OperationalError("no numeric code"), TypeError("defect"))
)
def test_unexpected_metadata_faults_propagate_without_retries(task_measurement, error):
    calls = []

    def operation():
        calls.append("once")
        raise error

    with pytest.raises(type(error)) as caught:
        task_measurement._retry_metadata(SimpleNamespace(deadline=float("inf")), operation)
    assert caught.value is error
    assert calls == ["once"]


def test_expired_observation_budget_never_calls_execute(task_measurement, monkeypatch):
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(task_measurement.time, "monotonic", lambda: 60.0)
        with pytest.raises(TimeoutError, match="allowance"):
            task_measurement._retry_metadata(
                SimpleNamespace(deadline=180.0), lambda: calls.append("new-work"), deadline=60.0
            )
    assert calls == []


def caller_context(module):
    """Build unit caller metadata from independent typed job fixtures."""
    original = hand_task_report(module).cells[0].jobs[0].job
    command = TaskExecuteRequest(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="execute_request",
        request_id=original.request_id,
        plan_id=original.plan_id,
    )
    job = TaskJobRecord.model_validate({**original.model_dump(), "request_digest": command.fingerprint})
    return command, job


def test_uncertain_admission_retries_exact_request_then_cleans_its_original_job(task_measurement, monkeypatch):
    """Caller protocol regression; physical principal remains the installed flow."""
    command, job = caller_context(task_measurement)
    view = TaskJobView(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="job_view",
        job=job,
        worker_active=False,
        elapsed_seconds=0.0,
    )
    calls, cancelled, lookups = [], [], []
    store = object()

    def execute(supplied):
        calls.append(supplied)
        if len(calls) == 1:
            raise contention(sqlite3.SQLITE_BUSY)  # The retained admission already exists.
        return view

    def retained(supplied_store, owner, key):
        lookups.append((supplied_store, owner, key))
        return job

    service = SimpleNamespace(store=store, owner=job.owner, execute=execute, cancel=cancelled.append)
    run = SimpleNamespace(service=service, commands=(command,), deadline=180.0)
    with monkeypatch.context() as patch:
        patch.setattr(task_measurement.time, "monotonic", lambda: 1.0)
        patch.setattr(task_measurement.time, "sleep", lambda _seconds: None)
        patch.setattr(task_measurement, "_views", lambda *_args: (view,))
        patch.setattr(task_measurement.execution_lifecycle, "retained_request", retained)
        with ExitStack() as cleanup:
            ids, _, _ = task_measurement._admit(run, cleanup, 60.0)
            assert ids == [job.job_id]
    assert len(calls) == 2
    assert all(supplied is command for supplied in calls)
    assert lookups == [(store, job.owner, command.request_id)]
    assert cancelled == [job.job_id]


def test_after_admission_nonretryable_failure_still_runs_registered_cleanup(task_measurement, monkeypatch):
    command, job = caller_context(task_measurement)
    cancelled = []

    def lost_response(_command):
        raise TypeError("after-admission defect must still propagate")

    run = SimpleNamespace(
        service=SimpleNamespace(store=object(), owner=job.owner, execute=lost_response, cancel=cancelled.append),
        commands=(command,),
        deadline=180.0,
    )
    with monkeypatch.context() as patch:
        patch.setattr(task_measurement.time, "monotonic", lambda: 1.0)
        patch.setattr(task_measurement.execution_lifecycle, "retained_request", lambda *_args: job)
        with pytest.raises(TypeError, match="after-admission"), ExitStack() as cleanup:
            task_measurement._admit(run, cleanup, 60.0)
    assert cancelled == [job.job_id]


@pytest.mark.parametrize("damage", ("owner", "request_id", "request_digest", "plan_id", "family", "absent"))
def test_uncertain_cleanup_never_cancels_another_retained_request(task_measurement, monkeypatch, damage):
    command, job = caller_context(task_measurement)
    selected = job
    if damage == "family":
        selected = object()
    elif damage == "absent":
        selected = None
    else:
        values = job.model_dump()
        values[damage] = "another-owner" if damage in {"owner", "request_id"} else "sha256:" + "f" * 64
        if damage == "plan_id":
            values[damage] = "f" * 64
        selected = TaskJobRecord.model_validate(values)
    cancelled = []
    run = SimpleNamespace(
        service=SimpleNamespace(store=object(), owner=job.owner, cancel=cancelled.append), deadline=180.0
    )
    with monkeypatch.context() as patch:
        patch.setattr(task_measurement.time, "monotonic", lambda: 1.0)
        patch.setattr(task_measurement.execution_lifecycle, "retained_request", lambda *_args: selected)
        if damage == "absent":
            task_measurement._cancel_command(run, command)
        else:
            with pytest.raises(ValueError, match="Cleanup"):
                task_measurement._cancel_command(run, command)
    assert cancelled == []


def test_late_uncertain_admission_cleans_retained_work_without_next_execute(task_measurement, monkeypatch):
    command, job = caller_context(task_measurement)
    view = TaskJobView(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="job_view",
        job=job,
        worker_active=False,
        elapsed_seconds=60.0,
    )
    calls, cancelled, clock = [], [], iter((1.0, 60.0, 61.0, 61.0))

    def late_response(supplied):
        calls.append(supplied)
        return view

    run = SimpleNamespace(
        service=SimpleNamespace(store=object(), owner=job.owner, execute=late_response, cancel=cancelled.append),
        commands=(command, command),
        deadline=180.0,
    )
    with monkeypatch.context() as patch:
        patch.setattr(task_measurement.time, "monotonic", lambda: next(clock))
        patch.setattr(task_measurement.execution_lifecycle, "retained_request", lambda *_args: job)
        with pytest.raises(TimeoutError, match="admission"), ExitStack() as cleanup:
            task_measurement._admit(run, cleanup, 60.0)
    assert calls == [command]
    assert cancelled == [job.job_id]


def test_late_admission_status_cannot_replace_eligible_views_or_start_next_request(task_measurement, monkeypatch):
    command, job = caller_context(task_measurement)
    view = TaskJobView(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="job_view",
        job=job,
        worker_active=False,
        elapsed_seconds=0.0,
    )
    calls, cancelled, clock = [], [], iter((1.0, 1.0, 60.0, 61.0, 61.0))

    def response(supplied):
        calls.append(supplied)
        return view

    run = SimpleNamespace(
        service=SimpleNamespace(store=object(), owner=job.owner, execute=response, cancel=cancelled.append),
        commands=(command, command),
        deadline=180.0,
    )
    with monkeypatch.context() as patch:
        patch.setattr(task_measurement.time, "monotonic", lambda: next(clock))
        patch.setattr(task_measurement, "_views", lambda *_args: (view,))
        patch.setattr(task_measurement.execution_lifecycle, "retained_request", lambda *_args: job)
        with pytest.raises(TimeoutError, match="Admission status"), ExitStack() as cleanup:
            task_measurement._admit(run, cleanup, 60.0)
    assert calls == [command]
    assert cancelled == [job.job_id]
