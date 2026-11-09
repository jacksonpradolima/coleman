"""Source-unit task flows with real Hedge, metadata, common driver and ACKs.

Installed-image selection and detached startup are substituted. The existing
trusted WorkerHooks supplies an explicit 1 MiB successful memory observation:
this in-parent unit must not charge the whole pytest process's inherited peak
to one worker. Actual guard predicates, output sampling and caps are unchanged.
The image is not an installed capture. Concrete codecs, SQL bodies, kernel
ownership, the common loop and complete mathematical state are real; separate
installed tests qualify the actual installed process and resource observation.
"""

from __future__ import annotations

import json
from contextlib import closing, contextmanager
from types import SimpleNamespace

import pytest

from coleman.application import execution_driver as driver
from coleman.application import execution_lifecycle as lifecycle
from coleman.application import execution_views as views
from coleman.application import task_runtime, tasks
from coleman.application.baseline_lock import baseline_lock
from coleman.application.tasks import TaskService
from coleman.application.workflow_store import WorkflowStore
from coleman.core.identity import Reference
from coleman.learning.reference import Hedge
from coleman.tasks.contracts import EncodedValue
from coleman.tasks.identity import ApprovedExecutor, HostBootstrapProfile
from coleman.tasks.requests import TaskExecuteRequest, TaskHostPolicy
from coleman.workflows.contracts import ExperimentPlan, WorkflowError
from coleman.workflows.lifecycle import HostPolicy

from .test_completed_prefix import _declared_image
from .test_reference_executor import source as source
from .test_storage_records import NATIVE, _native


def changed(value, **updates):
    """Revalidate an altered real immutable declaration, without unchecked copies."""
    return type(value).model_validate({**value.model_dump(), **updates})


def records(service):
    """Read the complete real job catalogue, without invoking a provider."""
    with closing(service.store.connect()) as connection:
        return tuple(item.to_json() for item in service.store.records(connection))


def retained_plans(service):
    """Read all immutable plan rows to detect partial failed admissions."""
    with closing(service.store.connect()) as connection:
        return tuple(tuple(row) for row in connection.execute("SELECT * FROM plans ORDER BY id"))


@pytest.fixture
def service_case(source, tmp_path, monkeypatch):
    """Select source bytes explicitly; preserve every principal execution body."""
    image = _declared_image(source)
    approval = ApprovedExecutor(recipe=image.recipe, implementation=image.binding)
    profile = HostBootstrapProfile(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="host_bootstrap",
        reference=Reference(id="declared-source-service-host", version="1"),
        executors=(approval,),
    )
    selected = source.package.bootstrap()
    calls = []
    sql = SimpleNamespace(depth=0)
    transaction = WorkflowStore.transaction

    @contextmanager
    def observed_transaction(store, **kwargs):
        with transaction(store, **kwargs) as connection:
            sql.depth += 1
            try:
                yield connection
            finally:
                sql.depth -= 1

    def outside(name):
        assert sql.depth == 0, name
        calls.append(name)

    def capture(recipe):
        outside("declared-image-selection")
        assert recipe.to_json() == image.recipe.to_json()
        return image

    def load(current, *, expected):
        outside("declared-provider-selection")
        assert current.model_dump_json() == approval.model_dump_json()
        assert expected.to_json() == image.to_json()
        return selected

    def schedule(root, *, bootstrap_profile):
        outside("detached-start-substitution")
        assert root == tmp_path
        assert bootstrap_profile is None or bootstrap_profile.to_json() == profile.to_json()

    def observe_method(cls, name):
        original = getattr(cls, name)

        def actual(instance, *args, **kwargs):
            outside(name)
            return original(instance, *args, **kwargs)

        monkeypatch.setattr(cls, name, actual)

    monkeypatch.setattr(WorkflowStore, "transaction", observed_transaction)
    monkeypatch.setattr(tasks, "capture_executor_identity", capture)
    monkeypatch.setattr(tasks, "load_approved_executor", load)
    monkeypatch.setattr(task_runtime, "load_approved_executor", load)
    monkeypatch.setattr(tasks, "ensure_supervisor", schedule)
    for name in ("validate", "encode", "decode"):
        observe_method(source.recipe.ModelCodec, name)
    observe_method(source.executor.HedgeExecutor, "prepare")
    for name in ("read_completed", "advance", "publish_coordinate", "validate", "reconcile"):
        observe_method(source.executor.PreparedHedge, name)
    service = TaskService(tmp_path, owner="source-unit", bootstrap_profile=profile)
    request = source.sample.planning_request()
    calls.clear()
    return SimpleNamespace(
        source=source,
        service=service,
        request=request,
        image=image,
        approval=approval,
        profile=profile,
        selected=selected,
        calls=calls,
        sql=sql,
        rss_samples=[],
    )


def admit(item, request_id="source-command"):
    """Plan and admit through the actual public facade, without math yet."""
    preview = item.service.plan(item.request)
    command = TaskExecuteRequest(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="execute_request",
        request_id=request_id,
        plan_id=preview.plan_id,
    )
    return command, item.service.execute(command)


def run_claimed(item, job_id):
    """Use the real lock/driver with explicit successful unit RSS observation."""
    service = item.service

    def unit_peak_rss():
        item.rss_samples.append(1.0)
        return 1.0

    with baseline_lock(service.store.root / "locks" / f"{job_id}.lock"):
        with service.store.transaction() as connection:
            current = lifecycle.owned_record(service.store, connection, job_id, service.owner)
            assert current.state == "queued"
            service.store.save_record(
                connection, views.changed(current, state="running", invocations=current.invocations + 1)
            )
        driver.run_with_hooks(
            service.store.root,
            job_id,
            bootstrap_profile=item.profile,
            hooks=driver.WorkerHooks(peak_rss_mib=unit_peak_rss),
        )


def direct_state(item):
    """Compute the complete independent direct state from original panels."""
    configuration = item.source.sample.configuration()
    trajectory = item.source.sample.inputs(configuration).trajectories[0]
    learner = Hedge(configuration.reference, owner=trajectory.reference)
    for panel in trajectory.panels:
        assert learner.observe(panel).status == "accepted"
    return learner.export_state()


def test_public_plan_admit_common_execution_report_and_retained_retry(service_case, monkeypatch):
    item = service_case
    discovery = item.service.discover()
    assert discovery.bootstrap == item.profile.binding
    assert discovery.executors == (item.image.recipe,)
    assert item.calls == []
    command, queued = admit(item)
    plan = item.service.saved_plan(command.plan_id)
    assert plan.source == item.request.source
    assert queued.job.state == "queued"
    assert "prepare" not in item.calls
    assert not (item.service.store.root / "jobs" / queued.job.job_id).exists()
    assert item.service.plan(item.request).plan_id == command.plan_id
    assert len(retained_plans(item.service)) == 1
    run_claimed(item, queued.job.job_id)
    assert item.rss_samples
    assert set(item.rss_samples) == {1.0}
    done = item.service.status(queued.job.job_id)
    assert done.job.state == "completed"
    assert done.worker_active is False
    assert done.eta_seconds is None
    completion = item.service.report(done.job.job_id)
    report = item.service.decode_result(done.job.job_id)
    assert completion.result.body == report.to_json()
    assert report.results[0].checkpoint.learner.to_json() == direct_state(item).to_json()
    assert report.results[0].proof.frontier.position == 3
    assert report.results[0].proof.observed_head.generation == 2
    before = records(item.service)
    item.calls.clear()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("retained command attempted provider loading")

    monkeypatch.setattr(tasks, "load_approved_executor", forbidden)
    assert item.service.execute(command).job.to_json() == done.job.to_json()
    assert records(item.service) == before
    assert item.calls == ["detached-start-substitution"]
    observer = TaskService(item.service.store.root, owner=item.service.owner)
    assert observer.discover().executors == ()
    assert observer.report(done.job.job_id).to_json() == completion.to_json()
    assert tuple(view.job.job_id for view in observer.reconnect()) == (done.job.job_id,)
    with pytest.raises(WorkflowError, match="explicit current bootstrap"):
        observer.decode_result(done.job.job_id)


def test_post_ack_interruption_public_resume_keeps_identity_and_only_remaining_panel(service_case, monkeypatch):
    item = service_case
    command, queued = admit(item)
    original = driver._Worker.record_progress

    def stop_after_ack(worker, program, context, progress, **kwargs):
        original(worker, program, context, progress, **kwargs)
        if progress.frontier.position == 2:
            raise InterruptedError("controlled response loss after actual U2 ACK")

    with monkeypatch.context() as patch:
        patch.setattr(driver._Worker, "record_progress", stop_after_ack)
        run_claimed(item, queued.job.job_id)
    interrupted = item.service.status(queued.job.job_id).job
    assert interrupted.state == "interrupted"
    assert interrupted.frontier is not None
    assert interrupted.frontier.position == 2
    assert interrupted.attempts == 1
    item.calls.clear()
    assert item.service.resume(interrupted.job_id).job.state == "queued"
    observed = []
    observe = Hedge.observe

    def record_actual(learner, panel):
        observed.append(panel.model_dump_json())
        return observe(learner, panel)

    with monkeypatch.context() as patch:
        patch.setattr(Hedge, "observe", record_actual)
        run_claimed(item, interrupted.job_id)
    report = item.service.decode_result(interrupted.job_id)
    expected_panel = item.source.sample.inputs().trajectories[0].panels[-1]
    assert observed == [expected_panel.model_dump_json()]
    assert report.results[0].checkpoint.learner.to_json() == direct_state(item).to_json()
    done = item.service.status(interrupted.job_id).job
    assert done.job_id == queued.job.job_id
    assert done.attempts == 2
    assert done.invocations == 2
    assert item.service.resume(done.job_id).job.to_json() == done.to_json()
    assert item.service.execute(command).job.to_json() == done.to_json()


@pytest.mark.parametrize("slot", ("configuration", "inputs", "coordinate"))
def test_concrete_invalid_values_refuse_before_plan_persistence_or_prepare(service_case, slot):
    item = service_case
    target = item.request.coordinates[0].body if slot == "coordinate" else getattr(item.request, slot)
    raw = json.loads(target.body)
    raw["schema_version"] = 2
    invalid = EncodedValue(codec=target.codec, body=json.dumps(raw))
    request = (
        changed(item.request, coordinates=(changed(item.request.coordinates[0], body=invalid),))
        if slot == "coordinate"
        else changed(item.request, **{slot: invalid})
    )
    with pytest.raises(ValueError, match="schema_version"):
        item.service.plan(request)
    assert retained_plans(item.service) == ()
    assert records(item.service) == ()
    assert "prepare" not in item.calls


def test_missing_or_unapproved_recipe_never_reaches_image_selection(service_case):
    item = service_case
    observer = TaskService(item.service.store.root, owner=item.service.owner)
    with pytest.raises(WorkflowError, match="explicit current bootstrap"):
        observer.plan(item.request)
    request = changed(item.request, executor=Reference(id="unapproved", version="1"))
    with pytest.raises(WorkflowError, match="current approved host catalogue"):
        item.service.plan(request)
    assert item.calls == []
    assert retained_plans(item.service) == ()


@pytest.mark.parametrize("capacity", ("generic_work", "aggregate_memory"))
def test_planning_caps_refuse_without_saving_work(service_case, tmp_path, capacity):
    item = service_case
    task_policy = (
        TaskHostPolicy(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="host_policy",
            caps=changed(item.request.limits, max_work_units=5),
        )
        if capacity == "generic_work"
        else None
    )
    policy = None if capacity == "generic_work" else HostPolicy(schema_version=1, max_active_memory_bytes=67108864)
    service = TaskService(
        tmp_path / capacity,
        owner="source-unit",
        bootstrap_profile=item.profile,
        policy=policy,
        task_policy=task_policy,
    )
    with pytest.raises(ValueError, match="host cap|aggregate active memory"):
        service.plan(item.request)
    assert retained_plans(service) == ()
    assert records(service) == ()
    assert "prepare" not in item.calls


def test_unknown_foreign_native_and_conflicting_handles_preserve_actual_records(service_case):
    item = service_case
    command, queued = admit(item)
    before = records(item.service)
    with pytest.raises(KeyError):
        item.service.saved_plan("0" * 64)
    with pytest.raises(KeyError):
        item.service.status("unknown-job")
    foreign = TaskService(item.service.store.root, owner="other")
    with pytest.raises(KeyError):
        foreign.saved_plan(command.plan_id)
    native = ExperimentPlan.from_json(NATIVE.read_bytes())
    lifecycle.save_plan(item.service.store, native, item.service.owner)
    with pytest.raises(WorkflowError, match="another execution family"):
        item.service.saved_plan(native.fingerprint[7:])
    conflict = changed(command, plan_id=native.fingerprint[7:])
    with pytest.raises(WorkflowError, match="Idempotency key conflicts"):
        item.service.execute(conflict)
    with pytest.raises(WorkflowError, match="no completed report"):
        item.service.report(queued.job.job_id)
    assert records(item.service) == before


def test_cancel_resume_and_expiry_retain_original_request_tombstone(service_case):
    item = service_case
    command, queued = admit(item)
    cancelled = item.service.cancel(queued.job.job_id).job
    assert cancelled.state == "cancelled"
    assert item.service.cancel(cancelled.job_id).job.to_json() == cancelled.to_json()
    assert item.service.resume(cancelled.job_id).job.state == "queued"
    item.service.cancel(cancelled.job_id)
    with item.service.store.transaction() as connection:
        current = lifecycle.owned_record(item.service.store, connection, cancelled.job_id, item.service.owner)
        item.service.store.save_record(connection, changed(current, created=0.0, changed=0.0))
    assert item.service.cleanup() == (cancelled.job_id,)
    expired = item.service.status(cancelled.job_id).job
    assert expired.state == "expired"
    assert item.service.execute(command).job.to_json() == expired.to_json()
    with pytest.raises(WorkflowError, match="Expired work"):
        item.service.resume(expired.job_id)


def test_native_retained_key_and_job_cannot_be_reinterpreted_as_generic(service_case):
    item = service_case
    native = ExperimentPlan.from_json(NATIVE.read_bytes())
    record = _native(native, owner=item.service.owner, request_id="native-retained")
    lifecycle.save_plan(item.service.store, native, item.service.owner)
    lifecycle.admit_record(item.service.store, native, record)
    preview = item.service.plan(item.request)
    command = TaskExecuteRequest(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="execute_request",
        request_id=record.request_id,
        plan_id=preview.plan_id,
    )
    before = records(item.service)
    item.calls.clear()
    with pytest.raises(WorkflowError, match="Idempotency key conflicts"):
        item.service.execute(command)
    with pytest.raises(WorkflowError, match="another execution family"):
        item.service.status(record.job_id)
    assert records(item.service) == before
    assert item.calls == []
