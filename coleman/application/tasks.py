"""Trusted neutral task facade over the existing durable execution lifecycle."""

from __future__ import annotations

import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter

from coleman.application import execution_lifecycle as lifecycle
from coleman.application.execution_bootstrap import bind_host, require_current_profile, validate_task_caps
from coleman.application.execution_views import Job
from coleman.application.executor_capture import LoadedExecutor, capture_executor_identity, load_approved_executor
from coleman.application.workflow_store import WorkflowStore
from coleman.application.workflow_supervisor import ensure_supervisor
from coleman.core.identity import Reference
from coleman.tasks.contracts import PlanId, TaskId, TaskJobRecord, TaskPlan
from coleman.tasks.identity import ApprovedExecutor, HostBootstrapProfile
from coleman.tasks.protocols import decode_value
from coleman.tasks.requests import (
    TaskDiscovery,
    TaskExecuteRequest,
    TaskHostPolicy,
    TaskJobView,
    TaskPlanningRequest,
    TaskPlanPreview,
)
from coleman.tasks.results import TaskCompletion
from coleman.workflows.contracts import WorkflowError
from coleman.workflows.lifecycle import HostPolicy


class TaskService:
    """Use approved concrete executors through the common durable coordinator.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent local POSIX workspace, shared with native workflows.
    owner : str, optional
        Host-supplied local principal, never a per-request transport argument.
    bootstrap_profile : HostBootstrapProfile or None, optional
        Current independently approved installed recipes and image bindings.
        None permits provider-free status, cancellation and outer report reads;
        it cannot reconstruct execution approval from workspace metadata.
    policy : HostPolicy or None, optional
        Existing common queue/job/plan/output/memory policy. Explicit conflicting
        reopen settings refuse; the native policy schema is unchanged.
    task_policy : TaskHostPolicy or None, optional
        Separate immutable generic work ceilings, without native build units.

    Notes
    -----
    Construction opens bounded metadata and registers equality constraints only.
    Mathematical work starts in the same fixed detached worker used by native
    TCPCI; there is one queue, scheduler, resource guard and execution loop.
    Current capture requires the documented immutable directory-wheel CPython
    no-bytecode profile. Hosts configure that process mode before loading code.
    This trusted local runtime is not a sandbox or distributed effects service.
    Direct learners remain usable without this facade or any registration.
    """

    def __init__(
        self,
        root: Path,
        *,
        owner: str = "local-workspace",
        bootstrap_profile: HostBootstrapProfile | None = None,
        policy: HostPolicy | None = None,
        task_policy: TaskHostPolicy | None = None,
    ):
        """Open the shared store; constructor parameters and side effects apply."""
        self.owner = TypeAdapter(TaskId).validate_python(owner)
        self.store = WorkflowStore(root, policy)
        self.bootstrap_profile = (
            None if bootstrap_profile is None else HostBootstrapProfile.model_validate(bootstrap_profile)
        )
        self.policy = bind_host(self.store, self.bootstrap_profile, task_policy)

    def discover(self) -> TaskDiscovery:
        """Describe current explicit approval without importing provider code.

        Returns
        -------
        TaskDiscovery
            Approved static recipes and separate generic ceilings. No source
            outcomes, implementation factory or mathematical update is read.
        """
        profile = self.bootstrap_profile
        return TaskDiscovery(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="discovery",
            bootstrap=None if profile is None else profile.binding,
            executors=() if profile is None else tuple(item.recipe for item in profile.executors),
            policy=self.policy,
        )

    def _approval(self, reference: Reference) -> ApprovedExecutor:
        """Select only a current operator-approved recipe, never a stored import."""
        if self.bootstrap_profile is None:
            raise WorkflowError("Task planning/execution requires the explicit current bootstrap profile")
        approval = next((item for item in self.bootstrap_profile.executors if item.recipe.reference == reference), None)
        if approval is None:
            raise WorkflowError("Requested executor is not in the current approved host catalogue")
        return approval

    def plan(self, request: TaskPlanningRequest) -> TaskPlanPreview:
        """Freeze concrete values, installed identity and coordinates before work.

        Parameters
        ----------
        request : TaskPlanningRequest
            Complete bounded values under an already approved recipe and codecs.

        Returns
        -------
        TaskPlanPreview
            Immutable owned handle and explicit work semantics; saved_plan reads
            the complete resolved plan without starting mathematical execution.

        Raises
        ------
        ValueError, TypeError, OSError, ImportError
            Missing approval, changed installed image, concrete codec mismatch,
            unknown versions, incompatible caps or exhausted common plan slots.

        Notes
        -----
        Capture/bootstrap/concrete decoding occur outside SQL. No prepare,
        learner factory, source outcome loader or state restore runs here.
        """
        current = TaskPlanningRequest.model_validate(request)
        profile = self.bootstrap_profile
        if profile is None:
            raise WorkflowError("Task planning requires the explicit current bootstrap profile")
        approval = self._approval(current.executor)
        image = capture_executor_identity(approval.recipe)
        executor = load_approved_executor(approval, expected=image)
        decode_value(executor.configuration_codec, current.configuration)
        decode_value(executor.input_codec, current.inputs)
        for item in current.coordinates:
            decode_value(executor.coordinate_codec, item.body)
        plan = TaskPlan(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="plan",
            executor=approval.recipe.binding,
            implementation=image,
            bootstrap=profile.binding,
            configuration=current.configuration,
            inputs=current.inputs,
            source=current.source,
            objective=current.objective,
            protocol=current.protocol,
            stage=current.stage,
            coordinates=current.coordinates,
            limits=current.limits,
        )
        with closing(self.store.connect()) as connection:
            require_current_profile(connection, self.bootstrap_profile)
            validate_task_caps(connection, plan)
        if plan.limits.max_memory_bytes > self.store.policy.max_active_memory_bytes:
            raise WorkflowError("One worker exceeds aggregate active memory capacity")
        lifecycle.save_plan(self.store, plan, self.owner)
        return TaskPlanPreview(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="plan_preview",
            plan_id=plan.fingerprint.removeprefix("sha256:"),
            digest=plan.fingerprint,
            executor=approval.recipe.reference,
            work_profile=approval.recipe.frontier,
            coordinates=len(plan.coordinates),
            target_work_units=sum(item.target.position for item in plan.coordinates),
        )

    def saved_plan(self, plan_id: str) -> TaskPlan:
        """Read an owned strict known plan with no provider discovery or restore.

        Parameters
        ----------
        plan_id : str
            Host-issued full 64-hex content handle.

        Returns
        -------
        TaskPlan
            Complete known task plan. A native handle refuses at this facade.
        """
        identity = TypeAdapter(PlanId).validate_python(plan_id)
        with closing(self.store.connect()) as connection:
            plan = self.store.read_plan(connection, identity, self.owner)
        if not isinstance(plan, TaskPlan):
            raise WorkflowError("Task facade requires a task plan, not another execution family")
        return plan

    def _validate_execution(self, plan: TaskPlan) -> LoadedExecutor:
        """Recheck current approval/image and all concrete inputs outside SQL."""
        if self.bootstrap_profile is None:
            raise WorkflowError("Task execution requires the explicit current bootstrap profile")
        plan.validate_bootstrap(self.bootstrap_profile)
        approval = self._approval(plan.implementation.recipe.reference)
        executor = load_approved_executor(approval, expected=plan.implementation)
        decode_value(executor.configuration_codec, plan.configuration)
        decode_value(executor.input_codec, plan.inputs)
        for item in plan.coordinates:
            decode_value(executor.coordinate_codec, item.body)
        with closing(self.store.connect()) as connection:
            require_current_profile(connection, self.bootstrap_profile)
            validate_task_caps(connection, plan)
        return executor

    def execute(self, request: TaskExecuteRequest) -> TaskJobView:
        """Admit one job with shared all-owner capacity and retained key semantics.

        Parameters
        ----------
        request : TaskExecuteRequest
            Owned saved-plan handle and retained caller-scoped idempotency key.

        Returns
        -------
        TaskJobView
            Original logical job on exact retries, including retained expiry.

        Raises
        ------
        ValueError, TypeError, OSError, ImportError
            Cross-family/payload key conflict, missing approval, incompatible
            concrete values/image or exhausted shared metadata/resource limits.

        Notes
        -----
        Current image/codecs are checked before new admission; the detached
        worker rechecks before prepare/source/restore. A committed admission
        survives client disconnect or acknowledgement timeout. Retry the same
        request key; do not invent another job after an uncertain response.
        """
        current = TaskExecuteRequest.model_validate(request)
        existing = lifecycle.retained_request(self.store, self.owner, current.request_id)
        if existing is not None:
            if not isinstance(existing, TaskJobRecord) or existing.request_digest != current.fingerprint:
                raise WorkflowError("Idempotency key conflicts with its retained payload or execution family")
            job = existing
        else:
            plan = self.saved_plan(current.plan_id)
            self._validate_execution(plan)
            now = time.time()
            proposed = TaskJobRecord(
                schema_version=1,
                contract="coleman-tasks-v1",
                artifact_type="job",
                job_id=uuid4().hex,
                owner=self.owner,
                request_id=current.request_id,
                request_digest=current.fingerprint,
                plan_id=current.plan_id,
                state="queued",
                stage=plan.stage,
                created=now,
                changed=now,
                total_coordinates=len(plan.coordinates),
            )
            job = lifecycle.admit_record(self.store, plan, proposed)
        self._start()
        return self.status(job.job_id)

    def _start(self) -> None:
        """Ensure the existing fixed scheduler with current explicit approval."""
        ensure_supervisor(self.store.root, bootstrap_profile=self.bootstrap_profile)

    def status(self, job_id: str) -> TaskJobView:
        """Inspect/reconcile actual ownership without loading an unavailable provider.

        Parameters
        ----------
        job_id : str
            Owned host-issued task identity; native handles refuse explicitly.

        Returns
        -------
        TaskJobView
            Actual operational state with unavailable ETA and no inferred result.
        """
        identity = TypeAdapter(TaskId).validate_python(job_id)
        with closing(self.store.connect()) as connection:
            self._task(lifecycle.owned_record(self.store, connection, identity, self.owner))
        job, active = lifecycle.reconcile_record(self.store, identity, self.owner)
        return TaskJobView(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="job_view",
            job=self._task(job),
            worker_active=active,
            elapsed_seconds=max(0.0, time.time() - job.created),
        )

    @staticmethod
    def _task(job: Job) -> TaskJobRecord:
        """Reject another known job family before task-specific interpretation."""
        if not isinstance(job, TaskJobRecord):
            raise WorkflowError("Task facade requires a task job, not another execution family")
        return job

    def cancel(self, job_id: str) -> TaskJobView:
        """Cancel through the shared serialized lifecycle with no provider import.

        Parameters
        ----------
        job_id : str
            Owned task identity. Repeated cancellation retains the same state.

        Returns
        -------
        TaskJobView
            Cancelled or already terminal snapshot. A live cancelled worker still
            consumes shared capacity until actual kernel ownership is released.
        """
        self.status(job_id)
        lifecycle.cancel_record(self.store, job_id, self.owner)
        return self.status(job_id)

    def resume(self, job_id: str) -> TaskJobView:
        """Requeue explicit compatible continuation without reset or new identity.

        Parameters
        ----------
        job_id : str
            Owned interrupted, failed or cancelled job; active retries retain it.

        Returns
        -------
        TaskJobView
            Same logical job through the shared queue and worker lifecycle.

        Raises
        ------
        ValueError, TypeError, OSError, ImportError
            Incompatible current approval/image/codecs, expired bytes, live
            cancelled ownership or exhausted capacities. State restore is only
            attempted later by the selected worker after its own identity check.
        """
        job = self.status(job_id).job
        if job.state not in ("completed", "queued", "running", "awaiting_evaluation"):
            plan = self.saved_plan(job.plan_id)
            self._validate_execution(plan)
            lifecycle.requeue_record(self.store, job, plan)
        self._start()
        return self.status(job_id)

    def reconnect(self) -> tuple[TaskJobView, ...]:
        """Reconcile owned tasks and schedule only with current explicit approval.

        Returns
        -------
        tuple of TaskJobView
            Bounded owned snapshots. Interrupted/cancelled work is never
            implicitly resumed; live workers retain their actual kernel locks.
        """
        with closing(self.store.connect()) as connection:
            identities = tuple(
                job.job_id
                for job in self.store.records(connection)
                if isinstance(job, TaskJobRecord) and job.owner == self.owner
            )
        result = tuple(self.status(identity) for identity in identities)
        self._start()
        return result

    def report(self, job_id: str) -> TaskCompletion:
        """Read complete outer evidence without granting stored import authority.

        Parameters
        ----------
        job_id : str
            Owned completed task identity, never a path or imported plan.

        Returns
        -------
        TaskCompletion
            Complete-last outer manifest with exact retained concrete body.
            Concrete provider-body validation is explicit through decode_result.

        Raises
        ------
        ValueError, OSError
            Incomplete/expired/changed bytes or mismatched complete-plan binding.
        """
        view = self.status(job_id)
        if view.job.state != "completed":
            raise WorkflowError("Task has no completed report")
        path = self.store.root / "jobs" / view.job.job_id / "complete.json"
        with path.open("rb") as stream:
            result = TaskCompletion.from_json(stream.read(16777217))
        result.validate_for(self.saved_plan(view.job.plan_id))
        if result.fingerprint != view.job.result_digest:
            raise WorkflowError("Task completion binding mismatch")
        return result

    def decode_result(self, job_id: str) -> object:
        """Validate a concrete result using current explicitly approved code.

        Parameters
        ----------
        job_id : str
            Owned complete task identity whose exact plan/image remains approved.

        Returns
        -------
        object
            Actual concrete result type selected by the trusted executor codec,
            never an uncontracted dictionary used across the business boundary.

        Raises
        ------
        ValueError, TypeError, OSError, ImportError
            Changed/unapproved image, invalid complete evidence or concrete body.
            This reads no source outcomes, restores no state and starts no work.
        """
        completion = self.report(job_id)
        plan = self.saved_plan(self.status(job_id).job.plan_id)
        executor = self._validate_execution(plan)
        return decode_value(executor.final_result_codec, completion.result)

    def cleanup(self) -> tuple[str, ...]:
        """Expire owned terminal task bytes through the shared retention rules.

        Returns
        -------
        tuple of str
            Expired task handles. Request tombstones/plans retain finite slots;
            active locks and every other facade/owner remain untouched.
        """
        return lifecycle.cleanup_records(self.store, self.owner, TaskJobRecord)
