"""Transport-neutral durable CI workflow application service for trusted local hosts."""

from __future__ import annotations

import hashlib
import time
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import polars as pl
from pydantic import TypeAdapter

from coleman.application import execution_lifecycle as lifecycle
from coleman.application.baseline_storage import read_baseline_artifact
from coleman.application.workflow_artifacts import read_trial_attempts, verify_report, verify_trial
from coleman.application.workflow_store import JobRecord, WorkflowStore
from coleman.results.domain import Measurement
from coleman.spec.selected_runtime import capture_native_runtime, require_native_runtime
from coleman.tasks.identity import HostBootstrapProfile
from coleman.workflows.contracts import DatasetManifest, ExperimentPlan, PlanningRequest, WorkflowError, WorkflowId
from coleman.workflows.lifecycle import (
    DatasetSummary,
    ExecutePlan,
    HostPolicy,
    JobView,
    PlanPreview,
    TrialCoordinate,
    TrialSummary,
    WorkflowDiscovery,
    WorkflowInspection,
    WorkflowReport,
)
from coleman.workflows.planning import make_plan

if TYPE_CHECKING:
    from coleman.application.workflow_downloads import _ArtifactManifest


def _preview(plan: ExperimentPlan) -> PlanPreview:
    """Return exact bounded coordinates without copying every full RunSpec into context."""
    return PlanPreview(
        schema_version=1,
        plan_id=plan.fingerprint[7:],
        digest=plan.fingerprint,
        dataset_id=plan.dataset.dataset_id,
        configurations=len({trial.configuration_id for trial in plan.trials}),
        trials=len(plan.trials),
        expected_build_rows=len(plan.trials) * len(plan.dataset.build_ids),
        coordinates=tuple(TrialCoordinate.from_trial(trial) for trial in plan.trials),
        evaluation=plan.request.evaluation,
    )


class WorkflowService:
    """Persistent host-owned workflows, detached from any CLI/MCP client lifetime.

    Parameters
    ----------
    root : pathlib.Path
        Trusted private POSIX local workspace. No operation accepts an untrusted
        path, URL, plugin, shell command or native specification override.
    owner : str
        Trusted local principal, never supplied by tool arguments. Caller-bound
        plans/jobs/imports are isolated; the dataset registry is host-owned.
    policy : HostPolicy or None
        Initial finite caps. A conflicting policy on reopen rejects explicitly.
    bootstrap_profile : HostBootstrapProfile or None, optional
        Explicit generic approval when sharing a workspace with task jobs. None
        preserves native startup and cannot authorize queued generic execution.

    Notes
    -----
    Uses short SQLite transactions, immutable artifacts and kernel execution locks.
    Native execution is detached into fixed package subprocesses. Disconnect/close
    does not cancel work. Reopening reconciles lost ownership as interrupted;
    continuation is explicit and rechecks producer/data/configuration compatibility.
    Only trusted local hosts on Linux/macOS local storage are supported. Resource
    sampling is not a hostile-code sandbox or a distributed exactly-once guarantee.
    """

    def __init__(
        self,
        root: Path,
        *,
        owner: str = "local-workspace",
        policy: HostPolicy | None = None,
        bootstrap_profile: HostBootstrapProfile | None = None,
    ):
        """Open durable metadata; no fixture, source acquisition or native work is implicit."""
        self.owner = TypeAdapter(WorkflowId).validate_python(owner)
        self.store = WorkflowStore(root, policy)
        self.bootstrap_profile = (
            None if bootstrap_profile is None else HostBootstrapProfile.model_validate(bootstrap_profile)
        )
        self._artifact_manifest: _ArtifactManifest | None = None

    def close(self) -> None:
        """Release the bounded connection-local artifact cache; durable jobs/evidence remain owned by the host."""
        self._artifact_manifest = None

    def reconnect(self) -> tuple[JobView, ...]:
        """Reconcile owned jobs and restart only the trusted workspace coordinator.

        Returns
        -------
        tuple[JobView, ...]
            Bounded retained caller-owned snapshots. Queued admissions are scheduled;
            interrupted/cancelled work requires explicit resume. Live kernel-owned
            workers remain active even if their previous coordinator disappeared.

        Raises
        ------
        ValueError, OSError
            Corrupt metadata, unsupported ownership storage or coordinator spawn failure.
        """
        with closing(self.store.connect()) as connection:
            identities = tuple(
                job.job_id
                for job in self.store.records(connection)
                if isinstance(job, JobRecord) and job.owner == self.owner
            )
        views = tuple(self.status(identity) for identity in identities)
        self._start()
        return views

    def register(self, manifest: DatasetManifest, *, snapshot_root: Path | None = None) -> DatasetManifest:
        """Admit a host-prepared exact-byte snapshot with finite aggregate registry bounds.

        Parameters
        ----------
        manifest : DatasetManifest
            Trusted snapshot produced by workflow_registry, including actual source
            and projection identities. This method is intentionally absent from MCP.
        snapshot_root : pathlib.Path or None
            Trusted temporary preparation root, outside the workspace. Its validated
            snapshot is copied only after transactional ID/count/byte admission.
            None preserves the existing workspace-snapshot registration API.

        Returns
        -------
        DatasetManifest
            Unchanged immutable registration; identical retries are idempotent.

        Raises
        ------
        ValueError, OSError
            Conflicting ID, missing/altered snapshot or registry/byte limit exceeded.

        Notes
        -----
        Publication and metadata admission share the registry transaction. A crash
        after atomic byte publication but before metadata commit may leave a snapshot;
        it consumes physical count/byte capacity, including on later retries. No
        unregistered file is silently deleted or treated as completed registration.
        """
        from coleman.application.workflow_artifacts import pin_file

        manifest = DatasetManifest.model_validate(manifest)
        if len(manifest.to_json().encode()) > 262144:
            raise WorkflowError("Registered source metadata exceeds 256 KiB")
        source_root = self.store.root if snapshot_root is None else snapshot_root.absolute()
        path = source_root / "datasets" / f"{manifest.sha256[7:]}.parquet"
        pin = pin_file(source_root, path)
        if (pin.sha256, pin.size_bytes) != (manifest.sha256, manifest.size_bytes):
            raise WorkflowError("Registered input bytes do not match the manifest")
        from coleman.application.workflow_registry import _publish_snapshot, inspect_history

        with path.open("rb") as stream:
            payload = stream.read(67108865)
        if ("sha256:" + hashlib.sha256(payload).hexdigest(), len(payload)) != (manifest.sha256, manifest.size_bytes):
            raise WorkflowError("Registered input changed during validation")
        frame = inspect_history(payload)
        dimensions = (
            tuple(sorted(frame["BuildId"].unique().to_list())),
            frame.height,
            int(max(frame.group_by("BuildId").len()["len"].to_list())),
        )
        if dimensions != (manifest.build_ids, manifest.rows, manifest.maximum_tests):
            raise WorkflowError("Registered dimensions differ from the actual native source bytes")
        with self.store.transaction() as connection:
            values = self.store.datasets(connection)
            existing = next((value for value in values if value.dataset_id == manifest.dataset_id), None)
            if existing is not None:
                if existing != manifest:
                    raise WorkflowError("Registered dataset ID cannot be silently replaced")
                _publish_snapshot(self.store, manifest, payload)
                return existing
            if len(values) >= self.store.policy.max_datasets:
                raise WorkflowError("Dataset registry is full; create another owned workspace")
            if sum(value.size_bytes for value in values) + manifest.size_bytes > self.store.policy.max_registered_bytes:
                raise WorkflowError("Aggregate registered input byte limit exceeded")
            _publish_snapshot(self.store, manifest, payload)
            connection.execute("INSERT INTO datasets VALUES (?,?)", (manifest.dataset_id, manifest.to_json()))
        return manifest

    def discover(self) -> WorkflowDiscovery:
        """Discover actual approved IDs, typed planning schema, defaults and host caps.

        Returns
        -------
        WorkflowDiscovery
            Bounded metadata only; no source outcomes or mutable host paths.
        """
        with closing(self.store.connect()) as connection:
            return WorkflowDiscovery(
                schema_version=1,
                datasets=tuple(DatasetSummary.from_manifest(item) for item in self.store.datasets(connection)),
                policy=self.store.policy,
                planning_schema=PlanningRequest.model_json_schema(),
                native_threads=pl.thread_pool_size(),
            )

    def plan(self, request: PlanningRequest) -> PlanPreview:
        """Resolve and save immutable executable coordinates before queue admission.

        Parameters
        ----------
        request : PlanningRequest
            Versioned bounded policy, parameter, budget, seed and evaluation choices.

        Returns
        -------
        PlanPreview
            Stable digest/handle and every expected coordinate, with separate counts.

        Raises
        ------
        KeyError, ValueError, OSError
            Unregistered/ineligible data, incompatible units, caps or storage failure.
            No native work starts; expansion is bounded before materialization.
        """
        request = PlanningRequest.model_validate(request)
        for name, value in request.limits.model_dump().items():
            if value > getattr(self.store.policy.caps, name):
                raise WorkflowError(f"Planning limit exceeds host cap: {name}")
        if request.limits.max_memory_bytes > self.store.policy.max_active_memory_bytes:
            raise WorkflowError("One worker exceeds aggregate active memory capacity")
        with closing(self.store.connect()) as connection:
            dataset = next(
                (item for item in self.store.datasets(connection) if item.dataset_id == request.dataset_id), None
            )
        if dataset is None:
            raise KeyError("Dataset is not registered")
        plan = make_plan(request, dataset, capture_native_runtime(), native_threads=pl.thread_pool_size())
        lifecycle.save_plan(self.store, plan, self.owner)
        return _preview(plan)

    def saved_plan(self, plan_id: str) -> ExperimentPlan:
        """Load a caller-owned fully resolved plan without executing it.

        Parameters
        ----------
        plan_id : str
            Saved full hexadecimal content handle.

        Returns
        -------
        ExperimentPlan
            Revalidated immutable plan; unknown schema/runtime is not migrated.
        """
        with closing(self.store.connect()) as connection:
            return self.store.plan(connection, TypeAdapter(WorkflowId).validate_python(plan_id), self.owner)

    def execute(self, request: ExecutePlan) -> JobView:
        """Idempotently admit a saved plan and start the fixed durable coordinator.

        Parameters
        ----------
        request : ExecutePlan
            Caller-scoped retained request key and saved plan handle; no overrides.

        Returns
        -------
        JobView
            Same logical job on identical retries, including after restart/expiry.

        Raises
        ------
        KeyError, ValueError, OSError
            Missing owned plan, key conflict, compatibility or aggregate admission
            bounds. No extra interactive confirmation is required.
        """
        request = ExecutePlan.model_validate(request)
        existing = lifecycle.retained_request(self.store, self.owner, request.request_id)
        if existing is not None:
            if not isinstance(existing, JobRecord) or existing.request_digest != request.fingerprint:
                raise WorkflowError("Idempotency key conflicts with its retained payload or execution family")
            job = existing
        else:
            plan = self.saved_plan(request.plan_id)
            require_native_runtime(plan.runtime)
            if plan.native_threads != pl.thread_pool_size():
                raise WorkflowError("Saved plan producer/runtime is incompatible; create a new plan")
            now = time.time()
            proposed = JobRecord(
                schema_version=1,
                job_id=uuid4().hex,
                owner=self.owner,
                request_id=request.request_id,
                request_digest=request.fingerprint,
                plan_id=request.plan_id,
                state="queued",
                stage="descriptive" if plan.request.evaluation == "descriptive" else "tuning",
                created=now,
                changed=now,
                total_trials=len(plan.trials),
            )
            job = lifecycle.admit_record(self.store, plan, proposed)
        self._start()
        return self.status(job.job_id)

    def _admission_limits(self, connection, jobs, plan) -> None:
        """Check new-job slot/byte reservations; resumed jobs retain their existing reservation."""
        lifecycle.check_admission(self.store, connection, jobs, plan)

    def _start(self) -> None:
        """Ensure a single fixed detached supervisor; no shell or user-selected module."""
        from coleman.application.workflow_supervisor import ensure_supervisor

        if self.bootstrap_profile is None:
            ensure_supervisor(self.store.root)
        else:
            ensure_supervisor(self.store.root, bootstrap_profile=self.bootstrap_profile)

    def _job(self, connection, job_id: str) -> JobRecord:
        """Look up only a validated caller-owned job, including retained tombstones."""
        identity = TypeAdapter(WorkflowId).validate_python(job_id)
        job = lifecycle.owned_record(self.store, connection, identity, self.owner)
        if not isinstance(job, JobRecord):
            raise WorkflowError("Native workflow facade requires a native workflow job")
        return job

    def status(self, job_id: str) -> JobView:
        """Inspect/reconcile a job using kernel ownership rather than stale PID metadata.

        Parameters
        ----------
        job_id : str
            Host-issued caller-owned durable identity.

        Returns
        -------
        JobView
            Actual lifecycle snapshot. Lost running ownership becomes interrupted;
            no checkpoint is executed automatically. ETA stays explicitly unavailable.
        """
        with closing(self.store.connect()) as connection:
            self._job(connection, job_id)
        job, active = lifecycle.reconcile_record(self.store, job_id, self.owner)
        if not isinstance(job, JobRecord):
            raise WorkflowError("Native workflow facade requires a native workflow job")
        return JobView(
            schema_version=1,
            job_id=job.job_id,
            request_id=job.request_id,
            plan_id=job.plan_id,
            state=job.state,
            stage=job.stage,
            completed_trials=job.completed_trials,
            total_trials=job.total_trials,
            current_trial=job.current_trial,
            checkpoint_step=job.checkpoint_step,
            attempts=job.attempts,
            invocations=job.invocations,
            elapsed_seconds=max(0.0, time.time() - job.created),
            eta_seconds=Measurement(
                name="remaining_wall_seconds",
                value=None,
                unit="s",
                meaning="coleman.workflow.remaining-wall.v1",
                direction="none",
                status="unavailable",
            ),
            worker_active=active,
            result_digest=job.result_digest,
            error=job.error,
        )

    def cancel(self, job_id: str) -> JobView:
        """Request cancellation transactionally; completed receipts win if committed first.

        Parameters
        ----------
        job_id : str
            Owned identity. Repeated cancellation is idempotent.

        Returns
        -------
        JobView
            Cancelled state or the already terminal state. A worker can remain
            physically active until its next guard sample; resume rejects meanwhile.
        """
        with closing(self.store.connect()) as connection:
            self._job(connection, job_id)
        lifecycle.cancel_record(self.store, job_id, self.owner)
        return self.status(job_id)

    def resume(self, job_id: str) -> JobView:
        """Explicitly requeue compatible interrupted/failed/cancelled native work.

        Parameters
        ----------
        job_id : str
            Owned retained job. Active/queued retries return that same job.

        Returns
        -------
        JobView
            Stable job with completed trials reused and incomplete native state
            restored by the existing checkpoint engine in a separate physical attempt.

        Raises
        ------
        ValueError
            Active cancelled owner, exhausted attempts/queue, expired or incompatible
            producer. Never silently substitutes a fresh run for rejected continuation.
        """
        self.status(job_id)
        with closing(self.store.connect()) as connection:
            job = self._job(connection, job_id)
        if job.state not in ("completed", "queued", "running", "awaiting_evaluation"):
            plan = self.saved_plan(job.plan_id)
            require_native_runtime(plan.runtime)
            if plan.native_threads != pl.thread_pool_size():
                raise WorkflowError("Incompatible runtime or exhausted worker invocation allowance")
            lifecycle.requeue_record(self.store, job, plan)
        self._start()
        return self.status(job_id)

    def inspect(self, job_id: str, *, offset: int = 0, limit: int = 4) -> WorkflowInspection:
        """Read bounded verified complete-trial pages, including honest partial progress.

        Parameters
        ----------
        job_id : str
            Owned retained identity.
        offset : int
            Zero-based logical trial page offset, at most 128.
        limit : int
            Between one and eight trial receipts; no entire input history is returned.

        Returns
        -------
        WorkflowInspection
            Complete published stage rows only; failed/incomplete attempts remain
            visible through durable progress/events rather than fabricated zero rows.
        """
        if type(offset) is not int or type(limit) is not int or not 0 <= offset <= 128 or not 1 <= limit <= 8:
            raise WorkflowError("Inspection requires bounded integer offset/limit")
        view = self.status(job_id)
        if view.state == "expired":
            raise WorkflowError("Expired workflow evidence is unavailable")
        plan = self.saved_plan(view.plan_id)
        root = self.store.root / "jobs" / job_id
        coordinates = plan.trials
        if view.stage == "evaluation":
            from coleman.application.workflow_evaluation import EvaluationHandoff, evaluation_trials

            coordinates = evaluation_trials(
                plan, read_baseline_artifact(root / "evaluation-handoff.json", EvaluationHandoff)
            )
        receipts, attempts = [], []
        page_coordinates = coordinates[offset : offset + limit]
        for trial in page_coordinates:
            path = root / "evidence" / trial.trial_id[7:] / view.stage / "complete.json"
            if path.exists():
                receipts.append(verify_trial(root, path, plan, trial, view.stage))
            attempts.extend(read_trial_attempts(root, plan, trial))
        page = WorkflowInspection(
            schema_version=1,
            job=view,
            trials=tuple(TrialSummary.from_result(receipt) for receipt in receipts),
            coordinates=tuple(TrialCoordinate.from_trial(trial) for trial in page_coordinates),
            attempts=tuple(attempts),
            next_offset=offset + limit if offset + limit < len(coordinates) else None,
        )
        if len(page.to_json().encode()) > 524288:
            raise WorkflowError("Inspection page exceeds context limit; request fewer trial receipts")
        return page

    def evaluate(self, job_id: str) -> JobView:
        """Freeze validation and queue audited continuation using actual native learning state.

        Parameters
        ----------
        job_id : str
            Owned protected job at awaiting_evaluation. Completed/evaluation retries
            return the same job without reserving another holdout access.

        Returns
        -------
        JobView
            Same logical job in evaluation. Full candidate, baseline, learning-state
            snapshots and protocol are saved before protected outcomes are executed.

        Raises
        ------
        ValueError, OSError
            Wrong stage, missing prefix evidence, incompatible producer/runtime or
            exhausted invocation/admission limits. Source eligibility is never invented.
        """
        from coleman.application.workflow_evaluation import evaluation_trials, prepare_evaluation

        with self.store.transaction() as connection:
            job = self._job(connection, job_id)
            plan = self.store.plan(connection, job.plan_id, self.owner)
            if plan.request.evaluation != "protected":
                raise WorkflowError("Descriptive replay exposes the full history; create a separate protected plan")
            if job.state != "completed" and job.stage != "evaluation":
                if job.state != "awaiting_evaluation":
                    raise WorkflowError("Protected evaluation requires a completed tuning prefix")
                plan = self.store.plan(connection, job.plan_id, self.owner)
                require_native_runtime(plan.runtime)
                if plan.native_threads != pl.thread_pool_size() or job.invocations >= plan.request.limits.max_attempts:
                    raise WorkflowError("Incompatible producer/runtime or exhausted invocation allowance")
                lifecycle.check_queue_capacity(self.store, self.store.records(connection))
                handoff = prepare_evaluation(self.store.root, self.store.root / "jobs" / job_id, plan)
                self.store.save_job(
                    connection,
                    JobRecord.model_validate(
                        {
                            **job.model_dump(),
                            "stage": "evaluation",
                            "state": "queued",
                            "completed_trials": 0,
                            "total_trials": len(evaluation_trials(plan, handoff)),
                            "checkpoint_step": 0,
                            "current_trial": None,
                            "changed": time.time(),
                        }
                    ),
                )
        self._start()
        return self.status(job_id)

    def report(self, job_id: str) -> WorkflowReport:
        """Read a complete-last workflow report without loading native checkpoints.

        Parameters
        ----------
        job_id : str
            Owned completed identity.

        Returns
        -------
        WorkflowReport
            Revalidated content bound to the published lifecycle digest.

        Raises
        ------
        ValueError, OSError
            Incomplete/expired/altered report or native member bytes.
        """
        view = self.status(job_id)
        if view.state != "completed":
            raise WorkflowError("Workflow has no completed report")
        root = self.store.root / "jobs" / job_id
        report = read_baseline_artifact(root / "complete.json", WorkflowReport)
        plan = self.saved_plan(view.plan_id)
        if report.fingerprint != view.result_digest or report.plan_fingerprint != plan.fingerprint:
            raise WorkflowError("Workflow completion binding mismatch")
        verify_report(root, plan, report)
        return report

    def cleanup(self) -> tuple[str, ...]:
        """Expire owned terminal bytes while retaining bounded nonrecyclable key tombstones.

        Returns
        -------
        tuple[str, ...]
            Expired job IDs. Active locks and nonterminal jobs are never removed.
            Explicit cleanup has filesystem side effects; plans/datasets remain.

        Notes
        -----
        Tombstones retain fixed slots for the workspace lifetime. Exhausted metadata
        requires a new explicitly selected workspace rather than recycling keys and
        accidentally executing an old retried request again.
        """
        return lifecycle.cleanup_records(self.store, self.owner, JobRecord)

    def export(self, job_id: str):
        """Export bounded self-contained reproducible evidence without executable state.

        Parameters
        ----------
        job_id : str
            Owned completed native workflow identity.

        Returns
        -------
        EvidenceBundle
            Complete abstract RunSpecs, source/runtime provenance, report and exact
            native Parquet bytes. Original local checkpoints are never exported.

        Raises
        ------
        ValueError, OSError
            Incomplete/expired/altered evidence or portable JSON over 2 MiB. Larger
            retained jobs remain available through bounded inspection and host storage.
        """
        import base64

        from coleman.application.workflow_exports import EvidenceBundle, WorkflowPortableFile

        report = self.report(job_id)
        view = self.status(job_id)
        plan = self.saved_plan(view.plan_id)
        root = self.store.root / "jobs" / job_id
        files, total = [], 0
        for receipt in report.trials:
            for pin in receipt.files:
                total += pin.size_bytes
                if total > 1048576:
                    raise WorkflowError(
                        "Native evidence exceeds inline export byte allowance; inspect retained host artifacts"
                    )
                files.append(
                    WorkflowPortableFile(
                        name=pin.name,
                        sha256=pin.sha256[7:],
                        size_bytes=pin.size_bytes,
                        content_base64=base64.b64encode((root / pin.name).read_bytes()).decode("ascii"),
                    )
                )
        bundle = EvidenceBundle(schema_version=1, plan=plan, report=report, files=tuple(files))
        if len(bundle.to_json().encode()) > 2097152:
            raise WorkflowError("Portable evidence exceeds 2 MiB; inspect retained host artifacts")
        return bundle

    def import_evidence(self, payload: str):
        """Validate and retain a caller-owned read-only bundle, never an executable plan.

        Parameters
        ----------
        payload : str
            Versioned portable JSON, at most 2 MiB; unknown versions, unsafe names,
            changed bytes and report/native inconsistencies reject before retention.

        Returns
        -------
        ImportedEvidence
            Stable caller-scoped handle. No file extraction, pickle load, dataset
            registration, native execution or scientific source admission occurs.

        Raises
        ------
        ValueError, OSError, BaselineBusyError
            Malformed evidence, exhausted bounded slots/bytes or storage failure.

        Notes
        -----
        A valid version/source header reserves conservative exposure before native
        outcome validation, even if later validation or retention capacity rejects.
        Exact-payload retries reuse that attempt; different wire encodings may
        conservatively overcount exposure while retaining the same content handle.
        Retention charges canonical UTF-8 bytes, including restored defaults,
        across all callers; retrying an existing handle consumes no extra space.
        """
        from coleman.application.workflow_exports import EvidenceBundle, ImportedEvidence, evidence_source_plan
        from coleman.application.workflow_exposure import reserve_import_exposure
        from coleman.results.domain import fingerprint

        if len(payload.encode()) > 2097152:
            raise WorkflowError("Portable workflow JSON exceeds 2 MiB")
        plan = evidence_source_plan(payload)
        attempt = fingerprint(["workflow.import-attempt.v1", self.owner, payload])[7:]
        reserve_import_exposure(self.store, plan, attempt)
        bundle = EvidenceBundle.from_json(payload)
        canonical = bundle.to_json()
        retained_size = len(canonical.encode("utf-8"))
        identity = fingerprint([self.owner, bundle.fingerprint])[7:]
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT payload FROM imports WHERE id=? AND owner=?", (identity, self.owner)
            ).fetchone()
            if existing is None:
                count, size = connection.execute(
                    "SELECT count(*),coalesce(sum(length(CAST(payload AS BLOB))),0) FROM imports"
                ).fetchone()
                if count >= self.store.policy.max_jobs or size + retained_size > self.store.policy.max_retained_bytes:
                    raise WorkflowError("Read-only import slots/bytes exhausted")
                connection.execute("INSERT INTO imports VALUES (?,?,?)", (identity, self.owner, canonical))
        return ImportedEvidence(schema_version=1, evidence_id=identity, digest=bundle.fingerprint)

    def imported_evidence(self, evidence_id: str):
        """Retrieve a caller-owned read-only bundle without changing holdout access or execution.

        Parameters
        ----------
        evidence_id : str
            Host-issued import handle; it cannot be used as a job/plan/checkpoint.

        Returns
        -------
        EvidenceBundle
            Fully revalidated bounded evidence; integrity is not authenticity.
        """
        from coleman.application.workflow_exports import EvidenceBundle

        identity = TypeAdapter(WorkflowId).validate_python(evidence_id)
        with closing(self.store.connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM imports WHERE id=? AND owner=?", (identity, self.owner)
            ).fetchone()
        if row is None:
            raise KeyError("No owned read-only evidence")
        bundle = EvidenceBundle.from_json(row[0])
        from coleman.results.domain import fingerprint

        expected_id = fingerprint([self.owner, bundle.fingerprint]).removeprefix("sha256:")
        if expected_id != identity:
            raise WorkflowError("Read-only evidence handle/content binding mismatch")
        return bundle
