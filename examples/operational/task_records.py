"""Versioned finite task-measurement records and unchanged resource profile.

These records distinguish declared work, SQL-acknowledged lower bounds and
unavailable observations. They do not schedule or resume any work.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from coleman.core.base import CoreContract
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.tasks.contracts import TaskJobRecord, TaskLimits
from coleman.tasks.identity import ExecutorImplementationIdentity, HostBootstrapProfile
from coleman.tasks.requests import TaskJobView
from coleman.workflows.contracts import MAX_WORKFLOW_ARTIFACT_ENTRIES
from coleman.workflows.lifecycle import HostPolicy

CASES = ((1, "small"), (3, "small"), (1, "larger"), (3, "larger"))
CELL_SECONDS = 60.0
OUTER_SECONDS = 180.0
# This reserve is inside the original outer cap; it never funds more work.
ENTRY_RESERVE_SECONDS = 30.0
THREAD_VARIABLES = frozenset(
    {
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
)


RESOURCE_ERRORS = frozenset(
    {
        "sampled_resource_limit",
        "Workflow observed wall-time allowance exhausted",
        "Workflow output byte/file limit exceeded",
        "Workflow sampled memory allowance exhausted or unavailable",
    }
)


def task_limits(size: str) -> TaskLimits:
    """Declare the original resource envelope for one authored workload.

    Parameters
    ----------
    size : str
        Exactly ``small`` or ``larger``; these select1 or32 planned updates.

    Returns
    -------
    TaskLimits
        Coordinate/work/attempt bounds,60-second worker allowance and4MiB
        output cap. Default memory guards remain enabled without substitution.

    Raises
    ------
    ValueError
        The supplied name is outside the declared finite experiment.
    """
    if size not in {"small", "larger"}:
        raise ValueError("Unknown authored task workload")
    coordinates, updates = (1, 1) if size == "small" else (4, 32)
    return TaskLimits(
        max_coordinates=coordinates,
        max_work_units=updates * 2,
        max_attempts=2,
        max_invocations=3,
        attempt_seconds=60.0,
        max_output_bytes=4 * 1024**2,
    )


class JobObservation(CoreContract):
    """Retain actual inactive terminal state, work and independently named bytes.

    Notes
    -----
    Before-cleanup state is a real individual view, not an atomic batch snapshot.
    Full-result/state fields are verified only for completed jobs; otherwise
    their nulls are explicitly not applicable. ACK bytes select only actual
    acknowledged current component blobs, excluding orphan generations. Artifact
    bytes count all inactive job files, independently of reserved output caps.
    """

    job: TaskJobRecord
    before_cleanup: TaskJobView
    worker_active: StrictBool
    planned_updates: Annotated[int, Field(strict=True, ge=1, le=32)]
    accepted_updates: Annotated[int, Field(strict=True, ge=0, le=32)]
    acknowledged_checkpoint_bytes: Annotated[int, Field(strict=True, ge=0, le=4 * 1024**2)]
    committed_state_digest: Fingerprint | None
    complete_state_status: Literal["verified", "not_applicable"]
    mathematical_digest: Fingerprint | None
    age_seconds: Annotated[Binary64, Field(ge=0)]
    result_bytes: Annotated[int, Field(strict=True, ge=1, le=4 * 1024**2)] | None
    final_checkpoint_representation_bytes: Annotated[int, Field(strict=True, ge=1, le=4 * 1024**2)] | None
    artifact_files: Annotated[int, Field(strict=True, ge=0, le=MAX_WORKFLOW_ARTIFACT_ENTRIES)]
    artifact_bytes: Annotated[int, Field(strict=True, ge=0, le=4 * 1024**2)]

    @model_validator(mode="after")
    def _work(self):
        """Require quiescent actual outcomes and honest acknowledged work."""
        if (
            self.before_cleanup.job.job_id,
            self.before_cleanup.job.plan_id,
            self.before_cleanup.job.request_digest,
            self.before_cleanup.job.owner,
            self.before_cleanup.job.request_id,
        ) != (
            self.job.job_id,
            self.job.plan_id,
            self.job.request_digest,
            self.job.owner,
            self.job.request_id,
        ):
            raise ValueError("Pre-cleanup observations must belong to the same original logical job")
        if self.worker_active or self.job.state not in {"completed", "cancelled", "failed"}:
            raise ValueError("Task observations require terminal jobs with all actual owners inactive")
        if self.accepted_updates > self.planned_updates:
            raise ValueError("Acknowledged work exceeds the exact original planned work")
        if self.acknowledged_checkpoint_bytes == 0:
            if self.accepted_updates or self.committed_state_digest is not None:
                raise ValueError("Work or committed state requires actual acknowledged checkpoint bytes")
        elif self.committed_state_digest is None:
            raise ValueError("Acknowledged state bytes require their retained content identity")
        if self.job.state == "failed" and self.job.error not in RESOURCE_ERRORS:
            raise ValueError("Unexpected execution defects cannot become a LIMITED resource result")
        return self

    @model_validator(mode="after")
    def _result(self):
        """Keep full direct/result verification separate from partial ACK work."""
        fields = (self.mathematical_digest, self.result_bytes, self.final_checkpoint_representation_bytes)
        if self.job.state == "completed":
            if self.complete_state_status != "verified" or any(value is None for value in fields):
                raise ValueError("Completed jobs require full actual result/direct-state verification")
            if self.accepted_updates != self.planned_updates:
                raise ValueError("Completed jobs require every planned update to be acknowledged")
        elif self.complete_state_status != "not_applicable" or any(value is not None for value in fields):
            raise ValueError("Incomplete jobs have no verified complete result or final direct-state comparison")
        return self


class TaskCell(CoreContract):
    """Observe one complete or explicitly LIMITED finite burst.

    Notes
    -----
    Sequential queued-view sweeps are not atomic concurrent-depth measurements.
    Host CPU excludes workers; host peak RSS is a lifetime high-water including
    previous cells/imports. Worker CPU/RSS stay explicitly unavailable through
    the public task view. Limits and planned work never increase after a failure.
    """

    complete: StrictBool
    disposition: Literal["completed", "observation_deadline", "resource_limit"]
    observation_status: Literal["observed"]
    batch: Literal[1, 3]
    size: Literal["small", "larger"]
    planned_updates: Annotated[int, Field(strict=True, ge=1, le=96)]
    input_bytes: Annotated[int, Field(strict=True, ge=1, le=2 * 1024**2)]
    max_queued_views_per_sweep: Annotated[int, Field(strict=True, ge=0, le=3)]
    worker_owner_observed: StrictBool
    elapsed_seconds: Annotated[Binary64, Field(ge=0)]
    observation_elapsed_seconds: Annotated[Binary64, Field(ge=0)]
    observation_limit_seconds: Annotated[Binary64, Field(ge=60, le=60)]
    host_cpu_seconds: Annotated[Binary64, Field(ge=0)]
    host_peak_rss_mib: Annotated[Binary64, Field(ge=0)] | None
    worker_cpu_seconds: None
    worker_peak_rss_mib: None
    worker_resource_status: Literal["unavailable"]
    task_limits: TaskLimits
    jobs: Annotated[tuple[JobObservation, ...], Field(min_length=1, max_length=3)]

    @model_validator(mode="after")
    def _work(self):
        """Reject omitted jobs, altered plans and dishonest measured work."""
        if len(self.jobs) != self.batch or len({row.job.job_id for row in self.jobs}) != self.batch:
            raise ValueError("A task cell requires every distinct originally admitted logical job")
        if (
            len({(row.job.owner, row.job.request_id) for row in self.jobs}) != self.batch
            or len({row.job.plan_id for row in self.jobs}) != 1
        ):
            raise ValueError("A cell requires distinct retained caller requests for its one exact original plan")
        coordinates, updates = (1, 1) if self.size == "small" else (4, 32)
        if self.planned_updates != self.batch * updates:
            raise ValueError("The originally declared cell workload cannot shrink after a limit")
        if any(row.job.total_coordinates != coordinates or row.planned_updates != updates for row in self.jobs):
            raise ValueError("Job work differs from the unchanged original finite case")
        if any(row.accepted_updates < row.job.completed_coordinates * (updates // coordinates) for row in self.jobs):
            raise ValueError("Completed coordinate claims exceed actual acknowledged work")
        if self.task_limits != task_limits(self.size) or self.max_queued_views_per_sweep > self.batch:
            raise ValueError("Task observations must preserve exact selected caps and actual sweep membership")
        return self

    @model_validator(mode="after")
    def _disposition(self):
        """Admit only explicit deadlines/resource outcomes, never silent defects."""
        if self.complete != (self.disposition == "completed"):
            raise ValueError("LIMITED cells must explicitly remain incomplete")
        if (self.disposition == "observation_deadline") != (self.observation_elapsed_seconds >= CELL_SECONDS):
            raise ValueError("A late completion/resource observation cannot retroactively pass the original window")
        if self.complete and any(row.job.state != "completed" for row in self.jobs):
            raise ValueError("A completed cell cannot hide partial/cancelled/resource-failed jobs")
        if self.disposition == "observation_deadline" and all(
            row.before_cleanup.job.state == "completed" and not row.before_cleanup.worker_active for row in self.jobs
        ):
            raise ValueError("Observation limits require an actual unfinished/owned job at the deadline")
        if self.disposition == "resource_limit" and not any(row.job.state == "failed" for row in self.jobs):
            raise ValueError("Resource-limited cells require an actual declared worker resource failure")
        return self


class NotStartedCell(CoreContract):
    """Retain a declared case that cannot fit the remaining original budget.

    Notes
    -----
    No job was admitted and no physical work/resource measurement is invented.
    Planned work stays in the grid; explicit unavailable/nulls are not zeros.
    The30-second entry reserve is inside the original180-second aggregate cap.
    """

    complete: Literal[False]
    disposition: Literal["not_started"]
    reason: Literal["outer_budget"]
    observation_status: Literal["unavailable"]
    batch: Literal[1, 3]
    size: Literal["small", "larger"]
    planned_updates: Annotated[int, Field(strict=True, ge=1, le=96)]
    remaining_seconds: Annotated[Binary64, Field(ge=0, le=180)]
    jobs: Annotated[tuple[JobObservation, ...], Field(max_length=0)]
    input_bytes: None
    elapsed_seconds: None
    observation_elapsed_seconds: None
    observation_limit_seconds: Annotated[Binary64, Field(ge=60, le=60)]
    host_cpu_seconds: None
    host_peak_rss_mib: None
    max_queued_views_per_sweep: None
    worker_owner_observed: None

    @model_validator(mode="after")
    def _unattempted(self):
        """Bind the full planned case and actual insufficient entry allowance."""
        updates = 1 if self.size == "small" else 32
        if (
            self.planned_updates != self.batch * updates
            or self.remaining_seconds >= CELL_SECONDS + ENTRY_RESERVE_SECONDS
        ):
            raise ValueError("NOTSTARTED must retain the original case and its real insufficient outer budget")
        return self


class TaskLoadReport(CoreContract):
    """Retain every finite cell and separate driver from executor dependencies.

    Notes
    -----
    ``driver_runtime`` optionally binds the actual selected CLI closure. None
    means unavailable identity and is omitted from serialization, preserving
    historical v1 bytes/digests. It neither changes the executor's independently
    captured closure nor claims a missing driver identity was observed.
    """

    complete: StrictBool
    outcome: Literal["completed", "limited"]
    planned_updates: Literal[132]
    outer_limit_seconds: Annotated[Binary64, Field(ge=180, le=180)]
    entry_reserve_seconds: Annotated[Binary64, Field(ge=30, le=30)]
    acknowledged_update_lower_bound: Annotated[int, Field(strict=True, ge=0, le=132)]
    runtime: InstalledRuntime
    driver_runtime: SelectedInstalledRuntime | None = Field(default=None, exclude_if=lambda value: value is None)
    bootstrap: HostBootstrapProfile
    implementation: ExecutorImplementationIdentity
    hardware: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
    native_threads: tuple[tuple[str, str], ...]
    host_policy: HostPolicy
    driver_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    example_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    helper_sources: Annotated[
        tuple[tuple[str, Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]], ...],
        Field(min_length=2, max_length=2),
    ]
    queue_evidence: Literal["observed", "insufficient"]
    cells: Annotated[tuple[TaskCell | NotStartedCell, ...], Field(min_length=4, max_length=4)]

    @model_validator(mode="after")
    def _installed_image(self):
        """Join the exact captured approval and actual selected/base runtime."""
        if self.driver_runtime is not None:
            self.driver_runtime.require_base(self.runtime, extras=("cli",), pins=(("click", "8.1.8"),))
        if len(self.bootstrap.executors) != 1 or (
            self.bootstrap.executors[0].implementation != self.implementation.binding
            or self.bootstrap.executors[0].recipe != self.implementation.recipe
        ):
            raise ValueError("The actual captured executor image must match its explicit current approval")
        self.implementation.runtime.require_base(self.runtime)
        return self

    @model_validator(mode="after")
    def _grid(self):
        """Require all ordered cases and derive honest queue evidence status."""
        if tuple(name for name, _ in self.helper_sources) != ("task_inputs.py", "task_records.py"):
            raise ValueError("Task reports require the exact copied helper-source inventory in canonical order")
        if tuple((cell.batch, cell.size) for cell in self.cells) != CASES:
            raise ValueError("Task load reports require all four exact ordered workload cells")
        if self.host_policy != HostPolicy(schema_version=1, max_running=1, max_queued=3):
            raise ValueError("Task load reports must retain the actual selected shared host policy")
        observed = all(
            isinstance(cell, TaskCell) and cell.max_queued_views_per_sweep > 0 for cell in self.cells if cell.batch == 3
        )
        if self.queue_evidence != ("observed" if observed else "insufficient"):
            raise ValueError("Queue pressure is an observed condition, not a requested burst size")
        return self

    @model_validator(mode="after")
    def _completion(self):
        """Bind declared completion to every original cell and actual ACK sum."""
        complete = all(cell.complete for cell in self.cells)
        if self.complete != complete or self.outcome != ("completed" if complete else "limited"):
            raise ValueError("An aggregate cannot call LIMITED/partial work complete")
        if sum(cell.planned_updates for cell in self.cells) != self.planned_updates:
            raise ValueError("The original four-case grid always plans132 panel updates")
        if (
            sum(row.accepted_updates for cell in self.cells for row in cell.jobs)
            != self.acknowledged_update_lower_bound
        ):
            raise ValueError("Acknowledged lower-bound work must come from actual inspected ACK states only")
        return self

    @model_validator(mode="after")
    def _mathematical_parity(self):
        """Keep actual complete math invariant under the submitted burst size."""
        for size in ("small", "larger"):
            digests = {
                job.mathematical_digest
                for cell in self.cells
                if cell.size == size
                for job in cell.jobs
                if job.complete_state_status == "verified"
            }
            if len(digests) > 1:
                raise ValueError("Batch size must not change actual complete mathematical states")
        return self
