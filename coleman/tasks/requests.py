"""Pure generic application requests and views, independent of transports."""

from typing import Annotated, Literal

from pydantic import Field, StrictBool

from coleman.core.identity import Reference
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.tasks._base import TaskContract
from coleman.tasks.contracts import EncodedValue, PlanId, TaskCoordinate, TaskId, TaskJobRecord, TaskLimits
from coleman.tasks.identity import ExecutorRecipe


class TaskPlanningRequest(TaskContract):
    """Select a host-approved recipe and concrete bounded values, never code.

    Attributes
    ----------
    executor : Reference
        Exact already approved recipe identity and version, not an import path.
    configuration, inputs : EncodedValue
        Concrete complete values at this serialization edge. The service decodes
        them through the approved codecs before application callbacks or SQL.
    source, objective : DefinitionBinding
        Explicit source and objective definitions, including their interpretation.
    protocol : DefinitionBinding or None
        Exact scientific protocol or explicit not-applicable declaration.
    stage : Reference
        Explicit logical stage, independent of native build/protected stages.
    coordinates, limits : tuple of TaskCoordinate, TaskLimits
        Complete bounded work catalogue and requested operational capacities.

    Notes
    -----
    Source acquisition and coordinate construction belong to trusted adapters
    or application code. This request accepts no path, URL, factory, module,
    package installation or executable serialized object. A recipe ID alone
    cannot approve a provider. No work or provider discovery occurs here.
    """

    artifact_type: Literal["planning_request"]
    executor: Reference
    configuration: EncodedValue
    inputs: EncodedValue
    source: DefinitionBinding
    objective: DefinitionBinding
    protocol: DefinitionBinding | None
    stage: Reference
    coordinates: Annotated[tuple[TaskCoordinate, ...], Field(min_length=1, max_length=128)]
    limits: TaskLimits = TaskLimits()


class TaskExecuteRequest(TaskContract):
    """Admit one saved plan under a retained caller-scoped idempotency key.

    Attributes
    ----------
    request_id : str
        Bounded key retained after expiry. Cross-family reuse also conflicts.
    plan_id : str
        Full 64-hex owned saved-plan handle; configuration overrides are forbidden.
    """

    artifact_type: Literal["execute_request"]
    request_id: TaskId
    plan_id: PlanId


class TaskHostPolicy(TaskContract):
    """Keep generic work caps separate from unchanged native build capacities.

    Attributes
    ----------
    caps : TaskLimits
        Immutable operator-selected ceilings for new generic plans. Workspace
        queue/job/plan/output/memory pools remain the existing shared HostPolicy.

    Notes
    -----
    Stored policy is an equality constraint, never provider import authority.
    Reopening with a conflicting policy refuses rather than silently widening.
    """

    artifact_type: Literal["host_policy"]
    caps: TaskLimits = TaskLimits()
    _maximum_bytes = 16 * 1024


class TaskJobView(TaskContract):
    """Expose provider-free operational state with honest unavailable ETA.

    Attributes
    ----------
    job : TaskJobRecord
        Complete typed operational metadata, separate from scientific evidence.
    worker_active : bool
        Actual kernel-lock observation; a state label alone proves no ownership.
    elapsed_seconds : float
        Nonnegative actual epoch elapsed seconds, never a source clock or fee.
    eta_seconds, eta_status : None, {'unavailable'}
        No estimated duration is invented; observed zero is not missingness.
    """

    artifact_type: Literal["job_view"]
    job: TaskJobRecord
    worker_active: StrictBool
    elapsed_seconds: Annotated[Binary64, Field(ge=0)]
    eta_seconds: None = None
    eta_status: Literal["unavailable"] = "unavailable"


class TaskPlanPreview(TaskContract):
    """Return the exact plan handle and work semantics without full source bodies.

    Attributes
    ----------
    plan_id, digest : str
        Bare owned content handle and full immutable SHA-256 identity.
    executor, work_profile : Reference, DefinitionBinding
        Selected recipe and exact declared work-unit interpretation.
    coordinates, target_work_units : int
        Logical trajectory count and sum of target positions, not attempts or
        independent scientific observations. Worst-case attempts are separately
        bounded by the saved TaskLimits.
    """

    artifact_type: Literal["plan_preview"]
    plan_id: PlanId
    digest: Fingerprint
    executor: Reference
    work_profile: DefinitionBinding
    coordinates: Annotated[int, Field(strict=True, ge=1, le=128)]
    target_work_units: Annotated[int, Field(strict=True, ge=0, le=100000)]


class TaskDiscovery(TaskContract):
    """Expose only currently supplied operator approval and immutable host caps.

    Attributes
    ----------
    bootstrap : DefinitionBinding or None
        Current explicitly supplied approval, or None for provider-free access.
    executors : tuple of ExecutorRecipe
        Approved static concrete codec/work recipes, not autodiscovered plugins.
    policy : TaskHostPolicy
        Separate finite generic ceilings. Common workspace pools are not changed.

    Notes
    -----
    Codec schema digests identify definitions; they are not a substitute for
    provider-authored human examples or dynamically generated JSON schemas.
    Discovery reads no source outcomes and performs no mathematical update.
    """

    artifact_type: Literal["discovery"]
    bootstrap: DefinitionBinding | None
    executors: Annotated[tuple[ExecutorRecipe, ...], Field(max_length=16)]
    policy: TaskHostPolicy


class TaskJobCommand(TaskContract):
    """Select an owned task handle without accepting host paths or principals.

    Attributes
    ----------
    job_id : str
        Bounded host-issued task identity. Owner comes from trusted host setup.
    """

    artifact_type: Literal["job_command"]
    job_id: TaskId


class TaskJobList(TaskContract):
    """Return a bounded reconnect snapshot without embedding source outcomes.

    Attributes
    ----------
    jobs : tuple of TaskJobView
        At most 128 caller-owned known snapshots, under the shared job ceiling.
    """

    artifact_type: Literal["job_list"]
    jobs: Annotated[tuple[TaskJobView, ...], Field(max_length=128)]


class TaskCleanup(TaskContract):
    """Report expired owned handles while their idempotency tombstones remain.

    Attributes
    ----------
    expired_jobs : tuple of str
        At most 128 owned handles; no arbitrary path is returned or accepted.
    """

    artifact_type: Literal["cleanup"]
    expired_jobs: Annotated[tuple[TaskId, ...], Field(max_length=128)]


class TaskFailure(TaskContract):
    """Expose a bounded expected transport failure without host exception text.

    Attributes
    ----------
    error : str
        Invalid input, absent owned handle, explicit conflict/limit or unavailable
        storage. An error is never serialized as a successful result or zero.
    """

    artifact_type: Literal["failure"]
    error: Literal["invalid_input", "not_found", "conflict_or_limit", "storage_unavailable"]
