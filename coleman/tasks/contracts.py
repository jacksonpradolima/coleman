"""Strict task plan, resource and progress declarations without native coupling."""

from __future__ import annotations

from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, decode_json, fingerprint
from coleman.core.reading import bounded_json_bytes
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.storage_contracts import AcknowledgedHead, CommitReceipt, CommitRequest, ExecutionCompatibility
from coleman.tasks._base import TaskContract
from coleman.tasks.identity import CodecDeclaration, ExecutorImplementationIdentity, HostBootstrapProfile

TaskId = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
PlanId = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
Position = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
TaskState = Literal[
    "queued", "running", "awaiting_evaluation", "interrupted", "failed", "cancelled", "completed", "expired"
]


def _finite_body(value: object) -> None:
    """Reject nonstandard JSON constants without changing exact decimal tokens.

    Parameters
    ----------
    value : object
        Duplicate-safe decoded body whose lexical depth/bytes are already bound.

    Raises
    ------
    ValueError
        A standard-library NaN/Infinity extension occurs outside a string.

    Notes
    -----
    Decimal tokens remain exact; concrete codecs separately enforce numerical
    interpretation. This does not normalize or re-encode the original body.
    """
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Task encoded bodies require finite JSON constants")
    if isinstance(value, dict):
        for child in value.values():
            _finite_body(child)
    elif isinstance(value, list):
        for child in value:
            _finite_body(child)


class EncodedValue(CoreImmutable):
    """Retain one concrete codec's exact canonical body at a persistence edge.

    Attributes
    ----------
    codec : CodecDeclaration
        Frozen concrete interpretation and selected byte ceiling.
    body : str
        Complete UTF-8 JSON. Concrete canonicality/fields require that codec.

    Notes
    -----
    Reading checks bounded JSON shape only, not a dynamically selected schema.
    It does not normalize optional omissions, signed zero or decimal tokens.
    """

    codec: CodecDeclaration
    body: Annotated[str, Field(strict=True, min_length=1, max_length=16 * 1024**2)]

    @model_validator(mode="after")
    def _body(self) -> Self:
        """Bound opaque bytes/depth before any concrete host codec can run."""
        bounded_json_bytes(self.body, self.codec.maximum_bytes)
        _finite_body(decode_json(self.body, exact_decimals=True))
        return self


class TaskFrontier(CoreImmutable):
    """Name a logical committed position using explicit executor-owned units.

    Attributes
    ----------
    profile : DefinitionBinding
        Complete definition of position/work semantics, not a native ordinal.
    coordinate : Reference
        Stable logical trajectory identity, independent of physical attempts.
    position : int
        Nonnegative exact committed count. Zero is a valid virgin frontier.
    """

    profile: DefinitionBinding
    coordinate: Reference
    position: Position


class TaskCoordinate(CoreImmutable):
    """Freeze one concrete work coordinate, explicit seed and target frontier.

    Attributes
    ----------
    reference : Reference
        Stable task coordinate identity and version.
    body : EncodedValue
        Concrete coordinate data, distinct from global config and source input.
    target : TaskFrontier
        Exact terminal position; its logical coordinate must match reference.
    seed : int or None
        Explicit uint64 RNG seed, or not applicable. Zero is never missing.
    """

    reference: Reference
    body: EncodedValue
    target: TaskFrontier
    seed: Seed | None

    @model_validator(mode="after")
    def _target(self) -> Self:
        """Refuse a target borrowed from another logical trajectory."""
        if self.target.coordinate != self.reference:
            raise ValueError("Task coordinate and target frontier must have the same reference")
        return self

    @property
    def fingerprint(self) -> str:
        """Identify this exact typed coordinate without embedding a self digest."""
        return fingerprint(type(self).model_validate(self).model_dump(mode="json"))


class TaskLimits(CoreImmutable):
    """Declare finite operational caps without reinterpreting native work caps.

    Attributes
    ----------
    max_coordinates, max_work_units : int, optional
        At most 128 logical trajectories and 100000 declared work units,
        including worst-case physical attempts under one named work profile.
    max_attempts, max_invocations : int, optional
        Per-coordinate physical attempts and separate job-worker starts, at most8.
    attempt_seconds : float, optional
        Observed wall-time allowance per worker invocation, at most3600seconds.
    max_memory_bytes, max_output_bytes : int, optional
        Sampled RSS and reserved job output byte ceilings, not sandbox guarantees.
    """

    max_coordinates: Annotated[int, Field(strict=True, ge=1, le=128)] = 32
    max_work_units: Annotated[int, Field(strict=True, ge=1, le=100000)] = 100000
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=8)] = 4
    max_invocations: Annotated[int, Field(strict=True, ge=1, le=8)] = 8
    attempt_seconds: Binary64 = 300.0
    max_memory_bytes: Annotated[int, Field(strict=True, ge=67108864, le=2147483648)] = 1073741824
    max_output_bytes: Annotated[int, Field(strict=True, ge=65536, le=67108864)] = 16777216

    @model_validator(mode="after")
    def _seconds(self) -> Self:
        """Reject zero/negative/excessive real worker wall-time allowances."""
        if not 0 < self.attempt_seconds <= 3600:
            raise ValueError("Task wall-time allowance must be positive and at most 3600 seconds")
        return self


class TaskPlan(TaskContract):
    """Freeze a concrete installed task independently of its later job handle.

    Attributes
    ----------
    executor, bootstrap : DefinitionBinding
        Exact recipe and separately operator-selected bootstrap profile.
    implementation : ExecutorImplementationIdentity
        Complete installed capture; stored equality does not grant authority.
    configuration, inputs : EncodedValue
        Separate codec roles, decoded before callbacks or transaction admission.
    source, objective : DefinitionBinding
        Explicit source and objective definitions, without inferred clocks/units.
    protocol : DefinitionBinding or None
        Declared scientific protocol, or explicitly not applicable.
    stage : Reference
        Logical executor stage, distinct from native tuning/build labels.
    coordinates, limits : tuple of TaskCoordinate, TaskLimits
        Complete finite ordered trajectories under one declared work profile.
    """

    artifact_type: Literal["plan"]
    executor: DefinitionBinding
    implementation: ExecutorImplementationIdentity
    bootstrap: DefinitionBinding
    configuration: EncodedValue
    inputs: EncodedValue
    source: DefinitionBinding
    objective: DefinitionBinding
    protocol: DefinitionBinding | None
    stage: Reference
    coordinates: Annotated[tuple[TaskCoordinate, ...], Field(min_length=1, max_length=128)]
    limits: TaskLimits = TaskLimits()

    @model_validator(mode="after")
    def _plan(self) -> Self:
        """Join recipe/codecs/profile and cap work before provider execution."""
        recipe = self.implementation.recipe
        if self.executor != recipe.binding:
            raise ValueError("Task executor must match the captured complete recipe")
        if self.configuration.codec != recipe.codec_for("configuration") or self.inputs.codec != recipe.codec_for(
            "inputs"
        ):
            raise ValueError("Task configuration/input codec roles must match the executor recipe")
        coordinate_codec = recipe.codec_for("coordinate")
        if any(item.body.codec != coordinate_codec for item in self.coordinates):
            raise ValueError("Task coordinates require the declared concrete coordinate codec")
        ids = [item.reference.id for item in self.coordinates]
        if len(ids) != len(set(ids)):
            raise ValueError("Task coordinates require unique logical IDs")
        if any(item.target.profile != recipe.frontier for item in self.coordinates):
            raise ValueError("Task work budgets require one exact declared frontier profile")
        if len(self.coordinates) > self.limits.max_coordinates:
            raise ValueError("Task coordinate count exceeds its declared capacity")
        if (
            sum(item.target.position for item in self.coordinates) * self.limits.max_attempts
            > self.limits.max_work_units
        ):
            raise ValueError("Task worst-case work exceeds its declared named-profile budget")
        return self

    def validate_bootstrap(self, profile: HostBootstrapProfile) -> None:
        """Join an independently supplied host profile without granting authority.

        Parameters
        ----------
        profile : HostBootstrapProfile
            Actual operator-selected current profile, never workspace authority.

        Raises
        ------
        ValueError
            Profile binding, complete recipe or approved capture does not match.

        Notes
        -----
        The runtime separately establishes the trusted profile source and
        verifies current installation/codecs before admission or provider load.
        """
        current = type(self).model_validate(self)
        host = HostBootstrapProfile.model_validate(profile)
        if current.bootstrap != host.binding:
            raise ValueError("Task plan must match the independently supplied bootstrap profile")
        approved = next((item for item in host.executors if item.recipe.binding == current.executor), None)
        if approved is None or approved.implementation != current.implementation.binding:
            raise ValueError("Task plan capture must match the operator-approved executor recipe")


class TaskJobRecord(TaskContract):
    """Keep neutral operational progress separate from scientific evidence.

    Attributes
    ----------
    job_id, owner, request_id, request_digest, plan_id : str
        Host-issued identity, principal, idempotency key, request digest and bare
        64-hex immutable plan handle. Full plan fingerprints remain explicit.
    state, stage : str, Reference
        Finite lifecycle and exact plan-owned stage. Neither establishes a lock.
    created, changed : float
        Actual operational epoch seconds, never inferred source availability.
    current_coordinate, frontier : str or None, TaskFrontier or None
        Exact coordinate digest and acknowledged logical position when known.
    completed_coordinates, total_coordinates, attempts, invocations : int
        Distinct trajectory, physical attempt and worker-start counts.
    error, result_digest : str or None
        Bounded safe failure and complete-last result identity when published.
    """

    artifact_type: Literal["job"]
    job_id: TaskId
    owner: TaskId
    request_id: TaskId
    request_digest: Fingerprint
    plan_id: PlanId
    state: TaskState
    stage: Reference
    created: Binary64
    changed: Binary64
    current_coordinate: Fingerprint | None = None
    frontier: TaskFrontier | None = None
    completed_coordinates: Annotated[int, Field(strict=True, ge=0, le=128)] = 0
    total_coordinates: Annotated[int, Field(strict=True, ge=1, le=128)]
    attempts: Annotated[int, Field(strict=True, ge=0, le=1024)] = 0
    invocations: Annotated[int, Field(strict=True, ge=0, le=8)] = 0
    error: Annotated[str, Field(strict=True, max_length=128)] | None = None
    result_digest: Fingerprint | None = None

    @model_validator(mode="after")
    def _lifecycle(self) -> Self:
        """Reject false completion, reversed clocks or unidentified frontiers."""
        if self.created < 0 or self.changed < self.created:
            raise ValueError("Task lifecycle clocks require nonnegative nondecreasing actual seconds")
        if self.completed_coordinates > self.total_coordinates:
            raise ValueError("Completed task coordinates exceed the declared total")
        if self.frontier is not None and self.current_coordinate is None:
            raise ValueError("Task progress frontier requires an exact current coordinate")
        if self.state == "completed" and (
            self.result_digest is None or self.completed_coordinates != self.total_coordinates
        ):
            raise ValueError("Completed tasks require every coordinate and a complete result digest")
        if self.state not in {"completed", "expired"} and self.result_digest is not None:
            raise ValueError("Only completed/expired tasks retain a complete result digest")
        return self

    def validate_for(self, plan: TaskPlan) -> None:
        """Check plan-owned job fields without loading a provider or state store.

        Parameters
        ----------
        plan : TaskPlan
            Exact retained plan; its bare content handle must match this record.

        Raises
        ------
        ValueError
            Wrong plan/stage/count, coordinate/frontier, or operational caps.
        """
        current = type(self).model_validate(self)
        checked = TaskPlan.model_validate(plan)
        if current.plan_id != checked.fingerprint.removeprefix("sha256:") or current.stage != checked.stage:
            raise ValueError("Task job must match its exact retained plan and stage")
        if current.total_coordinates != len(checked.coordinates):
            raise ValueError("Task job total must match the retained coordinate catalogue")
        if current.invocations > checked.limits.max_invocations or current.attempts > (
            len(checked.coordinates) * checked.limits.max_attempts
        ):
            raise ValueError("Task job counters exceed its immutable operational caps")
        if current.current_coordinate is not None:
            selected = next(
                (item for item in checked.coordinates if item.fingerprint == current.current_coordinate), None
            )
            if selected is None:
                raise ValueError("Task job current coordinate is not in its retained plan")
            if current.frontier is not None and (
                current.frontier.coordinate != selected.reference
                or current.frontier.profile != selected.target.profile
                or current.frontier.position > selected.target.position
            ):
                raise ValueError("Task job frontier does not match the current declared trajectory")


class OwnedCommitProof(TaskContract):
    """Join actual acknowledgement declarations without pretending to read storage.

    Attributes
    ----------
    plan_fingerprint, coordinate_fingerprint : str
        Complete immutable plan and coordinate bindings.
    frontier : TaskFrontier
        Declared logical current work position. Its meaning is verified against
        the actual decoded owner state, not inferred from publication count.
    collection, compatibility : DefinitionBinding, ExecutionCompatibility
        Complete collection-definition identity and explicit execution context.
    request, receipt, observed_head : CommitRequest, CommitReceipt, AcknowledgedHead
        Actual command/receipt/current observation supplied by the runtime.

    Notes
    -----
    Pure validation checks joins only. Runtime capability must reopen actual
    storage and verify these declarations before recording outer job progress.
    A request alone, artifact or forged model does not establish acceptance.
    """

    artifact_type: Literal["owned_commit_proof"]
    plan_fingerprint: Fingerprint
    coordinate_fingerprint: Fingerprint
    frontier: TaskFrontier
    collection: DefinitionBinding
    compatibility: ExecutionCompatibility
    request: CommitRequest
    receipt: CommitReceipt
    observed_head: AcknowledgedHead

    @model_validator(mode="after")
    def _acknowledged(self) -> Self:
        """Require a matching original command and nonrewinding current head."""
        accepted = self.receipt.accepted_head
        expected = self.request.expected
        if accepted.intent_fingerprint != self.request.intent.fingerprint:
            raise ValueError("Owned task proof receipt must acknowledge the exact original intent")
        if accepted.generation != self.request.intent.generation:
            raise ValueError("Owned task proof intent and acknowledged generation differ")
        if self.observed_head != self.receipt.current_head:
            raise ValueError("Owned task proof must retain the matching observed current head")
        if accepted.definition_fingerprint != self.collection.definition_fingerprint:
            raise ValueError("Owned task proof requires its exact collection definition")
        if self.observed_head.definition_fingerprint != self.collection.definition_fingerprint:
            raise ValueError("Owned task proof current head changed its collection definition")
        if expected is None:
            if self.request.intent.kind != "initialize" or accepted.generation != 0:
                raise ValueError("Virgin owned task proof requires initialization generation zero")
        elif (
            self.request.intent.kind == "initialize"
            or expected.generation + 1 != accepted.generation
            or (expected.session, expected.source_epoch, expected.definition_fingerprint)
            != (accepted.session, accepted.source_epoch, accepted.definition_fingerprint)
        ):
            raise ValueError("Owned task proof must join its exact consecutive predecessor")
        return self

    def validate_for(self, plan: TaskPlan, coordinate: TaskCoordinate) -> None:
        """Check supplied plan/coordinate/runtime joins without opening storage.

        Parameters
        ----------
        plan : TaskPlan
            Complete retained plan, not a SQL handle or unchecked dictionary.
        coordinate : TaskCoordinate
            Complete selected coordinate from that exact plan.

        Raises
        ------
        ValueError
            Wrong plan, coordinate, frontier, runtime or owner implementation.

        Notes
        -----
        This is declaration consistency only. Actual current store ACK and
        installation verification remain mandatory runtime responsibilities.
        """
        current = type(self).model_validate(self)
        checked = TaskPlan.model_validate(plan)
        selected = TaskCoordinate.model_validate(coordinate)
        if current.plan_fingerprint != checked.fingerprint or current.coordinate_fingerprint != selected.fingerprint:
            raise ValueError("Owned task proof must bind its complete retained plan and coordinate")
        if selected not in checked.coordinates:
            raise ValueError("Owned task proof coordinate is not a member of the retained plan")
        if (
            current.frontier.profile != selected.target.profile
            or current.frontier.coordinate != selected.reference
            or current.frontier.position > selected.target.position
        ):
            raise ValueError("Owned task proof frontier does not match the declared coordinate")
        if current.compatibility.runtime != checked.implementation.runtime:
            raise ValueError("Owned task proof runtime differs from its captured implementation")
        if checked.implementation.binding not in current.compatibility.owner_implementations:
            raise ValueError("Owned task proof must retain its captured owner implementation binding")
