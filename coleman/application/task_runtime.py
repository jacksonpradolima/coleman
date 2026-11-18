"""Compile an explicitly approved concrete task for the one execution driver."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from coleman.application.execution_program import CoordinateWork, ExecutionProgram
from coleman.application.executor_capture import load_approved_executor
from coleman.application.workflow_store import WorkflowStore
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import TaskJobRecord, TaskPlan
from coleman.tasks.identity import CodecDeclaration, HostBootstrapProfile
from coleman.tasks.protocols import (
    AttemptContext,
    CommittedProgress,
    CoordinateContext,
    CoordinateOperationV1,
    PreparationContext,
    PreparedExecutor,
    ProgressVerifier,
    PublicationReadbackV1,
    StageBoundary,
    ValueCodec,
    coordinate_operation,
    decode_value,
    encode_value,
)
from coleman.tasks.results import TaskCompletion


@dataclass(frozen=True)
class CompletionCodec:
    """Read only the known outer completion; provider validation stays explicit.

    Notes
    -----
    This codec's identity covers this versioned wrapper schema. It does not
    replace, infer or authorize the selected concrete final-result codec.
    """

    @property
    def declaration(self) -> CodecDeclaration:
        """Return the fixed wrapper definition, schema identity and 16 MiB cap."""
        schema = fingerprint(TaskCompletion.model_json_schema())
        return CodecDeclaration(
            binding=DefinitionBinding(
                reference=Reference(id="coleman/task-completion", version="1"),
                definition_fingerprint=fingerprint(["coleman/task-completion-codec-v1", schema, 16777216]),
            ),
            schema_fingerprint=schema,
            maximum_bytes=16777216,
        )

    def validate(self, value: object, /) -> TaskCompletion:
        """Revalidate all outer fields and joins, with no provider or storage I/O.

        Parameters
        ----------
        value : object
            Complete task completion candidate, including unchecked instances.

        Returns
        -------
        TaskCompletion
            Revalidated known outer value; concrete body validation is separate.
        """
        return TaskCompletion.model_validate(value)

    def encode(self, value: TaskCompletion, /) -> str:
        """Encode a complete outer value deterministically without migrations.

        Parameters
        ----------
        value : TaskCompletion
            Complete known outer result.

        Returns
        -------
        str
            Canonical finite bounded UTF-8 JSON, retaining concrete body bytes.
        """
        return self.validate(value).to_json()

    def decode(self, body: str, /) -> TaskCompletion:
        """Read bounded known JSON; unknown versions and duplicate keys reject.

        Parameters
        ----------
        body : str
            Complete outer JSON, at most 16 MiB, never an executable selector.

        Returns
        -------
        TaskCompletion
            Typed outer result with the original concrete body untouched.
        """
        return TaskCompletion.from_json(body)


@dataclass(frozen=True)
class _Boundary:
    """Preserve the actual scientific boundary around the common work callback."""

    selected: StageBoundary[Any, Any]
    plan: TaskPlan
    codec: ValueCodec[Any]

    def produce(self, load_completed: Callable[[], tuple[Any, ...]], /) -> TaskCompletion:
        """Wrap the selected concrete final result after its original boundary."""
        result = encode_value(self.codec, self.selected.produce(load_completed))
        completion = TaskCompletion(
            schema_version=1,
            contract="coleman-tasks-v1",
            artifact_type="completion",
            plan_fingerprint=self.plan.fingerprint,
            source=self.plan.source,
            implementation=self.plan.implementation.binding,
            stage=self.plan.stage,
            coordinates=tuple(item.fingerprint for item in self.plan.coordinates),
            result=result,
        )
        completion.validate_for(self.plan)
        return completion


@dataclass(frozen=True)
class _Progress:
    """Validate the selected concrete proof before actual storage verification."""

    selected: ProgressVerifier[Any]
    codec: ValueCodec[Any]

    def _concrete(self, value: CommittedProgress[Any]) -> CommittedProgress[Any]:
        """Preserve joins while decoding proof through its exact selected codec."""
        proof = decode_value(self.codec, encode_value(self.codec, value.proof))
        return CommittedProgress(value.plan_fingerprint, value.coordinate_fingerprint, value.frontier, proof)

    def validate(self, progress: CommittedProgress[Any], context: CoordinateContext, /) -> None:
        """Check concrete proof shape, then the actual current acknowledged store."""
        self.selected.validate(self._concrete(progress), context)

    def reconcile(self, context: CoordinateContext, /) -> CommittedProgress[Any] | None:
        """Recover actual current proof without turning declarations into ACKs."""
        match self.selected.reconcile(context):
            case None:
                return None
            case actual:
                return self._concrete(actual)


@dataclass(frozen=True)
class _Prepared:
    """Keep concrete coordinate validation at each common driver callback."""

    selected: PreparedExecutor[Any, Any, Any, Any]
    result_codec: ValueCodec[Any]
    boundary: StageBoundary[Any, TaskCompletion]
    progress_verifier: ProgressVerifier[Any] | None
    progress_codec: ValueCodec[Any] | None = None

    @property
    def coordinate_operation(self) -> CoordinateOperationV1[Any, Any, Any, Any] | None:
        """Wrap an optional scope with the same declared concrete host codecs."""
        selected = coordinate_operation(self.selected)
        return None if selected is None else _CoordinateOperation(selected, self)

    def read_completed(self, coordinate: Any, context: CoordinateContext, /) -> Any | None:
        """Read and concretely validate retained evidence; corruption is an error."""
        match self.selected.read_completed(coordinate, context):
            case None:
                return None
            case value:
                return decode_value(self.result_codec, encode_value(self.result_codec, value))

    def advance(
        self, coordinate: Any, context: AttemptContext, committed: Callable[[CommittedProgress[Any]], None], /
    ) -> Any:
        """Delegate one reserved attempt, preserving the actual committed callback."""
        return decode_value(
            self.result_codec, encode_value(self.result_codec, self.selected.advance(coordinate, context, committed))
        )

    def publish_coordinate(self, coordinate: Any, value: Any, context: CoordinateContext, /) -> None:
        """Publish through the owner, then verify its complete concrete readback."""
        publication = _publication_readback(self.selected)
        encoded = encode_value(self.result_codec, value)
        if publication is None:
            self.selected.publish_coordinate(coordinate, decode_value(self.result_codec, encoded), context)
            retained = self.read_completed(coordinate, context)
        else:
            actual = publication.publish_and_read(coordinate, decode_value(self.result_codec, encoded), context)
            retained = (
                None if actual is None else decode_value(self.result_codec, encode_value(self.result_codec, actual))
            )
        if retained is None or encode_value(self.result_codec, retained) != encoded:
            raise ValueError("Task coordinate publication differs from its concrete complete result")


@dataclass
class _Lifetime:
    """Reject leaked host operations after their own scope has closed."""

    active: bool = True

    def require_open(self) -> None:
        """Refuse use outside the corresponding entered coordinate operation."""
        if not self.active:
            raise ValueError("Task coordinate operation is closed")


@dataclass(frozen=True)
class _ScopedProgress:
    """Preserve concrete proof validation while binding its callback lifetime."""

    selected: ProgressVerifier[Any]
    lifetime: _Lifetime

    def validate(self, progress: CommittedProgress[Any], context: CoordinateContext, /) -> None:
        """Require the same live scope before proof and actual ACK verification."""
        self.lifetime.require_open()
        self.selected.validate(progress, context)

    def reconcile(self, context: CoordinateContext, /) -> CommittedProgress[Any] | None:
        """Recover evidence only while its coordinate operation remains open."""
        self.lifetime.require_open()
        return self.selected.reconcile(context)


@dataclass(frozen=True)
class _ScopedPrepared:
    """Expose only lifetime-bound concrete operations to the common driver."""

    selected: _Prepared
    lifetime: _Lifetime

    @property
    def boundary(self) -> StageBoundary[Any, TaskCompletion]:
        """Retain the already-selected outer scientific stage boundary."""
        self.lifetime.require_open()
        return self.selected.boundary

    @property
    def progress_verifier(self) -> ProgressVerifier[Any] | None:
        """Keep a captured verifier unusable after closing this exact scope."""
        self.lifetime.require_open()
        selected = self.selected.progress_verifier
        return None if selected is None else _ScopedProgress(selected, self.lifetime)

    def read_completed(self, coordinate: Any, context: CoordinateContext, /) -> Any | None:
        """Read complete concrete evidence within this live operation."""
        self.lifetime.require_open()
        return self.selected.read_completed(coordinate, context)

    def advance(
        self, coordinate: Any, context: AttemptContext, committed: Callable[[CommittedProgress[Any]], None], /
    ) -> Any:
        """Preserve the reserved attempt and common progress callback."""
        self.lifetime.require_open()
        return self.selected.advance(coordinate, context, committed)

    def publish_coordinate(self, coordinate: Any, value: Any, context: CoordinateContext, /) -> None:
        """Publish and validate actual readback before this operation closes."""
        self.lifetime.require_open()
        self.selected.publish_coordinate(coordinate, value, context)


@dataclass(frozen=True)
class _CoordinateOperation:
    """Apply existing result and proof codecs to an optional acquired provider."""

    selected: CoordinateOperationV1[Any, Any, Any, Any]
    parent: _Prepared

    @property
    def profile(self) -> Reference:
        """Retain the actual provider declaration for the pre-open check."""
        return self.selected.profile

    @contextmanager
    def open(self, coordinate: Any, context: CoordinateContext, /) -> Iterator[_ScopedPrepared]:
        """Close on every path and never suppress a common-driver failure."""
        lifetime = _Lifetime()
        failure: BaseException | None = None
        try:
            with self.selected.open(coordinate, context) as selected:
                try:
                    _require_capabilities(selected, require_progress=self.parent.progress_verifier is not None)
                    wrapped = _Prepared(
                        selected,
                        self.parent.result_codec,
                        self.parent.boundary,
                        _progress(selected, self.parent.progress_codec),
                        self.parent.progress_codec,
                    )
                    yield _ScopedPrepared(wrapped, lifetime)
                except BaseException as error:
                    # Cleanup must cover cancellation and process-level interrupts too.
                    failure = error
                    raise
                finally:
                    lifetime.active = False
            if failure is not None:
                raise failure  # A provider cannot turn a suppressed driver error into success.
        finally:
            lifetime.active = False


def prepare_task_program(
    store: WorkflowStore,
    job: TaskJobRecord,
    plan: TaskPlan,
    *,
    bootstrap_profile: HostBootstrapProfile | None,
    guard: Callable[[], None],
) -> ExecutionProgram[Any, Any, TaskCompletion, Any]:
    """Compile one verified installed executor outside metadata transactions.

    Parameters
    ----------
    store : WorkflowStore
        Existing trusted workspace, shared with the native executor.
    job, plan : TaskJobRecord, TaskPlan
        Complete retained metadata, checked together before provider discovery.
    bootstrap_profile : HostBootstrapProfile or None
        Independently supplied current operator approval. None refuses generic
        work; a stored profile cannot reconstruct authority.
    guard : callable
        Shared cancellation, wall-time, memory and output checks.

    Returns
    -------
    ExecutionProgram
        Concrete decoded coordinates and operations for the one shared driver.

    Raises
    ------
    ValueError, TypeError, OSError, ImportError
        Missing approval, changed installation/codecs, invalid concrete values
        or missing selected execution/progress capabilities. No implicit reset,
        package installation, migration or fallback to another executor occurs.

    Notes
    -----
    Rechecks the complete image before bootstrap, source preparation or restore.
    The immutable trusted installation must remain unchanged during execution.
    The provider owns mathematical state and actual ACK verification. This
    function creates no queue, work loop, transaction, learner or checkpoint.
    """
    current = TaskPlan.model_validate(plan)
    TaskJobRecord.model_validate(job).validate_for(current)
    if bootstrap_profile is None:
        raise ValueError("Generic task execution requires the current explicit host bootstrap profile")
    current.validate_bootstrap(bootstrap_profile)
    approval = next(item for item in bootstrap_profile.executors if item.recipe.binding == current.executor)
    guard()
    executor = load_approved_executor(approval, expected=current.implementation)
    configuration = decode_value(executor.configuration_codec, current.configuration)
    inputs = decode_value(executor.input_codec, current.inputs)
    coordinates = tuple(decode_value(executor.coordinate_codec, item.body) for item in current.coordinates)
    context = PreparationContext(
        current.fingerprint,
        current.source,
        current.implementation.binding,
        ExecutionCompatibility(
            schema_version=1,
            runtime=current.implementation.runtime,
            owner_implementations=(current.implementation.binding,),
        ),
        store.root / "jobs" / job.job_id,
    )
    guard()
    selected = executor.prepare(configuration, inputs, context)
    _require_prepared(selected, current)
    proof = _progress(selected, executor.progress_proof_codec)
    guard()
    works = tuple(
        CoordinateWork(
            item.fingerprint,
            item.reference,
            item.target,
            value,
            context.job_root / "evidence" / item.fingerprint.removeprefix("sha256:"),
            seed=item.seed,
        )
        for item, value in zip(current.coordinates, coordinates, strict=True)
    )
    return ExecutionProgram(
        context,
        works,
        _Prepared(
            selected,
            executor.coordinate_result_codec,
            _Boundary(selected.boundary, current, executor.final_result_codec),
            proof,
            executor.progress_proof_codec,
        ),
        CompletionCodec(),
        lambda value: value.fingerprint,
        context.job_root / "complete.json",
    )


def _progress(
    selected: PreparedExecutor[Any, Any, Any, Any], codec: ValueCodec[Any] | None
) -> ProgressVerifier[Any] | None:
    """Join optional proof codec and actual verifier before any reservation."""
    if selected.progress_verifier is None:
        return None
    if codec is None:
        raise ValueError("A task progress verifier requires its declared concrete proof codec")
    return _Progress(selected.progress_verifier, codec)


def _require_prepared(value: object, plan: TaskPlan) -> None:
    """Refuse incomplete declared capabilities before reserving physical work."""
    _require_capabilities(value, require_progress=plan.implementation.recipe.effects == "local_state")


def _require_capabilities(value: object, *, require_progress: bool) -> None:
    """Validate actual operations before entering the shared coordinate loop."""
    for name in ("read_completed", "advance", "publish_coordinate"):
        if not callable(getattr(value, name, None)):
            raise TypeError("Prepared task executor is missing a required coordinate operation")
    if not callable(getattr(getattr(value, "boundary", None), "produce", None)):
        raise TypeError("Prepared task executor requires a concrete stage boundary")
    verifier = getattr(value, "progress_verifier", None)
    if require_progress and verifier is None:
        raise ValueError("A local-state task requires actual progress verification and reconciliation")
    if verifier is not None and not all(callable(getattr(verifier, name, None)) for name in ("validate", "reconcile")):
        raise TypeError("Task progress capability requires actual validation and reconciliation")
    _publication_readback(value)
    coordinate_operation(value)


def _publication_readback(value: object) -> PublicationReadbackV1[Any, Any] | None:
    """Validate optional trusted capability metadata before physical publication.

    Parameters
    ----------
    value : object
        Actual prepared provider from the independently captured implementation.

    Returns
    -------
    PublicationReadbackV1 or None
        Supported complete operation, or the unchanged legacy publication path.

    Raises
    ------
    ValueError, TypeError
        Malformed or unknown profile, or a missing concrete operation. Unknown
        future versions never silently select the legacy path.

    Notes
    -----
    Runs during preparation before reservation and again before publication.
    Reads only the provider's capability declaration, never a request flag,
    artifact-selected executable, cached snapshot or storage acknowledgement.
    The declared profile is not evidence that its callbacks performed I/O.
    """
    capability = getattr(value, "publication_readback", None)
    if capability is None:
        return None
    profile = Reference.model_validate(getattr(capability, "profile", None))
    if profile != Reference(id="coleman-coordinate-publication-readback", version="1"):
        raise ValueError("Unsupported task coordinate publication/readback profile")
    if not callable(getattr(capability, "publish_and_read", None)):
        raise TypeError("Task publication/readback capability requires its concrete operation")
    return cast(PublicationReadbackV1[Any, Any], capability)
