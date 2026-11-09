"""Structural typed executor capabilities for one shared lifecycle coordinator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import EncodedValue, TaskFrontier
from coleman.tasks.identity import CodecDeclaration


class ValueCodec[T](Protocol):
    """Supply a selected concrete codec without requiring model inheritance.

    Notes
    -----
    Existing DomainRecord/WorkflowReport serializers remain valid. Encode and
    decode must preserve the complete concrete value and exact declared bytes;
    metadata/schema equality alone does not establish that property.
    """

    @property
    def declaration(self) -> CodecDeclaration:
        """Return the immutable interpretation selected by trusted host code."""
        ...

    def validate(self, value: object, /) -> T:
        """Validate a complete concrete value without I/O or coercive fallback.

        Parameters
        ----------
        value : object
            Candidate complete value, possibly an unchecked model instance.

        Returns
        -------
        T
            Revalidated concrete value retaining every declared field.
        """
        ...

    def decode(self, body: str, /) -> T:
        """Read exact bounded canonical content using the selected concrete type.

        Parameters
        ----------
        body : str
            Complete UTF-8 JSON text; no filename or dynamically selected type.

        Returns
        -------
        T
            Complete concrete value with unknown fields/versions rejected.
        """
        ...

    def encode(self, value: T, /) -> str:
        """Write a revalidated complete value within the declared byte ceiling.

        Parameters
        ----------
        value : T
            Concrete complete value, retaining optional omissions and units.

        Returns
        -------
        str
            Exact canonical UTF-8 JSON reversible by this same concrete codec.
        """
        ...


def decode_value[T](codec: ValueCodec[T], encoded: EncodedValue, /) -> T:
    """Decode an edge value with an independently selected concrete codec.

    Parameters
    ----------
    codec : ValueCodec
        Actual trusted concrete codec, never imported from encoded metadata.
    encoded : EncodedValue
        Complete bounded edge value and its exact declared interpretation.

    Returns
    -------
    T
        Validated concrete value whose canonical re-encoding is byte-identical.

    Raises
    ------
    ValueError, TypeError
        Wrong codec binding, invalid concrete content or noncanonical bytes.

    Notes
    -----
    Caller executes trusted codec callbacks outside metadata transactions.
    Byte reversibility is not a proof of arbitrary custom snapshot completeness.
    """
    declaration = CodecDeclaration.model_validate(codec.declaration)
    current = EncodedValue.model_validate(encoded)
    if current.codec != declaration:
        raise ValueError("Encoded task value does not match the selected concrete codec")
    value = codec.validate(codec.decode(current.body))
    if codec.encode(value) != current.body:
        raise ValueError("Encoded task value must retain the exact canonical concrete bytes")
    return value


def encode_value[T](codec: ValueCodec[T], value: T, /) -> EncodedValue:
    """Encode a concrete value and verify the same codec can read its exact bytes.

    Parameters
    ----------
    codec : ValueCodec
        Actual independently selected concrete interpretation.
    value : T
        Complete value; no CoreContract inheritance is required.

    Returns
    -------
    EncodedValue
        Complete bounded edge declaration, reversible with the same codec.

    Raises
    ------
    ValueError, TypeError
        Invalid concrete state, bytes, codec declaration or round-trip refusal.
    """
    encoded = EncodedValue(codec=codec.declaration, body=codec.encode(codec.validate(value)))
    decode_value(codec, encoded)
    return encoded


@dataclass(frozen=True)
class PreparationContext:
    """Pass admitted neutral execution bindings outside metadata transactions.

    Attributes
    ----------
    plan_fingerprint : str
        Exact complete plan digest, whether native or generic; never a SQL handle.
    source, implementation : DefinitionBinding
        Exact retained source and actual implementation identity declarations.
    compatibility : ExecutionCompatibility
        Actual current runtime/owner definitions, independently captured.
    job_root : pathlib.Path
        Trusted bounded host artifact location, never selected by request JSON.
    """

    plan_fingerprint: Fingerprint
    source: DefinitionBinding
    implementation: DefinitionBinding
    compatibility: ExecutionCompatibility
    job_root: Path


@dataclass(frozen=True)
class CoordinateContext:
    """Bind one shared-driver coordinate to its trusted artifact directory.

    Attributes
    ----------
    preparation : PreparationContext
        Exact admitted plan, runtime and trusted job root.
    coordinate_fingerprint, reference : str, Reference
        Exact coordinate content digest and stable logical identity.
    target : TaskFrontier
        Explicit named work semantics. Native profiles are ephemeral views and
        do not rewrite retained native build counters or coordinate wires.
    artifact_directory : pathlib.Path
        Driver-selected per-coordinate directory, outside metadata transactions.
    seed : int or None, optional
        Exact declared uint64 seed for this coordinate. Zero is valid; None
        means no seed is declared. It is separate from global preparation.

    Notes
    -----
    The coordinator never seeds or reseeds a global RNG. A provider may use the
    seed to initialize its owned RNG once, then restore its complete owned state
    on continuation instead of reseeding for each attempt.
    """

    preparation: PreparationContext
    coordinate_fingerprint: Fingerprint
    reference: Reference
    target: TaskFrontier
    artifact_directory: Path
    seed: int | None = None


@dataclass(frozen=True)
class AttemptContext:
    """Supply an actual reserved attempt and common resource guard to a kernel.

    Attributes
    ----------
    coordinate : CoordinateContext
        Exact trajectory and immutable surrounding execution bindings.
    attempt_id : str
        Stable actual physical reservation, created before advance effects.
    invocation : int
        Actual bounded worker-start count, distinct from learner updates.
    deadline : float
        Finite monotonic worker deadline, not a source or outcome clock.
    guard : callable
        Shared ownership/cancellation/deadline/RSS/output check.
    """

    coordinate: CoordinateContext
    attempt_id: str
    invocation: int
    deadline: float
    guard: Callable[[], None]


@dataclass(frozen=True)
class CommittedProgress[P]:
    """Carry concrete progress evidence for validation before outer metadata.

    Attributes
    ----------
    plan_fingerprint, coordinate_fingerprint : str
        Full immutable plan/coordinate content identities, not SQL handles.
    frontier : TaskFrontier
        Actual declared committed position under the selected work profile.
    proof : P
        Selected concrete receipt. A request or integer alone is insufficient.
    """

    plan_fingerprint: str
    coordinate_fingerprint: str
    frontier: TaskFrontier
    proof: P


class StageBoundary[R, O](Protocol):  # pylint: disable=too-few-public-methods
    """Wrap the common loader in the adapter's existing scientific scope.

    Notes
    -----
    May invoke the common loader at most once. A verified cached report may
    invoke it zero times. The shared driver supplies the one-shot effect guard.
    No adapter owns a second scheduling/admission/coordinate lifecycle loop.
    """

    def produce(self, load_completed: Callable[[], tuple[R, ...]], /) -> O:
        """Produce or reuse complete evidence under a declared adapter scope.

        Parameters
        ----------
        load_completed : callable
            Common coordinate driver; second calls must reject before effects.

        Returns
        -------
        O
            Complete final result, with no automatic scientific qualification.
        """
        ...


class ProgressVerifier[P](Protocol):
    """Verify actual store acceptance and state-derived logical work positions.

    Notes
    -----
    This is an optional runtime capability, distinct from a pure proof codec.
    Storage generations count publications; logical work is derived from the
    decoded current owner state under the selected named frontier profile.
    """

    def validate(self, progress: CommittedProgress[P], context: CoordinateContext, /) -> None:
        """Check actual current ACK, bindings and semantic frontier before metadata.

        Parameters
        ----------
        progress : CommittedProgress
            Complete candidate progress and its selected concrete proof.
        context : CoordinateContext
            Expected retained plan, coordinate, runtime and artifact location.
        """
        ...

    def reconcile(self, context: CoordinateContext, /) -> CommittedProgress[P] | None:
        """Recover actual current progress after an inner ACK/outer metadata gap.

        Parameters
        ----------
        context : CoordinateContext
            Complete expected trajectory and execution bindings.

        Returns
        -------
        CommittedProgress or None
            Actual validated current position, or an explicitly virgin state.
            Corrupt/incompatible/missing acknowledged restoration must raise.
        """
        ...


class OwnedContinuation[S, P](ProgressVerifier[P], Protocol):
    """Optionally restore/reconcile complete state without fake stateless state.

    Notes
    -----
    Selected by trusted capability wiring, not runtime reflection on JSON. An
    implementation reopens actual storage and compares current ACK/bindings;
    pure receipt validation alone never establishes that a commit happened.
    """

    @property
    def state_codec(self) -> ValueCodec[S]:
        """Return the selected complete continuation codec."""
        ...

    def restore(self, state: S, context: CoordinateContext, /) -> None:
        """Restore one validated complete state without silent fresh fallback.

        Parameters
        ----------
        state : S
            Complete state recovered with the exact selected state codec.
        context : CoordinateContext
            Exact immutable plan, coordinate and runtime bindings.
        """
        ...


class PreparedExecutor[W, R, O, P](Protocol):
    """Provide bounded coordinate operations; the shared driver owns all loops."""

    @property
    def boundary(self) -> StageBoundary[R, O]:
        """Return the native scientific or generic descriptive scope wrapper."""
        ...

    @property
    def progress_verifier(self) -> ProgressVerifier[P] | None:
        """Return actual progress verification/recovery, or stateless absence."""
        ...

    def read_completed(self, coordinate: W, context: CoordinateContext, /) -> R | None:
        """Read and verify a complete prior result, or return unavailable.

        Parameters
        ----------
        coordinate : W
            Decoded selected concrete coordinate.
        context : CoordinateContext
            Exact retained bindings and trusted artifact location.

        Returns
        -------
        R or None
            Complete verified result or absence; corruption must raise.
        """
        ...

    def advance(
        self, coordinate: W, context: AttemptContext, committed: Callable[[CommittedProgress[P]], None], /
    ) -> R:
        """Advance one reserved trajectory using its existing mathematical core.

        Parameters
        ----------
        coordinate : W
            Decoded selected concrete trajectory, never the whole job schedule.
        context : AttemptContext
            Already reserved physical attempt and shared guard callback.
        committed : callable
            Progress callback only after actual concrete acknowledgement.

        Returns
        -------
        R
            Complete trajectory result. Outer publication remains complete-last.
        """
        ...

    def publish_coordinate(self, coordinate: W, value: R, context: CoordinateContext, /) -> None:
        """Publish the selected concrete result without claiming job completion.

        Parameters
        ----------
        coordinate : W
            Exact decoded coordinate.
        value : R
            Complete result revalidated with the selected result codec.
        context : CoordinateContext
            Trusted bounded per-coordinate artifact location and bindings.
        """
        ...


class Executor[C, I, W, R, O, P](Protocol):  # pylint: disable=too-few-public-methods
    """Expose separate selected codecs and a per-coordinate executor factory.

    Notes
    -----
    Preparation/capture/codec callbacks occur outside metadata transactions.
    Direct mathematical callers need no task wrapper or runtime metadata.
    """

    @property
    def definition(self) -> DefinitionBinding:
        """Return the exact immutable executor definition selected by the host."""
        ...

    @property
    def configuration_codec(self) -> ValueCodec[C]:
        """Return the independently selected complete configuration codec."""
        ...

    @property
    def input_codec(self) -> ValueCodec[I]:
        """Return the selected source-input codec, separate from configuration."""
        ...

    @property
    def coordinate_codec(self) -> ValueCodec[W]:
        """Return the selected concrete per-trajectory coordinate codec."""
        ...

    @property
    def coordinate_result_codec(self) -> ValueCodec[R]:
        """Return the complete per-coordinate result codec."""
        ...

    @property
    def final_result_codec(self) -> ValueCodec[O]:
        """Return the complete final adapter-evidence codec."""
        ...

    @property
    def progress_proof_codec(self) -> ValueCodec[P] | None:
        """Return the selected progress codec, or explicit stateless absence."""
        ...

    def prepare(self, configuration: C, inputs: I, context: PreparationContext, /) -> PreparedExecutor[W, R, O, P]:
        """Prepare concrete operations without acquiring queue/store ownership.

        Parameters
        ----------
        configuration : C
            Complete selected configuration, already concretely validated.
        inputs : I
            Separate complete source input, without duplicate opaque mappings.
        context : PreparationContext
            Exact admitted plan, actual compatibility and trusted job root.

        Returns
        -------
        PreparedExecutor
            Coordinate capabilities used by the single common lifecycle driver.
        """
        ...
