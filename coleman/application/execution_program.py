"""Typed in-memory programs consumed by the single durable execution driver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint
from coleman.tasks.contracts import TaskFrontier
from coleman.tasks.protocols import PreparationContext, PreparedExecutor, ValueCodec


@dataclass(frozen=True)
class CoordinateWork[W]:
    """Join a concrete coordinate to its separately declared execution frontier.

    Attributes
    ----------
    coordinate_fingerprint : str
        Exact existing native trial ID or generic coordinate content digest.
        A native ID is never replaced by an intermediate adapter's digest.
    reference : Reference
        Stable logical trajectory, independent of attempts and publication.
    target : TaskFrontier
        Declared terminal position under the exact named work profile. A native
        profile explicitly counts native builds; it is not a storage generation.
    value : W
        Already decoded concrete coordinate. It may be the original native
        Trial; no native plan or value is converted into a generic wire record.
    artifact_directory : pathlib.Path
        Trusted driver/adapter-selected local directory under the owned job.
    seed : int or None, optional
        Exact declared uint64 seed, including valid zero, or None when
        undeclared. Forwarded to the coordinate context without RNG operations.

    Notes
    -----
    This is a trusted in-memory program value, not an ingress validator or a
    persisted schema. Compilation validates the selected codecs, IDs and paths
    before the common driver reserves physical work. Construction performs no
    I/O, discovery, checkpoint restoration or mathematical update.
    """

    coordinate_fingerprint: Fingerprint
    reference: Reference
    target: TaskFrontier
    value: W
    artifact_directory: Path
    seed: int | None = None


@dataclass(frozen=True)
class ExecutionProgram[W, R, O, P]:
    """Supply adapter operations to one scheduler, resource guard and work loop.

    Attributes
    ----------
    context : PreparationContext
        Verified plan/source/image bindings and the trusted owned job root.
    coordinates : tuple of CoordinateWork
        Exact ordered stage catalogue; seed repetitions are not inference units.
    executor : PreparedExecutor
        Coordinate operations and the scientific/descriptive boundary. The
        boundary receives the same guarded, at-most-once common work callback.
    output_codec : ValueCodec
        Concrete final-result codec. Legacy serializers retain their original
        omissions/units; they need not inherit a new task result class.
    output_fingerprint : callable
        Concrete content-identity function, evaluated outside SQL. Native
        results retain their original fingerprint algorithm and interpretation.
    output_path : pathlib.Path
        Trusted immutable completion artifact, published before completed status.
    finish_state : {'completed', 'awaiting_evaluation'}, optional
        Normal completion or the existing explicit native tuning handoff. The
        adapter declares this disposition; the driver owns the state transition.

    Notes
    -----
    Programs own no queue, SQLite transaction, retry engine or job work loop.
    Their operations execute outside metadata transaction bodies. Existing
    artifact/checkpoint mechanisms remain responsible for actual bytes; an
    output path or a returned object alone never certifies durable completion.
    """

    context: PreparationContext
    coordinates: tuple[CoordinateWork[W], ...]
    executor: PreparedExecutor[W, R, O, P]
    output_codec: ValueCodec[O]
    output_fingerprint: Callable[[O], Fingerprint]
    output_path: Path
    finish_state: Literal["completed", "awaiting_evaluation"] = "completed"
