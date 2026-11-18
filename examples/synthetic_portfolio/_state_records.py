"""Private complete records for one funded-portfolio storage operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from coleman.core.base import CoreImmutable
from coleman.core.primitives import Fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import PreparedCollection
from coleman.state.contracts import StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionManifest
from coleman.tasks.contracts import OwnedCommitProof
from coleman.tasks.protocols import CommittedProgress

from .task_contracts import PortfolioCheckpointV2 as PortfolioCheckpoint


@dataclass(frozen=True)
class _AcknowledgedState:
    """Keep one operation's fully read checkpoint, manifest and actual receipt."""

    checkpoint: PortfolioCheckpoint
    manifest: CollectionManifest
    progress: CommittedProgress[OwnedCommitProof]


@dataclass(frozen=True)
class _OpenedState:
    """Keep one actual open operation's store and completely selected snapshot."""

    store: LocalStateStore | None
    checkpoint: PortfolioCheckpoint | None
    manifest: CollectionManifest | None
    prepared: PreparedCollection | None = None


class _CheckpointIdentity(CoreImmutable):
    """Validate all fixed checkpoint metadata without constructing a state view.

    This private identity is neither a snapshot nor a validation witness. It
    contains no experiment state and grants no execution or acknowledgement.
    """

    # Repeat the public checkpoint identity without importing its full state view.
    # pylint: disable=duplicate-code
    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    binding: StateBinding
    plan_fingerprint: Fingerprint
    coordinate_fingerprint: Fingerprint
    source: DefinitionBinding
    rows: tuple[tuple[int, int], ...]
    # pylint: enable=duplicate-code


def _require_matching_progress(
    progress: CommittedProgress[OwnedCommitProof],
    candidate: OwnedCommitProof,
    actual: CommittedProgress[OwnedCommitProof],
) -> None:
    """Compare complete admitted proof fields after actual current acquisition."""
    if (progress.plan_fingerprint, progress.coordinate_fingerprint, progress.frontier) != (
        actual.plan_fingerprint,
        actual.coordinate_fingerprint,
        actual.frontier,
    ) or candidate.model_dump(exclude={"receipt": {"status"}}) != actual.proof.model_dump(
        exclude={"receipt": {"status"}}
    ):
        raise ValueError("Portfolio progress differs from the actual current ACK and complete state")
