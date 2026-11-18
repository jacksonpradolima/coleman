"""Explicit coordinate lifetime for one actual portfolio execution operation.

Only complete acquisitions produced by the adjacent owned state adapter are
held. Every reuse still reconciles actual SQL and fresh manifest/blob integrity.
No caller-provided snapshot, validation flag or cross-operation cache exists.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from coleman.core.identity import Reference
from coleman.state.local import LocalStateStore
from coleman.tasks.contracts import OwnedCommitProof
from coleman.tasks.protocols import AttemptContext, CommittedProgress, CoordinateContext, StageBoundary

from ._state import _forbidden, read_text
from ._state_records import _AcknowledgedState, _require_matching_progress
from .recipe import COORDINATE_RESULT, PROGRESS_PROOF, STATE_V2, ModelCodec
from .task_contracts import PortfolioCheckpointV2, PortfolioTrajectory
from .task_results import PortfolioCoordinateResultV2, PortfolioReportV2

if TYPE_CHECKING:
    from .executor import PreparedPortfolio


@dataclass(frozen=True)
class PortfolioCoordinateOperations:
    """Open independent, explicitly bounded operations for one prepared source.

    Parameters
    ----------
    prepared : PreparedPortfolio
        Original immutable preparation; construction opens no store or learner.
    """

    prepared: PreparedPortfolio

    @property
    def profile(self) -> Reference:
        """Return the declared semantic profile without performing I/O.

        Returns
        -------
        Reference
            ``coleman-coordinate-operation@1``, not approval or an actual ACK.
        """
        return Reference(id="coleman-coordinate-operation", version="1")

    @contextmanager
    def open(self, coordinate: PortfolioTrajectory, context: CoordinateContext, /) -> Iterator[ScopedPortfolio]:
        """Enter one independent sequential operation and dispose it on exit.

        Parameters
        ----------
        coordinate : PortfolioTrajectory
            Complete original source coordinate, revalidated before use.
        context : CoordinateContext
            Exact admitted plan, runtime, target and trusted artifact directory.

        Yields
        ------
        ScopedPortfolio
            Prepared-compatible proxy. It exposes no held state or validation
            witness and refuses every operation after closure.

        Raises
        ------
        ValueError, RuntimeError
            Wrong source/context or use outside the creating thread/lifetime.

        Notes
        -----
        The common host owns the finite loop, reservation and guard order. This
        context owns only its private acquisition lifetime, not a scheduler or
        atomic transaction. Accepted ACKs and published files survive failures.
        """
        scope = ScopedPortfolio(self.prepared, coordinate, context)
        try:
            yield scope
        finally:
            scope.close()


class ScopedPortfolio:
    """Keep actual complete acquisitions inside one entered coordinate operation.

    Parameters
    ----------
    prepared : PreparedPortfolio
        Immutable original preparation, shared without live mathematical state.
    coordinate : PortfolioTrajectory
        Complete original coordinate for this operation alone.
    context : CoordinateContext
        Exact original plan/runtime/source and host-owned artifact path.

    Notes
    -----
    Calls are sequential and stay in the creating thread. Separate operations
    own separate acquisitions and stores. Complete learner statistics, pending
    labels and original book receipts remain in durable checkpoints; this scope
    owns no additional continuation statistic. Closing discards its acquired
    views and refuses stale use. A failure does not roll back accepted effects.
    """

    def __init__(
        self, prepared: PreparedPortfolio, coordinate: PortfolioTrajectory, context: CoordinateContext
    ) -> None:
        """Validate bound declarations without creating a learner or opening I/O."""
        self._prepared = prepared
        self._context = context
        self._layout = prepared._layout(coordinate, context)  # pylint: disable=protected-access
        self._store: LocalStateStore | None = None
        self._held: _AcknowledgedState | None = None
        self._thread = threading.get_ident()
        self._active = True

    def _require_live(self) -> None:
        """Refuse disposed and cross-thread calls before any storage effect."""
        if not self._active:
            raise RuntimeError("Portfolio coordinate operation is closed")
        if threading.get_ident() != self._thread:
            raise RuntimeError("Portfolio coordinate operation requires its creating thread")

    def _require_coordinate(self, coordinate: PortfolioTrajectory, context: CoordinateContext) -> None:
        """Preserve complete public coordinate admission and the exact bound path."""
        self._require_live()
        if context != self._context:
            raise ValueError("Portfolio coordinate operation changed its complete bound context")
        self._prepared._layout(coordinate, context)  # pylint: disable=protected-access

    def _require_context(self, context: CoordinateContext) -> None:
        """Keep proof operations under their exact original complete context."""
        self._require_coordinate(self._layout.trajectory, context)

    def close(self) -> None:
        """Dispose acquired views idempotently, without undoing durable effects.

        Notes
        -----
        Performs no store command or physical-effect compensation. All later
        proxy/verifier/property use refuses, including a previously saved proxy.
        """
        self._active = False
        self._held = None
        self._store = None

    @property
    def boundary(self) -> StageBoundary[PortfolioCoordinateResultV2, PortfolioReportV2]:
        """Return the original pure output boundary while this operation is open.

        Returns
        -------
        StageBoundary
            Original source/order checks; no state is acquired here.
        """
        self._require_live()
        return self._prepared.boundary

    @property
    def progress_verifier(self) -> ScopedPortfolio:
        """Return this operation's real retained-proof verifier.

        Returns
        -------
        ScopedPortfolio
            Same live operation; the host still applies its declared proof codec.
        """
        self._require_live()
        return self

    @property
    def publication_readback(self) -> ScopedPortfolio:
        """Return actual publication/readback inside this same live operation.

        Returns
        -------
        ScopedPortfolio
            Handler under the separate publication/readback version-one profile.
        """
        self._require_live()
        return self

    @property
    def profile(self) -> Reference:
        """Return publication/readback semantics, never a state certificate.

        Returns
        -------
        Reference
            ``coleman-coordinate-publication-readback@1`` for this live handler.
        """
        self._require_live()
        return Reference(id="coleman-coordinate-publication-readback", version="1")

    @property
    def state_codec(self) -> ModelCodec[PortfolioCheckpointV2]:
        """Return the unchanged complete normalized checkpoint codec.

        Returns
        -------
        ModelCodec
            Selected version-two codec; no snapshot is captured or certified.
        """
        self._require_live()
        return STATE_V2

    def _retained(self) -> _AcknowledgedState | None:
        """Reconcile real SQL and integrity before using any held actual read."""
        self._require_live()
        if self._store is None or self._held is None:
            self._store, self._held = self._layout._open_reconciliation()  # pylint: disable=protected-access
            return self._held
        store, held = self._store, self._held
        request = held.progress.proof.request
        if request.intent.kind == "initialize":
            self._held = self._layout._acknowledged_state(store)  # pylint: disable=protected-access
            return self._held
        before = store.read_head()
        receipt = store.commit_transition(
            request,
            (self._layout.binding,),
            _forbidden,
            compatibility=self._context.preparation.compatibility,
        )
        if receipt.status != "retained":
            raise ValueError("Portfolio operation requires real retained original-command SQL")
        if (
            before == held.progress.proof.observed_head
            and receipt.accepted_head == receipt.current_head == before
            and held.manifest.fingerprint == receipt.current_head.manifest_fingerprint
        ):
            progress = self._layout._progress_from_checkpoint(  # pylint: disable=protected-access
                store, request, receipt, held.checkpoint
            )
            self._held = _AcknowledgedState(held.checkpoint, held.manifest, progress)
        else:
            self._held = self._layout._acknowledged_state(store)  # pylint: disable=protected-access
        return self._held

    def reconcile(self, context: CoordinateContext, /) -> CommittedProgress[OwnedCommitProof] | None:
        """Recover actual complete ACK-backed progress inside this operation.

        Parameters
        ----------
        context : CoordinateContext
            Exact original bound context, not a snapshot or validation flag.

        Returns
        -------
        CommittedProgress or None
            Real retained progress; genuinely absent/virgin state remains null.
        """
        self._require_context(context)
        acquired = self._retained()
        return None if acquired is None else acquired.progress

    def validate(self, progress: CommittedProgress[OwnedCommitProof], context: CoordinateContext, /) -> None:
        """Validate full proof and actual retained integrity before outer progress.

        Parameters
        ----------
        progress : CommittedProgress
            Incoming actual response, still admitted through the host proof codec.
        context : CoordinateContext
            Exact original plan, runtime, source and artifact path.

        Raises
        ------
        ValueError, RuntimeError, OSError
            Wrong/stale proof, unavailable/corrupt state or a disposed operation.
        """
        self._require_context(context)
        candidate = PROGRESS_PROOF.validate(progress.proof)
        acquired = self._retained()
        if acquired is None:
            raise ValueError("Portfolio progress requires actual acknowledged state")
        _require_matching_progress(progress, candidate, acquired.progress)

    def _complete(self) -> PortfolioCoordinateResultV2:
        """Project only actual retained current state acquired inside this scope."""
        acquired = self._retained()
        if acquired is None or self._store is None:
            raise ValueError("Portfolio completion requires its actual complete acknowledged state")
        return self._layout._completed_from_acknowledged(self._store, acquired)[0]  # pylint: disable=protected-access

    def read_completed(
        self, coordinate: PortfolioTrajectory, context: CoordinateContext, /
    ) -> PortfolioCoordinateResultV2 | None:
        """Read actual fixed result bytes and join them to complete current state.

        Parameters
        ----------
        coordinate : PortfolioTrajectory
            Original complete coordinate, fully admitted before file access.
        context : CoordinateContext
            Exact bound context and trusted owned artifact directory.

        Returns
        -------
        PortfolioCoordinateResultV2 or None
            Actual bounded canonical file result or unpublished absence. Presence
            alone never certifies completion; real current state is checked.
        """
        self._require_coordinate(coordinate, context)
        path = context.artifact_directory / "coordinate-result.json"
        if not path.exists():
            self._retained()
            return None
        result = COORDINATE_RESULT.decode(read_text(path, COORDINATE_RESULT.maximum_bytes))
        if result.to_json() != self._complete().to_json():
            raise ValueError("Portfolio result differs from its complete actual terminal state artifacts")
        return result

    def advance(
        self,
        coordinate: PortfolioTrajectory,
        context: AttemptContext,
        committed: Callable[[CommittedProgress[OwnedCommitProof]], None],
        /,
    ) -> PortfolioCoordinateResultV2:
        """Run original bounded batches with actual host callbacks and guards.

        Parameters
        ----------
        coordinate : PortfolioTrajectory
            Every original period and complete declared source coordinate.
        context : AttemptContext
            Actual reserved invocation, deadline and unchanged shared guard.
        committed : callable
            Host progress receiver, invoked once after each actual ACK. Its
            scoped verifier still performs full proof and retained/head checks.

        Returns
        -------
        PortfolioCoordinateResultV2
            Original terminal state/manifest references and all fourteen outcomes.

        Notes
        -----
        Learners exist only in actual store staging. Callback errors retain the
        accepted ACK and propagate; no observation or external effect is undone.
        """
        self._require_coordinate(coordinate, context.coordinate)
        context.guard()
        if self._store is None:
            self._store = self._layout.open(create=True)
        assert self._store is not None
        store = self._store
        self._layout.initialize(store)
        current = self._retained()
        if current is None:
            raise ValueError("Portfolio operation lost its actual initialized state")
        while current.progress.frontier.position < len(coordinate.periods):
            previous = current.progress.frontier.position
            self._held = self._layout._advance_acquisition(store, context.guard)  # pylint: disable=protected-access
            if self._held.progress.frontier.position <= previous:
                raise ValueError("Portfolio batch must advance its actual acknowledged period frontier")
            committed(self._held.progress)
            self._require_live()
            current = self._held
            assert current is not None
        return self._complete()

    def publish_and_read(
        self,
        coordinate: PortfolioTrajectory,
        value: PortfolioCoordinateResultV2,
        context: CoordinateContext,
        /,
    ) -> PortfolioCoordinateResultV2:
        """Write/read actual immutable bytes using this operation's full audit.

        Parameters
        ----------
        coordinate : PortfolioTrajectory
            Complete original bound coordinate, not an executable selector.
        value : PortfolioCoordinateResultV2
            Typed terminal reference and exact summaries; never a state witness.
        context : CoordinateContext
            Exact original plan/runtime/source and trusted artifact path.

        Returns
        -------
        PortfolioCoordinateResultV2
            Actual bounded file readback joined to fresh ACK/manifest/blob
            integrity, with full wire equality and no later-call cache.

        Notes
        -----
        The original public standalone readers remain fresh. A later refusal
        can leave immutable bytes present; outer completion belongs to the host.
        """
        self._require_coordinate(coordinate, context)
        selected = COORDINATE_RESULT.validate(value)
        if selected.to_json() != self._complete().to_json():
            raise ValueError("Portfolio result differs from its complete actual terminal state artifacts")
        assert self._store is not None
        return self._layout._publish_verified_bytes(self._store, selected)  # pylint: disable=protected-access

    def publish_coordinate(
        self,
        coordinate: PortfolioTrajectory,
        value: PortfolioCoordinateResultV2,
        context: CoordinateContext,
        /,
    ) -> None:
        """Publish with the same actual readback and fresh-integrity requirements.

        Parameters
        ----------
        coordinate, value, context : PortfolioTrajectory, PortfolioCoordinateResultV2, CoordinateContext
            Complete bound coordinate, actual terminal result and trusted context.
        """
        self.publish_and_read(coordinate, value, context)
