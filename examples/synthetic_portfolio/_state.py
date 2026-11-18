"""One real local-state boundary, with no task queue or coordinate scheduler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field

from coleman.artifacts import ArtifactWriter
from coleman.core.base import CoreImmutable
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.components import (
    ComponentBinding,
    ComponentCodec,
    ComponentValue,
    PreparedCollection,
    StagedCollection,
)
from coleman.state.contracts import StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import (
    CollectionDefinition,
    CollectionManifest,
    CommitIntent,
    CommitReceipt,
    CommitRequest,
    ComponentSpec,
)
from coleman.tasks.contracts import OwnedCommitProof, TaskFrontier
from coleman.tasks.protocols import CommittedProgress, CoordinateContext

from .experiment import Experiment, ExperimentStateV2
from .recipe import STATE_V2 as STATE
from .recipe import frontier_definition, reference
from .task_contracts import PortfolioCheckpointV2 as PortfolioCheckpoint
from .task_contracts import PortfolioConfiguration, PortfolioTrajectory
from .task_results import PortfolioCoordinateResultV2, PortfolioSnapshotReference, terminal_outcomes


def read_text(path: Path, maximum_bytes: int) -> str:
    """Acquire a bounded artifact before parsing a complete concrete value.

    Parameters
    ----------
    path : pathlib.Path
        Trusted host-selected local file, never supplied in request content.
    maximum_bytes : int
        Strict integer from one through16MiB, excluding no hidden suffix.

    Returns
    -------
    str
        Complete original UTF-8 text, without normalization.

    Raises
    ------
    ValueError, OSError, UnicodeError
        Oversized, unavailable or invalid UTF-8 artifact.
    """
    if type(maximum_bytes) not in (int,) or not 1 <= maximum_bytes <= 16 * 1024**2:
        raise ValueError("Synthetic Portfolio acquisition requires a strict positive capacity at most16MiB")
    with path.open("rb") as stream:
        content = stream.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ValueError("Synthetic Portfolio artifact exceeds its selected acquisition capacity")
    return content.decode("utf-8")


class PortfolioOwner:
    """Stage one real learner; construction occurs only inside accepted staging.

    Parameters
    ----------
    initial : PortfolioCheckpoint
        Exact complete immutable starting envelope and selected binding.

    Notes
    -----
    The actual composed experiment owner is serial and has immutable typed state.
    This wrapper adds no RNG, cache, hidden mutable vector or external effect.
    """

    def __init__(self, initial: PortfolioCheckpoint) -> None:
        """Create the actual configured learner inside its trusted factory."""
        self._initial = initial
        self.experiment = Experiment(strength=initial.experiment.strength, rows=initial.rows)

    def export_state(self) -> PortfolioCheckpoint:
        """Export complete real math with the exact fixed surrounding bindings.

        Returns
        -------
        PortfolioCheckpoint
            Full current experiment state, without runtime fields inside its math.
        """
        return PortfolioCheckpoint(
            **self._initial.model_dump(exclude={"experiment"}),
            # The normal outer constructor traverses every ExperimentState
            # field, including unchecked nested input. Retain complete audited
            # mathematical exports, without validating the full archive twice.
            experiment=self.experiment._capture_normalized_state(),  # pylint: disable=protected-access
        )

    def restore_state(self, snapshot: PortfolioCheckpoint) -> None:
        """Restore a complete context-matched snapshot before any update.

        Parameters
        ----------
        snapshot : PortfolioCheckpoint
            Exact concrete state with the same plan, source, owner and codec.

        Raises
        ------
        ValueError
            Any surrounding or inner mathematical binding differs. Existing
            live composition state remains unchanged on refusal.
        """
        current = STATE.validate(snapshot)
        if current.model_dump(exclude={"experiment"}) != self._initial.model_dump(exclude={"experiment"}):
            raise ValueError("Portfolio restore requires its exact complete execution bindings")
        # STATE.validate has just traversed the complete ExperimentState under
        # revalidate_instances=always. Keep the same strict staged restores,
        # without immediately repeating that identical inner model audit.
        self.experiment._restore_checked(current.experiment)  # pylint: disable=protected-access


def _forbidden(_stage: StagedCollection) -> None:
    """Refuse any body execution during an exact retained-command lookup."""
    raise AssertionError("Retained task reconciliation must execute zero transition callbacks")


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

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    binding: StateBinding
    plan_fingerprint: Fingerprint
    coordinate_fingerprint: Fingerprint
    source: DefinitionBinding
    rows: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Layout:
    """Keep pure bindings and original data separately from actual live owners.

    Parameters
    ----------
    configuration : PortfolioConfiguration
        Exact fixed mathematical law and publication batching rule.
    trajectory : PortfolioTrajectory
        Exact complete original coordinate and virgin inner state.
    context : CoordinateContext
        Actual admitted execution bindings and trusted local artifact directory.
    """

    configuration: PortfolioConfiguration
    trajectory: PortfolioTrajectory
    context: CoordinateContext

    @property
    def _identity(self) -> _CheckpointIdentity:
        """Validate fixed metadata without converting an unused virgin state.

        All configuration and trajectory fingerprints still use their complete
        public validators. This fresh value is never cached or used in place of
        a selected state reader, owner construction or mathematical restore.
        """
        context = self.context
        selected = StateBinding(
            owner=self.trajectory.reference,
            implementation=context.preparation.implementation,
            state_codec=STATE.declaration.binding.reference,
            configuration_fingerprint=fingerprint(
                {
                    "configuration": self.configuration.fingerprint,
                    "trajectory": self.trajectory.fingerprint,
                    "source": context.preparation.source.model_dump(mode="json"),
                    "plan": context.preparation.plan_fingerprint,
                    "coordinate": context.coordinate_fingerprint,
                }
            ),
        )
        return _CheckpointIdentity(
            schema_version=2,
            binding=selected,
            plan_fingerprint=context.preparation.plan_fingerprint,
            coordinate_fingerprint=context.coordinate_fingerprint,
            source=context.preparation.source,
            rows=self.trajectory.rows,
        )

    @property
    def initial(self) -> PortfolioCheckpoint:
        """Construct a complete validated virgin snapshot, never a real owner.

        The explicit v1-to-v2 conversion remains lossless and fully validated.
        Metadata-only operations do not request this unused state projection.
        """
        return PortfolioCheckpoint(
            **self._identity.model_dump(mode="python"),
            experiment=ExperimentStateV2.from_legacy(self.trajectory.initial_state),
        )

    @property
    def definition(self) -> CollectionDefinition:
        """Declare the sole complete actual composed experiment owner."""
        return CollectionDefinition(
            schema_version=1,
            reference=reference("synthetic-portfolio-owned-collection"),
            session=reference("session-" + self.context.coordinate_fingerprint[7:]),
            source_epoch=self.context.preparation.source.reference,
            children=(),
            components=(
                ComponentSpec(
                    binding=self._identity.binding,
                    child=None,
                    roles=(
                        "algorithm",
                        "predictor",
                        "transform",
                        "learner_pending",
                        "domain",
                        "applied_path",
                        "virtual_path",
                        "delivery",
                    ),
                    frontier_profile="none",
                ),
            ),
        )

    @property
    def binding(self) -> ComponentBinding[PortfolioCheckpoint, PortfolioOwner]:
        """Supply a fresh actual owner factory without invoking it."""
        return ComponentBinding(
            codec=ComponentCodec(self._identity.binding, PortfolioCheckpoint),
            new_owner=lambda: PortfolioOwner(self.initial),
            frontier=lambda _: None,
        )

    def open(self, *, create: bool = False) -> LocalStateStore | None:
        """Open exact existing state; only explicitly virgin absence can create.

        Parameters
        ----------
        create : bool, optional
            Permit explicit first creation. A retained marker without its store
            always refuses, including when creation was requested.

        Returns
        -------
        LocalStateStore or None
            Verified store, or genuinely absent state in this trusted profile.

        Raises
        ------
        ValueError, OSError
            Missing prior state, changed marker, corrupt/incompatible store.
        """
        return self._open_acquisition(create=create).store

    def open_with_checkpoint(
        self, *, create: bool = False
    ) -> tuple[LocalStateStore | None, PortfolioCheckpoint | None]:
        """Return an opened store and its same-operation complete checked state.

        Parameters
        ----------
        create : bool, optional
            Permit an explicitly virgin store, following ``open``'s rules.

        Returns
        -------
        tuple
            Actual store and complete snapshot from one selected load. Absence
            returns ``(None, None)``; a virgin store has no snapshot yet.

        Raises
        ------
        ValueError, OSError
            Invalid source/bindings/artifacts or a head changed during acquisition.

        Notes
        -----
        The fresh head must match this actual manifest. This is a point-in-time
        result, not a lease, caller validation token or cache for another call.
        Standalone ``open`` and ``checkpoint`` retain their independent reads.
        """
        opened = self._open_acquisition(create=create)
        if opened.store is not None:
            head = opened.store.read_head()
            expected = None if opened.manifest is None else opened.manifest.fingerprint
            observed = None if head is None else head.manifest_fingerprint
            if observed != expected:
                raise ValueError("Portfolio open snapshot requires its actual current matching head")
        return opened.store, opened.checkpoint

    def open_reconciled(
        self,
    ) -> tuple[LocalStateStore | None, CommittedProgress[OwnedCommitProof] | None, PortfolioCheckpoint | None]:
        """Open and recover real retained progress from one complete acquisition.

        Returns
        -------
        tuple
            Actual store, retained progress and complete selected checkpoint.
            Absence returns three nulls; a virgin store has null progress/state.

        Raises
        ------
        ValueError, OSError
            Changed source/definition/state, incompatible image, expired receipt
            or concurrently changed head. No substitute proof is produced.

        Notes
        -----
        Follows the original open/marker rules, including immutable marker I/O.
        The real retained SQL operation and final fresh-head checks still run.
        A newer head reloads/refuses; no owner or transition body runs. Reuse is
        confined to this call and never accepted from a caller or later call.
        """
        store, acquired = self._open_reconciliation()
        if acquired is None:
            return store, None, None
        return store, acquired.progress, acquired.checkpoint

    def _open_reconciliation(self) -> tuple[LocalStateStore | None, _AcknowledgedState | None]:
        """Reuse only this operation's actual full selected open acquisition."""
        opened = self._open_acquisition(create=False)
        if opened.store is None or opened.prepared is None:
            if opened.store is not None and opened.store.read_head() is not None:
                raise ValueError("Portfolio open snapshot requires its actual current matching head")
            return opened.store, None
        return opened.store, self._reconcile_selected(opened.store, self.binding, opened.prepared)

    def open_resolved(self, value: PortfolioCoordinateResultV2) -> tuple[LocalStateStore, PortfolioCheckpoint]:
        """Open and resolve exact terminal artifacts within one full acquisition.

        Parameters
        ----------
        value : PortfolioCoordinateResultV2
            Complete declared source, byte references, outcomes and actual proof.
            No path or previously validated snapshot is accepted.

        Returns
        -------
        tuple
            Actual opened store and its complete current selected checkpoint.

        Raises
        ------
        ValueError, OSError
            Missing/incomplete/corrupt state, changed source/metadata/summary,
            incompatible image, expired receipt or stale/advanced current head.

        Notes
        -----
        Performs actual retained SQL and all terminal manifest, digest, source,
        summary and fresh-head checks. Reading grants no execution or migration
        authority and repeats no mathematical work. Standalone readers remain
        fresh. No state survives this operation as a validation cache.
        """
        selected = PortfolioCoordinateResultV2.model_validate(value)
        store, acquired = self._open_reconciliation()
        if store is None or acquired is None:
            raise ValueError("Portfolio result requires the complete actual current state store")
        actual, checkpoint = self._completed_from_acknowledged(store, acquired)
        if selected.to_json() != actual.to_json():
            raise ValueError("Portfolio result differs from its complete actual terminal state artifacts")
        return store, checkpoint

    def _open_acquisition(self, *, create: bool) -> _OpenedState:
        """Perform the original open and full selected read once, without caching."""
        directory = self.context.artifact_directory
        path, marker = directory / "state", directory / "owned-definition.json"
        expected = self.definition.to_json()
        if marker.exists() and read_text(marker, 128 * 1024) != expected:
            raise ValueError("Synthetic Portfolio continuation definition changed")
        if not path.exists():
            if marker.exists() or (directory / "coordinate-result.json").exists():
                raise ValueError("Acknowledged typed Portfolio state is missing; fresh fallback is forbidden")
            if not create:
                return _OpenedState(None, None, None)
            directory.mkdir(parents=True, exist_ok=True)
            store = LocalStateStore.create(path, self.definition, self.context.preparation.compatibility)
        else:
            store = LocalStateStore.open(path)
        checkpoint = manifest = prepared = None
        if store.read_head() is not None:
            prepared = store.load((self.binding,), compatibility=self.context.preparation.compatibility)
            checkpoint = self._checkpoint_from_prepared(prepared)
            manifest = prepared.manifest
        ArtifactWriter(directory).write_immutable(marker, expected)
        return _OpenedState(store, checkpoint, manifest, prepared)

    def checkpoint(self, store: LocalStateStore) -> PortfolioCheckpoint:
        """Read every acknowledged field without creating/restoring a learner.

        Parameters
        ----------
        store : LocalStateStore
            Actual opened local collection.

        Returns
        -------
        PortfolioCheckpoint
            Complete current state under exact selected bindings.
        """
        binding = self.binding
        prepared = store.load((binding,), compatibility=self.context.preparation.compatibility)
        return self._checkpoint_from_prepared(prepared)

    def _checkpoint_from_prepared(self, prepared: PreparedCollection) -> PortfolioCheckpoint:
        """Check every host binding on a complete, same-operation store load.

        Parameters
        ----------
        prepared : PreparedCollection
            Complete selected-codec result of the actual local store's load;
            never metadata alone or an unchecked application assembly.

        Returns
        -------
        PortfolioCheckpoint
            Complete typed experiment under this exact coordinate and source.

        Raises
        ------
        ValueError
            Changed collection, source, mathematical law or work frontier.

        Notes
        -----
        The caller must obtain this value through the full store reader in the
        current operation. Nothing is retained between calls or accepted as an
        acknowledgement here; progress still requires the actual SQL receipt
        and a fresh matching head observation.
        """
        if prepared.manifest.definition.to_json() != self.definition.to_json():
            raise ValueError("Synthetic Portfolio store has a different complete collection definition")
        current = prepared.component(self.binding.codec)
        if (
            current.model_dump(exclude={"experiment"}) != self._identity.model_dump(mode="python")
            or current.experiment.strength != self.configuration.strength
            or current.experiment.tape_fingerprint != self.trajectory.initial_state.tape_fingerprint
        ):
            raise ValueError("Synthetic Portfolio checkpoint changed its complete source, math or execution bindings")
        if current.experiment.next_index > len(self.trajectory.periods):
            raise ValueError("Synthetic Portfolio state exceeds its declared original trajectory")
        return current

    def initialize(self, store: LocalStateStore) -> None:
        """Explicitly accept the exact supplied virgin snapshot if not initialized.

        Parameters
        ----------
        store : LocalStateStore
            Actual verified virgin or already acknowledged store.
        """
        if store.read_head() is not None:
            return
        initial = self.initial
        selected = self.binding
        request = CommitRequest(
            schema_version=1,
            expected=None,
            intent=CommitIntent(
                schema_version=1,
                generation=0,
                kind="initialize",
                content_fingerprint=initial.fingerprint,
            ),
        )
        store.initialize(
            request,
            (ComponentValue(selected.codec, initial),),
            (selected,),
            compatibility=self.context.preparation.compatibility,
        )

    def reconcile(self, store: LocalStateStore) -> CommittedProgress[OwnedCommitProof]:
        """Obtain a real retained receipt for the actual current accepted command.

        Parameters
        ----------
        store : LocalStateStore
            Actual initialized local store, reopened or just acknowledged.

        Returns
        -------
        CommittedProgress
            Exact current proof and logical position from complete decoded period frontier.

        Notes
        -----
        Reissuing the original intent hits the store's retained branch before
        any owner factory/restore/body. No receipt is fabricated from metadata.
        An unchanged accepted head can reuse this operation's complete selected
        load. A concurrently advanced current head requires another full read;
        both paths finish with the same fresh head/receipt comparison.
        """
        return self._acknowledged_state(store).progress

    def reconcile_with_checkpoint(
        self, store: LocalStateStore
    ) -> tuple[CommittedProgress[OwnedCommitProof], PortfolioCheckpoint]:
        """Return an actual retained proof and the same complete checked snapshot.

        Parameters
        ----------
        store : LocalStateStore
            Actual initialized store under this exact source and execution law.

        Returns
        -------
        tuple
            Retained progress and complete state from the same acquisition.

        Raises
        ------
        ValueError, OSError
            Changed source, data, manifest or head; no substitute proof is made.

        Notes
        -----
        Uses the real retained command and final fresh-head comparison. A newer
        current head follows the existing complete reload/refusal path. No owner
        callback runs, and no state is reused by a later operation.
        """
        acquired = self._acknowledged_state(store)
        return acquired.progress, acquired.checkpoint

    def _acknowledged_state(self, store: LocalStateStore) -> _AcknowledgedState:
        """Retain a complete selected load only within this one current call."""
        selected = self.binding
        prepared = store.load((selected,), compatibility=self.context.preparation.compatibility)
        return self._reconcile_selected(store, selected, prepared)

    def _reconcile_selected(
        self,
        store: LocalStateStore,
        selected: ComponentBinding[PortfolioCheckpoint, PortfolioOwner],
        prepared: PreparedCollection,
    ) -> _AcknowledgedState:
        """Join one actual selected load to the unchanged real retained path."""
        current = self._checkpoint_from_prepared(prepared)
        request = CommitRequest(
            schema_version=1,
            expected=prepared.manifest.predecessor,
            intent=prepared.manifest.intent,
        )
        if request.intent.kind == "initialize":
            receipt = store.initialize(
                request,
                prepared.values,
                (selected,),
                compatibility=self.context.preparation.compatibility,
            )
        else:
            receipt = store.commit_transition(
                request,
                (selected,),
                _forbidden,
                compatibility=self.context.preparation.compatibility,
            )
        if (
            receipt.accepted_head == receipt.current_head
            and receipt.accepted_head.manifest_fingerprint == prepared.manifest.fingerprint
        ):
            progress = self._progress_from_checkpoint(store, request, receipt, current)
            return _AcknowledgedState(current, prepared.manifest, progress)
        return self._progress_evidence(store, request, receipt)

    def progress(
        self,
        store: LocalStateStore,
        request: CommitRequest,
        receipt: CommitReceipt,
    ) -> CommittedProgress[OwnedCommitProof]:
        """Pair a real acceptance response with the current complete experiment.

        Parameters
        ----------
        store : LocalStateStore
            Actual current local store under the driver's coordinate lock.
        request, receipt : CommitRequest, CommitReceipt
            Exact original command and actual store response, never synthesized.

        Returns
        -------
        CommittedProgress
            Exact ACK joins with state-derived U under the static work profile.
        """
        return self._progress_evidence(store, request, receipt).progress

    def _progress_evidence(
        self, store: LocalStateStore, request: CommitRequest, receipt: CommitReceipt
    ) -> _AcknowledgedState:
        """Obtain complete current bytes before joining their actual ACK."""
        prepared = store.load((self.binding,), compatibility=self.context.preparation.compatibility)
        current = self._checkpoint_from_prepared(prepared)
        if prepared.manifest.fingerprint != receipt.current_head.manifest_fingerprint:
            raise ValueError("Synthetic Portfolio progress requires the actual current matching ACK")
        progress = self._progress_from_checkpoint(store, request, receipt, current)
        return _AcknowledgedState(current, prepared.manifest, progress)

    def _progress_from_checkpoint(
        self,
        store: LocalStateStore,
        request: CommitRequest,
        receipt: CommitReceipt,
        current: PortfolioCheckpoint,
    ) -> CommittedProgress[OwnedCommitProof]:
        """Join an actual receipt and complete checked state to a fresh head.

        Parameters
        ----------
        store : LocalStateStore
            Actual same-operation store; its head is read again here.
        request, receipt : CommitRequest, CommitReceipt
            Original accepted command and actual SQL acknowledgement.
        current : PortfolioCheckpoint
            Complete same-operation selected load, checked by this layout.

        Returns
        -------
        CommittedProgress
            Unchanged full proof and state-derived chronological frontier.

        Raises
        ------
        ValueError
            Another command advanced the head after receipt acquisition.
        """
        head = store.read_head()
        if head is None or head != receipt.current_head:
            raise ValueError("Synthetic Portfolio progress requires the actual current matching ACK")
        frontier = TaskFrontier(
            profile=frontier_definition(),
            coordinate=self.trajectory.reference,
            position=current.experiment.next_index,
        )
        proof = OwnedCommitProof(
            contract="coleman-tasks-v1",
            schema_version=1,
            artifact_type="owned_commit_proof",
            plan_fingerprint=current.plan_fingerprint,
            coordinate_fingerprint=current.coordinate_fingerprint,
            frontier=frontier,
            collection=DefinitionBinding(
                reference=self.definition.reference,
                definition_fingerprint=self.definition.fingerprint,
            ),
            compatibility=self.context.preparation.compatibility,
            request=request,
            receipt=receipt,
            observed_head=head,
        )
        return CommittedProgress(current.plan_fingerprint, current.coordinate_fingerprint, frontier, proof)

    def advance(self, store: LocalStateStore, guard: Callable[[], None]) -> CommittedProgress[OwnedCommitProof]:
        """Accept one bounded batch of actual original periods on fresh staging.

        Parameters
        ----------
        store : LocalStateStore
            Actual initialized state at a decoded original-period position.
        guard : callable
            Shared actual deadline/ownership/resource check, before each period.

        Returns
        -------
        CommittedProgress
            Real ACK followed by a complete state-derived logical frontier.

        Raises
        ------
        ValueError
            Missing or foreign collection, stale predecessor, terminal state or
            incompatible staged data. A refused transition accepts no new ACK.

        Notes
        -----
        Command ``portfolio-staged-batch-v2`` binds the complete expected ACK,
        original source and mathematical configuration. The ACK's verified
        definition digest preserves the full collection comparison; it never
        supplies a period count. Only the strictly restored staged experiment
        supplies that count. Old commands remain retained under their original
        intent and are never rewritten into this command generation.
        """
        return self._advance_acquisition(store, guard).progress

    def advance_with_checkpoint(
        self, store: LocalStateStore, guard: Callable[[], None]
    ) -> tuple[CommittedProgress[OwnedCommitProof], CommittedProgress[OwnedCommitProof], PortfolioCheckpoint]:
        """Advance, reconcile the same command and return its actual checked state.

        Parameters
        ----------
        store : LocalStateStore
            Actual initialized store for this exact original source and plan.
        guard : callable
            Existing deadline/ownership/resource guard before every period.

        Returns
        -------
        tuple
            Actual first acceptance progress, real retained-command progress,
            and complete snapshot joined to that retained progress.

        Raises
        ------
        ValueError, OSError
            Failed transition, retained/source/head mismatch or concurrent change.

        Notes
        -----
        Only the actual complete post-ACK selected load is reused, inside this
        call. The same original request is reissued with a forbidden body and
        must return a real retained receipt. Matching before/after fresh heads,
        manifest and receipts permit reuse; advancement takes the original full
        reload/refusal path. No callback snapshot, caller token or cross-call
        cache supplies authority. Standalone operations remain fresh.

        The transition and retained lookup are separate store operations, not
        an atomic transaction. The first transition can remain committed when
        a later lookup, read or head check fails. After an uncertain response,
        reconcile the original command retained in the acknowledged manifest
        and receipt; do not assume rollback or replay the transition body.
        """
        acquired = self._advance_acquisition(store, guard)
        accepted = acquired.progress
        request = accepted.proof.request
        before = store.read_head()
        receipt = store.commit_transition(
            request, (self.binding,), _forbidden, compatibility=self.context.preparation.compatibility
        )
        if receipt.status != "retained":
            raise ValueError("Portfolio reconciliation requires the real retained original command")
        if (
            before == accepted.proof.observed_head
            and receipt.accepted_head == receipt.current_head
            and receipt.current_head == before
            and acquired.manifest.fingerprint == receipt.current_head.manifest_fingerprint
        ):
            retained = self._progress_from_checkpoint(store, request, receipt, acquired.checkpoint)
            return accepted, retained, acquired.checkpoint
        current = self._progress_evidence(store, request, receipt)
        return accepted, current.progress, current.checkpoint

    def _advance_acquisition(self, store: LocalStateStore, guard: Callable[[], None]) -> _AcknowledgedState:
        """Keep the actual post-ACK selected acquisition inside this operation."""
        selected = self.binding
        head = store.read_head()
        if head is None or head.definition_fingerprint != self.definition.fingerprint:
            raise ValueError("Synthetic Portfolio advance requires its complete acknowledged collection definition")
        request = CommitRequest(
            schema_version=1,
            expected=head,
            intent=CommitIntent(
                schema_version=1,
                generation=head.generation + 1,
                kind="transition",
                content_fingerprint=fingerprint(
                    {
                        "command": "portfolio-staged-batch-v2",
                        "expected": head.model_dump(mode="json"),
                        "trajectory": self.trajectory.fingerprint,
                        "source": self.context.preparation.source.model_dump(mode="json"),
                        "configuration": self.configuration.fingerprint,
                        "maximum_periods": self.configuration.periods_per_commit,
                    }
                ),
            ),
        )

        def apply(stage: StagedCollection) -> None:
            """Observe only this bounded original batch on its fresh actual owner."""
            owner = stage.owner(selected)
            start = len(owner.experiment.periods)
            stop = min(start + self.configuration.periods_per_commit, len(self.trajectory.periods))
            if start >= stop:
                raise ValueError("Synthetic Portfolio cannot advance beyond its terminal original period")
            for position in range(start, stop):
                guard()
                owner.experiment.step(position)

        receipt = store.commit_transition(
            request,
            (selected,),
            apply,
            compatibility=self.context.preparation.compatibility,
        )
        return self._progress_evidence(store, request, receipt)

    def _completed_evidence(self, store: LocalStateStore) -> tuple[PortfolioCoordinateResultV2, PortfolioCheckpoint]:
        """Join one terminal retained receipt to complete current selected bytes."""
        acknowledged = self._acknowledged_state(store)
        return self._completed_from_acknowledged(store, acknowledged)

    def _completed_from_acknowledged(
        self, store: LocalStateStore, acknowledged: _AcknowledgedState
    ) -> tuple[PortfolioCoordinateResultV2, PortfolioCheckpoint]:
        """Retain all original terminal checks on same-operation full evidence."""
        current, progress, manifest = acknowledged.checkpoint, acknowledged.progress, acknowledged.manifest
        head = progress.proof.observed_head
        if (
            current.experiment.next_index != len(self.trajectory.periods)
            or progress.proof.receipt.accepted_head != head
            or manifest.fingerprint != head.manifest_fingerprint
            or store.read_head() != head
        ):
            raise ValueError("Portfolio completion requires the exact terminal current acknowledged full state")
        result = PortfolioCoordinateResultV2(
            schema_version=2,
            source=current.source,
            snapshot=PortfolioSnapshotReference(
                schema_version=1,
                component=manifest.components[0],
                manifest_fingerprint=manifest.fingerprint,
                manifest_byte_count=len(manifest.to_json().encode("utf-8")),
            ),
            experiment_fingerprint=current.experiment.fingerprint,
            proof=progress.proof,
            outcomes=terminal_outcomes(current.experiment),
        )
        return result, current

    def complete_result(self, store: LocalStateStore) -> PortfolioCoordinateResultV2:
        """Describe terminal current ACK artifacts without duplicating full state.

        Parameters
        ----------
        store : LocalStateStore
            Actual owned collection under this exact source and execution image.

        Returns
        -------
        PortfolioCoordinateResultV2
            Exact original state/manifest byte identities, terminal summaries
            and a real retained current receipt. Full receipts remain stored.

        Raises
        ------
        ValueError, OSError
            Incomplete, changed, incompatible or unavailable actual evidence.
        """
        return self._completed_evidence(store)[0]

    def resolve_result(self, store: LocalStateStore, value: PortfolioCoordinateResultV2) -> PortfolioCheckpoint:
        """Resolve exact current evidence through the existing strict state reader.

        Parameters
        ----------
        store : LocalStateStore
            Trusted operator-owned collection; no path is selected by a report.
        value : PortfolioCoordinateResultV2
            Complete declared metadata, source, summaries and terminal receipt.

        Returns
        -------
        PortfolioCheckpoint
            Every original mathematical state, book receipt and pending label.
            Resolution updates no learner and accepts no new state command.

        Raises
        ------
        ValueError, OSError
            Wrong source/identity/size/summary/receipt, tampered or stale artifact,
            incompatible current execution image or missing complete evidence.
        """
        selected = PortfolioCoordinateResultV2.model_validate(value)
        actual, checkpoint = self._completed_evidence(store)
        if selected.to_json() != actual.to_json():
            raise ValueError("Portfolio result differs from its complete actual terminal state artifacts")
        return checkpoint
