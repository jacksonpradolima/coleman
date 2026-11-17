"""One real local-state boundary, with no task queue or coordinate scheduler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import fingerprint
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
    CommitIntent,
    CommitReceipt,
    CommitRequest,
    ComponentSpec,
)
from coleman.tasks.contracts import OwnedCommitProof, TaskFrontier
from coleman.tasks.protocols import CommittedProgress, CoordinateContext

from .experiment import Experiment
from .recipe import STATE, frontier_definition, reference
from .task_contracts import PortfolioCheckpoint, PortfolioConfiguration, PortfolioTrajectory


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
            experiment=self.experiment._capture_state(),  # pylint: disable=protected-access
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
    def initial(self) -> PortfolioCheckpoint:
        """Construct only an immutable envelope, never a real owner."""
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
        return PortfolioCheckpoint(
            schema_version=1,
            binding=selected,
            plan_fingerprint=context.preparation.plan_fingerprint,
            coordinate_fingerprint=context.coordinate_fingerprint,
            source=context.preparation.source,
            rows=self.trajectory.rows,
            experiment=self.trajectory.initial_state,
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
                    binding=self.initial.binding,
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
        initial = self.initial
        return ComponentBinding(
            codec=ComponentCodec(initial.binding, PortfolioCheckpoint),
            new_owner=lambda: PortfolioOwner(initial),
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
        directory = self.context.artifact_directory
        path, marker = directory / "state", directory / "owned-definition.json"
        expected = self.definition.to_json()
        if marker.exists() and read_text(marker, 128 * 1024) != expected:
            raise ValueError("Synthetic Portfolio continuation definition changed")
        if not path.exists():
            if marker.exists() or (directory / "coordinate-result.json").exists():
                raise ValueError("Acknowledged typed Portfolio state is missing; fresh fallback is forbidden")
            if not create:
                return None
            directory.mkdir(parents=True, exist_ok=True)
            store = LocalStateStore.create(path, self.definition, self.context.preparation.compatibility)
        else:
            store = LocalStateStore.open(path)
        if store.read_head() is not None:
            self.checkpoint(store)
        ArtifactWriter(directory).write_immutable(marker, expected)
        return store

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
            current.model_dump(exclude={"experiment"}) != self.initial.model_dump(exclude={"experiment"})
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
        selected = self.binding
        request = CommitRequest(
            schema_version=1,
            expected=None,
            intent=CommitIntent(
                schema_version=1,
                generation=0,
                kind="initialize",
                content_fingerprint=self.initial.fingerprint,
            ),
        )
        store.initialize(
            request,
            (ComponentValue(selected.codec, self.initial),),
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
        selected = self.binding
        prepared = store.load((selected,), compatibility=self.context.preparation.compatibility)
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
            return self._progress_from_checkpoint(store, request, receipt, current)
        return self.progress(store, request, receipt)

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
        current = self.checkpoint(store)
        return self._progress_from_checkpoint(store, request, receipt, current)

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
        """
        start = self.checkpoint(store).experiment.next_index
        stop = min(start + self.configuration.periods_per_commit, len(self.trajectory.periods))
        if start == stop:
            raise ValueError("Synthetic Portfolio cannot advance beyond its terminal original period")
        selected = self.binding
        head = store.read_head()
        assert head is not None
        request = CommitRequest(
            schema_version=1,
            expected=head,
            intent=CommitIntent(
                schema_version=1,
                generation=head.generation + 1,
                kind="transition",
                content_fingerprint=fingerprint(
                    {"from": start, "through": stop, "source": self.trajectory.fingerprint}
                ),
            ),
        )

        def apply(stage: StagedCollection) -> None:
            """Observe only this bounded original batch on its fresh actual owner."""
            owner = stage.owner(selected)
            for position in range(start, stop):
                guard()
                owner.experiment.step(position)

        receipt = store.commit_transition(
            request,
            (selected,),
            apply,
            compatibility=self.context.preparation.compatibility,
        )
        return self.progress(store, request, receipt)
