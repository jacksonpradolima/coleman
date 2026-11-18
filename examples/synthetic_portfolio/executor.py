"""Structural external executor; the host owns all queue/coordinate scheduling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from hashlib import sha256

from coleman.artifacts import ArtifactWriter
from coleman.objectives.bindings import DefinitionBinding
from coleman.tasks.contracts import OwnedCommitProof
from coleman.tasks.protocols import AttemptContext, CommittedProgress, CoordinateContext, PreparationContext

from ._state import Layout, read_text
from .recipe import (
    CONFIGURATION,
    COORDINATE,
    COORDINATE_RESULT,
    FINAL_RESULT,
    INPUTS,
    PROGRESS_PROOF,
    ModelCodec,
    frontier_definition,
)
from .recipe import (
    STATE_V2 as STATE,
)
from .recipe import recipe as declare_recipe
from .task_contracts import (
    PortfolioCheckpointV2 as PortfolioCheckpoint,
)
from .task_contracts import (
    PortfolioConfiguration,
    PortfolioInputs,
    PortfolioTrajectory,
)
from .task_results import PortfolioCoordinateResultV2, PortfolioReportItemV2, PortfolioReportV2


def _report_advanced(
    previous: int,
    committed: Callable[[CommittedProgress[OwnedCommitProof]], None],
    current: CommittedProgress[OwnedCommitProof],
) -> None:
    """Keep the original frontier refusal before forwarding the actual callback."""
    if current.frontier.position <= previous:
        raise ValueError("Portfolio batch must advance its actual acknowledged period frontier")
    committed(current)


@dataclass(frozen=True)
class PreparedPortfolio:
    """Keep only immutable admitted inputs; no live learner survives a stage.

    Parameters
    ----------
    configuration, inputs : PortfolioConfiguration, PortfolioInputs
        Complete revalidated law and original source declaration.
    context : PreparationContext
        Exact admitted task identity, actual compatibility and trusted job root.
    """

    configuration: PortfolioConfiguration
    inputs: PortfolioInputs
    context: PreparationContext

    @property
    def boundary(self) -> _PreparedBoundary:
        """Return the scope wrapper with the complete immutable inventory.

        Notes
        -----
        Coordinate fingerprints include their task envelope, so the actual
        driver results provide that inventory at publication. A prepared kernel
        cannot derive those envelope hashes from inner trajectory bytes alone.
        """
        return _PreparedBoundary(self)

    @property
    def progress_verifier(self) -> PreparedPortfolio:
        """Expose actual storage recovery, distinct from the pure proof codec."""
        return self

    @property
    def state_codec(self) -> ModelCodec:
        """Return the complete outer state codec without capturing or restoring."""
        return STATE

    def _layout(self, trajectory: PortfolioTrajectory, context: CoordinateContext) -> Layout:
        """Join concrete selected data and all admitted context before I/O."""
        current = COORDINATE.validate(trajectory)
        known = next((item for item in self.inputs.trajectories if item.reference == current.reference), None)
        if known is None or known.to_json() != current.to_json():
            raise ValueError("Portfolio coordinate must match its complete original declared source")
        if (
            context.preparation != self.context
            or context.reference != current.reference
            or context.target.coordinate != current.reference
            or context.target.profile != frontier_definition()
            or context.target.position != len(current.periods)
        ):
            raise ValueError("Portfolio coordinate requires exact admitted context, work profile and target")
        return Layout(self.configuration, current, context)

    def _context_layout(self, context: CoordinateContext) -> Layout:
        """Resolve the source-owned trajectory without looking up code from JSON."""
        coordinate = next((item for item in self.inputs.trajectories if item.reference == context.reference), None)
        if coordinate is None:
            raise ValueError("Portfolio continuation has no matching original trajectory")
        return self._layout(coordinate, context)

    def reconcile(self, context: CoordinateContext, /) -> CommittedProgress[OwnedCommitProof] | None:
        """Reopen actual state and recover an ACK/outer-progress response gap.

        Parameters
        ----------
        context : CoordinateContext
            Exact retained task, coordinate, runtime and trusted artifact path.

        Returns
        -------
        CommittedProgress or None
            Current state-derived accepted periods and actual retained receipt, or virgin state.

        Raises
        ------
        ValueError, OSError
            Changed context, missing acknowledged state, corrupt/incompatible
            artifacts or failed exact original-command receipt reconciliation.
        """
        layout = self._context_layout(context)
        _store, progress, _checkpoint = layout.open_reconciled()
        return progress

    def validate(self, progress: CommittedProgress[OwnedCommitProof], context: CoordinateContext, /) -> None:
        """Check proof against actual current storage before outer metadata.

        Parameters
        ----------
        progress : CommittedProgress
            Real callback/recovery response, not an integer work claim.
        context : CoordinateContext
            Complete expected runtime and source context.

        Raises
        ------
        ValueError
            Wrong proof, stale frontier, missing state or incompatible binding.

        Notes
        -----
        A retained lookup changes receipt status from new to retained; all other
        proof bytes and the current ACK must match exactly. No math is repeated.
        """
        candidate = PROGRESS_PROOF.validate(progress.proof)
        actual = self.reconcile(context)
        if actual is None:
            raise ValueError("Portfolio progress requires actual acknowledged state")
        if (progress.plan_fingerprint, progress.coordinate_fingerprint, progress.frontier) != (
            actual.plan_fingerprint,
            actual.coordinate_fingerprint,
            actual.frontier,
        ) or candidate.model_dump(exclude={"receipt": {"status"}}) != actual.proof.model_dump(
            exclude={"receipt": {"status"}}
        ):
            raise ValueError("Portfolio progress differs from the actual current ACK and complete state")

    def read_completed(
        self, coordinate: PortfolioTrajectory, context: CoordinateContext, /
    ) -> PortfolioCoordinateResultV2 | None:
        """Read only a published complete result and verify its actual current ACK.

        Parameters
        ----------
        coordinate : PortfolioTrajectory
            Exact original declared trajectory.
        context : CoordinateContext
            Actual retained task and trusted result directory.

        Returns
        -------
        PortfolioCoordinateResultV2 or None
            Complete rechecked terminal reference and exact summaries, or
            unpublished absence. Full state remains in the actual ACK artifact.
        """
        layout = self._layout(coordinate, context)
        path = context.artifact_directory / "coordinate-result.json"
        if not path.exists():
            layout.open()  # Check a retained marker even when publication was interrupted.
            return None
        result = COORDINATE_RESULT.decode(read_text(path, COORDINATE_RESULT.maximum_bytes))
        self._require_result(layout, result)
        return result

    def _require_result(self, layout: Layout, result: PortfolioCoordinateResultV2) -> PortfolioCheckpoint:
        """Resolve all current state, summaries and exact original ACK bindings."""
        _store, checkpoint = layout.open_resolved(result)
        return checkpoint

    def resolve_result(self, result: PortfolioCoordinateResultV2, context: CoordinateContext, /) -> PortfolioCheckpoint:
        """Read full original evidence behind a compact result without updating it.

        Parameters
        ----------
        result : PortfolioCoordinateResultV2
            Exact typed result from the selected schema2 codec.
        context : CoordinateContext
            Trusted original plan, current implementation and owned directory.

        Returns
        -------
        PortfolioCheckpoint
            Complete normalized schema2 state, including every original receipt
            and pending label. Content references grant no path/import authority.

        Raises
        ------
        ValueError, OSError
            Wrong context, summaries, identities, sizes, stale ACK or missing/
            corrupt/incompatible artifacts. No mathematical update is attempted.
        """
        return self._require_result(self._context_layout(context), COORDINATE_RESULT.validate(result))

    def advance(
        self,
        coordinate: PortfolioTrajectory,
        context: AttemptContext,
        committed: Callable[[CommittedProgress[OwnedCommitProof]], None],
        /,
    ) -> PortfolioCoordinateResultV2:
        """Finish one original trajectory through bounded actual store transitions.

        Parameters
        ----------
        coordinate : PortfolioTrajectory
            Original finite loss trajectory; this is no job scheduling loop.
        context : AttemptContext
            Actual reserved attempt, deadline and common guard.
        committed : callable
            Called only after real ACKs. A callback failure leaves the accepted
            inner state recoverable, without repeating accepted observations.

        Returns
        -------
        PortfolioCoordinateResultV2
            Exact terminal state reference, typed outcomes and actual current
            receipt. The driver owns outer publication; full state stays stored.
        """
        layout = self._layout(coordinate, context.coordinate)
        context.guard()
        store = layout.open(create=True)
        assert store is not None
        layout.initialize(store)
        progress = layout.reconcile(store)
        result: PortfolioCoordinateResultV2 | None = None
        while progress.frontier.position < len(coordinate.periods):
            previous: int = progress.frontier.position
            progress, result = layout.advance_with_completion(
                store, context.guard, partial(_report_advanced, previous, committed)
            )
        return layout.complete_result(store) if result is None else result

    def publish_coordinate(
        self,
        coordinate: PortfolioTrajectory,
        value: PortfolioCoordinateResultV2,
        context: CoordinateContext,
        /,
    ) -> None:
        """Publish rechecked immutable result bytes, without a job-complete claim.

        Parameters
        ----------
        coordinate : PortfolioTrajectory
            Exact original source coordinate.
        value : PortfolioCoordinateResultV2
            Complete real ACK-backed reference and every terminal outcome. Full
            original evidence is rechecked before publishing this metadata.
        context : CoordinateContext
            Actual trusted publication directory and execution bindings.
        """
        current = COORDINATE_RESULT.validate(value)
        self._require_result(self._layout(coordinate, context), current)
        ArtifactWriter(context.artifact_directory).write_immutable(
            context.artifact_directory / "coordinate-result.json",
            COORDINATE_RESULT.encode(current),
        )


@dataclass(frozen=True)
class _PreparedBoundary:
    """Join original owner order when the common driver supplies envelope hashes."""

    prepared: PreparedPortfolio

    def produce(self, load_completed: Callable[[], tuple[PortfolioCoordinateResultV2, ...]], /) -> PortfolioReportV2:
        """Load once and require every exact original owner/source/plan result."""
        values = load_completed()
        if tuple(item.snapshot.component.binding.owner for item in values) != tuple(
            item.reference for item in self.prepared.inputs.trajectories
        ):
            raise ValueError("Portfolio boundary requires every original source trajectory in declared order")
        if any(
            item.proof.plan_fingerprint != self.prepared.context.plan_fingerprint
            or item.source != self.prepared.inputs.binding
            for item in values
        ):
            raise ValueError("Portfolio boundary requires its exact admitted plan and complete source")
        items = []
        for value in values:
            encoded = COORDINATE_RESULT.encode(value).encode("utf-8")
            items.append(
                PortfolioReportItemV2(
                    schema_version=2,
                    coordinate=value,
                    artifact_profile="portfolio-coordinate-result-v2/core-json-v1",
                    artifact_fingerprint="sha256:" + sha256(encoded).hexdigest(),
                    artifact_byte_count=len(encoded),
                )
            )
        return PortfolioReportV2(
            schema_version=2,
            interpretation="constructed_mechanism_only",
            results=tuple(items),
        )


class PortfolioExecutor:
    """Expose actual concrete codecs and one preparation factory, without I/O.

    Notes
    -----
    Direct shipped Portfolio callers need none of these adapters or registrations.
    The task driver owns attempts, queueing, resource guards and publication.
    """

    @property
    def configuration_codec(self) -> ModelCodec[PortfolioConfiguration]:
        """Return the selected complete mathematical configuration codec."""
        return CONFIGURATION

    @property
    def input_codec(self) -> ModelCodec[PortfolioInputs]:
        """Return the separate complete original-source codec."""
        return INPUTS

    @property
    def coordinate_codec(self) -> ModelCodec[PortfolioTrajectory]:
        """Return the selected original trajectory codec."""
        return COORDINATE

    @property
    def coordinate_result_codec(self) -> ModelCodec[PortfolioCoordinateResultV2]:
        """Return compact schema2 metadata for complete original ACK evidence."""
        return COORDINATE_RESULT

    @property
    def final_result_codec(self) -> ModelCodec[PortfolioReportV2]:
        """Return the complete compact schema2 outcomes-and-artifact codec."""
        return FINAL_RESULT

    @property
    def progress_proof_codec(self) -> ModelCodec[OwnedCommitProof]:
        """Return the pure proof codec, separate from actual reconciliation."""
        return PROGRESS_PROOF

    @property
    def state_codec(self) -> ModelCodec:
        """Return the complete continuation codec declared by this fixture."""
        return STATE

    @property
    def definition(self) -> DefinitionBinding:
        """Return the exact pure static recipe binding, without capture."""
        return declare_recipe().binding

    def prepare(
        self,
        configuration: PortfolioConfiguration,
        inputs: PortfolioInputs,
        context: PreparationContext,
        /,
    ) -> PreparedPortfolio:
        """Join complete values and actual host bindings without creating owners.

        Parameters
        ----------
        configuration, inputs : PortfolioConfiguration, PortfolioInputs
            Actual selected concrete values; full source stays separate from law.
        context : PreparationContext
            Admitted plan/source, captured implementation and trusted job root.

        Returns
        -------
        PreparedPortfolio
            Immutable capabilities; actual learners are created only in accepted
            local-store staging, never during preparation or stale retries.

        Raises
        ------
        ValueError
            Source or any initial full composed configuration differs.
        """
        selected, source = CONFIGURATION.validate(configuration), INPUTS.validate(inputs)
        if source.binding != context.source or any(
            item.initial_state.strength != selected.strength for item in source.trajectories
        ):
            raise ValueError(
                "Portfolio preparation requires exact source and complete coefficient-strength configurations"
            )
        return PreparedPortfolio(selected, source, context)
