"""Structural external executor; the host owns all queue/coordinate scheduling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from coleman.artifacts import ArtifactWriter
from coleman.objectives.bindings import DefinitionBinding
from coleman.tasks.contracts import OwnedCommitProof
from coleman.tasks.protocols import AttemptContext, CommittedProgress, CoordinateContext, PreparationContext

from . import recipe as declare_recipe
from ._state import Layout, read_text
from .contracts import HedgeConfiguration, HedgeCoordinateResult, HedgeInputs, HedgeReport, HedgeTrajectory
from .recipe import (
    CONFIGURATION,
    COORDINATE,
    COORDINATE_RESULT,
    FINAL_RESULT,
    INPUTS,
    PROGRESS_PROOF,
    STATE,
    ModelCodec,
    frontier_definition,
)


@dataclass(frozen=True)
class PreparedHedge:
    """Keep only immutable admitted inputs; no live learner survives a stage.

    Parameters
    ----------
    configuration, inputs : HedgeConfiguration, HedgeInputs
        Complete revalidated law and original source declaration.
    context : PreparationContext
        Exact admitted task identity, actual compatibility and trusted job root.
    """

    configuration: HedgeConfiguration
    inputs: HedgeInputs
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
    def progress_verifier(self) -> PreparedHedge:
        """Expose actual storage recovery, distinct from the pure proof codec."""
        return self

    @property
    def state_codec(self) -> ModelCodec:
        """Return the complete outer state codec without capturing or restoring."""
        return STATE

    def _layout(self, trajectory: HedgeTrajectory, context: CoordinateContext) -> Layout:
        """Join concrete selected data and all admitted context before I/O."""
        current = COORDINATE.validate(trajectory)
        known = next((item for item in self.inputs.trajectories if item.reference == current.reference), None)
        if known is None or known.to_json() != current.to_json():
            raise ValueError("Hedge coordinate must match its complete original declared source")
        if (
            context.preparation != self.context
            or context.reference != current.reference
            or context.target.coordinate != current.reference
            or context.target.profile != frontier_definition()
            or context.target.position != len(current.panels)
        ):
            raise ValueError("Hedge coordinate requires exact admitted context, work profile and target")
        return Layout(self.configuration, current, context)

    def _context_layout(self, context: CoordinateContext) -> Layout:
        """Resolve the source-owned trajectory without looking up code from JSON."""
        coordinate = next((item for item in self.inputs.trajectories if item.reference == context.reference), None)
        if coordinate is None:
            raise ValueError("Hedge continuation has no matching original trajectory")
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
            Current state-derived U and actual retained receipt, or virgin state.

        Raises
        ------
        ValueError, OSError
            Changed context, missing acknowledged state, corrupt/incompatible
            artifacts or failed exact original-command receipt reconciliation.
        """
        layout = self._context_layout(context)
        store = layout.open()
        return None if store is None or store.read_head() is None else layout.reconcile(store)

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
            raise ValueError("Hedge progress requires actual acknowledged state")
        if (progress.plan_fingerprint, progress.coordinate_fingerprint, progress.frontier) != (
            actual.plan_fingerprint,
            actual.coordinate_fingerprint,
            actual.frontier,
        ) or candidate.model_dump(exclude={"receipt": {"status"}}) != actual.proof.model_dump(
            exclude={"receipt": {"status"}}
        ):
            raise ValueError("Hedge progress differs from the actual current ACK and complete state")

    def read_completed(
        self, coordinate: HedgeTrajectory, context: CoordinateContext, /
    ) -> HedgeCoordinateResult | None:
        """Read only a published complete result and verify its actual current ACK.

        Parameters
        ----------
        coordinate : HedgeTrajectory
            Exact original declared trajectory.
        context : CoordinateContext
            Actual retained task and trusted result directory.

        Returns
        -------
        HedgeCoordinateResult or None
            Complete rechecked result, or unpublished absence. Corruption raises.
        """
        layout = self._layout(coordinate, context)
        path = context.artifact_directory / "coordinate-result.json"
        if not path.exists():
            layout.open()  # Check a retained marker even when publication was interrupted.
            return None
        result = COORDINATE_RESULT.decode(read_text(path, COORDINATE_RESULT.maximum_bytes))
        self._require_result(layout, result)
        return result

    def _require_result(self, layout: Layout, result: HedgeCoordinateResult) -> None:
        """Reject complete-looking files without exact terminal current state."""
        self.validate(
            CommittedProgress(
                result.proof.plan_fingerprint,
                result.proof.coordinate_fingerprint,
                result.proof.frontier,
                result.proof,
            ),
            layout.context,
        )
        store = layout.open()
        if store is None or result.checkpoint.to_json() != layout.checkpoint(store).to_json():
            raise ValueError("Hedge result must retain the complete actual current checkpoint")
        if result.proof.frontier.position != len(layout.trajectory.panels):
            raise ValueError("Hedge result requires every original panel to be acknowledged")

    def advance(
        self,
        coordinate: HedgeTrajectory,
        context: AttemptContext,
        committed: Callable[[CommittedProgress[OwnedCommitProof]], None],
        /,
    ) -> HedgeCoordinateResult:
        """Finish one original trajectory through bounded actual store transitions.

        Parameters
        ----------
        coordinate : HedgeTrajectory
            Original finite loss trajectory; this is no job scheduling loop.
        context : AttemptContext
            Actual reserved attempt, deadline and common guard.
        committed : callable
            Called only after real ACKs. A callback failure leaves the accepted
            inner state recoverable, without repeating accepted observations.

        Returns
        -------
        HedgeCoordinateResult
            Complete final state and an actual retained receipt for its current
            command. Outer coordinate/job publication is owned by the driver.
        """
        layout = self._layout(coordinate, context.coordinate)
        context.guard()
        store = layout.open(create=True)
        assert store is not None
        layout.initialize(store)
        while layout.checkpoint(store).learner.payload.update_count < len(coordinate.panels):
            committed(layout.advance(store, context.guard))
        progress = layout.reconcile(store)
        return HedgeCoordinateResult(
            contract="coleman-typed-hedge-reference-v1",
            schema_version=1,
            checkpoint=layout.checkpoint(store),
            proof=progress.proof,
        )

    def publish_coordinate(
        self,
        coordinate: HedgeTrajectory,
        value: HedgeCoordinateResult,
        context: CoordinateContext,
        /,
    ) -> None:
        """Publish rechecked immutable result bytes, without a job-complete claim.

        Parameters
        ----------
        coordinate : HedgeTrajectory
            Exact original source coordinate.
        value : HedgeCoordinateResult
            Complete real ACK-backed result, not a partial artifact locator.
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

    prepared: PreparedHedge

    def produce(self, load_completed: Callable[[], tuple[HedgeCoordinateResult, ...]], /) -> HedgeReport:
        """Load once and require every exact original owner/source/plan result."""
        values = load_completed()
        if tuple(item.checkpoint.binding.owner for item in values) != tuple(
            item.reference for item in self.prepared.inputs.trajectories
        ):
            raise ValueError("Hedge boundary requires every original source trajectory in declared order")
        if any(
            item.checkpoint.plan_fingerprint != self.prepared.context.plan_fingerprint
            or item.checkpoint.source != self.prepared.inputs.binding
            for item in values
        ):
            raise ValueError("Hedge boundary requires its exact admitted plan and complete source")
        return HedgeReport(
            contract="coleman-typed-hedge-reference-v1",
            schema_version=1,
            interpretation="constructed_mechanism_only",
            results=values,
        )


class HedgeExecutor:
    """Expose actual concrete codecs and one preparation factory, without I/O.

    Notes
    -----
    Direct shipped Hedge callers need none of these adapters or registrations.
    The task driver owns attempts, queueing, resource guards and publication.
    """

    @property
    def configuration_codec(self) -> ModelCodec[HedgeConfiguration]:
        """Return the selected complete mathematical configuration codec."""
        return CONFIGURATION

    @property
    def input_codec(self) -> ModelCodec[HedgeInputs]:
        """Return the separate complete original-source codec."""
        return INPUTS

    @property
    def coordinate_codec(self) -> ModelCodec[HedgeTrajectory]:
        """Return the selected original trajectory codec."""
        return COORDINATE

    @property
    def coordinate_result_codec(self) -> ModelCodec[HedgeCoordinateResult]:
        """Return the complete per-trajectory state-and-ACK result codec."""
        return COORDINATE_RESULT

    @property
    def final_result_codec(self) -> ModelCodec[HedgeReport]:
        """Return the complete constructed descriptive evidence codec."""
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
        configuration: HedgeConfiguration,
        inputs: HedgeInputs,
        context: PreparationContext,
        /,
    ) -> PreparedHedge:
        """Join complete values and actual host bindings without creating owners.

        Parameters
        ----------
        configuration, inputs : HedgeConfiguration, HedgeInputs
            Actual selected concrete values; full source stays separate from law.
        context : PreparationContext
            Admitted plan/source, captured implementation and trusted job root.

        Returns
        -------
        PreparedHedge
            Immutable capabilities; actual learners are created only in accepted
            local-store staging, never during preparation or stale retries.

        Raises
        ------
        ValueError
            Source or any initial full mathematical configuration differs.
        """
        selected, source = CONFIGURATION.validate(configuration), INPUTS.validate(inputs)
        if source.binding != context.source or any(
            item.initial_state.payload.configuration.to_json() != selected.reference.to_json()
            for item in source.trajectories
        ):
            raise ValueError("Hedge preparation requires exact source and complete mathematical configurations")
        return PreparedHedge(selected, source, context)
