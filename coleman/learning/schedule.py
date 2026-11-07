"""Optional explicit application clocks around the pure controlled mixture.

The schedule is fixed configuration, not an inferred market calendar. Releases
occur inside the direct decision callback, after an enclosing causal replay's
closed eligible-feedback drain. No scheduler or store is introduced.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.actions import AbstainAction, Action
from coleman.core.capabilities import Capability
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch, FullPanel
from coleman.core.identity import Reference
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.temporal import Timestamp
from coleman.learning.control_contracts import (
    ControlledConfig,
    ControlledPayload,
    MovementPeriod,
    _ControllerContract,
)
from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectMixture, DirectMixtureState, DirectOptions
from coleman.objectives.controls import IntervalCadence, ReleaseInterval
from coleman.state.contracts import AlgorithmState, StateBinding


class ControlSchedule(_ControllerContract):
    """Declare finite immutable release and movement-period application clocks.

    Attributes
    ----------
    profile : {'explicit-host-release-plan-v1'}
        Supplied UTC clocks; no business-day, timezone or clock inference.
    calendar : Reference or None
        Required configured cadence calendar for interval mode; null for rounds.
    intervals : tuple of ReleaseInterval
        Nonoverlapping chronological release intervals, with unique identities.
    periods : tuple of MovementPeriod
        Nonoverlapping chronological hard-budget periods, with unique identities.

    Notes
    -----
    Empty schedules are valid only for an untimed round/per-release law.
    Complete schedule bytes are part of the outer snapshot binding. Changing
    a schedule requires an explicitly new application lineage, never a hidden
    resumed-calendar substitution.
    """

    profile: Literal["explicit-host-release-plan-v1"] = "explicit-host-release-plan-v1"
    calendar: Reference | None
    intervals: Annotated[tuple[ReleaseInterval, ...], Field(max_length=4096)] = ()
    periods: Annotated[tuple[MovementPeriod, ...], Field(max_length=4096)] = ()

    @model_validator(mode="after")
    def _sequences(self) -> Self:
        """Retain complete chronological declarations without collapsing retries."""
        if (self.calendar is None) != (not self.intervals):
            raise ValueError("Exactly nonempty release intervals require a cadence calendar")
        for sequence in (self.intervals, self.periods):
            identities = tuple(item.interval.model_dump_json() for item in sequence)
            if len(set(identities)) != len(identities):
                raise ValueError("Scheduled logical interval identities must be unique")
            if any(later.start < previous.end for previous, later in zip(sequence, sequence[1:], strict=False)):
                raise ValueError("Scheduled intervals cannot overlap or rewind")
        return self


def _configuration(config: ControlledConfig, schedule: ControlSchedule) -> None:
    """Reject a schedule incompatible with the complete fixed mathematical law."""
    cadence = config.controls.cadence
    expected = cadence.calendar if isinstance(cadence, IntervalCadence) else None
    if schedule.calendar != expected:
        raise ValueError("Control schedule differs from the fixed cadence calendar")
    if (config.period_clock is None) != (not schedule.periods):
        raise ValueError("Exactly per-period movement requires scheduled movement periods")
    if any(item.calendar != config.period_clock for item in schedule.periods):
        raise ValueError("Scheduled movement periods use another declared clock")
    if max(len(schedule.intervals), len(schedule.periods)) > config.limits.maximum_intervals:
        raise ValueError("Control schedule exceeds selected retained interval capacity")


def _binding(inner: StateBinding, schedule: ControlSchedule) -> StateBinding:
    """Bind the complete producer bridge and original clock-plan bytes."""
    return StateBinding(
        owner=inner.owner,
        implementation=inner.implementation,
        state_codec=Reference(id="controlled-direct-schedule-state", version="1"),
        configuration_fingerprint=fingerprint(
            {"inner": inner.model_dump(mode="json"), "schedule": schedule.model_dump(mode="json")}
        ),
    )


class ControlledDirectState(_ControllerContract):
    """Capture actual mathematical, producer and immutable schedule state.

    Attributes
    ----------
    binding : StateBinding
        Outer owner/configuration/codec including the complete clock schedule.
    schedule : ControlSchedule
        Original supplied release/budget declarations, never read from a clock.
    direct : DirectMixtureState
        All original pending executions, D/U and complete ControlledPayload.
    """

    binding: StateBinding
    schedule: ControlSchedule
    direct: DirectMixtureState[ControlledPayload]

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Join schedule, fixed law and original producer binding before restore."""
        if self.binding != _binding(self.direct.binding, self.schedule):
            raise ValueError("Controlled application snapshot has an incompatible schedule binding")
        _configuration(self.direct.method.payload.config, self.schedule)
        if len(canonical_json(self.model_dump(mode="json")).encode("utf-8")) > self.direct.options.maximum_bytes:
            raise ValueError("Controlled application complete state exceeds selected bytes")
        return self


class _ScheduledBridge(DirectMixture[ControlledPayload]):
    """Apply explicit clocks inside captured decision/update host operations."""

    def __init__(self, learner: ControlledHedge, schedule: ControlSchedule, **options):
        """Retain immutable clocks before constructing the actual producer owner."""
        self._controller = learner
        self._schedule = schedule
        super().__init__(learner, **options)

    def _snapshot(
        self,
        method: AlgorithmState[ControlledPayload],
        *,
        pending: tuple[ExecutedDecision, ...],
        decisions: int,
        updates: int,
    ) -> DirectMixtureState[ControlledPayload]:
        """Admit complete outer schedule bytes before the producer commit."""
        result = super()._snapshot(method, pending=pending, decisions=decisions, updates=updates)
        ControlledDirectState(
            schema_version=1, binding=_binding(result.binding, self._schedule), schedule=self._schedule, direct=result
        )
        return result

    def _period(self, apply_at: Timestamp | None) -> None:
        """Admit only the explicitly covering budget period; never infer one."""
        if self._controller.config.period_clock is None:
            return
        if apply_at is None:
            raise ValueError("Scheduled per-period control requires explicit application time")
        matches = tuple(item for item in self._schedule.periods if item.start <= apply_at < item.end)
        if len(matches) != 1:
            raise ValueError("No declared movement period covers this actual processing instant")
        self._controller.begin_period(matches[0], apply_at=apply_at)

    def _before_decision(self, request: DecisionRequest) -> None:
        """Release at a declared point after the enclosing replay drain barrier."""
        at = None if request.timing is None else request.timing.decision_at
        self._period(at)
        if self._schedule.calendar is None:
            return
        if request.timing is None:
            raise ValueError("Interval-controlled decisions require explicit decision/cutoff clocks")
        points = tuple(item for item in self._schedule.intervals if item.release_at == at)
        if points:
            point = points[0]
            if point.cutoff != request.timing.information_cutoff:
                raise ValueError("Release cutoff differs from the captured decision's closed drain cutoff")
            self._controller.release(point, calendar=self._schedule.calendar, apply_at=request.timing.decision_at)

    def _action(self, request: DecisionRequest) -> Action:
        """Declare explicit abstention after an unresolved failed output attempt."""
        outcome = self._controller.unresolved_failure
        if outcome is not None and outcome.disposition in {"infeasible", "numerical_failure"}:
            return AbstainAction(
                schema_version=1, space=request.space, reason="controlled-output-" + outcome.disposition
            )
        return super()._action(request)

    def _observe(self, panel: FullPanel, *, eligible_at: Timestamp | None, apply_at: Timestamp | None) -> object:
        """Apply the actual supplied processing clock, independently of receipt."""
        self._period(apply_at)
        return self._controller.observe(panel, eligible_at=eligible_at, apply_at=apply_at)


class ControlledDirect:
    """Compose a pure controlled learner and one explicit immutable clock plan.

    Parameters
    ----------
    learner : ControlledHedge
        Actual independently owned mathematical state.
    schedule : ControlSchedule
        Complete fixed application clock declarations.
    options : DirectOptions
        Exact task/objective/horizon and finite producer/complete-state limits.

    Notes
    -----
    This optional adapter performs no scheduling, store access or physical
    execution. A containing CausalReplay supplies the closed drain barrier.
    Each wrapper, producer bridge and actual learner is an owned live root.
    It abstains explicitly after an unresolved negative output; a new compatible
    staged configuration or explicit mathematical recovery is application work.
    Trusted host failures invalidate the producer bridge without rollback.
    """

    def __init__(
        self,
        learner: ControlledHedge,
        schedule: ControlSchedule,
        options: DirectOptions,
    ):
        """Bind exact schedule/options and validate the whole initial snapshot."""
        checked = ControlSchedule.model_validate(schedule)
        _configuration(learner.config, checked)
        self._schedule = checked
        self._direct = _ScheduledBridge(
            learner,
            checked,
            task=options.task,
            objective=options.objective,
            horizon=options.horizon,
            maximum_pending=options.maximum_pending,
            maximum_bytes=options.maximum_bytes,
        )
        self._binding = _binding(self._direct.binding, checked)
        self.export_state()

    @property
    def learner(self) -> ControlledHedge:
        """Inspect the actual exclusively owned controller without external updates."""
        return self._direct._controller  # pylint: disable=protected-access

    @property
    def bridge(self) -> DirectMixture[ControlledPayload]:
        """Inspect the actual producer owner for an explicit live-root inventory."""
        return self._direct

    @property
    def binding(self) -> StateBinding:
        """Return complete fixed owner/method/producer/schedule interpretation."""
        return self._binding

    @property
    def state_type(self) -> type[ControlledDirectState]:
        """Return the trusted concrete complete application state codec."""
        return ControlledDirectState

    @property
    def capability(self) -> Capability:
        """Return full observed panel capability, with no regret/risk assertion."""
        return self._direct.capability

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Emit current coefficients or explicit abstention inside captured clocks.

        Parameters
        ----------
        request : DecisionRequest
            Complete original request; interval release requires exact clocks.

        Returns
        -------
        Decision
            Explicit offline-identity proposal, not physical execution.
        """
        return self._direct.decide(request)

    def update(
        self, feedback: FeedbackBatch, /, *, eligible_at: Timestamp | None = None, apply_at: Timestamp | None = None
    ) -> object:
        """Admit the actual original producer's complete observed loss panel.

        Parameters
        ----------
        feedback : FeedbackBatch
            Selected original-producer batch; no hidden correction selection.
        eligible_at, apply_at : datetime or None, optional
            Actual explicitly supplied clocks, required by interval controls.

        Returns
        -------
        object
            Typed held/released/negative mathematical disposition.
        """
        return self._direct.update(feedback, eligible_at=eligible_at, apply_at=apply_at)

    def export_state(self) -> ControlledDirectState:
        """Export every mathematical/producer/calendar field without I/O.

        Returns
        -------
        ControlledDirectState
            Complete strict version-one bounded application continuation.
        """
        return ControlledDirectState(
            schema_version=1, binding=self.binding, schedule=self._schedule, direct=self._direct.export_state()
        )

    def restore_state(self, state: ControlledDirectState, /) -> None:
        """Restore only the exact owner/clock-plan/complete producer lineage.

        Parameters
        ----------
        state : ControlledDirectState
            Complete matching strict v1 snapshot.

        Raises
        ------
        ValueError, TypeError
            Invalid/future version, mismatched plan, identity, joins or capacity.
        LearningApplicationError
            A trusted mathematical restore failed and invalidated the bridge.
        """
        checked = ControlledDirectState.model_validate(state)
        if checked.binding != self.binding or checked.schedule.to_json() != self._schedule.to_json():
            raise ValueError("Controlled application restore differs from the exact fixed clock-plan binding")
        self._direct.restore_state(checked.direct)
