"""Controlled full-information exponential mixtures with complete immutable state.

This owner applies one fixed exponential/share/joint-output law to each
nonempty released sum-loss block. Hard controls and a soft coefficient penalty
have independent meanings. No store, scheduler, transport or paid fee exists
at this mathematical layer.
"""

from __future__ import annotations

from coleman.core.actions import Weight
from coleman.core.feedback import FullPanel
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.temporal import Timestamp
from coleman.learning._controlled_output import release_proposal
from coleman.learning._controlled_state import (
    ApplicationClock,
    ReleaseContext,
    checked_state,
    controlled_binding,
    idle_outcome,
    initial_origin,
    logical_reference,
    net_usage,
    solved_outcome,
)
from coleman.learning.contracts import read_loss_panel
from coleman.learning.control_contracts import (
    AcceptedPanel,
    ClosedInterval,
    ControlledConfig,
    ControlledPayload,
    ControlOutcome,
    MovementPeriod,
)
from coleman.objectives.bindings import DefinitionBinding
from coleman.objectives.controls import (
    IntervalCadence,
    ReleaseInterval,
    RoundCadence,
    initial_exact_path_usage,
)
from coleman.state.contracts import AlgorithmState, StateBinding


def _replace(payload: ControlledPayload, **changes) -> ControlledPayload:
    """Revalidate a complete staged candidate instead of bypassing frozen invariants."""
    return ControlledPayload.model_validate(payload.model_dump() | changes)


def _processing(payload: ControlledPayload, clock: ApplicationClock, *, required: bool) -> None:
    """Require explicit monotone processing, retaining genuinely unknown times."""
    if required and clock.apply_at is None:
        raise ValueError("This controlled operation requires an explicit processing instant")
    if clock.apply_at is not None and payload.last_apply_at is not None and clock.apply_at < payload.last_apply_at:
        raise ValueError("Controlled processing cannot rewind its supplied application clock")
    movement = payload.config.controls.movement
    if movement is not None and movement.scope == "per_interval":
        if not payload.periods or clock.apply_at is None:
            raise ValueError("Per-interval movement requires an explicitly admitted period and application clock")
        period = payload.periods[-1]
        if not period.start <= clock.apply_at < period.end:
            raise ValueError("Application lies outside the current half-open movement period")


class ControlledHedge:
    """Own one constant-rate controlled learner, independently of any application.

    Parameters
    ----------
    config : ControlledConfig
        Complete versioned fixed law, loss interpretation, controls and limits.
    owner : Reference
        Stable logical state owner/version. No global registration occurs.

    Notes
    -----
    Only a closed immutable AlgorithmState is mutable through replacement.
    Invalid inputs/capacity failures leave it unchanged. A mathematically failed
    release retains its accepted panels, with U advanced and R unchanged;
    a failed round block must be retried explicitly before accepting more.
    Raw repeated panels are distinct observations, not durable receipt retries.
    The optional application/store layer owns producer and delivery identity.
    Calls on this mathematical owner must be sequential. Immutable candidates
    and a single state assignment do not synchronize concurrent callers; use
    independently owned instances and an explicitly qualified outer protocol.
    This algorithm has no inherited reference Hedge/Fixed Share guarantee.
    """

    def __init__(self, config: ControlledConfig, *, owner: Reference):
        """Validate the complete law and initial boxes before one state assignment."""
        selected = ControlledConfig.model_validate(config)
        binding = controlled_binding(selected, Reference.model_validate(owner))
        origin = initial_origin(selected, binding)
        payload = ControlledPayload(
            schema_version=1,
            config=selected,
            origin=origin,
            support=tuple(item.value > 0 for item in origin.weights),
            unresolved_failure=None,
        )
        self._state = checked_state(binding, payload)

    @property
    def config(self) -> ControlledConfig:
        """Return the complete immutable law; moving origins never rewrite it."""
        return self._state.payload.config

    @property
    def binding(self) -> StateBinding:
        """Return exact owner, algorithm, codec and fixed-configuration bindings."""
        return self._state.binding

    @property
    def members(self) -> tuple[Reference, ...]:
        """Return the complete fixed expert catalogue in canonical ID order."""
        return tuple(item.member for item in self.config.controls.initial_prior.weights)

    @property
    def loss_spec(self) -> ScalarSpec:
        """Return the declared dimensionless bounded [0,1] observation schema."""
        return self.config.loss_spec

    @property
    def transform(self) -> DefinitionBinding:
        """Return the exact declared transform binding, without fitting it."""
        return self.config.transform

    @property
    def weights(self) -> tuple[Weight, ...]:
        """Return actual emitted coefficient fractions, never holdings or trades."""
        return self._state.payload.origin.weights

    @property
    def update_count(self) -> int:
        """Return accepted complete panels U, including retained failed blocks."""
        return self._state.payload.update_count

    @property
    def release_count(self) -> int:
        """Return successful nonempty releases R, including optimal no-movement holds."""
        return self._state.payload.release_count

    @property
    def last_outcome(self) -> ControlOutcome | None:
        """Inspect the latest explicit disposition without applying a step."""
        return self._state.payload.last_outcome

    @property
    def unresolved_failure(self) -> ControlOutcome | None:
        """Inspect the blocked output frontier until a successful nonempty release.

        Returns
        -------
        ControlOutcome or None
            The original negative output still requiring recovery. Later held
            observations, empty points and new budget periods do not clear it.
            This is independent from the most recent observational receipt.
        """
        return self._state.payload.unresolved_failure

    def _commit(self, payload: ControlledPayload) -> None:
        """Validate complete content/capacity before the single live assignment."""
        self._state = checked_state(self.binding, payload)

    def observe(
        self,
        panel: FullPanel,
        /,
        *,
        eligible_at: Timestamp | None = None,
        apply_at: Timestamp | None = None,
    ) -> ControlOutcome:
        """Accept one complete panel and release only when its declared gate opens.

        Parameters
        ----------
        panel : FullPanel
            Every exact expert, one observed finite matching loss in [0,1].
        eligible_at, apply_at : datetime or None, optional
            Explicit eligibility and actual processing clocks. Interval cadence
            requires both; per-period movement requires application time. No
            receipt clock or current time is substituted for an unknown value.

        Returns
        -------
        ControlOutcome
            Held, released, infeasible or numerical-failure receipt. Accepted
            losses remain complete pending state until successful consumption.

        Raises
        ------
        ValueError, TypeError
            Invalid loss/clock, unadmitted period, full failed block or capacity;
            no state changes on these admission/serialization errors.
        """
        read_loss_panel(panel, members=self.members, loss_spec=self.loss_spec, transform=self.transform)
        before = self._state.payload
        clock = ApplicationClock(eligible_at=eligible_at, apply_at=apply_at)
        cadence = self.config.controls.cadence
        _processing(before, clock, required=isinstance(cadence, IntervalCadence))
        if isinstance(cadence, IntervalCadence) and clock.eligible_at is None:
            raise ValueError("Interval cadence requires explicit panel eligibility")
        if isinstance(cadence, RoundCadence) and len(before.pending) == cadence.panels:
            raise ValueError("A failed complete round block must be retried before accepting another panel")
        sequence = before.update_count + 1
        admitted = AcceptedPanel(
            schema_version=1,
            sequence=sequence,
            reference=logical_reference("accepted-panel", self.binding, sequence),
            panel=panel,
            eligible_at=clock.eligible_at,
            apply_at=clock.apply_at,
        )
        candidate = _replace(
            before,
            update_count=sequence,
            pending=(*before.pending, admitted),
            last_apply_at=clock.apply_at if clock.apply_at is not None else before.last_apply_at,
            last_outcome=None,
        )
        if isinstance(cadence, RoundCadence) and len(candidate.pending) == cadence.panels:
            candidate, outcome = self._attempt(candidate, candidate.pending, clock.apply_at, None)
        else:
            outcome = idle_outcome(candidate, "held")
            candidate = _replace(candidate, last_outcome=outcome)
        self._commit(candidate)
        return outcome

    def retry_release(self, *, apply_at: Timestamp | None = None) -> ControlOutcome:
        """Retry a retained complete failed round block under the current period.

        Parameters
        ----------
        apply_at : datetime or None, optional
            Explicit monotone processing instant; required for period movement.

        Returns
        -------
        ControlOutcome
            Original failure or one successful nonempty release. Repeating a
            successful release is rejected because its block was consumed.

        Raises
        ------
        ValueError
            Interval cadence, partial/absent block, clocks, periods or capacity.
        """
        before = self._state.payload
        cadence = self.config.controls.cadence
        if not isinstance(cadence, RoundCadence) or len(before.pending) != cadence.panels:
            raise ValueError("Explicit retry requires one complete retained round block")
        clock = ApplicationClock(apply_at=apply_at)
        _processing(before, clock, required=False)
        candidate, outcome = self._attempt(before, before.pending, clock.apply_at, None)
        self._commit(candidate)
        return outcome

    def release(self, interval: ReleaseInterval, *, calendar: Reference, apply_at: Timestamp) -> ControlOutcome:
        """Close one explicit interval after its application-owned drain barrier.

        Parameters
        ----------
        interval : ReleaseInterval
            Complete single cutoff/release point. Pending eligible panels are
            summed once; later panels never revise this closed attempt.
        calendar : Reference
            Exact declared interval-clock identity/version.
        apply_at : datetime
            Explicit processing instant equal to interval.release_at.

        Returns
        -------
        ControlOutcome
            Released, empty or negative result. An exact retained interval
            retry returns its original result without resetting state or usage.

        Raises
        ------
        ValueError
            Conflicting identity, wrong calendar/cadence, rewound/overlapping
            interval, wrong processing instant, movement period or capacity.

        Notes
        -----
        This pure call cannot prove an external drain completed. The optional
        replay host invokes it inside decide after CausalReplay's closed drain.
        """
        checked = ReleaseInterval.model_validate(interval)
        before = self._state.payload
        cadence = self.config.controls.cadence
        if not isinstance(cadence, IntervalCadence) or calendar != cadence.calendar:
            raise ValueError("Release requires the configured interval cadence/calendar")
        clock = ApplicationClock(apply_at=apply_at)
        if clock.apply_at != checked.release_at:
            raise ValueError("Actual processing must equal the declared single release point")
        for closed in before.intervals:
            if closed.interval.interval == checked.interval:
                if closed.interval.model_dump_json() != checked.model_dump_json():
                    raise ValueError("Conflicting body at one retained interval identity")
                return closed.outcome
        _processing(before, clock, required=True)
        if before.intervals and checked.start < before.intervals[-1].interval.end:
            raise ValueError("Release intervals cannot overlap or rewind")
        block = tuple(
            item
            for item in before.pending
            if item.eligible_at is not None
            and item.eligible_at <= checked.cutoff
            and item.apply_at is not None
            and item.apply_at <= checked.release_at
        )
        if block:
            candidate, outcome = self._attempt(before, block, clock.apply_at, checked)
        else:
            outcome = idle_outcome(before, "empty")
            candidate = _replace(before, last_outcome=outcome, last_apply_at=clock.apply_at)
        candidate = _replace(
            candidate, intervals=(*candidate.intervals, ClosedInterval(interval=checked, outcome=outcome))
        )
        self._commit(candidate)
        return outcome

    def begin_period(self, period: MovementPeriod, *, apply_at: Timestamp) -> bool:
        """Admit one explicit movement-budget period without releasing panels.

        Parameters
        ----------
        period : MovementPeriod
            Nonoverlapping half-open UTC period under the configured clock.
        apply_at : datetime
            Explicit monotone processing instant within this period.

        Returns
        -------
        bool
            True for one newly admitted period; False for an exact historical
            identity retry. A retry never rewinds/reset its origin or usage.

        Raises
        ------
        ValueError
            Wrong movement scope/clock, conflict, rewind or selected capacity.
        """
        selected = MovementPeriod.model_validate(period)
        before = self._state.payload
        movement = self.config.controls.movement
        if movement is None or movement.scope != "per_interval" or selected.calendar != self.config.period_clock:
            raise ValueError("Movement period requires the configured per-interval movement clock")
        clock = ApplicationClock(apply_at=apply_at)
        if clock.apply_at is None or not selected.start <= clock.apply_at < selected.end:
            raise ValueError("Period admission requires an explicit instant inside its half-open boundaries")
        for known in before.periods:
            if known.interval == selected.interval:
                if known.to_json() != selected.to_json():
                    raise ValueError("Conflicting body at one movement-period identity")
                return False
        if before.last_apply_at is not None and clock.apply_at < before.last_apply_at:
            raise ValueError("Movement-period admission cannot rewind processing")
        if before.periods and selected.start < before.periods[-1].end:
            raise ValueError("Movement periods cannot overlap or rewind")
        path = (
            initial_exact_path_usage(interval=selected.interval, movement=movement, origin=before.origin)
            if movement.measure == "accumulated_absolute"
            else None
        )
        candidate = _replace(
            before,
            periods=(*before.periods, selected),
            period_origin=before.origin,
            exact_path=path,
            last_apply_at=clock.apply_at,
        )
        self._commit(candidate)
        return True

    def _attempt(
        self,
        before: ControlledPayload,
        block: tuple[AcceptedPanel, ...],
        apply_at: Timestamp | None,
        interval: ReleaseInterval | None,
    ) -> tuple[ControlledPayload, ControlOutcome]:
        """Stage exactly one exp/share/joint solve and its finite complete receipt."""
        release = logical_reference(
            "controlled-release",
            self.binding,
            {
                "next": before.release_count + 1,
                "panels": [item.reference.model_dump(mode="json") for item in block],
                "interval": None if interval is None else interval.model_dump(mode="json"),
                "period": None if not before.periods else before.periods[-1].model_dump(mode="json"),
            },
        )
        attempt = release_proposal(
            self.config, before.origin, block, usage=net_usage(before), exact_path=before.exact_path
        )
        context = ReleaseContext(release, tuple(item.reference for item in block), apply_at)
        outcome, path = solved_outcome(before, self.binding, attempt, context)
        successful = outcome.disposition == "released"
        origin = outcome.emitted_origin if outcome.emitted_origin is not None else before.origin
        consumed = {item.reference.id for item in block} if successful else set()
        candidate = _replace(
            before,
            origin=origin,
            support=tuple(item.value > 0 for item in origin.weights),
            exact_path=path,
            release_count=outcome.release_count,
            consumed_panels=outcome.consumed_panels,
            pending=tuple(item for item in before.pending if item.reference.id not in consumed),
            last_apply_at=apply_at if apply_at is not None else before.last_apply_at,
            last_outcome=outcome,
            unresolved_failure=None if successful else outcome,
        )
        return candidate, outcome

    def export_state(self) -> AlgorithmState[ControlledPayload]:
        """Return complete immutable state, retaining pending losses and exact usage.

        Returns
        -------
        AlgorithmState
            Strict version-one owner/configuration-bound continuation. No I/O.
        """
        return checked_state(self.binding, self._state.payload)

    def restore_state(self, state: AlgorithmState[ControlledPayload], /) -> None:
        """Restore one explicitly compatible complete state without migration.

        Parameters
        ----------
        state : AlgorithmState
            Concrete ControlledPayload with the exact owner/law/codec binding.

        Raises
        ------
        ValueError, TypeError
            Unknown/future version, wrong owner/configuration/codec, malformed
            joins, missing pending/usage state or selected capacity. The live
            owner remains unchanged until all validation succeeds.
        """
        checked = AlgorithmState[ControlledPayload].model_validate(state)
        if checked.binding != self.binding or checked.payload.config.to_json() != self.config.to_json():
            raise ValueError("Controlled state differs from this exact owner/configuration/codec binding")
        self._state = checked_state(self.binding, checked.payload)
