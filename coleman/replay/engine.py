"""Single-owner causal replay with explicit processing clocks and no storage."""

from __future__ import annotations

# Private seeded hooks are shared only within the replay implementation family.
# pylint: disable=protected-access
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Literal, Protocol, Self, cast

from coleman.core.base import CoreContract
from coleman.core.capabilities import validate_decision
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.identity import Reference
from coleman.core.primitives import MAX_MEMBERS
from coleman.core.protocols import Decider, Stateful, Updater
from coleman.core.temporal import Timestamp
from coleman.observations.contracts import ObservationHistory, ObservationSnapshot
from coleman.replay.contracts import (
    PreparedDecision,
    ReplayApplication,
    ReplayContinuation,
    ReplayDecisionRecord,
    ReplayDrain,
    ReplayFeedback,
    ReplayOpportunity,
    ReplayParent,
    ReplayPolicy,
    ReplayState,
    ReplayUpdateRecord,
)
from coleman.replay.planning import (
    PendingRound,
    _applications,
    _capacity,
    _feedback_frontier,
    _frontiers,
    _history_update,
    _instant,
    _opportunity,
    _pending,
)
from coleman.replay.revisions import validate_feedback_graph


class ReplayConsumer[S: CoreContract](
    Decider[DecisionRequest, Decision], Updater[ReplayApplication, object], Stateful[S], Protocol
):
    """Compose direct protocols for one optional qualified replay host.

    Notes
    -----
    No inheritance or registration is required. ``S`` must be a concrete complete
    versioned snapshot: all statistics, RNG, caches, pending and decision effects
    have one host owner. Export is side-effect-free. Update receives the actual
    supplied processing clock, not an inferred receipt time. A host adapter can
    unwrap this envelope for an independently usable mathematical learner.
    """


class ReplayBusyError(RuntimeError):
    """Reject concurrent or reentrant operations on the same replay session.

    Notes
    -----
    The rejected operation performs no host call. Independent sessions require
    independent hosts; this process-local guard is not shared-writer recovery.
    """


class ReplayInvalidatedError(RuntimeError):
    """Refuse continued use after a possibly partial host call or stale binding.

    Notes
    -----
    Use a fresh host and an explicitly compatible complete continuation. No
    rollback, durable acknowledgement or external-effect reconciliation occurs.
    """


def _continuation_type[S: CoreContract](state_type: type[S]) -> type[ReplayContinuation[S]]:
    """Select the trusted concrete snapshot type, preserving every host field."""
    return cast(type[ReplayContinuation[S]], ReplayContinuation.__class_getitem__(state_type))


class CausalReplay[S: CoreContract]:
    """Own bounded delivery metadata and drive one explicitly owned host.

    Parameters
    ----------
    host : ReplayConsumer[S]
        Actual direct decider/updater and complete state owner. No history,
        future feedback table or unchosen outcome oracle is passed to it.
    policy : ReplayPolicy
        Explicit versioned source, capability, clocks and correction profile.
        An independent origin requires a null parent; use a matched fork for a
        child lineage.
    state_type : type[S]
        Concrete CoreContract subclass selected by trusted host code, never
        dynamically imported from an artifact.
    state_codec : Reference
        Exact host snapshot generation and interpretation.

    Raises
    ------
    ValueError, TypeError
        Invalid profile, state type, parent or complete exported snapshot.
    Exception
        Actual host export failures propagate. The host must be independently
        owned and its export side-effect-free.

    Notes
    -----
    This optional in-memory layer performs no I/O, reads no clock and owns no
    RNG. Explicit drains are processing inputs; changing their schedule may
    change revision selection. Host calls may mutate state. Any host-call or
    post-call validation failure invalidates this session; it is not rolled
    back or exposed as a successful continuation. Durable commits and external
    effects require a separately qualified outer protocol.
    """

    def __init__(
        self,
        host: ReplayConsumer[S],
        policy: ReplayPolicy,
        *,
        state_type: type[S],
        state_codec: Reference,
    ) -> None:
        """Validate an independent origin before retaining its complete state."""
        checked = ReplayPolicy.model_validate(policy)
        if checked.parent is not None:
            raise ValueError("A child replay requires a complete matched parent continuation")
        codec = Reference.model_validate(state_codec)
        self._configure(host, state_type)
        snapshot = self._snapshot()
        self._state = ReplayState(
            schema_version=1,
            policy=checked,
            history=None,
            state_codec=codec,
            host_fingerprint=snapshot.fingerprint,
        )

    def _configure(self, host: ReplayConsumer[S], state_type: type[S]) -> None:
        """Configure only process-local ownership; do not load a dynamic codec."""
        if not isinstance(state_type, type) or not issubclass(state_type, CoreContract) or state_type is CoreContract:
            raise TypeError("Replay requires an explicit concrete CoreContract host snapshot type")
        self._host = host
        self._state_type = state_type
        self._lock = Lock()
        self._invalidated = False

    def _validate_state(self, state: ReplayState) -> ReplayState:
        """Select the unchanged v1 contract unless an explicit profile overrides."""
        return ReplayState.model_validate(state)

    def _continuation(self, snapshot: S) -> ReplayContinuation[S]:
        """Compose the explicitly selected complete snapshot generation."""
        return _continuation_type(self._state_type)(schema_version=1, state=self._state, host_state=snapshot)

    def _state_fingerprint(self) -> str:
        """Keep the original validating identity path for complete v1 state."""
        return self._state.fingerprint

    def _same_state(self, candidate: ReplayState) -> bool:
        """Compare a newly validated candidate with the current owned state."""
        return candidate.to_json() == self._state.to_json()

    def _alive(self) -> None:
        """Prevent a failed host transition from becoming reusable metadata."""
        if self._invalidated:
            raise ReplayInvalidatedError("Replay host/session was invalidated; restore a fresh complete owner")

    @contextmanager
    def _operation(self) -> Iterator[None]:
        """Reject reentrancy instead of depending on accidental thread order."""
        if not self._lock.acquire(blocking=False):
            raise ReplayBusyError("Replay session already has an active operation")
        try:
            self._alive()
            yield
        finally:
            self._lock.release()

    @contextmanager
    def _host_boundary(self) -> Iterator[None]:
        """Invalidate on a host failure, preserving its original exception."""
        try:
            yield
        except BaseException:
            # This is cleanup, not recovery: interrupts are re-raised unchanged.
            # A callback may already have changed statistics or consumed RNG.
            self._invalidated = True
            raise

    def _snapshot(self) -> S:
        """Revalidate exactly the declared complete, side-effect-free snapshot."""
        with self._host_boundary():
            snapshot = self._host.export_state()
            if type(snapshot) is not self._state_type:  # pylint: disable=unidiomatic-typecheck
                raise TypeError("Host exported a different concrete snapshot codec/type")
            return self._state_type.model_validate(snapshot)

    def _verify_host(self) -> S:
        """Refuse a stale metadata value paired with an already advanced host."""
        snapshot = self._snapshot()
        if snapshot.fingerprint != self._state.host_fingerprint:
            self._invalidated = True
            raise ReplayInvalidatedError("Host state changed outside its replay owner")
        return snapshot

    @property
    def state(self) -> ReplayState:
        """Return retained immutable metadata, without claiming a durable commit.

        Returns
        -------
        ReplayState
            Current full inventory and causal records. Future admission may
            change its digest without changing the prior causal-prefix digest.

        Raises
        ------
        ReplayBusyError, ReplayInvalidatedError
            The session is active or a host transition failed.

        Notes
        -----
        This property calls no host. Export a complete continuation to verify
        the actual host binding before caller-owned persistence.
        """
        with self._operation():
            return self._state

    @property
    def pending(self) -> tuple[PendingRound, ...]:
        """Inspect explicit unapplied states without delivering values to a host.

        Returns
        -------
        tuple of PendingRound
            Audit reasons for unknown receipt, unavailable clocks, partial
            components, missing required values and explicit absence.

        Raises
        ------
        ReplayBusyError, ReplayInvalidatedError
            The session is active or invalidated.

        Notes
        -----
        Pending inventory is excluded from earlier causal-prefix identities.
        """
        with self._operation():
            return _pending(self._state)

    def admit_history(self, history: ObservationHistory) -> ReplayState:
        """Install an append-only source inventory without performing an update.

        Parameters
        ----------
        history : ObservationHistory
            Complete bounded revision graph. Every previous revision must
            remain byte-identical, including original first receipts.

        Returns
        -------
        ReplayState
            New immutable inventory, or the identical state for an exact retry.

        Raises
        ------
        ValueError
            Wrong source/version, deletion, conflicting revision or a backfill
            changing a closed frontier or any prior eligible decision view.
        ReplayBusyError, ReplayInvalidatedError
            This owner is active or unusable.

        Notes
        -----
        Reuses inclusive ObservationHistory.as_of. Legacy source revisions have
        no producer-phase annotation, so no same-time backfill is inferred.
        No source, host, storage or clock is invoked.
        """
        with self._operation():
            checked = _history_update(self._state, history)
            if self._state.history is not None and checked.to_json() == self._state.history.to_json():
                return self._state
            self._state = self._validate_state(self._state.model_copy(update={"history": checked}))
            return self._state

    def eligible_snapshot(self, cutoff: Timestamp) -> ObservationSnapshot:
        """Select source inputs for feature construction without advancing state.

        Parameters
        ----------
        cutoff : datetime
            Explicit zoned information instant; no ambient clock is used.

        Returns
        -------
        ObservationSnapshot
            Actual inclusive as_of view, with no future revisions.

        Raises
        ------
        ValueError
            No source history is admitted or the timestamp is invalid.
        ReplayBusyError, ReplayInvalidatedError
            This owner is active or unusable.

        Notes
        -----
        The caller may derive typed request features from this view. A matching
        digest is a consistency check, not proof of arbitrary feature derivation.
        """
        with self._operation():
            if self._state.history is None:
                raise ValueError("Admit a complete source history before selecting replay inputs")
            return self._state.history.as_of(_instant(cutoff))

    def _retained(self, opportunity: ReplayOpportunity) -> ReplayDecisionRecord | None:
        """Reconcile accepted logical retries before checking current frontiers."""
        for record in self._state.decisions:
            if record.opportunity.request.logical_identity == opportunity.request.logical_identity:
                if record.opportunity.to_json() != opportunity.to_json():
                    raise ValueError("Conflicting content for an already emitted logical replay opportunity")
                return record
        return None

    def prepare(self, opportunity: ReplayOpportunity) -> PreparedDecision:
        """Bind a timed request to its actual eligible view without a host call.

        Parameters
        ----------
        opportunity : ReplayOpportunity
            Timed core request, nullable producing semantics and exact target.

        Returns
        -------
        PreparedDecision
            Only the eligible snapshot and expected metadata/host bindings.

        Raises
        ------
        ValueError
            Conflicting retry, wrong round/cutoff/input digest, unsupported
            capability, semantic transition or undeclared changing universe.
        ReplayBusyError, ReplayInvalidatedError
            This owner is active or unusable.

        Notes
        -----
        Exact accepted retries retain their original view even after progress.
        New preparation becomes stale if session inventory subsequently changes.
        """
        with self._operation():
            checked = ReplayOpportunity.model_validate(opportunity)
            retained = self._retained(checked)
            snapshot = retained.snapshot if retained is not None else _opportunity(self._state, checked)
            return PreparedDecision(
                schema_version=1,
                opportunity=checked,
                snapshot=snapshot,
                state_fingerprint=self._state_fingerprint(),
                host_state_fingerprint=self._state.host_fingerprint,
            )

    def _synchronous(self, applications: tuple[ReplayApplication, ...]) -> None:
        """Refuse another synchronous opportunity while an outcome is pending."""
        if self._state.policy.delivery != "synchronous":
            return
        completed = {
            (row.application.feedback.lineage.model_dump_json(), row.application.executed.fingerprint)
            for row in self._state.updates
        }
        completed.update((row.feedback.lineage.model_dump_json(), row.executed.fingerprint) for row in applications)
        if any(
            (row.lineage.model_dump_json(), row.executed.fingerprint) not in completed for row in self._state.decisions
        ):
            raise ValueError("Synchronous replay cannot emit another opportunity while feedback remains pending")

    def emit(self, prepared: PreparedDecision, *, execution: Literal["offline_identity_v1"]) -> ReplayDecisionRecord:
        """Drain eligible feedback, choose once and retain simulated execution.

        Parameters
        ----------
        prepared : PreparedDecision
            Actual eligible view and expected state from prepare.
        execution : {'offline_identity_v1'}
            Required explicit offline unchanged-action simulation profile.

        Returns
        -------
        ReplayDecisionRecord
            Retained real proposal and explicitly simulated actual action. Exact
            accepted retries return the original record without a host call.

        Raises
        ------
        ValueError
            Bad/stale preparation, unsupported execution, clocks, capacity,
            synchronous delay, incompatible proposal or altered request.
        ReplayBusyError, ReplayInvalidatedError
            Concurrent/reentrant use, prior failure or externally changed host.
        Exception
            Real host failures propagate and invalidate this session.

        Notes
        -----
        All drain/request preflight precedes mutation. The host receives only
        DecisionRequest; no complete source inventory or outcome oracle crosses
        that boundary. This method performs no physical CI action or trade.
        """
        with self._operation():
            checked = PreparedDecision.model_validate(prepared)
            if execution != "offline_identity_v1":
                raise ValueError("Replay requires the explicit offline_identity_v1 execution profile")
            retained = self._retained(checked.opportunity)
            if retained is not None:
                return retained
            if (
                checked.state_fingerprint != self._state_fingerprint()
                or checked.host_state_fingerprint != self._state.host_fingerprint
            ):
                raise ValueError("Prepared replay decision is stale; prepare again from the current owner")
            snapshot = _opportunity(self._state, checked.opportunity)
            if snapshot.to_json() != checked.snapshot.to_json():
                raise ValueError("Prepared replay view differs from the actual current eligible snapshot")
            timing = checked.opportunity.request.timing
            if timing is None:
                raise ValueError("Causal replay requires explicit decision timing")
            applications = _applications(self._state, timing.information_cutoff, timing.decision_at)
            _capacity(self._state, len(applications) + 2)
            self._synchronous(applications)
            self._verify_host()
            self._drain(applications, timing.information_cutoff, timing.decision_at)
            with self._host_boundary():
                before = self._state.host_fingerprint
                proposal = Decision.model_validate(self._host.decide(checked.opportunity.request))
                if proposal.request.to_json() != checked.opportunity.request.to_json():
                    raise ValueError("Replay host changed the exact producing request")
                validate_decision(proposal, self._state.policy.capability)
                executed = ExecutedDecision(
                    schema_version=1,
                    proposal=proposal,
                    action=proposal.action,
                    mapping="unchanged",
                    sampling_disposition="preserved",
                    sampling=proposal.sampling,
                )
                after = self._snapshot()
                record = ReplayDecisionRecord(
                    schema_version=1,
                    lineage=self._state.policy.lineage,
                    opportunity=checked.opportunity,
                    executed=executed,
                    snapshot=snapshot,
                    decision_index=self._state.decision_count,
                    update_count_before=self._state.update_count,
                    accepted_at=timing.decision_at,
                    host_before=before,
                    host_after=after.fingerprint,
                    sequence=self._state.next_sequence,
                    execution=execution,
                )
                self._state = self._validate_state(
                    self._state.model_copy(
                        update={"decisions": (*self._state.decisions, record), "host_fingerprint": after.fingerprint}
                    )
                )
                return record

    def admit_feedback(self, values: tuple[ReplayFeedback, ...]) -> ReplayState:
        """Admit a complete bounded revision cohort without updating the host.

        Parameters
        ----------
        values : tuple of ReplayFeedback
            At most 4096 attempts, including exact retries. All parents needed
            for the combined graph must be present in retained/incoming data.

        Returns
        -------
        ReplayState
            Canonical full inventory. Exact duplicates are no-ops, including
            retries after acceptance; later corrections never reapply a round.

        Raises
        ------
        ValueError, TypeError
            Invalid graph, conflicting duplicate, wrong execution/target/member
            version, budget violation or late source-visible backfill.
        ReplayBusyError, ReplayInvalidatedError
            This owner is active or unusable.

        Notes
        -----
        A just-registered producer after the last drain may supply its feedback
        in a later causal phase. Other closed visible cohorts cannot be rewritten.
        Unknown receipt/null/partial facts remain pending, without zero filling.
        """
        with self._operation():
            if not isinstance(values, tuple) or len(values) > MAX_MEMBERS:
                raise ValueError("Replay admission requires at most 4096 typed feedback attempts")
            checked = tuple(ReplayFeedback.model_validate(value) for value in values)
            old = {(value.logical_key, value.batch.revision_id): value for value in self._state.feedback}
            unique = dict(old)
            for value in checked:
                key = value.logical_key, value.batch.revision_id
                previous = unique.setdefault(key, value)
                if previous.to_json() != value.to_json():
                    raise ValueError("Conflicting feedback content or first receipt at one revision identity")
            combined = validate_feedback_graph(
                tuple(unique.values()), self._state.decisions, self._state.policy.capability
            )
            for value in checked:
                key = value.logical_key, value.batch.revision_id
                if key in old:
                    continue  # Full canonical conflict checking already occurred.
                self._admission_frontier(value)
            candidate = self._validate_state(self._state.model_copy(update={"feedback": combined}))
            if not self._same_state(candidate):
                self._state = candidate
            return self._state

    def _admission_frontier(self, feedback: ReplayFeedback) -> None:
        """Distinguish a closed-cohort backfill from a later producer phase."""
        _feedback_frontier(self._state, feedback)

    def _same_drain(self, cutoff: Timestamp, processing: Timestamp) -> bool:
        """Recognize an exact closed-drain retry without another host call."""
        last = self._state._last_drain()
        if last is None:
            return False
        return (
            last.eligible_through == cutoff
            and last.apply_at == processing
            and last.decision_count == self._state.decision_count
            and last.update_count_after == self._state.update_count
        )

    def _drain(
        self, applications: tuple[ReplayApplication, ...], cutoff: Timestamp, processing: Timestamp
    ) -> tuple[ReplayUpdateRecord, ...]:
        """Run one prevalidated drain under the owner's existing operation guard."""
        if self._same_drain(cutoff, processing):
            return ()
        original = self._state
        records = []
        before = original.host_fingerprint
        with self._host_boundary():
            for index, application in enumerate(applications):
                self._host.update(application)
                snapshot = self._snapshot()
                record = ReplayUpdateRecord(
                    schema_version=1,
                    lineage=original.policy.lineage,
                    application=application,
                    host_before=before,
                    host_after=snapshot.fingerprint,
                    sequence=original.next_sequence + index,
                )
                records.append(record)
                before = snapshot.fingerprint
            drain = ReplayDrain(
                schema_version=1,
                lineage=original.policy.lineage,
                eligible_through=cutoff,
                apply_at=processing,
                decision_count=original.decision_count,
                update_count_before=original.update_count,
                update_count_after=original.update_count + len(records),
                host_before=original.host_fingerprint,
                host_after=before,
                sequence=original.next_sequence + len(records),
            )
            self._state = self._validate_state(
                original.model_copy(
                    update={
                        "updates": (*original.updates, *records),
                        "drains": (*original.drains, drain),
                        "eligible_through": cutoff,
                        "apply_at": processing,
                        "host_fingerprint": before,
                    }
                )
            )
        return tuple(records)

    def advance(self, *, eligible_through: Timestamp, apply_at: Timestamp) -> tuple[ReplayUpdateRecord, ...]:
        """Apply a closed cohort using independently supplied eligibility/time.

        Parameters
        ----------
        eligible_through : datetime
            Inclusive knowledge cutoff. Both this and apply_at are independently
            nondecreasing; eligibility must not exceed processing time.
        apply_at : datetime
            Actual supplied processing instant, retained in every application.

        Returns
        -------
        tuple of ReplayUpdateRecord
            Successful new completed rounds, ordered by eligibility then stable
            producing decision/feedback identities. Exact closed retries are empty.

        Raises
        ------
        ValueError
            Reversed clocks, incorrect logical delay, unsupported feedback,
            capacity or observed-unit/coverage errors; preflight is side-effect-free.
        ReplayBusyError, ReplayInvalidatedError
            This owner is active, failed or paired with a changed host.
        Exception
            Actual host failures propagate and invalidate the session.

        Notes
        -----
        Select latest source-visible revisions before checking maturity/nulls.
        An unready successor blocks its older ready ancestor. No clock is read,
        no application is backdated, and later corrections remain audit-only.
        """
        with self._operation():
            cutoff, processing = _frontiers(self._state, eligible_through, apply_at)
            if self._same_drain(cutoff, processing):
                return ()
            applications = _applications(self._state, cutoff, processing)
            self._verify_host()
            return self._drain(applications, cutoff, processing)

    def export_continuation(self) -> ReplayContinuation[S]:
        """Bind current metadata and complete actual host state in one envelope.

        Returns
        -------
        ReplayContinuation[S]
            Revalidated pure-data snapshot. Its caller may publish it using an
            existing immutable writer; publication is not performed here.

        Raises
        ------
        ValueError, TypeError, ReplayInvalidatedError
            Bad snapshot or stale host/metadata pairing.
        ReplayBusyError
            Another operation owns this session.

        Notes
        -----
        This qualifies an explicit host-owned composition, not arbitrary state
        completeness, durable acknowledgement or exactly-once physical effects.
        """
        with self._operation():
            snapshot = self._verify_host()
            return self._continuation(snapshot)

    @classmethod
    def from_continuation(
        cls,
        host: ReplayConsumer[S],
        continuation: ReplayContinuation[S],
        *,
        state_type: type[S],
        state_codec: Reference,
    ) -> Self:
        """Restore a matched complete snapshot into a fresh independently owned host.

        Parameters
        ----------
        host : ReplayConsumer[S]
            Fresh compatible complete owner; never a shared live host.
        continuation : ReplayContinuation[S]
            Exact metadata and concrete typed snapshot, validated before restore.
        state_type : type[S]
            Trusted concrete snapshot type, including its strict codec version.
        state_codec : Reference
            Must exactly match the retained host-state codec identity.

        Returns
        -------
        Self
            New process-local owner at the retained frontier.

        Raises
        ------
        ValueError, TypeError
            Incompatible snapshot/codec or host output after restore.
        Exception
            Actual restore/export failures propagate; that host is invalidated.

        Notes
        -----
        No file, pickle, dynamic import, migration or acknowledged frontier is
        loaded. The caller selects this snapshot; durable stale-checkpoint and
        external-effect reconciliation require a separate outer protocol.
        """
        checked = _continuation_type(state_type).model_validate(continuation)
        if checked.state.state_codec != Reference.model_validate(state_codec):
            raise ValueError("Replay continuation host-state codec differs from the selected codec")
        if type(checked.host_state) is not state_type:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Replay continuation requires its exact concrete snapshot type")
        snapshot = state_type.model_validate(checked.host_state)
        state = ReplayState.model_validate(checked.state)
        result = cls.__new__(cls)
        result._configure(host, state_type)
        result._state = state
        with result._host_boundary():
            host.restore_state(snapshot)
            result._verify_host()
        return result

    @classmethod
    def fork_from_continuation(
        cls,
        host: ReplayConsumer[S],
        continuation: ReplayContinuation[S],
        *,
        lineage: Reference,
        state_type: type[S],
        state_codec: Reference,
    ) -> Self:
        """Create an explicit child from a matched prefix without rewriting history.

        Parameters
        ----------
        host : ReplayConsumer[S]
            Fresh independent compatible owner. No parent live host is changed.
        continuation : ReplayContinuation[S]
            Complete retained snapshot selected before the affected transition.
        lineage : Reference
            New child identity/version, distinct from every retained lineage.
        state_type : type[S]
            Explicit complete concrete snapshot type.
        state_codec : Reference
            Exact compatible host snapshot generation.

        Returns
        -------
        Self
            Child retaining every parent causal record unchanged. Future drains
            and decisions belong to the child; original source clocks still apply.

        Raises
        ------
        ValueError, TypeError
            Reused lineage, incompatible snapshot/codec or invalid parent prefix.
        Exception
            Actual fresh-host restore errors propagate.

        Notes
        -----
        This is a reconstructed offline branch, never proof that a correction
        was knowable earlier or rollback of real actions. The caller retains the
        original lineage and may not silently backdate later source receipts.
        """
        checked = _continuation_type(state_type).model_validate(continuation)
        child = Reference.model_validate(lineage)
        state = checked.state
        parent_lineages = () if state.policy.parent is None else state.policy.parent.ancestors
        old_lineages = (
            state.policy.lineage,
            *parent_lineages,
            *(record.lineage for record in state.decisions),
            *(record.lineage for record in state.updates),
            *(record.lineage for record in state.drains),
        )
        if child in old_lineages:
            raise ValueError("A replay child must use a genuinely new lineage identity")
        parent = ReplayParent(
            lineage=state.policy.lineage,
            ancestors=(*parent_lineages, state.policy.lineage),
            state_fingerprint=state.fingerprint,
            prefix_fingerprint=state.causal_prefix_fingerprint,
            next_sequence=state.next_sequence,
            host_fingerprint=state.host_fingerprint,
        )
        policy = ReplayPolicy.model_validate(state.policy.model_copy(update={"lineage": child, "parent": parent}))
        replacement = ReplayState.model_validate(state.model_copy(update={"policy": policy}))
        envelope = _continuation_type(state_type)(schema_version=1, state=replacement, host_state=checked.host_state)
        return cls.from_continuation(host, envelope, state_type=state_type, state_codec=state_codec)
