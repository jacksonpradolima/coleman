"""Optional ordinal replay with explicit pure compaction and codec rebinding."""

from __future__ import annotations

# Private seeded hooks are shared only within the replay implementation family.
# pylint: disable=protected-access
from typing import Self, cast

from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import canonical_json, fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.replay.codec import DEFAULT_MAXIMUM_BYTES, _bounded
from coleman.replay.contracts import (
    ReplayContinuation,
    ReplayDecisionRecord,
    ReplayOpportunity,
    ReplayPolicy,
    ReplayState,
)
from coleman.replay.engine import CausalReplay, ReplayConsumer
from coleman.replay.retained_contracts import (
    MAXIMUM_REBINDS,
    OrdinalRetention,
    ReplaySeal,
    ReplayStateRebind,
    RetainedContinuation,
    RetainedReplayState,
    _envelope_type,
    _ordinal,
    ordinal_decision_id,
    rebind_boundary,
)

__all__ = ["RetainedReplay", "ordinal_decision_id", "rebind_retained"]


def _envelope[S: CoreContract](state: RetainedReplayState, host: S) -> RetainedContinuation[S]:
    """Validate the joint sixteen-MiB handoff before exposing or assigning it."""
    result = _envelope_type(type(host))(schema_version=2, state=state, host_state=host)
    _bounded(result.to_json(), DEFAULT_MAXIMUM_BYTES)
    return result


class RetainedReplay[S: CoreContract](CausalReplay[S]):
    """Run the shared causal engine under an explicit source-owned ordinal profile.

    Parameters
    ----------
    host : ReplayConsumer[S]
        Independently owned actual decider/updater and complete state owner.
    policy : ReplayPolicy
        Exact source, delivery, capability and correction contract; origin only.
    state_type : type[S]
        Trusted concrete complete host snapshot, including actual RNG and pending.
    state_codec : Reference
        Exact host state interpretation, never imported dynamically.
    retention : OrdinalRetention
        Source epoch and finite retained-tail bound; no automatic eviction.

    Raises
    ------
    ValueError, TypeError
        Invalid origin, profile, snapshot or unsupported model generation.
    Exception
        A real host export error propagates; no host update is performed here.

    Notes
    -----
    Existing prepare/emit/admission/advance semantics and explicit two-clock
    drains are shared with CausalReplay. Global counts never reset at a seal.
    No storage, runtime capture or acknowledgement occurs. The full handoff has
    a joint16MiB bound. Expired producers reject without discarded-byte equality.
    """

    # V1 construction would create the wrong schema; configure the shared owner explicitly.
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        host: ReplayConsumer[S],
        policy: ReplayPolicy,
        *,
        state_type: type[S],
        state_codec: Reference,
        retention: OrdinalRetention,
    ) -> None:
        """Create an explicit ordinal origin without interpreting old freeform IDs."""
        checked = ReplayPolicy.model_validate(policy)
        if checked.parent is not None:
            raise ValueError("Retained replay begins at an explicit independent origin")
        profile = OrdinalRetention.model_validate(retention)
        codec = Reference.model_validate(state_codec)
        self._configure(host, state_type)
        snapshot = self._snapshot()
        self._state = RetainedReplayState(
            schema_version=2,
            policy=checked,
            history=None,
            state_codec=codec,
            host_fingerprint=snapshot.fingerprint,
            retention=profile,
            seal=None,
        )
        _envelope(self._retained_state(), snapshot)

    def _retained_state(self) -> RetainedReplayState:
        """Require the selected generation rather than silently narrowing v1."""
        if not isinstance(self._state, RetainedReplayState):
            raise TypeError("Retained replay requires its explicit generation-two state")
        return self._state

    def _validate_state(self, state: ReplayState) -> RetainedReplayState:
        """Revalidate shared engine rebuilds under the selected retained contract."""
        return RetainedReplayState.model_validate(state)

    def _continuation(self, snapshot: S) -> RetainedContinuation[S]:
        """Use the actual concrete complete snapshot and bounded v2 envelope."""
        return _envelope(self._retained_state(), snapshot)

    def _state_fingerprint(self) -> str:
        """Hash only the actual owned state that already passed full validation.

        No cache is attached to models or copied to model_copy. Constructors,
        restores and every metadata replacement validate the whole state first;
        external readers/to_json/model_validate keep complete reconstruction.
        Metadata is typed immutable data; host state is a separate digest whose
        actual ownership is checked by _verify_host before any mutation.
        """
        return fingerprint(self._retained_state().model_dump(mode="json"))

    def _same_state(self, candidate: ReplayState) -> bool:
        """Compare fully validated owned values without repeating reconstruction."""
        return canonical_json(candidate.model_dump(mode="json")) == canonical_json(
            self._retained_state().model_dump(mode="json")
        )

    def _retained(self, opportunity: ReplayOpportunity) -> ReplayDecisionRecord | None:
        """Reject retired source ordinals before retry lookup or any host call."""
        _ordinal(self._retained_state(), opportunity)
        return super()._retained(opportunity)

    @property
    def state(self) -> RetainedReplayState:
        """Return the immutable bounded suffix and its explicit sealed boundary.

        Returns
        -------
        RetainedReplayState
            Current inventory; obtaining it neither calls a host nor acknowledges.

        Raises
        ------
        ReplayBusyError, ReplayInvalidatedError
            The inherited single-owner guard rejects active or failed sessions.
        """
        return cast(RetainedReplayState, super().state)

    def compact(self) -> RetainedContinuation[S]:
        """Seal a complete quiescent prefix without publishing or resetting it.

        Returns
        -------
        RetainedContinuation[S]
            Full typed host plus bounded empty suffix and immutable prefix seal.
            Empty origins return unchanged and receive no fabricated seal.

        Raises
        ------
        ValueError
            Pending rounds, unclosed phase, no new actual drain for rebind reset,
            missing history or a joint16MiB handoff overflow; old metadata stays.
        ReplayBusyError, ReplayInvalidatedError
            Concurrent use, stale host or failed prior transition.
        Exception
            Actual side-effect-free export failures propagate.

        Notes
        -----
        D/U/sequence, host/RNG, clocks, source epoch and causal commitment stay
        unchanged. Floor becomes completed D exclusively, retiring0 throughD-1.
        A durable owner must acknowledge this seal separately from transitions.
        """
        with self._operation():
            state = self._retained_state()
            snapshot = self._verify_host()
            if state.decision_count != state.update_count:
                raise ValueError("Cannot compact replay while producer rounds remain pending")
            if state.decision_count == 0:
                return _envelope(state, snapshot)
            if state.seal is not None and not state.drains:
                if state.rebinds:
                    raise ValueError("Resetting codec rebinds requires a new actual closed drain")
                return _envelope(state, snapshot)
            seal = _new_seal(state)
            candidate = RetainedReplayState.model_validate(
                state.model_copy(
                    update={"seal": seal, "decisions": (), "updates": (), "feedback": (), "drains": (), "rebinds": ()}
                )
            )
            result = _envelope(candidate, snapshot)
            self._state = candidate
            return result

    def export_continuation(self) -> RetainedContinuation[S]:
        """Export the actual full typed host and bounded retained frontier.

        Returns
        -------
        RetainedContinuation[S]
            Generation-two pure data; no commit or compatibility scan occurs.

        Raises
        ------
        ValueError, TypeError
            Joint16MiB overflow, invalid model or host-state mismatch.
        ReplayBusyError, ReplayInvalidatedError
            The owner is active, failed or externally changed.
        """
        return cast(RetainedContinuation[S], super().export_continuation())

    @classmethod
    def from_continuation(
        cls,
        host: ReplayConsumer[S],
        continuation: ReplayContinuation[S],
        *,
        state_type: type[S],
        state_codec: Reference,
    ) -> Self:
        """Restore a fully validated retained snapshot into a fresh owned host.

        Parameters
        ----------
        host : ReplayConsumer[S]
            Fresh compatible independent owner, never a partially shared host.
        continuation : RetainedContinuation[S]
            Complete matched data, validated and bounded before restore.
        state_type : type[S]
            Exact concrete complete caller-owned snapshot type.
        state_codec : Reference
            Must equal the current authorized seal/rebind target codec.

        Returns
        -------
        Self
            New owner with unchanged global counts and expired-producer fences.

        Raises
        ------
        ValueError, TypeError
            Wrong generation, concrete state, codec, bounds or replay invariants.
        Exception
            Real restore/export failures propagate and invalidate the new owner.

        Notes
        -----
        This does not prove a durable ack frontier or current installed runtime;
        the optional outer store must qualify those before calling this method.
        """
        checked = _envelope_type(state_type).model_validate(continuation)
        _bounded(checked.to_json(), DEFAULT_MAXIMUM_BYTES)
        if checked.state.state_codec != Reference.model_validate(state_codec):
            raise ValueError("Retained continuation codec differs from the selected codec")
        if type(checked.host_state) is not state_type:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Retained continuation requires its exact concrete host snapshot type")
        result = cls.__new__(cls)
        result._configure(host, state_type)
        result._state = checked.state
        with result._host_boundary():
            host.restore_state(checked.host_state)
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
        """Refuse retained forks rather than inventing an omitted v1 parent trace.

        Parameters
        ----------
        host : ReplayConsumer[S]
            Uncalled prospective owner.
        continuation : ReplayContinuation[S]
            Supplied checkpoint; no restore is attempted.
        lineage : Reference
            Proposed child identity, not allocated here.
        state_type : type[S]
            Prospective complete snapshot type.
        state_codec : Reference
            Prospective host interpretation.

        Raises
        ------
        ValueError
            Always. Existing CausalReplay v1 full matched forks remain available;
            no post-seal parent reconstruction is claimed by this profile.

        Notes
        -----
        Performs no host, RNG, storage or runtime call.
        """
        raise ValueError("Retained replay forks are unsupported; use an explicit complete v1 matched fork")


def _new_seal(state: RetainedReplayState) -> ReplaySeal:
    """Validate actual closed quiescence before removing any live record."""
    drain, first = state._last_drain(), state._first_opportunity()
    if state.decision_count != state.update_count:
        raise ValueError("Cannot compact replay while producer rounds remain pending")
    if drain is None or drain.sequence + 1 != state.next_sequence:
        raise ValueError("Replay compaction requires its final actual closed drain")
    if state.history is None or first is None:
        raise ValueError("Replay compaction requires complete source and producing semantics")
    ancestors = () if state.policy.parent is None else state.policy.parent.ancestors
    return ReplaySeal(
        profile="sealed-replay-prefix-v1",
        source_state_fingerprint=state.fingerprint,
        causal_commitment=state.causal_prefix_fingerprint,
        decision_count=state.decision_count,
        update_count=state.update_count,
        next_sequence=state.next_sequence,
        host_fingerprint=state.host_fingerprint,
        state_codec=state.state_codec,
        first_opportunity=first,
        last_drain=drain,
        ancestors=(*ancestors, state.policy.lineage),
        sealed_history=state.history,
    )


def rebind_retained[S: CoreContract, T: CoreContract](
    source: RetainedContinuation[S], target_host_state: T, *, target_codec: Reference, migration: DefinitionBinding
) -> RetainedContinuation[T]:
    """Record a known pure host-codec mapping at an empty quiescent seal.

    Parameters
    ----------
    source : RetainedContinuation[S]
        Complete actual old frontier, not merely a declared old digest.
    target_host_state : T
        Complete concrete target data produced by a separately reviewed mapping.
    target_codec : Reference
        Exact target state interpretation.
    migration : DefinitionBinding
        Known mapping definition and content fingerprint.

    Returns
    -------
    RetainedContinuation[T]
        Revalidated target data and bounded linked receipt; historical drains stay.

    Raises
    ------
    ValueError, TypeError
        Nonquiescent/unsealed/live-tail input, wrong model, full64-link chain,
        invalid target or joint16MiB overflow. Source values are immutable.

    Notes
    -----
    No host restore, runtime scan or I/O occurs. This checks content bindings,
    not arbitrary semantic equivalence of statistics, RNG, cache or pending data.
    The durable owner additionally compares the source against its actual ack.
    """
    if not isinstance(source, RetainedContinuation):
        raise TypeError("Replay codec migration requires a concrete retained continuation")
    checked = _envelope_type(type(source.host_state)).model_validate(source)
    _bounded(checked.to_json(), DEFAULT_MAXIMUM_BYTES)
    state = checked.state
    if state.seal is None or state.decision_count != state.update_count:
        raise ValueError("Replay codec migration requires a quiescent generation-two seal")
    if state.decisions or state.updates or state.drains or state.feedback:
        raise ValueError("Replay codec migration requires an empty sealed tail")
    if len(state.rebinds) >= MAXIMUM_REBINDS:
        raise ValueError("Replay codec migration exceeds the64-link receipt bound")
    _envelope_type(type(target_host_state))
    target = type(target_host_state).model_validate(target_host_state)
    codec = Reference.model_validate(target_codec)
    receipt = ReplayStateRebind(
        profile="replay-codec-rebind-v1",
        migration=DefinitionBinding.model_validate(migration),
        source_state_fingerprint=state.fingerprint,
        boundary_fingerprint=rebind_boundary(state),
        source_codec=state.state_codec,
        target_codec=codec,
        source_host_fingerprint=state.host_fingerprint,
        target_host_fingerprint=target.fingerprint,
    )
    result = RetainedReplayState.model_validate(
        state.model_copy(
            update={"state_codec": codec, "host_fingerprint": target.fingerprint, "rebinds": (*state.rebinds, receipt)}
        )
    )
    return _envelope(result, target)
