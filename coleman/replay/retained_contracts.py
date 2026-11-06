"""Explicit generation-two replay retention; complete v1 traces stay unchanged."""

from __future__ import annotations

# Private seeded hooks are shared only within the replay implementation family.
# pylint: disable=protected-access
from typing import Annotated, Literal, Self, cast

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint, Identifier, fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.observations.contracts import ObservationHistory
from coleman.replay.contracts import (
    Count,
    ReplayContinuation,
    ReplayDrain,
    ReplayOpportunity,
    ReplayState,
    _ReplaySeed,
)
from coleman.replay.revisions import _validate_complete_state, validate_opportunity

MAXIMUM_REBINDS = 64


def ordinal_decision_id(epoch: Reference, ordinal: int) -> Identifier:
    """Derive the declared source-owned identity of one logical opportunity.

    Parameters
    ----------
    epoch : Reference
        Stable source epoch; never minted again on retry or codec migration.
    ordinal : int
        Nonnegative global decision ordinal, rejecting bool and coercion.

    Returns
    -------
    str
        Full deterministic SHA-256 identifier under ordinal-decision-v1.

    Raises
    ------
    ValueError
        Invalid epoch or ordinal. Performs no I/O or host/RNG call.
    """
    checked = Reference.model_validate(epoch)
    if type(ordinal) is not int or ordinal < 0:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("A replay ordinal requires a nonnegative integer")
    return fingerprint({"profile": "ordinal-decision-v1", "epoch": checked.model_dump(mode="json"), "ordinal": ordinal})


class OrdinalRetention(CoreImmutable):
    """Opt into bounded quiescent retention without resetting logical time.

    Attributes
    ----------
    profile : {'quiescent-ordinal-v1'}
        Explicit restrictive identity and sealing profile.
    source_epoch : Reference
        Source-owned identity fixed across restart and codec migration.
    maximum_tail_records : int
        Joint retained decisions/feedback/updates/drains limit, 1 through4096.

    Notes
    -----
    No eviction occurs automatically. Exact retries are recognized only while
    records remain in the tail. A seal retires all completed producer ordinals.
    """

    profile: Literal["quiescent-ordinal-v1"]
    source_epoch: Reference
    maximum_tail_records: Annotated[int, Field(strict=True, ge=1, le=4096)]


class ReplayStateRebind(CoreImmutable):
    """Bind a reviewed host codec change at an unchanged sealed boundary.

    Attributes
    ----------
    profile : {'replay-codec-rebind-v1'}
        Explicit migration generation; not a field-name compatibility guess.
    migration : DefinitionBinding
        Exact reviewed data-mapping definition and digest.
    source_state_fingerprint : str
        Complete source replay-state digest checked by creator/prior ack owner.
    boundary_fingerprint : str
        Unchanged policy, epoch and immutable sealed boundary commitment.
    source_codec, target_codec : Reference
        Exact old/new host snapshot interpretations.
    source_host_fingerprint, target_host_fingerprint : str
        Complete host digests joined by this qualified mapping.

    Notes
    -----
    A standalone digest does not authenticate discarded source history.
    """

    profile: Literal["replay-codec-rebind-v1"]
    migration: DefinitionBinding
    source_state_fingerprint: Fingerprint
    boundary_fingerprint: Fingerprint
    source_codec: Reference
    target_codec: Reference
    source_host_fingerprint: Fingerprint
    target_host_fingerprint: Fingerprint


class ReplaySeal(CoreImmutable):
    """Retain one quiescent prefix boundary and its complete source anchor.

    Attributes
    ----------
    profile : {'sealed-replay-prefix-v1'}
        Explicit checkpoint-boundary profile, separate from v1 trace digests.
    source_state_fingerprint, causal_commitment : str
        Full actual source state and ordered profile2 causal commitment.
    decision_count, update_count, next_sequence : int
        Global counters; decision_count is the EXCLUSIVE retired ordinal floor.
    host_fingerprint : str
        Original complete host digest, unchanged by subsequent codec receipts.
    state_codec : Reference
        Original host snapshot interpretation at this seal.
    first_opportunity : ReplayOpportunity
        Complete stable producing semantics retained after the tail is removed.
    last_drain : ReplayDrain
        Literal final actual closed phase, including its original host digests.
    ancestors : tuple of Reference
        Ordered unique root-to-active lineage inventory, at most4096.
    sealed_history : ObservationHistory
        Complete bounded source anchor. No source parent is silently discarded.

    Raises
    ------
    ValueError
        Nonquiescent counters, incomplete phase or inconsistent source/lineages.
    """

    profile: Literal["sealed-replay-prefix-v1"]
    source_state_fingerprint: Fingerprint
    causal_commitment: Fingerprint
    decision_count: Count
    update_count: Count
    next_sequence: Count
    host_fingerprint: Fingerprint
    state_codec: Reference
    first_opportunity: ReplayOpportunity
    last_drain: ReplayDrain
    ancestors: Annotated[tuple[Reference, ...], Field(min_length=1, max_length=4096)]
    sealed_history: ObservationHistory

    @model_validator(mode="after")
    def _boundary(self) -> Self:
        """Require a real final quiescent phase, never a fabricated empty origin."""
        drain = self.last_drain
        if self.decision_count == 0 or self.decision_count != self.update_count:
            raise ValueError("A replay seal requires completed nonempty quiescence")
        if (drain.decision_count, drain.update_count_after) != (self.decision_count, self.update_count):
            raise ValueError("Seal counters differ from its final closed drain")
        if drain.sequence + 1 != self.next_sequence or drain.host_after != self.host_fingerprint:
            raise ValueError("Seal final sequence or host differs from its actual drain")
        if len(set(self.ancestors)) != len(self.ancestors) or self.ancestors[-1] != drain.lineage:
            raise ValueError("Seal ancestry must be unique and end at its active drain lineage")
        return self


def _fold_prefix(state: ReplayState, initial: str) -> str:
    """Fold actual records individually, making seal chunking irrelevant."""
    result = initial
    records = sorted((*state.decisions, *state.updates, *state.drains), key=lambda item: item.sequence)
    for record in records:
        result = fingerprint(
            {"profile": "replay-prefix-v2", "previous": result, "record": record.model_dump(mode="json")}
        )
    return result


def rebind_boundary(state: RetainedReplayState) -> str:
    """Identify immutable migration semantics without future inventory leakage.

    Parameters
    ----------
    state : RetainedReplayState
        Known validated retained metadata. The complete source-state digest is
        checked separately by the creator or the durable prior-ack owner.

    Returns
    -------
    str
        Deterministic policy/profile/seal digest, including its source anchor.

    Raises
    ------
    ValueError, TypeError
        Invalid nested content cannot be serialized canonically.

    Notes
    -----
    Later source inventory and causal-tail additions do not rewrite an older
    receipt. This is consistency evidence, not source authentication or I/O.
    """
    return fingerprint(
        {
            "profile": "replay-rebind-boundary-v1",
            "policy": state.policy.model_dump(mode="json"),
            "retention": state.retention.model_dump(mode="json"),
            "seal": None if state.seal is None else state.seal.model_dump(mode="json"),
        }
    )


class RetainedReplayState(ReplayState):
    """Read an explicit sealed prefix plus a bounded live replay suffix.

    Attributes
    ----------
    schema_version : int
        Required strict2; v1 readers reject this distinct generation.
    retention : OrdinalRetention
        Fixed source identity and bounded tail policy.
    seal : ReplaySeal or None
        Explicit origin when null; otherwise a quiescent retired prefix.
    rebinds : tuple of ReplayStateRebind
        At most64 chained host-codec receipts at this immutable seal.

    Notes
    -----
    Inherited fields keep their names and meanings. Global D/U/sequence are
    prefix counts plus actual suffix records. No restore, migration or I/O runs.
    """

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    retention: OrdinalRetention
    seal: ReplaySeal | None
    rebinds: Annotated[tuple[ReplayStateRebind, ...], Field(max_length=MAXIMUM_REBINDS)] = ()

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Replace zero-origin reconstruction with the shared explicit seed."""
        _retained_boundary(self)
        for record in self.decisions:
            _ordinal(self, record.opportunity)
        _validate_complete_state(self, indexed=True)
        return self

    def _seed(self) -> _ReplaySeed:
        """Supply the authorized codec target without rewriting an old drain."""
        seal = self.seal
        if seal is None:
            return _ReplaySeed()
        host = self.rebinds[-1].target_host_fingerprint if self.rebinds else seal.host_fingerprint
        return _ReplaySeed(
            seal.decision_count, seal.update_count, seal.next_sequence, host, seal.last_drain, seal.first_opportunity
        )

    @property
    def _record_limit(self) -> int:
        """Apply the explicitly selected joint retained-tail ceiling."""
        return self.retention.maximum_tail_records

    @property
    def decision_count(self) -> int:
        """Return lifetime successful decisions without a4096 lifetime ceiling.

        Returns
        -------
        int
            Exclusive next source-owned decision ordinal.
        """
        return self._seed().decision_count + len(self.decisions)

    @property
    def update_count(self) -> int:
        """Return lifetime completed rounds, preserving unequal D/U.

        Returns
        -------
        int
            Quiescent prefix updates plus actual successful suffix updates.
        """
        return self._seed().update_count + len(self.updates)

    @property
    def next_sequence(self) -> int:
        """Return the next lifetime causal position.

        Returns
        -------
        int
            Explicit prefix position plus actual contiguous suffix records.
        """
        return self._seed().next_sequence + len(self.decisions) + len(self.updates) + len(self.drains)

    @property
    def causal_prefix_fingerprint(self) -> str:
        """Return the profile2 record fold, independent of seal chunking.

        Returns
        -------
        str
            Named commitment excluding future source/feedback inventory.
        """
        initial = (
            self.seal.causal_commitment
            if self.seal is not None
            else fingerprint({"profile": "replay-prefix-v2", "stream": self.policy.stream.model_dump(mode="json")})
        )
        return _fold_prefix(self, initial)

    def _validate_ordinal(self, opportunity: ReplayOpportunity) -> None:
        """Refuse expired ordinals before effects; the exclusive floor is valid."""
        _ordinal(self, opportunity)
        if opportunity.request.round != self.decision_count:
            raise ValueError("Retained replay request.round must equal global decision_count")


class RetainedContinuation[S: CoreContract](ReplayContinuation[S]):
    """Pair generation-two replay with an explicit complete concrete host state.

    Attributes
    ----------
    schema_version : int
        Required strict2; no implicit conversion from a v1 continuation.
    state : RetainedReplayState
        Validated sealed prefix, bounded tail and exact expected host digest.
    host_state : S
        Actual complete concrete caller-owned snapshot; inherited checks apply.

    Notes
    -----
    Readers validate data only. Explicit runtime restoration remains caller-owned.
    """

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    state: RetainedReplayState

    @model_validator(mode="before")
    @classmethod
    def _concrete_codec(cls, value: object) -> object:
        """Reject erased generic parameters before any nested field validation."""
        arguments = cls.__pydantic_generic_metadata__["args"]
        if not arguments or not _concrete_state(arguments[0]):
            raise ValueError("Retained continuation requires an explicit concrete CoreContract host type")
        return value

    @model_validator(mode="after")
    def _snapshot(self) -> Self:
        """Reject erased/open generic codecs before inspecting host attributes."""
        arguments = type(self).__pydantic_generic_metadata__["args"]
        if not arguments or not _concrete_state(arguments[0]):
            raise ValueError("Retained continuation requires an explicit concrete CoreContract host type")
        if type(self.host_state) is not arguments[0]:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Retained continuation requires its exact concrete host snapshot type")
        if self.host_state.fingerprint != self.state.host_fingerprint:
            raise ValueError("Continuation host snapshot differs from the expected replay state")
        return self


def _concrete_state(state_type: object) -> bool:
    """Qualify one complete explicit model, including generic concreteness."""
    return (
        isinstance(state_type, type)
        and issubclass(state_type, CoreContract)
        and state_type is not CoreContract
        and not state_type.__pydantic_generic_metadata__.get("parameters")
    )


def _envelope_type[S: CoreContract](state_type: type[S]) -> type[RetainedContinuation[S]]:
    """Select a concrete host-owned model without importing an engine/registry."""
    if not _concrete_state(state_type):
        raise TypeError("Retained replay requires an explicit concrete CoreContract host snapshot type")
    return cast(type[RetainedContinuation[S]], RetainedContinuation.__class_getitem__(state_type))


def _ordinal(state: RetainedReplayState, opportunity: ReplayOpportunity) -> None:
    """Reject retired or mismatched source identity without guessing equality."""
    request = opportunity.request
    if request.round < state._seed().decision_count:
        raise ValueError("Replay producer ordinal expired below the exclusive sealed floor")
    if request.decision_id != ordinal_decision_id(state.retention.source_epoch, request.round):
        raise ValueError("Retained replay decision_id differs from its source epoch and ordinal")


def _retained_boundary(state: RetainedReplayState) -> None:
    """Validate immutable source/policy anchors and the finite codec receipt chain."""
    if state.policy.parent is not None:
        raise ValueError("Retained replay requires its explicit origin profile; matched forks remain v1")
    seal = state.seal
    if seal is None:
        if state.rebinds:
            raise ValueError("Codec rebinds require an explicit quiescent replay seal")
        return
    if seal.ancestors != (state.policy.lineage,):
        raise ValueError("Retained seal ancestry differs from its complete policy lineage")
    if state.history is None:
        raise ValueError("A retained seal requires its complete current source history")
    _sealed_history(state.history, seal)
    _seal_opportunity(state, seal)
    _rebind_chain(state, seal)


def _seal_opportunity(state: RetainedReplayState, seal: ReplaySeal) -> None:
    """Keep the original source-owned first opportunity fully meaningful."""
    first = seal.first_opportunity
    validate_opportunity(state.policy, None, first)
    if first.request.round != 0 or first.request.decision_id != ordinal_decision_id(state.retention.source_epoch, 0):
        raise ValueError("Seal first opportunity differs from its original source ordinal")
    timing = first.request.timing
    if (
        timing is None
        or timing.eligible_input_fingerprint != seal.sealed_history.as_of(timing.information_cutoff).fingerprint
    ):
        raise ValueError("Seal first opportunity differs from its complete source anchor")
    if timing.decision_at > seal.last_drain.apply_at or timing.information_cutoff > seal.last_drain.eligible_through:
        raise ValueError("Seal final drain cannot precede its original first opportunity")


def _rebind_chain(state: RetainedReplayState, seal: ReplaySeal) -> None:
    """Join finite codec receipts without rewriting a historical host digest."""
    source_codec, source_host = seal.state_codec, seal.host_fingerprint
    for receipt in state.rebinds:
        if receipt.boundary_fingerprint != rebind_boundary(state):
            raise ValueError("Codec rebind changes the immutable sealed replay boundary")
        if receipt.source_codec != source_codec or receipt.source_host_fingerprint != source_host:
            raise ValueError("Codec rebind chain is disconnected")
        source_codec, source_host = receipt.target_codec, receipt.target_host_fingerprint
    if state.state_codec != source_codec:
        raise ValueError("Retained state codec differs from its authorized seal/rebind target")


def _sealed_history(history: ObservationHistory, seal: ReplaySeal) -> None:
    """Preserve every anchored row and reject backfill across discarded views."""
    if (history.source_id, history.source_version) != (
        seal.sealed_history.source_id,
        seal.sealed_history.source_version,
    ):
        raise ValueError("Sealed source identity/version differs from the complete current history")
    old = {(row.metadata.observation_id, row.metadata.revision_id): row for row in seal.sealed_history.revisions}
    new = {(row.metadata.observation_id, row.metadata.revision_id): row for row in history.revisions}
    if any(key not in new or new[key].to_json() != row.to_json() for key, row in old.items()):
        raise ValueError("Retained history must preserve every sealed source revision byte-for-byte")
    for key, row in new.items():
        if key not in old and row.effective_metadata.available_at <= seal.last_drain.eligible_through:
            raise ValueError("Source history backfills the sealed replay information frontier")
