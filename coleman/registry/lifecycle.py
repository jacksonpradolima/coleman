"""Prospective bounded lifecycle history and immutable predecision eligibility."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint, Identifier, fingerprint
from coleman.core.temporal import Timestamp
from coleman.objectives.bindings import DefinitionBinding
from coleman.registry._transitions import _fold, _validate_history
from coleman.registry.definitions import Catalogue, MemberDefinition, MemberKind
from coleman.registry.limits import RegistryContract, RegistryLimits, admit_bytes

Round = Annotated[int, Field(strict=True, ge=0)]
Operation = Literal["introduce", "inactive", "unavailable", "resume", "retire", "reenter"]
Status = Literal["not_introduced", "active", "inactive", "unavailable", "retired"]
Reentry = Literal["retain", "reset", "restart"]


class PopulationPolicy(CoreImmutable):
    """Freeze one task/population namespace and causal clock profile.

    Attributes
    ----------
    population, task : Reference
        Stable population and task identities, independent of global catalogue.
    kind : {'expert', 'action'}
        Selected member namespace.
    clock : {'timed', 'untimed'}
        Timed always requires a supplied cutoff; untimed always requires null.
    profile : {'registry-round-v1'}, optional
        Conservative schedule: nondecreasing activation round and known time
        for actionable facts. Unknown timed knowledge remains audit-only.

    Notes
    -----
    This describes eligibility, not dynamic-learner/comparator qualification.
    Empty opportunities count only when the caller seals an explicit request;
    they never create an adaptive update or a physical-cost observation.
    No ambient clocks, population mutation, source checks or I/O occur.
    """

    population: Reference
    task: Reference
    kind: MemberKind
    clock: Literal["timed", "untimed"]
    profile: Literal["registry-round-v1"] = "registry-round-v1"


class LifecycleFact(RegistryContract):
    """Supply one ordered prospective transition or timed audit-only fact.

    Attributes
    ----------
    event_id : str
        Stable local retry identity; conflicting content rejects.
    sequence : int
        Contiguous acceptance sequence starting at0, independent of wall time.
    member : Reference
        Exact registered member version; replacement requires explicit restart.
    definition_fingerprint : str
        Exact immutable ``MemberDefinition`` content digest.
    operation : str
        Introduce, inactive, unavailable, resume, retire or reenter.
    effective_from_round, admitted_before_round : int
        Transition applies no earlier than both supplied round boundaries.
    known_at : datetime or None
        Required explicit zoned knowledge instant; unknown timed facts never
        become live. New knowledge requires a new prospectively admitted fact.
    reason : str
        Bounded by complete encoded bytes; retained supplied explanation.
    state_epoch, predecessor_epoch : Reference or None
        Introduction/reentry name an epoch. Reentry also names its exact prior
        epoch. Ordinary transitions retain it and require both fields null.
    disposition : {'retain', 'reset', 'restart'} or None
        Required only for reentry; reset retains the exact member definition,
        restart may change its version. Neither imports live historical losses.

    Notes
    -----
    Requires explicit schema1. Facts contain no performance values. Predecision
    unavailability differs from a producing active outcome later being missing.
    Pure construction performs no transition, delivery, migration or I/O.
    """

    event_id: Identifier
    sequence: Round
    member: Reference
    definition_fingerprint: Fingerprint
    operation: Operation
    effective_from_round: Round
    admitted_before_round: Round
    known_at: Timestamp | None
    reason: Identifier
    state_epoch: Reference | None
    predecessor_epoch: Reference | None
    disposition: Reentry | None

    @model_validator(mode="after")
    def _epoch_fields(self) -> Self:
        """Require explicit introduction/reentry and forbid ignored epoch data."""
        requirements = {
            "introduce": ((True, False, False), "Introduction requires only an explicit new state_epoch"),
            "reenter": ((True, True, True), "Reentry requires state_epoch, predecessor_epoch and disposition"),
        }
        expected, message = requirements.get(
            self.operation, ((False, False, False), "Ordinary lifecycle transitions retain epochs without epoch fields")
        )
        supplied = tuple(value is not None for value in (self.state_epoch, self.predecessor_epoch, self.disposition))
        if supplied != expected:
            raise ValueError(message)
        return self


class EligibilityMember(CoreImmutable):
    """Retain a member's exact predecision disposition without inventing loss.

    Attributes
    ----------
    member : Reference
        Exact historical member version.
    definition_fingerprint : str
        Exact immutable leaf digest.
    status : str
        Not introduced, active, inactive, unavailable or retired.
    state_epoch : Reference or None
        Selected explicit epoch, null before any eligible introduction.
    last_event_id, reason : str or None
        Last applied causal fact and supplied reason, null when none applied.

    Notes
    -----
    History preserves earlier epochs. This view neither imputes an outcome nor
    chooses whether a delayed-capable owner applies or audits old feedback.
    """

    member: Reference
    definition_fingerprint: Fingerprint
    status: Status
    state_epoch: Reference | None
    last_event_id: Identifier | None
    reason: Identifier | None


class EligibilitySnapshot(RegistryContract):
    """Seal complete supplied causal prefix and independently stable active set.

    Attributes
    ----------
    policy : PopulationPolicy
        Exact fixed population/task/clock profile.
    catalogue : DefinitionBinding
        Complete catalogue publication digest, separate from active identity.
    definitions : tuple of MemberDefinition
        Every exact leaf referenced by the retained fact prefix, no unrelated
        definitions. Historical versions remain distinguishable.
    facts : tuple of LifecycleFact
        Complete accepted prefix, including scheduled and unknown facts.
    round : int
        Explicit predecision logical boundary.
    cutoff : datetime or None
        Required for timed policy, forbidden for untimed policy.
    members : tuple of EligibilityMember
        All referenced versions and exact dispositions, independently recomputed.

    Notes
    -----
    At most4096 leaves/facts and16MiB. Active identity excludes round, cutoff,
    inactive history and catalogue provenance, but includes exact active leaf
    and epoch. Full fingerprint covers all fields and explicit nulls. No action,
    loss, learner update, authenticity or theorem is implied; performs no I/O.
    """

    policy: PopulationPolicy
    catalogue: DefinitionBinding
    definitions: Annotated[tuple[MemberDefinition, ...], Field(max_length=4096)]
    facts: Annotated[tuple[LifecycleFact, ...], Field(max_length=4096)]
    round: Round
    cutoff: Timestamp | None
    members: Annotated[tuple[EligibilityMember, ...], Field(max_length=4096)]

    @model_validator(mode="after")
    def _content(self) -> Self:
        """Recompute exact causal eligibility before accepting supplied views."""
        if (self.policy.clock == "timed") != (self.cutoff is not None):
            raise ValueError("Snapshot cutoff must match the fixed timed/untimed population policy")
        canonical = tuple(sorted(self.definitions, key=lambda item: (item.member.id, item.member.version)))
        references = [(item.member.id, item.member.version) for item in canonical]
        if len(references) != len(set(references)) or set(references) != {
            (fact.member.id, fact.member.version) for fact in self.facts
        }:
            raise ValueError("Snapshot definitions must exactly cover referenced historical versions")
        object.__setattr__(self, "definitions", canonical)
        _validate_history(self.policy, canonical, self.facts)
        expected = _fold(self.policy, canonical, self.facts, self.round, self.cutoff)
        if tuple(sorted(self.members, key=lambda row: (row.member.id, row.member.version))) != expected:
            raise ValueError("Snapshot dispositions differ from its exact causal prefix")
        object.__setattr__(self, "members", expected)
        admit_bytes(self, RegistryLimits(maximum_bytes=16 * 1024 * 1024))
        return self

    @property
    def active(self) -> tuple[Reference, ...]:
        """Return exact active versions in canonical logical member order.

        Returns
        -------
        tuple of Reference
            Empty when all members sleep/retire; no artificial zero coordinate.
        """
        return tuple(row.member for row in self.members if row.status == "active")

    @property
    def active_contract_fingerprint(self) -> str:
        """Identify only the exact selected mathematical population contract.

        Returns
        -------
        str
            Digest of fixed policy, active leaves and their state epochs.

        Notes
        -----
        Inactive facts, unrelated versions, round/cutoff and global publication
        identity are excluded so static replay does not see false changes.
        """
        leaves = {leaf.member: leaf for leaf in self.definitions}
        return fingerprint(
            {
                "schema_version": 1,
                "policy": self.policy.model_dump(mode="json"),
                "active": [
                    {
                        "definition": leaves[row.member].model_dump(mode="json"),
                        "state_epoch": row.state_epoch.model_dump(mode="json") if row.state_epoch else None,
                    }
                    for row in self.members
                    if row.status == "active"
                ],
            }
        )

    @property
    def prefix_fingerprint(self) -> str:
        """Return the complete accepted lifecycle-prefix content identity.

        Returns
        -------
        str
            Digest including unknown and future facts in acceptance order.
        """
        return fingerprint([fact.model_dump(mode="json") for fact in self.facts])

    @property
    def history_fingerprint(self) -> str:
        """Identify the exact fixed policy and this accepted causal fact prefix.

        Returns
        -------
        str
            Complete producing history identity for compact retained seals.

        Notes
        -----
        Includes future/unknown facts without making them eligible; no I/O.
        """
        return _history_fingerprint(self.policy, self.facts)

    def feasible_set(self, logical_id: str) -> Reference:
        """Bind a stable core space ID to the selective active contract.

        Parameters
        ----------
        logical_id : str
            Caller-owned stable finite-space identity.

        Returns
        -------
        Reference
            Existing core reference with explicit active digest as its version.

        Notes
        -----
        Called after snapshot hashing; creates no digest cycle or global state.
        """
        return Reference(id=logical_id, version=self.active_contract_fingerprint)


class OpportunitySeal(CoreImmutable):
    """Retain a compact local request retry binding without copying history.

    Attributes
    ----------
    logical_identity : str
        Existing core task+decision-ID identity, which intentionally omits round.
    round : int
        Explicit producing logical round.
    cutoff : datetime or None
        Producing knowledge cutoff, null exactly for an untimed population.
    request_fingerprint, snapshot_fingerprint, envelope_fingerprint : str
        Exact original core request, full snapshot and registered request joins.
    fact_count : int
        Exact accepted prefix length at production, including future/unknowns.
    history_fingerprint : str
        Digest of fixed population policy and the complete producing fact
        prefix. Restored history cannot silently rewrite previously sealed facts.

    Notes
    -----
    Optional replay/storage owns durable acceptance. This bounded pure record
    alone gives no exactly-once execution/update or concurrent commit guarantee.
    """

    logical_identity: Fingerprint
    round: Round
    cutoff: Timestamp | None
    request_fingerprint: Fingerprint
    snapshot_fingerprint: Fingerprint
    envelope_fingerprint: Fingerprint
    fact_count: Annotated[int, Field(strict=True, ge=0, le=4096)]
    history_fingerprint: Fingerprint


class RegistryState(RegistryContract):
    """Hold one bounded immutable population history and local request seals.

    Attributes
    ----------
    catalogue : Catalogue
        Exact immutable leaves; retained historical versions are never evicted.
    policy : PopulationPolicy
        Fixed population/task/kind/clock policy for this state lineage.
    limits : RegistryLimits
        Coupled selected counts/bytes, checked before successors are returned.
    facts : tuple of LifecycleFact, optional
        Append-only accepted prefix, default empty.
    producers : tuple of OpportunitySeal, optional
        Compact local request seals, canonical by logical opportunity identity.

    Notes
    -----
    No global registry, factories, learner state, execution engine, storage or
    I/O. Caller-owned values remain unchanged on invalid successor/refusal.
    Sealing an empty opportunity advances this history's prospective frontier
    without inventing an update or a physical charge.
    """

    catalogue: Catalogue
    policy: PopulationPolicy
    limits: RegistryLimits
    facts: Annotated[tuple[LifecycleFact, ...], Field(max_length=4096)] = ()
    producers: Annotated[tuple[OpportunitySeal, ...], Field(max_length=8192)] = ()

    @model_validator(mode="after")
    def _valid(self) -> Self:
        """Admit counts/bytes and verify every scheduled transition boundary."""
        if (
            len(self.catalogue.definitions) > self.limits.maximum_definitions
            or len(self.facts) > self.limits.maximum_facts
        ):
            raise ValueError("Registry definition/fact count exceeds selected capacity")
        if len(self.producers) > self.limits.maximum_producers:
            raise ValueError("Registry producer count exceeds selected capacity")
        admit_bytes(self, self.limits)
        identities = [item.logical_identity for item in self.producers]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate or conflicting retained logical opportunity")
        by_round = sorted(self.producers, key=lambda item: item.round)
        _validate_seal_clocks(self.policy, by_round)
        object.__setattr__(self, "producers", tuple(sorted(self.producers, key=lambda item: item.logical_identity)))
        leaves = tuple(item for item in self.catalogue.definitions if item.kind == self.policy.kind)
        _validate_history(self.policy, leaves, self.facts)
        _validate_retained_prefixes(self)
        return self

    @property
    def history_fingerprint(self) -> str:
        """Return the fixed population policy and complete fact-prefix digest.

        Returns
        -------
        str
            Canonical causal-history identity used by compact producing seals.

        Notes
        -----
        Revalidates current state before hashing; no callbacks or I/O occur.
        Retained-prefix validation has an explicit selected finite work ceiling.
        """
        current = RegistryState.model_validate(self)
        return _history_fingerprint(current.policy, current.facts)

    def append(self, fact: LifecycleFact) -> RegistryState:
        """Validate and append a prospective fact, or reconcile its exact retry.

        Parameters
        ----------
        fact : LifecycleFact
            Next contiguous event; after a sealed round r, admission must be at
            least r+1 even if supplied effective/knowledge clocks are older.

        Returns
        -------
        RegistryState
            New valid immutable value, or current value for an exact retry.

        Raises
        ------
        ValueError
            Conflicting retry, late admission, invalid schedule/transition,
            unknown definition/version or coupled count/byte exhaustion.

        Notes
        -----
        All applicable schedule and knowledge boundaries are checked using the
        conservative monotone profile; unsupported schedules explicitly reject.
        Raw retries are local content checks, not durable delivery guarantees.
        """
        current, event = RegistryState.model_validate(self), LifecycleFact.model_validate(fact)
        for previous in current.facts:
            if previous.event_id == event.event_id:
                if previous.fingerprint != event.fingerprint:
                    raise ValueError("Conflicting lifecycle event retry")
                return current
        frontier = max((item.round for item in current.producers), default=-1)
        if event.admitted_before_round <= frontier:
            raise ValueError("New lifecycle facts must be admitted prospectively beyond sealed decisions")
        if len(current.facts) >= current.limits.maximum_facts:
            raise ValueError("Registry lifecycle capacity exhausted")
        return RegistryState.model_validate(current.model_dump() | {"facts": (*current.facts, event)})

    def snapshot(  # pylint: disable=redefined-builtin
        self, round: int, cutoff: Timestamp | None = None
    ) -> EligibilitySnapshot:
        """Compute an exact predecision snapshot without changing registry state.

        Parameters
        ----------
        round : int
            Explicit nonnegative logical decision boundary.
        cutoff : datetime or None, optional
            Required timed knowledge boundary, forbidden for untimed policy.

        Returns
        -------
        EligibilitySnapshot
            Complete supplied causal prefix and all dispositions.

        Raises
        ------
        ValueError
            Wrong clock profile, selected byte capacity or attempt to rewrite a
            sealed historical snapshot with a changed prefix/catalogue.

        Notes
        -----
        No ambient time, learner call, seal, cost or update is produced.
        """
        current = RegistryState.model_validate(self)
        referenced = {(fact.member.id, fact.member.version) for fact in current.facts}
        leaves = tuple(
            item
            for item in current.catalogue.definitions
            if item.kind == current.policy.kind and (item.member.id, item.member.version) in referenced
        )
        # Validate the clock/round before comparing them inside the pure fold.
        from pydantic import TypeAdapter  # pylint: disable=import-outside-toplevel

        selected_round = TypeAdapter(Round).validate_python(round)
        selected_cutoff = TypeAdapter(Timestamp | None).validate_python(cutoff)
        if (current.policy.clock == "timed") != (selected_cutoff is not None):
            raise ValueError("Snapshot cutoff must match the fixed timed/untimed population policy")
        result = EligibilitySnapshot(
            schema_version=1,
            policy=current.policy,
            catalogue=DefinitionBinding(
                reference=current.catalogue.reference, definition_fingerprint=current.catalogue.fingerprint
            ),
            definitions=leaves,
            facts=current.facts,
            round=selected_round,
            cutoff=selected_cutoff,
            members=_fold(current.policy, leaves, current.facts, selected_round, selected_cutoff),
        )
        frontier = max((item.round for item in current.producers), default=-1)
        if selected_round <= frontier and not any(
            item.round == selected_round and item.snapshot_fingerprint == result.fingerprint
            for item in current.producers
        ):
            raise ValueError("Cannot rewrite a sealed producing snapshot")
        admit_bytes(result, current.limits)
        return result


def _validate_seal_clocks(policy: PopulationPolicy, seals: list[OpportunitySeal]) -> None:
    """Check accepted timed cutoff progression without consulting ambient time."""
    previous: OpportunitySeal | None = None
    for seal in seals:
        if (policy.clock == "timed") != (seal.cutoff is not None):
            raise ValueError("Retained request cutoff must match the fixed population clock policy")
        if previous is not None:
            _validate_seal_successor(previous, seal)
        previous = seal


def _validate_seal_successor(previous: OpportunitySeal, seal: OpportunitySeal) -> None:
    """Reject accepted-prefix rewinds and ambiguous boundaries at one round."""
    if seal.fact_count < previous.fact_count:
        raise ValueError("Retained producing fact_count must be nondecreasing across rounds")
    if seal.round == previous.round and (seal.fact_count, seal.cutoff, seal.snapshot_fingerprint) != (
        previous.fact_count,
        previous.cutoff,
        previous.snapshot_fingerprint,
    ):
        raise ValueError("Same-round producing seals must retain the same prefix, cutoff and snapshot")
    if previous.cutoff is not None and seal.cutoff is not None and seal.cutoff < previous.cutoff:
        raise ValueError("Accepted timed request cutoffs must be nondecreasing across rounds")


def _history_fingerprint(policy: PopulationPolicy, facts: tuple[LifecycleFact, ...]) -> str:
    """Hash the existing canonical policy and exact ordered causal fact prefix."""
    return fingerprint(
        {
            "schema_version": 1,
            "policy": policy.model_dump(mode="json"),
            "facts": [fact.model_dump(mode="json") for fact in facts],
        }
    )


def _validate_retained_prefixes(state: RegistryState) -> None:
    """Admit distinct-prefix work before copies/hashes and check late admissions."""
    lengths = {seal.fact_count for seal in state.producers}
    if any(length > len(state.facts) for length in lengths):
        raise ValueError("Retained producer fact_count exceeds available history")
    if sum(lengths) > state.limits.maximum_prefix_work:
        raise ValueError("Registry distinct retained prefix validation exceeds maximum_prefix_work")
    prefixes = {length: _history_fingerprint(state.policy, state.facts[:length]) for length in lengths}
    # A suffix-minimum scan checks every seal's future admissions in O(facts +
    # producers), avoiding repeated producer-by-fact Cartesian scans.
    suffix_minimum = [float("inf")] * (len(state.facts) + 1)
    for index in range(len(state.facts) - 1, -1, -1):
        suffix_minimum[index] = min(suffix_minimum[index + 1], state.facts[index].admitted_before_round)
    for seal in state.producers:
        if prefixes[seal.fact_count] != seal.history_fingerprint:
            raise ValueError("Retained producer history fingerprint differs from its exact causal prefix")
        if suffix_minimum[seal.fact_count] <= seal.round:
            raise ValueError("Facts after a sealed prefix must be admitted beyond the producing round")
