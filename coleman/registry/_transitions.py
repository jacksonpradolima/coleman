"""Private finite causal transition fold for the public lifecycle contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coleman.core.identity import Reference
from coleman.core.temporal import Timestamp
from coleman.registry.definitions import MemberDefinition

if TYPE_CHECKING:
    from coleman.registry.lifecycle import EligibilityMember, LifecycleFact, PopulationPolicy, Status


def _applicable(fact: LifecycleFact, policy: PopulationPolicy, round_number: int, cutoff: Timestamp | None) -> bool:
    """Use both prospective round clocks and the fixed knowledge policy."""
    if max(fact.effective_from_round, fact.admitted_before_round) > round_number:
        return False
    return policy.clock == "untimed" or (fact.known_at is not None and cutoff is not None and fact.known_at <= cutoff)


def _transition(previous: EligibilityMember | None, fact: LifecycleFact) -> EligibilityMember:
    """Check one actionable transition, including exact version/epoch lineage."""
    from coleman.registry.lifecycle import EligibilityMember  # pylint: disable=import-outside-toplevel

    status: Status
    if fact.operation == "introduce":
        if previous is not None:
            raise ValueError("Introduction requires a previously absent logical member")
        epoch, status = fact.state_epoch, "active"
    else:
        if previous is None:
            raise ValueError("Lifecycle transition requires an earlier eligible introduction")
        epoch = previous.state_epoch
        if fact.operation == "reenter":
            _reentry(previous, fact)
            epoch, status = fact.state_epoch, "active"
        else:
            status = _continuing_status(previous, fact)
    return EligibilityMember(
        member=fact.member,
        definition_fingerprint=fact.definition_fingerprint,
        status=status,
        state_epoch=epoch,
        last_event_id=fact.event_id,
        reason=fact.reason,
    )


def _continuing_status(previous: EligibilityMember, fact: LifecycleFact) -> Status:
    """Validate a same-definition lifecycle move and return its disposition."""
    if previous.member != fact.member or previous.definition_fingerprint != fact.definition_fingerprint:
        raise ValueError("Lifecycle transition cannot inherit another definition/version")
    allowed: dict[str, set[str]] = {
        "inactive": {"active"},
        "unavailable": {"active"},
        "resume": {"inactive", "unavailable"},
        "retire": {"active", "inactive", "unavailable"},
    }
    if previous.status not in allowed[fact.operation]:
        raise ValueError("Invalid lifecycle transition from current disposition")
    statuses: dict[str, Status] = {
        "inactive": "inactive",
        "unavailable": "unavailable",
        "resume": "active",
        "retire": "retired",
    }
    return statuses[fact.operation]


def _reentry(previous: EligibilityMember, fact: LifecycleFact) -> None:
    """Require retired origin, exact predecessor and explicit version restart."""
    if previous.status != "retired" or fact.predecessor_epoch != previous.state_epoch:
        raise ValueError("Reentry requires the exact retired predecessor epoch")
    same_definition = previous.member == fact.member and previous.definition_fingerprint == fact.definition_fingerprint
    same_epoch = previous.state_epoch == fact.state_epoch
    if fact.disposition == "retain" and (not same_definition or not same_epoch):
        raise ValueError("Retain reentry requires identical member definition and epoch")
    if fact.disposition == "reset" and (not same_definition or same_epoch):
        raise ValueError("Reset reentry requires the same definition and a new epoch")
    if fact.disposition == "restart" and same_epoch:
        raise ValueError("Restart reentry requires a new epoch")


def _validate_history(
    policy: PopulationPolicy, definitions: tuple[MemberDefinition, ...], facts: tuple[LifecycleFact, ...]
) -> None:
    """Validate a monotone schedule at every actionable boundary in one pass."""
    leaves = {item.member: (item.kind, item.fingerprint) for item in definitions}
    seen: set[str] = set()
    current: dict[str, EligibilityMember] = {}
    previous_round = -1
    previous_known: Timestamp | None = None
    epochs: dict[str, set[Reference]] = {}
    for index, fact in enumerate(facts):
        if fact.sequence != index or fact.event_id in seen:
            raise ValueError("Lifecycle sequence must be contiguous with unique event IDs")
        seen.add(fact.event_id)
        _require_definition(policy, leaves.get(fact.member), fact)
        if policy.clock == "untimed" and fact.known_at is not None:
            raise ValueError("Untimed population requires explicit null knowledge clocks")
        if policy.clock == "timed" and fact.known_at is None:
            continue
        activation = max(fact.effective_from_round, fact.admitted_before_round)
        if activation < previous_round or (
            previous_known is not None and fact.known_at is not None and fact.known_at < previous_known
        ):
            raise ValueError("Lifecycle profile requires nondecreasing activation rounds and known_at boundaries")
        previous_round, previous_known = activation, fact.known_at
        prior = current.get(fact.member.id)
        updated = _transition(prior, fact)
        _remember_epoch(epochs, fact)
        current[fact.member.id] = updated


def _remember_epoch(epochs: dict[str, set[Reference]], fact: LifecycleFact) -> None:
    """Track each logical member's epochs after its transition is validated."""
    used = epochs.setdefault(fact.member.id, set())
    if fact.operation in {"introduce", "reenter"} and fact.disposition != "retain":
        if fact.state_epoch in used:
            raise ValueError("New state epoch cannot reuse a retained historical epoch")
        if fact.state_epoch is not None:
            used.add(fact.state_epoch)


def _require_definition(policy: PopulationPolicy, leaf: tuple[str, str] | None, fact: LifecycleFact) -> None:
    """Validate exact namespace, member/version and immutable implementation."""
    if leaf != (policy.kind, fact.definition_fingerprint):
        raise ValueError("Lifecycle fact requires an exact registered population definition")


def _fold(
    policy: PopulationPolicy,
    definitions: tuple[MemberDefinition, ...],
    facts: tuple[LifecycleFact, ...],
    round_number: int,
    cutoff: Timestamp | None,
) -> tuple[EligibilityMember, ...]:
    """Retain all referenced versions while selecting the causal active epoch."""
    from coleman.registry.lifecycle import EligibilityMember  # pylint: disable=import-outside-toplevel

    rows = {
        leaf.member: EligibilityMember(
            member=leaf.member,
            definition_fingerprint=leaf.fingerprint,
            status="not_introduced",
            state_epoch=None,
            last_event_id=None,
            reason=None,
        )
        for leaf in definitions
    }
    current: dict[str, EligibilityMember] = {}
    for fact in facts:
        if _applicable(fact, policy, round_number, cutoff):
            updated = _transition(current.get(fact.member.id), fact)
            current[fact.member.id] = updated
            rows[fact.member] = updated
    return tuple(sorted(rows.values(), key=lambda row: (row.member.id, row.member.version)))
