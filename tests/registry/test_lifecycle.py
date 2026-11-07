"""Independent public synthetic lifecycle schedules and causal negative cases."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from coleman.core.identity import Reference
from coleman.registry.definitions import MemberDefinition
from coleman.registry.lifecycle import EligibilitySnapshot, LifecycleFact, PopulationPolicy, RegistryState

FIXTURES = Path(__file__).parents[1] / "fixtures/registry/v1"
START = datetime(2026, 1, 1, 9, tzinfo=UTC)


def ref(name, version="v1"):
    return Reference(id=name, version=version)


def history(*, count=None, clock="untimed", **changes):
    values = json.loads((FIXTURES / "history.json").read_text(encoding="utf-8"))
    if count is not None:
        values["facts"] = values["facts"][:count]
    values["policy"]["clock"] = clock
    if clock == "timed":
        for fact in values["facts"]:
            fact["known_at"] = START + timedelta(minutes=fact["admitted_before_round"])
    return RegistryState.model_validate(values | changes)


def fact(state, name="A", operation="introduce", round_number=0, **changes):
    member = ref(name)
    leaf = state.catalogue.resolve(state.policy.kind, member)
    values = dict(
        schema_version=1,
        event_id=f"event-{len(state.facts)}",
        sequence=len(state.facts),
        member=member,
        definition_fingerprint=leaf.fingerprint,
        operation=operation,
        effective_from_round=round_number,
        admitted_before_round=round_number,
        known_at=None,
        reason="public-synthetic-lifecycle",
        state_epoch=ref(name + ":e0") if operation == "introduce" else None,
        predecessor_epoch=None,
        disposition=None,
    )
    return LifecycleFact.model_validate(values | changes)


def test_hand_schedule_active_dispositions_and_exact_epochs():
    state = history()
    oracle = json.loads((FIXTURES / "oracle.json").read_text(encoding="utf-8"))
    for row in oracle["rounds"]:
        snapshot = state.snapshot(row["round"])
        assert [member.id for member in snapshot.active] == row["active"]
        assert [member.state_epoch.id for member in snapshot.members if member.status == "active"] == row["epochs"]
        assert snapshot.active_contract_fingerprint == row["active_contract_fingerprint"]
        assert EligibilitySnapshot.from_json(snapshot.to_json()) == snapshot
    assert [row.status for row in state.snapshot(1).members] == ["active", "inactive", "not_introduced"]
    assert state.snapshot(2).active == (ref("A"), ref("B"))
    assert state.snapshot(3).active == (ref("A"), ref("C"))
    assert not state.snapshot(5).active


def test_every_activation_and_knowledge_boundary_is_validated_before_successor():
    original = history(count=0)
    introduced = original.append(fact(original, round_number=10))
    before = introduced.to_json()
    earlier_round = fact(introduced, operation="retire", round_number=5)
    with pytest.raises(ValueError, match="nondecreasing"):
        introduced.append(earlier_round)
    assert introduced.to_json() == before
    timed = history(count=0, clock="timed")
    timed = timed.append(fact(timed, known_at=START + timedelta(minutes=10)))
    earlier_knowledge = fact(timed, operation="retire", known_at=START + timedelta(minutes=5))
    with pytest.raises(ValueError, match="known_at"):
        timed.append(earlier_knowledge)


def test_unknown_timed_knowledge_never_becomes_live_and_new_fact_is_prospective():
    state = history(count=0, clock="timed")
    unknown = fact(state, known_at=None)
    state = state.append(unknown)
    for cutoff in (START, START + timedelta(days=365)):
        assert not state.snapshot(0, cutoff).active
        assert state.snapshot(0, cutoff).members[0].status == "not_introduced"
    with pytest.raises(ValueError, match="fixed timed/untimed"):
        state.snapshot(0)
    known = fact(state, round_number=1, known_at=START, event_id="new-knowledge")
    state = state.append(known)
    assert not state.snapshot(0, START + timedelta(days=1)).active
    assert state.snapshot(1, START).active == (ref("A"),)
    assert state.facts[0] == unknown
    untimed = history(count=0)
    timed_fact = fact(untimed, known_at=START)
    with pytest.raises(ValueError, match="Untimed population"):
        untimed.append(timed_fact)


def test_future_introduction_waits_for_both_rounds_and_timed_cutoff():
    state = history(count=0, clock="timed")
    state = state.append(fact(state, round_number=2, effective_from_round=1, known_at=START))
    assert not state.snapshot(1, START).active
    assert not state.snapshot(2, START - timedelta(microseconds=1)).active
    assert state.snapshot(2, START).active == (ref("A"),)
    assert state.snapshot(3, START + timedelta(seconds=1)).active_contract_fingerprint == (
        state.snapshot(2, START).active_contract_fingerprint
    )
    assert state.snapshot(3, START + timedelta(seconds=1)).fingerprint != state.snapshot(2, START).fingerprint


def test_active_contract_ignores_inactive_fact_history_unrelated_versions_and_catalogue_publication():
    state = history(count=3)
    first = state.snapshot(1)
    changed_leaf = MemberDefinition.model_validate(
        state.catalogue.resolve("expert", ref("C")).model_dump() | {"member": ref("unrelated", "v17")}
    )
    catalogue = state.catalogue.register(changed_leaf)
    changed = RegistryState.model_validate(state.model_dump() | {"catalogue": catalogue}).snapshot(2)
    assert changed.active_contract_fingerprint == first.active_contract_fingerprint
    assert changed.fingerprint != first.fingerprint
    assert changed.prefix_fingerprint == first.prefix_fingerprint
    retired_inactive = state.append(fact(state, name="B", operation="retire", round_number=2)).snapshot(2)
    assert retired_inactive.active_contract_fingerprint == first.active_contract_fingerprint
    assert retired_inactive.prefix_fingerprint != first.prefix_fingerprint
    unavailable = state.append(fact(state, name="B", operation="resume", round_number=2))
    assert unavailable.snapshot(2).active_contract_fingerprint != first.active_contract_fingerprint


def test_retry_sequence_unknown_version_and_conflicting_implementation_leave_prior_bytes():
    state = history(count=0)
    event = fact(state)
    state = state.append(event)
    assert state.append(LifecycleFact.from_json(event.to_json())) == state
    before = state.to_json()
    for replacement in (
        {"reason": "conflict"},
        {"event_id": "wrong-sequence", "sequence": 3},
        {"event_id": "unknown-version", "sequence": 1, "member": ref("A", "v9")},
        {"event_id": "wrong-definition", "sequence": 1, "definition_fingerprint": "sha256:" + "f" * 64},
    ):
        candidate = LifecycleFact.model_validate(event.model_dump() | replacement)
        with pytest.raises(ValueError):
            state.append(candidate)
        assert state.to_json() == before


@pytest.mark.parametrize("operation", ["inactive", "unavailable", "resume", "retire", "reenter"])
def test_absent_member_cannot_transition(operation):
    state = history(count=0)
    extra = {"state_epoch": ref("A:e1"), "predecessor_epoch": ref("A:e0"), "disposition": "restart"}
    candidate = fact(state, operation=operation, **(extra if operation == "reenter" else {}))
    with pytest.raises(ValueError):
        state.append(candidate)


@pytest.mark.parametrize("sleep", ["inactive", "unavailable"])
def test_predecision_sleep_resume_and_later_active_missing_outcome_are_distinct(sleep):
    state = history(count=1)
    sleeping = state.append(fact(state, operation=sleep, round_number=1))
    assert not sleeping.snapshot(1).active
    assert sleeping.snapshot(1).members[0].state_epoch == ref("A:e0")
    resumed = sleeping.append(fact(sleeping, operation="resume", round_number=2))
    assert resumed.snapshot(2).active == (ref("A"),)
    assert resumed.snapshot(2).members[0].state_epoch == ref("A:e0")


def test_reentry_requires_explicit_definition_and_fresh_or_retained_epoch():
    state = history(count=1)
    state = state.append(fact(state, operation="retire", round_number=1))
    for disposition, epoch in (("retain", "A:e1"), ("reset", "A:e0"), ("restart", "A:e0")):
        candidate = fact(
            state,
            operation="reenter",
            round_number=2,
            disposition=disposition,
            state_epoch=ref(epoch),
            predecessor_epoch=ref("A:e0"),
        )
        with pytest.raises(ValueError):
            state.append(candidate)
    retained = state.append(
        fact(
            state,
            operation="reenter",
            round_number=2,
            disposition="retain",
            state_epoch=ref("A:e0"),
            predecessor_epoch=ref("A:e0"),
        )
    )
    assert retained.snapshot(2).members[0].state_epoch == ref("A:e0")
    for disposition in ("reset", "restart"):
        restarted = state.append(
            fact(
                state,
                operation="reenter",
                round_number=2,
                disposition=disposition,
                state_epoch=ref("A:e1"),
                predecessor_epoch=ref("A:e0"),
            )
        )
        assert restarted.snapshot(2).members[0].state_epoch == ref("A:e1")


def test_version_restart_returning_to_old_version_never_activates_two_versions_or_reuses_epoch():
    state = history(count=1)
    state = state.append(fact(state, operation="retire", round_number=1))
    first = state.catalogue.resolve("expert", ref("A"))
    second = MemberDefinition.model_validate(first.model_dump() | {"member": ref("A", "v2")})
    state = RegistryState.model_validate(state.model_dump() | {"catalogue": state.catalogue.register(second)})
    changed = fact(
        state,
        operation="reenter",
        round_number=2,
        member=second.member,
        definition_fingerprint=second.fingerprint,
        disposition="restart",
        state_epoch=ref("A:e1"),
        predecessor_epoch=ref("A:e0"),
    )
    state = state.append(changed)
    assert state.snapshot(2).active == (ref("A", "v2"),)
    duplicate_introduction = fact(state, operation="introduce", round_number=3)
    with pytest.raises(ValueError):
        state.append(duplicate_introduction)
    state = state.append(
        fact(state, operation="retire", round_number=3, member=second.member, definition_fingerprint=second.fingerprint)
    )
    reused_epoch = fact(
        state,
        operation="reenter",
        round_number=4,
        disposition="restart",
        state_epoch=ref("A:e0"),
        predecessor_epoch=ref("A:e1"),
    )
    with pytest.raises(ValueError, match="historical epoch"):
        state.append(reused_epoch)
    state = state.append(
        fact(
            state,
            operation="reenter",
            round_number=4,
            disposition="restart",
            state_epoch=ref("A:e2"),
            predecessor_epoch=ref("A:e1"),
        )
    )
    assert state.snapshot(4).active == (ref("A"),)
    assert [row.status for row in state.snapshot(4).members] == ["active", "retired"]


def test_snapshot_cannot_forge_dispositions_leaves_or_clock_policy():
    original = history().snapshot(1)
    for change in (
        {"members": ()},
        {"definitions": original.definitions[:-1]},
        {"cutoff": START},
        {"round": True},
        {"schema_version": 2},
    ):
        candidate = original.model_dump() | change
        with pytest.raises(ValueError):
            EligibilitySnapshot.model_validate(candidate)
    unknown_clock = original.policy.model_dump() | {"clock": "automatic"}
    with pytest.raises(ValueError):
        PopulationPolicy.model_validate(unknown_clock)
