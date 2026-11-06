"""Actual optional SQLite acknowledgement owns retries, stages and frontiers."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

from coleman.core.primitives import fingerprint
from coleman.state.components import ComponentBinding, ComponentValue
from coleman.state.local import LocalStateStore
from tests.state.support import command


@dataclass(frozen=True)
class Case:
    store: LocalStateStore
    bindings: tuple[ComponentBinding[Any, Any], ...]
    compatibility: Any
    initialization: Any
    receipt: Any


@pytest.fixture
def initialized(state_example, tmp_path):
    example = state_example
    definition, bindings = example.collection_layout()
    compatibility = example.execution_compatibility()
    values = example.origin_values(bindings)
    store = LocalStateStore.create(tmp_path / "store", definition, compatibility)
    initialize = command(0, None, kind="initialize", content="explicit-complete-origins")
    receipt = store.initialize(initialize, values, bindings, compatibility=compatibility)
    return Case(store, bindings, compatibility, initialize, receipt)


def step(stage, bindings, value=0.25):
    action = stage.owner(bindings[0]).step(value)
    stage.owner(bindings[-1]).apply(action.chosen)


def test_initialize_retained_retry_and_later_old_receipt_never_rewind_or_call_the_host(initialized):
    case = initialized
    request = command(1, case.receipt.current_head)
    calls = []

    def transition(stage):
        calls.append("actual-transition")
        step(stage, case.bindings)

    first = case.store.commit_transition(request, case.bindings, transition, compatibility=case.compatibility)
    second_request = command(2, first.current_head)
    second = case.store.commit_transition(second_request, case.bindings, transition, compatibility=case.compatibility)
    retry = case.store.commit_transition(request, case.bindings, transition, compatibility=case.compatibility)
    assert calls == ["actual-transition", "actual-transition"]
    assert retry.status == "retained"
    assert retry.accepted_head == first.accepted_head
    assert retry.current_head == second.current_head
    assert case.store.read_head() == second.current_head
    origins = ()
    initial_retry = case.store.initialize(case.initialization, origins, (), compatibility=case.compatibility)
    assert initial_retry.accepted_head == case.receipt.accepted_head
    assert initial_retry.current_head == second.current_head
    current = case.store.load(case.bindings, compatibility=case.compatibility)
    child = current.component(case.bindings[0].codec)
    assert child.state.decision_count == 2
    assert child.state.update_count == 2
    assert current.component(case.bindings[-1].codec).book.spent_seconds == 6


@pytest.mark.parametrize("conflict", ["input", "predecessor", "stale", "compatibility"])
def test_operational_conflicts_and_stale_parent_reject_before_any_new_factory_or_callback(initialized, conflict):
    case = initialized
    accepted = command(1, case.receipt.current_head)
    current = case.store.commit_transition(
        accepted, case.bindings, partial(step, bindings=case.bindings), compatibility=case.compatibility
    )
    calls = []

    def factory():
        calls.append("factory")
        return case.bindings[0].new_owner()

    wrapped = (
        ComponentBinding(case.bindings[0].codec, factory, case.bindings[0].frontier),
        *case.bindings[1:],
    )
    compatibility = case.compatibility
    if conflict == "input":
        request = command(1, case.receipt.current_head, content="different-command")
    elif conflict == "predecessor":
        request = command(1, current.current_head)
    elif conflict == "stale":
        request = command(2, case.receipt.current_head)
    else:
        request = accepted
        implementation = compatibility.owner_implementations[0]
        changed = type(implementation).model_validate(
            implementation.model_copy(update={"definition_fingerprint": fingerprint("another-owner-code")})
        )
        compatibility = type(compatibility).model_validate(
            compatibility.model_copy(update={"owner_implementations": (changed,)})
        )
    before = case.store.read_head()
    principal = case.store.commit_transition

    def callback(stage):
        calls.append("transition")

    with pytest.raises(ValueError):
        principal(request, wrapped, callback, compatibility=compatibility)
    assert calls == []
    assert case.store.read_head() == before


def test_callback_failure_discards_fresh_mutated_stage_and_preserves_all_acknowledged_components(initialized):
    case = initialized
    before = case.store.load(case.bindings, compatibility=case.compatibility)
    request = command(1, case.receipt.current_head)
    leaked_stage = []

    def transition(stage):
        leaked_stage.append(stage)
        step(stage, case.bindings)
        raise RuntimeError("deliberate-fault-after-real-host-update")

    with pytest.raises(RuntimeError, match="after-real-host-update"):
        case.store.commit_transition(request, case.bindings, transition, compatibility=case.compatibility)
    assert case.store.read_head() == case.receipt.current_head
    after = case.store.load(case.bindings, compatibility=case.compatibility)
    assert tuple(value.value.to_json() for value in after.values) == tuple(
        value.value.to_json() for value in before.values
    )
    binding = case.bindings[0]
    with pytest.raises(ValueError, match="no longer active"):
        leaked_stage[0].owner(binding)


def test_new_generation_cannot_publish_a_valid_older_causal_checkpoint(initialized):
    case = initialized
    old = case.store.load(case.bindings, compatibility=case.compatibility).component(case.bindings[0].codec)
    first = case.store.commit_transition(
        command(1, case.receipt.current_head),
        case.bindings,
        partial(step, bindings=case.bindings),
        compatibility=case.compatibility,
    )
    request = command(2, first.current_head)

    def rewind(stage):
        stage.owner(case.bindings[0]).restore_state(old)

    with pytest.raises(ValueError, match="prefix|rewind"):
        case.store.commit_transition(request, case.bindings, rewind, compatibility=case.compatibility)
    assert case.store.read_head() == first.current_head
    assert (
        case.store.load(case.bindings, compatibility=case.compatibility)
        .component(case.bindings[0].codec)
        .state.update_count
        == 1
    )


def test_transition_cannot_hide_an_unacknowledged_seal(initialized):
    case = initialized
    first = case.store.commit_transition(
        command(1, case.receipt.current_head),
        case.bindings,
        partial(step, bindings=case.bindings),
        compatibility=case.compatibility,
    )
    request = command(2, first.current_head)

    def hidden_seal(stage):
        stage.owner(case.bindings[0]).replay.compact()

    with pytest.raises(ValueError, match="sealed prefix|floor"):
        case.store.commit_transition(request, case.bindings, hidden_seal, compatibility=case.compatibility)
    assert case.store.read_head() == first.current_head
    seal_request = command(2, first.current_head, kind="seal")
    sealed = case.store.commit_transition(seal_request, case.bindings, hidden_seal, compatibility=case.compatibility)
    assert sealed.current_head.generation == 2
    actual = case.store.load(case.bindings, compatibility=case.compatibility).component(case.bindings[0].codec)
    assert actual.state.seal is not None
    assert actual.state.decisions == ()
    assert actual.state.decision_count == 1


def test_explicit_initialization_validates_all_actual_supplied_values_before_creating_any_owner(
    state_example, tmp_path
):
    example = state_example
    definition, bindings = example.collection_layout((0, 1))
    compatibility = example.execution_compatibility()
    values = example.origin_values(bindings)
    calls = []

    def factory(binding):
        calls.append(binding.codec.binding.owner.id)
        return binding.new_owner()

    watched = tuple(ComponentBinding(item.codec, partial(factory, item), item.frontier) for item in bindings)
    invalid_value = object.__new__(ComponentValue)
    object.__setattr__(invalid_value, "codec", values[-1].codec)
    object.__setattr__(invalid_value, "value", values[-1].value.model_copy(update={"schema_version": 9}))
    invalid = (*values[:-1], invalid_value)
    store = LocalStateStore.create(tmp_path / "invalid-last", definition, compatibility)
    request = command(0, None, kind="initialize")
    with pytest.raises(ValueError):
        store.initialize(request, invalid, watched, compatibility=compatibility)
    assert calls == []
    assert store.read_head() is None


def test_initialization_uses_the_running32_pending_states_instead_of_fresh_factory_defaults(
    state_example, pending_evidence, tmp_path
):
    example = state_example
    continuations, _, domain = pending_evidence
    definition, bindings = example.collection_layout(tuple(range(32)))
    compatibility = example.execution_compatibility()
    values = tuple(
        ComponentValue(binding.codec, state) for binding, state in zip(bindings, (*continuations, domain), strict=True)
    )
    store = LocalStateStore.create(tmp_path / "composed", definition, compatibility)
    request = command(0, None, kind="initialize", content="actual-running32-D95-U48")
    store.initialize(request, values, bindings, compatibility=compatibility)
    prepared = LocalStateStore.open(store.root).load(bindings, compatibility=compatibility)
    restored = tuple(prepared.component(binding.codec) for binding in bindings[:-1])
    assert sum(item.state.decision_count for item in restored) == 95
    assert sum(item.state.update_count for item in restored) == 48
    assert sum(len(item.host_state.pending) for item in restored) == 47
    assert sum(sum(item.host_state.loss_sums) for item in restored) == 16
    assert prepared.component(bindings[-1].codec).book.spent_seconds == 351
    assert tuple(item.to_json() for item in restored) == tuple(item.to_json() for item in continuations)


def test_corrupt_current_blob_never_falls_back_to_a_valid_prior_generation(initialized):
    case = initialized
    first = case.store.commit_transition(
        command(1, case.receipt.current_head),
        case.bindings,
        partial(step, bindings=case.bindings),
        compatibility=case.compatibility,
    )
    manifest = case.store.load(case.bindings, compatibility=case.compatibility).manifest
    blob = manifest.components[0]
    path = case.store.root / "blobs" / (blob.fingerprint[7:] + ".json")
    path.write_bytes(b"interrupted-current-child")
    principal = case.store.load
    with pytest.raises(ValueError, match="digest|fingerprint"):
        principal(case.bindings, compatibility=case.compatibility)
    assert first.current_head.generation == 1
