"""Actual fresh-state staging preserves admission order and declared live roots."""

from dataclasses import replace
from typing import cast

import pytest

from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import RootInventory
from coleman.registry.state import LiveOwner, OwnedBindings, OwnerInspection
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.components import ComponentBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import ExecutionCompatibility
from tests.registry.state_support import changed_recipe, command, definition, layout


@pytest.fixture(scope="module")
def compatibility():
    return ExecutionCompatibility(
        schema_version=1,
        runtime=SelectedInstalledRuntime.capture(extras=()),
        owner_implementations=(definition("counter-host"),),
    )


def initialize(tmp_path, fixture, compatibility):
    store = LocalStateStore.create(tmp_path / "store", fixture.definition, compatibility)
    owned = OwnedBindings(fixture.declaration, fixture.inspections)
    receipt = store.initialize(
        command(0, None, kind="initialize"), fixture.values, owned.bindings, compatibility=compatibility
    )
    fixture.calls.clear()
    return store, receipt.current_head


def unexpected_transition(_stage):
    """Fail if a refused transition reaches the actual store callback."""
    pytest.fail("callback ran")


def test_constructor_performs_no_factory_or_inventory_callback():
    fixture = layout()
    live = tuple(
        LiveOwner(inspection, owner) for inspection, owner in zip(fixture.inspections, fixture.origins, strict=True)
    )
    owned = OwnedBindings(fixture.declaration, fixture.inspections, live=live)
    assert len(owned.bindings) == 2
    assert fixture.calls == []


@pytest.mark.parametrize("limit", ["resources", "bytes"])
def test_selected_capacity_refuses_before_any_callback(limit):
    fixture = layout()
    limits = RegistryLimits(maximum_resources=1) if limit == "resources" else RegistryLimits(maximum_bytes=1)
    with pytest.raises(ValueError, match="capacity|maximum_bytes"):
        OwnedBindings(fixture.declaration, fixture.inspections, limits=limits)
    assert fixture.calls == []


def test_real_accepted_transition_and_retained_retry_have_one_callback_and_zero_retry_factories(
    tmp_path, compatibility
):
    fixture = layout()
    store, head = initialize(tmp_path, fixture, compatibility)
    live = tuple(
        LiveOwner(inspection, owner) for inspection, owner in zip(fixture.inspections, fixture.origins, strict=True)
    )
    owned = OwnedBindings(fixture.declaration, fixture.inspections, live=live)
    callbacks = []

    def increment(stage):
        callbacks.append("accepted")
        for inspection in reversed(fixture.inspections):
            owner = owned.owner(stage, inspection.binding)
            owner.values[0] += 1
            owner.rng.getrandbits(32)

    request = command(1, head)
    first = store.commit_transition(request, owned.bindings, increment, compatibility=compatibility)
    assert first.status == "new"
    assert callbacks == ["accepted"]
    assert sum(operation == "factory" for _child, operation in fixture.calls) == 2
    snapshot = store.load(owned.bindings, compatibility=compatibility)
    assert all(snapshot.component(inspection.binding.codec).values == (1,) for inspection in fixture.inspections)
    assert all(owner.values == [0] for owner in fixture.origins)
    fixture.calls.clear()
    second = store.commit_transition(request, owned.bindings, increment, compatibility=compatibility)
    assert second.status == "retained"
    assert callbacks == ["accepted"]
    assert fixture.calls == []
    conflicting = command(1, head, content="different")
    with pytest.raises(ValueError):
        store.commit_transition(conflicting, owned.bindings, increment, compatibility=compatibility)
    assert fixture.calls == []


def test_staged_alias_to_live_weights_rejects_before_restore_and_keeps_acknowledged_state(tmp_path, compatibility):
    fixture = layout(children=(0,))
    store, head = initialize(tmp_path, fixture, compatibility)
    inspection = fixture.inspections[0]
    original = inspection.binding

    def alias_factory():
        owner = original.new_owner()
        owner.values = fixture.origins[0].values
        return owner

    component = ComponentBinding(original.codec, alias_factory, original.frontier)
    inspected = OwnerInspection(component, inspection.inspect)
    live = (LiveOwner(inspected, fixture.origins[0]),)
    owned = OwnedBindings(fixture.declaration, (inspected,), live=live)
    request = command(1, head)
    with pytest.raises(ValueError, match="retained live root"):
        store.commit_transition(request, owned.bindings, unexpected_transition, compatibility=compatibility)
    assert not any(operation == "restore" for _child, operation in fixture.calls)
    assert fixture.origins[0].values == [0]
    assert store.read_head() == head


def test_two_fresh_wrappers_sharing_weights_reject_before_any_restore(tmp_path, compatibility):
    valid = layout()
    store, head = initialize(tmp_path, valid, compatibility)
    fixture = layout(shared_values=[0])
    owned = OwnedBindings(fixture.declaration, fixture.inspections)
    request = command(1, head)
    with pytest.raises(ValueError, match="alias"):
        store.commit_transition(request, owned.bindings, unexpected_transition, compatibility=compatibility)
    assert not any(operation == "restore" for _child, operation in fixture.calls)
    assert store.read_head() == head


def test_postrestore_replacement_alias_is_rejected_before_transition(tmp_path, compatibility):
    fixture = layout(children=(0,))
    store, head = initialize(tmp_path, fixture, compatibility)
    fixture.calls.clear()
    original = fixture.inspections[0]

    def replacement_factory():
        owner = original.binding.new_owner()
        owner.replacement = fixture.origins[0].values
        return owner

    component = ComponentBinding(original.binding.codec, replacement_factory, original.binding.frontier)
    inspection = OwnerInspection(component, original.inspect)
    owned = OwnedBindings(fixture.declaration, (inspection,), live=(LiveOwner(inspection, fixture.origins[0]),))
    request = command(1, head)
    with pytest.raises(ValueError, match="retained live root"):
        store.commit_transition(request, owned.bindings, unexpected_transition, compatibility=compatibility)
    assert sum(operation == "restore" for _child, operation in fixture.calls) == 1
    assert fixture.origins[0].values == [0]
    assert store.read_head() == head


@pytest.mark.parametrize("shape", ["roots-list", "recipes-list", "root-generator", "wrong-report"])
def test_malformed_host_inventory_rejects_without_normalizing_unbounded_shapes(tmp_path, compatibility, shape):
    fixture = layout(children=(0,))
    store, head = initialize(tmp_path, fixture, compatibility)
    original = fixture.inspections[0]

    def inspect(owner):
        result = original.inspect(owner)
        if shape == "roots-list":
            return RootInventory(result.recipes, cast(tuple, list(result.roots)))
        if shape == "recipes-list":
            return RootInventory(cast(tuple, list(result.recipes)), result.roots)
        if shape == "root-generator":
            return RootInventory(result.recipes, cast(tuple, iter(result.roots)))
        return cast(RootInventory, object())

    owned = OwnedBindings(fixture.declaration, (OwnerInspection(original.binding, inspect),))
    request = command(1, head)
    with pytest.raises(TypeError, match="RootInventory|tuple"):
        store.commit_transition(request, owned.bindings, unexpected_transition, compatibility=compatibility)
    assert store.read_head() == head


def test_inspected_stale_controls_reject_before_adaptive_callback(tmp_path, compatibility):
    fixture = layout(children=(0,))
    store, head = initialize(tmp_path, fixture, compatibility)
    original = fixture.inspections[0]

    def stale_factory():
        owner = original.binding.new_owner()
        owner.recipe = changed_recipe(owner.recipe, controls=definition("stale-controls"))
        return owner

    component = ComponentBinding(original.binding.codec, stale_factory, original.binding.frontier)
    owned = OwnedBindings(fixture.declaration, (OwnerInspection(component, original.inspect),))
    request = command(1, head)
    with pytest.raises(ValueError, match="recipe binding"):
        store.commit_transition(request, owned.bindings, unexpected_transition, compatibility=compatibility)
    assert not any(operation == "restore" for _child, operation in fixture.calls)


def test_wrapper_refuses_second_accepted_staging_and_unlisted_binding(tmp_path, compatibility):
    fixture = layout(children=(0,))
    store, head = initialize(tmp_path, fixture, compatibility)
    owned = OwnedBindings(fixture.declaration, fixture.inspections)
    original = fixture.inspections[0].binding

    def update(stage):
        unlisted = replace(original)
        with pytest.raises(ValueError, match="not in"):
            owned.owner(stage, unlisted)
        owned.owner(stage, original).values[0] += 1

    first = store.commit_transition(command(1, head), owned.bindings, update, compatibility=compatibility)
    request = command(2, first.current_head)
    with pytest.raises(ValueError, match="one accepted staging"):
        store.commit_transition(request, owned.bindings, update, compatibility=compatibility)
