"""Public owned-staging refusals preserve the no-callback admission boundary."""

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from coleman.registry.ownership import RootInventory
from coleman.registry.state import LiveOwner, OwnedBindings, OwnerInspection
from coleman.state.components import ComponentBinding, ComponentCodec
from tests.registry.state_support import changed_recipe, definition, layout


@pytest.mark.parametrize("kind", ["list", "empty", "over-cap", "wrong-value", "missing", "duplicate"])
def test_invalid_complete_inspection_inventory_rejects_before_host_callbacks(kind):
    fixture = layout()
    values = {
        "list": list(fixture.inspections),
        "empty": (),
        "over-cap": (fixture.inspections[0],) * 129,
        "wrong-value": (object(),),
        "missing": fixture.inspections[:1],
        "duplicate": (fixture.inspections[0], fixture.inspections[0]),
    }[kind]
    inspections = cast(tuple, values)
    with pytest.raises((ValueError, TypeError)):
        OwnedBindings(fixture.declaration, inspections)
    assert fixture.calls == []


@pytest.mark.parametrize("kind", ["component", "inspector", "codec", "factory", "frontier", "binding"])
def test_invalid_component_or_callback_binding_refuses_before_any_host_call(kind):
    fixture = layout(children=(0,))
    selected = fixture.inspections[0]
    component = selected.binding
    if kind == "component":
        inspected = replace(selected, binding=cast(ComponentBinding, object()))
    elif kind == "inspector":
        inspected = replace(selected, inspect=cast(Callable, None))
    elif kind == "codec":
        inspected = replace(selected, binding=replace(component, codec=cast(ComponentCodec, object())))
    elif kind in {"factory", "frontier"}:
        field = "new_owner" if kind == "factory" else "frontier"
        inspected = replace(selected, binding=replace(component, **{field: None}))
    else:
        binding = type(component.codec.binding).model_validate(
            component.codec.binding.model_dump()
            | {"configuration_fingerprint": definition("different-config").definition_fingerprint}
        )
        codec = type(component.codec)(binding, component.codec.state_type)
        inspected = replace(selected, binding=replace(component, codec=codec))
    with pytest.raises((ValueError, TypeError)):
        OwnedBindings(fixture.declaration, (inspected,))
    assert fixture.calls == []


@pytest.mark.parametrize("kind", ["list", "over-cap", "wrong-value", "other-inspector", "partial", "duplicate"])
def test_live_inventory_requires_exact_complete_selected_inspectors_before_callbacks(kind):
    fixture = layout()
    correct = tuple(
        LiveOwner(inspection, owner) for inspection, owner in zip(fixture.inspections, fixture.origins, strict=True)
    )
    variants = {
        "list": list(correct),
        "over-cap": correct[:1] * 129,
        "wrong-value": (object(),),
        "other-inspector": (replace(correct[0], inspection=replace(correct[0].inspection)), correct[1]),
        "partial": correct[:1],
        "duplicate": correct[:1] * 2,
    }
    live = cast(tuple, variants[kind])
    with pytest.raises((ValueError, TypeError)):
        OwnedBindings(fixture.declaration, fixture.inspections, live=live)
    assert fixture.calls == []


@pytest.mark.parametrize("method", ["restore_state", "export_state"])
def test_partial_fresh_factory_set_cannot_restore_or_export_one_owner_early(method):
    fixture = layout()
    owned = OwnedBindings(fixture.declaration, fixture.inspections)
    partial = owned.bindings[0].new_owner()
    fixture.calls.clear()
    operation = getattr(partial, method)
    arguments = (fixture.values[0].value,) if method == "restore_state" else ()
    with pytest.raises(ValueError, match="Complete owned factory inventory"):
        operation(*arguments)
    assert fixture.calls == []


@pytest.mark.parametrize("kind", ["recipes", "roots"])
def test_oversized_callback_report_is_bounded_before_normalization_or_restore(kind):
    fixture = layout(children=(0,))
    selected = fixture.inspections[0]

    def inspect(owner):
        original = selected.inspect(owner)
        if kind == "recipes":
            return RootInventory(original.recipes * 33, original.roots)
        return RootInventory(original.recipes, original.roots * 257)

    owned = OwnedBindings(fixture.declaration, (OwnerInspection(selected.binding, inspect),))
    with pytest.raises(ValueError, match="finite capacity"):
        owned.bindings[0].new_owner()
    assert not any(operation == "restore" for _child, operation in fixture.calls)


def test_two_component_reports_cannot_disagree_on_one_child_recipe_before_restore():
    fixture = layout()
    first, second = fixture.inspections
    replacement = changed_recipe(fixture.declaration.recipes[1], controls=definition("wrong-controls"))

    def inspect(owner):
        actual = first.inspect(owner)
        return RootInventory((replacement,), actual.roots)

    owned = OwnedBindings(fixture.declaration, (OwnerInspection(first.binding, inspect), second))
    owned.bindings[0].new_owner()
    with pytest.raises(ValueError, match="disagree on the exact recipe"):
        owned.bindings[1].new_owner()
    assert not any(operation == "restore" for _child, operation in fixture.calls)
