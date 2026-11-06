"""Finite declarative inventories and quotas reject ambiguity before storage."""

import pytest

from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.storage_contracts import ComponentSpec, ExecutionCompatibility, StorageLimits


@pytest.mark.parametrize("count", [128, 129, 256])
def test_implementation_inventory_allows_distinct_owners_plus_the_known_mapping(state_example, count):
    example = state_example
    runtime = example.execution_compatibility().runtime
    definitions = tuple(
        DefinitionBinding(
            reference=example.reference(f"authored-definition-{index:03d}"), definition_fingerprint=fingerprint(index)
        )
        for index in range(count)
    )
    value = ExecutionCompatibility(schema_version=1, runtime=runtime, owner_implementations=definitions)
    assert len(value.owner_implementations) == count
    assert ExecutionCompatibility.from_json(value.to_json()).to_json() == value.to_json()


def test_257_implementation_declarations_reject_before_a_store_or_owner_exists(state_example):
    example = state_example
    runtime = example.execution_compatibility().runtime
    definitions = tuple(
        DefinitionBinding(
            reference=example.reference(f"authored-definition-{index:03d}"), definition_fingerprint=fingerprint(index)
        )
        for index in range(257)
    )
    with pytest.raises(ValueError):
        ExecutionCompatibility(schema_version=1, runtime=runtime, owner_implementations=definitions)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_equal_implementation_ids_do_not_become_distinct_owners_by_changing_version(state_example, version):
    example = state_example
    current = example.execution_compatibility()
    first = current.owner_implementations[0]
    reference = type(first.reference)(id=first.reference.id, version=version)
    other = DefinitionBinding(reference=reference, definition_fingerprint=fingerprint("different-source"))
    definitions = (first, other)
    with pytest.raises(ValueError, match="unique"):
        ExecutionCompatibility(schema_version=1, runtime=current.runtime, owner_implementations=definitions)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_children", True),
        ("maximum_children", 33),
        ("maximum_components", 129),
        ("maximum_component_bytes", 16 * 1024**2 + 1),
        ("maximum_collection_bytes", 64 * 1024**2 + 1),
        ("receipt_window", 0),
        ("receipt_window", 65),
        ("maximum_root_bytes", 8 * 1024**3 + 1),
        ("maximum_entries", 16385),
        ("lock_wait_seconds", 0),
        ("lock_wait_seconds", -1),
        ("lock_wait_seconds", float("inf")),
        ("lock_wait_seconds", float("nan")),
        ("lock_wait_seconds", True),
    ],
)
def test_storage_bounds_are_explicit_and_never_coerced_or_nonfinite(field, value):
    payload = {"schema_version": 1, field: value}
    with pytest.raises(ValueError):
        StorageLimits.model_validate(payload)


def test_collection_declarations_canonicalize_order_but_not_duplicate_or_missing_owners(state_example):
    example = state_example
    original, _ = example.collection_layout((0, 1))
    reordered = type(original).model_validate(
        original.model_copy(
            update={"children": tuple(reversed(original.children)), "components": tuple(reversed(original.components))}
        )
    )
    assert reordered.to_json() == original.to_json()
    missing = original.model_copy(update={"components": original.components[1:]})
    with pytest.raises(ValueError, match="child|component"):
        type(original).model_validate(missing)
    duplicate = original.model_copy(update={"components": (*original.components, original.components[0])})
    with pytest.raises(ValueError, match="unique|duplicate"):
        type(original).model_validate(duplicate)


def test_replay_owner_cannot_hide_delivery_ownership_or_declare_it_twice(state_example):
    definition, _ = state_example.collection_layout()
    original = definition.components[0]
    missing_delivery = original.model_copy(update={"roles": ("algorithm",)})
    with pytest.raises(ValueError, match="delivery"):
        ComponentSpec.model_validate(missing_delivery)
    duplicates = original.model_copy(update={"roles": ("delivery", "delivery")})
    with pytest.raises(ValueError, match="unique"):
        ComponentSpec.model_validate(duplicates)
