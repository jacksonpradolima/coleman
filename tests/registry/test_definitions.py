"""Immutable definition/version identities, namespaces and recipe ownership."""

from itertools import permutations

import pytest

from coleman.core.identity import Reference
from coleman.registry.definitions import Catalogue, MemberDefinition, RecipeBinding
from coleman.registry.limits import RegistryLimits
from coleman.state.contracts import StateBinding
from tests.registry.test_lifecycle import history, ref


def test_catalogue_permutation_and_exact_registration_retry_preserve_canonical_bytes():
    catalogue = history(count=0).catalogue
    for permutation in permutations(catalogue.definitions):
        changed = Catalogue.model_validate(catalogue.model_dump() | {"definitions": permutation})
        assert changed.to_json() == catalogue.to_json()
        assert changed.fingerprint == catalogue.fingerprint
    first = catalogue.definitions[0]
    assert catalogue.register(MemberDefinition.from_json(first.to_json())) == catalogue
    assert Catalogue.from_json(catalogue.to_json()) == catalogue


def test_duplicate_keys_conflicts_namespaces_and_historical_versions_are_explicit():
    catalogue = history(count=0).catalogue
    first = catalogue.definitions[0]
    with pytest.raises(ValueError, match="Duplicate catalogue"):
        Catalogue(schema_version=1, reference=catalogue.reference, definitions=(first, first))
    conflict = MemberDefinition.model_validate(first.model_dump() | {"configuration_fingerprint": "sha256:" + "f" * 64})
    with pytest.raises(ValueError, match="Conflicting immutable"):
        catalogue.register(conflict)
    second = MemberDefinition.model_validate(first.model_dump() | {"member": ref("A", "v2")})
    action = MemberDefinition.model_validate(first.model_dump() | {"kind": "action"})
    extended = catalogue.register(second).register(action)
    assert extended.resolve("expert", ref("A")) == first
    assert extended.resolve("expert", ref("A", "v2")) == second
    assert extended.resolve("action", ref("A")) == action
    for member in (ref("A", "missing"), ref("unknown")):
        with pytest.raises(ValueError, match="Unknown catalogue"):
            extended.resolve("expert", member)
    assert first.binding.reference == first.member
    assert first.binding.definition_fingerprint == first.fingerprint


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_strict_versions_unknown_fields_and_unchecked_instances_reject(version):
    leaf = history(count=0).catalogue.definitions[0]
    values = leaf.model_dump() | {"schema_version": version}
    with pytest.raises(ValueError):
        MemberDefinition.model_validate(values)
    unchecked_leaf = leaf.model_copy(update={"schema_version": version})
    with pytest.raises(ValueError):
        unchecked_leaf.to_json()
    values.pop("schema_version")
    with pytest.raises(ValueError):
        MemberDefinition.model_validate(values)
    candidate_values = leaf.model_dump() | {"loss": 0.0}
    with pytest.raises(ValueError):
        MemberDefinition.model_validate(candidate_values)


def test_recipe_reuses_exact_state_owners_and_nullable_existing_definition_bindings():
    leaf = history(count=0).catalogue.definitions[0]
    owners = tuple(
        StateBinding(
            owner=ref(name),
            implementation=leaf.implementation,
            state_codec=ref("toy-state"),
            configuration_fingerprint=leaf.configuration_fingerprint,
        )
        for name in ("weights", "rng")
    )
    recipe = RecipeBinding(
        schema_version=1,
        child=ref("child-A"),
        task=ref("public-ci-task"),
        expert=leaf.binding,
        state_epoch=ref("A:e0"),
        owners=owners,
        objective=ref("toy-loss"),
        target=None,
        loss_transform=None,
        controls=None,
    )
    assert [owner.owner.id for owner in recipe.owners] == ["rng", "weights"]
    assert RecipeBinding.from_json(recipe.to_json()) == recipe
    assert '"controls":null' in recipe.to_json()
    candidate_values = recipe.model_dump() | {"owners": (owners[0], owners[0])}
    with pytest.raises(ValueError):
        RecipeBinding.model_validate(candidate_values)
    with pytest.raises(ValueError):
        Reference(id="A", version=" ")
    limited_catalogue = history(count=0).catalogue
    selected_limits = RegistryLimits(maximum_definitions=1)
    with pytest.raises(ValueError, match="capacity"):
        limited_catalogue.register(leaf, limits=selected_limits)
