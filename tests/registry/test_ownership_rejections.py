"""Ownership refusals preserve real declared resources before adaptive work."""

from dataclasses import replace
from hashlib import sha256
from random import Random
from types import SimpleNamespace
from typing import Any, cast

import pytest

from coleman.core.primitives import canonical_json, fingerprint
from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import (
    DeclaredRootGuard,
    OwnershipDeclaration,
    ResourceClaim,
    RootBinding,
    RootInventory,
    SharedPredictor,
)
from tests.registry.test_ownership import immutable_case, independent_case, ref, shared_case, state_owner


def mutable_values(inventory):
    """Inspect only the known concrete public toy values used by these cases."""
    result = []
    for item in inventory.roots:
        root = item.root
        if type(root) is list:
            value = tuple(root)
        elif type(root) is dict:
            value = tuple(sorted(root.items()))
        elif type(root) is Random:
            value = root.getstate()
        elif type(root) is SimpleNamespace:
            value = tuple(root.coefficients)
        else:
            value = None
        result.append((item.claim.resource.id, value))
    return tuple(result)


def refuse_declaration(declaration, inventory, updates, message):
    """Reach the real declaration validator while retaining the valid prior."""
    before, original = mutable_values(inventory), declaration.to_json()
    candidate = declaration.model_copy(update=updates)
    with pytest.raises(ValueError, match=message):
        OwnershipDeclaration.model_validate(candidate)
    assert declaration.to_json() == original
    assert mutable_values(inventory) == before
    DeclaredRootGuard(declaration).check(inventory)


def refuse_inventory(declaration, original, candidate, message, *, error: type[Exception] = ValueError):
    """Require a complete guard refusal before any hypothetical adaptive work."""
    before = mutable_values(original)
    roots = tuple(item.root for item in original.roots)
    guard = DeclaredRootGuard(declaration)
    reached_adaptive_work = False
    with pytest.raises(error, match=message):
        guard.check(candidate)
        reached_adaptive_work = True
    assert not reached_adaptive_work
    assert all(item.root is root for item, root in zip(original.roots, roots, strict=True))
    assert mutable_values(original) == before
    guard.check(original)


@pytest.mark.parametrize(
    "updates",
    [
        {"access": "shared"},
        {"owner": None},
        {"immutable_fingerprint": fingerprint("unsupported-byte-profile")},
        {"access": "immutable_bytes", "role": "input", "immutable_fingerprint": fingerprint("bytes")},
        {"access": "immutable_bytes", "owner": None, "role": "cache", "immutable_fingerprint": fingerprint("bytes")},
        {"access": "immutable_bytes", "owner": None, "role": "input"},
        {"access": "borrowed_predictor", "role": "predictor", "child": None},
        {"access": "borrowed_predictor", "role": "weights"},
    ],
)
def test_resource_access_profiles_reject_incomplete_or_contradictory_ownership(updates):
    declaration, inventory = independent_case()
    original = inventory.roots[1].claim
    before = mutable_values(inventory)
    candidate = original.model_copy(update=updates)
    with pytest.raises(ValueError):
        ResourceClaim.model_validate(candidate)
    assert mutable_values(inventory) == before
    DeclaredRootGuard(declaration).check(inventory)


def test_shared_consumers_cannot_repeat_a_logical_child_under_another_version():
    declaration, inventory, sharing = shared_case()
    before = mutable_values(inventory)
    candidate = sharing.model_copy(update={"consumers": (sharing.consumers[0], ref(sharing.consumers[0].id, "other"))})
    with pytest.raises(ValueError, match="unique child"):
        SharedPredictor.model_validate(candidate)
    assert mutable_values(inventory) == before
    DeclaredRootGuard(declaration).check(inventory)


@pytest.mark.parametrize(
    "change", ["child", "claim", "owner", "shared-owner", "no-owner", "missing-root", "extra-root"]
)
def test_duplicate_and_incomplete_logical_inventories_preserve_prior_roots(change):
    declaration, inventory = independent_case()
    recipes, resources = declaration.recipes, declaration.resources
    if change == "child":
        updates = {"recipes": (recipes[0], recipes[1].model_copy(update={"child": recipes[0].child}))}
        message = "unique child"
    elif change == "claim":
        updates, message = {"resources": (*resources, resources[0])}, "Duplicate resource claim"
    elif change == "owner":
        updates = {"recipes": (recipes[0], recipes[1].model_copy(update={"owners": recipes[0].owners}))}
        message = "unique state owners"
    elif change == "shared-owner":
        updates, message = {"shared_owners": (recipes[0].owners[0],)}, "unique state owners"
    elif change == "no-owner":
        updates = {"recipes": tuple(recipe.model_copy(update={"owners": ()}) for recipe in recipes)}
        message = "unique state owners"
    elif change == "missing-root":
        updates = {"resources": tuple(claim for claim in resources if claim.resource.id != "a-owner")}
        message = "enclosing owner root"
    else:
        owner_root = next(claim for claim in resources if claim.resource.id == "a-owner")
        updates = {"resources": (*resources, owner_root.model_copy(update={"resource": ref("second-owner-root")}))}
        message = "enclosing owner root"
    refuse_declaration(declaration, inventory, updates, message)


def test_collection_owner_capacity_is_coupled_across_individually_valid_recipes():
    declaration, inventory = independent_case()
    first, second = declaration.recipes
    owners = tuple(state_owner(f"capacity-owner-{index}") for index in range(129))
    recipes = (first.model_copy(update={"owners": owners[:128]}), second.model_copy(update={"owners": owners[128:]}))
    refuse_declaration(declaration, inventory, {"recipes": recipes}, "unique state owners")


@pytest.mark.parametrize("change", ["unknown-child", "child-version", "wrong-owning-child"])
def test_resource_child_join_requires_exact_version_and_owning_child(change):
    declaration, inventory = independent_case()
    claim = declaration.resources[0]
    if change == "unknown-child":
        child, message = ref("unknown-child"), "exact declared child/version"
    elif change == "child-version":
        assert claim.child is not None
        child, message = ref(claim.child.id, "other"), "exact declared child/version"
    else:
        child, message = declaration.recipes[1].child, "wrong child binding"
    resources = (claim.model_copy(update={"child": child}), *declaration.resources[1:])
    refuse_declaration(declaration, inventory, {"resources": resources}, message)


@pytest.mark.parametrize("change", ["duplicate-clause", "unused-clause", "missing-writer", "second-writer"])
def test_predictor_sharing_requires_one_actual_declared_writer_resource(change):
    declaration, inventory, sharing = shared_case()
    if change == "duplicate-clause":
        updates, message = {"shared_predictors": (sharing, sharing)}, "Duplicate shared predictor"
    elif change == "unused-clause":
        updates = {"shared_predictors": (sharing, sharing.model_copy(update={"resource": ref("undeclared-predictor")}))}
        message = "requires its owned root"
    elif change == "missing-writer":
        resources = tuple(
            claim
            for claim in declaration.resources
            if not (claim.resource == sharing.resource and claim.access == "owned")
        )
        updates, message = {"resources": resources}, "exactly one named writer"
    else:
        child = declaration.recipes[1]
        second = ResourceClaim(
            resource=sharing.resource, owner=child.owners[0], child=child.child, role="predictor", access="owned"
        )
        resources = tuple(
            claim
            for claim in declaration.resources
            if not (claim.resource == sharing.resource and claim.child == child.child)
        )
        updates, message = {"resources": (*resources, second)}, "exactly one named writer"
    refuse_declaration(declaration, inventory, updates, message)


@pytest.mark.parametrize("change", ["version", "role"])
def test_one_resource_id_cannot_hide_conflicting_versions_or_roles(change):
    declaration, inventory, sharing = shared_case()
    claims = list(declaration.resources)
    index = next(
        index for index, claim in enumerate(claims) if claim.resource == sharing.resource and claim.child is not None
    )
    update = {"resource": ref(sharing.resource.id, "other")} if change == "version" else {"role": "cache"}
    claims[index] = claims[index].model_copy(update=update)
    refuse_declaration(declaration, inventory, {"resources": tuple(claims)}, "conflicting versions or roles")


def test_borrowed_resource_must_name_the_exact_sole_writer_owner():
    declaration, inventory, sharing = shared_case()
    resources = tuple(
        claim.model_copy(update={"owner": declaration.recipes[0].owners[0]})
        if claim.resource == sharing.resource and claim.child == declaration.recipes[0].child
        else claim
        for claim in declaration.resources
    )
    refuse_declaration(declaration, inventory, {"resources": resources}, "owner|writer")


@pytest.mark.parametrize("change", ["writer-codec", "consumer-version"])
def test_sharing_clause_requires_exact_writer_codec_and_consuming_child_version(change):
    declaration, inventory, sharing = shared_case()
    if change == "writer-codec":
        update = {"writer": sharing.writer.model_copy(update={"state_codec": ref("other-codec")})}
        message = "sharing owner"
    else:
        first = sharing.consumers[0]
        update = {"consumers": (ref(first.id, "other"), *sharing.consumers[1:])}
        message = "consumers differ"
    clause = sharing.model_copy(update=update)
    refuse_declaration(declaration, inventory, {"shared_predictors": (clause,)}, message)


@pytest.mark.parametrize("change", ["digest", "mutable-input", "version"])
def test_shared_immutable_input_declarations_require_a_coherent_profile(change):
    declaration, inventory = immutable_case()
    claims = list(declaration.resources)
    index = next(
        index
        for index, claim in enumerate(claims)
        if claim.access == "immutable_bytes" and claim.child is not None and claim.child.id == "b"
    )
    if change == "digest":
        update, message = {"immutable_fingerprint": fingerprint("other")}, "coherent byte profile"
    elif change == "version":
        update, message = {"resource": ref(claims[index].resource.id, "other")}, "conflicting versions or roles"
    else:
        update = {"access": "owned", "owner": declaration.recipes[1].owners[0], "immutable_fingerprint": None}
        message = "coherent byte profile"
    claims[index] = claims[index].model_copy(update=update)
    refuse_declaration(declaration, inventory, {"resources": tuple(claims)}, message)


@pytest.mark.parametrize("change", ["record", "recipe-list", "missing-recipe", "extra-recipe", "entry"])
def test_runtime_inventory_refuses_wrong_shapes_and_counts_before_adaptive_work(change):
    declaration, inventory = independent_case()
    candidate: Any
    if change == "record":
        candidate, error, message = (
            SimpleNamespace(recipes=inventory.recipes, roots=inventory.roots),
            TypeError,
            "in-process tuples",
        )
    elif change == "recipe-list":
        candidate, error, message = (
            RootInventory(cast(Any, list(inventory.recipes)), inventory.roots),
            TypeError,
            "in-process tuples",
        )
    elif change == "missing-recipe":
        candidate, error, message = replace(inventory, recipes=inventory.recipes[:1]), ValueError, "every exact recipe"
    elif change == "extra-recipe":
        candidate = replace(inventory, recipes=(*inventory.recipes, inventory.recipes[0]))
        error, message = ValueError, "every exact recipe"
    else:
        entry = inventory.roots[0]
        lookalike = SimpleNamespace(claim=entry.claim, root=entry.root)
        candidate, error, message = (
            replace(inventory, roots=(lookalike, *inventory.roots[1:])),
            TypeError,
            "concrete RootBinding",
        )
    refuse_inventory(declaration, inventory, candidate, message, error=error)


def test_runtime_shape_rejection_does_not_enumerate_custom_containers():
    declaration, inventory = independent_case()
    calls = []

    class UnsupportedTuple(tuple):
        def __iter__(self):
            calls.append("iteration")
            return super().__iter__()

        def __len__(self):
            calls.append("length")
            return super().__len__()

    recipes = UnsupportedTuple(inventory.recipes)
    refuse_inventory(
        declaration, inventory, RootInventory(recipes, inventory.roots), "in-process tuples", error=TypeError
    )
    roots = UnsupportedTuple(inventory.roots)
    guard = DeclaredRootGuard(declaration)
    with pytest.raises(ValueError, match="bounded tuple"):
        guard.check_partial(roots)
    assert calls == []


@pytest.mark.parametrize("change", ["duplicate", "missing-bytes", "absent-byte-root"])
def test_runtime_exact_claim_inventory_cannot_drop_or_repeat_immutable_inputs(change):
    declaration, inventory = immutable_case()
    if change == "duplicate":
        candidate = replace(inventory, roots=(*inventory.roots[:-1], inventory.roots[0]))
        message = "duplicate or different declared resource"
    elif change == "missing-bytes":
        candidate = replace(
            inventory, roots=tuple(item for item in inventory.roots if item.claim.access != "immutable_bytes")
        )
        message = "every declared resource"
    else:
        candidate = replace(
            inventory,
            roots=tuple(
                RootBinding(item.claim, None) if item.claim.access == "immutable_bytes" else item
                for item in inventory.roots
            ),
        )
        message = "concrete canonical JSON bytes"
    refuse_inventory(declaration, inventory, candidate, message)


def test_actual_immutable_byte_capacity_is_checked_after_valid_declaration_admission():
    declaration, inventory = independent_case()
    payload = canonical_json({"features": "x" * 20000}).encode()
    digest = "sha256:" + sha256(payload).hexdigest()
    claims = tuple(
        ResourceClaim(
            resource=ref("large-shared-input"),
            owner=None,
            child=recipe.child,
            role="input",
            access="immutable_bytes",
            immutable_fingerprint=digest,
        )
        for recipe in declaration.recipes
    )
    declared = OwnershipDeclaration(
        schema_version=1, recipes=declaration.recipes, resources=(*declaration.resources, *claims)
    )
    roots = (*inventory.roots, *(RootBinding(claim, payload) for claim in claims))
    selected = RegistryLimits(maximum_bytes=len(payload) + 1)
    assert len(declared.to_json().encode()) < selected.maximum_bytes
    guard = DeclaredRootGuard(declared, limits=selected)
    before = mutable_values(inventory)
    candidate = RootInventory(inventory.recipes, roots)
    reached_adaptive_work = False
    with pytest.raises(ValueError, match="immutable resource bytes exceed"):
        guard.check(candidate)
        reached_adaptive_work = True
    assert not reached_adaptive_work
    assert mutable_values(inventory) == before
