"""Declared mathematical ownership and actual in-process alias rejection."""

from dataclasses import replace
from hashlib import sha256
from random import Random
from types import SimpleNamespace

import pytest

from coleman.core.identity import Reference
from coleman.core.primitives import canonical_json, fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.registry.definitions import RecipeBinding
from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import (
    DeclaredRootGuard,
    OwnershipDeclaration,
    PredictorStep,
    ResourceClaim,
    RootBinding,
    RootInventory,
    SharedPredictor,
    check_predictor_barrier,
)
from coleman.state.contracts import StateBinding


def ref(name, version="1"):
    return Reference(id=name, version=version)


def definition(name):
    return DefinitionBinding(reference=ref(name), definition_fingerprint=fingerprint(name))


def state_owner(name):
    return StateBinding(
        owner=ref(name),
        implementation=definition("toy-owner"),
        state_codec=ref("toy-state"),
        configuration_fingerprint=fingerprint({"owner": name}),
    )


def recipe(name):
    return RecipeBinding(
        schema_version=1,
        child=ref(name),
        task=ref("task-" + name),
        expert=definition("expert-" + name),
        state_epoch=ref("epoch-" + name),
        owners=(state_owner("owner-" + name),),
        objective=ref("objective"),
        target=definition("target"),
        loss_transform=definition("loss-transform"),
        controls=definition("bounds-cadence"),
    )


def independent_case():
    recipes = (recipe("a"), recipe("b"))
    claims = []
    roots = []
    for child in recipes:
        for role, root in (
            ("owner", object()),
            ("weights", [1.0, 2.0]),
            ("rng", Random(19)),
            ("predictor", SimpleNamespace(coefficients=[2.0])),
            ("cache", {"result": 3.0}),
        ):
            claim = ResourceClaim(
                resource=ref(child.child.id + "-" + role),
                owner=child.owners[0],
                child=child.child,
                role=role,
                access="owned",
            )
            claims.append(claim)
            roots.append(RootBinding(claim, root))
    declared = OwnershipDeclaration(schema_version=1, recipes=recipes, resources=tuple(claims))
    return declared, RootInventory(recipes, tuple(roots))


def swap_root(inventory, resource, root):
    return replace(
        inventory,
        roots=tuple(
            RootBinding(item.claim, root) if item.claim.resource.id == resource else item for item in inventory.roots
        ),
    )


def test_ownership_canonical_roundtrip_and_set_permutation():
    declaration, inventory = independent_case()
    permuted = OwnershipDeclaration(
        schema_version=1,
        recipes=tuple(reversed(declaration.recipes)),
        resources=tuple(reversed(declaration.resources)),
    )
    assert permuted.to_json() == declaration.to_json()
    assert OwnershipDeclaration.from_json(declaration.to_json()) == declaration
    DeclaredRootGuard(declaration).check(replace(inventory, recipes=tuple(reversed(inventory.recipes))))


@pytest.mark.parametrize("schema", [None, True, "1", 0, 2])
def test_ownership_rejects_missing_coerced_and_future_versions(schema):
    declaration, _ = independent_case()
    raw = declaration.model_dump(mode="json")
    if schema is None:
        del raw["schema_version"]
    else:
        raw["schema_version"] = schema
    with pytest.raises(ValueError):
        OwnershipDeclaration.model_validate(raw)


def test_ownership_wire_rejects_unknown_and_duplicate_fields():
    declaration, _ = independent_case()
    original = declaration.to_json()
    unknown = original[:-1] + ',"unknown":null}'
    with pytest.raises(ValueError):
        OwnershipDeclaration.from_json(unknown)
    duplicate = '{"schema_version":1,' + original[1:]
    with pytest.raises(ValueError, match="Duplicate"):
        OwnershipDeclaration.from_json(duplicate)


@pytest.mark.parametrize("role", ["weights", "rng", "predictor", "cache"])
def test_distinct_wrappers_reject_actual_mutable_aliases(role):
    declaration, inventory = independent_case()
    first = next(item.root for item in inventory.roots if item.claim.resource.id == "a-" + role)
    aliased = swap_root(inventory, "b-" + role, first)
    guard = DeclaredRootGuard(declaration)
    with pytest.raises(ValueError, match="alias"):
        guard.check(aliased)


def test_equal_values_and_equal_seeds_are_accepted_only_with_distinct_roots():
    declaration, inventory = independent_case()
    DeclaredRootGuard(declaration).check(inventory)
    first = next(item.root for item in inventory.roots if item.claim.resource.id == "a-rng")
    second = next(item.root for item in inventory.roots if item.claim.resource.id == "b-rng")
    assert first is not second
    assert first.getstate() == second.getstate()
    guard = DeclaredRootGuard(declaration)
    aliased = swap_root(inventory, "b-rng", first)
    with pytest.raises(ValueError, match="alias"):
        guard.check(aliased)


def test_root_custom_equality_and_hash_are_never_used():
    class IdentityOnly:
        def __eq__(self, _other):
            raise AssertionError("Runtime root equality must not run")

        def __hash__(self):
            raise AssertionError("Runtime root hashing must not run")

    declaration, inventory = independent_case()
    roots = tuple(RootBinding(item.claim, IdentityOnly()) for item in inventory.roots)
    DeclaredRootGuard(declaration).check(replace(inventory, roots=roots))


@pytest.mark.parametrize(
    "field", ["child", "task", "expert", "state_epoch", "target", "loss_transform", "controls", "owners"]
)
def test_actual_recipe_requires_exact_child_epoch_target_bounds_cadence_binding(field):
    declaration, inventory = independent_case()
    original = inventory.recipes[0]
    value = getattr(original, field)
    if field in {"child", "task", "state_epoch"}:
        changed = ref("wrong-" + field)
    elif field == "owners":
        changed = (state_owner("wrong-owner"),)
    else:
        changed = definition("wrong-" + field)
    assert changed != value
    actual = RecipeBinding.model_validate(original.model_copy(update={field: changed}))
    guard = DeclaredRootGuard(declaration)
    candidate = replace(inventory, recipes=(actual, inventory.recipes[1]))
    with pytest.raises(ValueError, match="binding"):
        guard.check(candidate)


@pytest.mark.parametrize("change", ["missing", "duplicate", "different", "list"])
def test_root_inventory_is_exact_complete_and_bounded(change):
    declaration, inventory = independent_case()
    if change == "missing":
        roots = inventory.roots[:-1]
    elif change == "duplicate":
        roots = (*inventory.roots[:-1], inventory.roots[0])
    elif change == "different":
        entry = inventory.roots[-1]
        roots = (
            *inventory.roots[:-1],
            RootBinding(entry.claim.model_copy(update={"resource": ref("other")}), entry.root),
        )
    else:
        roots = list(inventory.roots)
    guard = DeclaredRootGuard(declaration)
    candidate = replace(inventory, roots=roots)
    with pytest.raises(ValueError, match="resource|bounded"):
        guard.check(candidate)


def test_declaration_requires_complete_exact_named_state_owners():
    declaration, _ = independent_case()
    claims = tuple(item for item in declaration.resources if item.role != "owner")
    with pytest.raises(ValueError, match="enclosing owner"):
        OwnershipDeclaration(schema_version=1, recipes=declaration.recipes, resources=claims)
    wrong = declaration.resources[0].model_copy(update={"owner": state_owner("unbound")})
    with pytest.raises(ValueError, match="exact declared state owner"):
        OwnershipDeclaration(
            schema_version=1, recipes=declaration.recipes, resources=(wrong, *declaration.resources[1:])
        )


def immutable_case(root=None):
    declaration, inventory = independent_case()
    payload = canonical_json({"features": [2, 3], "source": "public-toy"}).encode()
    claims = tuple(
        ResourceClaim(
            resource=ref("immutable-input"),
            owner=None,
            child=child.child,
            role="input",
            access="immutable_bytes",
            immutable_fingerprint="sha256:" + sha256(payload).hexdigest(),
        )
        for child in declaration.recipes
    )
    declared = OwnershipDeclaration(
        schema_version=1,
        recipes=declaration.recipes,
        resources=(*declaration.resources, *claims),
    )
    actual = payload if root is None else root
    roots = (*inventory.roots, *(RootBinding(claim, actual) for claim in claims))
    return declared, RootInventory(inventory.recipes, roots)


def test_explicit_verified_immutable_canonical_bytes_can_be_shared():
    declaration, inventory = immutable_case()
    DeclaredRootGuard(declaration).check(inventory)
    assert inventory.roots[-1].root is inventory.roots[-2].root


@pytest.mark.parametrize("root", [([1, 2],), SimpleNamespace(frozen=True), b'{"x": 1}', b"[NaN]"])
def test_frozen_containers_and_unverified_bytes_are_not_immutable_profiles(root):
    declaration, inventory = immutable_case(root)
    guard = DeclaredRootGuard(declaration)
    with pytest.raises(ValueError, match="bytes|fingerprint"):
        guard.check(inventory)


def test_equal_byte_values_do_not_hide_conflicting_resource_root_identity():
    declaration, inventory = immutable_case()
    last = inventory.roots[-1]
    separate = bytes(bytearray(last.root))
    assert separate == last.root
    assert separate is not last.root
    altered = replace(inventory, roots=(*inventory.roots[:-1], RootBinding(last.claim, separate)))
    guard = DeclaredRootGuard(declaration)
    with pytest.raises(ValueError, match="conflicting live roots"):
        guard.check(altered)


@pytest.mark.parametrize("payload", [b'{"x": 1}', b"[NaN]", b"[" * 65 + b"0" + b"]" * 65])
def test_correct_byte_digest_does_not_replace_canonical_finite_and_depth_checks(payload):
    declaration, inventory = immutable_case()
    resources = tuple(
        claim.model_copy(update={"immutable_fingerprint": "sha256:" + sha256(payload).hexdigest()})
        if claim.access == "immutable_bytes"
        else claim
        for claim in declaration.resources
    )
    altered = OwnershipDeclaration.model_validate(declaration.model_copy(update={"resources": resources}))
    claims = {(claim.resource.id, "" if claim.child is None else claim.child.id): claim for claim in altered.resources}
    roots = tuple(
        RootBinding(claims[(item.claim.resource.id, "" if item.claim.child is None else item.claim.child.id)], payload)
        if item.claim.access == "immutable_bytes"
        else item
        for item in inventory.roots
    )
    guard = DeclaredRootGuard(altered)
    candidate = replace(inventory, roots=roots)
    with pytest.raises(ValueError):
        guard.check(candidate)


def test_enclosing_owner_and_nested_mutable_role_require_distinct_roots_in_this_profile():
    declaration, inventory = independent_case()
    owner = next(item.root for item in inventory.roots if item.claim.resource.id == "a-owner")
    guard = DeclaredRootGuard(declaration)
    aliased = swap_root(inventory, "a-predictor", owner)
    with pytest.raises(ValueError, match="alias"):
        guard.check(aliased)


def shared_case():
    declaration, inventory = independent_case()
    owner = state_owner("shared-predictor-owner")
    shared_root = {"coefficient": 0, "updates": 0}
    owner_claim = ResourceClaim(resource=ref("shared-owner"), owner=owner, child=None, role="owner", access="owned")
    writer = ResourceClaim(resource=ref("shared-predictor"), owner=owner, child=None, role="predictor", access="owned")
    borrowed = tuple(
        ResourceClaim(
            resource=writer.resource,
            owner=owner,
            child=child.child,
            role="predictor",
            access="borrowed_predictor",
        )
        for child in declaration.recipes
    )
    sharing = SharedPredictor(
        resource=writer.resource,
        writer=owner,
        consumers=tuple(child.child for child in declaration.recipes),
        operation=ref("fit-once"),
        barrier=ref("update-snapshot-consume"),
        snapshot_version=ref("snapshot", "7"),
    )
    declared = OwnershipDeclaration(
        schema_version=1,
        recipes=declaration.recipes,
        shared_owners=(owner,),
        resources=(*declaration.resources, owner_claim, writer, *borrowed),
        shared_predictors=(sharing,),
    )
    roots = (
        *inventory.roots,
        RootBinding(owner_claim, object()),
        RootBinding(writer, shared_root),
        *(RootBinding(claim, shared_root) for claim in borrowed),
    )
    return declared, RootInventory(inventory.recipes, roots), sharing


def barrier_steps(sharing):
    return tuple(
        PredictorStep(
            actor=actor,
            kind=kind,
            operation=sharing.operation,
            barrier=sharing.barrier,
            snapshot_version=sharing.snapshot_version,
        )
        for actor, kind in (
            (sharing.writer.owner, "update"),
            (sharing.writer.owner, "snapshot"),
            *((consumer, "consume") for consumer in sharing.consumers),
        )
    )


def test_intentional_predictor_sharing_has_one_update_and_one_coherent_version():
    declaration, inventory, sharing = shared_case()
    DeclaredRootGuard(declaration).check(inventory)
    predictor = next(item.root for item in inventory.roots if item.claim.resource.id == "shared-predictor")
    predictor["updates"] += 1
    predictor["coefficient"] = 9
    snapshot = canonical_json(predictor).encode()
    consumed = [(consumer, sharing.snapshot_version, snapshot) for consumer in sharing.consumers]
    check_predictor_barrier(sharing, barrier_steps(sharing))
    assert predictor["updates"] == 1
    assert consumed[0][1:] == consumed[1][1:]


@pytest.mark.parametrize("change", ["missing", "writer", "order", "snapshot", "barrier", "operation", "duplicate"])
def test_predictor_barrier_rejects_wrong_owner_order_and_versions(change):
    _, _, sharing = shared_case()
    steps = barrier_steps(sharing)
    if change == "missing":
        steps = steps[1:]
    elif change == "duplicate":
        steps = (steps[0], *steps)
    elif change == "order":
        steps = (*steps[:2], *reversed(steps[2:]))
    else:
        field = {"writer": "actor", "snapshot": "snapshot_version", "barrier": "barrier", "operation": "operation"}[
            change
        ]
        steps = (steps[0].model_copy(update={field: ref("wrong")}), *steps[1:])
    with pytest.raises(ValueError, match="barrier"):
        check_predictor_barrier(sharing, steps)


@pytest.mark.parametrize("change", ["missing", "writer", "consumer", "resource"])
def test_mutable_predictor_sharing_requires_exact_writer_and_consumers(change):
    declaration, _, sharing = shared_case()
    clauses = (
        ()
        if change == "missing"
        else (
            sharing.model_copy(
                update={
                    {"writer": "writer", "consumer": "consumers", "resource": "resource"}.get(change, "writer"): {
                        "writer": state_owner("other"),
                        "consumer": (ref("other"),),
                        "resource": ref("other"),
                    }.get(change)
                }
            ),
        )
    )
    candidate = declaration.model_copy(update={"shared_predictors": clauses})
    with pytest.raises(ValueError, match="sharing|consumers"):
        OwnershipDeclaration.model_validate(candidate)


def test_same_named_shared_predictor_cannot_resolve_to_two_real_roots():
    declaration, inventory, _ = shared_case()
    last = inventory.roots[-1]
    actual = replace(inventory, roots=(*inventory.roots[:-1], RootBinding(last.claim, dict(last.root))))
    guard = DeclaredRootGuard(declaration)
    with pytest.raises(ValueError, match="conflicting live roots"):
        guard.check(actual)


def test_selected_resource_and_actual_byte_capacities_refuse_without_callbacks():
    declaration, _ = independent_case()
    resource_limits = RegistryLimits(maximum_resources=1)
    with pytest.raises(ValueError, match="capacity"):
        DeclaredRootGuard(declaration, limits=resource_limits)
    declaration, _ = immutable_case()
    byte_limits = RegistryLimits(maximum_bytes=1)
    with pytest.raises(ValueError, match="byte capacity"):
        DeclaredRootGuard(declaration, limits=byte_limits)
