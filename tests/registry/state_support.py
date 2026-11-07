"""Small actual mutable hosts for accepted-staging tests, without a learner claim."""

from dataclasses import dataclass
from random import Random
from typing import Annotated

from pydantic import Field

from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.registry.definitions import RecipeBinding
from coleman.registry.ownership import OwnershipDeclaration, ResourceClaim, RootBinding, RootInventory
from coleman.registry.state import OwnerInspection
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
from coleman.state.contracts import StateBinding
from coleman.state.random_state import PythonRandomState, capture_random, restore_random
from coleman.state.storage_contracts import CollectionDefinition, CommitIntent, CommitRequest, ComponentSpec


def reference(name, version="1"):
    return Reference(id=name, version=version)


def definition(name):
    return DefinitionBinding(
        reference=reference(name), definition_fingerprint=fingerprint([name, "public-constructed-v1"])
    )


class CounterState(CoreContract):
    recipe: RecipeBinding
    values: tuple[Annotated[int, Field(strict=True)], ...]
    rng: PythonRandomState


class CounterOwner:
    def __init__(self, recipe, calls, *, values=None, replacement=None):
        self.recipe = recipe
        self.values = [0] if values is None else values
        self.rng = Random(100)
        self.calls = calls
        self.replacement = replacement

    def export_state(self):
        self.calls.append((self.recipe.child.id, "export"))
        return CounterState(
            schema_version=1, recipe=self.recipe, values=tuple(self.values), rng=capture_random(self.rng)
        )

    def restore_state(self, state):
        self.calls.append((self.recipe.child.id, "restore"))
        state = CounterState.model_validate(state)
        self.recipe = state.recipe
        self.values = list(state.values) if self.replacement is None else self.replacement
        self.rng = Random()
        restore_random(self.rng, state.rng)


@dataclass
class Layout:
    declaration: OwnershipDeclaration
    inspections: tuple[OwnerInspection, ...]
    definition: CollectionDefinition
    values: tuple[ComponentValue, ...]
    origins: tuple[CounterOwner, ...]
    calls: list


def layout(children=(0, 1), *, shared_values=None, replacement=None):
    calls, recipes, inspections, claims, specs, origins, values = [], [], [], [], [], [], []
    for child in children:
        identity = reference(f"ci-child-{child}")
        binding = StateBinding(
            owner=identity,
            implementation=definition("counter-host"),
            state_codec=reference("counter-state"),
            configuration_fingerprint=fingerprint({"child": child}),
        )
        recipe = RecipeBinding(
            schema_version=1,
            child=identity,
            task=identity,
            expert=definition("public-ci-recipe"),
            state_epoch=reference("initial-epoch"),
            owners=(binding,),
            objective=reference("constructed-loss"),
            target=None,
            loss_transform=None,
            controls=None,
        )
        recipes.append(recipe)
        child_claims = tuple(
            ResourceClaim(
                resource=reference(f"{identity.id}-{role}"), owner=binding, child=identity, role=role, access="owned"
            )
            for role in ("owner", "weights", "rng")
        )
        claims.extend(child_claims)

        def factory(selected=recipe):
            calls.append((selected.child.id, "factory"))
            return CounterOwner(selected, calls, values=shared_values, replacement=replacement)

        def inspect(owner, selected=child_claims):
            calls.append((owner.recipe.child.id, "inventory"))
            return RootInventory(
                (owner.recipe,),
                tuple(
                    RootBinding(claim, root)
                    for claim, root in zip(selected, (owner, owner.values, owner.rng), strict=True)
                ),
            )

        codec = ComponentCodec(binding, CounterState)
        component = ComponentBinding(codec, factory, lambda _state: None)
        inspections.append(OwnerInspection(component, inspect))
        specs.append(ComponentSpec(binding=binding, child=identity, roles=("algorithm",), frontier_profile="none"))
        origin = CounterOwner(recipe, calls)
        origins.append(origin)
        values.append(ComponentValue(codec, origin.export_state()))
    declaration = OwnershipDeclaration(schema_version=1, recipes=tuple(recipes), resources=tuple(claims))
    collection = CollectionDefinition(
        schema_version=1,
        reference=reference("owned-counter-definition"),
        session=reference("owned-counter-session"),
        source_epoch=reference("owned-counter-source"),
        children=tuple(recipe.child for recipe in recipes),
        components=tuple(specs),
    )
    calls.clear()
    return Layout(declaration, tuple(inspections), collection, tuple(values), tuple(origins), calls)


def command(generation, expected, kind="transition", content="increment"):
    return CommitRequest(
        schema_version=1,
        expected=expected,
        intent=CommitIntent(
            schema_version=1,
            generation=generation,
            kind=kind,
            content_fingerprint=fingerprint([content, generation, kind]),
        ),
    )


def changed_recipe(recipe, **updates):
    return RecipeBinding.model_validate({**recipe.model_dump(), **updates})
