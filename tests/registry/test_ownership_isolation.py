"""Actual independent host state, RNG traces, and refreshed staging roots."""

from random import Random

import pytest

from coleman.core.base import CoreContract
from coleman.core.protocols import Stateful
from coleman.registry.ownership import (
    DeclaredRootGuard,
    OwnershipDeclaration,
    ResourceClaim,
    RootBinding,
    RootInventory,
)
from coleman.state.random_state import PythonRandomState, capture_random, restore_random
from tests.registry.test_ownership import recipe, ref


class ToyState(CoreContract):
    weights: tuple[int, int]
    statistics: tuple[int, int]
    pending: tuple[tuple[str, int], ...]
    rng: PythonRandomState
    predictor: tuple[int, ...]
    transform: tuple[int, ...]
    cache: tuple[tuple[str, int], ...]
    environment: tuple[int, ...]
    path: tuple[int, ...]
    counters: tuple[int, int]
    cadence: tuple[int, int]
    movement: tuple[int, int]


class ToyRecipe:
    """A host-owned direct consumer; optional declarations do not drive it."""

    def __init__(self, seed=27):
        self.weights = [0, 0]
        self.statistics = {"count": 0, "total": 0}
        self.pending = {}
        self.rng = Random(seed)
        self.rng.gauss(0, 1)  # Leave a real cached Gaussian in the owned backend.
        self.predictor = [3]
        self.transform = [2]
        self.cache = {}
        self.environment = []
        self.path = []
        self.counters = [0, 0]
        self.cadence = [3, 0]
        self.movement = [100, 0]
        self.restore_calls = 0

    def decide(self, value):
        selected = self.rng.randrange(2)
        label = "local-" + str(self.counters[0])
        self.pending[label] = selected
        self.cache[label] = value * self.predictor[0] + self.transform[0]
        self.environment.append(value)
        self.path.append(selected)
        self.counters[0] += 1
        self.cadence[1] = self.counters[0] % self.cadence[0]
        self.movement[1] += selected
        return label, selected, self.cache[label], tuple(self.weights)

    def feedback(self, label, observed):
        selected = self.pending.pop(label)
        self.weights[selected] += observed
        self.statistics["count"] += 1
        self.statistics["total"] += observed
        self.counters[1] += 1

    def export_state(self):
        return ToyState(
            schema_version=1,
            weights=(self.weights[0], self.weights[1]),
            statistics=(self.statistics["count"], self.statistics["total"]),
            pending=tuple(sorted(self.pending.items())),
            rng=capture_random(self.rng),
            predictor=tuple(self.predictor),
            transform=tuple(self.transform),
            cache=tuple(sorted(self.cache.items())),
            environment=tuple(self.environment),
            path=tuple(self.path),
            counters=(self.counters[0], self.counters[1]),
            cadence=(self.cadence[0], self.cadence[1]),
            movement=(self.movement[0], self.movement[1]),
        )

    def restore_state(self, state):
        checked = ToyState.model_validate(state)
        self.restore_calls += 1
        self.weights = list(checked.weights)
        self.statistics = dict(zip(("count", "total"), checked.statistics, strict=True))
        self.pending = dict(checked.pending)
        self.rng = Random()
        restore_random(self.rng, checked.rng)
        self.predictor = list(checked.predictor)
        self.transform = list(checked.transform)
        self.cache = dict(checked.cache)
        self.environment = list(checked.environment)
        self.path = list(checked.path)
        self.counters = list(checked.counters)
        self.cadence = list(checked.cadence)
        self.movement = list(checked.movement)


ROLES = (
    "owner",
    "weights",
    "statistics",
    "pending",
    "rng",
    "predictor",
    "transform",
    "cache",
    "environment",
    "path",
    "counters",
    "cadence",
    "movement",
)


def declare():
    recipes = (recipe("a"), recipe("b"))
    claims = tuple(
        ResourceClaim(
            resource=ref(child.child.id + "-" + role),
            owner=child.owners[0],
            child=child.child,
            role=role,
            access="owned",
        )
        for child in recipes
        for role in ROLES
    )
    return OwnershipDeclaration(schema_version=1, recipes=recipes, resources=claims)


def inspect_roots(declaration, owners):
    roots = tuple(
        RootBinding(
            claim,
            owners[claim.child.id] if claim.role == "owner" else getattr(owners[claim.child.id], claim.role),
        )
        for claim in declaration.resources
        if claim.child.id in owners
    )
    recipes = tuple(child for child in declaration.recipes if child.child.id in owners)
    return RootInventory(recipes, roots)


def one_round(owner, value, observed):
    trace = owner.decide(value)
    owner.feedback(trace[0], observed)
    return trace


def isolated(inputs):
    owner = ToyRecipe()
    trace = tuple(one_round(owner, value, observed) for value, observed in inputs)
    return trace, owner.export_state()


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_interleaved_reordered_and_restored_direct_recipes_preserve_complete_isolated_trace(order):
    inputs = {"a": ((2, 1), (4, 3), (3, 2), (9, 1)), "b": ((7, 4), (1, 2), (6, 5), (8, 3))}
    expected = {name: isolated(values) for name, values in inputs.items()}
    owners = {name: ToyRecipe() for name in inputs}
    direct_protocol: Stateful[ToyState] = owners["a"]
    assert direct_protocol.export_state().rng.gauss_next is not None
    traces = {name: [] for name in inputs}
    declaration = declare()
    DeclaredRootGuard(declaration).check(inspect_roots(declaration, owners))
    for index in range(4):
        for name in order:
            traces[name].append(one_round(owners[name], *inputs[name][index]))
        if index == 1:
            old = owners
            snapshots = {name: owner.export_state() for name, owner in old.items()}
            guard = DeclaredRootGuard(declaration, inspect_roots(declaration, old))
            owners = {}
            for name in order:
                owners[name] = ToyRecipe(seed=999)
                guard.check_partial(inspect_roots(declaration, owners).roots)
            guard.check(inspect_roots(declaration, owners))
            for name in order:
                owners[name].restore_state(ToyState.from_json(snapshots[name].to_json()))
                guard.check(inspect_roots(declaration, owners))
            assert {name: owner.export_state() for name, owner in old.items()} == snapshots
    for name, owner in owners.items():
        assert tuple(traces[name]) == expected[name][0]
        assert owner.export_state().to_json() == expected[name][1].to_json()


@pytest.mark.parametrize("action", ["change", "reset", "retire"])
def test_changing_resetting_or_retiring_one_recipe_preserves_other_complete_state_and_rng(action):
    declaration = declare()
    owners = {"a": ToyRecipe(), "b": ToyRecipe()}
    one_round(owners["a"], 2, 5)
    owners["b"].decide(8)  # Keep real delayed feedback pending on B.
    before = owners["b"].export_state().to_json()
    DeclaredRootGuard(declaration).check(inspect_roots(declaration, owners))
    if action == "change":
        one_round(owners["a"], 9, 17)
        owners["a"].predictor[0] = 400
        owners["a"].transform[0] = 100
    elif action == "reset":
        owners["a"].restore_state(ToyRecipe(seed=81).export_state())
    else:
        del owners["a"]
    assert owners["b"].export_state().to_json() == before
    expected = ToyRecipe()
    expected.restore_state(ToyState.from_json(before))
    assert owners["b"].decide(11) == expected.decide(11)
    assert owners["b"].export_state().to_json() == expected.export_state().to_json()


@pytest.mark.parametrize("role", ["owner", "weights", "rng", "predictor", "cache", "pending"])
def test_staged_roots_cannot_alias_retained_live_roots_before_restore(role):
    declaration = declare()
    live = {"a": ToyRecipe(), "b": ToyRecipe()}
    one_round(live["a"], 6, 2)
    before = {name: owner.export_state().to_json() for name, owner in live.items()}
    guard = DeclaredRootGuard(declaration, inspect_roots(declaration, live))
    fresh = ToyRecipe()
    staged = {"a": fresh}
    if role == "owner":
        staged["a"] = live["b"]  # Even another child's new resource ID cannot hide it.
    else:
        setattr(fresh, role, getattr(live["b"], role))
    with pytest.raises(ValueError, match="retained live root"):
        guard.check_partial(inspect_roots(declaration, staged).roots)
    assert fresh.restore_calls == 0
    assert {name: owner.export_state().to_json() for name, owner in live.items()} == before


@pytest.mark.parametrize("source", ["retained", "staged"])
def test_post_restore_replacement_is_rechecked_before_adaptive_callback(source):
    declaration = declare()
    live = {"a": ToyRecipe(), "b": ToyRecipe()}
    snapshots = {name: owner.export_state() for name, owner in live.items()}
    guard = DeclaredRootGuard(declaration, inspect_roots(declaration, live))
    staged = {"a": ToyRecipe(), "b": ToyRecipe()}
    guard.check(inspect_roots(declaration, staged))
    staged["a"].restore_state(snapshots["a"])
    staged["a"].cache = live["b"].cache if source == "retained" else staged["b"].cache
    callbacks = []
    with pytest.raises(ValueError, match="alias"):
        guard.check(inspect_roots(declaration, staged))
        callbacks.append(staged["a"].decide(4))
    assert callbacks == []
    assert {name: owner.export_state() for name, owner in live.items()} == snapshots


def test_guard_keeps_strong_references_to_live_roots_even_after_host_replacement():
    declaration = declare()
    live = {"a": ToyRecipe(), "b": ToyRecipe()}
    original = live["a"].weights
    guard = DeclaredRootGuard(declaration, inspect_roots(declaration, live))
    live["a"].weights = list(original)
    staged = {"a": ToyRecipe()}
    staged["a"].weights = original
    with pytest.raises(ValueError, match="retained live root"):
        guard.check_partial(inspect_roots(declaration, staged).roots)
