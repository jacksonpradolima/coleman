"""Thirty-two unchanged public CI recipes with opt-in registry ownership.

This synthetic recovery example selects each child recipe from its own one-
expert population. Its inner CI actions remain the original A/B space. It does
not implement Hedge, pooled learning, or a predictive-efficacy experiment.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

from coleman.core.actions import ChoiceAction
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.registry.composition import RecipeExecution, RecipeFeedback, RegistryCheckpoint, require_recipe_eligibility
from coleman.registry.composition import RegistryOwner as CheckpointOwner
from coleman.registry.definitions import Catalogue, MemberDefinition, RecipeBinding
from coleman.registry.lifecycle import LifecycleFact, PopulationPolicy, RegistryState
from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import (
    DeclaredRootGuard,
    OwnershipDeclaration,
    ResourceClaim,
    RootBinding,
    RootInventory,
)
from coleman.registry.state import LiveOwner, OwnedBindings, OwnerInspection
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue
from coleman.state.contracts import StateBinding
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility


def load_host():
    """Load one explicitly selected public source, including the copied layout.

    Returns
    -------
    module
        The unchanged public ``examples/state/ci.py`` host. Installed execution
        uses the explicitly copied sibling ``owned_ci.py`` instead.

    Notes
    -----
    Only the fixed sibling and fixed repository path are considered. There is
    no source discovery, external consumer lookup or registration by name.
    """
    name = "authored_registry_owned_ci"
    if name in sys.modules:
        return sys.modules[name]
    copied = Path(__file__).with_name("owned_ci.py")
    source = copied if copied.is_file() else Path(__file__).parents[1] / "state" / "ci.py"
    specification = importlib.util.spec_from_file_location(name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError("The explicitly selected public owned CI source is required")
    host = importlib.util.module_from_spec(specification)
    sys.modules[name] = host
    specification.loader.exec_module(host)
    return host


def implementation() -> DefinitionBinding:
    """Bind the explicitly selected public collection adapter source.

    Returns
    -------
    DefinitionBinding
        Local content identity used only by the optional composed-state layer.

    Raises
    ------
    OSError
        The explicitly selected public host or this adapter source is unreadable.

    Notes
    -----
    Calls ``load_host``, which imports and executes the fixed public module on
    first use, then reads this adapter file to bind its exact bytes. Explicit
    collection layout and compatibility setup call this outer metadata boundary.
    """
    host = load_host()
    return DefinitionBinding(
        reference=host.reference("authored-registry-ci-collection"),
        definition_fingerprint="sha256:" + sha256(Path(__file__).read_bytes()).hexdigest(),
    )


class RegistryOwner(CheckpointOwner):
    """Adapt the SDK's complete registry owner to unchanged public CI emissions."""

    @property
    def recipe_bindings(self) -> tuple[RecipeBinding, ...]:
        """Read current immutable bindings for the trusted root inspector.

        Returns
        -------
        tuple of RecipeBinding
            Actual selected state bindings. Full checkpoint export/restore
            independently validates all retained histories and origins.
        """
        return self._state.ownership.recipes

    def emit(self, owner: Any, *, at: int | None = None):
        """Capture eligibility before an unchanged inner emission and retain it.

        Parameters
        ----------
        owner : ReplayOwner
            One actual fresh/restored public CI owner.
        at : int or None, optional
            Explicit input clock offset, using the unchanged host default.

        Returns
        -------
        ReplayDecisionRecord
            Original core/replay execution, without replacing its A/B universe.
        """
        host = load_host()
        ordinal = owner.replay.state.decision_count
        opportunity = host.opportunity(owner.child, ordinal, at=at)
        recipe = next(item for item in self._state.ownership.recipes if item.child == host.child_reference(owner.child))
        registry = next(item for item in self._state.registries if item.policy.task == recipe.task)
        snapshot = registry.snapshot(ordinal, cutoff=opportunity.request.timing.information_cutoff)
        require_recipe_eligibility(snapshot, recipe, opportunity.request)
        record = owner.replay.emit(owner.replay.prepare(opportunity), execution="offline_identity_v1")
        execution = RecipeExecution(schema_version=1, snapshot=snapshot, recipe=recipe, executed=record.executed)
        self.record(execution)
        return record

    def validate_feedback(self, offered: Any, *, apply_at: Any) -> None:
        """Join delayed feedback to its original recipe epoch before replay.

        Parameters
        ----------
        offered : ReplayFeedback
            Original unchanged replay feedback and actual target join.
        apply_at : datetime
            Explicit existing replay application instant.
        """
        origin = self._state.for_execution(offered.batch.decision_fingerprint)
        RecipeFeedback(schema_version=1, producer_fingerprint=origin.fingerprint, batch=offered.batch).validate_for(
            origin, load_host().capability(), apply_at=apply_at
        )

    def inactivate_next(self, owner: Any) -> None:
        """Admit one prospective lifecycle fact while retaining old pending joins.

        Parameters
        ----------
        owner : ReplayOwner
            Actual selected child at its current next-decision boundary.

        Notes
        -----
        This changes eligibility for a future opportunity only. Its existing
        pending outcomes still belong to their original eligible recipe epoch.
        No learner, RNG, cadence or path state changes here.
        """
        host = load_host()
        recipe = next(item for item in self._state.ownership.recipes if item.child == host.child_reference(owner.child))
        registry = next(item for item in self._state.registries if item.policy.task == recipe.task)
        boundary = owner.replay.state.decision_count
        fact = LifecycleFact(
            schema_version=1,
            event_id="inactive-after-pending",
            sequence=len(registry.facts),
            member=recipe.expert.reference,
            definition_fingerprint=recipe.expert.definition_fingerprint,
            operation="inactive",
            effective_from_round=boundary,
            admitted_before_round=boundary,
            known_at=host.instant(boundary + 1),
            reason="prospective-public-recipe-pause",
            state_epoch=None,
            predecessor_epoch=None,
            disposition=None,
        )
        self.replace_registry(registry.append(fact))


@dataclass(frozen=True)
class Layout:
    """Retain finite actual state codecs, declarations and trusted inspectors.

    Attributes
    ----------
    host : module
        Explicitly loaded unchanged public owned CI implementation.
    definition : CollectionDefinition
        Complete child and component inventory for the existing state store.
    bindings : tuple of ComponentBinding
        Original actual factories/codecs in child, domain and registry order.
    ownership : OwnershipDeclaration
        Exact recipe meanings and the complete declared mutable root inventory.
    origin : RegistryCheckpoint
        Explicit initial lifecycle histories and recipe bindings.
    inspections : tuple of OwnerInspection
        Trusted pure callbacks reporting actual roots for each exact binding.
    """

    host: Any
    definition: CollectionDefinition
    bindings: tuple[ComponentBinding[Any, Any], ...]
    ownership: OwnershipDeclaration
    origin: RegistryCheckpoint
    inspections: tuple[OwnerInspection[Any, Any], ...]

    def owned(self, *, live: tuple[Any, ...] = ()) -> OwnedBindings:
        """Create lazy wrappers; factories and inspectors run only in staging.

        Parameters
        ----------
        live : tuple, optional
            Complete actual retained owners in original binding order. Empty
            means no live collection is retained by this host.

        Returns
        -------
        OwnedBindings
            Optional wrappers compatible with the existing store protocol.
        """
        retained = (
            tuple(LiveOwner(item, owner) for item, owner in zip(self.inspections, live, strict=True)) if live else ()
        )
        return OwnedBindings(self.ownership, self.inspections, live=retained)

    def compatibility(self) -> ExecutionCompatibility:
        """Capture installed base metadata and both exact public source bindings.

        Returns
        -------
        ExecutionCompatibility
            Actual explicit outer-layer compatibility, with no extra selected.
        """
        previous = self.host.execution_compatibility()
        return ExecutionCompatibility(
            schema_version=1,
            runtime=previous.runtime,
            owner_implementations=(*previous.owner_implementations, implementation()),
        )


def _registry(host, child, owner):
    """Build a timed one-expert population without conflating inner actions."""
    identity = host.child_reference(child)
    expert = MemberDefinition(
        schema_version=1,
        kind="expert",
        member=host.reference(identity.id + "-owned-ci-expert"),
        implementation=owner.implementation,
        configuration_fingerprint=owner.configuration_fingerprint,
    )
    epoch = host.reference(identity.id + "-recipe-epoch")
    recipe = RecipeBinding(
        schema_version=1,
        child=identity,
        task=identity,
        expert=expert.binding,
        state_epoch=epoch,
        owners=(owner,),
        objective=host.reference("minimize-authored-ci-loss"),
        target=None,
        loss_transform=None,
        controls=None,
    )
    state = RegistryState(
        schema_version=1,
        catalogue=Catalogue(
            schema_version=1, reference=host.reference(identity.id + "-catalogue"), definitions=(expert,)
        ),
        policy=PopulationPolicy(
            population=host.reference(identity.id + "-population"), task=identity, kind="expert", clock="timed"
        ),
        limits=RegistryLimits(),
    ).append(
        LifecycleFact(
            schema_version=1,
            event_id="introduce",
            sequence=0,
            member=expert.member,
            definition_fingerprint=expert.fingerprint,
            operation="introduce",
            effective_from_round=0,
            admitted_before_round=0,
            known_at=host.instant(0),
            reason="explicit-public-synthetic-recipe",
            state_epoch=epoch,
            predecessor_epoch=None,
            disposition=None,
        )
    )
    return recipe, state


def _host_child(host, consumer):
    """Read this pinned public host's immutable configuration, without export."""
    if type(consumer) is not host.HostConsumer:  # pylint: disable=unidiomatic-typecheck
        raise TypeError("Ownership inspector requires the exact public HostConsumer class")
    # This adapter owns the pinned example: _state is its concrete immutable
    # HostState. Full snapshots still validate through the original codecs,
    # restore and export; this accessor reads only actual child configuration.
    state = consumer._state  # pylint: disable=protected-access
    if type(state) is not host.HostState or type(state.child) is not int:  # pylint: disable=unidiomatic-typecheck
        raise TypeError("Ownership inspector requires the exact immutable public HostState child")
    return state.child


def _inspect_child(recipe, claims, child, expected_policy, owner):
    """Read exact actual child configuration and real enclosing backing roots."""
    host = load_host()
    if type(owner.child) is not int or owner.child != child or _host_child(host, owner.host) != child:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Actual replay/host owner has the wrong child binding")
    if owner.replay.state.policy.fingerprint != expected_policy:
        raise ValueError("Actual replay owner has a stale task/expert capability binding")
    roots = {"owner": owner, "environment": owner.host, "session": owner.replay}
    return RootInventory((recipe,), tuple(RootBinding(claim, roots[claim.role]) for claim in claims))


def _inspect_outer(claim, owner):
    """Supply a real outer mutable root without inventing hidden backing state."""
    recipes = owner.recipe_bindings if isinstance(owner, RegistryOwner) else ()
    return RootInventory(recipes, (RootBinding(claim, owner),))


def collection_layout(children: tuple[int, ...] = tuple(range(32))) -> Layout:
    """Add one registry component to the unchanged complete owned CI layout.

    Parameters
    ----------
    children : tuple of int, optional
        Explicit bounded child inventory, default all32 existing public recipes.

    Returns
    -------
    Layout
        Complete34-owner profile for32 children:32 replay owners, one applied
        domain owner and one registry owner, with98 declared mutable roots.

    Notes
    -----
    The actual CI host owns immutable concrete snapshots and reconstructs a
    local RNG when deciding. Its mutable host/session roots are declared; this
    adapter invents no long-lived Random. The unchanged host schema retains
    statistics, pending, transform/cache, cadence, path and buffered RNG state.
    Root checks make the narrow declared roots/bindings checked claim. Factory,
    restore and inspection callbacks remain trusted, bounded and local.
    """
    host = load_host()
    old_definition, previous = host.collection_layout(children)
    registry_binding = StateBinding(
        owner=host.reference("zz-registry-ci-catalogue"),
        implementation=implementation(),
        state_codec=host.reference("authored-registry-ci-checkpoint"),
        configuration_fingerprint=fingerprint({"children": sorted(children), "profile": "unchanged-owned-ci-v1"}),
    )
    pairs = tuple(
        _registry(host, child, binding.codec.binding)
        for child, binding in zip(sorted(children), previous[:-1], strict=True)
    )
    recipes = tuple(item[0] for item in pairs)
    resources = []
    for recipe in recipes:
        resources.extend(
            ResourceClaim(
                resource=host.reference(recipe.child.id + "-" + role),
                owner=recipe.owners[0],
                child=recipe.child,
                role=role,
                access="owned",
            )
            for role in ("owner", "environment", "session")
        )
    outer = tuple(
        ResourceClaim(
            resource=host.reference(binding.owner.id + "-root"), owner=binding, child=None, role="owner", access="owned"
        )
        for binding in (previous[-1].codec.binding, registry_binding)
    )
    ownership = OwnershipDeclaration(
        schema_version=1,
        recipes=recipes,
        shared_owners=(previous[-1].codec.binding, registry_binding),
        resources=(*resources, *outer),
    )
    origin = RegistryCheckpoint(schema_version=1, registries=tuple(item[1] for item in pairs), ownership=ownership)
    registry_component = ComponentBinding(
        ComponentCodec(registry_binding, RegistryCheckpoint), partial(RegistryOwner, origin), lambda _state: None
    )
    definition = CollectionDefinition(
        schema_version=1,
        reference=host.reference("authored-registry-ci-definition"),
        session=host.reference("authored-registry-ci-session"),
        source_epoch=old_definition.source_epoch,
        children=old_definition.children,
        components=(
            *old_definition.components,
            ComponentSpec(binding=registry_binding, child=None, roles=("delivery",), frontier_profile="none"),
        ),
    )
    inspections = tuple(
        OwnerInspection(
            binding,
            partial(
                _inspect_child,
                recipe,
                tuple(claim for claim in ownership.resources if claim.child == recipe.child),
                child,
                host.policy(child).fingerprint,
            ),
        )
        for child, recipe, binding in zip(sorted(children), recipes, previous[:-1], strict=True)
    )
    inspections += (
        OwnerInspection(previous[-1], partial(_inspect_outer, outer[0])),
        OwnerInspection(registry_component, partial(_inspect_outer, outer[1])),
    )
    return Layout(host, definition, (*previous, registry_component), ownership, origin, inspections)


def pending_values(layout: Layout, *, order: tuple[int, ...] | None = None, isolated: bool = False):
    """Produce the real95/48 unequal frontier and complete outer origins.

    Parameters
    ----------
    layout : Layout
        Actual complete32-child declaration/codecs.
    order : tuple of int or None, optional
        Exact supplied permutation of participating children.
    isolated : bool, optional
        Emit one child's complete prefix at a time instead of interleaving.

    Returns
    -------
    tuple
        Complete actual owners and typed values. Each child has2..4 decisions
        and1..2 updates, preserving47 true pending outcomes in all32 children.
    """
    host = layout.host
    children = tuple(recipe.child for recipe in layout.ownership.recipes)
    indices = tuple(int(child.id.removeprefix("recipe-")) for child in children)
    selected = indices if order is None else order
    if len(selected) != len(indices) or set(selected) != set(indices):
        raise ValueError("Direct scheduling order must be an exact child permutation")
    owners = {child: host.ReplayOwner(child) for child in indices}
    domain, registry = host.Domain(), RegistryOwner(layout.origin)
    records = {child: [] for child in indices}
    for child in _emission_schedule(selected, isolated):
        record = registry.emit(owners[child])
        records[child].append(record)
        if not isinstance(record.executed.action, ChoiceAction):
            raise ValueError("Authored registry fixture requires the unchanged choice action")
        domain.apply(record.executed.action.chosen)
    for child in selected:
        _prefix_feedback(host, registry, owners[child], records[child])
    complete = (*(owners[child] for child in indices), domain, registry)
    values = tuple(
        ComponentValue(binding.codec, owner.export_state())
        for binding, owner in zip(layout.bindings, complete, strict=True)
    )
    return complete, values


def _emission_schedule(selected, isolated):
    """Bound the same original emissions under isolated/interleaved ordering."""
    if isolated:
        return tuple(child for child in selected for _ordinal in range(2 + child % 3))
    return tuple(child for ordinal in range(4) for child in selected if ordinal < 2 + child % 3)


def _prefix_feedback(host, registry, owner, records):
    """Apply the existing unchanged observed prefix for one actual child."""
    child = owner.child
    decisions, updates = 2 + child % 3, 1 + child % 2
    offered = tuple(
        host.feedback(record, decisions=decisions, available=decisions, value=((child + index) % 4) / 4)
        for index, record in enumerate(records[:updates])
    )
    at = host.instant(decisions)
    for item in offered:
        registry.validate_feedback(item, apply_at=at)
    owner.replay.admit_feedback(offered)
    owner.replay.advance(eligible_through=at, apply_at=at)


def complete_pending(layout: Layout, owners: tuple[Any, ...]) -> None:
    """Apply all47 real delayed outcomes through their original outer producer.

    Parameters
    ----------
    layout : Layout
        Exact selected child/task/expert/owner bindings.
    owners : tuple
        Complete fresh staged owners, with unchanged domain and registry last.

    Notes
    -----
    This performs no new decision, RNG draw, cadence advance or path charge.
    Existing retained replay validates original targets and actual chosen loss.
    """
    host, registry = layout.host, owners[-1]
    registry.inactivate_next(owners[0])
    for owner in owners[:-2]:
        child = owner.child
        before = owner.host.export_state()
        by_fingerprint = {record.executed.fingerprint: record for record in owner.replay.state.decisions}
        offered = tuple(
            host.feedback(
                by_fingerprint[pending.executed.fingerprint],
                decisions=before.decision_count,
                available=before.decision_count + 1,
                value=((child + pending.executed.proposal.request.round) % 4) / 4,
            )
            for pending in before.pending
        )
        at = host.instant(before.decision_count + 1)
        for item in offered:
            registry.validate_feedback(item, apply_at=at)
        owner.replay.admit_feedback(offered)
        owner.replay.advance(eligible_through=at, apply_at=at)


def run():
    """Return the complete public synthetic32-child pending-frontier receipt.

    Returns
    -------
    dict
        Actual95 decisions,48 applied updates,47 pending labels and95 original
        recipe producers, with declared roots/bindings checked. These synthetic
        mechanics establish no predictive advantage or physical CI execution.
    """
    layout = collection_layout()
    owners, values = pending_values(layout)
    reports = tuple(item.inspect(owner) for item, owner in zip(layout.inspections, owners, strict=True))
    DeclaredRootGuard(layout.ownership).check(
        RootInventory(
            owners[-1].recipe_bindings,
            tuple(root for report in reports for root in report.roots),
        )
    )
    states = [item.value.host_state for item in values[:-2]]
    return {
        "evidence": "public-synthetic-ci-registry",
        "children": len(states),
        "owners": len(values),
        "D": sum(item.decision_count for item in states),
        "U": sum(item.update_count for item in states),
        "pending": sum(len(item.pending) for item in states),
        "origins": len(values[-1].value.executions),
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
