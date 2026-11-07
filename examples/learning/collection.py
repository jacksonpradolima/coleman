"""Compose 32 actual shipped mixtures through the existing owned state store.

The authored streams qualify operational independence and complete recovery.
They establish neither independent observations, pooling, efficacy nor fees.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from coleman.core.actions import MixtureAction
from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.learning._controlled_state import controlled_binding
from coleman.learning.contracts import ReferenceConfig, ReferencePayload
from coleman.learning.control_contracts import ControlledConfig, MovementPeriod
from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectMixture, DirectMixtureState, DirectOptions
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.learning.reference import _implementation as reference_implementation
from coleman.learning.replay import ReplayMixture
from coleman.learning.schedule import ControlledDirect, ControlledDirectState, ControlSchedule
from coleman.objectives.bindings import DefinitionBinding
from coleman.objectives.controls import L1MovementPenalty, MovementAllowance, MovementLimit
from coleman.registry.composition import RecipeExecution, RecipeFeedback, RegistryCheckpoint, require_recipe_eligibility
from coleman.registry.composition import RegistryOwner as CheckpointOwner
from coleman.registry.definitions import Catalogue, MemberDefinition, RecipeBinding
from coleman.registry.lifecycle import LifecycleFact, PopulationPolicy, RegistryState
from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import (
    DeclaredRootGuard,
    OwnershipDeclaration,
    ResourceClaim,
    ResourceRole,
    RootBinding,
    RootInventory,
)
from coleman.registry.state import LiveOwner, OwnedBindings, OwnerInspection
from coleman.replay.contracts import ReplayPolicy
from coleman.replay.retained import RetainedReplay, ordinal_decision_id
from coleman.replay.retained_contracts import OrdinalRetention, RetainedContinuation
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue, child_frontier
from coleman.state.contracts import StateBinding
from coleman.state.storage_contracts import CollectionDefinition, ComponentSpec, ExecutionCompatibility

DECISIONS_PER_CHILD = 4
PREFIX_UPDATES = 2


def load_ci():
    """Load the explicitly selected public learning example, without discovery.

    Returns
    -------
    module
        The fixed sibling ci.py, or its installed copied learning_ci.py name.
    """
    name = "authored_learning_collection_ci"
    if name not in sys.modules:
        copied = Path(__file__).with_name("learning_ci.py")
        source = copied if copied.is_file() else Path(__file__).with_name("ci.py")
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError("The explicitly selected public learning CI source is required")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def reference(name: str) -> Reference:
    """Declare one exact local version-one identity.

    Parameters
    ----------
    name : str
        Explicit logical identity.

    Returns
    -------
    Reference
        Frozen identity without lookup.
    """
    return Reference(id=name, version="1")


EPOCH = reference("authored-shipped-learning-collection-epoch")


def implementation() -> DefinitionBinding:
    """Bind the exact public collection source at its outer metadata boundary.

    Returns
    -------
    DefinitionBinding
        Complete source-content identity, without constructing a learner.
    """
    return DefinitionBinding(
        reference=reference("authored-shipped-learning-collection"),
        definition_fingerprint="sha256:" + sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def _configuration(child: int) -> ReferenceConfig | ControlledConfig:
    """Select three actual references and positive-penalty moving/holding laws."""
    ci = load_ci()
    kind = child % 5
    if kind < 3:
        return ci.reference_configuration(
            ("hedge-constant", "hedge-anytime-cumulative", "fixed-share-excluding-origin")[kind]
        )
    config = ci.controlled_configuration(cadence=1 + (child // 5) % 2, strength=0.01 if kind == 3 else 1.0)
    movement = config.controls.movement
    if movement is None:
        raise ValueError("The selected public controlled law requires explicit movement")
    exact = MovementAllowance.model_validate(
        movement.model_dump(mode="python")
        | {
            "reference": reference("authored-learning-period-variation"),
            "scope": "per_interval",
            "limits": tuple(MovementLimit(member=limit.member, maximum=2.0) for limit in movement.limits),
        }
    )
    controls = type(config.controls).model_validate(config.controls.model_dump(mode="python") | {"movement": exact})
    return ControlledConfig.model_validate(
        config.model_dump(mode="python")
        | {"controls": controls, "period_clock": reference("authored-learning-periods")}
    )


@dataclass(frozen=True)
class ChildSpec:
    """Retain one immutable child law, schedule and complete outer codec.

    Attributes
    ----------
    child : int
        Explicit child index from zero through31.
    configuration : ReferenceConfig or ControlledConfig
        Fixed complete actual mathematical law, admitted before construction.
    schedule : ControlSchedule or None
        Explicit hard-budget period for controlled round cadence.
    binding : StateBinding
        Outer complete continuation interpretation, including all fixed inputs.
    """

    child: int
    configuration: ReferenceConfig | ControlledConfig
    schedule: ControlSchedule | None
    binding: StateBinding

    @property
    def identity(self) -> Reference:
        """Return the child's stable task and owner identity."""
        return reference(f"learning-child-{self.child:02}")

    @property
    def method_binding(self) -> StateBinding:
        """Read actual pure SDK metadata without constructing a prototype owner."""
        config = self.configuration
        if isinstance(config, ControlledConfig):
            return controlled_binding(config, self.identity)
        return StateBinding(
            owner=self.identity,
            implementation=reference_implementation(config.profile),
            state_codec=reference("coleman-reference-learning-state"),
            configuration_fingerprint=config.fingerprint,
        )

    @property
    def state_type(self) -> type[CoreContract]:
        """Return the trusted concrete host model without generic erasure."""
        if isinstance(self.configuration, ControlledConfig):
            return ControlledDirectState
        return DirectMixtureState[ReferencePayload]


def _spec(child: int, source: DefinitionBinding) -> ChildSpec:
    """Build pure complete configuration; no factory or inspector is invoked."""
    config = _configuration(child)
    ci = load_ci()
    schedule = None
    if isinstance(config, ControlledConfig):
        if config.period_clock is None:
            raise ValueError("Controlled collection requires its explicit movement-period clock")
        schedule = ControlSchedule(
            schema_version=1,
            calendar=None,
            periods=(
                MovementPeriod(
                    schema_version=1,
                    calendar=config.period_clock,
                    interval=reference("authored-learning-budget-period"),
                    start=ci.NOW,
                    end=ci.NOW + timedelta(minutes=20),
                ),
            ),
        )
    binding = StateBinding(
        owner=reference(f"learning-child-{child:02}"),
        implementation=source,
        state_codec=reference("authored-shipped-mixture-retained-continuation"),
        configuration_fingerprint=fingerprint(
            {
                "child": child,
                "configuration": config.model_dump(mode="json"),
                "schedule": None if schedule is None else schedule.model_dump(mode="json"),
                "profile": "actual-shipped-owned-mixtures-v1",
            }
        ),
    )
    return ChildSpec(child, config, schedule, binding)


def _application(spec: ChildSpec) -> tuple[Any, Any]:
    """Construct actual independent kernels and their real producer bridges."""
    config, ci = spec.configuration, load_ci()
    options = DirectOptions(task=spec.identity, objective=ci.OBJECTIVE, horizon=ci.HORIZON)
    if isinstance(config, ControlledConfig):
        if spec.schedule is None:
            raise ValueError("Controlled collection child requires its complete period schedule")
        method = ControlledHedge(config, owner=spec.identity)
        return method, ControlledDirect(method, spec.schedule, options)
    classes = {
        "hedge-constant": Hedge,
        "hedge-anytime-cumulative": AnytimeHedge,
        "fixed-share-excluding-origin": FixedShare,
    }
    method = classes[config.profile](config, owner=spec.identity)
    return method, DirectMixture(method, task=spec.identity, objective=ci.OBJECTIVE, horizon=ci.HORIZON)


def _policy(spec: ChildSpec, application) -> ReplayPolicy:
    """Bind the actual application's full-panel capability and child stream."""
    return ReplayPolicy(
        schema_version=1,
        stream=spec.identity,
        lineage=reference(spec.identity.id + "-live"),
        parent=None,
        source=load_ci().SOURCE,
        capability=application.capability,
        delivery="delayed",
        specialists=False,
    )


class ChildOwner:
    """Own one actual learner, producer, replay wrapper and retained session.

    Parameters
    ----------
    specification : ChildSpec
        Complete immutable child configuration and concrete interpretation.

    Notes
    -----
    Every mutable object in this chain is individually declared as a root.
    All sufficient mathematical, producer and delivery state is in the complete
    retained continuation; no dummy RNG or separate field root is invented.
    """

    def __init__(self, specification: ChildSpec):
        """Create one actual fresh chain, ready for explicit state restoration."""
        self.specification = specification
        self.method, self.application = _application(specification)
        self.consumer = ReplayMixture(self.application)
        self.session = RetainedReplay(
            self.consumer,
            _policy(specification, self.application),
            state_type=self.application.state_type,
            state_codec=reference("authored-learning-replay-host-state"),
            retention=OrdinalRetention(profile="quiescent-ordinal-v1", source_epoch=EPOCH, maximum_tail_records=512),
        )
        self.session.admit_history(load_ci().history())

    @property
    def producer(self):
        """Return the actual inner producer, without duplicating its root claim."""
        return self.application.bridge if isinstance(self.application, ControlledDirect) else self.application

    def export_state(self):
        """Return the actual complete mathematical/producer/delivery continuation."""
        return self.session.export_continuation()

    def restore_state(self, state, /) -> None:
        """Rebuild a fresh actual chain and replace it only after complete restore.

        Parameters
        ----------
        state : RetainedContinuation
            Exact concrete child continuation, including the complete fixed law.
        """
        method, application = _application(self.specification)
        consumer = ReplayMixture(application)
        session = RetainedReplay.from_continuation(
            consumer,
            state,
            state_type=application.state_type,
            state_codec=reference("authored-learning-replay-host-state"),
        )
        self.method, self.application, self.consumer, self.session = method, application, consumer, session


class RegistryOwner(CheckpointOwner):
    """Retain original actual mixture producers under the fixed recipe laws."""

    @property
    def recipe_bindings(self) -> tuple[RecipeBinding, ...]:
        """Read fixed recipe metadata without exporting the growing checkpoint."""
        return self._state.ownership.recipes

    def validate_feedback(self, offered, owner: ChildOwner, *, apply_at) -> None:
        """Validate the original producer before applying its delayed panel.

        Parameters
        ----------
        offered : ReplayFeedback
            Complete original typed panel and producer join.
        owner : ChildOwner
            Actual original child receiver, never a replacement recipe.
        apply_at : datetime
            Explicit actual processing instant.
        """
        original = self._state.for_execution(offered.batch.decision_fingerprint)
        RecipeFeedback(schema_version=1, producer_fingerprint=original.fingerprint, batch=offered.batch).validate_for(
            original, owner.application.capability, apply_at=apply_at
        )

    def emit(self, owner: ChildOwner):
        """Capture recipe eligibility before one actual shipped-mixture emission.

        Parameters
        ----------
        owner : ChildOwner
            Actual privately owned child chain.

        Returns
        -------
        ReplayDecisionRecord
            Original typed offline-identity mixture producer.
        """
        spec = owner.specification
        ordinal = owner.session.state.decision_count
        authored = load_ci().opportunity(owner.application, ordinal, ordinal)
        request = type(authored.request).model_validate(
            authored.request.model_dump(mode="python")
            | {"task": spec.identity, "decision_id": ordinal_decision_id(EPOCH, ordinal)}
        )
        offered = type(authored).model_validate(authored.model_dump(mode="python") | {"request": request})
        recipe = next(item for item in self.recipe_bindings if item.child == spec.identity)
        registry = next(item for item in self._state.registries if item.policy.task == spec.identity)
        snapshot = registry.snapshot(ordinal, cutoff=request.timing.information_cutoff)
        require_recipe_eligibility(snapshot, recipe, request)
        record = owner.session.emit(owner.session.prepare(offered), execution="offline_identity_v1")
        if not isinstance(record.executed.action, MixtureAction):
            raise ValueError("The public collection requires an actual emitted mixture")
        self.record(RecipeExecution(schema_version=1, snapshot=snapshot, recipe=recipe, executed=record.executed))
        return record


def _registry(spec: ChildSpec):
    """Declare a one-recipe expert population distinct from its inner experts."""
    config, ci = spec.configuration, load_ci()
    leaf = MemberDefinition(
        schema_version=1,
        kind="expert",
        member=reference(spec.identity.id + "-shipped-method"),
        implementation=spec.method_binding.implementation,
        configuration_fingerprint=config.fingerprint,
    )
    controlled = config.controls if isinstance(config, ControlledConfig) else None
    recipe = RecipeBinding(
        schema_version=1,
        child=spec.identity,
        task=spec.identity,
        expert=leaf.binding,
        state_epoch=reference(spec.identity.id + "-epoch"),
        owners=(spec.binding,),
        objective=ci.OBJECTIVE,
        target=None if controlled is None or controlled.target is None else controlled.target.binding,
        loss_transform=config.transform,
        controls=None if controlled is None else config.binding,
    )
    registry = RegistryState(
        schema_version=1,
        catalogue=Catalogue(
            schema_version=1, reference=reference(spec.identity.id + "-catalogue"), definitions=(leaf,)
        ),
        policy=PopulationPolicy(
            population=reference(spec.identity.id + "-population"), task=spec.identity, kind="expert", clock="timed"
        ),
        limits=RegistryLimits(),
    ).append(
        LifecycleFact(
            schema_version=1,
            event_id="introduce",
            sequence=0,
            member=leaf.member,
            definition_fingerprint=leaf.fingerprint,
            operation="introduce",
            effective_from_round=0,
            admitted_before_round=0,
            known_at=ci.NOW,
            reason="authored-shipped-method-collection",
            state_epoch=recipe.state_epoch,
            predecessor_epoch=None,
            disposition=None,
        )
    )
    return recipe, registry


def _roots(owner: ChildOwner) -> dict[str, object]:
    """Check actual pointer joins and expose each distinct mutable root once."""
    if owner.application.learner is not owner.method or owner.consumer.application is not owner.application:
        raise ValueError("The actual mathematical/producer/replay ownership chain disagrees")
    if owner.session._host is not owner.consumer:  # pylint: disable=protected-access
        raise ValueError("The actual retained session has another replay consumer")
    if owner.method.binding != owner.specification.method_binding:
        raise ValueError("Actual shipped kernel differs from its fixed declared law")
    if (
        owner.session.state.policy.model_dump_json()
        != _policy(owner.specification, owner.application).model_dump_json()
    ):
        raise ValueError("Actual session policy differs from its child and application")
    roots: dict[str, object] = {
        "owner": owner,
        "statistics": owner.method,
        "environment": owner.application,
        "input": owner.consumer,
        "session": owner.session,
    }
    if isinstance(owner.application, ControlledDirect):
        if owner.application.bridge.learner is not owner.method:
            raise ValueError("The scheduled producer has another mathematical learner")
        roots["pending"] = owner.application.bridge
    return roots


def _inspect_child(recipe, claims, owner):
    """Inspect actual live roots without constructing, restoring or exporting."""
    if owner.specification.identity != recipe.child:
        raise ValueError("Collection owner belongs to another declared child")
    roots = _roots(owner)
    return RootInventory((recipe,), tuple(RootBinding(claim, roots[claim.role]) for claim in claims))


def _inspect_registry(claim, owner):
    """Retain the real registry owner as one shared outer mutable root."""
    return RootInventory(owner.recipe_bindings, (RootBinding(claim, owner),))


@dataclass(frozen=True)
class Layout:
    """Retain one complete finite set of actual method components and roots.

    Attributes
    ----------
    definition, bindings : CollectionDefinition, tuple of ComponentBinding
        Existing local-store declaration and actual trusted codecs/factories.
    ownership, origin : OwnershipDeclaration, RegistryCheckpoint
        Fixed complete root declarations and original registry state.
    inspections : tuple of OwnerInspection
        Trusted callbacks used by explicit standalone checks or accepted staging.
    specifications : tuple of ChildSpec
        Complete canonical child laws and supplied clock plans.
    """

    definition: CollectionDefinition
    bindings: tuple[ComponentBinding[Any, Any], ...]
    ownership: OwnershipDeclaration
    origin: RegistryCheckpoint
    inspections: tuple[OwnerInspection[Any, Any], ...]
    specifications: tuple[ChildSpec, ...]

    def owned(self, *, live: tuple[Any, ...] = ()) -> OwnedBindings:
        """Create lazy accepted-staging guards, retaining any actual live owners.

        Parameters
        ----------
        live : tuple, optional
            Complete retained actual owners in canonical component order.

        Returns
        -------
        OwnedBindings
            Existing optional wrapper; construction invokes no callbacks.
        """
        retained = (
            tuple(LiveOwner(item, owner) for item, owner in zip(self.inspections, live, strict=True)) if live else ()
        )
        return OwnedBindings(self.ownership, self.inspections, live=retained)

    def compatibility(self) -> ExecutionCompatibility:
        """Capture actual installed base and explicitly selected public sources.

        Returns
        -------
        ExecutionCompatibility
            Existing metadata boundary, outside pure mathematical operations.
        """
        from coleman.spec.selected_runtime import SelectedInstalledRuntime  # pylint: disable=import-outside-toplevel

        ci = load_ci()
        if ci.__file__ is None:
            raise ValueError("The explicit public CI source must have a source path")
        source = DefinitionBinding(
            reference=reference("authored-learning-ci-source"),
            definition_fingerprint="sha256:" + sha256(Path(ci.__file__).read_bytes()).hexdigest(),
        )
        methods = {
            item.method_binding.implementation.reference.id: item.method_binding.implementation
            for item in self.specifications
        }
        return ExecutionCompatibility(
            schema_version=1,
            runtime=SelectedInstalledRuntime.capture(extras=()),
            owner_implementations=(implementation(), source, *methods.values()),
        )


def collection_layout(children: tuple[int, ...] = tuple(range(32))) -> Layout:
    """Declare actual shipped-mixture ownership without running any factory.

    Parameters
    ----------
    children : tuple of int, optional
        Explicit unique indices from zero through31; default all32.

    Returns
    -------
    Layout
        At full size,33 components and173 actual distinct mutable roots.

    Raises
    ------
    ValueError
        Invalid child inventory or incompatible complete declarations.
    """
    if type(children) is not tuple or not children or len(set(children)) != len(children):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Learning collection requires an explicit unique nonempty child tuple")
    if any(type(child) is not int or not 0 <= child < 32 for child in children):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Learning collection supports child indices zero through31")
    source = implementation()
    specs = tuple(_spec(child, source) for child in sorted(children))
    pairs = tuple(_registry(spec) for spec in specs)
    entries = tuple(_child_component(spec, pair[0]) for spec, pair in zip(specs, pairs, strict=True))
    bindings = tuple(item[0] for item in entries)
    inspections = tuple(item[1] for item in entries)
    resources = tuple(claim for item in entries for claim in item[2])
    components = tuple(item[3] for item in entries)
    ownership, checkpoint, registry, inspection, component = _registry_component(specs, pairs, resources, source)
    definition = CollectionDefinition(
        schema_version=1,
        reference=reference("authored-shipped-learning-definition"),
        session=reference("authored-shipped-learning-session"),
        source_epoch=EPOCH,
        children=tuple(spec.identity for spec in specs),
        components=(*components, component),
    )
    return Layout(definition, (*bindings, registry), ownership, checkpoint, (*inspections, inspection), specs)


def _child_component(spec: ChildSpec, recipe: RecipeBinding):
    """Bind one complete continuation to all its actual mutable object roots."""
    state_type = cast(type[RetainedContinuation[Any]], RetainedContinuation.__class_getitem__(spec.state_type))
    codec = ComponentCodec(spec.binding, state_type)
    binding = ComponentBinding(codec, partial(ChildOwner, spec), partial(child_frontier, child=spec.identity))
    # Each role points at a complete actual object, never invented field roots.
    roles: tuple[ResourceRole, ...] = ("owner", "statistics", "environment", "input", "session")
    if isinstance(spec.configuration, ControlledConfig):
        roles += ("pending",)
    claims = tuple(
        ResourceClaim(
            resource=reference(spec.identity.id + "-" + role),
            owner=spec.binding,
            child=spec.identity,
            role=role,
            access="owned",
        )
        for role in roles
    )
    inspection = OwnerInspection(binding, partial(_inspect_child, recipe, claims))
    component = ComponentSpec(
        binding=spec.binding,
        child=spec.identity,
        roles=("algorithm", "learner_pending", "delivery"),
        frontier_profile="replay-ordinal-v2",
    )
    return binding, inspection, claims, component


def _registry_component(specs, pairs, resources, source):
    """Add the actual outer producer registry without another delivery engine."""
    registry_binding = StateBinding(
        owner=reference("zz-learning-registry"),
        implementation=source,
        state_codec=reference("authored-shipped-learning-registry"),
        configuration_fingerprint=fingerprint([spec.binding.model_dump(mode="json") for spec in specs]),
    )
    claim = ResourceClaim(
        resource=reference("zz-learning-registry-root"),
        owner=registry_binding,
        child=None,
        role="owner",
        access="owned",
    )
    ownership = OwnershipDeclaration(
        schema_version=1,
        recipes=tuple(pair[0] for pair in pairs),
        shared_owners=(registry_binding,),
        resources=(*resources, claim),
    )
    checkpoint = RegistryCheckpoint(
        schema_version=1,
        registries=tuple(pair[1] for pair in pairs),
        ownership=ownership,
        limits=RegistryLimits(maximum_bytes=16 * 1024**2),
    )
    registry = ComponentBinding(
        ComponentCodec(registry_binding, RegistryCheckpoint), partial(RegistryOwner, checkpoint), lambda _state: None
    )
    inspection = OwnerInspection(registry, partial(_inspect_registry, claim))
    component = ComponentSpec(binding=registry_binding, child=None, roles=("delivery",), frontier_profile="none")
    return ownership, checkpoint, registry, inspection, component


def _feedback(registry: RegistryOwner, owner: ChildOwner, records, visible: float) -> None:
    """Apply complete original bounded outcomes through their actual producers."""
    ci = load_ci()
    at = ci.NOW + timedelta(minutes=visible)
    offered = []
    for record in records:
        ordinal = record.executed.proposal.request.round
        child = owner.specification.child
        losses = tuple(((child + factor * ordinal + factor) % 4) / 4 for factor in (1, 2, 3))
        item = ci.feedback(
            record,
            owner.specification.configuration,
            losses,
            visible=visible,
            delay=owner.session.state.decision_count - ordinal - 1,
        )
        registry.validate_feedback(item, owner, apply_at=at)
        offered.append(item)
    owner.session.admit_feedback(tuple(offered))
    owner.session.advance(eligible_through=at, apply_at=at)


def _inventory(layout: Layout, owners: tuple[Any, ...]) -> RootInventory:
    """Inspect the explicit standalone owners without constructing another root."""
    reports = tuple(item.inspect(owner) for item, owner in zip(layout.inspections, owners, strict=True))
    return RootInventory(owners[-1].recipe_bindings, tuple(root for report in reports for root in report.roots))


def pending_values(layout: Layout, *, order: tuple[int, ...] | None = None, isolated: bool = False):
    """Create actual running32-child values with two real pending panels each.

    Parameters
    ----------
    layout : Layout
        Fixed actual child codecs, laws and declarations.
    order : tuple of int or None, optional
        Exact supplied child scheduling permutation.
    isolated : bool, optional
        Finish each child's emission prefix before another child emits.

    Returns
    -------
    tuple
        Complete actual owners and typed component values; full size has128 D,
        64 U and64 true pending outcomes, with128 original registry producers.
    """
    indices = tuple(spec.child for spec in layout.specifications)
    selected = indices if order is None else order
    if type(selected) is not tuple or len(selected) != len(indices) or set(selected) != set(indices):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Learning schedule must be an exact child permutation")
    owners = {spec.child: ChildOwner(spec) for spec in layout.specifications}
    registry = RegistryOwner(layout.origin)
    complete = (*(owners[child] for child in indices), registry)
    guard = DeclaredRootGuard(layout.ownership)
    guard.check(_inventory(layout, complete))
    schedule = (
        tuple(child for child in selected for _ in range(DECISIONS_PER_CHILD))
        if isolated
        else tuple(child for _ in range(DECISIONS_PER_CHILD) for child in selected)
    )
    for child in schedule:
        record = registry.emit(owners[child])
        ordinal = record.executed.proposal.request.round
        if ordinal < PREFIX_UPDATES:
            _feedback(registry, owners[child], (record,), ordinal + 0.25)
    guard.check(_inventory(layout, complete))
    values = tuple(
        ComponentValue(binding.codec, owner.export_state())
        for binding, owner in zip(layout.bindings, complete, strict=True)
    )
    return complete, values


def complete_pending(layout: Layout, owners: tuple[Any, ...]) -> None:
    """Complete the original pending block without emitting another decision.

    Parameters
    ----------
    layout : Layout
        Original fixed declaration.
    owners : tuple
        Complete fresh/restored actual owners with registry last.
    """
    if len(owners) != len(layout.bindings):
        raise ValueError("Complete pending requires every actual component owner")
    registry = owners[-1]
    for owner in owners[:-1]:
        fingerprints = {item.fingerprint for item in owner.producer.export_state().pending}
        records = tuple(
            record for record in owner.session.state.decisions if record.executed.fingerprint in fingerprints
        )
        _feedback(registry, owner, records, DECISIONS_PER_CHILD + 1)


def summary(layout: Layout, owners: tuple[Any, ...]) -> dict[str, object]:
    """Report actual distinct mathematical, decision, pending and release clocks.

    Parameters
    ----------
    layout : Layout
        Complete fixed declarations.
    owners : tuple
        Actual complete child/registry owners.

    Returns
    -------
    dict
        Operational counts and explicit moving/optimal-hold ledger evidence.
    """
    children = owners[:-1]
    controllers = tuple(owner.method for owner in children if isinstance(owner.method, ControlledHedge))
    moving = tuple(method for method in controllers if _strength(method) == 0.01)
    holding = tuple(method for method in controllers if _strength(method) == 1.0)
    return {
        "evidence": "authored-shipped-learning-operational-collection",
        "children": len(children),
        "components": len(owners),
        "declared_roots": len(layout.ownership.resources),
        "D": sum(owner.session.state.decision_count for owner in children),
        "U": sum(owner.method.update_count for owner in children),
        "pending": sum(len(owner.producer.export_state().pending) for owner in children),
        "R_controlled": sum(method.release_count for method in controllers),
        "origins": len(owners[-1].export_state().executions),
        "moving_children": len(moving),
        "holding_children": len(holding),
        "moving_exact_usage_positive": all(any(_path_atoms(method)) for method in moving),
        "optimal_holds_spend_zero": all(not any(_path_atoms(method)) for method in holding),
    }


def _strength(method: ControlledHedge) -> float:
    """Require the actual positive-L1 law before classifying a child."""
    output = method.config.controls.output
    if not isinstance(output, L1MovementPenalty):
        raise ValueError("The controlled collection requires its explicit L1 profile")
    return output.strength


def _path_atoms(method: ControlledHedge) -> tuple[int, ...]:
    """Read the actual complete current-period ledger without rounding it."""
    path = method.export_state().payload.exact_path
    if path is None:
        raise ValueError("The running controlled collection requires its admitted exact period ledger")
    return tuple(int(item.atoms_decimal) for item in path.used)


def run() -> dict[str, object]:
    """Return the actual public32-child pending-state mechanics receipt.

    Returns
    -------
    dict
        Actual counters, exact ledgers and complete retained producer counts.
    """
    layout = collection_layout()
    owners, _values = pending_values(layout)
    return summary(layout, owners)


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
