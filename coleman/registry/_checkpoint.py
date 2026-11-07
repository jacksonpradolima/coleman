"""Bounded pure historical joins for optional complete recipe checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coleman.core.identity import Reference
from coleman.registry.definitions import Catalogue, MemberKind, RecipeBinding
from coleman.registry.lifecycle import RegistryState
from coleman.registry.limits import RegistryLimits
from coleman.registry.ownership import OwnershipDeclaration

if TYPE_CHECKING:
    from coleman.registry.composition import RecipeExecution, RegistryCheckpoint


def _checkpoint_tasks(
    registries: tuple[RegistryState, ...], ownership: OwnershipDeclaration
) -> tuple[dict[str, RegistryState], dict[str, RecipeBinding]]:
    """Resolve exact frozen recipes without mixing independent producer profiles."""
    tasks = {registry.policy.task.id: registry for registry in registries}
    if len(tasks) != len(registries):
        raise ValueError("Checkpoint registry tasks require unique logical identities")
    recipes = {recipe.task.id: recipe for recipe in ownership.recipes}
    if len(recipes) != len(ownership.recipes) or set(tasks) != set(recipes):
        raise ValueError("Checkpoint requires exactly one registry per declared recipe task")
    for identity, registry in tasks.items():
        recipe = recipes[identity]
        if registry.producers:
            raise ValueError("Owned-recipe checkpoint cannot mix standalone registered-request seals")
        if (registry.policy.kind, registry.policy.task) != ("expert", recipe.task):
            raise ValueError("Checkpoint recipe and expert-population task differ")
        leaf = registry.catalogue.resolve("expert", recipe.expert.reference)
        if leaf.binding != recipe.expert:
            raise ValueError("Checkpoint recipe differs from its exact immutable expert definition")
    return tasks, recipes


def _checkpoint_origins(
    executions: tuple[RecipeExecution, ...],
    tasks: dict[str, RegistryState],
    recipes: dict[str, RecipeBinding],
    publications: dict[str, Catalogue],
) -> None:
    """Verify original logical producer uniqueness, frozen bindings and frontiers."""
    members: dict[str, frozenset[tuple[MemberKind, Reference]]] = {
        identity: frozenset((leaf.kind, leaf.member) for leaf in publication.definitions)
        for identity, publication in publications.items()
    }
    seen = set()
    for execution in executions:
        if execution.logical_identity in seen:
            raise ValueError("Checkpoint contains duplicate or conflicting logical producers")
        seen.add(execution.logical_identity)
        registry = tasks.get(execution.recipe.task.id)
        if registry is None or recipes[execution.recipe.task.id].to_json() != execution.recipe.to_json():
            raise ValueError("Checkpoint producer differs from its frozen complete recipe binding")
        _historical_origin(registry, execution, publications, members)
    _execution_frontiers(executions)


def _registry_for(checkpoint: RegistryCheckpoint, task_id: str) -> RegistryState:
    """Resolve one exact retained logical task without newest-version inference."""
    for registry in checkpoint.registries:
        if registry.policy.task.id == task_id:
            return registry
    raise ValueError("Unknown checkpoint registry task")


def _raw_fact_count(registry: Any) -> int:
    """Count only supplied flat history before nested contract validation."""
    history = (
        registry.facts
        if isinstance(registry, RegistryState)
        else (registry.get("facts", ()) if isinstance(registry, dict) else ())
    )
    return len(history) if isinstance(history, (tuple, list)) else 0


def _raw_leaf_count(catalogue: Any) -> int:
    """Count flat publication leaves without hashing or copying their values."""
    leaves = (
        catalogue.definitions
        if isinstance(catalogue, Catalogue)
        else (catalogue.get("definitions", ()) if isinstance(catalogue, dict) else ())
    )
    return len(leaves) if isinstance(leaves, (tuple, list)) else 0


def _publication_pool(
    supplied: tuple[Catalogue, ...], registries: tuple[RegistryState, ...], limits: RegistryLimits
) -> dict[str, Catalogue]:
    """Resolve complete retained preimages without inventing old publications."""
    current = _current_publications(registries)
    pool = _normalized_publications(supplied, current, limits)
    _validate_publication_leaves(pool, current)
    return pool


def _current_publications(registries: tuple[RegistryState, ...]) -> dict[Reference, Catalogue]:
    """Require one complete current publication for each catalogue namespace."""
    current: dict[Reference, Catalogue] = {}
    for registry in registries:
        publication = registry.catalogue
        prior = current.get(publication.reference)
        if prior is not None and prior.fingerprint != publication.fingerprint:
            raise ValueError("Current catalogue namespace/version has conflicting complete publications")
        current[publication.reference] = publication
    return current


def _normalized_publications(
    supplied: tuple[Catalogue, ...], current: dict[Reference, Catalogue], limits: RegistryLimits
) -> dict[str, Catalogue]:
    """Bound raw normalization then charge each retained complete preimage once."""
    raw_leaves = sum(len(item.definitions) for item in supplied) + sum(
        len(item.definitions) for item in current.values()
    )
    if raw_leaves > 2 * limits.maximum_checkpoint_work:
        raise ValueError("Checkpoint exceeds its bounded publication normalization work capacity")
    pool: dict[str, Catalogue] = {}
    for publication in supplied:
        identity = publication.fingerprint
        if identity in pool:
            raise ValueError("Checkpoint contains duplicate explicit catalogue publications")
        pool[identity] = publication
    for publication in current.values():
        pool.setdefault(publication.fingerprint, publication)
    if len(pool) > limits.maximum_publications:
        raise ValueError("Checkpoint publication capacity exhausted")
    if sum(len(item.definitions) for item in pool.values()) > limits.maximum_checkpoint_work:
        raise ValueError("Checkpoint exceeds its publication-leaf validation work capacity")
    return pool


def _validate_publication_leaves(pool: dict[str, Catalogue], current: dict[Reference, Catalogue]) -> None:
    """Resolve immutable leaf bytes through bounded maps instead of pair products."""
    leaves = {
        reference: {(leaf.kind, leaf.member): leaf.to_json() for leaf in publication.definitions}
        for reference, publication in current.items()
    }
    for publication in pool.values():
        latest = leaves.get(publication.reference)
        if latest is None:
            raise ValueError("Checkpoint publication has an unknown catalogue namespace/version")
        for leaf in publication.definitions:
            if latest.get((leaf.kind, leaf.member)) != leaf.to_json():
                raise ValueError("Checkpoint publication conflicts with immutable retained leaves")


def _historical_origin(
    registry: RegistryState,
    execution: RecipeExecution,
    publications: dict[str, Catalogue],
    members: dict[str, frozenset[tuple[MemberKind, Reference]]],
) -> None:
    """Join an original complete snapshot to retained immutable history/leaves."""
    snapshot = execution.snapshot
    if snapshot.policy != registry.policy or len(snapshot.facts) > len(registry.facts):
        raise ValueError("Recipe producer belongs to another registry policy or history")
    if snapshot.catalogue.reference != registry.catalogue.reference:
        raise ValueError("Recipe producer belongs to another catalogue namespace/version")
    publication = publications.get(snapshot.catalogue.definition_fingerprint)
    if publication is None or publication.reference != snapshot.catalogue.reference:
        raise ValueError("Recipe producer requires its exact retained catalogue publication")
    if any(left.to_json() != right.to_json() for left, right in zip(snapshot.facts, registry.facts, strict=False)):
        raise ValueError("Recipe producer does not retain an exact accepted lifecycle prefix")
    if any(fact.admitted_before_round <= snapshot.round for fact in registry.facts[len(snapshot.facts) :]):
        raise ValueError("Lifecycle fact was admitted into an already-produced recipe round")
    for leaf in snapshot.definitions:
        # Prefix facts already pin leaf digests; the pool verified immutable
        # content. Here the original publication must actually contain the leaf.
        if (leaf.kind, leaf.member) not in members[snapshot.catalogue.definition_fingerprint]:
            raise ValueError("Unknown catalogue member in retained original publication")


def _execution_frontiers(executions: tuple[RecipeExecution, ...]) -> None:
    """Preserve the writer's strict per-task round and known-cutoff contract."""
    tasks: dict[str, list[RecipeExecution]] = {}
    for execution in executions:
        tasks.setdefault(execution.recipe.task.id, []).append(execution)
    for values in tasks.values():
        ordered = sorted(values, key=lambda item: item.snapshot.round)
        rounds = [item.snapshot.round for item in ordered]
        if len(rounds) != len(set(rounds)):
            raise ValueError("Checkpoint task opportunities require distinct increasing rounds")
        prefixes = [len(item.snapshot.facts) for item in ordered]
        if any(left > right for left, right in zip(prefixes, prefixes[1:], strict=False)):
            raise ValueError("Checkpoint task accepted history prefixes cannot rewind")
        cutoffs = [item.snapshot.cutoff for item in ordered if item.snapshot.cutoff is not None]
        if any(left > right for left, right in zip(cutoffs, cutoffs[1:], strict=False)):
            raise ValueError("Checkpoint task knowledge cutoffs cannot rewind")


def _registry_successor(previous: RegistryState, current: RegistryState) -> None:
    """Require causal extension while preserving selected interpretation limits."""
    if previous.policy != current.policy or previous.limits != current.limits:
        raise ValueError("Registry extension cannot change its fixed policy or selected limits")
    if previous.catalogue.reference != current.catalogue.reference:
        raise ValueError("Registry extension cannot change its catalogue namespace/version")
    for leaf in previous.catalogue.definitions:
        if current.catalogue.resolve(leaf.kind, leaf.member).to_json() != leaf.to_json():
            raise ValueError("Registry extension changes an immutable historical definition")
    if len(current.facts) < len(previous.facts) or any(
        left.to_json() != right.to_json() for left, right in zip(previous.facts, current.facts, strict=False)
    ):
        raise ValueError("Registry extension cannot rewind or replace accepted lifecycle facts")
