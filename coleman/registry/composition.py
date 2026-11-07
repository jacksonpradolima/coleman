"""Bind eligible recipes to their own actions and complete owned checkpoints.

A recipe's expert identity differs from the action universe it controls. These
optional envelopes preserve existing inner execution bytes. They add no learner,
delivery engine, automatic state transfer or physical execution guarantee.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from coleman.core.capabilities import Capability, validate_feedback
from coleman.core.decisions import DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch
from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp
from coleman.registry._checkpoint import (
    _checkpoint_origins,
    _checkpoint_tasks,
    _publication_pool,
    _raw_fact_count,
    _raw_leaf_count,
    _registry_for,
    _registry_successor,
)
from coleman.registry.definitions import Catalogue, MemberDefinition, RecipeBinding
from coleman.registry.lifecycle import EligibilitySnapshot, RegistryState
from coleman.registry.limits import RegistryContract, RegistryLimits, admit_bytes
from coleman.registry.ownership import OwnershipDeclaration


def require_recipe_eligibility(snapshot: EligibilitySnapshot, recipe: RecipeBinding, request: DecisionRequest) -> None:
    """Reject ineligible recipes before preparing or invoking their inner host.

    Parameters
    ----------
    snapshot : EligibilitySnapshot
        Complete predecision expert-population view with supplied causal clocks.
    recipe : RecipeBinding
        Exact selected expert/epoch, task, objective and declared configuration.
    request : DecisionRequest
        Inner action request; its action members can differ from expert IDs.

    Raises
    ------
    ValueError
        Inactive, unavailable, retired, unknown/version/epoch mismatch, or wrong
        task/round/objective/cutoff. No host or mutable owner is touched.

    Notes
    -----
    Revalidates all three inputs. This checks supplied eligibility and exact
    bindings, not authenticity, unseen controls or a selection probability.
    The actual emitted implementation reference is checked by RecipeExecution.
    """
    _eligible_recipe(
        EligibilitySnapshot.model_validate(snapshot),
        RecipeBinding.model_validate(recipe),
        DecisionRequest.model_validate(request),
    )


def _eligible_recipe(
    snapshot: EligibilitySnapshot, recipe: RecipeBinding, request: DecisionRequest
) -> MemberDefinition:
    """Share already-validated preconditions with the actual producer envelope."""
    if snapshot.policy.kind != "expert" or snapshot.policy.task != recipe.task:
        raise ValueError("Recipe execution requires its exact expert-population task")
    if (request.task, request.round) != (recipe.task, snapshot.round):
        raise ValueError("Recipe execution differs from the producing task or round")
    cutoff = None if request.timing is None else request.timing.information_cutoff
    if (cutoff, request.timing is not None) != (snapshot.cutoff, snapshot.policy.clock == "timed"):
        raise ValueError("Recipe execution differs from the fixed clock profile or cutoff")
    rows = [row for row in snapshot.members if row.member == recipe.expert.reference]
    if len(rows) != 1:
        raise ValueError("Recipe execution requires one exact registered expert version")
    row = rows[0]
    if (row.status, row.definition_fingerprint, row.state_epoch) != (
        "active",
        recipe.expert.definition_fingerprint,
        recipe.state_epoch,
    ):
        raise ValueError("Recipe execution requires its exact active expert definition and epoch")
    if request.objective != recipe.objective:
        raise ValueError("Recipe execution differs from its declared objective")
    return next(item for item in snapshot.definitions if item.member == row.member)


class RecipeExecution(RegistryContract):
    """Bind a selected eligible recipe to its unchanged inner action execution.

    Attributes
    ----------
    snapshot : EligibilitySnapshot
        Expert-population eligibility captured before the inner decision, with
        the same task, round and information cutoff. Source clocks are supplied
        attestations; this envelope does not authenticate their chronology.
    recipe : RecipeBinding
        Exact active expert/epoch, objective and complete declared state owners.
    executed : ExecutedDecision
        Original inner execution. Its action members can differ from the expert
        population: an eligible CI recipe may itself choose test plan A or B.

    Raises
    ------
    ValueError
        Wrong population/task/round/cutoff, expert definition, epoch, objective
        or implementation reference.

    Notes
    -----
    This is recipe eligibility, not a sampled recipe-selection probability,
    expert mixture, proof of execution, or verification of unseen controls.
    Optional checkpoint/replay layers retain the original outer producer join.
    Construction performs no host call, state transfer, registration or I/O.
    """

    snapshot: EligibilitySnapshot
    recipe: RecipeBinding
    executed: ExecutedDecision

    @model_validator(mode="after")
    def _join(self) -> Self:
        """Keep recipe eligibility separate from its actual controlled actions."""
        request = self.executed.proposal.request
        leaf = _eligible_recipe(self.snapshot, self.recipe, request)
        if leaf.implementation.reference != self.executed.proposal.implementation:
            raise ValueError("Recipe execution differs from its declared implementation reference")
        return self

    @property
    def logical_identity(self) -> str:
        """Return unchanged inner task/opportunity identity, distinct from content.

        Returns
        -------
        str
            Existing core logical digest. Complete retries compare outer content
            too, including recipe/epoch, cutoff and the inner logical round.
        """
        return self.executed.proposal.request.logical_identity


class RecipeFeedback(RegistryContract):
    """Retain feedback for one exact original eligible-recipe execution.

    Attributes
    ----------
    producer_fingerprint : str
        Complete original RecipeExecution digest, including old epoch/controls.
    batch : FeedbackBatch
        Unchanged inner feedback with its own exact executed-decision digest.

    Notes
    -----
    Retiring/restarting a recipe never redirects feedback to a replacement.
    The host explicitly decides whether a retained old owner applies or audits
    it. This envelope performs no update, delivery, imputation or I/O.
    """

    producer_fingerprint: Fingerprint
    batch: FeedbackBatch

    def validate_for(
        self, producer: RecipeExecution, capability: Capability, *, apply_at: Timestamp | None = None
    ) -> None:
        """Check original outer identity then existing actual feedback semantics.

        Parameters
        ----------
        producer : RecipeExecution
            Original immutable producer, never today's replacement recipe.
        capability : Capability
            Actual inner consumer's coverage, quantity, missingness and delay
            requirements. Eligible recipe status cannot satisfy these itself.
        apply_at : datetime or None, optional
            Explicit actual application instant for timed feedback validation.

        Raises
        ------
        ValueError
            Outer/inner producer mismatch or any existing feedback invariant.

        Notes
        -----
        No current catalogue lookup, learner update or durable deduplication.
        """
        current = RecipeFeedback.model_validate(self)
        origin = RecipeExecution.model_validate(producer)
        if current.producer_fingerprint != origin.fingerprint:
            raise ValueError("Recipe feedback requires its original outer producer")
        validate_feedback(origin.executed, current.batch, capability, apply_at=apply_at)


class RegistryCheckpoint(RegistryContract):
    """Retain eligibility, frozen ownership and complete original recipe joins.

    Attributes
    ----------
    profile : {'owned-recipes-v1'}, optional
        Complete independent-recipe origins. Standalone registered-request seals
        are a separate producing profile and cannot be mixed into these states.
    registries : tuple of RegistryState
        One through32 distinct expert-population tasks, matching the declared
        recipes. Each retains its bounded immutable definitions and history.
    ownership : OwnershipDeclaration
        Complete fixed mathematical/resource inventory for this collection.
    executions : tuple of RecipeExecution, optional
        Original producers, canonical by logical identity. These are origins,
        not a global execution-order log; owned host/domain state retains path
        order. Retained producers are never silently evicted.
    catalogues : tuple of Catalogue, optional
        Deduplicated complete publication preimages for original snapshots.
        Current publications are included automatically; historical publications
        remain retained after unrelated definition growth. At most the selected
        maximum_publications publications; their leaf/byte work is coupled too.
    limits : RegistryLimits, optional
        Coupled complete encoded-byte and producer-count admission. Defaults
        are2MiB/4096 producers; hard16MiB/8192. Full origins count actual bytes,
        so simultaneous Cartesian maxima need not fit.

    Notes
    -----
    This static-ownership profile permits prospective lifecycle history growth.
    Changing recipe controls, epoch, owner/code/configuration or layout requires
    an explicit new compatible collection/restart with retained old producers;
    it is not an implicit codec migration. Direct recipe contracts remain
    independently usable. Existing storage supplies atomic acknowledgement,
    retries and concurrency; this pure model supplies none of those itself.
    Original-prefix/suffix and publication-leaf validation work uses selected
    maximum_checkpoint_work (default65,536; hard1,048,576). Wire/dictionary input
    admits this work before recursive producer validation. Explicit trusted
    attribute extraction still enforces final bounds; getters are not sandboxed.
    Separate item maxima do not imply Cartesian capacity.
    """

    profile: Literal["owned-recipes-v1"] = "owned-recipes-v1"
    registries: Annotated[tuple[RegistryState, ...], Field(min_length=1, max_length=32)]
    ownership: OwnershipDeclaration
    executions: Annotated[tuple[RecipeExecution, ...], Field(max_length=8192)] = ()
    catalogues: Annotated[tuple[Catalogue, ...], Field(max_length=4096)] = ()
    limits: RegistryLimits = Field(default_factory=RegistryLimits)

    @model_validator(mode="before")
    @classmethod
    def _work(cls, value: Any) -> Any:
        """Admit coupled flat work before recursive nested execution validation."""
        if isinstance(value, RegistryCheckpoint):
            registries, executions = value.registries, value.executions
            catalogues, limits = value.catalogues, value.limits
            ownership = value.ownership
        elif isinstance(value, dict):
            registries, executions = value.get("registries", ()), value.get("executions", ())
            catalogues, limits = value.get("catalogues", ()), value.get("limits", RegistryLimits())
            ownership = value.get("ownership")
        else:
            return value
        if not all(isinstance(items, (tuple, list)) for items in (registries, executions, catalogues)):
            return value
        selected = RegistryLimits.model_validate(limits)
        if isinstance(ownership, OwnershipDeclaration):
            resources = ownership.resources
        elif isinstance(ownership, dict):
            resources = ownership.get("resources", ())
        else:
            resources = ()
        if isinstance(resources, (tuple, list)) and len(resources) > selected.maximum_resources:
            raise ValueError("Checkpoint ownership resource capacity exhausted")
        if len(registries) > 32 or len(executions) > selected.maximum_producers:
            raise ValueError("Checkpoint exceeds its finite registry/producer counts")
        if len(catalogues) > selected.maximum_publications:
            raise ValueError("Checkpoint publication capacity exhausted")
        work = sum(_raw_fact_count(registry) for registry in registries) * len(executions)
        work += sum(_raw_leaf_count(catalogue) for catalogue in catalogues)
        if work > selected.maximum_checkpoint_work:
            raise ValueError("Checkpoint exceeds its coupled history validation work capacity")
        return value

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Verify exact original joins without confusing action and recipe sets."""
        if len(self.ownership.resources) > self.limits.maximum_resources:
            raise ValueError("Checkpoint ownership resource capacity exhausted")
        tasks, recipes = _checkpoint_tasks(self.registries, self.ownership)
        publications = _publication_pool(self.catalogues, self.registries, self.limits)
        work = sum(len(registry.facts) for registry in self.registries) * len(self.executions)
        work += sum(len(catalogue.definitions) for catalogue in publications.values())
        if work > self.limits.maximum_checkpoint_work:
            raise ValueError("Checkpoint exceeds its normalized coupled validation work capacity")
        object.__setattr__(self, "catalogues", tuple(sorted(publications.values(), key=lambda item: item.fingerprint)))
        if len(self.executions) > self.limits.maximum_producers:
            raise ValueError("Checkpoint producer capacity exhausted")
        _checkpoint_origins(self.executions, tasks, recipes, publications)
        object.__setattr__(self, "registries", tuple(sorted(self.registries, key=lambda item: item.policy.task.id)))
        object.__setattr__(self, "executions", tuple(sorted(self.executions, key=lambda item: item.logical_identity)))
        admit_bytes(self, self.limits)
        return self

    def record(self, execution: RecipeExecution) -> RegistryCheckpoint:
        """Return a new complete origin set or reconcile an exact local retry.

        Parameters
        ----------
        execution : RecipeExecution
            Actual original inner execution and eligibility captured before it.

        Returns
        -------
        RegistryCheckpoint
            New immutable value, or current value for exact original content.

        Raises
        ------
        ValueError
            Conflict, stale/current snapshot mismatch, backward new round or
            capacity exhaustion. Existing caller values remain unchanged.

        Notes
        -----
        An exact retained retry keeps its original view even after catalogue
        growth. This pure helper does not execute, persist or deduplicate work.
        """
        current = RegistryCheckpoint.model_validate(self)
        origin = RecipeExecution.model_validate(execution)
        for previous in current.executions:
            if previous.logical_identity == origin.logical_identity:
                if previous.to_json() != origin.to_json():
                    raise ValueError("Conflicting original recipe execution for one logical opportunity")
                return current
        if len(current.executions) >= current.limits.maximum_producers:
            raise ValueError("Checkpoint producer capacity exhausted")
        registry = _registry_for(current, origin.recipe.task.id)
        actual = registry.snapshot(origin.snapshot.round, cutoff=origin.snapshot.cutoff)
        if actual.to_json() != origin.snapshot.to_json():
            raise ValueError("New recipe execution requires the actual current predecision registry snapshot")
        older = [item.snapshot.round for item in current.executions if item.recipe.task == origin.recipe.task]
        if older and origin.snapshot.round <= max(older):
            raise ValueError("New recipe execution cannot rewind its task opportunity frontier")
        return RegistryCheckpoint.model_validate({**current.model_dump(), "executions": (*current.executions, origin)})

    def for_execution(self, executed_fingerprint: str) -> RecipeExecution:
        """Resolve the original outer producer of one retained inner execution.

        Parameters
        ----------
        executed_fingerprint : str
            Exact unchanged inner ExecutedDecision fingerprint.

        Returns
        -------
        RecipeExecution
            Original predecision snapshot and recipe/epoch/ownership binding.

        Raises
        ------
        ValueError
            Unknown execution; no display-name or current-epoch fallback.
        """
        for execution in self.executions:
            if execution.executed.fingerprint == executed_fingerprint:
                return execution
        raise ValueError("Unknown original recipe execution")

    def replace_registry(self, registry: RegistryState) -> RegistryCheckpoint:
        """Extend one existing population without altering unrelated recipe state.

        Parameters
        ----------
        registry : RegistryState
            Prospective existing-task successor preserving exact prior history,
            fixed policy, selected limits and immutable old definition leaves.

        Returns
        -------
        RegistryCheckpoint
            Complete immutable successor; owned learner state is untouched.

        Raises
        ------
        ValueError
            Unknown task, changed policy/limits or historical rewind/conflict.
        """
        current = RegistryCheckpoint.model_validate(self)
        replacement = RegistryState.model_validate(registry)
        before = _registry_for(current, replacement.policy.task.id)
        _registry_successor(before, replacement)
        values = tuple(replacement if item.policy.task == before.policy.task else item for item in current.registries)
        return RegistryCheckpoint.model_validate({**current.model_dump(), "registries": values})


class RegistryOwner:
    """Expose complete registry origins through the existing Stateful protocol.

    Parameters
    ----------
    checkpoint : RegistryCheckpoint
        Complete initial value. Ownership and limits remain fixed for this
        profile; a changed configuration needs an explicit compatible restart.

    Notes
    -----
    Direct in-memory state performs no I/O. Compose its concrete codec with the
    existing state store for durable acceptance; the raw methods are not a new
    event engine and do not claim exactly-once physical work.
    """

    def __init__(self, checkpoint: RegistryCheckpoint) -> None:
        """Retain revalidated initial complete state without runtime capture."""
        self._state = RegistryCheckpoint.model_validate(checkpoint)
        self._ownership = self._state.ownership.to_json()
        self._limits = self._state.limits

    def export_state(self) -> RegistryCheckpoint:
        """Return revalidated complete original producers and registry histories.

        Returns
        -------
        RegistryCheckpoint
            Immutable complete declared state, without capture or publication.
        """
        return RegistryCheckpoint.model_validate(self._state)

    def restore_state(self, state: RegistryCheckpoint, /) -> None:
        """Restore compatible complete state into this explicit fresh owner.

        Parameters
        ----------
        state : RegistryCheckpoint
            Complete known-version state with exact fixed ownership and limits.

        Raises
        ------
        ValueError
            Unknown/invalid state, changed recipe configuration or lost initial
            history. The previous owner value remains unchanged on refusal.

        Notes
        -----
        Restore preserves this owner's retained origins and history. Create a
        fresh explicitly configured owner for another compatible lineage. Only
        the existing optional store supplies a durably acknowledged minimum.
        """
        checked = RegistryCheckpoint.model_validate(state)
        if checked.ownership.to_json() != self._ownership or checked.limits != self._limits:
            raise ValueError("Registry owner restore changes its fixed ownership or selected limits")
        for registry in self._state.registries:
            _registry_successor(registry, _registry_for(checked, registry.policy.task.id))
        origins = {execution.logical_identity: execution for execution in checked.executions}
        for execution in self._state.executions:
            retained_origin = origins.get(execution.logical_identity)
            if retained_origin is None or retained_origin.to_json() != execution.to_json():
                raise ValueError("Registry owner restore cannot remove or replace retained original producers")
        retained = {catalogue.fingerprint for catalogue in checked.catalogues}
        if any(catalogue.fingerprint not in retained for catalogue in self._state.catalogues):
            raise ValueError("Registry owner restore cannot remove retained catalogue publications")
        self._state = checked

    def record(self, execution: RecipeExecution) -> None:
        """Record one complete origin through the pure bounded checkpoint helper.

        Parameters
        ----------
        execution : RecipeExecution
            Exact actual execution and previously captured eligibility.

        Raises
        ------
        ValueError
            Conflicting producer, stale snapshot/frontier or capacity refusal.

        Notes
        -----
        An exact local retry leaves state unchanged; durability is external.
        """
        self._state = self._state.record(execution)

    def replace_registry(self, registry: RegistryState) -> None:
        """Extend one registry through its pure causal successor check.

        Parameters
        ----------
        registry : RegistryState
            Same-task fixed-policy immutable extension.

        Raises
        ------
        ValueError
            Incompatible interpretation, rewind/conflict or capacity refusal.
        """
        self._state = self._state.replace_registry(registry)
