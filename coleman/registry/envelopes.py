"""Optional producing request/execution/feedback joins retaining core v1 bytes."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.core.actions import AbstainAction, VectorSpace, space_members
from coleman.core.capabilities import Capability, validate_feedback
from coleman.core.decisions import DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch
from coleman.core.identity import ordered_references
from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp
from coleman.registry.definitions import RecipeBinding
from coleman.registry.lifecycle import EligibilitySnapshot, OpportunitySeal, RegistryState
from coleman.registry.limits import RegistryContract, RegistryLimits, admit_bytes


class RegistryRequest(RegistryContract):
    """Join a predecision registry snapshot to unchanged core request content.

    Attributes
    ----------
    request : DecisionRequest
        Existing finite-space request with identical task/round/cutoff, exact
        active members and active-contract digest as feasible-set version.
    snapshot : EligibilitySnapshot
        Complete producing causal prefix, including predecision dispositions.
    recipes : tuple of RecipeBinding, optional
        At most32 exact active expert/epoch bindings, ordered by child ID.
        Supported for expert populations only. Action-population consumers use
        external ownership declarations because learner and action epochs differ.

    Raises
    ------
    ValueError
        Wrong task/round/clock, active space/version, recipe semantics or bounds.

    Notes
    -----
    The existing core request serializes exactly as before. Full snapshot digest
    stays in this outer envelope; the core space version binds only the active
    contract. No direct learner is required to use this optional registry.
    Construction performs no decision, acceptance, factories or I/O.
    """

    request: DecisionRequest
    snapshot: EligibilitySnapshot
    recipes: Annotated[tuple[RecipeBinding, ...], Field(max_length=32)] = ()

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Reject semantic mismatches before any registered host call."""
        if self.request.task != self.snapshot.policy.task or self.request.round != self.snapshot.round:
            raise ValueError("Registered request must match snapshot task and round")
        cutoff = None if self.request.timing is None else self.request.timing.information_cutoff
        if cutoff != self.snapshot.cutoff or (self.request.timing is not None) != (
            self.snapshot.policy.clock == "timed"
        ):
            raise ValueError("Registered request timing must match the fixed population policy and cutoff")
        if isinstance(self.request.space, VectorSpace):
            raise ValueError("Registered population profile supports finite member spaces only")
        if self.request.space.members != self.snapshot.active:
            raise ValueError("Registered request must retain exact predecision active members")
        if self.request.space.feasible_set.version != self.snapshot.active_contract_fingerprint:
            raise ValueError("Registered core space version must bind the selective active contract")
        self._recipe_bindings()
        admit_bytes(self, RegistryLimits(maximum_bytes=16 * 1024 * 1024))
        return self

    def _recipe_bindings(self) -> None:
        """Separate expert recipes from action-member ownership semantics."""
        if self.recipes and self.snapshot.policy.kind != "expert":
            raise ValueError("Action populations bind consumer recipes through external ownership declarations")
        ordered_references(tuple(recipe.child for recipe in self.recipes))
        active = {row.member: row for row in self.snapshot.members if row.status == "active"}
        for recipe in self.recipes:
            row = active.get(recipe.expert.reference)
            if row is None:
                raise ValueError("Recipe must match the exact active expert, epoch, task and objective")
            supplied = (recipe.expert.definition_fingerprint, recipe.state_epoch, recipe.task, recipe.objective)
            expected = (row.definition_fingerprint, row.state_epoch, self.request.task, self.request.objective)
            if supplied != expected:
                raise ValueError("Recipe must match the exact active expert, epoch, task and objective")
        object.__setattr__(self, "recipes", tuple(sorted(self.recipes, key=lambda recipe: recipe.child.id)))


class RegisteredExecution(RegistryContract):
    """Retain an unchanged core execution and its original outer producer join.

    Attributes
    ----------
    request : RegistryRequest
        Exact original predecision snapshot and core request.
    executed : ExecutedDecision
        Existing actual-action envelope with its exact original core request.

    Raises
    ------
    ValueError
        Core request mismatch, changed active execution space, nonabstaining
        empty execution or encoded content above16MiB.

    Notes
    -----
    This profile conservatively preserves the registered active space across
    host mapping; mappings within that space remain core-defined. Empty active
    execution requires existing AbstainAction and fabricates no feedback/update.
    The fingerprint is the outer producer identity; construction does no work,
    routing, state transfer, durable acceptance or I/O.
    """

    request: RegistryRequest
    executed: ExecutedDecision

    @model_validator(mode="after")
    def _producer(self) -> Self:
        """Check exact core and registry joins without current-catalogue lookup."""
        if self.executed.proposal.request.fingerprint != self.request.request.fingerprint:
            raise ValueError("Registered execution must retain the exact original core request")
        if self.executed.action.space.model_dump(mode="json") != self.request.request.space.model_dump(mode="json"):
            raise ValueError("Registered execution must retain the exact active contract space")
        if not space_members(self.executed.action.space) and not isinstance(self.executed.action, AbstainAction):
            raise ValueError("Empty active execution requires explicit AbstainAction")
        admit_bytes(self, RegistryLimits(maximum_bytes=16 * 1024 * 1024))
        return self


class RegisteredFeedback(RegistryContract):
    """Bind unchanged core feedback to one exact original registered producer.

    Attributes
    ----------
    producer_fingerprint : str
        Full ``RegisteredExecution.fingerprint``, including original snapshot.
    batch : FeedbackBatch
        Existing feedback envelope; its decision fingerprint still binds the
        exact inner ExecutedDecision. Missing values/statuses remain unchanged.

    Notes
    -----
    Call ``validate_for`` with the original producer and actual capability before
    applying. Retirement, reentry and new versions never redirect this join to
    today's member/epoch. No update, imputation, delivery, deduplication or I/O.
    """

    producer_fingerprint: Fingerprint
    batch: FeedbackBatch

    def validate_for(
        self, producer: RegisteredExecution, capability: Capability, *, apply_at: Timestamp | None = None
    ) -> None:
        """Check outer producer then reuse exact existing feedback validation.

        Parameters
        ----------
        producer : RegisteredExecution
            Original immutable producer, including historical expert epochs.
        capability : Capability
            Actual consumer's required coverage, values, clocks and delays.
        apply_at : datetime or None, optional
            Explicit application instant required by timed core feedback.

        Raises
        ------
        ValueError
            Wrong outer producer, inner execution, incomplete active panel,
            unsupported missingness/delay or any existing capability invariant.

        Notes
        -----
        Performs no current-catalogue lookup, state mutation or I/O. Static and
        dynamic learner compatibility still belongs to the supplied capability
        and optional replay policy, with no comparator/theorem inferred.
        """
        current, origin = RegisteredFeedback.model_validate(self), RegisteredExecution.model_validate(producer)
        if current.producer_fingerprint != origin.fingerprint:
            raise ValueError("Feedback outer producer does not match the original registered execution")
        validate_feedback(origin.executed, current.batch, capability, apply_at=apply_at)


def seal_request(state: RegistryState, request: RegistryRequest) -> RegistryState:
    """Return bounded immutable local acceptance of an exact producing request.

    Parameters
    ----------
    state : RegistryState
        Current declared population, history and compact retained retry seals.
    request : RegistryRequest
        Exact predecision snapshot and unchanged core request to seal.

    Returns
    -------
    RegistryState
        Prospective immutable successor, or current value for an exact retry.

    Raises
    ------
    ValueError
        Same task/decision-ID has conflicting round or outer binding, snapshot
        differs from current history, backward new round or capacity exhaustion.

    Notes
    -----
    Counts/full envelope bytes are admitted before returning any successor.
    Full history is stored once; each seal retains only compact scalar fields.
    This pure helper neither executes a host nor provides durable exactly-once
    or concurrency guarantees. Optional existing replay/store owns those.
    """
    current, envelope = RegistryState.model_validate(state), RegistryRequest.model_validate(request)
    admit_bytes(envelope, current.limits)
    binding = OpportunitySeal(
        logical_identity=envelope.request.logical_identity,
        round=envelope.request.round,
        cutoff=envelope.snapshot.cutoff,
        request_fingerprint=envelope.request.fingerprint,
        snapshot_fingerprint=envelope.snapshot.fingerprint,
        envelope_fingerprint=envelope.fingerprint,
        fact_count=len(envelope.snapshot.facts),
        history_fingerprint=envelope.snapshot.history_fingerprint,
    )
    for previous in current.producers:
        if previous.logical_identity == binding.logical_identity:
            if previous != binding:
                raise ValueError("Conflicting registered retry for the same task and decision ID")
            return current
    if len(current.producers) >= current.limits.maximum_producers:
        raise ValueError("Registry retained producer capacity exhausted")
    if binding.round < max((item.round for item in current.producers), default=-1):
        raise ValueError("New producing requests cannot move backward across sealed rounds")
    expected = current.snapshot(binding.round, envelope.snapshot.cutoff)
    if expected.fingerprint != envelope.snapshot.fingerprint:
        raise ValueError("Producing snapshot does not match the current exact registry history")
    return RegistryState.model_validate(current.model_dump() | {"producers": (*current.producers, binding)})
