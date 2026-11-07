"""Explicit preflight for registered inputs to the unchanged replay engine.

Import this optional seam separately from pure catalogue/ownership contracts.
It validates a supplied actual replay policy and preserves its original request,
semantic-context, ordinal, source and feedback rules; it creates no runtime.
"""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from coleman.core.actions import space_members
from coleman.registry.envelopes import RegistryRequest
from coleman.registry.limits import RegistryContract
from coleman.replay.contracts import ReplayOpportunity, ReplayPolicy
from coleman.replay.revisions import validate_opportunity


class RegisteredReplayOpportunity(RegistryContract):
    """Preflight a registered request against its actual existing replay policy.

    Attributes
    ----------
    registered : RegistryRequest
        Exact predecision registry view and unchanged core request.
    opportunity : ReplayOpportunity
        Same complete core request, original target and optional full semantics.
        Pass this unchanged to the existing session's ``prepare``/``emit``.
    policy : ReplayPolicy
        Actual selected session policy, including explicit specialist/delivery
        declarations and the real host capability. No learner is created here.
    first : ReplayOpportunity or None
        Exact first emitted opportunity for an existing session, or explicit
        null at an independent origin. The session independently revalidates
        its actual retained first request, next ordinal and source snapshot.

    Raises
    ------
    ValueError
        Mismatched core request, unsupported action/feedback semantics, changed
        static universe/context, or a synchronous empty opportunity.

    Notes
    -----
    Static replay rejects active-contract changes. Specialist replay permits
    changing finite spaces only under its host attestation; it still freezes
    SemanticContext. Changing member-bound priors/controls is therefore not
    enabled by this wrapper. No specialist comparator guarantee is inferred.

    A synchronous empty opportunity rejects before a host call: existing absent
    feedback cannot complete it. Delayed empty input, if the host capability
    permits it, remains pending without a fabricated zero-loss/no-op update.
    Direct registry and durable-direct components may abstain independently.

    Pure construction performs no prepare, decision, source access, state
    mutation, acceptance or I/O. The host supplies the actual policy/first view;
    this preflight cannot police calls made outside the supported interface.
    """

    registered: RegistryRequest
    opportunity: ReplayOpportunity
    policy: ReplayPolicy
    first: ReplayOpportunity | None

    @model_validator(mode="after")
    def _compatible(self) -> Self:
        """Reject unsupported replay combinations before any host emission."""
        if self.registered.request.to_json() != self.opportunity.request.to_json():
            raise ValueError("Registered replay must retain the exact original core request")
        if self.policy.delivery == "synchronous" and not space_members(self.opportunity.request.space):
            raise ValueError("Synchronous registered replay cannot complete an empty opportunity")
        validate_opportunity(self.policy, self.first, self.opportunity)
        return self
