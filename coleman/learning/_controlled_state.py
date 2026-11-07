"""Pure complete-state construction helpers for the controlled learner owner."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, NamedTuple, Self

from pydantic import model_validator

from coleman.core.actions import Weight
from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.core.temporal import Timestamp
from coleman.learning._controlled_output import ReleaseProposal
from coleman.learning.contracts import LearningFailure
from coleman.learning.control_contracts import ControlledConfig, ControlledPayload, ControlOutcome
from coleman.objectives.arithmetic import checked_fraction
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExactPathUsage,
    L1MovementPenalty,
    MovementLimit,
    MovementUsage,
    advance_exact_path_usage,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS
from coleman.state.contracts import AlgorithmState, StateBinding


class ReleaseContext(NamedTuple):
    """Identify one attempted release and its original block/application clock.

    Attributes
    ----------
    reference : Reference
        Owner/configuration-scoped logical attempt, independent from paid cost.
    panels : tuple of Reference
        Complete ordered accepted-block membership.
    apply_at : datetime or None
        Explicit actual processing time, genuinely unknown when null.
    """

    reference: Reference
    panels: tuple[Reference, ...]
    apply_at: Timestamp | None


class ApplicationClock(CoreImmutable):
    """Normalize explicit optional clocks without consulting ambient time."""

    eligible_at: Timestamp | None = None
    apply_at: Timestamp | None = None

    @model_validator(mode="after")
    def _order(self) -> Self:
        """Keep known eligibility no later than known application."""
        if self.eligible_at is not None and self.apply_at is not None and self.eligible_at > self.apply_at:
            raise ValueError("Eligibility cannot follow the actual processing instant")
        return self


def controlled_binding(config: ControlledConfig, owner: Reference) -> StateBinding:
    """Bind one fixed mathematical law and its closed immutable state codec.

    Parameters
    ----------
    config : ControlledConfig
        Complete immutable method/loss/controls definition.
    owner : Reference
        Explicit local mathematical state identity/version.

    Returns
    -------
    StateBinding
        Complete fixed interpretation, without registration or source lookup.
    """
    return StateBinding(
        owner=owner,
        implementation=DefinitionBinding(
            reference=Reference(id="coleman-" + config.profile, version="1"),
            definition_fingerprint=fingerprint(
                {
                    "profile": config.profile,
                    "version": 1,
                    "operation_order": config.numerical_profile,
                    "state": "actual-emitted-origin-and-complete-pending-panels",
                    "solver": "binary64-grid-breakpoint-kkt-v1",
                }
            ),
        ),
        state_codec=Reference(id="coleman-controlled-learning-state", version="1"),
        configuration_fingerprint=config.fingerprint,
    )


def checked_state(binding: StateBinding, payload: ControlledPayload) -> AlgorithmState[ControlledPayload]:
    """Validate all continuation invariants and the selected complete byte cap.

    Parameters
    ----------
    binding : StateBinding
        Exact owner/method/configuration/codec interpretation.
    payload : ControlledPayload
        Complete finite typed continuation, including pending observations.

    Returns
    -------
    AlgorithmState
        Independently validated immutable snapshot; no live state is assigned.

    Raises
    ------
    ValueError, TypeError
        Invalid version/content, selected count limits or whole-state byte cap.
    """
    result = AlgorithmState[ControlledPayload](schema_version=1, binding=binding, payload=payload)
    if len(result.to_json().encode("utf-8")) > payload.config.limits.maximum_bytes:
        raise ValueError("Controlled complete state exceeds the selected byte capacity")
    return result


def logical_reference(kind: str, binding: StateBinding, value: object) -> Reference:
    """Derive a deterministic local owner/configuration-scoped operation identity.

    Parameters
    ----------
    kind : str
        Bounded literal operation namespace supplied by trusted code.
    binding : StateBinding
        Complete owner/law/codec interpretation, never an object address.
    value : object
        Finite canonical JSON operation coordinates, such as an accepted index.

    Returns
    -------
    Reference
        Version-one content identity; a digest authenticates no source.
    """
    return Reference(
        id=kind + "-" + fingerprint({"binding": binding.model_dump(mode="json"), "value": value}), version="1"
    )


def initial_origin(config: ControlledConfig, binding: StateBinding) -> CoefficientOrigin:
    """Retain the declared initial penalty origin or bind explicit initial weights.

    Parameters
    ----------
    config : ControlledConfig
        Complete law with already validated initial static-box membership.
    binding : StateBinding
        Exact owner/law binding used for a projection-only initial identity.

    Returns
    -------
    CoefficientOrigin
        Complete original emitted coefficients, without hidden normalization.
    """
    output = config.controls.output
    if isinstance(output, L1MovementPenalty):
        return output.origin
    return CoefficientOrigin(
        schema_version=1,
        reference=logical_reference("initial-origin", binding, 0),
        lineage=DefinitionLineage(
            source=config.reference, source_fingerprint=config.fingerprint, declared_at=None, supersedes=None
        ),
        weights=config.controls.initial_prior.weights,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )


def emitted_origin(
    payload: ControlledPayload,
    binding: StateBinding,
    release: Reference,
    weights: tuple[Weight, ...],
    clock: Timestamp | None,
) -> CoefficientOrigin:
    """Version actual finite emitted coefficients, preserving their original bytes.

    Parameters
    ----------
    payload : ControlledPayload
        Original complete continuation and previously committed origin.
    binding : StateBinding
        Fixed mathematical owner/law interpretation.
    release : Reference
        Exact successful nonempty attempt identity.
    weights : tuple of Weight
        Actual qualified binary64 coefficient fractions, including signed zero.
    clock : datetime or None
        Explicit application instant; unknown remains null.

    Returns
    -------
    CoefficientOrigin
        New snapshot with exact superseded-origin lineage. No owner mutates.
    """
    return CoefficientOrigin(
        schema_version=1,
        reference=logical_reference("emitted-origin", binding, release.model_dump(mode="json")),
        lineage=DefinitionLineage(
            source=payload.config.reference,
            source_fingerprint=payload.config.fingerprint,
            declared_at=clock,
            supersedes=payload.origin.binding,
        ),
        weights=weights,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )


def net_usage(payload: ControlledPayload) -> MovementUsage | None:
    """Derive a net-displacement attestation; never round an accumulated ledger.

    Parameters
    ----------
    payload : ControlledPayload
        Complete current and period-start emitted coefficient origins.

    Returns
    -------
    MovementUsage or None
        Rounded finite display attestation for net-only period movement. The
        geometry derives exact net displacement from both original origins.
        Other movement profiles return null and retain their own exact ledger.

    Raises
    ------
    ValueError
        Required period state is absent or a nonzero display is unrepresentable.
    """
    movement = payload.config.controls.movement
    if movement is None or movement.scope != "per_interval" or movement.measure != "net_displacement":
        return None
    if payload.period_origin is None or not payload.periods:
        raise ValueError("Per-interval movement requires an admitted current period")
    used = tuple(
        MovementLimit(member=old.member, maximum=checked_fraction(abs(Fraction(new.value) - Fraction(old.value))))
        for old, new in zip(payload.period_origin.weights, payload.origin.weights, strict=True)
    )
    return MovementUsage(interval=payload.periods[-1].interval, origin=payload.period_origin, used=used)


def idle_outcome(payload: ControlledPayload, disposition: Literal["held", "empty"]) -> ControlOutcome:
    """Construct a genuine held/empty receipt without a hidden mathematical step.

    Parameters
    ----------
    payload : ControlledPayload
        Complete accepted/consumed/pending frontiers and actual coefficients.
    disposition : {'held', 'empty'}
        Accepted unreleased panel or explicitly closed interval without panels.

    Returns
    -------
    ControlOutcome
        Finite no-op receipt, with not-applicable proposal and no output fee.
    """
    return ControlOutcome(
        schema_version=1,
        disposition=disposition,
        law=payload.config.binding,
        update_count=payload.update_count,
        release_count=payload.release_count,
        consumed_panels=payload.consumed_panels,
        weights=payload.origin.weights,
        origin=payload.origin,
        emitted_origin=None,
        release=None,
        panels=tuple(item.reference for item in payload.pending),
        summed_losses=(),
        proposal=None,
        proposal_status="not_applicable",
        output=None,
        input_fingerprint=None,
        solver_profile=None,
        solver_inputs=None,
        movement_usage=None,
        exact_path_before=None,
        failure=None,
    )


def _attempt_failure(attempt: ReleaseProposal) -> LearningFailure | None:
    """Keep the actual negative mathematical disposition and its finite reason."""
    if attempt.failure is not None:
        return attempt.failure
    if attempt.solution is not None and attempt.solution.status == "optimal":
        return None
    code = "infeasible" if attempt.solution is not None and attempt.solution.status == "infeasible" else "numeric_range"
    return LearningFailure(code=code, message="The joint controlled output did not qualify")


def _advanced_origin(
    payload: ControlledPayload,
    binding: StateBinding,
    release: Reference,
    attempt: ReleaseProposal,
    clock: Timestamp | None,
) -> tuple[CoefficientOrigin, ExactPathUsage | None]:
    """Charge only one successfully emitted vector, including zero movement."""
    solution = attempt.solution
    if solution is None or solution.status != "optimal":
        return payload.origin, payload.exact_path
    if solution.weights is None:
        raise RuntimeError("An optimal solver result must contain emitted coefficients")
    origin = emitted_origin(payload, binding, release, solution.weights, clock)
    path = payload.exact_path
    if path is not None:
        path = advance_exact_path_usage(usage=path, origin=payload.origin, emitted=origin, release=release)
    return origin, path


def solved_outcome(
    payload: ControlledPayload,
    binding: StateBinding,
    attempt: ReleaseProposal,
    context: ReleaseContext,
) -> tuple[ControlOutcome, ExactPathUsage | None]:
    """Construct one complete release receipt without assigning the live owner.

    Parameters
    ----------
    payload : ControlledPayload
        Original accepted-panel continuation, before this attempt is consumed.
    binding : StateBinding
        Complete fixed owner/law interpretation.
    attempt : ReleaseProposal
        Typed exp/share/joint-output result, including explicit numerical refusal.
    context : ReleaseContext
        Exact release/block membership and supplied actual processing clock.

    Returns
    -------
    tuple of ControlOutcome and ExactPathUsage or None
        Complete finite disposition and next exact accumulated coefficient
        ledger. Failure leaves weights, R and usage unchanged. Optimal holds
        consume one nonempty release and charge zero actual movement.

    Raises
    ------
    ValueError
        Inconsistent typed joins or an actual emitted movement over its budget.
    RuntimeError
        A solver marks optimality without supplying actual coefficients.
    """
    failure = _attempt_failure(attempt)
    successful = failure is None
    origin, path = _advanced_origin(payload, binding, context.reference, attempt, context.apply_at)
    disposition: Literal["released", "infeasible", "numerical_failure"] = "released"
    if failure is not None:
        disposition = "infeasible" if failure.code == "infeasible" else "numerical_failure"
    solution = attempt.solution
    outcome = ControlOutcome(
        schema_version=1,
        disposition=disposition,
        law=payload.config.binding,
        update_count=payload.update_count,
        release_count=payload.release_count + int(successful),
        consumed_panels=payload.consumed_panels + (len(context.panels) if successful else 0),
        weights=origin.weights,
        origin=payload.origin,
        emitted_origin=origin if successful else None,
        release=context.reference,
        panels=context.panels,
        summed_losses=attempt.summed_losses,
        proposal=attempt.proposal,
        proposal_status="observed" if attempt.proposal is not None else "unavailable",
        output=attempt.output,
        input_fingerprint=fingerprint(
            {
                "proposal_input": attempt.input_fingerprint,
                "release": context.reference.model_dump(mode="json"),
                "apply_at": None if context.apply_at is None else context.apply_at.isoformat(),
            }
        ),
        solver_profile=None if solution is None else "binary64-grid-breakpoint-kkt-v1",
        solver_inputs=None if solution is None else solution.geometry.inputs_fingerprint,
        movement_usage=net_usage(payload),
        exact_path_before=payload.exact_path,
        failure=failure,
    )
    return outcome, path
