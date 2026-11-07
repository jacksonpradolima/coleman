"""Pure released-block exponential, declared-target and joint-output seam."""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

from coleman.core.actions import Weight
from coleman.core.identity import ordered_references
from coleman.core.numeric import validate_simplex
from coleman.core.primitives import canonical_json, fingerprint
from coleman.learning._math import exp_update
from coleman.learning.contracts import (
    FailureCode,
    LearningFailure,
    LearningNumericalError,
    _panel_rows,
    read_loss_panel,
)
from coleman.learning.control_contracts import AcceptedPanel, AggregateLoss, ControlledConfig
from coleman.objectives.arithmetic import checked_fraction
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExactPathUsage,
    L1MovementPenalty,
    MovementUsage,
    ProjectionOutput,
    exact_mixture_intervals,
)
from coleman.objectives.preferences import DeclaredTargetShare
from coleman.optimization.mixture import MixtureSolveResult, solve_mixture_output

RELEASE_PROPOSAL_PROFILE = "released-block-exp-share-joint-v1"


class ReleaseProposal(NamedTuple):
    """Retain one finite released-block proposal and its explicit disposition.

    Attributes
    ----------
    summed_losses : tuple of AggregateLoss
        Canonical derived per-expert SUM losses; these may exceed one and are
        not another observed bounded loss panel.
    proposal : tuple of Weight or None
        Post-exponential, post-target proposal, or null if its numerical stage
        failed. Solver infeasibility retains the original proposal.
    output : ProjectionOutput or L1MovementPenalty
        Complete selected per-step output, with the actual current origin.
    input_fingerprint : str
        Complete canonical input identity, distinct from geometry-only identity.
    solution : MixtureSolveResult or None
        Actual joint solve disposition, or null if proposal construction failed.
        Rational diagnostics remain in process and are never coerced to floats.
    failure : LearningFailure or None
        Explicit numerical/infeasible refusal, or null on qualified output.

    Notes
    -----
    This immutable result commits no coefficient, panel, clock or budget state.
    Its owner consumes a block only after accepting a successful joint output.
    """

    summed_losses: tuple[AggregateLoss, ...]
    proposal: tuple[Weight, ...] | None
    output: ProjectionOutput | L1MovementPenalty
    input_fingerprint: str
    solution: MixtureSolveResult | None
    failure: LearningFailure | None


def _block(
    configuration: ControlledConfig, origin: CoefficientOrigin, panels: tuple[AcceptedPanel, ...]
) -> tuple[AcceptedPanel, ...]:
    """Admit immutable bounded shapes and original accepted-panel ordering."""
    # pylint: disable-next=unidiomatic-typecheck
    if type(panels) is not tuple or not 1 <= len(panels) <= configuration.limits.maximum_pending:
        raise ValueError("A release requires a nonempty bounded immutable accepted-panel block")
    members = tuple(weight.member for weight in origin.weights)
    if members != tuple(bound.member for bound in configuration.controls.bounds.bounds):
        raise ValueError("Release origin differs from the exact configured members and versions")
    schema_bytes = len(configuration.loss_spec.model_dump_json().encode("utf-8"))
    schema_bytes += len(configuration.transform.reference.model_dump_json().encode("utf-8"))
    member_bytes = sum(len(member.model_dump_json().encode("utf-8")) for member in members)
    if len(panels) * (schema_bytes * len(members) + member_bytes) > configuration.limits.maximum_bytes:
        raise ValueError("Release loss schemas exceed the selected complete-byte capacity")
    for item in panels:
        if type(item) is not AcceptedPanel:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("A release requires concrete accepted-panel envelopes")
        _panel_rows(item.panel, len(members))
    checked = tuple(AcceptedPanel.model_validate(item) for item in panels)
    sequences = tuple(item.sequence for item in checked)
    if sequences != tuple(sorted(set(sequences))):
        raise ValueError("Release panels require increasing unique accepted sequence numbers")
    ordered_references(tuple(item.reference for item in checked))
    return checked


def _step_output(configuration: ControlledConfig, origin: CoefficientOrigin) -> ProjectionOutput | L1MovementPenalty:
    """Rebind only the moving L1 origin, retaining the fixed law declaration."""
    selected = configuration.controls.output
    if isinstance(selected, L1MovementPenalty):
        return L1MovementPenalty.model_validate(selected.model_dump(mode="python") | {"origin": origin})
    return selected


def _input_fingerprint(
    configuration: ControlledConfig,
    origin: CoefficientOrigin,
    panels: tuple[AcceptedPanel, ...],
    output: ProjectionOutput | L1MovementPenalty,
    context: tuple[MovementUsage | None, ExactPathUsage | None],
) -> str:
    """Bound complete source inputs before numerical stages and fingerprinting."""
    usage, exact_path = context
    payload = {
        "profile": RELEASE_PROPOSAL_PROFILE,
        "configuration": configuration.model_dump(mode="json"),
        "origin": origin.model_dump(mode="json"),
        "panels": [item.model_dump(mode="json") for item in panels],
        "output": output.model_dump(mode="json"),
        "usage": None if usage is None else usage.model_dump(mode="json"),
        "exact_path": None if exact_path is None else exact_path.model_dump(mode="json"),
    }
    if len(canonical_json(payload).encode("utf-8")) > configuration.limits.maximum_bytes:
        raise ValueError("Release inputs exceed the selected complete-byte capacity")
    return fingerprint(payload)


def _sum_losses(
    configuration: ControlledConfig, origin: CoefficientOrigin, panels: tuple[AcceptedPanel, ...]
) -> tuple[AggregateLoss, ...]:
    """Sum admitted observed panel values exactly before one binary64 rounding."""
    members = tuple(weight.member for weight in origin.weights)
    totals = [Fraction() for _ in members]
    for item in panels:
        losses = read_loss_panel(
            item.panel, members=members, loss_spec=configuration.loss_spec, transform=configuration.transform
        )
        totals = [total + Fraction(value) for total, value in zip(totals, losses, strict=True)]
    return tuple(
        AggregateLoss(member=member, value=checked_fraction(total))
        for member, total in zip(members, totals, strict=True)
    )


def _target_share(weights: tuple[Weight, ...], target: DeclaredTargetShare | None) -> tuple[Weight, ...]:
    """Blend the declared target once with exact stored-value arithmetic."""
    if target is None or target.alpha == 0:
        return weights
    fraction = Fraction(target.alpha)
    exact = tuple(
        (1 - fraction) * Fraction(weight.value) + fraction * Fraction(anchor.value)
        for weight, anchor in zip(weights, target.weights, strict=True)
    )
    try:
        values = tuple(checked_fraction(value) for value in exact)
    except ValueError as exc:
        raise LearningNumericalError(
            "positive_underflow", "A positive declared-target share underflows binary64"
        ) from exc
    try:
        validate_simplex(values)
    except ValueError as exc:
        raise LearningNumericalError("normalization", "Declared-target share violates binary64-simplex-v1") from exc
    return tuple(Weight(member=weight.member, value=value) for weight, value in zip(weights, values, strict=True))


def _solver_failure(solution: MixtureSolveResult) -> LearningFailure | None:
    """Map exact/rounded solve refusals without fabricating zero diagnostics."""
    if solution.status == "optimal":
        return None
    codes: dict[str, FailureCode] = {
        "nonrepresentable_output": "positive_underflow",
        "rounding_envelope": "normalization",
        "rounded_box": "normalization",
        "work_limit": "capacity",
    }
    code = "infeasible" if solution.status == "infeasible" else codes.get(solution.reason, "numeric_range")
    return LearningFailure(code=code, message=f"Joint coefficient output refused: {solution.reason}")


def _compute(
    configuration: ControlledConfig,
    origin: CoefficientOrigin,
    summed: tuple[AggregateLoss, ...],
    output: ProjectionOutput | L1MovementPenalty,
    context: tuple[MovementUsage | None, ExactPathUsage | None],
) -> tuple[tuple[Weight, ...] | None, MixtureSolveResult | None, LearningFailure | None]:
    """Apply exponential, one target share and one joint solve in fixed order."""
    try:
        exponential = exp_update(origin.weights, tuple(item.value for item in summed), configuration.eta)
        proposal = _target_share(exponential, configuration.controls.target)
    except LearningNumericalError as exc:
        return None, None, exc.failure
    usage, exact_path = context
    solution = solve_mixture_output(
        proposal=proposal,
        bounds=configuration.controls.bounds,
        origin=origin,
        output=output,
        movement=configuration.controls.movement,
        usage=usage,
        exact_path=exact_path,
    )
    return proposal, solution, _solver_failure(solution)


def release_proposal(
    configuration: ControlledConfig,
    origin: CoefficientOrigin,
    panels: tuple[AcceptedPanel, ...],
    *,
    usage: MovementUsage | None = None,
    exact_path: ExactPathUsage | None = None,
) -> ReleaseProposal:
    """Construct one sum-loss released-block proposal and actual joint output.

    Parameters
    ----------
    configuration : ControlledConfig
        Immutable fixed constant-rate, target, bounds and output law.
    origin : CoefficientOrigin
        Actual current emitted vector and exact version.
    panels : tuple of AcceptedPanel
        Nonempty complete observed block in increasing accepted sequence order.
        Panels retain their original identities, units, transform and clocks.
    usage : MovementUsage or None, optional
        Explicit legacy period usage, independently absent from soft penalties.
    exact_path : ExactPathUsage or None, optional
        Authoritative exact emitted-path accounting, exclusive with usage.

    Returns
    -------
    ReleaseProposal
        Finite summed inputs, post-share proposal, per-step output and actual
        solver result; numerical failure remains explicit without state mutation.

    Raises
    ------
    ValueError, TypeError
        Malformed/incomplete input, wrong versions/units, unknown usage or
        exceeded selected count/complete-byte capacity.

    Notes
    -----
    Every observed panel enters the SUM once, followed by one exponential step,
    one declared-target share and one selected joint output. No mean loss,
    per-panel share/penalty, hidden clock, interval consumption, period reset or
    durable deduplication occurs. Exact sums and blends round once; positive
    underflow is failure, while mathematical structural zeros remain valid.
    The input digest binds complete source inputs, never scientific authenticity.
    """
    declared = ControlledConfig.model_validate(configuration)
    current = CoefficientOrigin.model_validate(origin)
    block = _block(declared, current, panels)
    context = (
        None if usage is None else MovementUsage.model_validate(usage),
        None if exact_path is None else ExactPathUsage.model_validate(exact_path),
    )
    exact_mixture_intervals(
        declared.controls.bounds, current, movement=declared.controls.movement, usage=context[0], exact_path=context[1]
    )
    output = _step_output(declared, current)
    identity = _input_fingerprint(declared, current, block, output, context)
    summed = _sum_losses(declared, current, block)
    proposal, solution, failure = _compute(declared, current, summed, output, context)
    return ReleaseProposal(summed, proposal, output, identity, solution, failure)
