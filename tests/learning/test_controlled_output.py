"""Actual released-block recurrence, target share and joint-output composition."""

import math
from fractions import Fraction
from typing import cast

import pytest

from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.learning._controlled_output import release_proposal
from coleman.learning.control_contracts import AcceptedPanel, ControlledConfig
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExpertBound,
    ExpertBounds,
    L1MovementPenalty,
    MixtureControls,
    MovementAllowance,
    MovementLimit,
    ProjectionOutput,
    RoundCadence,
    initial_exact_path_usage,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, DeclaredTargetShare, InitialPrior


def ref(name, version="1"):
    return Reference(id=name, version=version)


def metadata(name):
    return {
        "schema_version": 1,
        "reference": ref(name),
        "lineage": DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
    }


def weights(values):
    return tuple(Weight(member=ref(f"expert-{i}"), value=value) for i, value in enumerate(values))


def origin(values, name="emitted-initial"):
    return CoefficientOrigin(
        **metadata(name), weights=weights(values), semantics=COEFFICIENT_SEMANTICS, profile="binary64-simplex-v1"
    )


def configuration(
    *, prior=(0.5, 0.5), eta=0.0, target=None, alpha=0.5, upper=None, strength=None, movement=None, cadence=1
):
    current = origin(prior)
    initial = InitialPrior(
        **metadata("prior"), weights=current.weights, semantics=COEFFICIENT_SEMANTICS, profile="binary64-simplex-v1"
    )
    shared = None
    if target is not None:
        shared = DeclaredTargetShare(
            **metadata("target"),
            weights=weights(target),
            semantics=COEFFICIENT_SEMANTICS,
            profile="binary64-simplex-v1",
            alpha=alpha,
        )
    bounds = ExpertBounds(
        **metadata("bounds"),
        semantics=COEFFICIENT_SEMANTICS,
        bounds=tuple(
            ExpertBound(member=weight.member, lower=0.0, upper=1.0 if upper is None else upper[i])
            for i, weight in enumerate(current.weights)
        ),
    )
    output = ProjectionOutput()
    if strength is not None:
        output = L1MovementPenalty(
            **metadata("penalty"), semantics=COEFFICIENT_SEMANTICS, strength=strength, origin=current
        )
    controls = MixtureControls(
        **metadata("controls"),
        initial_prior=initial,
        target=shared,
        bounds=bounds,
        movement=movement,
        cadence=RoundCadence(panels=cadence),
        output=output,
    )
    return ControlledConfig(
        **metadata("controlled-law"),
        controls=controls,
        profile="controlled-hedge-projection" if strength is None else "controlled-hedge-l1-movement",
        eta=eta,
        loss_spec=ScalarSpec(
            dimension=ref("loss"),
            semantics=UnitSemantics(unit="1", meaning="authored-bounded-loss"),
            role="loss",
            minimum=0,
            maximum=1,
        ),
        transform=DefinitionBinding(
            reference=ref("identity-transform"), definition_fingerprint=fingerprint("identity")
        ),
        period_clock=None if movement is None or movement.scope == "per_release" else ref("period-clock"),
    )


def accepted(config, values, sequence=1):
    return AcceptedPanel(
        schema_version=1,
        sequence=sequence,
        reference=ref(f"panel-{sequence}"),
        eligible_at=None,
        apply_at=None,
        panel=FullPanel(
            rows=tuple(
                FeedbackRow(
                    member=weight.member,
                    values=(
                        OutcomeValue(
                            spec=config.loss_spec,
                            value=value,
                            status="unavailable" if value is None else "observed",
                            origin="synthetic",
                            transform=config.transform.reference,
                        ),
                    ),
                )
                for weight, value in zip(config.controls.initial_prior.weights, values, strict=True)
            )
        ),
    )


def values(vector):
    return tuple(item.value for item in vector)


def test_sum_exponential_then_one_target_share_precedes_actual_joint_output():
    config = configuration(eta=math.log(2), target=(0.25, 0.75), upper=(0.5, 1.0), cadence=2)
    current = origin((0.5, 0.5))
    block = (accepted(config, (0.0, 1.0), 1), accepted(config, (0.0, 1.0), 2))
    before = config.to_json(), current.to_json(), tuple(item.to_json() for item in block)
    result = release_proposal(config, current, block)
    assert values(result.summed_losses) == (0.0, 2.0)
    assert result.proposal is not None
    assert values(result.proposal) == pytest.approx((0.525, 0.475), abs=3e-16)
    assert result.solution is not None and result.solution.status == "optimal"
    assert result.solution.weights == weights((0.5, 0.5))
    assert result.failure is None
    assert before == (config.to_json(), current.to_json(), tuple(item.to_json() for item in block))


def test_per_release_l1_origin_moves_without_rewriting_the_fixed_configuration():
    config = configuration(strength=0.25)
    current = origin((0.75, 0.25), "emitted-later")
    block = (accepted(config, (0.0, 0.0)),)
    before = config.to_json()
    result = release_proposal(config, current, block)
    assert isinstance(result.output, L1MovementPenalty)
    assert isinstance(config.controls.output, L1MovementPenalty)
    assert result.output.origin.to_json() == current.to_json()
    assert result.output.strength == config.controls.output.strength
    assert result.solution is not None and result.solution.weights == current.weights
    assert result.failure is None and config.to_json() == before
    assert config.controls.output.origin.reference.id == "emitted-initial"


def test_lambda_zero_and_projection_share_proposal_geometry_and_qualified_output():
    plain = configuration(eta=math.log(2), target=(0.25, 0.75), upper=(0.6, 1.0))
    penalized = configuration(eta=math.log(2), target=(0.25, 0.75), upper=(0.6, 1.0), strength=0.0)
    current = origin((0.5, 0.5))
    block = (accepted(plain, (0.0, 1.0)),)
    a = release_proposal(plain, current, block)
    b = release_proposal(penalized, current, block)
    assert a.proposal == b.proposal
    assert a.solution is not None and b.solution is not None
    assert a.solution.geometry == b.solution.geometry and a.solution.certificate == b.solution.certificate
    assert a.solution.weights == b.solution.weights
    assert a.input_fingerprint != b.input_fingerprint


@pytest.mark.parametrize("alpha,expected", [(0.0, (1.0, 0.0)), (0.25, (0.75, 0.25)), (1.0, (0.0, 1.0))])
def test_declared_share_preserves_and_restores_structural_zero_support(alpha, expected):
    config = configuration(prior=(1.0, 0.0), eta=5.0, target=(0.0, 1.0), alpha=alpha)
    current = origin((1.0, 0.0))
    result = release_proposal(config, current, (accepted(config, (1.0, 0.0)),))
    assert result.proposal == weights(expected)
    assert result.solution is not None and result.solution.weights == weights(expected)
    assert result.failure is None


def test_share_positive_underflow_is_explicit_and_retains_finite_inputs():
    config = configuration(prior=(0.0, 1.0), target=(math.ulp(0.0), 1.0), alpha=0.5)
    current = origin((0.0, 1.0))
    block = (accepted(config, (0.0, 0.0)),)
    result = release_proposal(config, current, block)
    assert result.failure is not None and result.failure.code == "positive_underflow"
    assert result.proposal is result.solution is None
    assert values(result.summed_losses) == (0.0, 0.0)
    assert result.input_fingerprint == release_proposal(config, current, block).input_fingerprint


def test_exponential_underflow_cannot_be_silently_repaired_by_a_later_target():
    config = configuration(eta=10000.0, target=(0.0, 1.0), alpha=1.0)
    result = release_proposal(config, origin((0.5, 0.5)), (accepted(config, (0.0, 1.0)),))
    assert result.failure is not None and result.failure.code == "positive_underflow"
    assert result.proposal is result.solution is None and values(result.summed_losses) == (0.0, 1.0)


def test_original_proposal_survives_exact_infeasibility_under_legacy_admission():
    hard = MovementAllowance(
        **metadata("zero-movement"),
        semantics=COEFFICIENT_SEMANTICS,
        scope="per_release",
        measure="accumulated_absolute",
        limits=tuple(MovementLimit(member=ref(f"expert-{i}"), maximum=0.0) for i in range(3)),
    )
    config = configuration(prior=(1 / 3,) * 3, movement=hard)
    current = origin((1 / 3,) * 3)
    result = release_proposal(config, current, (accepted(config, (0.0,) * 3),))
    assert result.failure is not None and result.failure.code == "infeasible"
    assert result.proposal == current.weights
    assert result.solution is not None and result.solution.status == "infeasible"
    legacy = result.solution.geometry.legacy_intersection
    assert legacy is not None and legacy.status == "feasible" and not legacy.exact_feasible


def test_solver_rounding_failure_retains_its_actual_solution_and_proposal():
    config = configuration(prior=(math.ulp(0.0), 1.0))
    current = origin((math.ulp(0.0), 1.0))
    result = release_proposal(config, current, (accepted(config, (0.0, 0.0)),))
    assert result.failure is not None and result.failure.code == "positive_underflow"
    assert result.proposal == current.weights
    assert result.solution is not None and result.solution.status == "numerical_failure"
    assert result.solution.reason == "nonrepresentable_output"


def test_exact_usage_limits_the_joint_output_without_spending_the_ledger():
    hard = MovementAllowance(
        **metadata("interval-movement"),
        semantics=COEFFICIENT_SEMANTICS,
        scope="per_interval",
        measure="accumulated_absolute",
        limits=tuple(MovementLimit(member=ref(f"expert-{i}"), maximum=0.125) for i in range(2)),
    )
    config = configuration(prior=(1.0, 0.0), target=(0.0, 1.0), movement=hard, strength=0.01)
    current = origin((1.0, 0.0))
    ledger = initial_exact_path_usage(interval=ref("period"), movement=hard, origin=current)
    before = ledger.to_json()
    result = release_proposal(config, current, (accepted(config, (0.0, 0.0)),), exact_path=ledger)
    assert result.solution is not None and result.solution.weights == weights((0.875, 0.125))
    assert result.failure is None and ledger.to_json() == before
    assert values(result.proposal) == (0.5, 0.5)


def test_full_input_identity_distinguishes_original_blocks_with_equal_sum_and_geometry():
    config = configuration(cadence=2)
    current = origin((0.5, 0.5))
    a = (accepted(config, (0.0, 1.0), 1), accepted(config, (1.0, 0.0), 2))
    b = (accepted(config, (1.0, 0.0), 1), accepted(config, (0.0, 1.0), 2))
    first = release_proposal(config, current, a)
    second = release_proposal(config, current, b)
    assert first.summed_losses == second.summed_losses and first.proposal == second.proposal
    assert first.solution is not None and second.solution is not None
    assert first.solution.geometry.inputs_fingerprint == second.solution.geometry.inputs_fingerprint
    assert first.input_fingerprint != second.input_fingerprint
    assert first.input_fingerprint != first.solution.geometry.inputs_fingerprint
    assert first == release_proposal(config, current, a)


def test_loss_sum_rounds_once_instead_of_repeatedly_losing_small_positive_terms():
    config = configuration(cadence=3)
    block = tuple(accepted(config, (loss, 0.0), i + 1) for i, loss in enumerate((1.0, 2**-53, 2**-53)))
    result = release_proposal(config, origin((0.5, 0.5)), block)
    assert result.summed_losses[0].value == math.nextafter(1.0, math.inf)
    assert result.summed_losses[0].value > 1.0
    assert result.failure is None


@pytest.mark.parametrize("shape", ["empty", "list", "duplicate", "reverse"])
def test_invalid_block_shape_order_or_duplicate_does_not_reach_mathematical_output(shape):
    config = configuration(cadence=2)
    first, second = accepted(config, (0.0, 1.0), 1), accepted(config, (1.0, 0.0), 2)
    block = cast(
        tuple[AcceptedPanel, ...],
        {"empty": (), "list": [first], "duplicate": (first, first), "reverse": (second, first)}[shape],
    )
    current = origin((0.5, 0.5))
    with pytest.raises(ValueError, match="block|sequence"):
        release_proposal(config, current, block)


def test_missing_observed_loss_is_rejected_without_zero_imputation():
    config = configuration()
    block = (accepted(config, (None, 0.0)),)
    current = origin((0.5, 0.5))
    with pytest.raises(ValueError, match="observed loss"):
        release_proposal(config, current, block)


def test_selected_complete_byte_capacity_rejects_full_inputs_before_numerical_work():
    base = configuration()
    data = base.model_dump(mode="python")
    data["limits"] = {**data["limits"], "maximum_bytes": len(base.to_json().encode("utf-8")) + 128}
    config = ControlledConfig.model_validate(data)
    current = origin((0.5, 0.5))
    block = (accepted(config, (0.0, 1.0)),)
    with pytest.raises(ValueError, match="complete-byte capacity"):
        release_proposal(config, current, block)


def test_release_origin_requires_the_configured_expert_version_before_loss_work():
    config, original = configuration(), origin((0.5, 0.5))
    changed = Weight(member=ref(original.weights[0].member.id, "2"), value=0.5)
    current = CoefficientOrigin.model_validate(
        original.model_dump(mode="python") | {"weights": (changed, original.weights[1])}
    )
    block = (accepted(config, (0.0, 1.0)),)
    before = config.to_json(), current.to_json(), block[0].to_json()
    with pytest.raises(ValueError, match="exact configured members and versions"):
        release_proposal(config, current, block)
    assert (config.to_json(), current.to_json(), block[0].to_json()) == before


def test_release_requires_concrete_accepted_panel_envelopes():
    config, current = configuration(), origin((0.5, 0.5))
    document = accepted(config, (0.0, 1.0)).model_dump(mode="python")
    block = (cast(AcceptedPanel, document),)
    with pytest.raises(ValueError, match="concrete accepted-panel envelopes"):
        release_proposal(config, current, block)


def test_complete_block_schema_bytes_are_coupled_even_inside_panel_count_capacity():
    base = configuration(cadence=32)
    data = base.model_dump(mode="python")
    data["limits"] = {
        **data["limits"],
        "maximum_pending": 32,
        "maximum_bytes": len(base.to_json().encode("utf-8")) + 128,
    }
    config, current = ControlledConfig.model_validate(data), origin((0.5, 0.5))
    block = tuple(accepted(config, (0.0, 1.0), sequence) for sequence in range(1, 33))
    before = config.to_json(), current.to_json(), tuple(item.to_json() for item in block)
    assert len(block) == config.limits.maximum_pending
    with pytest.raises(ValueError, match="loss schemas exceed the selected complete-byte capacity"):
        release_proposal(config, current, block)
    assert (config.to_json(), current.to_json(), tuple(item.to_json() for item in block)) == before


def test_valid_residual_boundary_target_share_has_an_explicit_rounding_failure():
    third = 1 / 3
    step = math.ulp(third)
    prior = (third, third, third + 33 * step)
    target = (third + step, third + step, third + 31 * step)
    limit = Fraction(1, 2**49)
    assert sum(Fraction(value) for value in prior) == sum(Fraction(value) for value in target) == 1 + limit
    exact_share = tuple((Fraction(left) + Fraction(right)) / 2 for left, right in zip(prior, target, strict=True))
    assert sum(exact_share) == 1 + limit
    assert sum(Fraction(float(value)) for value in exact_share) == 1 + limit + Fraction(1, 2**54)
    config, current = configuration(prior=prior, target=target, alpha=0.5), origin(prior)
    block = (accepted(config, (0.0, 0.0, 0.0)),)
    before = config.to_json(), current.to_json(), block[0].to_json()
    result = release_proposal(config, current, block)
    assert result.failure is not None and result.failure.code == "normalization"
    assert result.proposal is None and result.solution is None
    assert tuple(item.value for item in result.summed_losses) == (0.0, 0.0, 0.0)
    assert result.input_fingerprint.startswith("sha256:")
    assert (config.to_json(), current.to_json(), block[0].to_json()) == before
