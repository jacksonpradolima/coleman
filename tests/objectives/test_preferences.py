"""Qualify explicit subjective inputs without turning them into observations."""

import json
from fractions import Fraction
from itertools import permutations
from typing import Any, cast

import pytest

from coleman.core.actions import Weight
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.preferences import (
    COEFFICIENT_SEMANTICS,
    AnchorEvaluation,
    DeclaredTargetShare,
    HumanJudgment,
    InitialPrior,
    ObjectiveWeight,
    ObjectiveWeights,
    PreferenceEvaluation,
    ReferencePoint,
    ReferencePoints,
    StaticQuadraticAnchor,
    coefficient_weights,
    evaluate_static_anchor,
    has_full_support,
    weight_objectives,
)


def _ref(name, version="v1"):
    return Reference(id=name, version=version)


def _metadata(name="subjective-definition"):
    return {
        "schema_version": 1,
        "reference": _ref(name),
        "lineage": DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
    }


def _weights(values=(0.75, 0.25)):
    return tuple(Weight(member=_ref(f"expert-{index}"), value=value) for index, value in enumerate(values))


def _coefficient_fields(values=(0.75, 0.25)):
    return {
        **_metadata(),
        "weights": _weights(values),
        "profile": "binary64-simplex-v1",
        "semantics": COEFFICIENT_SEMANTICS,
    }


def _spec(name="subjective-value", role="objective", bounds=(0.0, 1.0), unit="1"):
    return ScalarSpec(
        dimension=_ref(name),
        semantics=UnitSemantics(unit=unit, meaning=f"{name}-v1"),
        role=role,
        minimum=bounds[0],
        maximum=bounds[1],
    )


def _meaning(name="loss-a", horizon=None):
    return QuantityMeaning(spec=_spec(name, "loss"), interpretation="normalized_loss", horizon=horizon)


def _objective():
    return ObjectiveWeights(
        **_metadata(),
        terms=(
            ObjectiveWeight(quantity=_meaning(), weight=0.75),
            ObjectiveWeight(quantity=_meaning("loss-b"), weight=0.25),
        ),
        output=_spec(),
    )


def _values(status="observed"):
    return (
        Scalar(spec=_meaning().spec, value=0.2 if status == "observed" else None, status=status),
        Scalar(spec=_meaning("loss-b").spec, value=0.6, status="observed"),
    )


def test_initial_prior_does_not_execute_or_replace_a_baseline():
    prior = InitialPrior(**_coefficient_fields((0.0, 1.0)))
    assert not has_full_support(prior)
    assert has_full_support(InitialPrior(**_coefficient_fields()))
    baseline = _values()[0]
    before = baseline.model_dump_json()
    no_preference = None
    selected = baseline if no_preference is None else weight_objectives(no_preference, _values())
    assert selected is baseline
    assert baseline.model_dump_json() == before
    assert "feedback" not in prior.model_dump()


@pytest.mark.parametrize("alpha", [0.0, 1.0, 0.25])
def test_target_strength_is_a_share_fraction_separate_from_penalty(alpha):
    target = DeclaredTargetShare(**_coefficient_fields((0.0, 1.0)), alpha=alpha)
    assert DeclaredTargetShare.from_json(target.to_json()).fingerprint == target.fingerprint
    assert target.alpha == alpha
    assert target.weights[0].value == 0


@pytest.mark.parametrize("alpha", [-0.1, 1.01, True, float("inf"), float("nan"), "0.1"])
def test_invalid_share_strength_rejects(alpha):
    with pytest.raises(ValueError):
        DeclaredTargetShare(**_coefficient_fields(), alpha=alpha)


@pytest.mark.parametrize("values", [(0.0, 0.0), (-0.1, 1.1), (0.2, 0.2)])
def test_invalid_initial_coefficients_are_not_repaired(values):
    fields = _coefficient_fields(values)
    with pytest.raises(ValueError):
        InitialPrior(**fields)


def test_nonfinite_initial_coefficients_reject_at_weight_and_prior_boundaries():
    member = _ref("expert-0")
    nonfinite = float("inf")
    with pytest.raises(ValueError, match="finite"):
        Weight(member=member, value=nonfinite)
    fields = _coefficient_fields()
    fields["weights"] = (
        {"member": member.model_dump(), "value": nonfinite},
        {"member": _ref("expert-1").model_dump(), "value": 0.0},
    )
    with pytest.raises(ValueError, match="finite"):
        InitialPrior(**fields)


def test_coefficient_semantics_and_duplicate_expert_versions_are_explicit():
    fields = _coefficient_fields()
    fields["semantics"] = UnitSemantics(unit="1", meaning="probability-v1")
    with pytest.raises(ValueError, match="coefficient-fraction"):
        InitialPrior(**fields)
    with pytest.raises(ValueError, match="Duplicate"):
        coefficient_weights(
            (Weight(member=_ref("same"), value=0.5), Weight(member=_ref("same", "v2"), value=0.5)),
            "binary64-simplex-v1",
        )
    with pytest.raises(ValueError):
        coefficient_weights(_weights(), cast(Any, "future-profile"))
    with pytest.raises(ValueError):
        coefficient_weights(cast(Any, list(_weights())), "binary64-simplex-v1")


def test_subjective_arithmetic_and_canonical_permutations_have_one_identity():
    definition = _objective()
    identities = set()
    for terms in permutations(definition.terms):
        candidate = ObjectiveWeights.model_validate({**definition.model_dump(), "terms": terms})
        for values in permutations(_values()):
            result = weight_objectives(candidate, values)
            assert result.output is not None
            assert result.output.value == 0.3
            assert result.output.spec.role == "objective"
            assert result.status == "evaluated"
            assert PreferenceEvaluation.from_json(result.to_json()) == result
            identities.add((candidate.fingerprint, result.fingerprint))
    assert len(identities) == 1


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_missing_subjective_input_never_becomes_zero_even_at_zero_weight(status):
    definition = _objective()
    zero_term = ObjectiveWeight(quantity=_meaning(), weight=0.0)
    other_term = ObjectiveWeight(quantity=_meaning("loss-b"), weight=1.0)
    definition = ObjectiveWeights.model_validate({**definition.model_dump(), "terms": (zero_term, other_term)})
    result = weight_objectives(definition, _values(status))
    assert result.status == "insufficient_evidence"
    assert result.output is None
    assert result.inputs[0].status == status
    assert result.inputs[0].value is None


@pytest.mark.parametrize("bad", [(), (_values()[0],), (_values()[0], _values()[0])])
def test_missing_extra_or_duplicate_objective_dimensions_reject(bad):
    with pytest.raises(ValueError):
        weight_objectives(_objective(), bad)


def test_objective_schema_horizon_and_interpretation_cannot_be_substituted():
    wrong = Scalar(spec=_spec("different", "loss"), value=0.2, status="observed")
    with pytest.raises(ValueError):
        weight_objectives(_objective(), (wrong, _values()[1]))
    wrong_meaning = Scalar(spec=_spec("loss-a", "loss", unit="percent"), value=0.2, status="observed")
    with pytest.raises(ValueError, match="schema"):
        weight_objectives(_objective(), (wrong_meaning, _values()[1]))
    terms = (
        ObjectiveWeight(quantity=_meaning(), weight=0.5),
        ObjectiveWeight(quantity=_meaning("b", _ref("day")), weight=0.5),
    )
    with pytest.raises(ValueError, match="horizons"):
        ObjectiveWeights(**_metadata(), terms=terms, output=_spec())
    rank = QuantityMeaning(spec=_spec("rank", "feature"), interpretation="priority_score", horizon=None)
    with pytest.raises(ValueError, match="normalized losses"):
        ObjectiveWeight(quantity=rank, weight=1.0)
    with pytest.raises(ValueError):
        ObjectiveWeights(**_metadata(), terms=_objective().terms, output=_spec(role="loss"))


def test_static_anchor_is_repeated_subjective_penalty_not_an_initial_prior():
    definition = StaticQuadraticAnchor(**_coefficient_fields(), strength=0.4, output=_spec(bounds=(0.0, None)))
    result = evaluate_static_anchor(definition, _weights((0.5, 0.5)))
    assert result.output.value == 0.05
    assert AnchorEvaluation.from_json(result.to_json()) == result
    assert evaluate_static_anchor(definition, _weights((0.5, 0.5))) == result
    assert result.output.value is not None
    assert result.output.value * 2 == 0.1
    zero = StaticQuadraticAnchor.model_validate({**definition.model_dump(), "strength": 0.0})
    assert evaluate_static_anchor(zero, _weights((0.5, 0.5))).output.value == 0
    scaled = StaticQuadraticAnchor.model_validate({**definition.model_dump(), "strength": 4.0})
    assert evaluate_static_anchor(scaled, _weights((0.5, 0.5))).output.value == 0.5
    assert definition.binding != scaled.binding


@pytest.mark.parametrize("strength", [-1.0, True, "1", float("nan"), float("inf")])
def test_invalid_static_penalty_strength_rejects(strength):
    with pytest.raises(ValueError):
        StaticQuadraticAnchor(**_coefficient_fields(), strength=strength, output=_spec())


def test_static_anchor_rejects_versions_bad_output_and_nonzero_operation_underflow():
    definition = StaticQuadraticAnchor(**_coefficient_fields(), strength=1e-320, output=_spec())
    changed = (Weight(member=_ref("expert-0", "v2"), value=0.75), _weights()[1])
    with pytest.raises(ValueError, match="IDs and versions"):
        evaluate_static_anchor(definition, changed)
    tiny = _weights((0.75 + 2**-51, 0.25 - 2**-51))
    with pytest.raises(ValueError, match="underflows"):
        evaluate_static_anchor(definition, tiny)
    with pytest.raises(ValueError, match="objective"):
        StaticQuadraticAnchor(**_coefficient_fields(), strength=0.4, output=_spec(role="loss"))


def test_reference_points_and_judgments_retain_units_without_generating_feedback():
    meaning = QuantityMeaning(
        spec=_spec("duration", "resource", (0.0, None), "s"), interpretation="resource", horizon=None
    )
    point = ReferencePoint(quantity=meaning, value=Scalar(spec=meaning.spec, value=3.0, status="observed"))
    declaration = ReferencePoints(**_metadata(), points=(point,))
    assert ReferencePoints.from_json(declaration.to_json()) == declaration
    judgment = HumanJudgment(**_metadata("judgment"), left=_ref("unit"), right=_ref("integration"), relation="left")
    assert HumanJudgment.from_json(judgment.to_json()) == judgment
    assert "reward" not in judgment.model_dump()
    assert "feedback" not in judgment.model_dump()
    with pytest.raises(ValueError):
        HumanJudgment(**_metadata(), left=_ref("unit"), right=_ref("unit", "v2"), relation="tie")
    with pytest.raises(ValueError):
        ReferencePoint(quantity=meaning, value=Scalar(spec=meaning.spec, value=None, status="unknown"))


@pytest.mark.parametrize("version", [0, 2, 99, True, "1"])
def test_old_and_future_preference_schema_versions_are_not_guessed(version):
    payload = _objective().model_dump()
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        ObjectiveWeights.model_validate(payload)


def test_exact_rational_anchor_oracle_and_numerical_overflow():
    definition = StaticQuadraticAnchor(**_coefficient_fields(), strength=2.0, output=_spec(bounds=(0.0, None)))
    result = evaluate_static_anchor(definition, _weights((0.0, 1.0)))
    expected = Fraction(2) * (Fraction(3, 4) ** 2 + Fraction(3, 4) ** 2)
    assert result.output.value == float(expected)
    huge = StaticQuadraticAnchor(**_coefficient_fields((1.0, 0.0)), strength=1e308, output=_spec(bounds=(0.0, None)))
    with pytest.raises(ValueError, match="overflow"):
        evaluate_static_anchor(huge, _weights((0.0, 1.0)))


def test_versioned_golden_judgment_does_not_acquire_outcome_semantics():
    golden = (
        '{"kind":"human_judgment","left":{"id":"unit","version":"v1"},'
        '"lineage":{"declared_at":null,"source":null,"source_fingerprint":null,"supersedes":null},'
        '"reference":{"id":"judgment","version":"v1"},"relation":"left",'
        '"right":{"id":"integration","version":"v1"},"schema_version":1}'
    )
    judgment = HumanJudgment.from_json(golden)
    assert judgment.to_json() == golden
    assert judgment.left == _ref("unit")
    duplicate = golden.replace('"relation":"left"', '"relation":"right","relation":"left"')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        HumanJudgment.from_json(duplicate)


def test_unchecked_preference_receipts_cannot_claim_completed_or_empirical_output():
    result = weight_objectives(_objective(), _values())
    assert result.output is not None
    with pytest.raises(ValueError, match="observed support"):
        result.model_copy(update={"status": "insufficient_evidence", "output": None}).to_json()
    empirical = Scalar(spec=_spec(role="loss"), value=0.3, status="observed")
    with pytest.raises(ValueError, match="specification"):
        result.model_copy(update={"output": empirical}).to_json()
    anchor = StaticQuadraticAnchor(**_coefficient_fields(), strength=0.4, output=_spec())
    receipt = evaluate_static_anchor(anchor, _weights())
    with pytest.raises(ValueError, match="observed objective"):
        receipt.model_copy(update={"output": empirical}).to_json()


@pytest.mark.parametrize("route", ["python", "json", "reader"])
@pytest.mark.parametrize("kind", ["objective", "anchor"])
def test_receipt_reader_recomputes_actual_math_from_full_specification(kind, route):
    if kind == "objective":
        result = weight_objectives(_objective(), _values())
    else:
        definition = StaticQuadraticAnchor(**_coefficient_fields(), strength=0.4, output=_spec())
        result = evaluate_static_anchor(definition, _weights((0.5, 0.5)))
    assert result.output is not None
    payload = result.model_dump(mode="json")
    payload["output"]["value"] = 0.9
    assert payload["definition"] == result.definition.model_dump(mode="json")
    read = {
        "python": type(result).model_validate,
        "json": type(result).model_validate_json,
        "reader": type(result).from_json,
    }[route]
    with pytest.raises(ValueError, match="specification"):
        read(payload if route == "python" else json.dumps(payload))
    forged = result.model_copy(update={"output": Scalar(spec=result.output.spec, value=0.9, status="observed")})
    with pytest.raises(ValueError):
        forged.to_json()


def test_binding_or_specification_changes_cannot_preserve_receipt_identity():
    result = weight_objectives(_objective(), _values())
    payload = result.model_dump(mode="json")
    payload["definition"]["definition_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="specification"):
        PreferenceEvaluation.model_validate(payload)
    payload = result.model_dump(mode="json")
    del payload["specification"]
    with pytest.raises(ValueError):
        PreferenceEvaluation.model_validate(payload)
