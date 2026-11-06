"""Check finite evidence, conservative fallback and immutable input semantics."""

import json
from itertools import permutations

import pytest

from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.feasibility import (
    AlternativeFeasibility,
    ConstraintEvidence,
    FallbackDecision,
    FiniteAlternative,
    FiniteFeasibilityRequest,
    FiniteFeasibilityResult,
    ScalarConstraint,
    check_finite_feasibility,
    select_fallback,
)


def _ref(name, version="v1"):
    return Reference(id=name, version=version)


def _metadata(name="duration-limit"):
    return {
        "schema_version": 1,
        "reference": _ref(name),
        "lineage": DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
    }


def _meaning(unit="s", version="v1"):
    return QuantityMeaning(
        spec=ScalarSpec(
            dimension=_ref("duration", version),
            semantics=UnitSemantics(unit=unit, meaning="test-duration-v1"),
            role="resource",
            minimum=0.0,
        ),
        interpretation="resource",
        horizon=None,
    )


def _constraint(lower=None, upper=3.0, name="duration-limit"):
    return ScalarConstraint(**_metadata(name), quantity=_meaning(), lower=lower, upper=upper)


def _row(name="unit", value=2.0, status="observed"):
    return FiniteAlternative(member=_ref(name), values=(Scalar(spec=_meaning().spec, value=value, status=status),))


def _request(rows=None, constraints=None, complete=True):
    return FiniteFeasibilityRequest(
        schema_version=1,
        universe=(_ref("unit"), _ref("integration")),
        universe_complete=complete,
        alternatives=(_row(), _row("integration", 4.0)) if rows is None else rows,
        constraints=(_constraint(),) if constraints is None else constraints,
    )


def test_actual_ci_scalar_constraints_retain_raw_evidence_and_qualify_baseline():
    request = _request()
    before = request.to_json()
    result = check_finite_feasibility(request)
    assert result.status == "feasible"
    assert result.feasible_members == (_ref("unit"),)
    assert request.to_json() == before
    assert FiniteFeasibilityResult.from_json(result.to_json()) == result
    qualified = select_fallback(result, _ref("unit"))
    assert qualified.kind == "baseline" and qualified.member == _ref("unit")
    assert FallbackDecision.from_json(qualified.to_json()) == qualified
    assert select_fallback(result, _ref("integration")).reason == "unqualified_baseline"
    assert select_fallback(result, _ref("unit", "v2")).kind == "abstain"
    assert select_fallback(result).reason == "no_baseline"
    assert result.alternatives[0].evidence[0].value is not None
    assert result.alternatives[0].evidence[0].value.value == 4.0


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_null_states_and_missing_rows_do_not_fabricate_feasibility(status):
    request = _request((_row("unit", None, status),))
    result = check_finite_feasibility(request)
    assert result.status == "insufficient_evidence" and not result.feasible_members
    by_member = {row.member.id: row for row in result.alternatives}
    assert by_member["unit"].evidence[0].value is not None
    assert by_member["unit"].evidence[0].value.status == status
    assert by_member["integration"].present is False and not by_member["integration"].evidence
    assert select_fallback(result, _ref("unit")).kind == "abstain"


def test_missing_dimension_is_distinct_from_known_zero_and_negative_values():
    missing = FiniteAlternative(member=_ref("unit"), values=())
    result = check_finite_feasibility(_request((missing,)))
    evidence = next(row for row in result.alternatives if row.member.id == "unit").evidence[0]
    assert evidence.reason == "missing_dimension" and evidence.value is None
    zero = check_finite_feasibility(_request((_row(value=0.0),)))
    assert zero.feasible_members == (_ref("unit"),)
    signed = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_ref("delta"), semantics=UnitSemantics(unit="s", meaning="duration-change-v1"), role="objective"
        ),
        interpretation="raw_outcome",
        horizon=None,
    )
    constraint = ScalarConstraint(**_metadata(), quantity=signed, lower=-2.0, upper=0.0)
    value = Scalar(spec=signed.spec, value=-1.0, status="observed")
    request = _request((FiniteAlternative(member=_ref("unit"), values=(value,)),), constraints=(constraint,))
    assert check_finite_feasibility(request).feasible_members == (_ref("unit"),)


def test_complete_rejections_differ_from_incomplete_catalogue_and_empty_universe():
    rows = (_row(value=4.0), _row("integration", 5.0))
    assert check_finite_feasibility(_request(rows)).status == "infeasible"
    assert check_finite_feasibility(_request(rows, complete=False)).status == "insufficient_evidence"
    for complete, expected in ((True, "infeasible"), (False, "insufficient_evidence")):
        request = FiniteFeasibilityRequest(
            schema_version=1, universe=(), universe_complete=complete, alternatives=(), constraints=()
        )
        assert check_finite_feasibility(request).status == expected


def test_valid_conflicting_constraints_are_not_malformed_or_repaired():
    lower = _constraint(lower=4.0, upper=None, name="minimum")
    upper = _constraint(upper=3.0, name="maximum")
    result = check_finite_feasibility(_request((), constraints=(lower, upper)))
    assert result.status == "infeasible"
    assert len(result.conflicting_constraints) == 2
    assert not result.feasible_members
    with pytest.raises(ValueError, match="lower bound"):
        _constraint(lower=4.0, upper=3.0)
    with pytest.raises(ValueError, match="at least one"):
        _constraint(lower=None, upper=None)


def test_constraint_can_conflict_with_declared_measurement_bounds():
    impossible = _constraint(lower=None, upper=-1.0)
    result = check_finite_feasibility(_request((), constraints=(impossible,)))
    assert result.status == "infeasible" and result.conflicting_constraints == (impossible.binding,)


def test_schema_units_horizon_and_member_versions_are_never_inferred():
    wrong = FiniteAlternative(
        member=_ref("unit"), values=(Scalar(spec=_meaning("ms").spec, value=2.0, status="observed"),)
    )
    with pytest.raises(ValueError, match="schema"):
        _request((wrong,))
    wrong_version = FiniteAlternative(member=_ref("unit", "v2"), values=())
    with pytest.raises(ValueError, match="exact members"):
        _request((wrong_version,))
    conflicting = ScalarConstraint(**_metadata("other-limit"), quantity=_meaning(version="v2"), lower=None, upper=3.0)
    with pytest.raises(ValueError, match="quantity schema"):
        _request(constraints=(_constraint(), conflicting))
    with pytest.raises(ValueError, match="Duplicate"):
        _request((_row(), _row()))


def test_equality_bound_and_unconstrained_values_are_explicit():
    equal = _constraint(lower=2.0, upper=2.0)
    assert check_finite_feasibility(_request(constraints=(equal,))).feasible_members == (_ref("unit"),)
    assert check_finite_feasibility(_request(constraints=())).feasible_members == (_ref("integration"), _ref("unit"))
    lower = _constraint(lower=3.0, upper=None)
    result = check_finite_feasibility(_request(constraints=(lower,)))
    unit = next(row for row in result.alternatives if row.member.id == "unit")
    assert unit.evidence[0].reason == "below_lower"


def test_metamorphic_set_ordering_and_repeated_checks_are_deterministic():
    request = _request()
    identities = set()
    for universe in permutations(request.universe):
        for rows in permutations(request.alternatives):
            candidate = FiniteFeasibilityRequest.model_validate(
                {**request.model_dump(), "universe": universe, "alternatives": rows}
            )
            result = check_finite_feasibility(candidate)
            assert check_finite_feasibility(candidate) == result
            identities.add((candidate.fingerprint, result.fingerprint))
    assert len(identities) == 1


@pytest.mark.parametrize("version", [0, 2, 99, True, "1"])
def test_unknown_schemas_never_migrate_implicitly(version):
    payload = _request().model_dump()
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        FiniteFeasibilityRequest.model_validate(payload)


@pytest.mark.parametrize("value", [True, "1", float("inf"), float("nan")])
def test_nonfinite_or_coerced_constraint_limits_reject(value):
    with pytest.raises(ValueError):
        _constraint(upper=value)


def test_receipt_status_cannot_claim_a_completed_comparison_without_observation():
    missing = Scalar(spec=_meaning().spec, value=None, status="unknown")
    with pytest.raises(ValueError):
        ConstraintEvidence(constraint=_constraint().binding, value=missing, status="pass", reason="satisfied")
    with pytest.raises(ValueError):
        ConstraintEvidence(
            constraint=_constraint().binding, value=missing, status="unknown", reason="missing_dimension"
        )
    with pytest.raises(ValueError):
        ConstraintEvidence(constraint=_constraint().binding, value=None, status="fail", reason="satisfied")
    with pytest.raises(ValueError):
        AlternativeFeasibility(member=_ref("unit"), present=False, evidence=(), status="feasible")
    result = check_finite_feasibility(_request((_row(value=None, status="unknown"),)))
    with pytest.raises(ValueError):
        result.model_copy(update={"status": "feasible"}).to_json()


def test_fallback_cannot_claim_an_unqualified_execution():
    result = check_finite_feasibility(_request())
    fallback = select_fallback(result)
    with pytest.raises(ValueError):
        fallback.model_copy(update={"member": _ref("unit")}).to_json()


def test_declared_schema_maximum_and_receipt_conflicts_are_checked():
    bounded = QuantityMeaning(
        spec=ScalarSpec(
            dimension=_ref("quality"),
            semantics=UnitSemantics(unit="1", meaning="quality-v1"),
            role="objective",
            maximum=1.0,
        ),
        interpretation="raw_outcome",
        horizon=None,
    )
    constraint = ScalarConstraint(**_metadata(), quantity=bounded, lower=2.0, upper=None)
    assert check_finite_feasibility(_request((), constraints=(constraint,))).status == "infeasible"
    result = check_finite_feasibility(_request())
    evidence = next(row for row in result.alternatives if row.status == "feasible").evidence
    with pytest.raises(ValueError, match="absent alternative"):
        AlternativeFeasibility(member=_ref("unit"), present=False, evidence=evidence, status="feasible")
    with pytest.raises(ValueError, match="complete request"):
        result.model_copy(
            update={"conflicting_constraints": (_constraint().binding,), "status": "infeasible"}
        ).to_json()


@pytest.mark.parametrize("route", ["python", "json", "reader", "fallback"])
def test_finite_receipt_cannot_admit_four_seconds_under_three_second_limit(route):
    request = _request()
    result = check_finite_feasibility(request)
    payload = result.model_dump(mode="json")
    for row in payload["alternatives"]:
        if row["member"]["id"] == "integration":
            row["status"] = "feasible"
            row["evidence"][0]["status"] = "pass"
            row["evidence"][0]["reason"] = "satisfied"
    assert payload["request_fingerprint"] == request.fingerprint
    if route == "fallback":
        forged = result.model_copy(
            update={
                "alternatives": tuple(AlternativeFeasibility.model_validate(row) for row in payload["alternatives"])
            }
        )
        with pytest.raises(ValueError, match="complete request"):
            select_fallback(forged, _ref("integration"))
    else:
        read = {
            "python": FiniteFeasibilityResult.model_validate,
            "json": FiniteFeasibilityResult.model_validate_json,
            "reader": FiniteFeasibilityResult.from_json,
        }[route]
        with pytest.raises(ValueError, match="complete request"):
            read(payload if route == "python" else json.dumps(payload))


@pytest.mark.parametrize("rows", [(), None])
def test_conflicting_canonical_signed_zero_schemas_reject_both_orders_even_without_rows(rows):
    first = _constraint(name="first")
    second = ScalarConstraint.model_validate(
        {
            **_constraint(name="second").model_dump(),
            "quantity": _meaning().model_copy(update={"spec": _meaning().spec.model_copy(update={"minimum": -0.0})}),
        }
    )
    for constraints in permutations((first, second)):
        with pytest.raises(ValueError, match="quantity schema"):
            _request(rows=rows, constraints=constraints)


@pytest.mark.parametrize("field", ["request_fingerprint", "universe_complete"])
def test_finite_reader_binds_request_digest_and_completeness(field):
    result = check_finite_feasibility(_request())
    payload = result.model_dump(mode="json")
    payload[field] = "sha256:" + "f" * 64 if field == "request_fingerprint" else False
    with pytest.raises(ValueError, match="complete request"):
        FiniteFeasibilityResult.model_validate(payload)
