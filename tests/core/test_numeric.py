"""Independent numeric ingress, missingness and bounded simplex invariants."""

import itertools
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError

from coleman.core.actions import ChoiceAction, ChoiceSpace, SubsetSpace, VectorSpace
from coleman.core.base import CoreContract, CoreImmutable, VersionedContract
from coleman.core.decisions import DecisionRequest
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import SIMPLEX_PROFILE, Binary64, Scalar, ScalarSpec, validate_simplex
from coleman.core.primitives import MAX_DIMENSIONS, MAX_MEMBERS, canonical_json, fingerprint
from coleman.core.units import UnitSemantics


class NumericContract(CoreContract):
    value: Binary64


class NestedNumeric(CoreImmutable):
    value: Binary64


def scalar_spec(**changes):
    return ScalarSpec.model_validate(
        {
            "dimension": {"id": "net_increment", "version": "1"},
            "semantics": {"unit": "credit", "meaning": "synthetic-net-increment-v1"},
            "role": "objective",
            **changes,
        }
    )


@pytest.mark.parametrize("value", [0.0, -0.0, -2.5, 0, 1, 2**60, Decimal("0.1"), Decimal("-0")])
def test_binary64_declared_projection_preserves_represented_value(value):
    actual = TypeAdapter(Binary64).validate_python(value)
    assert type(actual) is float
    assert actual.hex() == float(value).hex()


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "0.1",
        b"0.1",
        Fraction(1, 3),
        2**53 + 1,
        2**60 + 1,
        10**400,
        Decimal("1e-400"),
        Decimal("-1e-400"),
        Decimal("1e309"),
        Decimal("NaN"),
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_binary64_rejects_lossy_integer_underflow_and_unsupported_kinds(value):
    with pytest.raises(ValueError):
        TypeAdapter(Binary64).validate_python(value)


def test_custom_callbacks_and_numeric_subclasses_cannot_execute_implicitly():
    calls = []

    class Foreign:
        def __float__(self):
            calls.append("executed")
            return 0.5

    class FloatSubclass(float):
        pass

    for value in (Foreign(), FloatSubclass(0.5)):
        with pytest.raises(ValueError):
            TypeAdapter(Binary64).validate_python(value)
    assert calls == []


@pytest.mark.parametrize("contract", [NumericContract, NestedNumeric])
@pytest.mark.parametrize("raw", ["1e-400", "-1e-400", "9007199254740993", "1e309", "NaN", "true", '"0.5"'])
def test_all_supported_model_json_readers_preserve_numeric_guards(contract, raw):
    prefix = '"schema_version":1,' if contract is NumericContract else ""
    payload = "{" + prefix + '"value":' + raw + "}"
    with pytest.raises(ValueError):
        contract.model_validate_json(payload)
    if contract is NumericContract:
        with pytest.raises(ValueError):
            contract.from_json(payload)


def test_guarded_json_rejects_duplicate_keys_and_field_relaxation():
    for payload in (
        '{"schema_version":1,"value":0.0,"value":1.0}',
        '{"schema_version":1,"value":0.0,"extra":1}',
    ):
        with pytest.raises(ValueError):
            NumericContract.model_validate_json(payload)
    for extra in ("allow", "ignore"):
        with pytest.raises(ValueError, match="forbid"):
            NumericContract.model_validate_json('{"schema_version":1,"value":0.0}', extra=extra)
    sample = bytearray(b'{"schema_version":1,"value":0.1}')
    result = NumericContract.model_validate_json(sample, by_name=True, context={"host": "test"})
    assert result.to_json() == '{"schema_version":1,"value":0.1}'
    assert NumericContract.from_json(result.to_json()) == result


def guarded_field_cases():
    reference = Reference(id="public-ci-member", version="1")
    space = ChoiceSpace(feasible_set=reference, members=(reference,))
    choice = ChoiceAction(schema_version=1, space=space, chosen=reference)
    request = DecisionRequest(
        schema_version=1, task=reference, decision_id="build-1", round=0, space=space, objective=reference
    )
    vector = VectorSpace(feasible_set=reference, coordinates=(), signed=False)
    subset = SubsetSpace(feasible_set=reference, members=(), minimum_selected=0, maximum_selected=0)
    return (
        (choice, "schema_version", True),
        (choice, "schema_version", 1.0),
        (request, "round", True),
        (vector, "signed", 1),
        (subset, "minimum_selected", False),
    )


@pytest.mark.parametrize("sample,field,invalid", guarded_field_cases())
def test_new_readers_cannot_relax_version_count_or_boolean_contracts(sample, field, invalid):
    contract = type(sample)
    valid = sample.model_dump(mode="json")
    corrupted = {**valid, field: invalid}
    for payload in (valid, corrupted):
        with pytest.raises(ValueError, match="strict fields"):
            contract.model_validate(payload, strict=False)
        with pytest.raises(ValueError, match="strict fields"):
            contract.model_validate_json(canonical_json(payload), strict=False)
    with pytest.raises(ValueError):
        contract.model_validate(corrupted)
    with pytest.raises(ValueError):
        contract.model_validate_json(canonical_json(corrupted))
    if isinstance(sample, CoreContract):
        with pytest.raises(ValueError):
            contract.from_json(canonical_json(corrupted))
    assert contract.model_validate_json(canonical_json(valid)) == sample


def test_additional_strict_python_mode_does_not_promise_json_array_to_tuple_coercion():
    assert NumericContract.model_validate_json('{"schema_version":1,"value":0.5}', strict=True).value == 0.5
    choice = guarded_field_cases()[0][0]
    assert type(choice).model_validate(choice.model_dump(), strict=True) == choice
    with pytest.raises(ValueError):
        type(choice).model_validate_json(choice.model_dump_json(), strict=True)


def test_legacy_pydantic_strictness_options_keep_their_original_behavior():
    assert VersionedContract.model_validate({"schema_version": True}, strict=False).schema_version == 1
    assert VersionedContract.model_validate_json('{"schema_version":1.0}', strict=False).schema_version == 1


@pytest.mark.parametrize("version", [None, 0, 2, 999, "1", True])
def test_new_contract_versions_fail_closed(version):
    raw = canonical_json({"schema_version": version, "value": 0.0})
    for reader in (NumericContract.from_json, NumericContract.model_validate_json):
        with pytest.raises(ValueError):
            reader(raw)
    with pytest.raises(ValueError):
        NumericContract.from_json('{"value":0.0}')


def test_observed_zero_signed_negative_and_null_states_remain_distinct():
    spec = scalar_spec(minimum=-2.0, maximum=1.0)
    zero = Scalar(spec=spec, value=0.0, status="observed")
    negative = Scalar(spec=spec, value=-2.0, status="observed")
    assert zero.value == 0.0 and negative.value == -2.0
    for status in ("unknown", "unavailable", "not_applicable"):
        missing = Scalar(spec=spec, value=None, status=status)
        assert missing.spec == zero.spec and missing.status != zero.status
        with pytest.raises(ValueError):
            Scalar(spec=spec, value=0.0, status=status)
    for payload in ({"spec": spec, "status": "unknown"}, {"spec": spec, "value": None, "status": "observed"}):
        with pytest.raises(ValueError):
            Scalar.model_validate(payload)
    with pytest.raises(ValueError):
        Scalar(spec=spec, value=math.nextafter(-2.0, -math.inf), status="observed")
    with pytest.raises(ValueError):
        Scalar(spec=spec, value=math.nextafter(1.0, math.inf), status="observed")
    with pytest.raises(ValueError):
        scalar_spec(minimum=2.0, maximum=1.0)


def test_scalar_roles_meanings_and_zero_representation_are_not_conflated():
    objective = scalar_spec()
    resource = scalar_spec(role="resource")
    assert objective != resource
    assert scalar_spec(role="action").role == "action"
    other = scalar_spec(semantics={"unit": "credit", "meaning": "gross-increment-v1"})
    assert other != objective
    positive = Scalar(spec=objective, value=0.0, status="observed")
    negative = Scalar(spec=objective, value=-0.0, status="observed")
    assert fingerprint(positive.model_dump(mode="json")) != fingerprint(negative.model_dump(mode="json"))
    for model in (positive, negative):
        restored = Scalar.model_validate_json(model.model_dump_json())
        assert isinstance(restored.value, float) and isinstance(model.value, float)
        assert restored.value.hex() == model.value.hex()


def test_unchecked_nested_values_revalidate_and_boundaries_are_frozen():
    corrupt = scalar_spec().model_copy(update={"minimum": 2.0, "maximum": 1.0})
    with pytest.raises(ValueError):
        Scalar(spec=corrupt, value=0.0, status="observed")
    sample = Scalar(spec=scalar_spec(), value=0.0, status="observed")
    with pytest.raises(ValidationError):
        sample.value = 1.0
    unchecked = NumericContract.model_construct(schema_version=1, value=True)
    with pytest.raises(ValueError):
        unchecked.to_json()
    for contract in (NumericContract, NestedNumeric):
        for extra in ("allow", "ignore"):
            with pytest.raises(ValueError, match="forbid"):
                contract.model_validate({"value": 0.0, "surprise": 1}, extra=extra)


def test_exact_simplex_residual_boundary_and_permutations():
    boundary = math.ldexp(1.0, -49)
    below = math.nextafter(boundary, 0.0)
    above = math.nextafter(boundary, math.inf)
    assert SIMPLEX_PROFILE == "binary64-simplex-v1"
    for vector in ((0.1, 0.2, 0.7), (1 / 3,) * 3, (1.0, boundary), (0.0, 1.0, -0.0)):
        exact_sum = sum((Fraction(*number.as_integer_ratio()) for number in vector), Fraction())
        assert abs(exact_sum - 1) <= Fraction(1, 2**49)
        for permutation in itertools.permutations(vector):
            before = tuple(value.hex() for value in permutation)
            validate_simplex(permutation)
            assert tuple(value.hex() for value in permutation) == before
    with pytest.raises(ValueError):
        validate_simplex((1.0, above))
    validate_simplex((1.0, below))
    # Rounded fsum alone cannot decide the exact stored residual at the boundary.
    assert math.fsum((1.0, above)) == math.fsum((1.0, boundary))


@pytest.mark.parametrize("weights", [(), (0.0,), (1,), (True,), (math.nan,), (math.inf,), (-1e-300, 1.0), (1.1,)])
def test_invalid_simplex_shape_kind_support_and_range(weights):
    with pytest.raises(ValueError):
        validate_simplex(weights)


def test_simplex_limits_and_decimal_context_do_not_change_validation():
    vector = (1 / MAX_MEMBERS,) * MAX_MEMBERS
    validate_simplex(vector)
    boundary_component = 1 / MAX_MEMBERS + math.ldexp(1.0, -49)
    boundary = vector[:-1] + (boundary_component,)
    validate_simplex(boundary)
    validate_simplex(vector[:-1] + (math.nextafter(boundary_component, 0.0),))
    with pytest.raises(ValueError):
        validate_simplex(vector[:-1] + (math.nextafter(boundary_component, math.inf),))
    with localcontext() as context:
        context.prec = 2
        validate_simplex((0.1, 0.2, 0.7))
    with pytest.raises(ValueError):
        validate_simplex(vector + (0.0,))
    with pytest.raises(ValueError):
        validate_simplex(cast(tuple[float, ...], [1.0]))
    assert MAX_DIMENSIONS == 64


def test_references_canonicalize_sets_and_reject_duplicates_or_corrupt_instances():
    a, b = Reference(id="a", version="1"), Reference(id="b", version="2")
    assert ordered_references((b, a)) == (a, b)
    assert ordered_references(()) == ()
    for values in ((a, a), (a, Reference(id="a", version="2")), (a,) * (MAX_MEMBERS + 1)):
        with pytest.raises(ValueError):
            ordered_references(values)
    with pytest.raises(ValueError):
        ordered_references((a.model_copy(update={"id": ""}),))
    with pytest.raises(ValueError):
        ordered_references(cast(tuple[Reference, ...], [a]))
    assert UnitSemantics(unit="s", meaning="duration-v1") != UnitSemantics(unit="ms", meaning="duration-v1")
