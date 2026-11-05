"""Independent finite action spaces, feasibility examples and frozen v1 wires."""

import hashlib
import itertools
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from coleman.core.actions import (
    AbstainAction,
    AcquisitionAction,
    AcquisitionSpace,
    Action,
    ChoiceAction,
    ChoiceSpace,
    MixtureAction,
    MixtureSpace,
    RankingAction,
    RankingSpace,
    SlateAction,
    SlateSpace,
    SubsetAction,
    SubsetSpace,
    VectorAction,
    VectorSpace,
    Weight,
    selected_members,
    space_members,
)
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarRole, ScalarSpec
from coleman.core.units import UnitSemantics

FIXTURES = Path(__file__).parents[1] / "fixtures/core/v1"


def ref(name: str, version: str = "v1") -> Reference:
    return Reference(id=name, version=version)


def scalar_spec(
    name: str = "allocation",
    *,
    role: ScalarRole = "action",
    unit: str = "1",
    meaning: str = "resource-share",
    minimum: float | None = None,
    maximum: float | None = None,
) -> ScalarSpec:
    return ScalarSpec(
        dimension=ref(name),
        semantics=UnitSemantics(unit=unit, meaning=meaning),
        role=role,
        minimum=minimum,
        maximum=maximum,
    )


def test_rankings_are_all_and_only_permutations_of_the_named_versioned_universe():
    members = (ref("a"), ref("b"), ref("c"))
    fingerprints = set()
    for order in itertools.permutations(members):
        variants = {
            RankingAction(
                schema_version=1,
                space=RankingSpace(members=universe, feasible_set=ref("permutation")),
                order=order,
            ).fingerprint
            for universe in itertools.permutations(members)
        }
        assert len(variants) == 1
        fingerprints.update(variants)
    assert len(fingerprints) == 6
    space = RankingSpace(members=members, feasible_set=ref("permutation"))
    for invalid in (
        members[:-1],
        (*members, ref("d")),
        (ref("a"), ref("a"), ref("c")),
        (ref("a", "v2"), ref("b"), ref("c")),
    ):
        with pytest.raises(ValueError):
            RankingAction(schema_version=1, space=space, order=invalid)


@pytest.mark.parametrize("space_type", [RankingSpace, ChoiceSpace, MixtureSpace, AcquisitionSpace])
def test_space_duplicates_do_not_collapse_even_when_the_versions_match(space_type):
    extras = {"fidelities": (ref("low"),)} if space_type is AcquisitionSpace else {}
    if space_type is MixtureSpace:
        extras = {"profile": "binary64-simplex-v1"}
    for duplicate in ((ref("a"), ref("a")), (ref("a"), ref("a", "v2"))):
        with pytest.raises(ValueError):
            space_type(members=duplicate, feasible_set=ref("feasible"), **extras)


def test_vector_coordinate_schema_preserves_signed_zero_in_its_bounds():
    declared = scalar_spec(minimum=0.0, maximum=1.0)
    changed = scalar_spec(minimum=-0.0, maximum=1.0)
    space = VectorSpace(coordinates=(declared,), signed=False, feasible_set=ref("bounded-vector"))
    with pytest.raises(ValueError, match="spec|schema"):
        VectorAction(schema_version=1, space=space, values=(Scalar(spec=changed, value=0.5, status="observed"),))


def test_subsets_have_set_identity_while_slate_order_remains_a_decision():
    members = (ref("a"), ref("b"), ref("c"))
    subset_space = SubsetSpace(members=members, feasible_set=ref("bounded"), minimum_selected=0, maximum_selected=2)
    slate_space = SlateSpace(members=members, feasible_set=ref("bounded"), minimum_selected=0, maximum_selected=2)
    subsets = set()
    slates = set()
    for size in range(3):
        for selected in itertools.combinations(members, size):
            subset_variants = {
                SubsetAction(schema_version=1, space=subset_space, selected=order).fingerprint
                for order in itertools.permutations(selected)
            }
            assert len(subset_variants) == 1
            subsets.update(subset_variants)
        for selected in itertools.permutations(members, size):
            slates.add(SlateAction(schema_version=1, space=slate_space, selected=selected).fingerprint)
    assert len(subsets) == 7
    assert len(slates) == 10
    for action_type, space in ((SubsetAction, subset_space), (SlateAction, slate_space)):
        for invalid in ((ref("a"), ref("a")), members, (ref("outside"),), (ref("a", "v2"),)):
            with pytest.raises(ValueError):
                action_type.model_validate({"schema_version": 1, "space": space, "selected": invalid})


@pytest.mark.parametrize("space_type", [SubsetSpace, SlateSpace])
@pytest.mark.parametrize("minimum,maximum", [(-1, 1), (1, 0), (0, 3), (True, 1), (0, 1.0)])
def test_cardinality_limits_are_explicit_inclusive_integers(space_type, minimum, maximum):
    with pytest.raises(ValueError):
        space_type(
            members=(ref("a"), ref("b")),
            feasible_set=ref("bounded"),
            minimum_selected=minimum,
            maximum_selected=maximum,
        )


def test_mixture_retains_complete_weights_and_never_repairs_or_imputes_them():
    space = MixtureSpace(members=(ref("b"), ref("a")), feasible_set=ref("simplex"), profile="binary64-simplex-v1")
    weights = (Weight(member=ref("a"), value=0.25), Weight(member=ref("b"), value=0.75))
    result = MixtureAction(schema_version=1, space=space, weights=weights)
    assert result.weights == weights
    assert result.fingerprint == MixtureAction(schema_version=1, space=space, weights=weights[::-1]).fingerprint
    assert tuple(weight.value for weight in result.weights) == (0.25, 0.75)
    for invalid in (
        weights[:1],
        (weights[0], weights[0]),
        (Weight(member=ref("a"), value=0.25), Weight(member=ref("b", "v2"), value=0.75)),
        (Weight(member=ref("a"), value=-0.25), Weight(member=ref("b"), value=1.25)),
        (Weight(member=ref("a"), value=0.5), Weight(member=ref("b"), value=0.25)),
    ):
        with pytest.raises(ValueError):
            MixtureAction(schema_version=1, space=space, weights=invalid)


def test_signed_vectors_distinguish_zero_valid_negative_and_missing_values():
    coordinate = scalar_spec(minimum=-1.0, maximum=1.0)
    signed = VectorSpace(coordinates=(coordinate,), signed=True, feasible_set=ref("signed-interval"))
    for value in (-1.0, -0.25, 0.0, 1.0):
        action = VectorAction(
            schema_version=1, space=signed, values=(Scalar(spec=coordinate, value=value, status="observed"),)
        )
        assert action.values[0].value == value
    unsigned = VectorSpace(coordinates=(coordinate,), signed=False, feasible_set=ref("unsigned-interval"))
    with pytest.raises(ValueError):
        VectorAction(
            schema_version=1, space=unsigned, values=(Scalar(spec=coordinate, value=-0.25, status="observed"),)
        )
    for status in ("unknown", "unavailable", "not_applicable"):
        with pytest.raises(ValueError):
            VectorAction(schema_version=1, space=signed, values=(Scalar(spec=coordinate, value=None, status=status),))
    for incompatible in (
        scalar_spec(unit="s"),
        scalar_spec(meaning="unrelated-ratio"),
        scalar_spec(role="resource"),
        scalar_spec(name="different-coordinate"),
    ):
        with pytest.raises(ValueError):
            VectorAction(
                schema_version=1, space=signed, values=(Scalar(spec=incompatible, value=0.0, status="observed"),)
            )
    with pytest.raises(ValueError):
        VectorAction(schema_version=1, space=signed, values=())


def test_fidelity_is_a_versioned_choice_and_abstention_is_explicit():
    space = AcquisitionSpace(
        members=(ref("sample-a"),), fidelities=(ref("screen"), ref("full")), feasible_set=ref("budget")
    )
    action = AcquisitionAction(schema_version=1, space=space, chosen=ref("sample-a"), fidelity=ref("screen"))
    assert selected_members(action) == (ref("sample-a"),)
    for chosen, fidelity in ((ref("sample-b"), ref("screen")), (ref("sample-a"), ref("screen", "v2"))):
        with pytest.raises(ValueError):
            AcquisitionAction(schema_version=1, space=space, chosen=chosen, fidelity=fidelity)
    empty = ChoiceSpace(members=(), feasible_set=ref("no-feasible-alternatives"))
    abstained = AbstainAction(schema_version=1, space=empty, reason="no-feasible-alternatives")
    assert space_members(abstained.space) == ()
    with pytest.raises(ValueError):
        ChoiceAction(schema_version=1, space=empty, chosen=ref("invented"))


def test_public_action_dispatch_rejects_wrong_shapes_and_unknown_kind():
    adapter = TypeAdapter(Action)
    space = ChoiceSpace(members=(ref("a"),), feasible_set=ref("choice"))
    action = ChoiceAction(schema_version=1, space=space, chosen=ref("a"))
    assert adapter.validate_python(action.model_dump(mode="python")) == action
    for patch in ({"kind": "ranking"}, {"kind": "new-action-v99"}, {"order": (ref("a"),)}):
        with pytest.raises(ValueError):
            adapter.validate_python({**action.model_dump(mode="python"), **patch})
    with pytest.raises(ValueError):
        selected_members(
            RankingAction(
                schema_version=1, space=RankingSpace(members=(ref("a"),), feasible_set=ref("rank")), order=(ref("a"),)
            )
        )
    assert space_members(space) == (ref("a"),)


def test_unsafe_model_copies_are_revalidated_before_serialization():
    valid = ChoiceAction(
        schema_version=1, space=ChoiceSpace(members=(ref("a"),), feasible_set=ref("choice")), chosen=ref("a")
    )
    for unsafe in (valid.model_copy(update={"chosen": ref("outside")}), valid.model_copy(update={"schema_version": 2})):
        with pytest.raises(ValueError):
            unsafe.to_json()
        with pytest.raises(ValueError):
            _ = unsafe.fingerprint


@pytest.mark.parametrize("value", [True, float("inf"), float("-inf"), float("nan"), "0.25"])
def test_allocation_values_do_not_coerce_booleans_strings_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        Weight(member=ref("a"), value=value)


def test_empty_rank_and_vector_are_explicit_actions_but_empty_mixture_is_not():
    rank = RankingAction(schema_version=1, space=RankingSpace(members=(), feasible_set=ref("empty-rank")), order=())
    vector = VectorAction(
        schema_version=1, space=VectorSpace(coordinates=(), signed=True, feasible_set=ref("empty-vector")), values=()
    )
    assert rank.kind == "ranking" and vector.kind == "vector"
    assert space_members(vector.space) == ()
    with pytest.raises(ValueError):
        MixtureAction(
            schema_version=1,
            space=MixtureSpace(members=(), feasible_set=ref("empty-simplex"), profile="binary64-simplex-v1"),
            weights=(),
        )


@pytest.mark.parametrize(
    "filename,model",
    [
        ("choice-action.json", ChoiceAction),
        ("ranking-action.json", RankingAction),
        ("mixture-action.json", MixtureAction),
    ],
)
def test_version_one_golden_bytes_and_independent_sha256(filename, model):
    wire = (FIXTURES / filename).read_bytes().removesuffix(b"\n")
    digest = json.loads((FIXTURES / "digests.json").read_text())[filename]
    assert hashlib.sha256(wire).hexdigest() == digest
    value = model.from_json(wire)
    assert value.to_json().encode() == wire
    assert value.fingerprint == "sha256:" + digest
    assert model.model_validate_json(wire) == value
    assert model.from_json(value.to_json()) == value
    payload = json.loads(wire)
    for version in (None, 0, 2, True, "1"):
        invalid = {**payload, "schema_version": version}
        if version is None:
            invalid.pop("schema_version")
        with pytest.raises(ValueError):
            model.from_json(json.dumps(invalid))
