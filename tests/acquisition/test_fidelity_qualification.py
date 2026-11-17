"""Independent resource-schedule binding for empirical fidelity evidence."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from coleman.acquisition.contracts import AllocationPlan, AllocationState, FidelityQualification

GOLDEN = Path(__file__).parent / "fixtures/v1/qualified-fidelities.json"
UNBOUND = Path(__file__).parent / "fixtures/v1/prototype-unbound-qualification.json"


@pytest.mark.parametrize(
    "change",
    (
        "tiny_early_targets",
        "changed_early_target",
        "changed_terminal_target",
        "changed_early_version",
        "changed_terminal_version",
        "changed_early_identity",
        "changed_resource_units",
    ),
)
def test_same_count_cannot_reuse_qualification_for_different_schedule(ci_pool, change):
    plan = ci_pool.make_plan()
    raw = plan.model_dump(mode="json")
    original = plan.qualification
    assert original is not None
    if change == "tiny_early_targets":
        raw["fidelities"][0]["resource"]["value"] = "0.000000000001"
        raw["fidelities"][1]["resource"]["value"] = "0.000000000002"
    elif change == "changed_early_target":
        raw["fidelities"][0]["resource"]["value"] = "0.5"
    elif change == "changed_terminal_target":
        raw["fidelities"][2]["resource"]["value"] = "5"
    elif change == "changed_early_version":
        raw["fidelities"][0]["reference"]["version"] = "2"
    elif change == "changed_terminal_version":
        raw["fidelities"][2]["reference"]["version"] = "2"
    elif change == "changed_early_identity":
        raw["fidelities"][0]["reference"]["id"] = "another-first-proxy"
    else:
        semantics = {"unit": "seconds", "meaning": "another-resource-v1"}
        raw["budget"]["semantics"] = semantics
        for fidelity in raw["fidelities"]:
            fidelity["resource"]["semantics"] = semantics
    # All changed schedules have the original number of positive increasing
    # targets. The retained evidence was measured at the *original* targets.
    with pytest.raises(ValueError, match="qualified fidelity schedule"):
        AllocationPlan.model_validate(raw)
    assert plan.qualification is original


def test_equal_exact_resource_values_preserve_qualified_plan_and_digest(ci_pool):
    plan = ci_pool.make_plan()
    raw = plan.model_dump(mode="json")
    for row in raw["fidelities"]:
        row["resource"]["value"] = str(Decimal(row["resource"]["value"]).quantize(Decimal("0.000")))
    same = AllocationPlan.model_validate(raw)
    assert same.to_json() == plan.to_json()
    assert same.fingerprint == plan.fingerprint


def test_independently_authored_qualification_golden_retains_full_ordered_schedule():
    raw = json.loads(GOLDEN.read_text())
    qualified = FidelityQualification.from_json(GOLDEN.read_text())
    assert qualified.model_dump(mode="json") == raw
    expected = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert qualified.to_json() == expected
    assert qualified.fingerprint == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()
    assert [row.resource.value for row in qualified.fidelities] == [Decimal(1), Decimal(2), Decimal(4)]
    assert FidelityQualification.from_json(expected).to_json() == expected


@pytest.mark.parametrize(
    "change", ("missing_schedule", "short_case", "duplicate_target", "mixed_units", "mixed_objectives")
)
def test_standalone_qualification_requires_its_actual_schedule_and_comparable_pairs(change):
    raw = json.loads(GOLDEN.read_text())
    if change == "missing_schedule":
        del raw["fidelities"]
    elif change == "short_case":
        raw["cases"][0]["early"].pop()
    elif change == "duplicate_target":
        raw["fidelities"][1]["reference"] = raw["fidelities"][0]["reference"]
    elif change == "mixed_units":
        raw["fidelities"][1]["resource"]["semantics"]["unit"] = "seconds"
    else:
        another = json.loads(json.dumps(raw["cases"][0]))
        another["case_id"] = "another-objective"
        for value in (*another["early"], another["terminal"]):
            value["spec"]["dimension"]["id"] = "different-objective"
        raw["cases"].append(another)
    with pytest.raises(ValueError):
        FidelityQualification.model_validate(raw)


@pytest.mark.parametrize("version", (0, 2, True, "1"))
def test_qualification_unknown_versions_reject_without_inference(version):
    raw = json.loads(GOLDEN.read_text())
    raw["schema_version"] = version
    with pytest.raises(ValueError):
        FidelityQualification.model_validate(raw)


def test_actual_unbound_prototype_artifact_cannot_borrow_a_later_plans_schedule(ci_pool):
    text = UNBOUND.read_text()
    assert (
        hashlib.sha256(text.encode()).hexdigest() == "4aa101d3dc31a2952dcb5aba7be094f708c2c0b1d176427c9a189ec880806ea0"
    )
    with pytest.raises(ValueError, match="fidelities"):
        FidelityQualification.from_json(text)
    raw = ci_pool.make_plan().model_dump(mode="json")
    raw["qualification"] = json.loads(text)
    with pytest.raises(ValueError, match="fidelities"):
        AllocationPlan.model_validate(raw)


def test_checkpoint_cannot_replace_plan_targets_while_retaining_old_qualification(ci_pool):
    from coleman.acquisition.engine import inspect_allocation
    from coleman.acquisition.session import TerminalAllocator

    owner = TerminalAllocator(ci_pool.make_plan())
    state = owner.export_state()
    before = state.to_json()
    changed = state.plan.fidelities[0].model_copy(update={"resource": ci_pool.quantity("0.000000000001")})
    plan = state.plan.model_copy(update={"fidelities": (changed, *state.plan.fidelities[1:])})
    forged = state.model_copy(update={"plan": plan})
    raw = forged.model_dump(mode="json")
    with pytest.raises(ValueError, match="qualified fidelity schedule"):
        AllocationState.model_validate(raw)
    with pytest.raises(ValueError, match="qualified fidelity schedule"):
        inspect_allocation(forged)
    with pytest.raises(ValueError, match="qualified fidelity schedule"):
        owner.restore_state(forged)
    assert owner.export_state().to_json() == before
