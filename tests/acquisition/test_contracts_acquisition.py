"""Versioned independent golden input, guarded ingress and identity boundaries."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from coleman.acquisition.contracts import AcquisitionReceipt, AllocationPlan, AllocationState
from coleman.acquisition.engine import accept_receipt, inspect_allocation
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar
from coleman.core.quantities import Quantity
from coleman.core.units import UnitSemantics

GOLDEN = Path(__file__).parent / "fixtures/v1/terminal-plan.json"


def test_independent_versioned_plan_golden_round_trip_and_digest():
    authored = json.loads(GOLDEN.read_text())
    plan = AllocationPlan.from_json(GOLDEN.read_text())
    assert plan.model_dump(mode="json") == authored
    assert AllocationPlan.from_json(plan.to_json()) == plan
    # Independent standard-library content digest; no production fingerprint helper.
    encoded = json.dumps(authored, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert plan.to_json() == encoded
    assert plan.fingerprint == "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
    state = AllocationState(schema_version=1, plan=plan)
    assert AllocationState.from_json(state.to_json()) == state
    purchase = inspect_allocation(state).next_purchase
    assert purchase is not None
    action = purchase.action
    assert action.kind == "acquisition"
    assert action.chosen.id == "recipe-A"


@pytest.mark.parametrize("version", [0, 2, 99])
def test_unknown_versions_and_extra_fields_rejected(version):
    raw = json.loads(GOLDEN.read_text())
    raw["schema_version"] = version
    with pytest.raises(ValueError):
        AllocationPlan.model_validate(raw)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_objective_or_qualification_is_rejected(ci_pool, value):
    plan = ci_pool.make_plan()
    with pytest.raises(ValueError):
        Scalar(spec=plan.outcome, value=value, status="observed")
    data = plan.model_dump(mode="json")
    data["qualification"]["maximum_error"] = value
    with pytest.raises(ValueError):
        AllocationPlan.model_validate(data)


def test_incompatible_units_rejected_before_acceptance(ci_pool):
    state = AllocationState(schema_version=1, plan=ci_pool.make_plan())
    purchase = inspect_allocation(state).next_purchase
    assert purchase is not None
    wrong = Quantity(value=Decimal("1"), semantics=UnitSemantics(unit="seconds", meaning="wall-clock-v1"))
    receipt = AcquisitionReceipt(
        schema_version=1,
        purchase=purchase,
        producer=Reference(id="meter", version="1"),
        outcome=Scalar(spec=state.plan.outcome, value=0, status="observed"),
        charge=wrong,
        delay_steps=0,
    )
    with pytest.raises(ValueError, match="incompatible budget units"):
        accept_receipt(state, receipt)
    assert not state.receipts


def test_changed_objective_plan_and_substituted_request_are_rejected(ci_pool):
    state = AllocationState(schema_version=1, plan=ci_pool.make_plan())
    purchase = inspect_allocation(state).next_purchase
    assert purchase is not None
    receipt = ci_pool.CIPool().acquire(purchase, state.plan)
    wrong = receipt.purchase.action.model_copy(update={"chosen": state.plan.candidates[1].reference})
    forged = receipt.model_copy(update={"purchase": receipt.purchase.model_copy(update={"action": wrong})})
    with pytest.raises(ValueError, match="exact pending purchase"):
        accept_receipt(state, forged)
    raw = state.model_dump(mode="json")
    raw["schema_version"] = 2
    with pytest.raises(ValueError):
        AllocationState.model_validate(raw)
    raw = json.loads(GOLDEN.read_text())
    raw["extra"] = "forbidden"
    with pytest.raises(ValueError):
        AllocationPlan.model_validate(raw)


def test_future_nested_version_duplicate_keys_and_nonfinite_json_do_not_fallback():
    text = GOLDEN.read_text()
    future_version = text.replace('"schema_version": 1', '"schema_version": 2')
    with pytest.raises(ValueError):
        AllocationPlan.from_json(future_version)
    duplicate_key = text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    with pytest.raises(ValueError):
        AllocationPlan.from_json(duplicate_key)
    nonfinite = text.replace('"minimum": null', '"minimum": NaN')
    with pytest.raises(ValueError):
        AllocationPlan.from_json(nonfinite)


def test_resource_quantum_is_canonical_without_altering_exact_value_or_old_quantity_class():
    raw = json.loads(GOLDEN.read_text())
    raw["budget"]["value"] = "2.000000000000"
    raw["fidelities"][0]["resource"]["value"] = "1.0"
    plan = AllocationPlan.model_validate(raw)
    assert str(plan.budget.value) == "2"
    assert str(plan.fidelities[0].resource.value) == "1"
    assert plan == AllocationPlan.from_json(plan.to_json())
    legacy = Quantity(value=Decimal("2.00"), semantics=plan.budget.semantics)
    assert legacy.value.as_tuple().exponent == -2


def test_in_memory_future_or_substituted_checkpoint_cannot_bypass_public_boundaries(ci_pool):
    state = AllocationState(schema_version=1, plan=ci_pool.make_plan())
    forged = state.model_copy(update={"schema_version": 2})
    with pytest.raises(ValueError):
        inspect_allocation(forged)
    receipt = ci_pool.CIPool().acquire(inspect_allocation(state).next_purchase, state.plan)
    with pytest.raises(ValueError):
        accept_receipt(forged, receipt)


@pytest.mark.parametrize(
    "case",
    [
        "negative_error",
        "same_partition_version",
        "duplicate_training",
        "missing_terminal",
        "missing_early",
        "different_objective",
        "wrong_budget_units",
        "reversed_fidelities",
        "nonobjective",
        "excessive_budget",
        "unused_setup",
        "missing_setup",
    ],
)
def test_invalid_plan_boundaries_fail_before_any_experiment_or_charge(ci_pool, case):
    raw = ci_pool.make_plan().model_dump(mode="json")
    if case == "negative_error":
        raw["qualification"]["maximum_error"] = -1
    elif case == "same_partition_version":
        raw["qualification"]["training_partition"] = {
            "id": raw["qualification"]["evaluation_partition"]["id"],
            "version": "2",
        }
    elif case == "duplicate_training":
        raw["qualification"]["cases"] *= 2
    elif case == "missing_terminal":
        raw["qualification"]["cases"][0]["terminal"].update(value=None, status="unavailable")
    elif case == "missing_early":
        raw["qualification"]["cases"][0]["early"][0].update(value=None, status="unknown")
    elif case == "different_objective":
        raw["qualification"]["cases"][0]["early"][0]["spec"]["semantics"]["meaning"] = "other-score-v1"
    elif case == "wrong_budget_units":
        raw["fidelities"][0]["resource"]["semantics"]["unit"] = "seconds"
    elif case == "reversed_fidelities":
        raw["fidelities"] = list(reversed(raw["fidelities"]))
    elif case == "nonobjective":
        raw["outcome"]["role"] = "feature"
    elif case == "excessive_budget":
        raw["budget"]["value"] = "1000000000001"
    elif case == "unused_setup":
        raw["setups"] = [{"group": {"id": "unused", "version": "1"}, "resource": raw["budget"]}]
    else:
        raw["candidates"][0]["setup_group"] = {"id": "missing", "version": "1"}
    with pytest.raises(ValueError):
        AllocationPlan.model_validate(raw)


def test_direct_replay_bounds_and_restore_require_complete_original_plan(ci_pool):
    from coleman.acquisition.engine import replay
    from coleman.acquisition.session import TerminalAllocator

    state = AllocationState(schema_version=1, plan=ci_pool.make_plan())
    purchase = inspect_allocation(state).next_purchase
    receipt = ci_pool.CIPool().acquire(purchase, state.plan)
    with pytest.raises(ValueError, match="immutable tuple"):
        replay(state.plan, [])  # type: ignore[arg-type]  # Deliberately invalid runtime boundary.
    with pytest.raises(ValueError, match="exceeds 640"):
        replay(state.plan, (receipt,) * 641)
    changed = TerminalAllocator(ci_pool.make_plan(budget=15))
    with pytest.raises(ValueError, match="exact original plan"):
        changed.restore_state(state)


def test_exact_delivery_rejects_signed_zero_content_change(ci_pool):
    from coleman.acquisition.session import TerminalAllocator

    owner = TerminalAllocator(ci_pool.make_plan())
    receipt = ci_pool.CIPool().acquire(owner.inspect().next_purchase, owner.export_state().plan)
    assert receipt.outcome.value == 0.0
    owner.accept(receipt)
    before = owner.export_state().to_json()
    altered = AcquisitionReceipt.model_validate(
        receipt.model_copy(
            update={
                "outcome": receipt.outcome.model_copy(update={"value": -0.0}),
            }
        )
    )
    assert receipt == altered  # Pydantic numeric equality is not content identity.
    assert receipt.fingerprint != altered.fingerprint
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        owner.accept(altered)
    assert owner.export_state().to_json() == before
    owner.accept(AcquisitionReceipt.from_json(receipt.to_json()))
    assert owner.export_state().to_json() == before


def test_exact_restore_and_outcome_spec_reject_signed_zero_configuration_change(ci_pool):
    from coleman.acquisition.session import TerminalAllocator

    plan = ci_pool.make_plan("equal_allocation_v1")
    altered_spec = type(plan.outcome).model_validate(plan.outcome.model_copy(update={"minimum": -0.0}))
    altered_plan = AllocationPlan.model_validate(plan.model_copy(update={"outcome": altered_spec}))
    assert plan == altered_plan
    assert plan.fingerprint != altered_plan.fingerprint
    owner = TerminalAllocator(plan)
    before = owner.export_state().to_json()
    altered_state = AllocationState(schema_version=1, plan=altered_plan)
    with pytest.raises(ValueError, match="exact original plan"):
        owner.restore_state(altered_state)
    assert owner.export_state().to_json() == before
    receipt = ci_pool.CIPool().acquire(owner.inspect().next_purchase, plan)
    altered = AcquisitionReceipt.model_validate(
        receipt.model_copy(
            update={
                "outcome": Scalar(spec=altered_spec, value=0.0, status="observed"),
            }
        )
    )
    with pytest.raises(ValueError, match="incompatible objective"):
        owner.accept(altered)
    assert owner.export_state().to_json() == before
