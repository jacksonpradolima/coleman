"""Independent terminal-allocation arithmetic and purchased-information checks."""

from decimal import Decimal

import pytest

from coleman.acquisition.contracts import AcquisitionReceipt, AllocationPlan, AllocationState, SetupCharge
from coleman.acquisition.engine import accept_receipt, inspect_allocation
from coleman.acquisition.session import TerminalAllocator
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar
from coleman.core.primitives import fingerprint


@pytest.mark.parametrize(
    ("method", "billing", "charged", "reads"),
    [
        ("equal_allocation_v1", "incremental", 16, 16),
        ("successive_halving_v1", "incremental", 8, 8),
        ("successive_halving_v1", "stateless_rerun", 12, 12),
    ],
)
def test_real_ci_pool_has_independently_counted_physical_charges(ci_pool, method, billing, charged, reads):
    owner, pool = ci_pool.run(method, billing)
    report = owner.inspect()
    assert report.status == "completed"
    assert report.winner.id == "A"
    assert report.known_charge.value == Decimal(charged)
    assert len(pool.reads) == reads
    assert owner.export_state().plan.budget.value - report.known_charge.value == 16 - charged
    if method == "successive_halving_v1":
        assert [(row.fidelity.id, [ref.id for ref in row.eliminated]) for row in report.eliminations] == [
            ("builds-1", ["C", "D"]),
            ("builds-2", ["B"]),
        ]
        assert not any(member in {"C", "D"} and index > 0 for member, index in pool.reads)
        assert not any(member == "B" and index > 1 for member, index in pool.reads)


def test_reference_trace_and_every_increment_elimination_restart(ci_pool):
    plan = ci_pool.make_plan()
    state = AllocationState(schema_version=1, plan=plan)
    pool = ci_pool.CIPool()
    trace = []
    while (purchase := inspect_allocation(state).next_purchase) is not None:
        prior = next(
            (row for row in reversed(state.receipts) if row.purchase.action.chosen == purchase.action.chosen), None
        )
        trace.append((purchase.action.chosen.id, purchase.action.fidelity.id, purchase.quoted_charge.value))
        receipt = pool.acquire(purchase, plan, prior)
        original = accept_receipt(state, receipt)
        # Restart after every increment, including immediately before/after both
        # eliminations. No provider prefix cache is needed for continuation.
        state = AllocationState.from_json(original.to_json())
        assert state == original
        assert accept_receipt(state, receipt) == state
        assert state.fingerprint == original.fingerprint
    assert trace == [
        ("A", "builds-1", 1),
        ("B", "builds-1", 1),
        ("C", "builds-1", 1),
        ("D", "builds-1", 1),
        ("A", "builds-2", 1),
        ("B", "builds-2", 1),
        ("A", "builds-4", 2),
    ]
    assert inspect_allocation(state).known_charge.value == 8


def test_same_receipt_retry_does_not_add_charge_and_conflict_does_not_mutate(ci_pool):
    owner = TerminalAllocator(ci_pool.make_plan())
    purchase = owner.inspect().next_purchase
    receipt = ci_pool.CIPool().acquire(purchase, owner.export_state().plan)
    owner.accept(receipt)
    before = owner.export_state()
    owner.accept(receipt)
    assert owner.export_state() == before
    changed = AcquisitionReceipt.model_validate(receipt.model_copy(update={"delay_steps": 1}))
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        owner.accept(changed)
    assert owner.export_state() == before


@pytest.mark.parametrize(("amount", "status"), [(None, "cost_unknown"), (2, "cost_overrun"), (0, "running")])
def test_actual_unknown_excessive_and_free_costs_are_distinct(ci_pool, amount, status):
    state = AllocationState(schema_version=1, plan=ci_pool.make_plan())
    purchase = inspect_allocation(state).next_purchase
    assert purchase is not None
    receipt = AcquisitionReceipt(
        schema_version=1,
        purchase=purchase,
        producer=Reference(id="meter", version="1"),
        outcome=Scalar(spec=state.plan.outcome, value=0, status="observed"),
        charge=None if amount is None else ci_pool.quantity(amount),
        delay_steps=5,
    )
    state = accept_receipt(state, receipt)
    report = inspect_allocation(state)
    assert report.status == status
    assert report.known_charge.value == (0 if amount is None else amount)
    assert report.unknown_charges == (1 if amount is None else 0)
    assert state.receipts[0].delay_steps == 5
    if status != "running":
        assert report.next_purchase is None
        assert report.winner is None


@pytest.mark.parametrize("missing", ["unknown", "unavailable", "not_applicable"])
def test_paid_missing_outcome_stops_without_imputation(ci_pool, missing):
    state = AllocationState(schema_version=1, plan=ci_pool.make_plan())
    purchase = inspect_allocation(state).next_purchase
    assert purchase is not None
    receipt = AcquisitionReceipt(
        schema_version=1,
        purchase=purchase,
        producer=Reference(id="meter", version="1"),
        outcome=Scalar(spec=state.plan.outcome, value=None, status=missing),
        charge=ci_pool.quantity(1),
        delay_steps=0,
    )
    report = inspect_allocation(accept_receipt(state, receipt))
    assert report.status == "outcome_missing"
    assert report.winner is None
    assert report.known_charge.value == 1
    assert report.next_purchase is None


def test_bounded_budget_is_never_silently_rounded_up_or_source_truncated(ci_pool):
    state = AllocationState(schema_version=1, plan=ci_pool.make_plan(budget=0))
    report = inspect_allocation(state)
    assert report.status == "budget_exhausted"
    assert report.next_purchase is None
    assert len(state.plan.candidates) == 4
    assert report.known_charge.value == 0
    assert report.winner is None


def test_shared_setup_overlap_is_charged_once_and_full_reruns_remain_explicit(ci_pool):
    plan = ci_pool.make_plan("equal_allocation_v1", budget=20)
    group = Reference(id="shared-build-image", version="1")
    candidates = tuple(type(row)(reference=row.reference, setup_group=group) for row in plan.candidates)
    plan = AllocationPlan.model_validate(
        plan.model_copy(
            update={
                "candidates": candidates,
                "setups": (SetupCharge(group=group, resource=ci_pool.quantity(3)),),
            }
        )
    )
    state = AllocationState(schema_version=1, plan=plan)
    quotes = []
    while (purchase := inspect_allocation(state).next_purchase) is not None:
        quotes.append(purchase.quoted_charge.value)
        receipt = AcquisitionReceipt(
            schema_version=1,
            purchase=purchase,
            producer=Reference(id="meter", version="1"),
            outcome=Scalar(spec=plan.outcome, value=0, status="observed"),
            charge=purchase.quoted_charge,
            delay_steps=0,
        )
        state = accept_receipt(state, receipt)
    assert quotes == [7, 4, 4, 4]  # 4*4 replay units plus one shared setup of3.
    assert inspect_allocation(state).known_charge.value == 19
    report = inspect_allocation(state)
    assert report.winner is not None
    assert report.winner.id == "A"  # Stable no-benefit tie.


def test_fidelity_qualification_rejects_unfair_training_and_protected_contamination(ci_pool):
    plan = ci_pool.make_plan()
    raw = plan.model_dump(mode="json")
    raw["qualification"]["cases"][0]["early"][0]["value"] = 1
    with pytest.raises(ValueError, match="Training fidelity gate failed"):
        AllocationPlan.model_validate(raw)
    raw = plan.model_dump(mode="json")
    raw["qualification"]["training_source"] = raw["source_fingerprint"]
    with pytest.raises(ValueError, match="distinct training"):
        AllocationPlan.model_validate(raw)
    raw = plan.model_dump(mode="json")
    raw["qualification"] = None
    with pytest.raises(ValueError, match="requires training-only"):
        AllocationPlan.model_validate(raw)


def test_candidate_permutation_is_metamorphically_identical(ci_pool):
    plan = ci_pool.make_plan()
    permuted = AllocationPlan.model_validate(plan.model_copy(update={"candidates": tuple(reversed(plan.candidates))}))
    assert plan.to_json() == permuted.to_json()
    assert plan.fingerprint == permuted.fingerprint
    assert plan.space == permuted.space


def test_signed_negative_objectives_and_no_benefit_are_valid(ci_pool):
    plan = ci_pool.make_plan("equal_allocation_v1")
    spec = plan.outcome.model_copy(update={"minimum": None})
    plan = AllocationPlan.model_validate(plan.model_copy(update={"outcome": spec}))
    state = AllocationState(schema_version=1, plan=plan)
    while (purchase := inspect_allocation(state).next_purchase) is not None:
        state = accept_receipt(
            state,
            AcquisitionReceipt(
                schema_version=1,
                purchase=purchase,
                producer=Reference(id="signed-source", version="1"),
                outcome=Scalar(spec=spec, value=-2, status="observed"),
                charge=ci_pool.quantity(0),
                delay_steps=0,
            ),
        )
    assert inspect_allocation(state).status == "completed"
    report = inspect_allocation(state)
    assert report.winner is not None
    assert report.winner.id == "A"
    assert inspect_allocation(state).known_charge.value == 0


def test_early_rank_reversal_is_an_honest_negative_with_unpurchased_terminal_hidden(ci_pool):
    # Training evidence is fixed by make_plan before any values below are used.
    # It cannot guarantee that a new held-out path follows its proxy ranking.
    plan = ci_pool.make_plan()
    read = []
    # The source host owns this table. The allocator receives only purchased rows.
    source = {
        ("A", "builds-1"): 0.0,
        ("B", "builds-1"): 0.1,
        ("C", "builds-1"): 0.8,
        ("D", "builds-1"): 1.0,
        ("A", "builds-2"): 0.0,
        ("B", "builds-2"): 0.1,
        ("A", "builds-4"): 0.8,
        ("D", "builds-4"): 0.0,
    }
    source_hash = fingerprint([[*key, value] for key, value in sorted(source.items())])
    qualification = plan.qualification
    assert qualification is not None
    qualification = type(qualification).model_validate(
        qualification.model_copy(update={"evaluation_source": source_hash})
    )
    plan = AllocationPlan.model_validate(
        plan.model_copy(
            update={
                "source_fingerprint": source_hash,
                "qualification": qualification,
            }
        )
    )
    state = AllocationState(schema_version=1, plan=plan)
    while (purchase := inspect_allocation(state).next_purchase) is not None:
        key = (purchase.action.chosen.id, purchase.action.fidelity.id)
        read.append(key)
        state = accept_receipt(
            state,
            AcquisitionReceipt(
                schema_version=1,
                purchase=purchase,
                producer=Reference(id="held-out-reversal-source", version="1"),
                outcome=Scalar(spec=plan.outcome, value=source[key], status="observed"),
                charge=purchase.quoted_charge,
                delay_steps=0,
            ),
        )
    report = inspect_allocation(state)
    assert report.winner is not None
    assert report.winner.id == "A"
    assert ("D", "builds-4") not in read
    before = state.fingerprint
    # A separate evaluator can discover the recommendation loses to D. That
    # terminal value was never input to elimination, tuning or the checkpoint.
    assert source[("D", "builds-4")] < source[(report.winner.id, "builds-4")]
    assert state.fingerprint == before
    assert inspect_allocation(state) == report


@pytest.mark.parametrize("change", ["objective", "resource", "members"])
def test_real_provider_rejects_relabeling_or_different_scope_before_reading(ci_pool, change):
    from coleman.core.units import UnitSemantics

    plan = ci_pool.make_plan("equal_allocation_v1")
    raw = plan.model_dump(mode="json")
    if change == "objective":
        raw["outcome"]["semantics"] = {"unit": "seconds", "meaning": "measured-ci-duration-v1"}
    elif change == "resource":
        wrong = UnitSemantics(unit="seconds", meaning="other-resource-v1").model_dump(mode="json")
        raw["budget"]["semantics"] = wrong
        for fidelity in raw["fidelities"]:
            fidelity["resource"]["semantics"] = wrong
    else:
        raw["candidates"] = raw["candidates"][:2]
    plan = AllocationPlan.model_validate(raw)
    owner = TerminalAllocator(plan)
    pool = ci_pool.CIPool()
    purchase = owner.inspect().next_purchase
    with pytest.raises(ValueError):
        pool.acquire(purchase, plan)
    assert pool.reads == []
    assert owner.export_state().receipts == ()


def test_real_incremental_provider_rejects_substituted_prior_producer_and_objective(ci_pool):
    plan = ci_pool.make_plan()
    owner = TerminalAllocator(plan)
    pool = ci_pool.CIPool()
    # Four actual purchases reach the A,target2 increment; prior is A,target1.
    for _ in range(4):
        owner.accept(pool.acquire(owner.inspect().next_purchase, plan))
    purchase = owner.inspect().next_purchase
    prior = owner.export_state().receipts[0]
    wrong_source = AcquisitionReceipt.model_validate(
        prior.model_copy(
            update={
                "producer": Reference(id="substituted-producer", version="1"),
            }
        )
    )
    before = list(pool.reads)
    with pytest.raises(ValueError, match="original plan, producer"):
        pool.acquire(purchase, plan, wrong_source)
    altered_spec = prior.outcome.spec.model_copy(update={"minimum": -0.0})
    wrong_spec = AcquisitionReceipt.model_validate(
        prior.model_copy(
            update={
                "outcome": Scalar(spec=altered_spec, value=0, status="observed"),
            }
        )
    )
    with pytest.raises(ValueError, match="original plan, producer"):
        pool.acquire(purchase, plan, wrong_spec)
    assert pool.reads == before
