"""Pure bounded allocation replay, importing observations but no checkpoint owners."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from coleman.acquisition.model import AcquisitionReceipt, AllocationPlan, Count, Purchase, _spec_content
from coleman.core.actions import AcquisitionAction
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.core.quantities import Quantity, decimal_text, decimal_value, ticks

Status = Literal["running", "completed", "budget_exhausted", "cost_unknown", "cost_overrun", "outcome_missing"]


class Elimination(CoreImmutable):
    """Retain one deterministic stage boundary and its actual discarded members."""

    fidelity: Reference
    eliminated: tuple[Reference, ...]
    retained: tuple[Reference, ...]


class AllocationReport(CoreContract):
    """Describe accepted knowledge, terminal status and exact observed resource use.

    Attributes
    ----------
    status : str
        Completed terminal identification, pending work, budget exhaustion or an
        explicit conservative missing/cost stop. No adaptive victory is implied.
    known_charge : Quantity
        Sum of observed actual resource charges, including charged failures.
    unknown_charges : int
        Unknown accepted amounts; zero known charge is never their substitute.
    active : tuple of Reference
        Remaining candidates, not a ranked full permutation.
    eliminations : tuple of Elimination
        Derived stage trace, including stable ID tie resolution.
    winner : Reference or None
        Best observed terminal candidate among survivors only on completion.
        Eliminated terminal outcomes remain unknown to the allocator.
    next_purchase : Purchase or None
        Deterministic pending request, null after a stopping outcome.

    Notes
    -----
    Reading/replaying performs no provider call, filesystem operation or fee
    conversion. Known resource is not necessarily money or executed-book cost.
    """

    status: Status
    known_charge: Quantity
    unknown_charges: Count
    active: tuple[Reference, ...]
    eliminations: tuple[Elimination, ...]
    winner: Reference | None
    next_purchase: Purchase | None


@dataclass
class _Cursor:  # pylint: disable=too-many-instance-attributes
    # Schedule, accepted accounting and elimination fields have distinct meanings.
    """Hold local replay arithmetic only; complete state is the immutable ledger."""

    active: tuple[Reference, ...]
    stage: int
    status: Status = "running"
    spent: int = 0
    unknown: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    previous: dict[str, int] = field(default_factory=dict)
    used_groups: set[tuple[str, str]] = field(default_factory=set)
    eliminations: list[Elimination] = field(default_factory=list)
    winner: Reference | None = None


def _quantity(plan: AllocationPlan, amount: int) -> Quantity:
    """Construct an exact budget-unit amount without ambient Decimal rounding."""
    return Quantity(value=Decimal(decimal_text(decimal_value(amount))), semantics=plan.budget.semantics)


def _request(plan: AllocationPlan, cursor: _Cursor, ordinal: int) -> Purchase | None:
    """Derive the sole next purchase and refuse a quote above remaining budget."""
    if cursor.status != "running":
        return None
    member = next(row for row in cursor.active if row.id not in cursor.scores)
    fidelity = plan.fidelities[cursor.stage]
    previous = cursor.previous.get(member.id)
    target = ticks(fidelity.resource.value)
    amount = target
    if previous is not None and plan.billing == "incremental":
        amount -= ticks(plan.fidelities[previous].resource.value)
    candidate = next(row for row in plan.candidates if row.reference == member)
    group = candidate.setup_group
    if group is not None and (group.id, group.version) not in cursor.used_groups:
        amount += ticks(next(row.resource.value for row in plan.setups if row.group == group))
    if cursor.spent + amount > ticks(plan.budget.value):
        cursor.status = "budget_exhausted"
        return None
    action = AcquisitionAction(schema_version=1, space=plan.space, chosen=member, fidelity=fidelity.reference)
    prior = None if previous is None else plan.fidelities[previous].reference
    identity = fingerprint({"plan": plan.fingerprint, "ordinal": ordinal, "action": action.fingerprint})
    return Purchase(
        schema_version=1,
        purchase_id=identity,
        ordinal=ordinal,
        action=action,
        prior_fidelity=prior,
        quoted_charge=_quantity(plan, amount),
    )


def _stage_complete(plan: AllocationPlan, cursor: _Cursor) -> None:
    """Select by actual purchased scores, with stable IDs for exact ties."""
    if len(cursor.scores) != len(cursor.active):
        return
    sign = 1 if plan.direction == "minimize" else -1
    ordered = tuple(sorted(cursor.active, key=lambda row: (sign * cursor.scores[row.id], row.id)))
    if cursor.stage == len(plan.fidelities) - 1:
        cursor.winner = ordered[0]
        cursor.status = "completed"
        return
    keep = (len(ordered) + 1) // 2
    retained = tuple(sorted(ordered[:keep], key=lambda row: row.id))
    cursor.eliminations.append(
        Elimination(
            fidelity=plan.fidelities[cursor.stage].reference,
            eliminated=tuple(sorted(ordered[keep:], key=lambda row: row.id)),
            retained=retained,
        )
    )
    cursor.active = retained
    cursor.stage += 1
    cursor.scores.clear()


def _apply(plan: AllocationPlan, cursor: _Cursor, receipt: AcquisitionReceipt) -> None:
    """Account the actual receipt before applying any conservative stopping rule."""
    if _spec_content(receipt.outcome.spec) != _spec_content(plan.outcome):
        raise ValueError("Purchased outcome has incompatible objective units/semantics")
    charge = receipt.charge
    if charge is not None and charge.semantics != plan.budget.semantics:
        raise ValueError("Actual charge has incompatible budget units/semantics")
    if charge is None:
        cursor.unknown += 1
        cursor.status = "cost_unknown"
    else:
        cursor.spent += ticks(charge.value)
        if charge.value > receipt.purchase.quoted_charge.value or cursor.spent > ticks(plan.budget.value):
            cursor.status = "cost_overrun"
    if cursor.status != "running":
        return
    if receipt.outcome.status != "observed":
        cursor.status = "outcome_missing"
        return
    member = receipt.purchase.action.chosen
    cursor.previous[member.id] = cursor.stage
    group = next(row.setup_group for row in plan.candidates if row.reference == member)
    if group is not None:
        cursor.used_groups.add((group.id, group.version))
    value = receipt.outcome.value
    if value is None:  # Defensive narrowing; Scalar already enforces this invariant.
        raise ValueError("Observed outcome is missing its value")
    cursor.scores[member.id] = value
    _stage_complete(plan, cursor)


def replay(plan: AllocationPlan, receipts: tuple[AcquisitionReceipt, ...]) -> AllocationReport:
    """Validate and reconstruct a bounded accepted ledger without reading a pool.

    Parameters
    ----------
    plan : AllocationPlan
        Frozen complete plan; callers normally use ``inspect_allocation``.
    receipts : tuple of AcquisitionReceipt
        At most 640 purchased observations in exact deterministic request order.

    Returns
    -------
    AllocationReport
        Exact continuation and conservative resource/knowledge status.

    Raises
    ------
    ValueError
        Too many receipts, a substituted/out-of-order request, post-stop data or
        incompatible observation/charge semantics. No partial state is returned.

    Notes
    -----
    This primitive performs no acknowledgement or physical execution. Replaying
    validates every original transition, with no unpurchased outcome access.
    """
    plan = AllocationPlan.model_validate(plan)
    if not isinstance(receipts, tuple):
        raise ValueError("Accepted acquisition receipts require an immutable tuple")
    receipts = tuple(AcquisitionReceipt.model_validate(receipt) for receipt in receipts)
    if len(receipts) > 640:
        raise ValueError("Terminal allocation ledger exceeds 640 acquisitions")
    first = len(plan.fidelities) - 1 if plan.method == "equal_allocation_v1" else 0
    cursor = _Cursor(active=tuple(row.reference for row in plan.candidates), stage=first)
    for ordinal, receipt in enumerate(receipts):
        expected = _request(plan, cursor, ordinal)
        if expected is None or receipt.purchase.to_json() != expected.to_json():
            raise ValueError("Receipt is not the exact pending purchase, or allocation already stopped")
        _apply(plan, cursor, receipt)
    pending = _request(plan, cursor, len(receipts))
    return AllocationReport(
        schema_version=1,
        status=cursor.status,
        known_charge=_quantity(plan, cursor.spent),
        unknown_charges=cursor.unknown,
        active=cursor.active,
        eliminations=tuple(cursor.eliminations),
        winner=cursor.winner,
        next_purchase=pending,
    )
