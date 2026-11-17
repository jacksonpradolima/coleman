"""Constructed CI terminal-identification pool with explicit resource accounting.

Run from the repository: ``python examples/acquisition/ci_pool.py``. Installed
qualification copies this standalone example outside the checkout first.
"""

from decimal import Decimal
from typing import Literal

from coleman.acquisition.contracts import (
    AcquisitionReceipt,
    AllocationPlan,
    CalibrationCase,
    Candidate,
    Fidelity,
    FidelityQualification,
    Purchase,
)
from coleman.acquisition.session import TerminalAllocator
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.quantities import Quantity
from coleman.core.units import UnitSemantics

RESOURCE = UnitSemantics(unit="build_replays", meaning="constructed-ci-prefix-build-replays-v1")
OBJECTIVE = ScalarSpec(
    dimension=Reference(id="missed-fault-fraction", version="1"),
    semantics=UnitSemantics(unit="1", meaning="missed-faults-at-two-tests-per-build-v1"),
    role="objective",
    minimum=0,
    maximum=1,
)
# This full constructed pool is declared before either method runs. The planner
# gets its fingerprint, candidate references and purchased scores, never this table.
ORDERS = {"A": ("t1", "t2"), "B": ("t1", "t3"), "C": ("t2", "t3"), "D": ("t3", "t4")}
FAULTS = ("t1", "t2", "t1", "t1")
SOURCE = fingerprint({"orders": {key: list(value) for key, value in ORDERS.items()}, "faults": list(FAULTS)})


def quantity(value: int | str | Decimal) -> Quantity:
    """Construct an exact resource amount in declared build-replay units.

    Parameters
    ----------
    value : int or str or Decimal
        Nonnegative finite amount with at most 12 fractional places.

    Returns
    -------
    Quantity
        Exact model-resource value, without physical execution or I/O.

    Raises
    ------
    ValueError
        Invalid/nonfinite, negative or out-of-profile amount.
    """
    return Quantity(value=Decimal(str(value)), semantics=RESOURCE)


def make_plan(
    method: Literal["equal_allocation_v1", "successive_halving_v1"] = "successive_halving_v1",
    billing: Literal["incremental", "stateless_rerun"] = "incremental",
    budget: int = 16,
) -> AllocationPlan:
    """Freeze one method before any protected pool observation is purchased.

    Parameters
    ----------
    method : str, optional
        Terminal-only uniform or explicit finite halving schedule.
    billing : str, optional
        Incremental build replays or fully charged stateless prefix reruns.
    budget : int, optional
        Maximum admitted build-replay resource, default 16 for both methods.

    Returns
    -------
    AllocationPlan
        Complete constructed pool/configuration with no purchased outcomes.

    Raises
    ------
    ValueError
        Invalid configuration, budget or fidelity qualification.

    Notes
    -----
    The tiny empirical training gate demonstrates the contract, not sufficient
    evidence for adoption or a held-out guarantee. Construction performs no I/O.
    """
    fidelities = tuple(
        Fidelity(reference=Reference(id=f"builds-{value}", version="1"), resource=quantity(value))
        for value in (1, 2, 4)
    )
    qualification = None
    if method == "successive_halving_v1":
        # Independent development cases are supplied explicitly. The threshold
        # admits an empirical training gate; it is not a test-set error bound.
        qualification = FidelityQualification(
            schema_version=1,
            training_partition=Reference(id="development-builds", version="1"),
            evaluation_partition=Reference(id="evaluation-builds", version="1"),
            training_source=fingerprint({"development": [0.0, 0.0, 0.0]}),
            evaluation_source=SOURCE,
            maximum_error=0,
            fidelities=fidelities,
            cases=(
                CalibrationCase(
                    case_id="development-stable-order",
                    producer=Reference(id="constructed-development-ci", version="1"),
                    early=(Scalar(spec=OBJECTIVE, value=0, status="observed"),) * 2,
                    terminal=Scalar(spec=OBJECTIVE, value=0, status="observed"),
                ),
            ),
        )
    return AllocationPlan(
        schema_version=1,
        reference=Reference(id="constructed-ci-terminal-comparison", version="1"),
        source_fingerprint=SOURCE,
        candidates=tuple(Candidate(reference=Reference(id=key, version="1")) for key in ORDERS),
        fidelities=fidelities,
        method=method,
        billing=billing,
        budget=quantity(budget),
        outcome=OBJECTIVE,
        direction="minimize",
        qualification=qualification,
    )


class CIPool:  # pylint: disable=too-few-public-methods
    # One explicit execution operation plus a read audit is the complete protocol.
    """Execute only requested CI prefixes, using an accepted prior receipt.

    This host-owned executor is outside the mathematical planner. Its measured
    prefix aggregate comes from the allocator ledger; the example uses
    idempotent source reads, not physical remote CI jobs. Stateless reruns use
    the full prefix and charge every build again.
    """

    def __init__(self):
        """Initialize an independent read audit; no execution cache affects continuation."""
        self.reads: list[tuple[str, int]] = []

    @staticmethod
    def _admitted(purchase: Purchase, plan: AllocationPlan) -> None:
        """Reject foreign source/objective/units/membership before any source read."""
        if purchase.action.space != plan.space or plan.source_fingerprint != SOURCE:
            raise ValueError("CI executor requires its exact admitted source and action space")
        if canonical_json(plan.outcome.model_dump(mode="json")) != canonical_json(OBJECTIVE.model_dump(mode="json")):
            raise ValueError("CI executor computes its fixed missed-fault objective; it cannot relabel measurements")
        if plan.budget.semantics != RESOURCE:
            raise ValueError("CI executor meters build replays, not another resource unit")
        expected_members = tuple(Reference(id=key, version="1") for key in sorted(ORDERS))
        if plan.space.members != expected_members:
            raise ValueError("CI executor requires its complete admitted versioned candidate pool")

    @staticmethod
    def _previous(purchase: Purchase, plan: AllocationPlan, prior: AcquisitionReceipt | None) -> tuple[int, float]:
        """Recover the original accepted prefix; stateless reruns start at zero."""
        if plan.billing != "incremental" or purchase.prior_fidelity is None:
            return 0, 0.0
        if prior is None or prior.purchase.action.chosen != purchase.action.chosen:
            raise ValueError("Incremental execution needs the original accepted prior receipt")
        if (
            prior.purchase.action.space != purchase.action.space
            or prior.producer != Reference(id="constructed-ci-prefix-executor", version="1")
            or canonical_json(prior.outcome.spec.model_dump(mode="json"))
            != canonical_json(OBJECTIVE.model_dump(mode="json"))
        ):
            raise ValueError("Incremental execution requires its original plan, producer and objective aggregate")
        if prior.purchase.action.fidelity != purchase.prior_fidelity or prior.outcome.value is None:
            raise ValueError("Incremental execution has an incompatible prior aggregate")
        start = int(next(row.resource.value for row in plan.fidelities if row.reference == purchase.prior_fidelity))
        return start, prior.outcome.value * start

    def acquire(
        self,
        purchase: Purchase,
        plan: AllocationPlan,
        prior: AcquisitionReceipt | None = None,
    ) -> AcquisitionReceipt:
        """Execute the actual chosen candidate/fidelity, returning its own score.

        Parameters
        ----------
        purchase : Purchase
            Exact proposed candidate and integer cumulative build target.
        plan : AllocationPlan
            Frozen constructed source and resource-billing profile.
        prior : AcquisitionReceipt or None, optional
            Original accepted previous aggregate for incremental continuation.

        Returns
        -------
        AcquisitionReceipt
            Purchased score and actual newly read build count. Mutates only the
            independent read audit; no cache is required by continuation.

        Raises
        ------
        ValueError
            Different source, unsupported target or incompatible prior aggregate.

        Notes
        -----
        This constructed executor performs local deterministic source reads,
        not physical CI jobs. Its producer reference is not authentication.
        """
        self._admitted(purchase, plan)
        member = purchase.action.chosen.id
        target = next(row.resource.value for row in plan.fidelities if row.reference == purchase.action.fidelity)
        if target != int(target) or not 1 <= target <= len(FAULTS):
            raise ValueError("Constructed CI fidelity requires a supported integer build target")
        stop = int(target)
        start, total = self._previous(purchase, plan, prior)
        for index in range(start, stop):
            self.reads.append((member, index))
            total += float(FAULTS[index] not in ORDERS[member])
        return AcquisitionReceipt(
            schema_version=1,
            purchase=purchase,
            producer=Reference(id="constructed-ci-prefix-executor", version="1"),
            outcome=Scalar(spec=plan.outcome, value=total / stop, status="observed"),
            charge=quantity(stop - start),
            delay_steps=0,
        )


def run(
    method: Literal["equal_allocation_v1", "successive_halving_v1"] = "successive_halving_v1",
    billing: Literal["incremental", "stateless_rerun"] = "incremental",
) -> tuple[TerminalAllocator, CIPool]:
    """Execute a full real constructed CI pool with the actual allocation core.

    Parameters
    ----------
    method : str, optional
        Frozen terminal comparison profile, default finite successive halving.
    billing : str, optional
        Explicit resumed-prefix or stateless-rerun accounting.

    Returns
    -------
    tuple of TerminalAllocator and CIPool
        Complete actual ledger and independently counted source-read audit.

    Raises
    ------
    ValueError
        Unsupported or incompatible experiment contract.

    Notes
    -----
    No network, external physical work, durable acknowledgement or evaluation
    inference is implied. Charged work may differ under a matched budget ceiling.
    """
    plan = make_plan(method, billing)
    owner = TerminalAllocator(plan)
    pool = CIPool()
    while (purchase := owner.inspect().next_purchase) is not None:
        prior = next(
            (
                row
                for row in reversed(owner.export_state().receipts)
                if row.purchase.action.chosen == purchase.action.chosen
            ),
            None,
        )
        owner.accept(pool.acquire(purchase, plan, prior))
    return owner, pool


if __name__ == "__main__":
    for selected_method, selected_billing in (
        ("equal_allocation_v1", "incremental"),
        ("successive_halving_v1", "incremental"),
        ("successive_halving_v1", "stateless_rerun"),
    ):
        allocator, _ = run(selected_method, selected_billing)
        print(selected_method, selected_billing, allocator.inspect().to_json())
