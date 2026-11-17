"""Public terminal-allocation contracts and fully verified ledger checkpoints."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.acquisition._replay import replay
from coleman.acquisition.model import (
    AcquisitionReceipt,
    AllocationPlan,
    CalibrationCase,
    Candidate,
    Fidelity,
    FidelityQualification,
    Purchase,
    SetupCharge,
)
from coleman.core.base import CoreContract

__all__ = [
    "AcquisitionReceipt",
    "AllocationPlan",
    "AllocationState",
    "CalibrationCase",
    "Candidate",
    "Fidelity",
    "FidelityQualification",
    "Purchase",
    "SetupCharge",
]


class AllocationState(CoreContract):
    """Complete bounded ledger checkpoint; replay derives pending/elimination state.

    Attributes
    ----------
    plan : AllocationPlan
        Frozen method, pool, cost model, objective and qualification.
    receipts : tuple of AcquisitionReceipt
        Ordered accepted receipts only, at most 640. Request IDs, observations,
        original producers, delays, actual charges and shared-group usage survive.

    Notes
    -----
    The deterministic next request is reconstructed exactly from this ledger;
    there is no RNG, hidden iterator, pending response or mutable cache to omit.
    JSON reading verifies every transition, but is not durable acknowledgement.
    Use the existing state codec/store for publication. Unknown versions fail;
    version 1 has no legacy checkpoint migration or future-reader fallback.
    """

    plan: AllocationPlan
    receipts: Annotated[tuple[AcquisitionReceipt, ...], Field(max_length=640)] = ()

    @model_validator(mode="after")
    def _ledger(self) -> Self:
        """Validate all accepted transitions with the pure bounded replay core."""
        replay(self.plan, self.receipts)
        return self
