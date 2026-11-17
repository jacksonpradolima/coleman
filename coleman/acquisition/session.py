"""Small state owner usable directly or with the existing composed-state store."""

from coleman.acquisition.contracts import AcquisitionReceipt, AllocationPlan, AllocationState
from coleman.acquisition.engine import AllocationReport, accept_receipt, inspect_allocation


class TerminalAllocator:
    """Own one bounded terminal-allocation ledger, without executing experiments.

    Parameters
    ----------
    plan : AllocationPlan
        Complete frozen candidate/fidelity/cost/objective specification.

    Notes
    -----
    This owner is deterministic and needs no RNG. It does not import providers,
    storage, CLI, MCP or a runner. Mutations below are local and unacknowledged
    until the host publishes them through its selected state mechanism.
    """

    def __init__(self, plan: AllocationPlan) -> None:
        """Initialize an empty ledger without reading outcomes or acquiring work.

        Parameters
        ----------
        plan : AllocationPlan
            Complete validated version 1 experiment contract.
        """
        self._state = AllocationState(schema_version=1, plan=plan)

    def inspect(self) -> AllocationReport:
        """Return the accepted status and deterministic actual acquisition request.

        Returns
        -------
        AllocationReport
            Explicit status, charges, eliminations and next request, without I/O.
        """
        return inspect_allocation(self._state)

    def accept(self, receipt: AcquisitionReceipt) -> AllocationReport:
        """Apply original purchased feedback with exact delivery deduplication.

        Parameters
        ----------
        receipt : AcquisitionReceipt
            Exact pending purchase or identical accepted retry.

        Returns
        -------
        AllocationReport
            New unacknowledged local status and next request.

        Raises
        ------
        ValueError
            Conflicting duplicate, out-of-order delivery or incompatible contract.
            Local state remains unchanged on failure.
        """
        updated = accept_receipt(self._state, receipt)
        report = inspect_allocation(updated)
        self._state = updated
        return report

    def export_state(self) -> AllocationState:
        """Return complete immutable state, without publishing or acknowledging it.

        Returns
        -------
        AllocationState
            Frozen plan and full bounded original-producer receipt ledger.
        """
        return self._state

    def restore_state(self, state: AllocationState) -> None:
        """Explicitly restore complete compatible state on this local owner.

        Parameters
        ----------
        state : AllocationState
            Complete validated ledger with exactly this owner's frozen plan.

        Raises
        ------
        ValueError
            Unknown version, invalid transition or changed configuration.

        Notes
        -----
        No external effect is replayed. Direct callers select their checkpoint;
        durable accepted-head and concurrent-writer protection belong to the
        existing state store, not this in-memory owner.
        """
        checked = AllocationState.model_validate(state)
        if checked.plan.to_json() != self._state.plan.to_json():
            raise ValueError("Allocation continuation requires the exact original plan")
        self._state = checked
