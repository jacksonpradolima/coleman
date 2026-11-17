"""Pure typed terminal-allocation inspection and receipt-delivery transitions."""

from coleman.acquisition._replay import AllocationReport, Elimination, replay
from coleman.acquisition.contracts import AcquisitionReceipt, AllocationState

__all__ = ["AllocationReport", "Elimination", "accept_receipt", "inspect_allocation", "replay"]


def inspect_allocation(state: AllocationState) -> AllocationReport:
    """Return the accepted resource/knowledge status and deterministic next action.

    Parameters
    ----------
    state : AllocationState
        Complete immutable checkpoint, including all accepted original receipts.

    Returns
    -------
    AllocationReport
        Fresh bounded replay with no provider calls or side effects.
    """
    checked = AllocationState.model_validate(state)
    return replay(checked.plan, checked.receipts)


def accept_receipt(state: AllocationState, receipt: AcquisitionReceipt) -> AllocationState:
    """Accept one purchased observation, deduplicating exact logical delivery.

    Parameters
    ----------
    state : AllocationState
        Current complete ledger; publication/host acknowledgement is separate.
    receipt : AcquisitionReceipt
        Original exact requested action, producer, score, delay and actual charge.

    Returns
    -------
    AllocationState
        New complete state; an identical previously accepted delivery returns the
        unchanged complete state and adds neither observation nor charge.

    Raises
    ------
    ValueError
        Conflicting duplicate content, foreign/stale request, post-stop delivery,
        invalid numeric value or incompatible units. Original state is unchanged.

    Notes
    -----
    Delivery deduplication is bounded to this full finite run's ledger. It does
    not deduplicate actual external work, make execution atomic with storage, or
    authorize a retry after an uncertain provider effect. Persist the request
    before execution and reconcile uncertain effects with the producer.
    """
    state = AllocationState.model_validate(state)
    receipt = AcquisitionReceipt.model_validate(receipt)
    for accepted in state.receipts:
        if accepted.purchase.purchase_id == receipt.purchase.purchase_id:
            if accepted.to_json() != receipt.to_json():
                raise ValueError("Conflicting duplicate acquisition receipt")
            return state
    return AllocationState(schema_version=1, plan=state.plan, receipts=(*state.receipts, receipt))
