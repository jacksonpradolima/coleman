"""Pure conservative source-exposure qualification, separate from inference."""

from __future__ import annotations

from typing import Literal

from coleman.evaluation.design.selection import SelectionManifest
from coleman.evaluation.design.source_contracts import SourcePartitionManifest, SourcePrefix, SourceReceipt


def _narrow(receipt: SourceReceipt, manifests: dict[str, SourcePartitionManifest]) -> bool:
    """Require complete matching metadata for every narrower partition claim."""
    request = receipt.request
    if request.potential_test_exposure or not receipt.partition_manifest_verified:
        return False
    manifest = manifests.get(str(request.partition_manifest_fingerprint))
    return manifest is not None and manifest.scope == request.scope and manifest.snapshot == request.snapshot


def _development_reasons(
    selection: SelectionManifest,
    earlier: tuple[SourceReceipt, ...],
    manifests: dict[str, SourcePartitionManifest],
) -> set[str]:
    """Keep protected trials and unknown development history visible."""
    reasons = set()
    accesses = {prior.fingerprint: prior for prior in earlier}
    attempted = {attempt.trial_id for attempt in selection.ledger.attempts}
    for trial in selection.ledger.trials:
        if trial.partition in {"test", "unknown"}:
            reasons.add("development_used_protected_or_unknown_data")
        observed = (
            trial.status != "declared" or trial.results or trial.completed_at is not None or trial.trial_id in attempted
        )
        if observed and not trial.source_accesses:
            reasons.add("development_access_history_unverified")
        for digest in trial.source_accesses:
            access = accesses.get(digest)
            if access is None or not _narrow(access, manifests) or access.requested_at > selection.actual_selected_at:
                reasons.add("development_access_history_unverified")
    return reasons


def source_reasons(
    prefix: SourcePrefix,
    receipt: SourceReceipt,
    selection: SelectionManifest,
    mode: Literal["produced", "imported"],
    manifests: tuple[SourcePartitionManifest, ...],
) -> tuple[str, ...]:
    """Explain why source history cannot qualify as first protected use.

    Parameters
    ----------
    prefix : SourcePrefix
        Complete internally valid source prefix retained by the adapter.
    receipt : SourceReceipt
        Exact current reserved evaluation/import access within that prefix.
    selection : SelectionManifest
        Complete supplied development trials, including unsuccessful ones.
    mode : {'produced', 'imported'}
        Declared before loading. Imports retain their original evaluation identity
        and never gain fresh confirmation by being re-encoded.
    manifests : tuple of SourcePartitionManifest
        Exact metadata actually verified by the adapter for narrow prior reads.

    Returns
    -------
    tuple of str
        Stable sorted reasons. An empty tuple only qualifies declared mediated
        history; it does not authenticate upstream access or prove independence.

    Raises
    ------
    ValueError
        Current receipt is absent/mismatched or manifests contain duplicates.

    Notes
    -----
    Source ordinal order survives clock ties/backward wall-clock changes. Seeds,
    candidates, snapshots and scope versions cannot create a clean new holdout.
    Legacy evaluation access eligibility remains an additional independent gate.
    """
    index = receipt.sequence - 1
    if index >= len(prefix.receipts) or prefix.receipts[index] != receipt:
        raise ValueError("Evaluation source reservation is not in the retained prefix")
    known = {manifest.fingerprint: manifest for manifest in manifests}
    if len(known) != len(manifests):
        raise ValueError("Source qualification requires unique manifest identities")
    reasons = set()
    if prefix.scope.prehistory != "declared_unexposed":
        reasons.add("source_prehistory_not_declared_unexposed")
    if mode == "imported" or receipt.request.purpose != "protected_evaluation":
        reasons.add("imported_or_preexisting_results")
    if any(not _narrow(prior, known) for prior in prefix.receipts[:index]):
        reasons.add("previous_protected_source_access")
    reasons.update(_development_reasons(selection, prefix.receipts[:index], known))
    return tuple(sorted(reasons))
