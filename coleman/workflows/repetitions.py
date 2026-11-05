"""Pure native-receipt projection into the existing complete-grid corpus contract."""

from __future__ import annotations

from datetime import datetime

from coleman.corpus.repeated import RepeatedCorpusReport
from coleman.corpus.repeated_contracts import CorpusRepetition, repetition_execution_id
from coleman.evaluation.protected import FrozenCandidate, FrozenEvaluation
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance, decode_json
from coleman.workflows.contracts import ExperimentPlan, WorkflowError
from coleman.workflows.evidence import FrozenWorkflowConfiguration
from coleman.workflows.lifecycle import NativeRow, TrialResult
from coleman.workflows.planning import protected_repetition_design


def _native_candidates(plan: ExperimentPlan, frozen: FrozenEvaluation) -> dict[str, FrozenCandidate]:
    """Bind the two typed frozen workflow contexts without reading source observations."""
    candidates = {}
    for candidate in frozen.candidates:
        context = FrozenWorkflowConfiguration.model_validate(decode_json(candidate.configuration_json))
        if context.workflow_plan != plan.fingerprint or context.source_scope != plan.dataset.source_fingerprint:
            raise WorkflowError("Repeated native evidence differs from its frozen source context")
        candidates[context.configuration_id] = candidate
    if len(candidates) != 2 or {item.candidate_id for item in candidates.values()} != {"baseline", "challenger"}:
        raise WorkflowError("Repeated native evidence requires distinct frozen baseline and challenger")
    return candidates


def _native_rows(receipt: TrialResult) -> dict[str, NativeRow]:
    """Index exact native source membership, rejecting duplicate build rows instead of overwriting them."""
    rows = {f"build-{row.source_build_id}": row for row in receipt.rows}
    if len(rows) != len(receipt.rows):
        raise WorkflowError("Repeated native evidence contains duplicate source builds")
    return rows


def native_repetition_evidence(
    plan: ExperimentPlan,
    frozen: FrozenEvaluation,
    receipts: tuple[TrialResult, ...],
    *,
    assembled_at: datetime,
) -> RepeatedCorpusReport:
    """Project actual native receipts through the authoritative repeated-corpus producer.

    Parameters
    ----------
    plan : ExperimentPlan
        Validated immutable source, native coordinates, objective and seed grid.
    frozen : FrozenEvaluation
        Complete baseline/challenger context. The existing design validator binds
        its source membership, metric units, cutoffs and design fingerprints.
    receipts : tuple[TrialResult, ...]
        Completed evaluation receipts, including original native rows and pins.
        Prefix rows remain unchanged; only registered protected builds contribute
        cells. No native Parquet, checkpoint, source file or outcome loader is opened.
    assembled_at : datetime
        Explicit aware UTC-compatible assembly timestamp, after source availability
        and within the frozen window. The caller records actual execution time;
        reconstruction establishes consistency, not an external clock attestation.

    Returns
    -------
    RepeatedCorpusReport
        Existing versioned complete-grid evidence with unchanged dimensionless
        binary64 native metrics. Its paired_observations method alone defines
        equal-seed/equal-build means, unit identities, source chronology and derived
        provenance. Seeds never create independent repository families.

    Raises
    ------
    ValueError
        Wrong context/receipt, duplicate or missing native membership, invalid
        source/assembly clocks, non-finite values or an incomplete candidate grid.

    Notes
    -----
    Pure bounded projection with no I/O, RNG, worker, transport or audit side effects.
    Execution and local/portable readers share this producer to prevent divergence
    between reported source metadata/provenance and unchanged native receipts.
    """
    design = protected_repetition_design(plan.request, plan.dataset, analysis_plan=frozen.plan)
    design.require_frozen(frozen)
    candidates = _native_candidates(plan, frozen)
    coordinates = {trial.trial_id: trial for trial in plan.trials}
    raw = []
    frozen_digest = frozen.fingerprint
    for receipt in receipts:
        trial = coordinates.get(receipt.trial_id)
        if trial is None or receipt.plan_fingerprint != plan.fingerprint or receipt.stage != "evaluation":
            raise WorkflowError("Repeated native evidence requires planned evaluation receipts")
        candidate = candidates.get(trial.configuration_id)
        if candidate is None:
            raise WorkflowError("Repeated native evidence contains an unselected configuration")
        rows = _native_rows(receipt)
        source_revision = receipt.fingerprint
        configuration = candidate.configuration_fingerprint
        execution_id = repetition_execution_id(candidate.candidate_id, trial.seed)
        for build in design.builds:
            row = rows.get(build.member_id)
            if row is None:
                raise WorkflowError("Repeated native evidence is missing a protected source build")
            raw.append(
                CorpusRepetition(
                    candidate_id=candidate.candidate_id,
                    member_id=build.member_id,
                    seed=trial.seed,
                    state="completed",
                    record=DomainRecord(
                        schema_version=1,
                        namespace="ci.workflow.repetitions",
                        run_id=receipt.native_run_id,
                        execution_id=execution_id,
                        decision_id=build.member_id,
                        data_fingerprint=frozen.data_fingerprint,
                        available_at=assembled_at,
                        provenance=ResultProvenance(
                            producer="coleman.workflow.native-projection",
                            producer_version="1",
                            spec_fingerprint=configuration,
                            evaluation_fingerprint=frozen_digest,
                            source_revision=source_revision,
                        ),
                        measurements=(
                            Measurement(
                                name=design.raw_metric,
                                value=getattr(row, plan.request.objective),
                                unit="1",
                                meaning=design.raw_semantics.meaning,
                                direction="maximize",
                                status="observed",
                            ),
                        ),
                    ),
                )
            )
    return RepeatedCorpusReport(
        schema_version=1, design=design, frozen=frozen, rows=tuple(raw), assembled_at=assembled_at
    )
