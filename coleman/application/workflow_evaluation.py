"""Frozen native-state handoff and audited comparisons using existing corpus statistics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

from coleman.application.baseline_storage import read_baseline_artifact
from coleman.application.workflow_artifacts import pin_file, read_attempts, verify_trial
from coleman.artifacts import ArtifactWriter
from coleman.corpus.repeated_contracts import (
    RepetitionDesign,
)
from coleman.evaluation.protected import (
    ComparisonHypothesis,
    EvaluationAudit,
    FrozenCandidate,
    FrozenEvaluation,
    TemporalPlan,
    evaluate,
    freeze,
)
from coleman.results.domain import Fingerprint, decode_json
from coleman.workflows.contracts import ExperimentPlan, Trial, WorkflowContract, WorkflowError
from coleman.workflows.evidence import FrozenLearningState, FrozenWorkflowConfiguration
from coleman.workflows.lifecycle import FilePin, TrialResult, WorkflowReport, report_assumptions
from coleman.workflows.planning import protected_repetition_design
from coleman.workflows.repetitions import native_repetition_evidence


class EvaluationHandoff(WorkflowContract):
    """Immutable protected continuation decision, persisted before holdout access.

    Parameters
    ----------
    design : RepetitionDesign
        Existing full seed/build/family/clock/metric aggregation contract.
    frozen : FrozenEvaluation
        Existing candidates, learned-state identity and entire hypothesis family.
    selected_configuration, challenger_configuration, baseline_configuration : str
        Validation choice and best UCB versus same-budget Random comparison.
        A tie chooses Random, then smaller c, then canonical configuration identity.
    checkpoints : tuple[FilePin, ...]
        Private immutable prefix state snapshots; never transported or executed
        from an imported bundle. Continuation uses the original native stores.
    """

    design: RepetitionDesign
    frozen: FrozenEvaluation
    selected_configuration: Fingerprint
    challenger_configuration: Fingerprint
    baseline_configuration: Fingerprint
    checkpoints: tuple[FilePin, ...]


def choose_configuration(plan: ExperimentPlan, results: tuple[TrialResult, ...], *, ucb_only: bool = False) -> str:
    """Choose the largest declared objective mean with a conservative deterministic tie rule.

    Parameters
    ----------
    plan : ExperimentPlan
        Frozen objective and complete coordinates.
    results : tuple[TrialResult, ...]
        Complete stage receipts. Protected selection uses validation rows only.
    ucb_only : bool
        Select a challenger for comparison even when Random wins validation.

    Returns
    -------
    str
        Stable configuration fingerprint; this is a descriptive choice, not evidence
        that a policy is better. Equal binary64 observations are averaged rationally.
    """
    values: dict[str, list[float]] = {}
    coordinates = {item.trial_id: item for item in plan.trials}
    validation_steps = set(range(1, len(plan.dataset.build_ids) + 1))
    if plan.request.evaluation == "protected":
        assert plan.dataset.temporal_plan is not None
        validation_steps = {
            i for i, decision in enumerate(plan.dataset.split_by_build(), 1) if decision.partition == "validation"
        }
    for result in results:
        trial = coordinates[result.trial_id]
        if ucb_only and trial.policy != "UCB":
            continue
        values.setdefault(trial.configuration_id, []).extend(
            getattr(row, plan.request.objective) for row in result.rows if row.step in validation_steps
        )
    representatives = {trial.configuration_id: trial for trial in plan.trials}

    def order(config: str):
        """Compare exact means before deterministic baseline/parameter/identity tie breaks."""
        trial = representatives[config]
        return (
            -sum((Fraction(value) for value in values[config]), Fraction()) / len(values[config]),
            trial.policy != "Random",
            trial.exploration or 0.0,
            config,
        )

    return min(values, key=order)


def prepare_evaluation(workspace: Path, root: Path, plan: ExperimentPlan) -> EvaluationHandoff:
    """Freeze validation choice, source protocol and actual prefix state before any test access.

    Parameters
    ----------
    workspace, root : pathlib.Path
        Trusted shared audit root and owned job artifact directory.
    plan : ExperimentPlan
        Eligible constructed protected workflow with one ratio budget.

    Returns
    -------
    EvaluationHandoff
        Complete immutable prefix handoff. Identical retries reuse it; no protected
        outcome is read. The worker derives its operational assembly context after
        acquiring execution ownership. Prefix artifacts are verified first.

    Raises
    ------
    ValueError, OSError
        Missing prefix completion, invalid metadata/state or immutable conflict.
    """
    target = root / "evaluation-handoff.json"
    if target.exists():
        return read_baseline_artifact(target, EvaluationHandoff)
    results = tuple(
        verify_trial(root, root / "evidence" / trial.trial_id[7:] / "tuning" / "complete.json", plan, trial, "tuning")
        for trial in plan.trials
    )
    selected, challenger = choose_configuration(plan, results), choose_configuration(plan, results, ucb_only=True)
    baseline = next(trial.configuration_id for trial in plan.trials if trial.policy == "Random")
    assert plan.dataset.temporal_plan is not None
    source_plan = plan.dataset.temporal_plan
    # The original input membership/partitions stay fixed. Derived metrics cannot
    # be backdated: reserve a declared assembly window before accessing holdout.
    assembly_end = max(
        source_plan.test_end,
        datetime.now(UTC)
        + timedelta(seconds=plan.request.limits.attempt_seconds * plan.request.limits.max_attempts + 60),
    )
    analysis_plan = TemporalPlan.model_validate({**source_plan.model_dump(), "test_end": assembly_end})
    design = protected_repetition_design(plan.request, plan.dataset, analysis_plan=analysis_plan)
    pins = []
    by_candidate = []
    for name, config in (("challenger", challenger), ("baseline", baseline)):
        learning = []
        for trial in plan.trials:
            if trial.configuration_id != config:
                continue
            native = root / "native" / trial.trial_id[7:]
            source_files = sorted(path for path in native.rglob("*") if path.is_file() and "checkpoints" in path.parts)
            if not source_files:
                raise WorkflowError("Protected continuation requires actual committed learning state")
            trial_pins = []
            for path in source_files:
                destination = root / "frozen-learning" / trial.trial_id[7:] / path.relative_to(native)
                with path.open("rb") as stream:
                    ArtifactWriter(root).write_immutable_stream(
                        destination,
                        iter(lambda stream=stream: stream.read(1048576), b""),
                        maximum_bytes=plan.request.limits.max_output_bytes,
                    )
                pin = pin_file(root, destination)
                pins.append(pin)
                trial_pins.append(pin)
            learning.append(
                FrozenLearningState(trial_id=trial.trial_id, seed=trial.seed, checkpoints=tuple(trial_pins))
            )
        by_candidate.append(
            FrozenCandidate.from_configuration(
                name,
                FrozenWorkflowConfiguration(
                    configuration_id=config,
                    source_scope=plan.dataset.source_fingerprint,
                    workflow_plan=plan.fingerprint,
                    learning_state=tuple(learning),
                    corpus_repetition_design=design.fingerprint,
                ).model_dump(mode="json"),
            )
        )
    from coleman.application.workflow_validation import protected_protocol

    frozen = freeze(
        tuple(by_candidate),
        plan.dataset.fingerprint,
        analysis_plan,
        protected_protocol(plan),
        (
            ComparisonHypothesis(
                hypothesis_id="selected-ucb-versus-random", candidate_id="challenger", baseline_id="baseline"
            ),
        ),
    )
    handoff = EvaluationHandoff(
        schema_version=1,
        design=design,
        frozen=frozen,
        selected_configuration=selected,
        challenger_configuration=challenger,
        baseline_configuration=baseline,
        checkpoints=tuple(pins),
    )
    EvaluationAudit(workspace / "audit").save_freeze(frozen)
    ArtifactWriter(root).write_immutable(target, handoff.to_json())
    return handoff


def evaluation_trials(plan: ExperimentPlan, handoff: EvaluationHandoff) -> tuple[Trial, ...]:
    """Resolve the two predeclared same-budget candidates without adding configurations.

    Parameters
    ----------
    plan : ExperimentPlan
        Retained complete source/runtime/configuration context.
    handoff : EvaluationHandoff
        Persisted validation choice and comparison family.

    Returns
    -------
    tuple[Trial, ...]
        All declared seeds for challenger and Random baseline, in canonical order.
    """
    return tuple(
        trial
        for trial in plan.trials
        if trial.configuration_id in (handoff.challenger_configuration, handoff.baseline_configuration)
    )


def prepare_evaluation_attempt(
    workspace: Path, root: Path, plan: ExperimentPlan, initial: EvaluationHandoff, invocation: int
) -> EvaluationHandoff:
    """Freeze an assembly window after worker ownership, preserving the original decision and audit scope.

    Parameters
    ----------
    workspace, root : pathlib.Path
        Trusted shared audit directory and private job directory. This writes one
        immutable context per admitted worker invocation, before protected access.
    plan : ExperimentPlan
        Original source membership, protocol and finite wall-time/attempt allowance.
    initial : EvaluationHandoff
        Unchanged pre-queue selection and prefix-state snapshots. These snapshots
        never acquire later learned state or a newly selected candidate on resume.
    invocation : int
        Actual owned worker start, one through the plan's finite cap, at most eight.

    Returns
    -------
    EvaluationHandoff
        Original source partitions, choices, learning state and hypothesis family,
        with a current bounded assembly cutoff. Its existing upstream holdout ID
        retains earlier accesses even though the operational context digest changes.

    Raises
    ------
    ValueError, OSError
        Invalid invocation, conflicting publication or context over 16 MiB. No
        outcome is read, access receipt reserved or checkpoint unpickled here.

    Notes
    -----
    Queue time and time between recoveries do not consume this execution window.
    Only the derived-metric assembly cutoff changes; the original handoff remains
    immutable. Reusing a completed report requires no new window or protected access.
    """
    if type(invocation) is not int or not 1 <= invocation <= plan.request.limits.max_attempts:
        raise WorkflowError("Protected execution context requires an admitted worker invocation")
    cutoff = max(
        initial.design.plan.test_end,
        datetime.now(UTC)
        + timedelta(seconds=plan.request.limits.attempt_seconds * plan.request.limits.max_attempts + 60),
    )
    analysis = TemporalPlan.model_validate({**initial.design.plan.model_dump(), "test_end": cutoff})
    design = RepetitionDesign.model_validate({**initial.design.model_dump(), "plan": analysis})
    candidates = tuple(_candidate_for_design(candidate, design) for candidate in initial.frozen.candidates)
    frozen = freeze(
        candidates, initial.frozen.data_fingerprint, analysis, initial.frozen.protocol, initial.frozen.comparison_family
    )
    context = EvaluationHandoff.model_validate({**initial.model_dump(), "design": design, "frozen": frozen})
    payload = context.to_json()
    if len(payload.encode()) > 16777216:
        raise WorkflowError("Protected execution context exceeds its 16 MiB reader limit; reduce the plan")
    EvaluationAudit(workspace / "audit").save_freeze(frozen)
    ArtifactWriter(root).write_immutable(root / "evaluation-attempts" / f"{invocation:02d}.json", payload)
    return context


def _candidate_for_design(candidate: FrozenCandidate, design: RepetitionDesign) -> FrozenCandidate:
    """Rebind only assembly bookkeeping, retaining exact candidate/source/prefix-state membership."""
    configuration = FrozenWorkflowConfiguration.model_validate(decode_json(candidate.configuration_json))
    rebound = FrozenWorkflowConfiguration.model_validate(
        {**configuration.model_dump(), "corpus_repetition_design": design.fingerprint}
    )
    return FrozenCandidate.from_configuration(candidate.candidate_id, rebound.model_dump(mode="json"))


def _require_current_window(handoff: EvaluationHandoff) -> None:
    """Reject expired assembly bookkeeping before another protected-access reservation."""
    if datetime.now(UTC) > handoff.design.plan.test_end:
        raise WorkflowError("Protected assembly window expired; resume with a new execution context")


def require_continuation(root: Path, trial: Trial, prefix_steps: int) -> None:
    """Reject missing/reset protected state before invoking native checkpoint restoration.

    Parameters
    ----------
    root : pathlib.Path
        Trusted job directory containing persisted immutable prefix snapshots.
    trial : Trial
        Selected logical coordinate whose original native paths must be reused.
    prefix_steps : int
        Fully committed training/validation frontier, in processed build ordinals.

    Raises
    ------
    ValueError, OSError
        Missing/backward state or changed initial prefix checkpoint bytes. Advanced
        frontiers still undergo the existing native checkpoint/recovery validation.
        This function never unpickles, repairs or substitutes a fresh run.
    """
    native = root / "native" / trial.trial_id[7:]
    paths = list(native.rglob("progress_ex1.json"))
    if len(paths) != 1:
        raise WorkflowError("Protected continuation cannot reset missing native learning state")
    with paths[0].open("rb") as stream:
        payload = stream.read(16385)
    if len(payload) > 16384:
        raise WorkflowError("Protected checkpoint progress exceeds its bounded schema")
    step = decode_json(payload).get("step_committed")
    if type(step) is not int or step < prefix_steps:
        raise WorkflowError("Protected checkpoint is behind its frozen prefix")
    if step == prefix_steps:
        handoff = read_baseline_artifact(root / "evaluation-handoff.json", EvaluationHandoff)
        prefix = Path("frozen-learning") / trial.trial_id[7:]
        pins = [pin for pin in handoff.checkpoints if Path(pin.name).is_relative_to(prefix)]
        if not pins:
            raise WorkflowError("Protected trial lacks its frozen learning-state membership")
        for pin in pins:
            actual = native / Path(pin.name).relative_to(prefix)
            current = pin_file(root, actual)
            if (current.sha256, current.size_bytes) != (pin.sha256, pin.size_bytes):
                raise WorkflowError("Native initial continuation state differs from its frozen snapshot")


def compare_continuation(
    workspace: Path, root: Path, plan: ExperimentPlan, handoff: EvaluationHandoff, execute_continuation
) -> WorkflowReport:
    """Reserve audited holdout access before continuing native state and aggregating evidence.

    Parameters
    ----------
    workspace, root : pathlib.Path
        Shared persistent access ledger and private job directory.
    plan : ExperimentPlan
        Frozen eligible constructed workflow, without fabricated empirical admission.
    handoff : EvaluationHandoff
        Persisted invocation context with complete candidates/protocol/prefix-state
        binding and an assembly cutoff frozen after worker ownership.
    execute_continuation : callable
        Trusted native callback returning actual complete evaluation stage receipts.

    Returns
    -------
    WorkflowReport
        Existing repeated-corpus aggregation and protected comparison evidence.
        Constructed demonstrations always retain an insufficient-evidence product
        interpretation and conservative fallback, independent of a numerical winner.

    Raises
    ------
    ValueError, OSError
        Changed frozen state, expired assembly window, incomplete/native evidence,
        audit exhaustion or unsupported metric. Failed access remains in the ledger.
    """
    for pin in handoff.checkpoints:
        if pin_file(root, root / pin.name) != pin:
            raise WorkflowError("Frozen learning-state snapshot changed")
    audit = EvaluationAudit(workspace / "audit")
    if len(audit.history(handoff.frozen.protocol.holdout_id)) >= 512:
        raise WorkflowError("Protected scope access slots exhausted")
    completed = []

    def observations():
        """Run only after the existing evaluator durably reserves protected access."""
        receipts = execute_continuation()
        completed.extend(receipts)
        assembled = datetime.now(UTC)
        repeated = native_repetition_evidence(plan, handoff.frozen, receipts, assembled_at=assembled)
        ArtifactWriter(root).write_immutable(
            root / "repetitions" / f"{repeated.fingerprint[7:]}.json", repeated.to_json()
        )
        return repeated.paired_observations()

    from coleman.application.workflow_exposure import source_lock, synchronize_exposures

    with source_lock(workspace, plan.dataset.source_fingerprint):
        synchronize_exposures(workspace, handoff.frozen)
        if len(audit.history(handoff.frozen.protocol.holdout_id)) >= 512:
            raise WorkflowError("Protected scope access slots exhausted")
        _require_current_window(handoff)
        comparison = evaluate(handoff.frozen, audit, observations, actor="local-workflow")
    audit.save_report(comparison)
    return WorkflowReport(
        schema_version=1,
        plan_fingerprint=plan.fingerprint,
        trials=tuple(completed),
        selected_configuration=handoff.selected_configuration,
        fallback_configuration=handoff.baseline_configuration,
        physical_attempts=read_attempts(root, plan),
        interpretation="insufficient_evidence",
        comparison=comparison,
        assumptions=report_assumptions(plan),
    )
