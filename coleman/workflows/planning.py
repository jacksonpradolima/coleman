"""Pure bounded planning over existing sweep primitives; no execution or transport."""

from __future__ import annotations

import math
from decimal import Decimal

from coleman.corpus.repeated_contracts import BUDGET_RATIO, CorpusBuild, RepetitionDesign
from coleman.evaluation.protected import TemporalPlan
from coleman.results.domain import canonical_json, fingerprint
from coleman.selection.contracts import Quantity, UnitSemantics
from coleman.spec import RunSpec
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.sweep import SweepAxis, SweepSpec, expand_sweep
from coleman.workflows.contracts import (
    MAX_WORKFLOW_ARTIFACT_ENTRIES,
    DatasetManifest,
    ExperimentPlan,
    PlanningRequest,
    Trial,
    WorkflowError,
)


def artifact_entry_bound(request: PlanningRequest, *, trials: int, builds: int) -> int:
    """Reserve conservative directory/file capacity before expanding a native sweep.

    Parameters
    ----------
    request : PlanningRequest
        Declared maximum physical invocations, including interrupted attempts.
    trials, builds : int
        Positive resolved coordinate count and registered native build count.

    Returns
    -------
    int
        Entry allowance, not bytes: 128 shared entries, 64 directories/metadata
        entries per trial, and 5 build entries plus 16 transient/receipt entries
        per trial/invocation. Five covers native Parquet, per-build checkpoints,
        both immutable stage copies and frozen learning checkpoints. Reserving
        the complete tree again for every invocation conservatively covers retries.

    Notes
    -----
    Pure metadata arithmetic; no grid allocation, outcomes, RNG or I/O. It shares
    the worker's fixed 32,768-entry ceiling. Byte/RSS/deadline limits still require
    runtime observation; this reservation does not predict physical artifact size.
    """
    return 128 + trials * (64 + request.limits.max_attempts * (5 * builds + 16))


def _cardinality(request: PlanningRequest, dataset: DatasetManifest) -> None:
    """Reject oversized/invalid search work before any grid list is materialized."""
    budgets, coefficients = len(request.budget.values), len(request.ucb_exploration)
    variable = [size for size in (budgets, coefficients) if size > 1]
    if request.mode == "zip" and len(set(variable)) > 1 and "UCB" in request.policies:
        raise WorkflowError("Zip variable dimensions must have matching lengths")
    ucb = math.prod(variable) if request.mode == "grid" else max(variable, default=1)
    configurations = budgets + (ucb if "UCB" in request.policies else 0)
    trials = configurations * len(request.seeds)
    limits = request.limits
    if dataset.size_bytes >= limits.max_output_bytes:
        raise WorkflowError("Registered input must leave space in the output byte allowance for native artifacts")
    if configurations > limits.max_configurations or trials > limits.max_trials:
        raise WorkflowError("Search space exceeds configuration/trial limits before expansion")
    if trials * len(dataset.build_ids) * limits.max_attempts > limits.max_build_work:
        raise WorkflowError("Worst-case native build work, including retries, exceeds the declared cap")
    if request.evaluation == "protected":
        protected_repetition_design(request, dataset)
    if artifact_entry_bound(request, trials=trials, builds=len(dataset.build_ids)) > MAX_WORKFLOW_ARTIFACT_ENTRIES:
        raise WorkflowError("Worst-case workflow artifact entries exceed the worker ceiling before expansion")


def _protected_request(request: PlanningRequest) -> None:
    """Require executable native protected settings before interpreting a ratio budget."""
    if request.evaluation != "protected":
        raise WorkflowError("Protected repetition design requires a protected request")
    if request.limits.max_attempts < 2:
        raise WorkflowError("Protected execution requires at least two worker invocations for tuning and evaluation")
    if request.budget.mode.value != "ratio" or len(request.budget.values) != 1 or "UCB" not in request.policies:
        raise WorkflowError(
            "Protected v1 requires one ratio budget and Random/UCB; other budgets support descriptive replay"
        )


def _protected_eligibility(dataset: DatasetManifest) -> None:
    """Validate source eligibility in native build order without reading outcomes."""
    if dataset.temporal_plan is None or not dataset.metadata:
        raise WorkflowError("Protected execution requires host-registered temporal/group metadata")
    if dataset.evidence != "constructed_mechanism":
        raise WorkflowError("Insufficient evidence: source registration does not grant empirical protected admission")
    if not all(item.costs_known_at_decision for item in dataset.metadata):
        raise WorkflowError("Protected execution requires declared predecision scheduling costs")
    decisions = dataset.split_by_build()
    if not all(item.eligible for item in decisions):
        raise WorkflowError("Protected execution rejects late/unavailable outcomes and overlapping groups")
    partitions = tuple(item.partition for item in decisions)
    if set(partitions) != {"train", "validation", "test"}:
        raise WorkflowError("Protected execution requires nonempty train, validation and test stages")
    order = {"train": 0, "validation": 1, "test": 2}
    if list(partitions) != sorted(partitions, key=lambda item: order[str(item)]):
        raise WorkflowError("Native stage continuation requires chronological contiguous partitions")


def protected_repetition_design(
    request: PlanningRequest, dataset: DatasetManifest, *, analysis_plan: TemporalPlan | None = None
) -> RepetitionDesign:
    """Validate the actual protected repetition contract before expanding native trials.

    Parameters
    ----------
    request : PlanningRequest
        Protected request with explicit seeds, objective and one native ratio
        budget. The existing design validates units and its per-candidate cell cap.
    dataset : DatasetManifest
        Complete declared build/observation identities and source split. Decisions
        join by observation/revision identity into the original native build order.
    analysis_plan : TemporalPlan or None
        Optional metric assembly window. Only test_end may extend the original
        source cutoff; source train/validation boundaries and field mappings remain
        fixed. None uses the unchanged source temporal plan for preflight.

    Returns
    -------
    RepetitionDesign
        Existing authoritative corpus contract, including the 16,384 build/seed
        cells-per-candidate bound, complete membership and sibling/revision guards.
        Seeds remain repetitions, never additional independent repository families.

    Raises
    ------
    ValueError
        Ineligible source/request, shifted partitions/mappings, invalid native
        units, ambiguous membership or excessive cells. No sweep is expanded.

    Notes
    -----
    Pure bounded metadata validation: no outcome, file, RNG, worker or audit access.
    The application uses this same factory when freezing its actual assembly
    window, preventing planning/admission from accepting an unbuildable design.
    """
    request, dataset = PlanningRequest.model_validate(request), DatasetManifest.model_validate(dataset)
    _protected_request(request)
    _protected_eligibility(dataset)
    assert dataset.temporal_plan is not None
    source = dataset.temporal_plan
    analysis = source if analysis_plan is None else TemporalPlan.model_validate(analysis_plan)
    expected = TemporalPlan.model_validate({**source.model_dump(), "test_end": analysis.test_end})
    if analysis != expected or analysis.test_end < source.test_end:
        raise WorkflowError("Analysis window must retain source partitions and field mappings")
    decisions = dataset.split_by_build()
    builds = tuple(
        CorpusBuild(
            member_id=f"build-{item.build_id}", project_id=item.unit_id, unit_id=item.unit_id, metadata=item.observation
        )
        for item, decision in zip(dataset.metadata, decisions, strict=True)
        if decision.partition == "test"
    )
    metric = request.objective
    return RepetitionDesign(
        schema_version=1,
        fold_id="workflow-protected-v1",
        data_fingerprint=dataset.fingerprint,
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        plan=analysis,
        warmup_end=analysis.train_end,
        budget=Quantity(value=Decimal(str(request.budget.values[0])), semantics=BUDGET_RATIO),
        seeds=request.seeds,
        builds=builds,
        raw_metric=f"workflow_{metric}",
        raw_semantics=UnitSemantics(unit="1", meaning=f"coleman.native.{metric}.binary64.v1"),
        cost_projection="binary32_carried_binary64_seconds_v1",
    )


def base_spec(request: PlanningRequest, policy: str) -> RunSpec:
    """Return normalized native configuration without ambient host paths or side effects.

    Parameters
    ----------
    request : PlanningRequest
        Validated native budgets, RNFail policy parameters and seed coordinates.
    policy : str
        Host-approved Random or UCB native name.

    Returns
    -------
    RunSpec
        Complete normalized specification; execution supplies owned local paths.
        Legacy native run identity and sweep semantics are unchanged.
    """
    return RunSpec.model_validate(
        {
            "execution": {"independent_executions": 1, "parallel_pool_size": 1},
            "experiment": {
                "datasets_dir": "inputs",
                "datasets": [request.dataset_id],
                "policies": [policy],
                "rewards": ["RNFail"],
                "budget": {"mode": request.budget.mode, "values": [request.budget.values[0]]},
            },
            "algorithm": {} if policy == "Random" else {"ucb": {"rnfail": {"c": request.ucb_exploration[0]}}},
            "checkpoint": {"enabled": True, "interval": 1},
            "results": {"out_dir": "native", "sink": "parquet", "top_k_prioritization": 10000},
        }
    )


def resolve_trials(request: PlanningRequest, dataset: DatasetManifest) -> tuple[Trial, ...]:
    """Count before expansion and reuse native grid/zip/seed execution coordinates.

    Parameters
    ----------
    request : PlanningRequest
        Typed dimensions; grid declaration order is immaterial, zip correspondence
        remains meaningful. Random never receives irrelevant UCB axes.
    dataset : DatasetManifest
        Actual frozen membership used to bound worst-case build work.

    Returns
    -------
    tuple[Trial, ...]
        Unique coordinates sorted by content identity, independent of execution order.

    Raises
    ------
    ValueError
        Invalid, duplicate, incompatible or oversized dimensions before expansion.
    """
    request, dataset = PlanningRequest.model_validate(request), DatasetManifest.model_validate(dataset)
    _cardinality(request, dataset)
    trials = []
    for policy in sorted(request.policies):
        params = {}
        if len(request.budget.values) > 1:
            params["experiment.budget.values"] = [[value] for value in request.budget.values]
        if policy == "UCB" and len(request.ucb_exploration) > 1:
            params["algorithm.ucb.rnfail.c"] = list(request.ucb_exploration)
        if not params:
            params = {"experiment.policies": [[policy]]}
        axis = SweepAxis(mode=request.mode, params=params)
        specs = expand_sweep(base_spec(request, policy), SweepSpec(axes=[axis], seeds=sorted(request.seeds)))
        for spec in specs:
            assert spec.execution.seed is not None
            coefficient = spec.algorithm.model_dump()["ucb"]["rnfail"]["c"] if policy == "UCB" else None
            config = {
                "policy": policy,
                "c": coefficient,
                "budget_mode": request.budget.mode.value,
                "budget_value": spec.experiment.budget.values[0],
                "reward": "RNFail",
            }
            resolved = spec.model_dump(mode="json")
            configuration = spec.model_dump(mode="json")
            configuration["execution"]["seed"] = None
            config_id = fingerprint(["coleman.workflow.configuration.v1", dataset.fingerprint, config, configuration])
            trials.append(
                Trial(
                    trial_id=fingerprint([config_id, spec.execution.seed]),
                    configuration_id=config_id,
                    policy=policy,
                    exploration=coefficient,
                    budget_value=spec.experiment.budget.values[0],
                    seed=spec.execution.seed,
                    resolved_spec_json=canonical_json(resolved),
                )
            )
    return tuple(sorted(trials, key=lambda trial: trial.trial_id))


def make_plan(
    request: PlanningRequest, dataset: DatasetManifest, runtime: InstalledRuntime, *, native_threads: int = 1
) -> ExperimentPlan:
    """Freeze resolved inputs, runtime and exact bounded trial coordinates without I/O.

    Parameters
    ----------
    request : PlanningRequest
        Client-selected approved dimensions and discovered defaults.
    dataset : DatasetManifest
        Host-registered exact input/evidence snapshot.
    runtime : InstalledRuntime
        Existing conservative producer/base-runtime identity captured by the host.
    native_threads : int
        Explicit Polars pool size for execution/continuation compatibility. Defaults
        to a one-thread hypothetical plan; the application captures actual host size.

    Returns
    -------
    ExperimentPlan
        Canonical executable plan; equivalent grid ordering retains its fingerprint.

    Raises
    ------
    ValueError
        Ineligible or oversized inputs, before any native execution is admitted.
    """
    request = PlanningRequest.model_validate(request)
    payload = request.model_dump()
    payload["policies"], payload["seeds"] = sorted(request.policies), sorted(request.seeds)
    if request.mode == "grid":
        payload["ucb_exploration"] = sorted(request.ucb_exploration)
        payload["budget"] = {**request.budget.model_dump(), "values": sorted(request.budget.values)}
    canonical = PlanningRequest.model_validate(payload)
    return ExperimentPlan(
        schema_version=1,
        request=canonical,
        dataset=dataset,
        runtime=runtime,
        trials=resolve_trials(canonical, dataset),
        native_threads=native_threads,
    )
