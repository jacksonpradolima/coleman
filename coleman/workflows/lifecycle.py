"""Durable workflow lifecycle, progress and evidence response contracts."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.evaluation.protected import FamilyReport
from coleman.results.domain import Fingerprint, Finite, Measurement
from coleman.selection.contracts import Immutable
from coleman.workflows.contracts import (
    DatasetManifest,
    ExperimentPlan,
    Trial,
    WorkflowContract,
    WorkflowError,
    WorkflowId,
    WorkflowLimits,
)

State = Literal[
    "queued", "running", "awaiting_evaluation", "interrupted", "failed", "cancelled", "completed", "expired"
]
Stage = Literal["descriptive", "tuning", "evaluation"]
TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted", "expired", "awaiting_evaluation"})


class HostPolicy(WorkflowContract):
    """Finite workspace reservations; stable across transport/coordinator restarts.

    Parameters
    ----------
    caps : WorkflowLimits
        Maximum accepted planning/worker caps; larger requests reject explicitly.
    max_running, max_queued, max_jobs, max_plans, max_datasets : int
        Distinct active/queue/retained metadata bounds. Expired job tombstones and
        saved plans retain their slots; IDs are never silently recycled.
    max_retained_bytes : int
        Aggregate admitted job output reservations, in bytes.
    max_active_memory_bytes : int
        Sum of memory reservations for workers that actually own execution locks.
    max_registered_bytes : int
        Aggregate encoded dataset bytes admitted by trusted host registration.
    retention_seconds : float
        Positive time after a terminal transition until logical artifact expiry.
        Explicit cleanup removes expired bytes while keeping identity tombstones.
    """

    caps: WorkflowLimits = WorkflowLimits(max_configurations=64, max_trials=128)
    max_running: Annotated[int, Field(strict=True, ge=1, le=4)] = 1
    max_queued: Annotated[int, Field(strict=True, ge=0, le=64)] = 8
    max_jobs: Annotated[int, Field(strict=True, ge=1, le=128)] = 64
    max_plans: Annotated[int, Field(strict=True, ge=1, le=128)] = 64
    max_datasets: Annotated[int, Field(strict=True, ge=1, le=32)] = 16
    max_retained_bytes: Annotated[int, Field(strict=True, ge=65536, le=2147483648)] = 536870912
    max_active_memory_bytes: Annotated[int, Field(strict=True, ge=67108864, le=8589934592)] = 2147483648
    max_registered_bytes: Annotated[int, Field(strict=True, ge=65536, le=536870912)] = 134217728
    retention_seconds: Annotated[Finite, Field(gt=0, le=604800)] = 86400.0


class ExecutePlan(WorkflowContract):
    """Caller-scoped durable idempotency request; no execution configuration override.

    Parameters
    ----------
    request_id : str
        Bounded caller key retained after expiry. Different plan/key payloads
        conflict; identical retries return the same logical job across restarts.
    plan_id : str
        Full lowercase SHA-256 hex of a saved host-approved plan.
    """

    request_id: WorkflowId
    plan_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class PlanPreview(WorkflowContract):
    """Bounded discovery/planning result without duplicating a large resolved plan.

    Parameters
    ----------
    plan_id, digest : str
        Stable retained plan handle and complete content fingerprint.
    dataset_id : str
        Approved registry ID; no mutable host path.
    configurations, trials, expected_build_rows : int
        Separate configuration/repetition/native descriptive-output counts.
    coordinates : tuple[TrialCoordinate, ...]
        Exact canonical policy/parameter/budget/seed coordinates, bounded to 128.
    evaluation : str
        Descriptive or protected; protected stage row counts are separate.
    """

    plan_id: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    digest: Fingerprint
    dataset_id: WorkflowId
    configurations: Annotated[int, Field(strict=True, ge=1, le=64)]
    trials: Annotated[int, Field(strict=True, ge=1, le=128)]
    expected_build_rows: Annotated[int, Field(strict=True, ge=1, le=65536)]
    coordinates: Annotated[tuple[TrialCoordinate, ...], Field(min_length=1, max_length=128)]
    evaluation: Literal["descriptive", "protected"]


class TrialCoordinate(Immutable):
    """Compact mathematical coordinate; the retained plan has complete RunSpecs.

    Parameters
    ----------
    trial_id, configuration_id : str
        Canonical logical identities, independent of physical attempt order.
    policy : str
        Approved Random or UCB name.
    exploration : float or None
        UCB coefficient; None means not applicable for Random.
    budget_value : float
        Quantity in the saved plan's explicit scheduling mode/unit.
    seed : int
        Algorithmic repetition, not independent evidence.
    """

    trial_id: Fingerprint
    configuration_id: Fingerprint
    policy: Literal["Random", "UCB"]
    exploration: Finite | None
    budget_value: Finite
    seed: Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]

    @classmethod
    def from_trial(cls, trial: Trial) -> TrialCoordinate:
        """Project a resolved trial without dropping its declared mathematical identity.

        Parameters
        ----------
        trial : Trial
            Revalidated saved plan coordinate.

        Returns
        -------
        TrialCoordinate
            Small structured tool result with no host path or complete dataset.
        """
        return cls.model_validate(trial.model_dump(exclude={"resolved_spec_json"}))


PlanPreview.model_rebuild()


class JobView(WorkflowContract):
    """Durable, caller-owned progress; snapshots do not certify completion by themselves.

    Parameters
    ----------
    job_id, request_id, plan_id : str
        Stable logical identities retained independently of transport lifetime.
    state : str
        Versioned lifecycle, including an explicit protected-evaluation boundary.
    stage : str
        Descriptive replay, tuning prefix or protected continuation.
    completed_trials, total_trials : int
        Counts for the current stage, not statistical inference units.
    current_trial : str or None
        Current logical coordinate; None means unavailable/no active coordinate.
    checkpoint_step : int
        Last actual durable native step reported for the current attempt.
    attempts : int
        Retained physical trial invocations; retries are not new configurations.
    invocations : int
        Physical job-worker starts, separately bounded even before a trial begins.
    elapsed_seconds : float
        Observed total job age in seconds, including queues/disconnection.
    eta_seconds : Measurement
        Explicit unavailable remaining-time estimate; no invented ETA.
    worker_active : bool
        Kernel-verified ownership, never health inferred from a PID.
    result_digest : str or None
        Present only after a complete verified workflow receipt is published.
    error : str or None
        Bounded stable failure reason without host paths or exception tracebacks.
    """

    job_id: WorkflowId
    request_id: WorkflowId
    plan_id: WorkflowId
    state: State
    stage: Stage
    completed_trials: Annotated[int, Field(strict=True, ge=0, le=128)]
    total_trials: Annotated[int, Field(strict=True, ge=1, le=128)]
    current_trial: Fingerprint | None = None
    checkpoint_step: Annotated[int, Field(strict=True, ge=0, le=512)] = 0
    attempts: Annotated[int, Field(strict=True, ge=0, le=2048)] = 0
    invocations: Annotated[int, Field(strict=True, ge=0, le=8)] = 0
    elapsed_seconds: Annotated[Finite, Field(ge=0)]
    eta_seconds: Measurement
    worker_active: Annotated[bool, Field(strict=True)]
    result_digest: Fingerprint | None = None
    error: Annotated[str, Field(strict=True, max_length=128)] | None = None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Prevent false completion and contradictory progress/receipt declarations."""
        if self.completed_trials > self.total_trials:
            raise ValueError("Completed trials exceed declared stage coordinates")
        if (self.state == "completed" and self.result_digest is None) or (
            self.state not in ("completed", "expired") and self.result_digest is not None
        ):
            raise ValueError("Only completed/expired workflows retain a completion digest")
        return self


class NativeRow(Immutable):
    """Explicit native per-build summary; old Parquet columns are never renamed in place.

    Parameters
    ----------
    step, source_build_id : int
        Local round ordinal and exact registered source BuildId.
    napfd, apfdc : float
        Original finite dimensionless quality metrics, not probabilities or money.
    tests_ran, tests_not_ran : int
        Native selected/unselected test counts.
    modeled_full_seconds : float
        Recorded full-suite modeled duration, separate from observed runtime.
    process_cpu_seconds, process_wall_seconds : float or None
        Actual process cumulative samples in seconds for this native attempt;
        unavailable readings stay None and are not zero.
    ranking : tuple[str, ...]
        Complete actual native order retained for independent reference checks.
    attempt_id : str
        Actual native execution_id, bound to a retained physical attempt event.
        Resumed output can retain a prefix from an earlier physical invocation.
    """

    step: Annotated[int, Field(strict=True, ge=1, le=512)]
    source_build_id: Annotated[int, Field(strict=True)]
    napfd: Finite
    apfdc: Finite
    tests_ran: Annotated[int, Field(strict=True, ge=0)]
    tests_not_ran: Annotated[int, Field(strict=True, ge=0)]
    modeled_full_seconds: Annotated[Finite, Field(ge=0)]
    process_cpu_seconds: Annotated[Finite, Field(ge=0)] | None
    process_wall_seconds: Annotated[Finite, Field(ge=0)] | None
    ranking: Annotated[tuple[Annotated[str, Field(strict=True, max_length=4096)], ...], Field(max_length=10000)]
    attempt_id: WorkflowId

    @model_validator(mode="after")
    def _ranking(self) -> Self:
        """Require complete unique native membership, including valid zero selected tests."""
        if len(self.ranking) != self.tests_ran + self.tests_not_ran or len(set(self.ranking)) != len(self.ranking):
            raise ValueError("Native ranking must contain every selected/unselected testcase exactly once")
        return self


class FilePin(Immutable):
    """Immutable evidence membership; paths are relative artifact names, never executable inputs.

    Parameters
    ----------
    name : str
        Canonical relative path validated by the application storage boundary.
    sha256 : str
        Identified content digest.
    size_bytes : int
        Exact completed bytes; incomplete temporary writes have no pin/receipt.
    """

    name: Annotated[str, Field(strict=True, min_length=1, max_length=512)]
    sha256: Fingerprint
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=67108864)]

    @model_validator(mode="after")
    def _relative(self) -> Self:
        """Reject aliases, traversal, backslashes and absolute artifact names."""
        path = PurePosixPath(self.name)
        if path.is_absolute() or ".." in path.parts or "\\" in self.name or path.as_posix() != self.name:
            raise ValueError("Artifact names must be canonical relative paths")
        return self


class NativeSemantics(Immutable):
    """Fixed units/meanings of workflow projections; legacy native bytes are unchanged.

    Parameters
    ----------
    quality_unit, quality_direction : str
        Dimensionless NAPFD/APFDc, both maximized; neither is money/probability.
    napfd_meaning, apfdc_meaning : str
        Original native full-universe binary64 metric semantics, explicitly versioned.
    duration_unit, duration_meaning : str
        Modeled source seconds, using explicit binary64 fsum. Native testcase
        costs separately cross the existing binary32 boundary.
    runtime_unit, runtime_meaning : str
        Observed cumulative per-physical-attempt process samples in seconds. Null
        means unavailable; resuming may change the physical sample origin.
    count_unit : str
        Selected/unselected testcase counts, distinct from statistical units.
    """

    quality_unit: Literal["1"] = "1"
    quality_direction: Literal["maximize"] = "maximize"
    napfd_meaning: Literal["coleman.native.napfd.binary64.v1"] = "coleman.native.napfd.binary64.v1"
    apfdc_meaning: Literal["coleman.native.apfdc.binary64.v1"] = "coleman.native.apfdc.binary64.v1"
    duration_unit: Literal["s"] = "s"
    duration_meaning: Literal["source-binary64-duration-fsum-v1"] = "source-binary64-duration-fsum-v1"
    runtime_unit: Literal["s"] = "s"
    runtime_meaning: Literal["native-physical-attempt-cumulative-sample-v1"] = (
        "native-physical-attempt-cumulative-sample-v1"
    )
    count_unit: Literal["count"] = "count"


class TrialResult(WorkflowContract):
    """Complete-last native stage receipt; rows reference the actual legacy output.

    Parameters
    ----------
    trial_id, plan_fingerprint : str
        Declared logical coordinate and complete plan context. The coordinate
        definition is checked against that plan; attribution of native bytes to
        its seed/treatment/source remains an unverified producer assertion.
    stage : str
        Descriptive, tuning prefix or protected continuation.
    native_run_id : str
        Producer-asserted native configuration identity, never a replacement
        workflow ID or verified portable provenance. Its hash includes host paths
        absent from portable evidence; unchanged Parquet does not bind this claim.
    rows : tuple[NativeRow, ...]
        Complete prefix/output rows for this stage; protected filtering happens
        only after real stage-bounded native execution, never post-hoc tuning.
    files : tuple[FilePin, ...]
        Actual completed native result membership; checkpoints are host-owned.
    semantics : NativeSemantics
        Explicit fixed units and versioned meaning for every native row projection.
    """

    trial_id: Annotated[
        Fingerprint,
        Field(description="Declared plan coordinate; native seed/treatment/source attribution is producer asserted."),
    ]
    plan_fingerprint: Fingerprint
    stage: Stage
    native_run_id: Annotated[
        str,
        Field(
            strict=True,
            pattern=r"^[0-9a-f]{12}$",
            description="Producer-asserted native ID; private path-derived identity is not verified by portable bytes.",
        ),
    ]
    rows: Annotated[tuple[NativeRow, ...], Field(min_length=1, max_length=512)]
    files: Annotated[tuple[FilePin, ...], Field(min_length=1, max_length=4096)]
    semantics: NativeSemantics = NativeSemantics()

    @property
    def native_execution_bindings_verified(self) -> Literal[False]:
        """Return false: portable native bytes do not prove this receipt's trial/seed/treatment assignment.

        This trust qualification is not serialized, preserving canonical v1
        receipt/report digests. Logical plan definitions and metric integrity are
        validated independently; trusting execution attribution requires producer
        admission or independent reproduction.
        """
        return False


def report_assumptions(plan: ExperimentPlan) -> tuple[str, ...]:
    """Return the exact source qualifications and mode-specific producer limitations.

    Parameters
    ----------
    plan : ExperimentPlan
        Frozen source declarations and descriptive/protected interpretation mode.

    Returns
    -------
    tuple[str, ...]
        Original source assumptions followed by the unchanged v1 producer caveats
        in canonical order. No outcome is inspected and no state is modified.
    """
    if plan.request.evaluation == "descriptive":
        return plan.dataset.assumptions + ("Seeds are repetitions, not independent repositories.",)
    return plan.dataset.assumptions + (
        "Constructed source; no empirical source admission.",
        "Seeds/builds are aggregated within declared repository families; no automatic production promotion.",
    )


class WorkflowReport(WorkflowContract):
    """Bounded descriptive/evaluation report; a tuning selection never certifies superiority.

    Parameters
    ----------
    plan_fingerprint : str
        Complete resolved plan/data/producer binding.
    trials : tuple[TrialResult, ...]
        Canonical retained stage receipts, distinct from retry attempts.
    selected_configuration : str or None
        Deterministic validation/descriptive choice; no automatic production action.
    fallback_configuration : str or None
        Same-budget Random configuration retained for conservative operational use
        when evidence is insufficient. Never an automatic production activation.
    interpretation : str
        Explicit descriptive or protected-insufficient conclusion.
    assumptions : tuple[str, ...]
        Retained source qualification and declared evidence limitations.
    comparison : FamilyReport or None
        Existing audited evaluator report, with independent-unit/multiplicity rules.
    physical_attempts : tuple[PhysicalAttempt, ...]
        Retained pre-execution reservations, including interruptions and prefix
        stages. Native row IDs bind observed runtime to these physical events.
    """

    plan_fingerprint: Fingerprint
    trials: Annotated[tuple[TrialResult, ...], Field(max_length=128)]
    selected_configuration: Fingerprint | None
    fallback_configuration: Fingerprint | None = None
    interpretation: Literal["descriptive_only", "insufficient_evidence"]
    assumptions: Annotated[tuple[str, ...], Field(max_length=32)]
    comparison: FamilyReport | None = None
    physical_attempts: Annotated[tuple[PhysicalAttempt, ...], Field(max_length=1024)] = ()


class DatasetSummary(Immutable):
    """Compact approved-source identity; the saved plan retains complete metadata.

    Parameters
    ----------
    dataset_id, digest, source_fingerprint, projection_fingerprint : str
        Approved ID and exact manifest/source/projection identities.
    rows, builds, maximum_tests : int
        Observations, builds and largest testcase universe; not inference units.
    duration_unit, evidence : str
        Explicit seconds and source qualification; registration grants no admission.
    protected_eligible : bool
        Whether declared source metadata supports this initial protected integration.
    eligibility_reason : str or None
        Safe explanation of missing clocks/costs/admission rather than invented values.
    independent_units : int or None
        Declared repository-family count; None means unavailable. Distinct IDs
        alone do not establish independence and seeds never increase this count.
    assumptions : tuple[str, ...]
        Bounded source limitations without raw histories or host paths.
    """

    dataset_id: WorkflowId
    digest: Fingerprint
    source_fingerprint: Fingerprint
    projection_fingerprint: Fingerprint
    rows: Annotated[int, Field(strict=True, ge=1, le=1000000)]
    builds: Annotated[int, Field(strict=True, ge=1, le=512)]
    maximum_tests: Annotated[int, Field(strict=True, ge=1, le=10000)]
    duration_unit: Literal["s"] = "s"
    evidence: Literal["constructed_mechanism", "descriptive_source"]
    protected_eligible: bool
    eligibility_reason: Annotated[str, Field(max_length=512)] | None
    independent_units: Annotated[int, Field(strict=True, ge=1, le=512)] | None
    assumptions: Annotated[tuple[Annotated[str, Field(max_length=512)], ...], Field(max_length=16)]

    @classmethod
    def from_manifest(cls, manifest: DatasetManifest) -> DatasetSummary:
        """Summarize approved identity/eligibility without loading native outcome bytes.

        Parameters
        ----------
        manifest : DatasetManifest
            Validated host registration; no I/O or source admission is performed.

        Returns
        -------
        DatasetSummary
            Compact discovery metadata with explicit evidence restrictions.
        """
        from coleman.workflows.planning import _protected_eligibility

        reason = None
        try:
            _protected_eligibility(manifest)
        except WorkflowError as error:
            reason = str(error)
        return cls(
            dataset_id=manifest.dataset_id,
            digest=manifest.fingerprint,
            source_fingerprint=manifest.source_fingerprint,
            projection_fingerprint=manifest.projection_fingerprint,
            rows=manifest.rows,
            builds=len(manifest.build_ids),
            maximum_tests=manifest.maximum_tests,
            evidence=manifest.evidence,
            protected_eligible=reason is None,
            eligibility_reason=reason,
            independent_units=len({item.unit_id for item in manifest.metadata}) if manifest.metadata else None,
            assumptions=manifest.assumptions,
        )


class WorkflowDiscovery(WorkflowContract):
    """Structured host capabilities; approved IDs and defaults carry their origin.

    Parameters
    ----------
    datasets : tuple[DatasetSummary, ...]
        Bounded source identities, without full temporal histories or host paths.
    policy : HostPolicy
        Immutable persisted resource/admission limits, including retention.
    planning_schema : dict
        Generated strict PlanningRequest JSON Schema; this dictionary is confined
        to transport discovery, never used as a domain execution configuration.
    default_origin : str
        Versioned package defaults; a client must still select a registered dataset.
    policies, reward : tuple[str, ...], str
        Actual initial native allowlist and fixed RNFail transform, without plugin imports.
    operations : tuple[str, ...]
        Shared application operation names; adapters publish their own typed tool/CLI names.
    native_threads : int
        Actual host Polars pool size, frozen into execution/continuation compatibility.
    """

    datasets: Annotated[tuple[DatasetSummary, ...], Field(max_length=32)]
    policy: HostPolicy
    planning_schema: dict
    default_origin: Literal["coleman-workflow-defaults-v1"] = "coleman-workflow-defaults-v1"
    policies: tuple[Literal["Random", "UCB"], ...] = ("Random", "UCB")
    reward: Literal["RNFail"] = "RNFail"
    operations: tuple[str, ...] = (
        "plan",
        "execute",
        "status",
        "cancel",
        "resume",
        "inspect",
        "evaluate",
        "report",
        "export",
        "import_evidence",
        "artifact_page",
        "read_artifact",
    )
    native_threads: Annotated[int, Field(strict=True, ge=1, le=1024)]


class BuildSummary(Immutable):
    """Bounded native row projection; complete rankings stay in immutable artifacts.

    Parameters
    ----------
    step, source_build_id : int
        Actual native ordinal and registered source build ID.
    napfd, apfdc : float
        Unchanged finite dimensionless quality; neither is probability or money.
    tests_ran, tests_not_ran : int
        Actual native selected/unselected testcase counts.
    modeled_full_seconds, process_cpu_seconds, process_wall_seconds : float or None
        Explicit modeled source seconds and nullable observed cumulative per-attempt
        process seconds, with the parent summary's fixed NativeSemantics.
    ranking_digest, ranking_size : str, int
        Complete actual ranking's content fingerprint and testcase count. This
        projection never pretends an omitted/truncated order is the native ranking.
    attempt_id : str
        Actual native execution_id bound to its retained physical event.
    """

    step: Annotated[int, Field(strict=True, ge=1, le=512)]
    source_build_id: Annotated[int, Field(strict=True)]
    napfd: Finite
    apfdc: Finite
    tests_ran: Annotated[int, Field(strict=True, ge=0)]
    tests_not_ran: Annotated[int, Field(strict=True, ge=0)]
    modeled_full_seconds: Annotated[Finite, Field(ge=0)]
    process_cpu_seconds: Annotated[Finite, Field(ge=0)] | None
    process_wall_seconds: Annotated[Finite, Field(ge=0)] | None
    ranking_digest: Fingerprint
    ranking_size: Annotated[int, Field(strict=True, ge=0, le=10000)]
    attempt_id: WorkflowId


class TrialSummary(WorkflowContract):
    """Explicit inspection projection, distinct from a full completed native receipt.

    Parameters
    ----------
    trial_id, receipt_digest : str
        Actual logical coordinate and complete original receipt fingerprint.
    stage, native_run_id : str
        Completed native stage and producer-asserted native configuration ID.
        This claim cannot be recomputed without private host execution paths.
    native_run_id_verification : str
        Always producer_asserted_unverified; native bytes verify metric values
        and declared membership, not the path-derived native ID or its producer.
    native_execution_bindings_verified : bool
        Always false. Native assignment to a trial/seed, policy, budget and source
        is producer asserted; the coordinate definition itself is plan-checked.
    rows : tuple[BuildSummary, ...]
        Finite native quantities, excluding entire testcase ranking histories.
    files_count : int
        Actual complete native Parquet membership count. Retrieve file handles
        through artifact listing after workflow completion; private checkpoints stay local.
    semantics : NativeSemantics
        Fixed explicit units, precision and meaning shared by the complete receipt.
    """

    artifact_type: Literal["coleman-workflow-trial-summary-v1"] = "coleman-workflow-trial-summary-v1"
    trial_id: Fingerprint
    receipt_digest: Fingerprint
    native_run_id_verification: Literal["producer_asserted_unverified"] = "producer_asserted_unverified"
    native_execution_bindings_verified: Literal[False] = False
    stage: Stage
    native_run_id: Annotated[
        str,
        Field(
            strict=True,
            pattern=r"^[0-9a-f]{12}$",
            description="Producer-asserted native ID; private path-derived identity is not verified by portable bytes.",
        ),
    ]
    rows: Annotated[tuple[BuildSummary, ...], Field(min_length=1, max_length=512)]
    files_count: Annotated[int, Field(strict=True, ge=1, le=4096)]
    semantics: NativeSemantics

    @classmethod
    def from_result(cls, receipt: TrialResult) -> TrialSummary:
        """Summarize real completed native evidence without placing full rankings in tool context.

        Parameters
        ----------
        receipt : TrialResult
            Revalidated complete stage receipt, whose original bytes/identity remain unchanged.

        Returns
        -------
        TrialSummary
            Clearly labeled small projection, linked to its full receipt by fingerprint.
        """
        from coleman.results.domain import fingerprint

        rows = tuple(
            BuildSummary(
                **row.model_dump(exclude={"ranking"}),
                ranking_digest=fingerprint(list(row.ranking)),
                ranking_size=len(row.ranking),
            )
            for row in receipt.rows
        )
        return cls(
            schema_version=1,
            trial_id=receipt.trial_id,
            receipt_digest=receipt.fingerprint,
            stage=receipt.stage,
            native_run_id=receipt.native_run_id,
            rows=rows,
            files_count=len(receipt.files),
            semantics=receipt.semantics,
        )


class WorkflowInspection(WorkflowContract):
    """Bounded stage-result page; partial evidence never implies job completion.

    Parameters
    ----------
    job : JobView
        Fresh durable lifecycle snapshot.
    trials : tuple[TrialSummary, ...]
        At most eight explicitly labeled complete-stage projections per page;
        full receipts, native rankings and original bytes remain in artifacts.
    coordinates : tuple[TrialCoordinate, ...]
        Logical stage coordinates on this page, including unfinished trials.
    attempts : tuple[PhysicalAttempt, ...]
        Retained bounded physical invocations for these coordinates, including
        earlier stage/interrupted reservations. A start event is not completion.
    next_offset : int or None
        Next page coordinate; None means no more published stage receipts.
    """

    job: JobView
    trials: Annotated[tuple[TrialSummary, ...], Field(max_length=8)]
    coordinates: Annotated[tuple[TrialCoordinate, ...], Field(max_length=8)] = ()
    attempts: Annotated[tuple[PhysicalAttempt, ...], Field(max_length=64)] = ()
    next_offset: Annotated[int, Field(strict=True, ge=0, le=128)] | None


WorkflowDiscovery.model_rebuild()


class PhysicalAttempt(WorkflowContract):
    """Immutable pre-execution reservation; distinct from logical trial and result.

    Parameters
    ----------
    trial_id, plan_fingerprint : str
        Saved exact coordinate and executable context.
    attempt_id : str
        Job/trial/ordinal physical invocation ID used by native execution_id.
    stage : str
        Stage requesting native work; a continuation does not create a new trial.
    worker_invocation, ordinal : int
        Bounded worker start and per-trial attempt reservation, starting at one.
    started_epoch_seconds : float
        Actual operational wall-clock epoch seconds, never source availability.
        A crash after reservation can mean no native computation occurred.
    """

    trial_id: Fingerprint
    attempt_id: WorkflowId
    plan_fingerprint: Fingerprint
    stage: Stage
    worker_invocation: Annotated[int, Field(strict=True, ge=1, le=8)]
    ordinal: Annotated[int, Field(strict=True, ge=1, le=8)]
    started_epoch_seconds: Finite


WorkflowInspection.model_rebuild()
WorkflowReport.model_rebuild()
