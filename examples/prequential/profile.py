"""Frozen public CI inputs, contracts, clocks and bounded compatibility metadata."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.core.decisions import DecisionRequest
from coleman.core.feedback import FullPanel
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, Identifier, canonical_json, fingerprint
from coleman.core.splits import TemporalObservation, TemporalPlan
from coleman.core.temporal import Timestamp
from coleman.core.timing import DecisionTiming
from coleman.evaluation.design.contracts import (
    BudgetPlan,
    CandidatePlan,
    ExperimentDesign,
    InferenceRequest,
    StrategyBindings,
    Threshold,
)
from coleman.evaluation.design.limits import ProtocolLimits
from coleman.evaluation.design.selection import AttemptLedger, CellIdentity, SelectionManifest
from coleman.evaluation.design.source_contracts import SourceMember, SourceScope, SourceSnapshot
from coleman.evaluation.prequential.comparison import ComparisonDisposition, ComparisonPlan
from coleman.evaluation.prequential.conformance import (
    ReferenceConformancePlan,
    ReferenceConformanceReport,
)
from coleman.evaluation.prequential.contracts import (
    ComponentObservation,
    ConformanceTrace,
    ExecutionKey,
    PrequentialContract,
    StepEvidence,
)
from coleman.evaluation.prequential.execution import PrequentialStep
from coleman.evaluation.protected.contracts import ComparisonHypothesis, FrozenCandidate
from coleman.objectives.bindings import DefinitionBinding
from coleman.replay.contracts import ReplayDecisionRecord, ReplayFeedback
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.storage_contracts import (
    ExecutionCompatibility,
)

from . import model

MODEL = model

TABLE = (
    ((2.0, 6.0), (2.0, 6.0), (6.0, 2.0)),
    ((6.0, 2.0), (6.0, 2.0), (2.0, 6.0)),
    ((4.0, 4.0), (4.0, 4.0), (4.0, 4.0)),
    ((1.0, 7.0), (7.0, 1.0), (3.0, 5.0)),
)


WORLDS = tuple("world-" + str(index) for index in range(len(TABLE)))


STAGE_PAIRS = (
    ("coefficient_learning", "fixed", "plain"),
    ("declared_target_share", "plain", "target"),
    ("expert_bounds", "target", "bounded"),
    ("movement_allowance", "bounded", "movement"),
    ("round_cadence", "movement", "cadence"),
    ("l1_penalty", "cadence", "penalized"),
    ("past_only_forecast_learning", "penalized", "online"),
    ("modeled_accounting", "penalized", "accounting_off"),
)


REFERENCE_RECIPE = MODEL.definition(
    "direct-fixed-ci-reference",
    "Independent fixed halves, forecast 4/4, full tie-break; direct finite-budget reference; no learning.",
)


def primary_component(step: StepEvidence) -> ComponentObservation:
    """Read the explicitly named terminal metric without relying on component order.

    Parameters
    ----------
    step : StepEvidence
        Complete later observations from one original finite CI action.

    Returns
    -------
    ComponentObservation
        terminal_net_seconds in modeled seconds; missing evidence raises ValueError.

    Notes
    -----
    Pure lookup; does not aggregate prefix values or infer physical cost."""
    for component in step.components:
        if component.metric.name == "terminal_net_seconds":
            return component
    raise ValueError("Complete CI evidence requires its named terminal primary component")


def decision_timing(request: DecisionRequest) -> DecisionTiming:
    """Require original complete logical timing without fabricating unknown clocks.

    Parameters
    ----------
    request : DecisionRequest
        Original coefficient or corresponding separate finite-domain request.

    Returns
    -------
    DecisionTiming
        Supplied zoned cutoff/decision/target clocks; missing timing raises ValueError.

    Notes
    -----
    Pure lookup; modeled seconds do not imply elapsed wall-clock occupancy."""
    if request.timing is None:
        raise ValueError("The CI example requires explicitly timed original decisions")
    return request.timing


class FrozenExample(PrequentialContract):
    """Retain the original design/selection/rule before any outcome callback."""

    design: ExperimentDesign
    selection: SelectionManifest
    plan: ComparisonPlan
    environment: ExecutionCompatibility

    @model_validator(mode="after")
    def _joins(self):
        """Reject replacement selection, family or fallback during continuation."""
        self.selection.require_design(self.design)
        if self.plan.design_fingerprint != self.design.fingerprint or self.selection.rule != self.plan.binding:
            raise ValueError("Frozen example declarations have incompatible original bindings")
        self.plan.validate_for(self.design, self.selection)
        return self


class ObservationInput(PrequentialContract):
    """Retain original observations and input clocks for explicit reproduction.

    Attributes
    ----------
    step : StepEvidence
        Original finite action and all five modeled/synthetic measurements. Their
        shared actual delivery/application clocks match the feedback batch.
    coefficient : ReplayDecisionRecord
        Exact expert-mixture producer, distinct from the finite CI action.
    feedback : ReplayFeedback
        Complete original expert losses, equal to modeled seconds divided by
        eight. Only an eligible actual replay application advances the predictor.
    expert_seconds : tuple of float
        Raw fast/full source values in modeled seconds, each in [0, 8]. These
        values become known after the decision; they are never a paid cost.

    Notes
    -----
    This example-specific version-one envelope is a complete retained input,
    rather than an invitation to reconstruct missing times or future outcomes.
    Unknown versions, missing components and conflicting clocks raise ValueError
    before a state transition. Constructing it has no storage side effects.
    """

    step: StepEvidence
    coefficient: ReplayDecisionRecord
    feedback: ReplayFeedback
    expert_seconds: tuple[Binary64, Binary64]

    @model_validator(mode="after")
    def _original(self):
        """Keep explicit components, processing clocks and complete losses distinct."""
        self._component_values()
        self._source_clocks()
        self._expert_panel()
        return self

    def _component_values(self) -> None:
        """Require every declared known component, without imputing missing values."""
        if {item.metric.name for item in self.step.components} != {item.name for item in MODEL.metrics()}:
            raise ValueError("A complete source input requires every explicit domain component")
        if any(item.value.status != "observed" or item.value.value is None for item in self.step.components):
            raise ValueError("The complete authored source profile requires known component values, without imputation")

    def _source_clocks(self) -> None:
        """Join actual source delivery/application timing to the original producer."""
        primary = primary_component(self.step)
        if (
            self.feedback.batch.timing is None
            or self.feedback.batch.timing.usable_at is None
            or primary.applied_at is None
        ):
            raise ValueError("A complete ready input requires known original eligibility/application clocks")
        if any(
            item.timing != self.feedback.batch.timing or item.applied_at != primary.applied_at
            for item in self.step.components
        ):
            raise ValueError("Domain components and expert feedback require the same original source processing clocks")
        if self.feedback.batch.decision_fingerprint != self.step.receipt.coefficient_execution_fingerprint:
            raise ValueError("Original expert feedback differs from its coefficient producer")
        PrequentialStep(schema_version=1, coefficient=self.coefficient, domain=self.step.receipt)

    def _expert_panel(self) -> None:
        """Preserve each original expert's seconds and exact bounded-loss transform."""
        panel = self.feedback.batch.feedback
        if (
            not isinstance(panel, FullPanel)
            or tuple(row.member for row in panel.rows) != MODEL.MEMBERS
            or any(len(row.values) != 1 for row in panel.rows)
        ):
            raise ValueError("Reproduction requires the original complete expert catalogue")
        if tuple(row.values[0].value for row in panel.rows) != tuple(value / 8.0 for value in self.expert_seconds):
            raise ValueError("Reproduced expert seconds differ from their original bounded losses")
        if any(value < 0.0 or value > 8.0 for value in self.expert_seconds):
            raise ValueError("The authored duration source is bounded to zero through eight seconds")
        if any(value != 0.0 and value / 8.0 == 0.0 for value in self.expert_seconds):
            raise ValueError("This binary64 normalization cannot silently erase a nonzero expert duration")


class SourceTick(PrequentialContract):
    """Retain one source delivery and every candidate's original observation.

    Attributes
    ----------
    world : str
        One authored independent-unit identifier; rounds and seeds are not units.
    ordinal : int
        Zero-based decision index, bounded to this three-decision example.
    decision_at : datetime
        Common originally captured UTC decision/cutoff/target tick. Modeled
        seconds do not imply physical elapsed time after this instant.
    observations : tuple of ObservationInput
        Every frozen candidate exactly once, sharing the raw source panel and
        original actual availability/application clocks. No outcome is included
        in the earlier decision-state checkpoint.

    Notes
    -----
    Construction validates this exact public table revision and performs no
    I/O. Missing, duplicated, differently timed or changed-source inputs raise
    ValueError. Reproduction retains these clocks rather than recapturing them.
    """

    world: Identifier
    ordinal: Annotated[int, Field(strict=True, ge=0, le=2)]
    decision_at: Timestamp
    observations: Annotated[tuple[ObservationInput, ...], Field(max_length=16)]

    @model_validator(mode="after")
    def _tick(self):
        """Require the complete fixed mode family and one original logical round."""
        if any(not item.step.components for item in self.observations):
            raise ValueError("A complete source tick cannot contain missing component evidence")
        modes = tuple(item.step.receipt.key.cell.candidate_id for item in self.observations)
        if set(modes) != set(MODEL.MODES) or len(modes) != len(MODEL.MODES):
            raise ValueError("A source tick requires every frozen candidate exactly once")
        if any(
            item.step.receipt.key.cell.member_id != self.world
            or item.step.receipt.executed.proposal.request.round != self.ordinal
            or decision_timing(item.step.receipt.executed.proposal.request).decision_at != self.decision_at
            for item in self.observations
        ):
            raise ValueError("Source tick changes its original world or common decision clock")
        source = self.observations[0]
        if any(
            item.expert_seconds != source.expert_seconds
            or item.feedback.batch.timing != source.feedback.batch.timing
            or primary_component(item.step).timing != primary_component(source.step).timing
            or primary_component(item.step).applied_at != primary_component(source.step).applied_at
            for item in self.observations
        ):
            raise ValueError("All candidates require the same complete source and eligibility/application calendar")
        if self.world not in WORLDS or source.expert_seconds != TABLE[WORLDS.index(self.world)][self.ordinal]:
            raise ValueError("Source delivery differs from this declared public table revision")
        return self


class SourceTape(PrequentialContract):
    """Retain the finite original model input/outcome tape, not new observations.

    Attributes
    ----------
    ticks : tuple of SourceTick
        Exactly twelve unique authored world/round deliveries. Each includes
        the full nine-candidate source panel and original captured UTC clocks.

    Notes
    -----
    This bounded version-one wire format supports reproduction of the same
    exposed evidence. It grants neither source access nor a new confirmation;
    callers must reserve any protected disclosure before reading its body.
    Incomplete grids and unknown versions raise ValueError without side effects.
    """

    ticks: Annotated[tuple[SourceTick, ...], Field(max_length=12)]

    @model_validator(mode="after")
    def _complete(self):
        """Keep the complete authored world/round grid without inventing units."""
        keys = [(tick.world, tick.ordinal) for tick in self.ticks]
        if len(keys) != 12 or set(keys) != {(world, ordinal) for world in WORLDS for ordinal in range(3)}:
            raise ValueError("Completed source tape requires every declared world/round exactly once")
        return self


class CachedExample(PrequentialContract):
    """Keep complete later comparison evidence separate from result-free metadata."""

    report_fingerprint: Fingerprint
    conformance: ReferenceConformanceReport
    disposition: ComparisonDisposition
    traces: Annotated[tuple[ConformanceTrace, ...], Field(max_length=16)]
    tape: SourceTape


class EmissionTick(PrequentialContract):
    """Freeze one common logical tick before any candidate emits that round."""

    ordinal: Annotated[int, Field(strict=True, ge=0, le=2)]
    decision_at: Timestamp


class Invocation(PrequentialContract):
    """Record one actual loader invocation, independently of logical learning."""

    attempt_id: Identifier
    started_at: Timestamp


def read_contract(path: Path, kind, maximum: int = 16 * 1024**2):
    """Read one bounded complete local contract without inventing missing evidence.

    Parameters
    ----------
    path : pathlib.Path
        Trusted original regular artifact; symlinks and missing files are refused.
    kind : type
        Explicit trusted versioned reader, never selected from artifact bytes.
    maximum : int, optional
        Maximum original UTF-8 bytes, default 16 MiB; reads at most one extra byte.

    Returns
    -------
    object
        Actual parsed contract; unknown versions and malformed bytes are refused.

    Raises
    ------
    ValueError, OSError
        Incomplete, oversized, incompatible or unreadable artifact. Protected
        access must already be reserved before reading result-bearing bodies."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular complete example artifact")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Example artifact exceeds its declared byte allowance")
    return kind.from_json(data)


def write_contract(root: Path, relative: str, value) -> None:
    """Publish complete canonical bytes through the existing immutable artifact writer.

    Parameters
    ----------
    root : pathlib.Path
        Trusted local artifact root; created by the existing writer as needed.
    relative : str
        Explicit bounded local artifact key.
    value : versioned contract
        Complete finite serializable evidence or result-free metadata.

    Raises
    ------
    ValueError, OSError
        Conflicting immutable bytes, invalid path/contract or publication failure.
        Does not replace existing evidence or claim cross-ledger atomicity."""
    content = value.to_json() + "\n"
    if len(content.encode("utf-8")) > 16 * 1024**2:
        raise ValueError("Example complete artifact exceeds sixteen MiB")
    ArtifactWriter(root).write_immutable(root / relative, content)


def frozen_declarations(root: Path) -> tuple[FrozenExample, bool]:
    """Persist or reopen the exact original experiment before outcome access.

    Parameters
    ----------
    root : pathlib.Path
        Trusted local output root; this operation reads or atomically creates its
        result-free frozen-example.json and captures installed/code identities.

    Returns
    -------
    tuple of FrozenExample and bool
        Original complete design, selection, fallback plan and compatibility; the
        boolean is true on reopening, requiring verification/import semantics.

    Raises
    ------
    ValueError, OSError
        Invalid/future metadata, current source or BASE-law drift, changed copied
        code/runtime, exhausted capacities or immutable publication failure.

    Notes
    -----
    Never updates selected_at or replaces the hypothesis family on retry. No
    source outcomes, learner callbacks or experiment store are opened here."""
    path = root / "frozen-example.json"
    if path.exists():
        frozen = read_contract(path, FrozenExample)
        current = {mode: canonical_json(_candidate(mode).model_dump(mode="json")) for mode in MODEL.MODES}
        old = {
            row.candidate.candidate_id: canonical_json(row.model_dump(mode="json")) for row in frozen.design.candidates
        }
        if current != old or _snapshot().model_dump_json() != frozen.design.snapshot.model_dump_json():
            raise ValueError(
                "Current candidate laws or source interpretation differ from the original frozen experiment"
            )
        if compatibility().fingerprint != frozen.environment.fingerprint:
            raise ValueError("Current runtime or copied example code differs from the original frozen execution")
        return frozen, True
    design, selection, plan = declarations()
    frozen = FrozenExample(schema_version=1, design=design, selection=selection, plan=plan, environment=compatibility())
    write_contract(root, "frozen-example.json", frozen)
    return frozen, False


def _snapshot() -> SourceSnapshot:
    """Bind the complete literal source and its fixed interpretation without clocks."""
    data = fingerprint(
        {
            "schema_version": 1,
            "table_seconds": [[list(pair) for pair in world] for world in TABLE],
            "worlds": list(WORLDS),
            "modeled_plan_requirements_seconds": [2.0, 6.0],
            "budget": "8 if round1 or odd world, else4",
        }
    )
    return SourceSnapshot(
        reference=MODEL.ref("public-ci-table-v1"),
        data_fingerprint=data,
        interpretation=MODEL.definition(
            "ci-table-interpretation",
            "world, step, fast/full modeled seconds; full information supplied after each decision.",
        ),
        captured_at=None,
    )


def _candidate(mode: str) -> CandidatePlan:
    """Freeze every declared stage without inspecting a protected outcome."""
    config = MODEL.configuration(mode)
    charge_rate = 0.0 if mode == "accounting_off" else 4.0
    base = FrozenCandidate.from_configuration(
        mode,
        {
            "method": config.model_dump(mode="json"),
            "prediction": "expanding-complete-past-mean" if mode == "online" else "frozen-four-seconds",
            "domain_adapter": MODEL.ADAPTER.model_dump(mode="json"),
            "intervention": MODEL.INTERVENTION.model_dump(mode="json"),
            "modeled_seconds_per_full_coefficient_L1": charge_rate,
            "execution_diagnostic": "Retain infeasible unprojected proposal as diagnostic only; never deploy it.",
            "calendar": (
                "Capture one common UTC tick after previous complete source panel application; three decisions."
            ),
            "frozen_descriptive_stage_pairs": [list(pair) for pair in STAGE_PAIRS],
        },
    )
    laws = {
        "implementation": "Actual shipped " + config.profile,
        "initialization": (
            "Fresh independent complete owners; coefficients one half per expert; empty predictor history."
        ),
        "update_law": config.to_json(),
        "refit_law": "Past mature complete-panel expanding mean."
        if mode == "online"
        else "Frozen supplied forecasts 4/4; no prediction updates used.",
        "transform_law": "Every synthetic expert duration is in [0,8] seconds; loss=duration/8, no fit or clipping.",
        "eligibility": (
            "Use only earlier received/matured panels; all candidates share each round's explicit clock/input panel."
        ),
        "accounting": (
            f"Own-path model: {charge_rate} seconds per full coefficient L1; "
            "not lambda, physical cost, financial turnover or a paid fee."
        ),
        "stopping": "All three decisions in each of four declared worlds; retain every mode and terminal outcome.",
    }
    return CandidatePlan(
        candidate=base,
        role="fixed_baseline" if mode == "fixed" else "candidate",
        strategy=StrategyBindings(
            **{name: MODEL.definition("ci-" + mode + "-" + name, law) for name, law in laws.items()},
            initial_state_fingerprint=None,
            update_scope="within_unit",
        ),
    )


def declarations() -> tuple[ExperimentDesign, SelectionManifest, ComparisonPlan]:
    """Freeze a nine-mode, four-world family before the real audited loader.

    Returns
    -------
    tuple of ExperimentDesign, SelectionManifest and ComparisonPlan
        Complete BASE laws, all contrasts, exact baseline conformance/terminal
        mapping and development-only declared lambda. Source is already exposed.
    """
    source_plan = TemporalPlan(
        train_end=datetime(2018, 1, 1, tzinfo=UTC),
        validation_end=datetime(2018, 12, 31, tzinfo=UTC),
        test_end=datetime(2020, 12, 31, tzinfo=UTC),
    )
    comparison_plan = TemporalPlan.model_validate(
        source_plan.model_copy(update={"test_end": datetime.now(UTC) + timedelta(hours=2)})
    )
    primary = MODEL.metric()
    limits = ProtocolLimits(
        schema_version=1,
        max_candidates=16,
        max_hypotheses=16,
        max_cells=64,
        max_aggregate_inputs=64,
        max_physical_attempts=128,
        max_output_references=128,
        max_configuration_bytes=16 * 1024,
        max_raw_record_bytes=4 * 1024,
        max_design_bytes=512 * 1024,
        max_selection_bytes=512 * 1024,
        max_freeze_bytes=512 * 1024,
        max_metadata_bytes=1024**2,
        max_envelope_bytes=4 * 1024**2,
        admission_margin_bytes=1024**2,
        max_graph_items=262144,
    )
    candidates = tuple(_candidate(mode) for mode in MODEL.MODES)
    snapshot = _snapshot()
    data = snapshot.data_fingerprint
    design = ExperimentDesign(
        schema_version=1,
        reference=MODEL.ref("public-prequential-ci-comparison"),
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        scope=SourceScope(
            schema_version=1,
            reference=MODEL.ref("public.authored-prequential-ci-table"),
            upstream=MODEL.definition(
                "authored-ci-worlds", "The complete four-world public table in ci.py; never empirical CI evidence."
            ),
            partition_definition=MODEL.definition(
                "original-ci-membership",
                canonical_json(source_plan.model_dump(mode="json")),
            ),
            prehistory="known_exposed",
        ),
        snapshot=snapshot,
        source_plan=source_plan,
        comparison_plan=comparison_plan,
        members=tuple(
            SourceMember(
                member_id=world,
                task=MODEL.DOMAIN_TASK,
                path=MODEL.ref("independent-authored-" + world),
                unit_id="authored-unit-" + world,
                metadata=TemporalObservation(
                    observation_id="source-" + world,
                    group_id="source-group-" + world,
                    event_at=datetime(2020, 1, index + 1, tzinfo=UTC),
                    available_at=datetime(2020, 1, index + 1, 1, tzinfo=UTC),
                ),
            )
            for index, world in enumerate(WORLDS)
        ),
        candidates=candidates,
        seeds=(0,),
        metrics=MODEL.metrics(),
        primary_metric=primary.name,
        hypotheses=tuple(
            ComparisonHypothesis(hypothesis_id=mode + "-versus-fixed", candidate_id=mode, baseline_id="fixed")
            for mode in MODEL.MODES
            if mode != "fixed"
        ),
        inference=InferenceRequest(
            method="paired_bounded_mean_hoeffding_holm_v1",
            dependence="independent",
            assignment="complete_paired_units",
            analysis="fixed_final",
            practical_effect=Threshold(value=0.1, semantics=primary.quantity.spec.semantics),
            minimum_units=20,
            independent_unit="authored-whole-world",
            independence_basis=MODEL.definition(
                "authored-unit-declaration",
                "Four authored distinct worlds are mechanics fixtures, not empirical independent observations; "
                "no extra n from rounds/seeds.",
            ),
            bounds_basis=MODEL.definition(
                "terminal-model-bounds",
                "Expert durations 0..8; full coefficient L1 <=2 at rate0 or4 gives terminal net0..16 modeled seconds.",
            ),
        ),
        budget=BudgetPlan(
            maximum_trials=0,
            maximum_physical_attempts=128,
            maximum_output_references=128,
            stopping=MODEL.definition(
                "finite-ci-comparison-budget",
                "Exactly nine frozen modes times four worlds; "
                "grouped Python simulation and separate direct reference work; "
                "no physical cost measurements.",
            ),
        ),
        limits=limits,
    )
    baseline = design.candidate("fixed").candidate
    keys = tuple(execution_key("fixed", world, ordinal) for world in WORLDS for ordinal in range(3))
    conformance = ReferenceConformancePlan(
        schema_version=1,
        reference=MODEL.ref("full-fixed-ci-conformance"),
        lineage=MODEL.lineage(),
        adapter=MODEL.ADAPTER,
        reference_recipe=REFERENCE_RECIPE,
        baseline_candidate_id="fixed",
        base_configuration_fingerprint=baseline.configuration_fingerprint,
        data_fingerprint=data,
        opportunities=keys,
        metrics=MODEL.metrics(),
    )
    plan = ComparisonPlan(
        schema_version=1,
        reference=MODEL.ref("ci-comparison-fallback-rule"),
        lineage=MODEL.lineage(),
        design_fingerprint=design.fingerprint,
        conformance_plan=conformance,
        baseline=baseline,
        required_hypotheses=tuple(row.hypothesis_id for row in design.hypotheses),
        cell_result_opportunities=tuple(execution_key("fixed", world, 2) for world in WORLDS),
        fallback="baseline",
    )
    selection = SelectionManifest(
        schema_version=1,
        design_fingerprint=design.fingerprint,
        ledger=AttemptLedger(schema_version=1, stage="development"),
        selected_candidate_ids=MODEL.MODES,
        rule=plan.binding,
        actual_selected_at=datetime.now(UTC),
    )
    return design, selection, plan


def execution_key(mode: str, world: str, ordinal: int) -> ExecutionKey:
    """Bind one independent comparison cell to a separate CI-domain opportunity.

    Parameters
    ----------
    mode : str
        Frozen base candidate identifier.
    world : str
        Authored independent-unit member identifier, distinct from a seed/round.
    ordinal : int
        Zero-based logical decision index in this finite example.

    Returns
    -------
    ExecutionKey
        Exact versioned domain task, member/seed and modeled build identity.
        Construction is pure and grants no source or deployment permission."""
    return ExecutionKey(
        cell=CellIdentity(candidate_id=mode, member_id=world, seed=0),
        task=MODEL.DOMAIN_TASK,
        decision_id="modeled-build-" + str(ordinal),
    )


def compatibility() -> ExecutionCompatibility:
    """Capture actual installed base and all six explicitly copied example sources."""
    sources = tuple(
        DefinitionBinding(
            reference=MODEL.ref("public-prequential-" + name),
            definition_fingerprint="sha256:" + sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest(),
        )
        for name in ("__init__", "ci", "model", "profile", "session", "reference")
    )
    return ExecutionCompatibility(
        schema_version=1,
        runtime=SelectedInstalledRuntime.capture(extras=()),
        owner_implementations=(*sources, MODEL.ADAPTER),
    )
