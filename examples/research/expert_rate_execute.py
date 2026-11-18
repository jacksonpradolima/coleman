"""Actual shipped learners and simple host controls for the frozen entry study."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from fractions import Fraction
from importlib import resources
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, Self, cast

from pydantic import Field, model_validator

# Selector keys reject bool and arbitrary callback-bearing subclasses.
# pylint: disable=unidiomatic-typecheck
from coleman.application.resource_usage import output_exceeds, peak_rss_mib
from coleman.artifacts import ArtifactWriter
from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec, validate_simplex
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import (
    LearningLimits,
    LearningUpdate,
    ReferenceConfig,
    ReferencePayload,
    ReferenceProfile,
    read_loss_panel,
)
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, InitialPrior
from coleman.results.domain import Measurement
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.components import ComponentCodec
from coleman.state.contracts import AlgorithmState, StateBinding
from examples.research.expert_rate_records import (
    METRICS,
    REFERENCE_PROFILES,
    Case,
    CaseRun,
    Checkpoint,
    Family,
    FamilyRun,
    Record,
    Setting,
    Step,
    StudyInput,
    Vector,
    admit_protocol,
)

MEMBERS = tuple(Reference(id="ci-recipe-" + suffix, version="1") for suffix in ("a", "b", "c"))
LOSS = ScalarSpec(
    dimension=Reference(id="expert-rate-entry-loss", version="1"),
    semantics=UnitSemantics(unit="1", meaning="authored-relative-ci-recipe-penalty-v1"),
    role="loss",
    minimum=0.0,
    maximum=1.0,
)
TRANSFORM = DefinitionBinding(
    reference=Reference(id="entry-rational-to-binary64", version="1"),
    definition_fingerprint=fingerprint(
        {"source": "locally-frozen-rational-loss", "conversion": "once-float-Fraction", "minimum": 0, "maximum": 1}
    ),
)
LINEAGE = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


class ResourceLimitError(RuntimeError):
    """Preserve a known declared observation cap, distinct from a software error."""


class BaselineState(Record):
    """Complete minimal fixed or bounded periodic state, without a new learner claim."""

    binding: StateBinding
    setting: Setting
    update_count: Annotated[int, Field(strict=True, ge=0, le=64)]
    coefficients: Vector
    history: Annotated[tuple[Vector, ...], Field(max_length=64)]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if self.setting.family not in ("periodic", "fixed-uniform"):
            raise ValueError("Host baseline state requires its actual simple-control family")
        validate_simplex(self.coefficients)
        if len(self.history) != (self.update_count if self.setting.family == "periodic" else 0):
            raise ValueError("Periodic history and fixed-state absence must match the actual update clock")
        if any(not 0 <= x <= 1 for row in self.history for x in row):
            raise ValueError("Host history requires original bounded observed losses")
        if self.coefficients != baseline_weights(self.history, self.setting.window):
            raise ValueError("Host coefficients differ from the exact declared last refit")
        return self


def baseline_weights(history: tuple[Vector, ...], window: int | None) -> Vector:
    """Compute the actual last eight-panel refit from retained eligible history.

    Parameters
    ----------
    history : tuple
        Original once-converted loss panels in order, at most64.
    window : int or None
        Frozen trailing window; null selects the untuned fixed-uniform control.

    Returns
    -------
    tuple of float
        Equal mass on exact trailing minimizers, held until the next refit.

    Notes
    -----
    Uses Fraction over stored binary64 values; no future row or oracle trace
    enters the actual host control. This is distinct from the independent
    oracle's full-prefix evaluation at each opportunity.
    """
    stop = len(history) - len(history) % 8
    if window is None or stop == 0:
        return 1 / 3, 1 / 3, 1 / 3
    totals = [sum((Fraction(row[i]) for row in history[max(0, stop - window) : stop]), Fraction()) for i in range(3)]
    minimum = min(totals)
    count = totals.count(minimum)
    return cast(Vector, tuple(1 / count if value == minimum else 0.0 for value in totals))


class HostBaseline:
    """Execute the declared fixed/periodic control, retaining full owned state.

    Parameters
    ----------
    setting : Setting
        Exact frozen fixed-uniform or periodic host-baseline definition.
    binding : StateBinding
        Original owner/source/configuration interpretation, never a paid cost.
    """

    def __init__(self, setting: Setting, binding: StateBinding):
        self._state = BaselineState(
            schema_version=1,
            binding=binding,
            setting=setting,
            update_count=0,
            coefficients=(1 / 3, 1 / 3, 1 / 3),
            history=(),
        )

    @property
    def weights(self) -> tuple[Weight, ...]:
        """Return actual coefficient fractions, never action propensities."""
        return tuple(
            Weight(member=member, value=value) for member, value in zip(MEMBERS, self._state.coefficients, strict=True)
        )

    @property
    def update_count(self) -> int:
        """Return accepted complete panels U, not independent observations."""
        return self._state.update_count

    def observe(self, panel: FullPanel) -> None:
        """Admit complete original feedback and advance the actual simple law.

        Parameters
        ----------
        panel : FullPanel
            All original members with observed compatible synthetic losses.

        Raises
        ------
        ValueError
            Incompatible feedback or the frozen64-panel capacity is exhausted.
        """
        losses = cast(Vector, read_loss_panel(panel, members=MEMBERS, loss_spec=LOSS, transform=TRANSFORM))
        if self.update_count >= 64:
            raise ValueError("The fixed study control has exhausted its original panel capacity")
        history = (*self._state.history, losses) if self._state.setting.family == "periodic" else ()
        self._state = BaselineState(
            schema_version=1,
            binding=self._state.binding,
            setting=self._state.setting,
            update_count=self.update_count + 1,
            coefficients=baseline_weights(history, self._state.setting.window),
            history=history,
        )

    def export_state(self) -> BaselineState:
        """Return complete validated state; no artifact or ACK is published."""
        return BaselineState.model_validate(self._state)

    def restore_state(self, state: BaselineState) -> None:
        """Restore complete context-matched state atomically, without replaying work.

        Parameters
        ----------
        state : BaselineState
            Every original sufficient statistic and held coefficient.

        Raises
        ------
        ValueError
            The selected owner, setting or complete state is incompatible.
        """
        checked = BaselineState.model_validate(state)
        if checked.binding != self._state.binding or checked.setting != self._state.setting:
            raise ValueError("Host baseline restore requires its exact original owner and law")
        self._state = checked


type Owner = Hedge | AnytimeHedge | FixedShare | HostBaseline


def _new_owner(protocol: StudyInput, case: Case, setting: Setting) -> Owner:
    """Construct one fresh real owner at U=0, with no cross-case state transfer.

    Parameters
    ----------
    protocol, case, setting : StudyInput, Case, Setting
        Exact original locally frozen source and frozen family/configuration.

    Returns
    -------
    Hedge or AnytimeHedge or FixedShare or HostBaseline
        Actual qualified reference or the separately labeled simple control.
    """
    identity = fingerprint([protocol.protocol_content, case.case_id, setting.model_dump(mode="json")])
    owner = Reference(id="entry-owner-" + identity[7:], version="1")
    if setting.family in ("periodic", "fixed-uniform"):
        binding = StateBinding(
            owner=owner,
            implementation=DefinitionBinding(
                reference=Reference(id="entry-host-" + setting.family, version="1"), definition_fingerprint=identity
            ),
            state_codec=Reference(id="entry-host-state", version="1"),
            configuration_fingerprint=identity,
        )
        return HostBaseline(setting, binding)
    profile = REFERENCE_PROFILES[setting.family]
    prior = InitialPrior(
        schema_version=1,
        reference=Reference(id="entry-uniform-prior", version="1"),
        lineage=LINEAGE,
        weights=tuple(Weight(member=m, value=1 / 3) for m in MEMBERS),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    config = ReferenceConfig(
        schema_version=1,
        reference=Reference(id="entry-config-" + identity[7:], version="1"),
        lineage=LINEAGE,
        profile=cast(ReferenceProfile, profile),
        prior=prior,
        loss_spec=LOSS,
        transform=TRANSFORM,
        eta=setting.eta,
        alpha=setting.alpha,
        anytime_multiplier=setting.multiplier,
        limits=LearningLimits(maximum_members=3, maximum_updates=64),
    )
    constructor = {"constant": Hedge, "anytime": AnytimeHedge, "fixed-share": FixedShare}[setting.family]
    return constructor(config, owner=owner)


def _select(protocol: StudyInput, case_id: str, family: Family, slot: int) -> tuple[Case, Setting]:
    """Resolve only original logical keys, never caller-supplied losses/rates."""
    if type(case_id) is not str or type(slot) is not int or type(family) is not str:
        raise ValueError("Case and setting selectors require their strict original logical keys")
    case = next((c for c in protocol.cases if c.case_id == case_id), None)
    setting = next((s for s in protocol.settings if s.family == family and s.slot == slot), None)
    if case is None or setting is None:
        raise ValueError("Unknown original study case or configuration")
    return case, setting


def new_owner(protocol: StudyInput, case_id: str, family: Family, slot: int) -> Owner:
    """Create a fresh actual owner only after full original source admission.

    Parameters
    ----------
    protocol : StudyInput
        Original complete typed input; every field is rechecked, not its label.
    case_id : str
        Exact original case identity, not a caller-provided loss array.
    family : str
        Original family identity.
    slot : int
        Strict original setting index, selecting parameters from admitted source.

    Returns
    -------
    Hedge or AnytimeHedge or FixedShare or HostBaseline
        A fresh U=0 owner under the exact original source and frozen treatment.

    Raises
    ------
    ValueError, TypeError
        Changed source/view/type/version or unknown case/setting key.
    """
    original = admit_protocol(protocol)
    case, setting = _select(original, case_id, family, slot)
    return _new_owner(original, case, setting)


def loss_panel(losses: Vector) -> FullPanel:
    """Bind an actual complete observed synthetic row to its fixed units.

    Parameters
    ----------
    losses : tuple of float
        Three once-converted original bounded losses, not inferred alternatives.

    Returns
    -------
    FullPanel
        Explicit observed synthetic values in the original member order.
    """
    rows = tuple(
        FeedbackRow(
            member=member,
            values=(
                OutcomeValue(
                    spec=LOSS, value=value, status="observed", origin="synthetic", transform=TRANSFORM.reference
                ),
            ),
        )
        for member, value in zip(MEMBERS, losses, strict=True)
    )
    return FullPanel(rows=rows)


def metric(name: str, value: Fraction | float | None, *, unit: str = "1") -> Measurement:
    """Keep signed/zero observations and unavailable measurements separate.

    Parameters
    ----------
    name : str
        Nonreserved explicit study metric name.
    value : Fraction or float or None
        Observed original quantity; null explicitly means unavailable.
    unit : str, optional
        Declared unit1, s or MiB, with no implicit conversion.

    Returns
    -------
    Measurement
        Existing versioned measurement semantics, never a paid-cost projection.
    """
    return Measurement(
        name=name,
        value=None if value is None else float(value),
        unit=unit,
        meaning=name + "-actual-binary64-source-v1",
        direction="none" if name == "coefficient_L1" else "minimize",
        status="unavailable" if value is None else "observed",
    )


def _checkpoint(owner: Owner) -> Checkpoint:
    state = owner.export_state()
    encoded = ComponentCodec(state.binding, type(state)).encode(state)
    content = encoded.content.decode("utf-8")
    return Checkpoint(
        position=owner.update_count, content=content, digest="sha256:" + hashlib.sha256(content.encode()).hexdigest()
    )


def restore_checkpoint(owner: Owner, checkpoint: Checkpoint) -> None:
    """Validate complete original checkpoint bytes before restoring an owner.

    Parameters
    ----------
    owner : Hedge or AnytimeHedge or FixedShare or HostBaseline
        Same original family/configuration/source/case owner.
    checkpoint : Checkpoint
        Full selected-codec bytes and exact accepted-panel position.

    Raises
    ------
    ValueError, TypeError
        Wrong version, type, original binding, clock, bytes or complete state.

    Notes
    -----
    Reading performs no mathematical update or publication. The actual owner's
    public atomic restore performs its own full configuration/binding checks.
    This direct read is neither an execution ACK nor an external-effect retry.
    """
    checked = Checkpoint.model_validate(checkpoint)
    if isinstance(owner, HostBaseline):
        state = BaselineState.from_json(checked.content)
        position = state.update_count
    else:
        state = AlgorithmState[ReferencePayload].from_json(checked.content)
        position = state.payload.update_count
    if position != checked.position or state.to_json() != checked.content:
        raise ValueError("Checkpoint bytes and original accepted update clock disagree")
    if isinstance(owner, HostBaseline):
        owner.restore_state(BaselineState.from_json(checked.content))
    else:
        owner.restore_state(AlgorithmState[ReferencePayload].from_json(checked.content))


def _observe_step(owner: Owner, index: int, row: Vector) -> tuple[Step | None, Checkpoint | None, str | None]:
    """Observe once, checking atomic refusal and complete selected-state evidence."""
    before = cast(Vector, tuple(w.value for w in owner.weights))
    previous = owner.export_state().to_json()
    outcome = owner.observe(loss_panel(row))
    if isinstance(outcome, LearningUpdate) and outcome.status == "numerical_failure":
        if owner.export_state().to_json() != previous:
            raise ValueError("A numerical refusal changed complete original learner state")
        return None, None, outcome.failure.code if outcome.failure else "invalid-refusal"
    checkpoint = _checkpoint(owner)
    return (
        Step(
            d=index,
            accepted_u=owner.update_count,
            before=before,
            after=cast(Vector, tuple(w.value for w in owner.weights)),
            state_digest=checkpoint.digest,
        ),
        checkpoint,
        None,
    )


def _terminal_metrics(case: Case, trace: tuple[Step, ...], completed: bool) -> tuple[Measurement, ...]:
    """Report actual before-update loss, explicit retrospective diagnostics and L1."""
    if not completed:
        return tuple(metric(name, None) for name in METRICS)
    before = tuple(s.before for s in trace)
    total = _actual_score(before, case.panels)
    fixed, switched = _actual_comparators(case.panels)
    movement = sum(
        (
            sum((abs(Fraction(a) - Fraction(b)) for a, b in zip(s.after, s.before, strict=True)), Fraction())
            for s in trace
        ),
        Fraction(),
    )
    values = [
        total,
        total / len(case.panels),
        _actual_score(before[case.informative_start :], case.panels[case.informative_start :])
        / (len(case.panels) - case.informative_start),
        total - fixed,
        total - switched,
        movement,
    ]
    return tuple(metric(n, v) for n, v in zip(METRICS, values, strict=True))


def _actual_score(weights, rows) -> Fraction:
    """Accumulate actual original products; the oracle uses a separate expression."""
    total = Fraction()
    for coefficients, losses in zip(weights, rows, strict=True):
        for coefficient, loss in zip(coefficients, losses, strict=True):
            total += Fraction(coefficient) * Fraction(loss)
    return total


def _actual_comparators(rows) -> tuple[Fraction, Fraction]:
    """Report finite retrospective costs with a streaming zero/one-switch DP.

    This computes diagnostics only, never a deployed path. The independent
    oracle separately enumerates original prefixes and every split/pair.
    """
    fixed = [Fraction(), Fraction(), Fraction()]
    switched = fixed.copy()
    for row in rows:
        best_prefix = min(fixed)
        switched = [min(prior, best_prefix) + Fraction(loss) for prior, loss in zip(switched, row, strict=True)]
        fixed = [prior + Fraction(loss) for prior, loss in zip(fixed, row, strict=True)]
    return min(fixed), min(switched)


def _continued_owner(protocol, case, setting, checkpoint):
    """Restore the actual full midpoint state into a fresh matching owner."""
    restored = _new_owner(protocol, case, setting)
    restore_checkpoint(restored, checkpoint)
    return restored


def _run_case(protocol: StudyInput, case: Case, setting: Setting, guard: Callable[[], None]) -> CaseRun:
    """Run actual chronological mathematics under one declared caller guard.

    Parameters
    ----------
    protocol, case, setting : StudyInput, Case, Setting
        Exact complete original source and trial law.
    guard : callable
        Existing caller-owned finite deadline/RSS/output check before and after
        each complete admitted step. Only ResourceLimitError is a limited outcome.

    Returns
    -------
    CaseRun
        Complete original trace or honest limited/numerical-refusal prefix.

    Raises
    ------
    Exception
        Unexpected validation/software errors propagate; they cannot become a
        completed or ordinary resource-limited experiment.
    """
    owner = _new_owner(protocol, case, setting)
    trace, snapshots = [], [_checkpoint(owner)]
    status, reason, attempted = "completed", None, 0
    try:
        guard()
        for row in case.panels:
            guard()
            attempted += 1
            step, checkpoint, failure = _observe_step(owner, owner.update_count, row)
            if failure is not None:
                status, reason = "numerical_failure", failure
                break
            assert step is not None and checkpoint is not None
            trace.append(step)
            if owner.update_count == len(case.panels) // 2:
                snapshots.append(checkpoint)
                owner = _continued_owner(protocol, case, setting, checkpoint)
            guard()
    except ResourceLimitError as error:
        status, reason = "resource_limited", str(error)
    if owner.update_count != snapshots[-1].position:
        snapshots.append(_checkpoint(owner))
    return CaseRun(
        schema_version=1,
        case_id=case.case_id,
        setting=setting,
        status=status,
        reason=reason,
        configuration_fingerprint=fingerprint(setting.model_dump(mode="json")),
        horizon=len(case.panels),
        attempted=attempted,
        accepted_u=owner.update_count,
        trace=tuple(trace),
        checkpoints=tuple(snapshots),
        metrics=_terminal_metrics(case, tuple(trace), status == "completed"),
    )


def run_case(protocol: StudyInput, case_id: str, family: Family, slot: int, guard: Callable[[], None]) -> CaseRun:
    """Execute one real case selected by original keys after full source admission.

    Parameters
    ----------
    protocol : StudyInput
        Complete original typed source; copied/constructed substitutions refuse.
    case_id, family : str
        Exact original source and family identities.
    slot : int
        Strict original frozen-setting index.
    guard : callable
        Caller-owned actual deadline/resource check; unexpected errors propagate.

    Returns
    -------
    CaseRun
        Actual full trace or an honest known-limit/numerical-refusal prefix.

    Raises
    ------
    ValueError, TypeError
        Invalid original data or unknown selectors before any owner is created.
    """
    original = admit_protocol(protocol)
    case, setting = _select(original, case_id, family, slot)
    return _run_case(original, case, setting, guard)


def _actual_runtime(protocol: StudyInput):
    """Read actual source/runtime, rejecting a substituted installed reference."""
    runtime = InstalledRuntime.capture()
    driver = SelectedInstalledRuntime.capture(extras=("cli",))
    sdk = resources.files("coleman")
    for source, expected in protocol.source_files:
        if (
            source.startswith("coleman/")
            and hashlib.sha256(sdk.joinpath(source[8:]).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Actual installed source differs from the frozen reference-method assessment")

    return runtime, driver


def _unstarted_case(case: Case, setting: Setting) -> CaseRun:
    """Retain original unattempted work with unavailable accepted state/scores."""
    return CaseRun(
        schema_version=1,
        case_id=case.case_id,
        setting=setting,
        status="not_started",
        configuration_fingerprint=fingerprint(setting.model_dump(mode="json")),
        reason="earlier-family-resource-limit",
        horizon=len(case.panels),
        attempted=0,
        accepted_u=None,
        trace=(),
        checkpoints=(),
        metrics=tuple(metric(n, None) for n in METRICS),
    )


class _FamilyContext(NamedTuple):
    """Keep one process's admitted inputs and measurement origins together."""

    protocol: StudyInput
    coordinate: tuple[Family, int]
    output: Path
    started: float
    runtime: tuple[InstalledRuntime, SelectedInstalledRuntime]


def _family_disposition(
    cases: tuple[CaseRun, ...], limited: str | None
) -> tuple[Literal["completed", "resource_limited", "numerical_failure"], str | None]:
    """Distinguish an explicit resource cap, numerical refusal and full completion."""
    if limited is not None:
        return "resource_limited", limited
    if any(c.status != "completed" for c in cases):
        return "numerical_failure", "one-or-more-original-cases-have-a-numerical-refusal"
    return "completed", None


def run_family(protocol: StudyInput, family: Family, repetition: int, output: Path) -> FamilyRun:
    """Execute one actual bounded family process and preserve each original case.

    Parameters
    ----------
    protocol : StudyInput
        Original hash-pinned source and complete frozen17-setting catalogue.
    family : str
        One of the five original families; no adaptive method is selected.
    repetition : int
        Physical timing coordinate0..2, never an independent observation.
    output : pathlib.Path
        Trusted unique process directory; immutable case files survive a timeout.

    Returns
    -------
    FamilyRun
        Complete full family grid or an explicit resource-limited grid.

    Raises
    ------
    Exception
        Unexpected math, source, validation or publication errors propagate.

    Notes
    -----
    Original180seconds,512MiB and64MiB aggregate output caps are retained.
    The parent enforces the hard process deadline including imports. This
    cooperative inner guard is not a sandbox or an external-effect protocol.
    """
    started = time.monotonic()
    protocol = admit_protocol(protocol)
    if type(repetition) is not int or not 0 <= repetition <= 2 or type(family) is not str:
        raise ValueError("Family execution requires its strict original physical coordinate")
    if family not in {s.family for s in protocol.settings}:
        raise ValueError("Unknown family cannot spend a formal process workload")
    runtime = _actual_runtime(protocol)

    def guard() -> None:
        if time.monotonic() - started >= 180:
            raise ResourceLimitError("family-observation-deadline-180s")
        memory = peak_rss_mib()
        if memory is None or memory > 512:
            raise ResourceLimitError("family-peak-rss-unavailable-or-above512MiB")

    cases = []
    limited = None
    for setting in (s for s in protocol.settings if s.family == family):
        for case in protocol.cases:
            if limited is None:
                value = _run_case(protocol, case, setting, guard)
                if value.status == "resource_limited":
                    limited = value.reason
            else:
                value = _unstarted_case(case, setting)
            content = value.to_json()
            if output_exceeds(64 * 1024**2 - len(content.encode()), root=output.parent, maximum_entries=8192):
                raise ResourceLimitError("study-retained-output-above64MiB")
            ArtifactWriter(output).write_immutable(output / f"setting-{setting.slot}-{case.case_id}.json", content)
            cases.append(value)
    return _finish_family(
        _FamilyContext(protocol, (family, repetition), output, started, runtime), tuple(cases), limited
    )


def _finish_family(context: _FamilyContext, cases: tuple[CaseRun, ...], limited: str | None) -> FamilyRun:
    """Retain actual grid/bytes/inner resources before whole-process measurement."""
    status, reason = _family_disposition(cases, limited)
    result = FamilyRun(
        schema_version=1,
        protocol_content=context.protocol.protocol_content,
        family=context.coordinate[0],
        repetition=context.coordinate[1],
        status=status,
        reason=reason,
        runtime=context.runtime[0],
        driver_runtime=context.runtime[1],
        source_files=tuple(
            (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(Path(__file__).parent.glob("expert_rate_*.py"))
        ),
        protocol_input_bytes=Path(__file__).with_name("expert-rate-entry-protocol-v2.json").stat().st_size,
        typed_input_bytes=len(context.protocol.to_json().encode()),
        logical_checkpoint_bytes=sum(
            len(checkpoint.content.encode()) for case_result in cases for checkpoint in case_result.checkpoints
        ),
        logical_case_bytes=sum(
            (context.output / f"setting-{case_result.setting.slot}-{case_result.case_id}.json").stat().st_size
            for case_result in cases
        ),
        attempted=sum(c.attempted for c in cases),
        cases=tuple(cases),
        resources=(
            metric("process_wall", time.monotonic() - context.started, unit="s"),
            metric("process_cpu", time.process_time(), unit="s"),
            metric("peak_rss", peak_rss_mib(), unit="MiB"),
        ),
    )
    result.require_grid(context.protocol)
    return result
