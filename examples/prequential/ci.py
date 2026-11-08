"""Compare real expert learners through a bounded, explicitly modeled CI workflow.

Copy this explicit six-file example package together and run ci.py with the
qualified wheel installed, including from outside the checkout under python -I.
Modeled seconds are neither measured CI occupancy nor paid physical costs."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.projection import project_frozen
from coleman.evaluation.design.reports import ProtocolReport, ReportLocator
from coleman.evaluation.design.selection import AttemptLedger, CellIdentity, PhysicalAttempt
from coleman.evaluation.prequential.comparison import ComparisonDisposition, assess_comparison
from coleman.evaluation.prequential.conformance import (
    ReferenceConformancePlan,
    ReferenceConformanceReport,
    check_reference_conformance,
)
from coleman.evaluation.prequential.contracts import (
    ConformanceTrace,
)
from coleman.evaluation.prequential.execution import PrequentialStep
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import (
    CommitRequest,
)

JSON_SUFFIX = ".json"

__all__ = (
    "MODEL",
    "PROFILE",
    "CommitRequest",
    "LocalStateStore",
    "PrequentialStep",
    "ReferenceConformancePlan",
    "FrozenExample",
    "ObservationInput",
    "SourceTick",
    "SourceTape",
    "CachedExample",
    "EmissionTick",
    "Invocation",
    "TABLE",
    "WORLDS",
    "STAGE_PAIRS",
    "REFERENCE_RECIPE",
    "_primary",
    "_timing",
    "_read",
    "_write",
    "_key",
    "frozen_declarations",
    "declarations",
    "compatibility",
    "reference_from_originals",
    "_trajectory_diagnostics",
    "ExampleResult",
    "run",
    "assess_existing",
    "main",
)


def _package():
    """Load only this explicit sibling package, including under isolated Python."""
    name = "coleman_public_prequential_example"
    path = Path(__file__).with_name("__init__.py")
    if name in sys.modules:
        existing = sys.modules[name]
        origin = getattr(existing, "__file__", None)
        if origin is None or Path(origin).resolve() != path.resolve():
            raise ValueError("A different public CI example package already owns this process name")
        return existing
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    if spec is None or spec.loader is None:
        raise RuntimeError("Copy every declared public prequential example file together")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_package()
if TYPE_CHECKING:
    from . import model, profile, reference
    from .profile import (
        REFERENCE_RECIPE,
        STAGE_PAIRS,
        TABLE,
        WORLDS,
        CachedExample,
        EmissionTick,
        FrozenExample,
        Invocation,
        ObservationInput,
        SourceTape,
        SourceTick,
        compatibility,
        declarations,
        frozen_declarations,
    )
    from .profile import (
        decision_timing as _timing,
    )
    from .profile import (
        execution_key as _key,
    )
    from .profile import (
        primary_component as _primary,
    )
    from .profile import (
        read_contract as _read,
    )
    from .profile import (
        write_contract as _write,
    )
    from .reference import reference_from_originals
    from .session import Session
else:
    model = importlib.import_module("coleman_public_prequential_example.model")
    profile = importlib.import_module("coleman_public_prequential_example.profile")
    reference = importlib.import_module("coleman_public_prequential_example.reference")
    Session = importlib.import_module("coleman_public_prequential_example.session").Session
    FrozenExample = profile.FrozenExample
    ObservationInput = profile.ObservationInput
    SourceTick = profile.SourceTick
    SourceTape = profile.SourceTape
    CachedExample = profile.CachedExample
    EmissionTick = profile.EmissionTick
    Invocation = profile.Invocation
    TABLE = profile.TABLE
    WORLDS = profile.WORLDS
    STAGE_PAIRS = profile.STAGE_PAIRS
    REFERENCE_RECIPE = profile.REFERENCE_RECIPE
    _primary = profile.primary_component
    _timing = profile.decision_timing
    _read = profile.read_contract
    _write = profile.write_contract
    _key = profile.execution_key
    frozen_declarations = profile.frozen_declarations
    declarations = profile.declarations
    compatibility = profile.compatibility
    reference_from_originals = reference.reference_from_originals

MODEL, PROFILE = model, profile
_reference, _summary = reference.reference_trace, reference.human_summary
_trajectory_diagnostics = reference.trajectory_diagnostics


@dataclass(frozen=True)
class ExampleResult:
    """Return complete inspectable evidence alongside the compact human result."""

    report: ProtocolReport
    conformance: ReferenceConformanceReport
    disposition: ComparisonDisposition
    traces: tuple[ConformanceTrace, ...]
    tape: SourceTape
    summary: dict[str, Any]


def run(root: Path, *, drift: Literal["action", "cost", "time"] | None = None) -> ExampleResult:
    """Run actual learners, joint checkpoints and the existing protected audit.

    Parameters
    ----------
    root : pathlib.Path
        Trusted local output directory. Retained declarations, original source
        ticks and real state acknowledgements permit explicit continuation.
        Completed results use the existing reserved cached-report disclosure.
    drift : {'action', 'cost', 'time'} or None, optional
        Deliberately corrupt one independent reference assertion, retaining an
        inspectable failed-conformance result and the conservative fallback.

    Returns
    -------
    ExampleResult
        Real protected report, full reference check, scoped disposition, complete
        candidate traces and explicit captured clock tape. Nothing is promoted.
    """
    frozen, resumed = frozen_declarations(root)
    audit = ProtocolAudit(root / "audit", limits=frozen.design.limits)
    locator_path = root / "completed-locator.json"
    if not locator_path.exists():
        locator_path = root / "prepared-result-locator.json"
    if locator_path.exists():
        result = _prepared_result(root, frozen, audit, locator_path)
    else:
        traces: list[ConformanceTrace] = []
        tape: list[SourceTick] = []
        mode = "imported" if resumed else "produced"

        def loader() -> RawPanel:
            """Produce the complete frozen grid only after both actual audit reservations."""
            return _produce(root, frozen, traces, tape, mode=mode)

        report = audit.evaluate(frozen.design, frozen.selection, loader, actor="public-ci-example", mode=mode)
        if not isinstance(report, ProtocolReport):
            raise RuntimeError("The predeclared bounded CI formulation unexpectedly became unsupported")
        _write(root, "prepared-result-locator.json", report.locator)
        result = assess_existing(frozen, report, tuple(traces), SourceTape(schema_version=1, ticks=tuple(tape)))
    _finalize(root, result)
    # Faults are held-evidence diagnostics after the immutable base publication.
    return result if drift is None else assess_existing(frozen, result.report, result.traces, result.tape, drift=drift)


def _finalize(root: Path, result: ExampleResult) -> None:
    """Complete original helper artifacts; never replace a prepared report identity."""
    _write(
        root,
        "completed-evidence.json",
        CachedExample(
            schema_version=1,
            report_fingerprint=result.report.fingerprint,
            conformance=result.conformance,
            disposition=result.disposition,
            traces=result.traces,
            tape=result.tape,
        ),
    )
    _write(root, "completed-locator.json", result.report.locator)


def _prepared_result(root: Path, frozen: FrozenExample, audit: ProtocolAudit, locator_path: Path) -> ExampleResult:
    """Reserve original report disclosure before any prepared result-bearing body."""
    locator = _read(locator_path, ReportLocator)
    view = audit.read_report(locator, actor="public-ci-example-cached-reader")
    if view.original.panel.design != frozen.design or view.original.panel.selection != frozen.selection:
        raise ValueError("Prepared report differs from its original frozen selection")
    body = root / "completed-evidence.json"
    if body.exists():
        cached = _read(body, CachedExample)
        if cached.report_fingerprint != view.original.fingerprint:
            raise ValueError("Cached comparison differs from its reserved original report")
        cached.disposition.validate_for(frozen.plan, cached.conformance, view.original)
        traces, tape = cached.traces, cached.tape
    else:
        tape = SourceTape(
            schema_version=1,
            ticks=tuple(
                _read(root / "ticks" / (world + "-" + str(ordinal) + JSON_SUFFIX), SourceTick)
                for ordinal in range(3)
                for world in WORLDS
            ),
        )
        traces = tuple(
            ConformanceTrace(
                schema_version=1,
                recipe=MODEL.ADAPTER,
                steps=tuple(
                    original.step
                    for tick in tape.ticks
                    for original in tick.observations
                    if original.step.receipt.key.cell.candidate_id == candidate
                ),
            )
            for candidate in MODEL.MODES
        )
    return assess_existing(frozen, view.original, traces, tape)


def assess_existing(
    frozen: FrozenExample, report: ProtocolReport, traces: tuple[ConformanceTrace, ...], tape: SourceTape, *, drift=None
) -> ExampleResult:
    """Assess already held original evidence without another experiment.

    Parameters
    ----------
    frozen : FrozenExample
        Original law/family/selection/source/compatibility declaration.
    report : ProtocolReport
        Complete original protected report, after its mediated read/disclosure.
    traces : tuple of ConformanceTrace
        Complete actual candidate trajectories, including every prefix decision.
    tape : SourceTape
        Immutable original inputs and clocks; the independent reference reads this,
        never the actual output trace being tested for a mapping/clock defect.
    drift : {'action', 'cost', 'time'} or None, optional
        Deliberately wrong reference assertion for a diagnostic regression; it
        changes neither source observations nor the immutable original report.

    Returns
    -------
    ExampleResult
        Full independent conformance, preserved family conclusions and fallback.

    Raises
    ------
    ValueError
        Any original family/metric/terminal/cost/source join or complete tape differs.

    Notes
    -----
    Pure held-data calculation: no source read, callback, deployment or new
    confirmation. A source read must already have been reserved by the caller."""
    plan = frozen.plan
    actual = next(trace for trace in traces if trace.steps[0].receipt.key.cell.candidate_id == "fixed")
    expected = _reference(tape, drift=drift)
    conformance = check_reference_conformance(plan.conformance_plan, expected, actual)
    disposition = assess_comparison(plan, conformance, report)
    summary = _summary(report, conformance, disposition, traces)
    return ExampleResult(report, conformance, disposition, traces, tape, summary)


def _produce(
    root: Path, frozen: FrozenExample, traces: list, tape: list, *, mode: Literal["produced", "imported"]
) -> RawPanel:
    """Produce the complete finite grid only after both real audit reservations."""
    invocation, prior_attempts = _prepare_invocation(root, frozen)
    sessions = _open_sessions(root, frozen)
    _execute_grid(root, frozen, sessions, tape)
    cells, produced = _completed_grid(sessions)
    traces.extend(produced)
    ledger = _attempt_ledger(invocation, prior_attempts)
    panel = RawPanel(
        schema_version=1,
        design=frozen.design,
        selection=frozen.selection,
        rows=cells,
        ledger=ledger,
        mode=mode,
        assembled_at=datetime.now(UTC),
    )
    _write(
        root,
        "work/completed-" + invocation.attempt_id + JSON_SUFFIX,
        AttemptLedger(
            schema_version=1,
            stage="protected",
            attempts=tuple(item for item in ledger.attempts if invocation.attempt_id in item.attempt_id),
        ),
    )
    return panel


def _prepare_invocation(root: Path, frozen: FrozenExample):
    """Refuse exhausted capacities before any new physical attempt or owner callback."""
    design = frozen.design
    previous = sorted((root / "work").glob("invocation-*.json")) if (root / "work").exists() else []
    prior_attempts = _previous_attempts(root, previous)
    if len(prior_attempts) + len(MODEL.MODES) > design.budget.maximum_physical_attempts:
        raise ValueError("Actual attempted loader work exceeds the frozen global budget")
    if (
        sum(len(attempt.outputs) for attempt in prior_attempts) + design.raw_cells
        > design.budget.maximum_output_references
    ):
        raise ValueError("Actual loader output references exceed the frozen global budget")
    invocation = Invocation(schema_version=1, attempt_id=uuid4().hex, started_at=datetime.now(UTC))
    _write(root, "work/invocation-" + invocation.attempt_id + JSON_SUFFIX, invocation)
    return invocation, prior_attempts


def _execute_grid(root: Path, frozen: FrozenExample, sessions: dict, tape: list) -> None:
    """Retain common emission/source clocks before real per-owner transitions."""
    evaluation = project_frozen(frozen.design, frozen.selection).fingerprint
    for ordinal in range(3):
        tick = _emission_tick(root, ordinal)
        for session in sessions.values():
            session.emit(ordinal, tick.decision_at)
        for world in WORLDS:
            delivery = _source_delivery(root, sessions, world, tick, evaluation)
            for original in delivery.observations:
                sessions[original.step.receipt.key.cell.candidate_id, world].apply_original(original)
            tape.append(delivery)


def _open_sessions(root: Path, frozen: FrozenExample) -> dict[tuple[str, str], Session]:
    """Open complete independent owners without replacing their original law."""
    sessions = {}
    for candidate in MODEL.MODES:
        for world in WORLDS:
            recipe = MODEL.Recipe(
                candidate,
                MODEL.configuration(candidate),
                MODEL.ControlSchedule(schema_version=1, calendar=None),
                frozen.design.candidate(candidate).candidate.configuration_fingerprint,
                frozen.design.snapshot.data_fingerprint,
                world,
            )
            path = root / "sessions" / candidate / world
            sessions[candidate, world] = Session(path, recipe, frozen.environment, create=not path.exists())
    return sessions


def _emission_tick(root: Path, ordinal: int) -> EmissionTick:
    """Persist a common actual tick before any candidate emits this round."""
    clock_path = root / "ticks" / ("emission-" + str(ordinal) + JSON_SUFFIX)
    if clock_path.exists():
        tick = _read(clock_path, EmissionTick)
        if tick.ordinal != ordinal:
            raise ValueError("Retained emission tick belongs to another round")
        return tick
    tick = EmissionTick(schema_version=1, ordinal=ordinal, decision_at=datetime.now(UTC))
    _write(root, "ticks/emission-" + str(ordinal) + JSON_SUFFIX, tick)
    return tick


def _source_delivery(root: Path, sessions: dict, world: str, tick: EmissionTick, evaluation: str) -> SourceTick:
    """Retain all original outcomes after emission and before their applications."""
    relative = "ticks/" + world + "-" + str(tick.ordinal) + JSON_SUFFIX
    if (root / relative).exists():
        delivery = _read(root / relative, SourceTick)
    else:
        visible, applied = datetime.now(UTC), datetime.now(UTC)
        originals = tuple(
            sessions[candidate, world].prepare_observation(
                tick.ordinal,
                TABLE[WORLDS.index(world)][tick.ordinal],
                evaluation,
                available_at=visible,
                apply_at=applied,
            )
            for candidate in MODEL.MODES
        )
        delivery = SourceTick(
            schema_version=1, world=world, ordinal=tick.ordinal, decision_at=tick.decision_at, observations=originals
        )
        _write(root, relative, delivery)
    if delivery.world != world or delivery.ordinal != tick.ordinal or delivery.decision_at != tick.decision_at:
        raise ValueError("Retained source delivery differs from its original emission tick")
    return delivery


def _completed_grid(sessions: dict) -> tuple[tuple[RawCell, ...], tuple[ConformanceTrace, ...]]:
    """Retain every prefix and bind only the named terminal step to each raw cell."""
    cells, traces = [], []
    for candidate in MODEL.MODES:
        steps = []
        for world in WORLDS:
            _, state = sessions[candidate, world].snapshots()
            steps.extend(state.observations)
            cells.append(
                RawCell(
                    candidate_id=candidate,
                    member_id=world,
                    seed=0,
                    state="completed",
                    evidence=_primary(state.observations[-1]).original,
                )
            )
        traces.append(ConformanceTrace(schema_version=1, recipe=MODEL.ADAPTER, steps=tuple(steps)))
    return tuple(cells), tuple(traces)


def _previous_attempts(root: Path, previous: list[Path]) -> tuple[PhysicalAttempt, ...]:
    """Retain real interrupted work as unavailable, never erase it on continuation."""
    attempts = []
    for path in previous:
        original = _read(path, Invocation, 16 * 1024)
        completed = root / "work" / ("completed-" + original.attempt_id + JSON_SUFFIX)
        if completed.exists():
            attempts.extend(_read(completed, AttemptLedger).attempts)
        else:
            attempts.extend(
                PhysicalAttempt(
                    attempt_id=original.attempt_id + "-" + candidate,
                    group_id="simulation-" + candidate,
                    trial_id=None,
                    candidate_id=candidate,
                    state="unavailable",
                    started_at=original.started_at,
                    completed_at=None,
                    outputs=(),
                    costs=(),
                )
                for candidate in MODEL.MODES
            )
    return tuple(attempts)


def _attempt_ledger(current: Invocation, previous: tuple[PhysicalAttempt, ...]) -> AttemptLedger:
    """Join bounded earlier attempts to this actual completed invocation."""
    attempts = list(previous)
    completed_at = datetime.now(UTC)
    attempts.extend(
        PhysicalAttempt(
            attempt_id=current.attempt_id + "-" + candidate,
            group_id="simulation-" + candidate,
            trial_id=None,
            candidate_id=candidate,
            state="completed",
            started_at=current.started_at,
            completed_at=completed_at,
            outputs=tuple(CellIdentity(candidate_id=candidate, member_id=world, seed=0) for world in WORLDS),
            costs=(),
        )
        for candidate in MODEL.MODES
    )
    return AttemptLedger(schema_version=1, stage="protected", attempts=tuple(attempts))


def main() -> None:
    """Write actual local evidence and print the full human result as JSON."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("prequential-ci-output")
    result = run(root)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
