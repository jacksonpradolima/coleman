"""Development-only choice and separately charged independent study assessment."""

from __future__ import annotations

import hashlib
import math
import time
from fractions import Fraction
from pathlib import Path

from coleman.application.resource_usage import peak_rss_mib
from examples.research.expert_rate_execute import (
    METRICS,
    ResourceLimitError,
    _checkpoint,
    _new_owner,
    restore_checkpoint,
)
from examples.research.expert_rate_math import comparator_losses, periodic_trace, reference_trace, score
from examples.research.expert_rate_process_records import Attempt, DevelopmentChoice, Gate, OracleResult
from examples.research.expert_rate_records import (
    REFERENCE_PROFILES,
    Artifact,
    DevelopmentFamily,
    DevelopmentScore,
    Family,
    FamilyRun,
    StudyInput,
    admit_protocol,
    read_artifact,
)

ORDER: tuple[Family, ...] = ("constant", "anytime", "fixed-share", "periodic", "fixed-uniform")
TIES: tuple[Family, ...] = ("fixed-uniform", "constant", "anytime", "fixed-share", "periodic")


def read_family(root: Path, artifact: Artifact, protocol: StudyInput) -> FamilyRun:
    """Read actual digest/size-bound bytes and require the full pinned ordered grid.

    Parameters
    ----------
    root : pathlib.Path
        Trusted original study root containing internally named artifacts.
    artifact : Artifact
        Exact retained path, size and SHA, not a completion witness.
    protocol : StudyInput
        Original full source; the reader checks its pinned content again.

    Returns
    -------
    FamilyRun
        Fully validated original family data with every configuration/case.

    Raises
    ------
    ValueError, OSError
        Altered/truncated/oversized content, version or original grid mismatch.
    """
    if Path(artifact.path).is_absolute() or ".." in Path(artifact.path).parts:
        raise ValueError("Family artifact requires its internal relative path")
    with (root / artifact.path).open("rb") as stream:
        content = stream.read(16 * 1024**2 + 1)
    if len(content) != artifact.size_bytes or hashlib.sha256(content).hexdigest() != artifact.sha256:
        raise ValueError("Family artifact bytes differ from their retained exact identity")
    result = FamilyRun.from_json(content)
    result.require_grid(protocol)
    return result


def _exact_case_loss(run, case) -> Fraction:
    return score(tuple(s.before for s in run.trace), case.panels)


def _ideal_trace(run, case):
    """Select the independent ideal recurrence, without a production kernel."""
    setting = run.setting
    if setting.family in ("constant", "anytime", "fixed-share"):
        profile = REFERENCE_PROFILES[setting.family]
        ideal = reference_trace(
            profile, case.panels, eta=setting.eta, multiplier=setting.multiplier, alpha=setting.alpha
        )
    else:
        # Window64 has no refit before the same first eight-panel boundary;
        # fixed-uniform is checked separately, without this adaptive oracle.
        ideal = periodic_trace(case.panels, setting.window or 64)
    return ideal


def _coefficient_error(trace, ideal, fixed):
    """Require every BEFORE/AFTER coefficient to satisfy the declared envelope."""
    maximum = 0.0
    for index, step in enumerate(trace):
        for actual, expected in ((step.before, ideal.before[index]), (step.after, ideal.after[index])):
            for observed, target in zip(actual, expected, strict=True):
                value = 1 / 3 if fixed else float(target)
                maximum = max(maximum, abs(observed - value))
                if not math.isclose(observed, value, abs_tol=2e-14, rel_tol=2e-13):
                    raise ValueError("Actual coefficient is outside its locally frozen Decimal80 envelope")
    return maximum


def _check_case(run, case) -> tuple[float, float]:
    setting = run.setting
    ideal = _ideal_trace(run, case)
    maximum = _coefficient_error(run.trace, ideal, setting.family == "fixed-uniform")
    total = _exact_case_loss(run, case)
    fixed, switched = comparator_losses(case.panels)
    movement = sum(
        (abs(Fraction(a) - Fraction(b)) for s in run.trace for a, b in zip(s.after, s.before, strict=True)), Fraction()
    )
    informative = score(
        tuple(s.before for s in run.trace[case.informative_start :]), case.panels[case.informative_start :]
    ) / (len(case.panels) - case.informative_start)
    values = (total, total / len(case.panels), informative, total - fixed, total - switched, movement)
    if tuple((m.name, m.value) for m in run.metrics) != tuple(zip(METRICS, map(float, values), strict=True)):
        raise ValueError("Actual observed metrics differ from exact original stored-binary64 arithmetic")
    ideal_score = sum(
        (
            Fraction(1, 3) * Fraction(x) if setting.family == "fixed-uniform" else Fraction(p) * Fraction(x)
            for vector, row in zip(ideal.before, case.panels, strict=True)
            for p, x in zip(vector, row, strict=True)
        ),
        Fraction(),
    )
    error = float(abs(total - ideal_score))
    if error > 1e-11:
        raise ValueError("Actual cumulative loss is outside its locally frozen ideal-score envelope")
    return maximum, error


def _read_original_family(root, protocol, family, repetition):
    """Bind actual bytes, grid and process coordinates before independent checks."""
    path = root / f"r{repetition}-{family}" / "family.json"
    content = read_artifact(path)
    reference = Artifact(
        path=str(path.relative_to(root)), sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    )
    result = read_family(root, reference, protocol)
    if result.status != "completed" or result.repetition != repetition or result.family != family:
        raise ValueError("Independent oracle requires every exact original completed physical slot")
    return result


def _oracle_guard(started: float) -> None:
    """Apply the original oracle deadline/RSS cap without conflating zero/null."""
    memory = peak_rss_mib()
    if time.monotonic() - started >= 180 or memory is None or memory > 512:
        raise ResourceLimitError("original-independent-oracle-wall-or-rss-cap")


def _checkpoint_recovery(run, case, protocol) -> None:
    """Exercise selected full-state readers, independently of expected arithmetic."""
    owner = _new_owner(protocol, case, run.setting)
    for checkpoint in run.checkpoints:
        restore_checkpoint(owner, checkpoint)
        expected = run.trace[0].before if checkpoint.position == 0 else run.trace[checkpoint.position - 1].after
        if tuple(w.value for w in owner.weights) != expected or _checkpoint(owner) != checkpoint:
            raise ValueError("Complete selected checkpoint does not restore the original trace position/state")


def _recorded_attempts(root: Path) -> tuple[Attempt, ...]:
    """Read every original versioned process slot, without dropping failures."""
    return tuple(
        Attempt.from_json(read_artifact(root / f"r{repetition}-{family}-attempt.json"))
        for repetition in range(3)
        for family in ORDER[repetition:] + ORDER[:repetition]
    )


def _assessed_gate(root: Path, protocol: StudyInput, started: float) -> Gate:
    """Charge the final complete assessment to the same original oracle window."""
    choice = DevelopmentChoice.from_json(read_artifact(root / "selection.json"))
    gate = entry_gate(root, _recorded_attempts(root), protocol, choice)
    _oracle_guard(started)
    return gate


def oracle(root: Path, protocol: StudyInput) -> OracleResult:
    """Check all actual repeated traces against separately authored arithmetic.

    Parameters
    ----------
    root : pathlib.Path
        Actual completed family artifacts from exactly the original15 slots.
    protocol : StudyInput
        Original immutable once-converted48-case/17-setting input.

    Returns
    -------
    OracleResult
        Full step count, numerical residual maxima and exact repetition equality.

    Raises
    ------
    ResourceLimitError, ValueError
        Original180s/512MiB limits, incomplete data, arithmetic or identity mismatch.

    Notes
    -----
    Performs no learner observation or fourth physical trial. Decimal80 and
    Fraction work is charged to this separate declared oracle process.
    """
    started = time.monotonic()
    protocol = admit_protocol(protocol)
    first = {}
    checked, maxima = 0, (0.0, 0.0)
    for repetition in range(3):
        for family in ORDER:
            result = _read_original_family(root, protocol, family, repetition)
            for index, run in enumerate(result.cases):
                _oracle_guard(started)
                case = next(c for c in protocol.cases if c.case_id == run.case_id)
                _checkpoint_recovery(run, case, protocol)
                errors = _check_case(run, case)
                maxima = max(maxima[0], errors[0]), max(maxima[1], errors[1])
                checked += len(run.trace)
                digest = hashlib.sha256(run.to_json().encode()).hexdigest()
                key = (family, index)
                if repetition == 0:
                    first[key] = digest
                elif first[key] != digest:
                    raise ValueError("Repeated physical trials changed complete mathematics/checkpoint identities")
    return OracleResult(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        checked_steps=checked,
        maximum_coefficient_error=maxima[0],
        maximum_score_error=maxima[1],
        repetitions_equal=True,
        gate=_assessed_gate(root, protocol, started),
    )


def development_projection(result: FamilyRun, protocol: StudyInput) -> DevelopmentFamily | None:
    """Project every development treatment; evaluation fields never enter choice.

    Parameters
    ----------
    result : FamilyRun
        Actual full worker result, including unsuccessful evaluation treatments.
    protocol : StudyInput
        Original pinned source defining development membership, never statuses.

    Returns
    -------
    DevelopmentFamily or None
        Complete development-only values, or unavailable when any ORIGINAL
        development treatment failed. No family/configuration is dropped.

    Raises
    ------
    ValueError
        Original source/grid mismatch before projection.
    """
    result.require_grid(protocol)
    sources = {c.case_id: c for c in protocol.cases if c.partition == "development"}
    rows = []
    for run in result.cases:
        if run.case_id not in sources:
            continue
        if run.status != "completed":
            return None
        case = sources[run.case_id]
        rows.append(
            DevelopmentScore(
                case_id=case.case_id,
                setting=run.setting,
                horizon=len(case.panels),
                exact_loss=str(_exact_case_loss(run, case)),
            )
        )
    projection = DevelopmentFamily(
        schema_version=1, protocol_content=protocol.protocol_content, family=result.family, rows=tuple(rows)
    )
    projection.require_grid(protocol)
    return projection


def select_development(projections: tuple[DevelopmentFamily, ...], protocol: StudyInput):
    """Select only from all five complete typed development-only families.

    Parameters
    ----------
    projections : tuple of DevelopmentFamily
        All five original families, with no evaluation values/status fields.
    protocol : StudyInput
        Complete admitted source; setting/family ties retain original order.

    Returns
    -------
    tuple
        Selected family and every family/slot pair in predeclared tie order.

    Raises
    ------
    ValueError
        Missing, duplicate, substituted or incomplete development treatments.
    """
    if tuple(p.family for p in projections) != TIES:
        raise ValueError("Selection requires all original families in declared tie order")
    selected, scores = [], []
    for projection in projections:
        projection.require_grid(protocol)
        slots = tuple(s.slot for s in protocol.settings if s.family == projection.family)
        means = tuple(
            sum((Fraction(r.exact_loss) / r.horizon for r in projection.rows if r.setting.slot == slot), Fraction())
            / 24
            for slot in slots
        )
        index = min(enumerate(means), key=lambda pair: (pair[1], pair[0]))[0]
        selected.append((projection.family, slots[index]))
        scores.append(means[index])
    winner = min(range(5), key=lambda i: (scores[i], i))
    return TIES[winner], tuple(selected)


def freeze_choice(root: Path, protocol: StudyInput) -> DevelopmentChoice | None:
    """Read only five development artifacts before opening any full family data.

    Parameters
    ----------
    root : pathlib.Path
        Original study root; every original family must retain its projection.
    protocol : StudyInput
        Exact original source defining all fixed selection inputs.

    Returns
    -------
    DevelopmentChoice or None
        Complete choice and exact source artifacts, or explicitly unavailable.

    Raises
    ------
    ValueError, OSError
        Invalid typed source, bytes or original development grid.
    """
    projections, artifacts = [], []
    for family in TIES:
        name = f"r0-{family}/development.json"
        path = root / name
        if not path.exists():
            return None
        content = read_artifact(path)
        if len(content) > 16 * 1024**2:
            raise ValueError("Development projection exceeds its original artifact bound")
        projections.append(DevelopmentFamily.from_json(content))
        artifacts.append(Artifact(path=name, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)))
    family, slots = select_development(tuple(projections), protocol)
    return DevelopmentChoice(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        selected_family=family,
        selected_slots=slots,
        development_artifacts=tuple(artifacts),
    )


def _evaluation_values(root, attempts, protocol):
    """Read full evaluation evidence only after the immutable choice exists."""
    values = {}
    for attempt in attempts:
        if attempt.repetition != 0:
            continue
        if attempt.status != "completed" or attempt.artifact is None:
            raise ValueError("Gate cannot drop failed original families/configurations")
        result = read_family(root, attempt.artifact, protocol)
        for run in result.cases:
            case = next(c for c in protocol.cases if c.case_id == run.case_id)
            values[(run.setting.family, run.setting.slot, case.case_id)] = _exact_case_loss(run, case) / len(
                case.panels
            )
    return values


def _shape_excess(shape, values, protocol, choice):
    """Average original rotations within shape; never count independent units."""
    chosen, selected = choice.selected_family, dict(choice.selected_slots)
    cases = tuple(c for c in protocol.cases if c.partition == "evaluation-exposed" and c.shape == shape)
    deployed = sum((values[(chosen, selected[chosen], c.case_id)] for c in cases), Fraction()) / len(cases)
    best = min(
        sum((values[(s.family, s.slot, c.case_id)] for c in cases), Fraction()) / len(cases)
        for s in protocol.settings
        if s.family in ("constant", "anytime", "fixed-share")
    )
    return shape, float(deployed - best)


def entry_gate(root: Path, attempts: tuple[Attempt, ...], protocol: StudyInput, choice: DevelopmentChoice) -> Gate:
    """Assess exposed rate shapes using the already frozen development-only choice.

    Parameters
    ----------
    root : pathlib.Path
        Original completed study root, after immutable selection publication.
    attempts : tuple of Attempt
        All15 checked physical slots; only repetition0 supplies finite effects.
    protocol : StudyInput
        Original source with all48cases/17configurations and exposed-role status.
    choice : DevelopmentChoice
        Complete previously frozen development-only selection. Never retuned.

    Returns
    -------
    Gate
        Retain baselines or support later formulation review, without inference.

    Raises
    ------
    ValueError, OSError
        Missing/changed original selection, source, complete grid or evidence.
    """
    retained = DevelopmentChoice.from_json(read_artifact(root / "selection.json"))
    if retained.to_json() != choice.to_json() or choice.protocol_content != protocol.protocol_content:
        raise ValueError("Evaluation requires the exact already-frozen original development choice")
    values = _evaluation_values(root, attempts, protocol)
    shapes = ("stationary-gap", "stationary-small-gap", "neutral-prefix", "contrast-pulses")
    excess = tuple(_shape_excess(shape, values, protocol, choice) for shape in shapes)
    chosen = choice.selected_family
    decision = "formulation_review_supported" if sum(v > 1 / 64 for _, v in excess) >= 2 else "retain_baselines"
    return Gate(
        schema_version=1,
        selected_family=chosen,
        selected_slots=choice.selected_slots,
        shape_excess=tuple(excess),
        decision=decision,
    )
