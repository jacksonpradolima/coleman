"""Execute the finite public baseline entry protocol without selecting a new learner.

Run ``python -m examples.research.expert_rate_entry --help`` with the optional
``coleman[cli]`` profile. Importing this module imports no Click command.
"""

from __future__ import annotations

import hashlib
import resource
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Literal, NamedTuple

from coleman.application.resource_usage import output_exceeds
from coleman.artifacts import ArtifactWriter
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from examples.research.expert_rate_assess import development_projection, freeze_choice, oracle, read_family
from examples.research.expert_rate_execute import ResourceLimitError, run_family
from examples.research.expert_rate_process_records import (
    Attempt,
    ChildSummary,
    DevelopmentChoice,
    Gate,
    OracleResult,
    StudyReport,
    UnavailableChoice,
)
from examples.research.expert_rate_records import Artifact, Family, StudyInput, load_protocol, read_artifact

ORDER: tuple[Family, ...] = ("constant", "anytime", "fixed-share", "periodic", "fixed-uniform")


class Assessment(NamedTuple):
    """Keep the single independent-oracle disposition and its nullable units."""

    oracle: Artifact | None
    wall_seconds: float | None
    gate: Gate | None
    status: Literal["completed", "resource_limited", "numerical_failure"]
    reason: str | None


class BaselineExecution(NamedTuple):
    """Retain fixed-slot outcomes, frozen choice and complete parent wall windows."""

    attempts: tuple[Attempt, ...]
    choice: DevelopmentChoice | None
    choice_artifact: Artifact | None
    pass_wall_seconds: tuple[float, float, float]
    wall_seconds: float


def source_files() -> tuple[tuple[str, str], ...]:
    """Read every actual driver source identity, without importing a plugin."""
    return tuple(
        (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(Path(__file__).parent.glob("expert_rate_*.py"))
    )


def publish(root: Path, name: str, content: str) -> Artifact:
    """Publish bounded immutable original bytes, accounting for retained output.

    Parameters
    ----------
    root : pathlib.Path
        Trusted fresh study directory, not a hostile-filesystem sandbox.
    name : str
        Internally selected relative artifact name.
    content : str
        Complete validated canonical DTO bytes, at most16MiB.

    Returns
    -------
    Artifact
        Exact content digest/size and relative path, not source authentication.

    Raises
    ------
    ResourceLimitError, ValueError, OSError
        Original16MiB/64MiB caps or immutable publication cannot be satisfied.
    """
    raw = content.encode("utf-8")
    if len(raw) > 16 * 1024**2 or output_exceeds(64 * 1024**2 - len(raw), root=root, maximum_entries=8192):
        raise ResourceLimitError("original-artifact-or-aggregate-output-cap")
    ArtifactWriter(root).write_immutable(root / name, content)
    return Artifact(path=name, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _child(protocol: StudyInput, family: Family, repetition: int, root: Path) -> None:
    directory = root / f"r{repetition}-{family}"
    result = run_family(protocol, family, repetition, directory)
    artifact = publish(root, str((directory / "family.json").relative_to(root)), result.to_json())
    development = development_projection(result, protocol) if repetition == 0 else None
    development_artifact = None
    if development is not None:
        development_artifact = publish(root, f"r0-{family}/development.json", development.to_json())
    summary = ChildSummary(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        family=family,
        repetition=repetition,
        status=result.status,
        reason=result.reason,
        attempted=result.attempted,
        source_files=result.source_files,
        artifact=artifact,
        development_artifact=development_artifact,
    )
    publish(root, f"r{repetition}-{family}/summary.json", summary.to_json())
    # Identical case bytes now live inside the fully published family artifact.
    # Remove only these exact duplicate prefix files, never the sole evidence.
    for run in result.cases:
        path = directory / f"setting-{run.setting.slot}-{run.case_id}.json"
        if path.read_text() != run.to_json():
            raise ValueError("Partial evidence changed before duplicate consolidation")
        path.unlink()


def _process(command: list[str], root: Path, name: str, timeout: float) -> tuple[bool, float, float]:
    start, cpu = time.monotonic(), resource.getrusage(resource.RUSAGE_CHILDREN)
    with (root / f"{name}.log").open("wb") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return False, time.monotonic() - start, _child_cpu(cpu)
    if result.returncode not in (0, 75):
        raise RuntimeError(f"Unexpected child software failure ({result.returncode}); retain {name}.log")
    elapsed = time.monotonic() - start
    return result.returncode == 0 and elapsed <= timeout, elapsed, _child_cpu(cpu)


def _child_cpu(before) -> float:
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime


def _attempt(
    root: Path,
    protocol: StudyInput,
    coordinate: tuple[Family, int],
    remaining: float,
    sources: tuple[tuple[str, str], ...],
) -> Attempt:
    """Launch at most once; reserve the complete original workload even on death."""
    family, repetition = coordinate
    reserved = sum(len(c.panels) for c in protocol.cases) * sum(s.family == family for s in protocol.settings)
    if remaining <= 0:
        return Attempt(
            schema_version=1,
            family=family,
            repetition=repetition,
            status="not_started",
            reason="original-pass-or-study-cap",
            reserved_calls=0,
            attempted_calls=None,
            wall_seconds=None,
            cpu_seconds=None,
            artifact=None,
        )
    name = f"r{repetition}-{family}"
    command = [
        sys.executable,
        "-m",
        "examples.research.expert_rate_entry",
        "--output",
        str(root),
        "--family",
        family,
        "--repetition",
        str(repetition),
    ]
    ok, wall, cpu = _process(command, root, name, min(180, remaining))
    if not ok:
        return Attempt(
            schema_version=1,
            family=family,
            repetition=repetition,
            status="resource_limited",
            reason="original-process-deadline-or-resource-cap",
            reserved_calls=reserved,
            attempted_calls=None,
            wall_seconds=wall,
            cpu_seconds=cpu,
            artifact=None,
        )
    value = ChildSummary.from_json(read_artifact(root / name / "summary.json"))
    if value.protocol_content != protocol.protocol_content or (value.family, value.repetition, value.source_files) != (
        family,
        repetition,
        sources,
    ):
        raise ValueError("Child original identity/source differs from this frozen attempt")
    return Attempt(
        schema_version=1,
        family=family,
        repetition=repetition,
        status=value.status,
        reason=value.reason,
        reserved_calls=reserved,
        attempted_calls=value.attempted,
        wall_seconds=wall,
        cpu_seconds=cpu,
        artifact=value.artifact,
    )


def _assess_finished(root, protocol, attempts, choice, wall_limited: bool = False) -> Assessment:
    """Charge one independent oracle only after every original baseline completes."""
    if wall_limited:
        return Assessment(None, None, None, "resource_limited", "original-pass-or-baseline-wall-cap")
    if not all(a.status == "completed" for a in attempts) or choice is None:
        if any(a.status == "numerical_failure" for a in attempts):
            return Assessment(None, None, None, "numerical_failure", "actual-original-numerical-refusal")
        return Assessment(None, None, None, "resource_limited", "one-or-more-original-physical-slots-incomplete")
    ok, wall, _ = _process(
        [sys.executable, "-m", "examples.research.expert_rate_entry", "--output", str(root), "--oracle"],
        root,
        "oracle",
        180,
    )
    if not ok:
        return Assessment(None, wall, None, "resource_limited", "original-independent-oracle-process-cap")
    content = read_artifact(root / "oracle.json")
    checked = OracleResult.from_json(content)
    if checked.protocol_content != protocol.protocol_content or checked.checked_steps != 77112:
        raise ValueError("Independent oracle did not check the full original workload")
    if (checked.gate.selected_family, checked.gate.selected_slots) != (choice.selected_family, choice.selected_slots):
        raise ValueError("Assessed decision differs from the original frozen development-only choice")
    artifact = Artifact(path="oracle.json", sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    return Assessment(artifact, wall, checked.gate, "completed", None)


def _verify_attempt(root: Path, attempt: Attempt, protocol: StudyInput) -> None:
    """Require actual full source/grid/byte evidence before publishing completion."""
    if attempt.artifact is None:
        return
    result = read_family(root, attempt.artifact, protocol)
    if result.runtime != InstalledRuntime.capture() or result.driver_runtime != SelectedInstalledRuntime.capture(
        extras=("cli",)
    ):
        raise ValueError("Family data differs from the actual same-host installed study runtime")
    if result.source_files != source_files():
        raise ValueError("Actual family data differs from the frozen driver source inventory")
    if (result.family, result.repetition, result.status, result.reason, result.attempted) != (
        attempt.family,
        attempt.repetition,
        attempt.status,
        attempt.reason,
        attempt.attempted_calls,
    ):
        raise ValueError("Returned process bookkeeping differs from the actual complete family artifact")


def _run_baseline_passes(root, protocol, sources) -> BaselineExecution:
    """Run only the original15 slots, retaining full parent validation overhead."""
    started, attempts = time.monotonic(), []
    choice, choice_artifact = None, None
    pass_walls = []
    for repetition in range(3):
        pass_start = time.monotonic()
        for family in ORDER[repetition:] + ORDER[:repetition]:
            remaining = min(900 - (time.monotonic() - pass_start), 2700 - (time.monotonic() - started))
            attempts.append(_attempt(root, protocol, (family, repetition), remaining, sources))
        if repetition == 0:
            choice = freeze_choice(root, protocol)
            if choice is not None:
                choice_artifact = publish(root, "selection.json", choice.to_json())
            else:
                publish(
                    root,
                    "selection-unavailable.json",
                    UnavailableChoice(schema_version=1, protocol_content=protocol.protocol_content).to_json(),
                )
        # Selection or its refusal is now immutable. Only now may the parent
        # inspect full evaluation fields and advertise verified family status.
        for attempt in attempts[-5:]:
            _verify_attempt(root, attempt, protocol)
            publish(root, f"r{attempt.repetition}-{attempt.family}-attempt.json", attempt.to_json())
        pass_walls.append(time.monotonic() - pass_start)
    return BaselineExecution(
        tuple(attempts),
        choice,
        choice_artifact,
        (pass_walls[0], pass_walls[1], pass_walls[2]),
        time.monotonic() - started,
    )


def run_study(root: Path) -> StudyReport:
    """Run the one fixed public baseline attempt in a new exclusively owned root.

    Parameters
    ----------
    root : pathlib.Path
        Trusted absent output directory. Atomic mkdir rejects a duplicate or
        concurrent invocation; this function supplies no automatic retry.

    Returns
    -------
    StudyReport
        Actual original15 slots and honest complete/limited disposition.

    Raises
    ------
    FileExistsError, RuntimeError, ValueError, OSError
        Duplicate namespace, unexpected software/source/contract/publication error.
        Such errors never publish an ordinary completed/limited aggregate.

    Notes
    -----
    Original180s child,900s pass,2700s baseline,512MiB/16MiB/64MiB and77,112
    work caps remain unchanged. Unknown killed-child work stays reserved and
    unavailable. A separate single180s oracle runs only after all slots complete.
    Every physical attempt is retained; repetitions are not independent units.
    Invoke from the source checkout or a dedicated copied example root that
    contains the supplied ``examples`` namespace; children use that module route.
    """
    root.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol(Path(__file__).with_name("expert-rate-entry-protocol-v2.json"))
    sources = source_files()
    baseline = _run_baseline_passes(root, protocol, sources)
    assessment = _assess_finished(
        root,
        protocol,
        baseline.attempts,
        baseline.choice,
        max(baseline.pass_wall_seconds) > 900 or baseline.wall_seconds > 2700,
    )
    if source_files() != sources:
        raise ValueError("Driver source changed during the actual physical study")
    report = StudyReport(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        status=assessment.status,
        reason=assessment.reason,
        runtime=InstalledRuntime.capture(),
        driver_runtime=SelectedInstalledRuntime.capture(extras=("cli",)),
        source_files=sources,
        attempts=baseline.attempts,
        reserved_calls=sum(a.reserved_calls for a in baseline.attempts),
        known_attempted_calls=sum(a.attempted_calls or 0 for a in baseline.attempts),
        unknown_attempts=sum(a.status != "not_started" and a.attempted_calls is None for a in baseline.attempts),
        oracle=assessment.oracle,
        oracle_wall_seconds=assessment.wall_seconds,
        pass_wall_seconds=baseline.pass_wall_seconds,
        baseline_wall_seconds=baseline.wall_seconds,
        development_choice=baseline.choice_artifact,
        gate=assessment.gate,
    )
    publish(root, "complete.json" if assessment.status == "completed" else "limited.json", report.to_json())
    return report


def main() -> None:
    """Invoke the optional finite Click driver with fixed original resource caps.

    Notes
    -----
    Requires ``coleman[cli]``; importing the mathematical/core API does not
    import Click. Output roots are trusted local paths. No rate, workload,
    repetition or deadline override is offered by this locally frozen driver.
    """
    try:
        click = import_module("click")
    except ImportError as error:
        raise SystemExit("Install coleman[cli] to run this optional research command") from error

    def invoke(output, family, repetition, independent):
        protocol = load_protocol(Path(__file__).with_name("expert-rate-entry-protocol-v2.json"))
        try:
            if independent:
                if family is not None:
                    raise click.UsageError("--oracle and --family are separate declared process roles")
                publish(output, "oracle.json", oracle(output, protocol).to_json())
            elif family is not None:
                _child(protocol, family, repetition, output)
            else:
                run_study(output)
        except ResourceLimitError as error:
            click.echo(str(error), err=True)
            raise SystemExit(75) from error

    command = click.Command(
        "expert-rate-entry",
        callback=invoke,
        params=[
            click.Option(("--output",), type=click.Path(path_type=Path), required=True),
            click.Option(("--family",), type=click.Choice(ORDER)),
            click.Option(("--repetition",), type=click.IntRange(0, 2), default=0),
            click.Option(("--oracle", "independent"), is_flag=True),
        ],
    )
    command.main()


if __name__ == "__main__":
    main()
