"""Measure bounded local lifecycle overhead with isolated actual learner runs.

Run copied files against a noneditable wheel: python measure.py --output results.
Every case runs off/noop/real SDK memory profiles in fresh processes; no network
exporter or statistical superiority claim is involved. ``--quick`` selects one
small workload. No report is published if any child fails or changes the math.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.core.base import CoreContract
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.telemetry.resources import ProcessResourceTracker


def _example():
    """Load only the explicitly copied, source-bound public sibling example."""
    name = "coleman_operational_ci_example"
    specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name("ci.py"))
    if specification is None or specification.loader is None:
        raise ImportError("The consciously copied ci.py source is required")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class Observation(CoreContract):
    """Record own-executable measurements separately from authored CI losses."""

    experts: Annotated[int, Field(strict=True, ge=2, le=32)]
    panels: Annotated[int, Field(strict=True, ge=1, le=32)]
    trajectories: Annotated[int, Field(strict=True, ge=1, le=4)]
    mode: Literal["off", "noop", "memory"]
    elapsed_seconds: Annotated[Binary64, Field(ge=0)]
    cpu_seconds: Annotated[Binary64, Field(ge=0)]
    peak_rss_mib: Annotated[Binary64, Field(ge=0)] | None
    current_rss_mib: Annotated[Binary64, Field(ge=0)] | None
    parent_elapsed_seconds: Annotated[Binary64, Field(ge=0)]
    accepted_updates: Annotated[int, Field(strict=True, ge=1, le=128)]
    updates_per_second: Annotated[Binary64, Field(ge=0)] | None
    mathematical_output: Fingerprint
    snapshot_bytes: Annotated[int, Field(strict=True, ge=0, le=1048576)]

    @model_validator(mode="after")
    def _throughput(self):
        """Keep accepted work and unavailable zero-duration throughput explicit."""
        if self.accepted_updates != self.panels * self.trajectories:
            raise ValueError("Observed update count differs from the completed workload")
        expected = self.accepted_updates / self.elapsed_seconds if self.elapsed_seconds else None
        if self.updates_per_second != expected:
            raise ValueError("Throughput must use actual completed work and elapsed seconds")
        return self


class Report(CoreContract):
    """Keep finite benchmark evidence and separately selected tool identities.

    Notes
    -----
    ``driver_runtime`` records the command's optional CLI closure, independently
    of the telemetry SDK being measured. Its absence means the old report did
    not record that identity; it is omitted from JSON so v1 bytes/digests do
    not change. Reading never invents or captures a missing runtime.
    """

    complete: Literal[True]
    profile: Literal["quick", "matrix"]
    runtime: InstalledRuntime
    driver_runtime: SelectedInstalledRuntime | None = Field(default=None, exclude_if=lambda value: value is None)
    instrumentation_runtime: SelectedInstalledRuntime
    hardware: str
    native_threads: tuple[tuple[str, str], ...]
    driver_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    example_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    measurements: Annotated[tuple[Observation, ...], Field(min_length=3, max_length=24)]

    @model_validator(mode="after")
    def _driver_identity(self):
        """Keep optional command tooling bound to the same exact base image."""
        driver = self.driver_runtime
        if driver is not None and (
            driver.selected_extras != ("cli",)
            or not any(item.name == "click" and item.version == "8.1.8" for item in driver.distributions)
            or any(
                getattr(self.runtime, key) != getattr(driver, key)
                for key in (
                    "package_fingerprint",
                    "python_implementation",
                    "python_version",
                    "operating_system",
                    "machine",
                )
            )
            or not {(item.name, item.version) for item in self.runtime.distributions}
            <= {(item.name, item.version) for item in driver.distributions}
        ):
            raise ValueError("CLI driver identity must select cli and preserve the exact base runtime")
        return self

    @model_validator(mode="after")
    def _complete(self):
        """Reject missing/duplicate mode cells and unequal mathematical outputs."""
        selected = self.instrumentation_runtime
        if selected.selected_extras != ("telemetry",) or not {
            "opentelemetry-api",
            "opentelemetry-sdk",
            "opentelemetry-exporter-otlp-proto-http",
        } <= {d.name for d in selected.distributions}:
            raise ValueError("Operational reports require the actual selected telemetry dependency identity")
        if any(
            getattr(self.runtime, key) != getattr(selected, key)
            for key in ("package_fingerprint", "python_implementation", "python_version", "operating_system", "machine")
        ):
            raise ValueError("Base and instrumentation identities must describe the same actual runtime")
        if not {(d.name, d.version) for d in self.runtime.distributions} <= {
            (d.name, d.version) for d in selected.distributions
        }:
            raise ValueError("Selected telemetry identity must preserve every exact base dependency version")
        cases = _cases(self.profile == "quick")
        expected = tuple((*case, mode) for case in cases for mode in ("off", "noop", "memory"))
        actual = tuple((r.experts, r.panels, r.trajectories, r.mode) for r in self.measurements)
        if actual != expected:
            raise ValueError("Complete operational reports require every exact ordered case/mode")
        for start in range(0, len(self.measurements), 3):
            if len({r.mathematical_output for r in self.measurements[start : start + 3]}) != 1:
                raise ValueError("Instrumentation must retain identical mathematical output")
        return self


def _cases(quick: bool) -> tuple[tuple[int, int, int], ...]:
    """Return the fixed public workload grid, not an adaptive benchmark choice."""
    return (
        ((2, 2, 1),)
        if quick
        else tuple(
            (experts, panels, trajectories) for experts in (2, 16) for panels in (2, 8) for trajectories in (1, 2)
        )
    )


def child(mode: Literal["off", "noop", "memory"], experts: int, panels: int, trajectories: int) -> Observation:
    """Measure real learner work after imports; lifetime peak includes imports.

    Parameters
    ----------
    mode : {'off', 'noop', 'memory'}
        Explicit local instrumentation profile; memory requires the optional SDK.
    experts, panels, trajectories : int
        Existing example caps:2..32 experts,1..32 panels and1..4 independent
        learner objects. The authored loss tape is not independent histories.

    Returns
    -------
    Observation
        Process seconds/MiB and actual complete work/throughput. The parent
        clock is initially zero, filled only by the actual subprocess caller.

    Raises
    ------
    ValueError, ImportError, RuntimeError
        Invalid workload/mode, missing SDK or numerical/measurement failure.

    Notes
    -----
    Reads own process clocks/resources and executes the existing learner.
    Creates no worker service, network exporter or persistent artifact.
    """
    example = _example()
    workload = example.Workload(schema_version=1, experts=experts, panels=panels, trajectories=trajectories)
    tracker = ProcessResourceTracker()
    result = example.run(workload, mode)
    measured = tracker.sample()
    return Observation(
        schema_version=1,
        experts=experts,
        panels=panels,
        trajectories=trajectories,
        mode=mode,
        elapsed_seconds=measured.wall_time_seconds,
        cpu_seconds=measured.cpu_time_seconds,
        peak_rss_mib=measured.peak_rss_mib,
        current_rss_mib=measured.current_rss_mib,
        parent_elapsed_seconds=0.0,
        accepted_updates=sum(state.payload.update_count for state in result.final_states),
        updates_per_second=(panels * trajectories / measured.wall_time_seconds) if measured.wall_time_seconds else None,
        mathematical_output=fingerprint([s.model_dump(mode="json") for s in result.final_states]),
        snapshot_bytes=sum(len(s.to_json().encode()) for s in result.snapshots),
    )


def measure(*, quick: bool = False) -> Report:
    """Compare matched finite cases; a parent boot/import clock is separate.

    Parameters
    ----------
    quick : bool, optional
        True selects one two-expert/two-panel/one-trajectory case. Otherwise
        experts2/16, panels2/8 and trajectories1/2 give8 cases and24 subprocesses.

    Returns
    -------
    Report
        All profiles and exact matching mathematical-output digests. Measured
        deltas are framework/instrumentation observations, not CI monetary cost.

    Raises
    ------
    RuntimeError, subprocess.TimeoutExpired, ImportError, ValueError
        A child fails, exceeds its30-second measurement fence, changes math,
        or the actual CLI/telemetry dependency identity is unavailable.

    Notes
    -----
    This API starts Click child commands and therefore requires the CLI extra,
    even when the parent itself is called programmatically. The report captures
    that actual selected closure separately from the telemetry-only identity.
    Cases run sequentially. Fresh processes measure cold process/import cost;
    the inner clock excludes imports and includes instrumentation construction
    and shutdown. Dependency caches are retained; this is not an installation
    benchmark or a representative performance distribution. A coverage/profiler
    run must be identified separately, not promoted to production latency.
    """
    cases = _cases(quick)
    rows = []
    for experts, panels, trajectories in cases:
        expected = None
        for mode in ("off", "noop", "memory"):
            started = time.perf_counter()
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--child",
                    mode,
                    str(experts),
                    str(panels),
                    str(trajectories),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Operational child failed: {result.stdout}\n{result.stderr}")
            row = Observation.from_json(result.stdout)
            row = Observation.model_validate(
                {**row.model_dump(), "parent_elapsed_seconds": time.perf_counter() - started}
            )
            if expected is not None and row.mathematical_output != expected:
                raise RuntimeError("Instrumentation changed the exact mathematical output")
            expected = row.mathematical_output
            rows.append(row)
    return Report(
        schema_version=1,
        complete=True,
        profile="quick" if quick else "matrix",
        runtime=InstalledRuntime.capture(),
        driver_runtime=SelectedInstalledRuntime.capture(extras=("cli",)),
        instrumentation_runtime=SelectedInstalledRuntime.capture(extras=("telemetry",)),
        hardware=(
            f"{platform.platform()};machine={platform.machine()};processor={platform.processor()};cpus={os.cpu_count()}"
        ),
        native_threads=tuple(
            (key, os.environ.get(key, "unset"))
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "POLARS_MAX_THREADS",
            )
        ),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        example_sha256=hashlib.sha256(Path(__file__).with_name("ci.py").read_bytes()).hexdigest(),
        measurements=tuple(rows),
    )


def publish(report: Report, root: Path) -> Path:
    """Publish one immutable complete report using the existing atomic writer.

    Parameters
    ----------
    report : Report
        Complete supported grid with identical matched mathematical outputs.
        Source/runtime fingerprints are content identity, not authentication.
    root : pathlib.Path
        Trusted caller-owned output root; created if absent. This is not an
        untrusted-filesystem containment or arbitrary-path authorization API.

    Returns
    -------
    pathlib.Path
        Content-addressed final JSON, the sole completion authority. Identical
        concurrent writes are idempotent under the existing local writer.

    Raises
    ------
    ValueError, OSError
        Invalid/incomplete/oversized report, conflicting bytes or actual local
        write/fsync/publication failure. No partial JSON is called complete.
    """
    checked = Report.model_validate(report)
    content = checked.to_json()
    if len(content.encode()) > 2097152:
        raise ValueError("Operational report exceeds its declared2MiB output cap")
    path = root / (checked.fingerprint.removeprefix("sha256:") + ".json")
    ArtifactWriter(root).write_immutable(path, content)
    return path


def main() -> None:
    """Run one bounded suite through the optional Click command.

    Raises
    ------
    SystemExit
        Help, malformed flags or a missing CLI extra exit before workload or
        publication; missing tooling includes explicit installation guidance.

    Notes
    -----
    Imports Click only on invocation. The four-value child option retains the
    original30-second process fence; the parent records its selected CLI
    closure separately from the telemetry SDK. Partial child output never
    becomes a complete file. Paths are explicit trusted operator inputs.
    """
    try:
        import click
    except ImportError as error:
        raise SystemExit("Install the reviewed COLEMAN wheel with its [cli] extra") from error

    @click.command(help=__doc__)
    @click.option("--quick", is_flag=True, help="Measure the original single small workload.")
    @click.option("--output", type=click.Path(path_type=Path), default=Path("operational-results"), show_default=True)
    @click.option(
        "--child",
        "child_spec",
        type=(
            click.Choice(("off", "noop", "memory")),
            click.IntRange(2, 32),
            click.IntRange(1, 32),
            click.IntRange(1, 4),
        ),
    )
    def _command(
        quick: bool, output: Path, child_spec: tuple[Literal["off", "noop", "memory"], int, int, int] | None
    ) -> None:
        """Select the original real child or complete finite parent route."""
        if child_spec is not None:
            mode, experts, panels, trajectories = child_spec
            print(child(mode, experts, panels, trajectories).to_json())
        else:
            print(publish(measure(quick=quick), output))

    _command()


if __name__ == "__main__":
    main()
