"""Private bounded process adapter invoking existing native services, never a second runner."""

from __future__ import annotations

import base64
import hashlib
import os
import stat
import sys
import threading
from importlib.metadata import version
from pathlib import Path

from coleman.application import resource_usage
from coleman.application.contracts import ExperimentResult, Limits, PortableFile, PortableNativeProvenance, ResultBundle
from coleman.application.registry import Configuration, NativeConfiguration
from coleman.artifacts import ArtifactWriter
from coleman.results.domain import decode_json
from coleman.selection.contracts import Immutable


class WorkItem(Immutable):
    """Private validated host-to-worker snapshot; not accepted by public transports."""

    configuration: Configuration
    limits: Limits


def _worker_peak_rss_mib() -> float | None:
    """Delegate own-executable peak RSS while retaining original call-time hooks.

    Returns
    -------
    float or None
        Original MiB sample, including observed zero and unavailable None.

    Raises
    ------
    TypeError, OverflowError, ImportError
        Unexpected sampling failure, as documented by resource_usage. Expected
        Linux read/parse failures remain unavailable rather than observed zero.

    Notes
    -----
    Passes this module's current Path and sys.platform bindings to the one
    shared sampler. Existing host/test patches remain effective at call time.
    No worker limit, completion state or artifact is changed by sampling.
    """
    return resource_usage.peak_rss_mib(_path_factory=Path, _platform=sys.platform)


def _check_resources(limits: Limits) -> None:
    """Refuse unknown/over-limit own-executable peak RSS or worker artifact bytes synchronously."""
    peak = _worker_peak_rss_mib()
    try:
        refused = peak is None or peak * 1048576 > limits.max_memory_bytes or _output_exceeds(limits.max_output_bytes)
    except OSError:
        refused = True
    if refused:
        os._exit(74)


def _output_exceeds(maximum_bytes: int, *, root: Path = Path("run"), maximum_entries: int = 128) -> bool:
    """Delegate the bounded output walk with the original worker defaults.

    Parameters
    ----------
    maximum_bytes : int
        Aggregate regular-file byte ceiling; equality remains permitted.
    root : pathlib.Path, optional
        Trusted output root, default the relative path run.
    maximum_entries : int, optional
        Enumerated descendant ceiling, default 128, including disappearing entries.

    Returns
    -------
    bool
        Original refusal predicate for bytes, entry count or descendant symlinks.

    Raises
    ------
    OSError
        Non-transient traversal/metadata failure; guards retain fail-closed handling.

    Notes
    -----
    Uses the sole resource_usage walker and this module's current stat binding.
    No output is written, removed or marked complete by this compatibility bridge.
    """
    return resource_usage.output_exceeds(maximum_bytes, root=root, maximum_entries=maximum_entries, _stat=stat)


def _guard(limits: Limits, stopped: threading.Event) -> None:
    """Sample own peak RSS and output bytes; exit without a completed receipt on exhaustion.

    The parent separately enforces wall-clock deadline and cancellation. Sampling
    is a resource guard for trusted native code, with possible transient overshoot.
    Unknown peak RSS fails closed rather than silently disabling the memory cap.
    """
    while not stopped.is_set():
        _check_resources(limits)
        stopped.wait(0.02)


def _experiment(configuration: NativeConfiguration, limits: Limits) -> ExperimentResult:
    """Run native RunSpec unchanged; export native data/spec bytes and explicitly redacted provenance."""
    from coleman.api import run
    from coleman.spec.models import RunSpec

    for item in configuration.inputs:
        path = Path("inputs") / item.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(item.content_base64))
    native = run(RunSpec.model_validate_json(configuration.spec_json))
    root = Path(str(native.artifacts_dir))
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                content = stream.read(limits.max_output_bytes + 1)
            total += len(content)
            if len(files) >= 64 or total > limits.max_output_bytes:
                sys.exit(74)
            if path == root / "provenance.json":
                raw = decode_json(content)
                provenance = PortableNativeProvenance(
                    schema_version=1,
                    source_sha256="sha256:" + hashlib.sha256(content).hexdigest(),
                    producer_version=version("coleman"),
                    python_version=raw["python_version"],
                    platform=raw["platform"],
                )
                content = provenance.to_json().encode("utf-8")
            files.append(
                PortableFile(
                    name=path.relative_to(root).as_posix(),
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                    content_base64=base64.b64encode(content).decode("ascii"),
                )
            )
    return ExperimentResult(
        schema_version=1, run_id=native.run_id, work_units=configuration.work_units, files=tuple(files)
    )


def execute(item: WorkItem) -> ResultBundle:
    """Invoke the native solver/runner with admitted work caps and no transport imports.

    Parameters
    ----------
    item : WorkItem
        Complete validated private host snapshot, including concrete problem
        configuration and finite work, memory and output allowances.

    Returns
    -------
    ResultBundle
        Original typed native result bound to the admitted configuration. Work
        units retain the selected problem's meaning; they are not paid costs.

    Notes
    -----
    The native experiment branch writes relative input and runner artifacts in
    the trusted worker directory. Solver branches retain their existing bounded
    algorithms. The process host separately owns deadline/resource guards and
    complete-last publication; this call publishes no completed-job receipt.
    Native validation and I/O errors propagate without a reset or migration.
    """
    from coleman.allocation import AllocationLimits, AllocationProblem, select_allocation
    from coleman.selection import CoverageProblem, SearchLimits, solve_coverage

    configuration, limits = item.configuration, item.limits
    problem = configuration.problem
    if isinstance(problem, CoverageProblem):
        result = solve_coverage(
            problem,
            method="branch_bound_v1",
            limits=SearchLimits(
                max_work=limits.max_work,
                deadline_seconds=None,
                max_candidates=limits.max_candidates,
                max_working_bytes=min(1073741824, limits.max_memory_bytes),
            ),
        )
    elif isinstance(problem, AllocationProblem):
        result = select_allocation(
            problem,
            limits=AllocationLimits(
                max_work=min(64, limits.max_work),
                deadline_seconds=None,
                max_working_bytes=min(1073741824, limits.max_memory_bytes),
            ),
        )
    else:
        result = _experiment(problem, limits)
    return ResultBundle(
        schema_version=1,
        spec_id=configuration.spec_id,
        configuration_fingerprint=configuration.fingerprint,
        kind=configuration.kind,
        result=result,
    )


def main() -> None:
    """Read the private fixed request file and publish JSON last; exceptions cannot complete a job."""
    item = WorkItem.model_validate_json(Path("request.json").read_bytes())
    _check_resources(item.limits)
    stopped = threading.Event()
    guard = threading.Thread(target=_guard, args=(item.limits, stopped), daemon=True)
    guard.start()
    try:
        bundle = execute(item)
        _check_resources(item.limits)
        content = bundle.to_json()
        encoded_size = len(content.encode("utf-8"))
        if encoded_size > item.limits.max_output_bytes or _output_exceeds(item.limits.max_output_bytes - encoded_size):
            sys.exit(74)
        ArtifactWriter(Path.cwd()).write_immutable(Path("result.json"), content)
    finally:
        stopped.set()
        guard.join(timeout=1)


if __name__ == "__main__":
    main()
