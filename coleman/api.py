"""
coleman.api - Library-first public API.

External projects can install a COLEMAN GitHub release wheel and drive experiments
programmatically via :func:`run`, :func:`run_many`, or :func:`sweep`.

Functions
---------
run
    Execute a single experiment from a resolved ``RunSpec``.
run_many
    Execute multiple specs, optionally in parallel.
sweep
    Expand a base spec × sweep definition and return concrete specs.
load_spec
    Load and validate a ``RunSpec`` from YAML (with pack resolution).
save_resolved
    Persist a resolved spec as canonical JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from coleman.capabilities import require_capability
from coleman.spec.io import load_spec, save_resolved
from coleman.spec.run_id import compute_run_id
from coleman.spec.sweep import expand_sweep

if TYPE_CHECKING:
    from coleman.runner import RunnerExtension
    from coleman.spec.models import RunSpec
    from coleman.spec.sweep import SweepSpec


class RunResult:
    """Lightweight container for the outcome of a single run.

    Parameters
    ----------
    run_id : str
        Deterministic run identifier.
    spec : RunSpec
        The resolved spec that was executed.
    metrics : dict[str, Any]
        Summary metrics collected during the run.
    artifacts_dir : str | None
        Path to the run's artefact directory, if any.
    """

    def __init__(
        self,
        *,
        run_id: str,
        spec: RunSpec,
        metrics: dict[str, Any] | None = None,
        artifacts_dir: str | None = None,
    ) -> None:
        """Initialise a RunResult.

        Parameters
        ----------
        run_id : str
            Deterministic run identifier.
        spec : RunSpec
            The resolved spec that was executed.
        metrics : dict[str, Any] or None
            Summary metrics.  Defaults to empty dict.
        artifacts_dir : str or None
            Path to the run artefact directory, if any.
        """
        self.run_id = run_id
        self.spec = spec
        self.metrics = metrics or {}
        self.artifacts_dir = artifacts_dir

    def __repr__(self) -> str:
        """Return a concise string representation of the result."""
        return f"RunResult(run_id={self.run_id!r}, metrics_keys={list(self.metrics.keys())})"


def run(spec: RunSpec) -> RunResult:
    """Execute a native TCPCI replay from a resolved specification.

    Parameters
    ----------
    spec : RunSpec
        Validated configuration, including dataset paths, policies, rewards,
        budgets and optional checkpoint, result and telemetry settings.

    Returns
    -------
    RunResult
        Resolved specification, configuration-derived run ID and artifact
        directory. ``metrics`` is currently empty; inspect native result files
        for per-build observations.

    Raises
    ------
    OSError
        If configuration/provenance cannot be persisted or execution encounters
        an unsuppressed filesystem error.

    Notes
    -----
    Writes resolved configuration and provenance before execution beneath
    ``spec.results.out_dir / compute_run_id(spec)``. Native results and enabled
    checkpoints use that run directory; an enabled manifest is generated afterward.
    A directory's existence does not certify a successful run.

    Resume requires compatible input bytes, runtime and configuration. The run ID
    hashes configuration with legacy normalization rules; it does not hash all
    external dataset bytes. The runner replays recorded outcomes rather than
    launching the subject's test commands. Execution errors propagate according
    to the native runner's error-handling contracts.

    See Also
    --------
    run_many : Execute multiple resolved specifications.
    run_with_extension : Use native lifecycle callbacks.
    """
    require_capability("ci")

    from coleman.manifest import generate_manifest
    from coleman.runner import run_experiment
    from coleman.spec.provenance import save_provenance

    rid = compute_run_id(spec)

    # Persist resolved spec + provenance.
    run_dir = Path(spec.results.out_dir) / rid
    save_resolved(spec, run_dir / "spec.resolved.json", redact_sensitive=True)
    save_provenance(run_dir, redact_sensitive=True)

    # Execute the experiment with per-run output paths so results and
    # checkpoints do not collide across different run ids or sweeps.
    execution_spec = spec.model_dump()
    execution_spec["_run_id"] = rid
    execution_spec["results"]["out_dir"] = str(run_dir / "results")
    checkpoint_cfg = execution_spec.get("checkpoint")
    if isinstance(checkpoint_cfg, dict) and "base_dir" in checkpoint_cfg:
        checkpoint_cfg["base_dir"] = str(run_dir / "checkpoints")
    run_experiment(execution_spec)

    if spec.results.manifest_enabled:
        generate_manifest(run_dir, run_id=rid)

    return RunResult(run_id=rid, spec=spec, artifacts_dir=str(run_dir))


def run_with_extension(spec: RunSpec, extension: RunnerExtension) -> RunResult:
    """Execute a native replay with extension lifecycle callbacks.

    Parameters
    ----------
    spec : RunSpec
        Validated native experiment configuration; budgets retain their declared
        ratio, seconds or test-count semantics.
    extension : RunnerExtension
        Runner extension receiving native lifecycle callbacks. Worker callbacks
        execute in their owning process and must respect its state boundaries.

    Returns
    -------
    RunResult
        Configuration-derived run identity and artifact directory. Native result
        files contain observations; this container does not aggregate metrics.

    Raises
    ------
    OSError
        If run configuration, provenance or other unsuppressed file I/O fails.

    Notes
    -----
    Has the same configuration/provenance, output, checkpoint and optional
    manifest side effects as ``run``. Extension errors follow the existing
    runner callback policy. Callbacks supplement orchestration; they do not
    replace its scheduling or checkpoint implementation.

    See Also
    --------
    run : Execute without an explicitly supplied runner extension.
    """
    require_capability("ci")

    from coleman.manifest import generate_manifest
    from coleman.runner import run_experiment_with_extension
    from coleman.spec.provenance import save_provenance

    rid = compute_run_id(spec)

    run_dir = Path(spec.results.out_dir) / rid
    save_resolved(spec, run_dir / "spec.resolved.json", redact_sensitive=True)
    save_provenance(run_dir, redact_sensitive=True)

    execution_spec = spec.model_dump()
    execution_spec["_run_id"] = rid
    execution_spec["results"]["out_dir"] = str(run_dir / "results")
    checkpoint_cfg = execution_spec.get("checkpoint")
    if isinstance(checkpoint_cfg, dict) and "base_dir" in checkpoint_cfg:
        checkpoint_cfg["base_dir"] = str(run_dir / "checkpoints")

    run_experiment_with_extension(execution_spec, extension)

    if spec.results.manifest_enabled:
        generate_manifest(run_dir, run_id=rid)

    return RunResult(run_id=rid, spec=spec, artifacts_dir=str(run_dir))


def run_many(
    specs: list[RunSpec],
    *,
    max_workers: int = 1,
) -> list[RunResult]:
    """Execute resolved specifications sequentially or in worker processes.

    Parameters
    ----------
    specs : list[RunSpec]
        Ordered configurations. Each run retains its own declared data, budgets
        and output/checkpoint namespace.
    max_workers : int, optional
        Number of specification-level workers. Values at most one select the
        sequential path; values greater than one create a process pool. Native
        execution can independently create its configured workers.

    Returns
    -------
    list[RunResult]
        Results in input order after all executions succeed.

    Raises
    ------
    ValueError
        If duplicate run IDs are detected before parallel execution. Distinct
        specifications must not race to write the same run directory.

    Notes
    -----
    Each child has ``run``'s filesystem and execution side effects. An error
    propagates rather than returning a partial result list; other workers may
    already have written artifacts. Sequential duplicate IDs are permitted and
    follow native continuation behavior. Worker count is orchestration, not
    physical parallel test-execution evidence.

    See Also
    --------
    run : Execute one specification.
    sweep : Expand a candidate specification grid.
    """
    if specs:
        require_capability("ci")

    if max_workers > 1:
        ids = [compute_run_id(s) for s in specs]
        if len(set(ids)) != len(ids):
            msg = (
                "Duplicate run_id values detected among specs. "
                "Parallel execution would cause racy writes to the same "
                "run directory. Deduplicate specs or run sequentially."
            )
            raise ValueError(msg)

        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(run, specs))

    return [run(s) for s in specs]


def sweep(base: RunSpec, sweep_spec: SweepSpec) -> list[RunSpec]:
    """Expand a specification and sweep declaration without executing runs.

    Parameters
    ----------
    base : RunSpec
        Validated template. Expansion constructs new validated specifications
        without mutating this object.
    sweep_spec : SweepSpec
        Grid/zip axes and optional seed repetitions. Zip lists must have equal
        lengths; grid axes form a Cartesian product.

    Returns
    -------
    list[RunSpec]
        Concrete specifications in deterministic expansion order.

    Raises
    ------
    ValueError
        If an axis or dotted parameter path is invalid, or expanded values fail
        specification validation.

    Notes
    -----
    Performs no dataset execution or artifact writes. Expansion materializes
    its output list, so callers must bound the candidate grid. Seeds reproduce
    algorithmic randomness; they are not independent source observations.

    See Also
    --------
    run_many : Execute the resulting specifications.
    """
    return expand_sweep(base, sweep_spec)


__all__ = [
    "RunResult",
    "load_spec",
    "run",
    "run_many",
    "run_with_extension",
    "save_resolved",
    "sweep",
]
