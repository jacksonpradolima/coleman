"""
coleman.api - Library-first public API.

External projects can ``pip install coleman`` and drive experiments
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

from coleman.spec.io import load_spec, save_resolved
from coleman.spec.run_id import compute_run_id
from coleman.spec.sweep import expand_sweep

if TYPE_CHECKING:
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
    """Execute a single experiment from a resolved *spec*.

    This is the canonical evaluation entry-point.  Both the CLI and
    library callers route through here.

    Parameters
    ----------
    spec : RunSpec
        Fully resolved run specification.

    Returns
    -------
    RunResult
        Outcome container with deterministic ``run_id`` and summary
        metrics.
    """
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


def run_with_extension(spec: RunSpec, extension: Any) -> RunResult:
    """Execute one run using runner extension callbacks without replacing orchestration."""
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
    """Execute multiple specs, optionally in parallel.

    Parameters
    ----------
    specs : list[RunSpec]
        List of resolved run specifications.
    max_workers : int
        Maximum number of parallel workers (``1`` = sequential).

    Returns
    -------
    list[RunResult]
        Results in the same order as *specs*.

    Raises
    ------
    ValueError
        If duplicate ``run_id`` values are detected when
        ``max_workers > 1`` (parallel writes to the same directory
        would cause data corruption).
    """
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
    """Expand *base* × *sweep_spec* into concrete specs.

    Parameters
    ----------
    base : RunSpec
        Template specification.
    sweep_spec : SweepSpec
        Sweep definition (axes + optional seeds).

    Returns
    -------
    list[RunSpec]
        Expanded specs in deterministic order.
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
