"""Compatibility entry into the single shared durable execution driver."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from coleman.application import execution_driver
from coleman.application.resource_usage import output_exceeds as _output_exceeds
from coleman.application.resource_usage import peak_rss_mib as _worker_peak_rss_mib
from coleman.application.workflow_store import WorkflowStore


def _read_job(store: WorkflowStore, job_id: str):
    """Retain the original call-time startup-visibility hook for native workers."""
    return execution_driver.read_job(store, job_id)


def _live_read_job(store: WorkflowStore, job_id: str):
    """Resolve the actual legacy read symbol on every call, including startup."""
    return _read_job(store, job_id)


def _live_peak() -> float | None:
    """Resolve the existing own-executable sampler symbol at actual call time."""
    return _worker_peak_rss_mib()


def _live_output(*args, **kwargs) -> bool:
    """Resolve the one shared output walker through its legacy call-time hook."""
    return _output_exceeds(*args, **kwargs)


def run_job(root: Path, job_id: str, *, progress_hook: Callable[[str, int], None] | None = None) -> None:
    """Execute the original native entry through the common job lifecycle.

    Parameters
    ----------
    root : pathlib.Path
        Trusted persistent workspace created by WorkflowStore.
    job_id : str
        Already kernel-owned scheduler identity, never an arbitrary run directory.
    progress_hook : callable or None, optional
        Original native build hook after checkpoint publication, outer metadata
        and resource guard; never accepted through CLI/MCP.

    Notes
    -----
    Delegates exactly once, retaining call-time legacy read/resource hooks and
    unchanged native Trial, report, checkpoint/RNG and error semantics. No work
    loop, scheduler or second resource-guard implementation lives here. Later
    failures leave honest partial evidence; a directory is not completion.
    """
    hooks = execution_driver.WorkerHooks(
        read_job=_live_read_job,
        peak_rss_mib=_live_peak,
        output_exceeds=_live_output,
    )
    execution_driver.run_with_hooks(root, job_id, progress_hook=progress_hook, hooks=hooks)
