"""Kernel-backed local ownership shared by durable workflow clients and workers."""

from __future__ import annotations

from pathlib import Path

from coleman.application.baseline_lock import BaselineBusyError, baseline_lock


def worker_active(root: Path, job_id: str) -> bool:
    """Probe a retained execution lock without using a PID as proof of ownership.

    Parameters
    ----------
    root : pathlib.Path
        Trusted workspace root, on supported POSIX local storage.
    job_id : str
        Validated, host-issued logical job ID.

    Returns
    -------
    bool
        True only while a process holds the exact retained inode's kernel lock.
        A false snapshot does not authorize execution; scheduling must acquire
        and retain that same lock across its transactional transition and spawn.

    Raises
    ------
    ValueError, OSError
        Unsafe or unavailable lock storage; never guessed inactive.
    """
    try:
        with baseline_lock(root / "locks" / f"{job_id}.lock"):
            return False
    except BaselineBusyError:
        return True
