"""Local attempt serialization for matrix orchestration and its existing native workers."""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path

from coleman.corpus.materialization import validate_source_storage


class BaselineBusyError(RuntimeError):
    """A live local process owns the same matrix or native execution; retry after it finishes."""


@contextmanager
def baseline_lock(path: Path, *, handoff: bool = False):
    """Hold a nonblocking local POSIX lock, optionally handing ownership to a child.

    Parameters
    ----------
    path : pathlib.Path
        Trusted retained lock file; symlinks, nonregular or multiply linked files
        reject. Callers must serialize their own threads and use local storage.
    handoff : bool
        If true, close without an explicit unlock at context exit. A fixed trusted
        subprocess may retain the yielded descriptor through explicit pass_fds;
        kernel ownership then survives parent death until its last copy closes.
        Default false preserves existing matrix/worker unlock semantics.

    Yields
    ------
    int
        Close-on-exec descriptor, inherited only when a trusted caller explicitly
        requests it. The parent must retain this context until spawn completes.

    Raises
    ------
    BaselineBusyError
        A live local process already owns this execution; no work is completed.
    ValueError, OSError
        Unsafe/unsupported local storage or lock/publication I/O failure.

    Notes
    -----
    Retained files avoid inode replacement races. This is neither a distributed
    lock nor protection against a hostile filesystem. PIDs never establish
    ownership. Existing matrix children independently hold native execution locks
    so a killed parent cannot make their running work look abandoned.
    """
    import fcntl

    validate_source_storage(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Baseline lock requires one regular local file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BaselineBusyError("Baseline execution is already active; retry after its owner finishes") from error
        try:
            yield descriptor
        finally:
            if not handoff:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
