"""Shared own-executable memory and bounded output observations for local workers.

Importing this module initializes no worker, domain, transport or telemetry
provider. Guards retain responsibility for interpreting an unavailable sample
and for applying their own declared resource limits.
"""

from __future__ import annotations

import stat
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType


def peak_rss_mib(*, _path_factory: Callable[[str], Path] | None = None, _platform: str | None = None) -> float | None:
    """Observe this executable's peak resident memory in MiB.

    Parameters
    ----------
    _path_factory : callable or None, optional
        Private compatibility hook for the original worker's call-time Path
        binding. None selects this module's current Path binding. It is called
        only on Linux, inside the original kernel-read error boundary.
    _platform : str or None, optional
        Private compatibility hook for the original worker's platform binding.
        None selects the current sys.platform; no platform is inferred from a
        saved result or an unavailable measurement.

    Returns
    -------
    float or None
        Peak resident memory in MiB (1,048,576 bytes), including observed zero.
        Linux's VmHWM kB field uses 1,024-byte units and is divided by 1,024.
        Missing, malformed, negative or unreadable Linux metadata returns None.
        Other platforms retain the existing telemetry tracker's peak value,
        including its unavailable result.

    Raises
    ------
    TypeError, OverflowError, ImportError
        Unexpected factory/numeric failure or unavailable telemetry code.
        Expected Linux OSError and ValueError failures instead return None;
        unexpected errors from the unchanged non-Linux tracker propagate.

    Notes
    -----
    Linux reads /proc/self/status rather than inherited pre-exec getrusage
    peaks. Non-Linux calls lazily construct the existing ProcessResourceTracker,
    including its normal clock and process-metadata reads. Nothing is written
    or executed externally. This sampled observation is neither a hard memory
    reservation nor a hostile-code sandbox; callers must refuse unknown memory
    when their execution contract requires a known bounded peak.
    """
    platform_name = sys.platform if _platform is None else _platform
    if platform_name != "linux":
        from coleman.telemetry.resources import ProcessResourceTracker

        return ProcessResourceTracker().sample().peak_rss_mib
    path_factory = Path if _path_factory is None else _path_factory
    try:
        with path_factory("/proc/self/status").open(encoding="ascii") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB":
                        return None
                    value = int(fields[1])
                    return value / 1024 if value >= 0 else None
    except (OSError, ValueError):
        return None
    return None


def output_exceeds(
    maximum_bytes: int,
    *,
    root: Path = Path("run"),
    maximum_entries: int = 128,
    _stat: ModuleType | None = None,
) -> bool:
    """Check sampled descendant entries and regular-file bytes under a trusted root.

    Parameters
    ----------
    maximum_bytes : int
        Declared maximum aggregate regular-file bytes. Equality is permitted;
        this helper does not introduce validation or change existing cap values.
    root : pathlib.Path, optional
        Trusted output directory, default the relative path run. Relative paths
        are interpreted against the caller's current working directory.
    maximum_entries : int, optional
        Maximum enumerated descendants, default 128. Directories, symlinks and
        entries disappearing before lstat still consume their enumeration slot.
    _stat : types.ModuleType or None, optional
        Private compatibility hook for the original worker's call-time stat
        binding. None selects this module's current stat binding; mode predicates
        are looked up only at their original points in the walk.

    Returns
    -------
    bool
        True for an exceeded entry/byte ceiling or any enumerated symlink.
        False means the sampled traversal did not encounter these refusals.
        A directory is not declared complete or protected against later writes.

    Raises
    ------
    OSError
        Traversal or metadata failure other than an entry disappearing before
        lstat. Callers retain their existing fail-closed refusal handling.
    TypeError, AttributeError
        Unsupported trusted root, cap or mode-provider input. No coercion or
        implicit default replacement is performed.

    Notes
    -----
    Streams the existing rglob traversal without retaining an unbounded listing.
    Uses lstat and refuses descendant symlinks before counting regular-file
    bytes; an absent temporary entry is skipped after its slot is counted.
    Performs metadata reads only. The host must provide a trusted root; this is
    neither a hostile-filesystem sandbox nor an atomic snapshot or disk quota.
    The original worker's run/128 defaults and strict-greater-than boundaries
    are preserved, including their existing behavior for raw unvalidated caps.
    """
    modes = stat if _stat is None else _stat
    total = 0
    for count, path in enumerate(root.rglob("*"), start=1):
        if count > maximum_entries:
            return True
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if modes.S_ISLNK(metadata.st_mode):
            return True
        if modes.S_ISREG(metadata.st_mode):
            total += metadata.st_size
        if total > maximum_bytes:
            return True
    return False
