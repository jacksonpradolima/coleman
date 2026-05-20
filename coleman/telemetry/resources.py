"""Helpers for sampling process resource usage during experiments."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from types import ModuleType

try:
    import resource as _resource_module
except ImportError:  # pragma: no cover - not expected on Linux, but keeps portability
    resource_module: ModuleType | None = None
else:
    resource_module = _resource_module


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time process resource sample."""

    current_rss_mib: float | None
    peak_rss_mib: float | None
    cpu_time_seconds: float
    wall_time_seconds: float
    cpu_utilization_percent: float | None


class ProcessResourceTracker:
    """Track CPU and memory usage for the current process.

    Notes
    -----
    * Current RSS is read from ``/proc/self/statm`` when available.
    * Peak RSS uses ``resource.getrusage``.
    * CPU utilization is computed over the interval between consecutive samples.
    """

    def __init__(self) -> None:
        """Initialise the tracker and record the start wall and CPU times."""
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()
        self._last_wall = self._start_wall
        self._last_cpu = self._start_cpu

    def sample(self) -> ResourceSnapshot:
        """Collect one resource sample for the current process."""
        now_wall = time.perf_counter()
        now_cpu = time.process_time()

        delta_wall = now_wall - self._last_wall
        delta_cpu = now_cpu - self._last_cpu
        cpu_utilization_percent = (delta_cpu / delta_wall) * 100 if delta_wall > 0 else None

        self._last_wall = now_wall
        self._last_cpu = now_cpu

        return ResourceSnapshot(
            current_rss_mib=_get_current_rss_mib(),
            peak_rss_mib=_get_peak_rss_mib(),
            cpu_time_seconds=now_cpu - self._start_cpu,
            wall_time_seconds=now_wall - self._start_wall,
            cpu_utilization_percent=cpu_utilization_percent,
        )


def _get_current_rss_mib() -> float | None:
    """Return the current resident set size in MiB when available."""
    statm_path = "/proc/self/statm"
    if not os.path.exists(statm_path):
        return None

    try:
        with open(statm_path, encoding="utf-8") as statm_file:
            fields = statm_file.read().split()
        if len(fields) < 2:
            return None

        resident_pages = int(fields[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (resident_pages * page_size) / (1024 * 1024)
    except (OSError, ValueError):
        return None


def _get_peak_rss_mib() -> float | None:
    """Return the peak resident set size in MiB when available."""
    if resource_module is None:
        return None

    try:
        peak_rss = resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None

    if sys.platform == "darwin":
        return peak_rss / (1024 * 1024)
    return peak_rss / 1024
