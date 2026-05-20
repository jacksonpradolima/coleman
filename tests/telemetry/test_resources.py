"""Tests for process resource tracking helpers."""

import os
import time
from unittest.mock import patch

import pytest

from coleman.telemetry.resources import (
    ProcessResourceTracker,
    ResourceSnapshot,
    _get_current_rss_mib,
    _get_peak_rss_mib,
)


def test_process_resource_tracker_returns_monotonic_samples():
    tracker = ProcessResourceTracker()

    first = tracker.sample()
    time.sleep(0.01)
    second = tracker.sample()

    assert second.wall_time_seconds >= first.wall_time_seconds
    assert second.cpu_time_seconds >= first.cpu_time_seconds
    assert second.current_rss_mib is None or second.current_rss_mib >= 0
    assert second.peak_rss_mib is None or second.peak_rss_mib >= 0


def test_process_resource_tracker_init_zero_times():
    """Immediately after construction wall/cpu times start at 0."""
    tracker = ProcessResourceTracker()
    snap = tracker.sample()
    assert snap.wall_time_seconds >= 0.0
    assert snap.cpu_time_seconds >= 0.0


def test_process_resource_tracker_snapshot_type():
    tracker = ProcessResourceTracker()
    snap = tracker.sample()
    assert isinstance(snap, ResourceSnapshot)


def test_process_resource_tracker_cpu_utilization_zero_delta():
    """CPU utilization is None when wall_time delta is zero."""
    tracker = ProcessResourceTracker()
    # Force last_wall == now by patching perf_counter to return the same value
    with (
        patch("coleman.telemetry.resources.time.perf_counter", return_value=tracker._start_wall),
        patch("coleman.telemetry.resources.time.process_time", return_value=tracker._start_cpu),
    ):
        snap = tracker.sample()
    assert snap.cpu_utilization_percent is None


def test_process_resource_tracker_cpu_utilization_positive():
    """CPU utilization is non-negative given normal time deltas."""
    tracker = ProcessResourceTracker()
    snap = tracker.sample()
    if snap.cpu_utilization_percent is not None:
        assert snap.cpu_utilization_percent >= 0.0


def test_get_current_rss_mib_on_linux():
    """/proc/self/statm-based RSS should return a positive float on Linux."""
    rss = _get_current_rss_mib()
    if os.path.exists("/proc/self/statm"):
        assert rss is not None
        assert rss > 0
    else:
        assert rss is None


def test_get_current_rss_mib_missing_file():
    """Returns None when /proc/self/statm is unavailable."""
    with patch("coleman.telemetry.resources.os.path.exists", return_value=False):
        assert _get_current_rss_mib() is None


def test_get_current_rss_mib_bad_content():
    """Returns None when file content cannot be parsed."""
    with patch("builtins.open", side_effect=OSError("no perm")):
        result = _get_current_rss_mib()
    assert result is None


def test_get_peak_rss_mib_returns_positive_or_none():
    """Peak RSS is either a positive float or None (no resource module)."""
    rss = _get_peak_rss_mib()
    assert rss is None or rss > 0


def test_get_peak_rss_mib_none_when_no_resource_module():
    """If the resource module is absent, _get_peak_rss_mib returns None."""
    with patch("coleman.telemetry.resources.resource_module", None):
        assert _get_peak_rss_mib() is None


def test_get_peak_rss_mib_darwin_divides_by_megabyte():
    """MacOS path (sys.platform == 'darwin') divides ru_maxrss by 1024*1024."""
    from unittest.mock import MagicMock

    fake_resource = MagicMock()
    fake_resource.getrusage.return_value.ru_maxrss = 1024 * 1024 * 128  # 128 MiB in bytes

    with (
        patch("coleman.telemetry.resources.resource_module", fake_resource),
        patch("coleman.telemetry.resources.sys") as mock_sys,
    ):
        mock_sys.platform = "darwin"
        result = _get_peak_rss_mib()

    assert result == pytest.approx(128.0)


def test_get_current_rss_mib_ioerror_on_read():
    """Returns None when the /proc/self/statm file exists but reading raises OSError."""
    with (
        patch("coleman.telemetry.resources.os.path.exists", return_value=True),
        patch("builtins.open", side_effect=OSError("permission denied")),
    ):
        result = _get_current_rss_mib()
    assert result is None


def test_get_peak_rss_mib_returns_none_on_getrusage_error():
    """Cover _get_peak_rss_mib except branch (lines 94-95)."""
    from unittest.mock import MagicMock

    fake_resource = MagicMock()
    fake_resource.RUSAGE_SELF = 0
    fake_resource.getrusage.side_effect = OSError("boom")

    with patch("coleman.telemetry.resources.resource_module", fake_resource):
        assert _get_peak_rss_mib() is None


def test_get_current_rss_mib_returns_none_for_short_statm_fields():
    """Cover len(fields) < 2 branch (line 78)."""
    from unittest.mock import mock_open

    with (
        patch("coleman.telemetry.resources.os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="123")),
    ):
        assert _get_current_rss_mib() is None
