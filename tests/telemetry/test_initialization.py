"""Real SDK partial construction must not outlive a no-op fallback."""

import threading
from unittest.mock import patch

import pytest

from coleman.telemetry import otel


@pytest.mark.parametrize("failure", ("OTLPSpanExporter", "BatchSpanProcessor", "TracerProvider"))
def test_real_sdk_trace_construction_failure_disposes_acquired_threads_and_defers_registration(failure):
    from opentelemetry import metrics, trace

    assert otel._HAS_OTEL
    meter_before, tracer_before = metrics.get_meter_provider(), trace.get_tracer_provider()
    threads_before = {thread.ident for thread in threading.enumerate()}
    # Only the selected constructor fails. Real metrics/reader/exporter and
    # prior trace resources are constructed and shut down by production code.
    with patch.object(otel, failure, side_effect=RuntimeError("controlled partial SDK construction failure")):
        result = otel.get_telemetry(enabled=True, export_interval_millis=60000)
    assert isinstance(result, otel.NoOpTelemetry)
    assert metrics.get_meter_provider() is meter_before
    assert trace.get_tracer_provider() is tracer_before
    assert {thread.ident for thread in threading.enumerate()} == threads_before


def test_cleanup_attempts_all_owned_resources_without_masking_original_failure(caplog):
    calls = []

    class Shutdown:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def shutdown(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("controlled cleanup failure")

    otel._dispose_partial(None, Shutdown("first", True), Shutdown("second"))
    assert calls == ["first", "second"]
    assert "controlled cleanup failure" in caplog.text
