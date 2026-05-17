"""Tests for the telemetry module (no-op adapter and factory)."""

from unittest.mock import patch

from coleman.telemetry.otel import NoOpTelemetry, get_telemetry

# ============================================================================
# NoOpTelemetry
# ============================================================================


class TestNoOpTelemetry:
    def test_record_cycle(self):
        t = NoOpTelemetry()
        t.record_cycle()  # should not raise
        t.record_cycle(attributes={"scenario": "test"})

    def test_run_markers(self):
        t = NoOpTelemetry()
        t.mark_run_started(attributes={"experiment": "1"})
        t.mark_run_finished(attributes={"experiment": "1"})

    def test_record_latency(self):
        t = NoOpTelemetry()
        t.record_latency("bandit_update", 0.5)
        t.record_latency("prioritization", 1.2, attributes={"policy": "UCB"})

    def test_record_fitness(self):
        t = NoOpTelemetry()
        t.record_fitness(0.8, 0.6)
        t.record_fitness(0.9, 0.7, attributes={"scenario": "test"})

    def test_record_resource_snapshot(self):
        t = NoOpTelemetry()
        t.record_resource_snapshot(128.0, 256.0, 75.0)
        t.record_resource_snapshot(None, None, None, attributes={"worker_id": "1"})

    def test_record_experiment_resources(self):
        t = NoOpTelemetry()
        t.record_experiment_resources(12.5, 4.2, 256.0)
        t.record_experiment_resources(12.5, 4.2, None, attributes={"execution_id": "exec-1"})

    def test_span_context_manager(self):
        t = NoOpTelemetry()
        with t.span("test_span") as s:
            assert s is None  # NoOp yields None
        with t.span("test_span", attributes={"step": 1}) as s:
            assert s is None  # NoOp still yields None with attributes


# ============================================================================
# get_telemetry factory
# ============================================================================


class TestGetTelemetry:
    def test_disabled_returns_noop(self):
        t = get_telemetry(enabled=False)
        assert isinstance(t, NoOpTelemetry)

    def test_disabled_with_resource_attributes_returns_noop(self):
        """resource_attributes are accepted but ignored when disabled."""
        t = get_telemetry(enabled=False, resource_attributes={"run_id": "abc123"})
        assert isinstance(t, NoOpTelemetry)

    def test_enabled_without_sdk_returns_noop(self):
        """Even if enabled, without OTel SDK we get NoOp (graceful fallback)."""
        # Since the test environment may or may not have OTel installed,
        # the important thing is that it doesn't crash
        t = get_telemetry(enabled=True)
        # Should be either Telemetry or NoOpTelemetry (both work)
        assert hasattr(t, "record_cycle")
        assert hasattr(t, "record_latency")
        assert hasattr(t, "record_fitness")
        assert hasattr(t, "span")

    def test_enabled_with_resource_attributes_does_not_crash(self):
        """Passing resource_attributes should not crash regardless of SDK availability."""
        t = get_telemetry(
            enabled=True,
            resource_attributes={"execution_id": "test|tr=0.50|exp=1|abc12345", "run_id": "abc123def456"},
        )
        assert hasattr(t, "record_cycle")
        assert hasattr(t, "span")

    def test_interface_consistency(self):
        """Verify NoOpTelemetry has the same public methods as the interface."""
        noop = NoOpTelemetry()
        expected_methods = [
            "record_cycle",
            "record_latency",
            "record_fitness",
            "record_resource_snapshot",
            "record_experiment_resources",
            "mark_run_started",
            "mark_run_finished",
            "flush",
            "span",
        ]
        for method in expected_methods:
            assert hasattr(noop, method)
            assert callable(getattr(noop, method))

    def test_telemetry_raises_importerror_without_otel_sdk(self):
        """Line 114: Telemetry() raises ImportError when OTel SDK is absent."""
        from coleman.telemetry.otel import Telemetry

        with patch("coleman.telemetry.otel._HAS_OTEL", False):
            import pytest

            with pytest.raises(ImportError, match="opentelemetry SDK is required"):
                Telemetry()

    def test_get_telemetry_exception_falls_back_to_noop(self):
        """Lines 415-417: get_telemetry catches Telemetry init exceptions."""
        with (
            patch("coleman.telemetry.otel._HAS_OTEL", True),
            patch("coleman.telemetry.otel.Telemetry.__init__", side_effect=RuntimeError("otel broke")),
        ):
            t = get_telemetry(enabled=True)
        assert isinstance(t, NoOpTelemetry)


def test_telemetry_with_fake_otel_stack_covers_instrument_paths():
    """Execute Telemetry init and methods with stubbed OTel dependencies."""
    from contextlib import contextmanager

    from coleman.telemetry.otel import Telemetry

    class FakeCounter:
        def __init__(self):
            self.events = []

        def add(self, value, attributes=None):
            self.events.append((value, attributes))

    class FakeHistogram:
        def __init__(self):
            self.events = []

        def record(self, value, attributes=None):
            self.events.append((value, attributes))

    class FakeMeter:
        def create_up_down_counter(self, *args, **kwargs):  # noqa: ARG002
            return FakeCounter()

        def create_counter(self, *args, **kwargs):  # noqa: ARG002
            return FakeCounter()

        def create_histogram(self, *args, **kwargs):  # noqa: ARG002
            return FakeHistogram()

    class FakeMeterProvider:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.meter = FakeMeter()

        def get_meter(self, name):  # noqa: ARG002
            return self.meter

        def force_flush(self):
            return None

    class FakeTracer:
        @contextmanager
        def start_as_current_span(self, name, attributes=None):  # noqa: ARG002
            yield None

    class FakeTracerProvider:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.tracer = FakeTracer()

        def add_span_processor(self, processor):  # noqa: ARG002
            return None

        def get_tracer(self, name):  # noqa: ARG002
            return self.tracer

        def force_flush(self):
            return None

    class FakeResource:
        @staticmethod
        def create(attrs):  # noqa: ARG004
            return object()

    class FakeMetricsModule:
        @staticmethod
        def set_meter_provider(provider):  # noqa: ARG004
            return None

    class FakeTraceModule:
        @staticmethod
        def set_tracer_provider(provider):  # noqa: ARG004
            return None

    with (
        patch("coleman.telemetry.otel._HAS_OTEL", True),
        patch("coleman.telemetry.otel.Resource", FakeResource),
        patch("coleman.telemetry.otel.OTLPMetricExporter", lambda endpoint: object()),
        patch(
            "coleman.telemetry.otel.PeriodicExportingMetricReader", lambda exporter, export_interval_millis: object()
        ),
        patch("coleman.telemetry.otel.MeterProvider", FakeMeterProvider),
        patch("coleman.telemetry.otel.otel_metrics", FakeMetricsModule()),
        patch("coleman.telemetry.otel.OTLPSpanExporter", lambda endpoint: object()),
        patch("coleman.telemetry.otel.BatchSpanProcessor", lambda exporter: object()),
        patch("coleman.telemetry.otel.TracerProvider", FakeTracerProvider),
        patch("coleman.telemetry.otel.otel_trace", FakeTraceModule()),
    ):
        t = Telemetry()
        t.mark_run_started({"k": "v"})
        t.record_cycle({"k": "v"})
        t.record_latency("bandit_update", 0.1, {"k": "v"})
        t.record_latency("prioritization", 0.2, {"k": "v"})
        t.record_latency("evaluation", 0.3, {"k": "v"})
        t.record_fitness(0.9, 0.8, {"k": "v"})
        t.record_resource_snapshot(10.0, 20.0, 30.0, {"k": "v"})
        t.record_experiment_resources(12.0, 5.0, 20.0, {"k": "v"})
        t.record_checkpoint_save({"k": "v"})
        with t.span("demo", attributes={"step": 1}) as span:
            assert span is None
        t.flush()
        t.mark_run_finished({"k": "v"})

        # Ensure at least one instrument captured data.
        assert t.cycles_total.events
