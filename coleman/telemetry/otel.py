"""
coleman.telemetry.otel - OpenTelemetry Initialization and No-Op Fallback.

When ``opentelemetry`` is installed **and** telemetry is enabled the module
configures a ``Telemetry`` facade that exposes pre-defined metrics and
trace helpers.

When the SDK is absent or telemetry is disabled, ``NoOpTelemetry`` provides
the same interface with near-zero overhead.

Metric names (for documentation / Grafana dashboard authors)
------------------------------------------------------------
* ``coleman.run_active``             – up/down counter
* ``coleman.cycles_total``            – counter
* ``coleman.bandit_update_latency``   – histogram (seconds)
* ``coleman.prioritization_latency``  – histogram (seconds)
* ``coleman.evaluation_latency``      – histogram (seconds)
* ``coleman.napfd``                   – histogram
* ``coleman.apfdc``                   – histogram
* ``coleman.process_memory_rss``      – histogram (MiB)
* ``coleman.process_memory_peak_rss`` – histogram (MiB)
* ``coleman.process_cpu_utilization`` – histogram (%)
* ``coleman.experiment_wall_time``    – histogram (seconds)
* ``coleman.experiment_cpu_time``     – histogram (seconds)
* ``coleman.checkpoint_saves_total``  – counter
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing OpenTelemetry; fall back gracefully
# ---------------------------------------------------------------------------
_HAS_OTEL = False
otel_metrics: Any = None
otel_trace: Any = None
OTLPMetricExporter: Any = None
OTLPSpanExporter: Any = None
MeterProvider: Any = None
PeriodicExportingMetricReader: Any = None
Resource: Any = None
TracerProvider: Any = None
BatchSpanProcessor: Any = None

try:
    otel_metrics = importlib.import_module("opentelemetry.metrics")
    otel_trace = importlib.import_module("opentelemetry.trace")
    OTLPMetricExporter = importlib.import_module(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter"
    ).OTLPMetricExporter
    OTLPSpanExporter = importlib.import_module("opentelemetry.exporter.otlp.proto.http.trace_exporter").OTLPSpanExporter
    MeterProvider = importlib.import_module("opentelemetry.sdk.metrics").MeterProvider
    PeriodicExportingMetricReader = importlib.import_module(
        "opentelemetry.sdk.metrics.export"
    ).PeriodicExportingMetricReader
    Resource = importlib.import_module("opentelemetry.sdk.resources").Resource
    TracerProvider = importlib.import_module("opentelemetry.sdk.trace").TracerProvider
    BatchSpanProcessor = importlib.import_module("opentelemetry.sdk.trace.export").BatchSpanProcessor
    _HAS_OTEL = True
except ImportError:  # pragma: no cover – tested via NoOpTelemetry path
    pass


class Telemetry:
    """OpenTelemetry facade for Coleman.

    Parameters
    ----------
    service_name : str
        OTLP service/resource name.
    otlp_endpoint : str
        OTLP HTTP endpoint.

    Attributes
    ----------
    meter : opentelemetry.metrics.Meter
        OTel meter for creating instruments.
    tracer : opentelemetry.trace.Tracer
        OTel tracer for creating spans.
    """

    def __init__(
        self,
        service_name: str = "coleman",
        otlp_endpoint: str = "http://localhost:4318",
        export_interval_millis: int = 5000,
        resource_attributes: dict[str, str] | None = None,
    ) -> None:
        """Initialise OpenTelemetry metrics and traces providers.

        Parameters
        ----------
        service_name : str
            OTLP service/resource name.
        otlp_endpoint : str
            OTLP HTTP endpoint base URL.
        export_interval_millis : int
            Metric export interval in milliseconds.
        resource_attributes : dict[str, str] or None
            Extra OTel resource attributes merged with ``service.name``.

        Raises
        ------
        ImportError
            If the ``opentelemetry`` SDK is not installed.
        """
        if not _HAS_OTEL:
            raise ImportError(
                "opentelemetry SDK is required for Telemetry. Install it with: pip install coleman[telemetry]"
            )

        attrs: dict[str, str] = dict(resource_attributes or {})
        attrs["service.name"] = service_name
        resource = Resource.create(attrs)

        # Metrics
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics"),
            export_interval_millis=export_interval_millis,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        otel_metrics.set_meter_provider(meter_provider)
        self._meter_provider = meter_provider
        self.meter = meter_provider.get_meter("coleman")

        # Traces
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")))
        otel_trace.set_tracer_provider(tracer_provider)
        self._tracer_provider = tracer_provider
        self.tracer = tracer_provider.get_tracer("coleman")

        # Pre-create instruments
        self.run_active = self.meter.create_up_down_counter(
            "coleman.run_active", description="Currently active experiment runs"
        )
        self.cycles_total = self.meter.create_counter("coleman.cycles_total", description="Total cycles processed")
        self.bandit_update_latency = self.meter.create_histogram(
            "coleman.bandit_update_latency", unit="s", description="Bandit update latency"
        )
        self.prioritization_latency = self.meter.create_histogram(
            "coleman.prioritization_latency", unit="s", description="Prioritization latency"
        )
        self.evaluation_latency = self.meter.create_histogram(
            "coleman.evaluation_latency", unit="s", description="Evaluation latency"
        )
        self.napfd = self.meter.create_histogram("coleman.napfd", description="NAPFD distribution")
        self.apfdc = self.meter.create_histogram("coleman.apfdc", description="APFDc distribution")
        self.process_memory_rss = self.meter.create_histogram(
            "coleman.process_memory_rss", unit="MiB", description="Process resident memory usage"
        )
        self.process_memory_peak_rss = self.meter.create_histogram(
            "coleman.process_memory_peak_rss", unit="MiB", description="Process peak resident memory usage"
        )
        self.process_cpu_utilization = self.meter.create_histogram(
            "coleman.process_cpu_utilization", unit="%", description="Process CPU utilization"
        )
        self.experiment_wall_time = self.meter.create_histogram(
            "coleman.experiment_wall_time", unit="s", description="End-to-end experiment duration"
        )
        self.experiment_cpu_time = self.meter.create_histogram(
            "coleman.experiment_cpu_time", unit="s", description="CPU time consumed by one experiment"
        )
        self.checkpoint_saves_total = self.meter.create_counter(
            "coleman.checkpoint_saves_total", description="Total checkpoints saved"
        )

    def record_cycle(self, attributes: dict[str, Any] | None = None) -> None:
        """Increment the cycle counter.

        Parameters
        ----------
        attributes : dict or None
            Optional OTel attributes (avoid high cardinality).
        """
        self.cycles_total.add(1, attributes=attributes)

    def mark_run_started(self, attributes: dict[str, Any] | None = None) -> None:
        """Mark an experiment run as active."""
        self.run_active.add(1, attributes=attributes)

    def mark_run_finished(self, attributes: dict[str, Any] | None = None) -> None:
        """Mark an experiment run as finished."""
        self.run_active.add(-1, attributes=attributes)

    def record_latency(self, name: str, duration: float, attributes: dict[str, Any] | None = None) -> None:
        """Record a latency observation.

        Parameters
        ----------
        name : str
            One of ``bandit_update``, ``prioritization``, ``evaluation``.
        duration : float
            Duration in seconds.
        attributes : dict or None
            Optional OTel attributes.
        """
        hist = getattr(self, f"{name}_latency", None)
        if hist is not None:
            hist.record(duration, attributes=attributes)

    def record_fitness(self, fitness: float, cost: float, attributes: dict[str, Any] | None = None) -> None:
        """Record NAPFD/APFDc observations.

        Parameters
        ----------
        fitness : float
            NAPFD value.
        cost : float
            APFDc value.
        attributes : dict or None
            Optional OTel attributes.
        """
        self.napfd.record(fitness, attributes=attributes)
        self.apfdc.record(cost, attributes=attributes)

    def record_resource_snapshot(
        self,
        current_rss_mib: float | None,
        peak_rss_mib: float | None,
        cpu_utilization_percent: float | None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record a point-in-time process resource sample."""
        if current_rss_mib is not None:
            self.process_memory_rss.record(current_rss_mib, attributes=attributes)
        if peak_rss_mib is not None:
            self.process_memory_peak_rss.record(peak_rss_mib, attributes=attributes)
        if cpu_utilization_percent is not None:
            self.process_cpu_utilization.record(cpu_utilization_percent, attributes=attributes)

    def record_experiment_resources(
        self,
        wall_time_seconds: float,
        cpu_time_seconds: float,
        peak_rss_mib: float | None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record final resource totals for one experiment."""
        self.experiment_wall_time.record(wall_time_seconds, attributes=attributes)
        self.experiment_cpu_time.record(cpu_time_seconds, attributes=attributes)
        if peak_rss_mib is not None:
            self.process_memory_peak_rss.record(peak_rss_mib, attributes=attributes)

    def record_checkpoint_save(self, attributes: dict[str, Any] | None = None) -> None:
        """Record one successful checkpoint save event."""
        self.checkpoint_saves_total.add(1, attributes=attributes)

    def flush(self) -> None:
        """Force pending metrics and traces to be exported."""
        self._meter_provider.force_flush()
        self._tracer_provider.force_flush()

    @contextlib.contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Generator[Any]:
        """Create a trace span context manager.

        Parameters
        ----------
        name : str
            Span name.
        attributes : dict or None
            Span attributes (step allowed here, unlike metrics).

        Yields
        ------
        opentelemetry.trace.Span
        """
        with self.tracer.start_as_current_span(name, attributes=attributes) as s:
            yield s


class NoOpTelemetry:
    """No-op telemetry stub with the same public interface as ``Telemetry``.

    All methods are instant no-ops.  Used when telemetry is disabled or the
    OpenTelemetry SDK is not installed.
    """

    def record_cycle(self, attributes: dict[str, Any] | None = None) -> None:
        """No-op.

        Parameters
        ----------
        attributes : dict or None
            Ignored.
        """

    def mark_run_started(self, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""

    def mark_run_finished(self, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""

    def record_latency(self, name: str, duration: float, attributes: dict[str, Any] | None = None) -> None:
        """No-op.

        Parameters
        ----------
        name : str
            Ignored.
        duration : float
            Ignored.
        attributes : dict or None
            Ignored.
        """

    def record_fitness(self, fitness: float, cost: float, attributes: dict[str, Any] | None = None) -> None:
        """No-op.

        Parameters
        ----------
        fitness : float
            Ignored.
        cost : float
            Ignored.
        attributes : dict or None
            Ignored.
        """

    def record_resource_snapshot(
        self,
        current_rss_mib: float | None,
        peak_rss_mib: float | None,
        cpu_utilization_percent: float | None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """No-op."""

    def record_experiment_resources(
        self,
        wall_time_seconds: float,
        cpu_time_seconds: float,
        peak_rss_mib: float | None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """No-op."""

    def record_checkpoint_save(self, attributes: dict[str, Any] | None = None) -> None:
        """No-op."""

    def flush(self) -> None:
        """No-op flush."""

    @contextlib.contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Generator[None]:
        """No-op context manager.

        Parameters
        ----------
        name : str
            Ignored.
        attributes : dict or None
            Ignored.

        Yields
        ------
        None
        """
        yield None


def get_telemetry(
    enabled: bool = False,
    service_name: str = "coleman",
    otlp_endpoint: str = "http://localhost:4318",
    export_interval_millis: int = 5000,
    resource_attributes: dict[str, str] | None = None,
) -> Telemetry | NoOpTelemetry:
    """Return the appropriate telemetry implementation.

    Parameters
    ----------
    enabled : bool
        Whether telemetry is enabled.
    service_name : str
        OTLP service name.
    otlp_endpoint : str
        OTLP endpoint URL.
    export_interval_millis : int
        Metric export interval in milliseconds.
    resource_attributes : dict[str, str] or None
        Extra OTel resource attributes (e.g. ``execution_id``, ``run_id``).
        Attached to the Resource, not to individual metric labels.

    Returns
    -------
    Telemetry or NoOpTelemetry
        Active telemetry if enabled and SDK present, else no-op.
    """
    if not enabled:
        return NoOpTelemetry()

    if not _HAS_OTEL:
        logger.warning(
            "Telemetry enabled but opentelemetry SDK not installed. "
            "Install with: pip install coleman[telemetry]. "
            "Falling back to NoOpTelemetry."
        )
        return NoOpTelemetry()

    try:
        return Telemetry(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
            export_interval_millis=export_interval_millis,
            resource_attributes=resource_attributes,
        )
    except Exception:
        logger.exception("Failed to initialize OpenTelemetry; falling back to NoOpTelemetry")
        return NoOpTelemetry()
