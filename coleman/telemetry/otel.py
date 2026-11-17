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


def _dispose_partial(*owners: Any) -> None:
    """Attempt every partial-owner shutdown without masking initialization failure."""
    for owner in owners:
        if owner is not None:
            try:
                owner.shutdown()
            except Exception:  # Cleanup must still attempt the other independent provider.
                logger.exception("Failed to dispose a partially initialized telemetry resource")


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
        RuntimeError, OSError
            SDK/provider/instrument construction or registration failure after
            acquired owned resources have been offered shutdown. The factory
            logs these failures and retains its existing no-op fallback.

        Notes
        -----
        Construction is explicitly opt-in: successful providers own exporter
        threads and normal process-wide SDK registration. Failed construction
        disposes acquired resources before propagating its original exception;
        registration waits until providers and instruments are constructed.
        SDK global registrations retain their existing write-once semantics.
        """
        if not _HAS_OTEL:
            raise ImportError(
                "opentelemetry SDK is required for Telemetry. Install the COLEMAN telemetry extra "
                "from a GitHub release wheel: https://github.com/jacksonpradolima/coleman#installation"
            )

        attrs: dict[str, str] = dict(resource_attributes or {})
        attrs["service.name"] = service_name
        resource = Resource.create(attrs)

        self._initialize(resource, otlp_endpoint, export_interval_millis)

    def _initialize(self, resource: Any, endpoint: str, interval: int) -> None:
        """Transfer shutdown ownership only after each parent accepts its child."""
        metric_owner = span_owner = tracer_owner = None
        try:
            metric_owner = OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics")
            metric_owner = PeriodicExportingMetricReader(metric_owner, export_interval_millis=interval)
            metric_owner = MeterProvider(resource=resource, metric_readers=[metric_owner])
            self._meter_provider = metric_owner
            self.meter = metric_owner.get_meter("coleman")

            span_owner = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
            span_owner = BatchSpanProcessor(span_owner)
            tracer_owner = TracerProvider(resource=resource)
            tracer_owner.add_span_processor(span_owner)
            span_owner, tracer_owner = tracer_owner, None
            self._tracer_provider = span_owner
            self.tracer = span_owner.get_tracer("coleman")
            self._create_instruments()
            # These retain the SDK's normal process-wide write-once behavior.
            # No partial exporter/provider is registered during construction.
            otel_metrics.set_meter_provider(self._meter_provider)
            otel_trace.set_tracer_provider(self._tracer_provider)
        except BaseException:
            _dispose_partial(tracer_owner, span_owner, metric_owner)
            raise

    def _create_instruments(self) -> None:
        """Construct the unchanged native instruments before global registration."""
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
        """Increment the existing native active-run up/down counter.

        Parameters
        ----------
        attributes : dict or None, optional
            Native series labels, default none. The host owns their cardinality
            and must use identical labels for the matching finished event.

        Notes
        -----
        Records through the configured SDK; it neither starts nor acknowledges
        a job. Calls are not deduplicated and must be balanced by the host.
        """
        self.run_active.add(1, attributes=attributes)

    def mark_run_finished(self, attributes: dict[str, Any] | None = None) -> None:
        """Decrement the existing native active-run counter after a host event.

        Parameters
        ----------
        attributes : dict or None, optional
            Same native series labels as the corresponding started event.

        Notes
        -----
        Exports an observation only. It cannot complete a job, validate an
        artifact, release a worker lock or prove exactly-once physical execution.
        Repeated/unbalanced calls retain the legacy counter behavior.
        """
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
        """Record existing current-executable resource observations.

        Parameters
        ----------
        current_rss_mib : float or None
            Sampled executable RSS in MiB (1048576bytes), or unavailable.
        peak_rss_mib : float or None
            Executable lifetime high-water RSS in MiB; may include earlier work.
        cpu_utilization_percent : float or None
            CPU/wall interval ratio times100; can exceed100 with native threads.
        attributes : dict or None, optional
            Host-controlled native labels; no per-step cardinality limit is
            imposed by this retained legacy facade.

        Notes
        -----
        None omits the sample, never records zero. These are not child-process
        totals, exclusive per-policy allocations or subject CI infrastructure.
        The existing facade trusts the caller's units/values and exports to its
        SDK; use typed lifecycle snapshots for strict finite/status validation.
        """
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
        """Record one attempt's cumulative seconds and executable lifetime peak.

        Parameters
        ----------
        wall_time_seconds : float
            Monotonic elapsed seconds since this attempt tracker was created.
        cpu_time_seconds : float
            Current-executable CPU seconds since tracker creation, not fees or
            total subject workload. Do not sum cumulative intermediate samples.
        peak_rss_mib : float or None
            Executable lifetime high-water MiB, or unavailable (sample omitted).
        attributes : dict or None, optional
            Existing host-controlled native labels and their cardinality policy.

        Notes
        -----
        Writes SDK observations only, without changing run/task completion.
        Legacy caller validation and artifact meanings remain unchanged.
        """
        self.experiment_wall_time.record(wall_time_seconds, attributes=attributes)
        self.experiment_cpu_time.record(cpu_time_seconds, attributes=attributes)
        if peak_rss_mib is not None:
            self.process_memory_peak_rss.record(peak_rss_mib, attributes=attributes)

    def record_checkpoint_save(self, attributes: dict[str, Any] | None = None) -> None:
        """Record one host-declared successful checkpoint-save event.

        Parameters
        ----------
        attributes : dict or None, optional
            Existing native labels. The caller owns finite cardinality and
            calls only after its actual successful checkpoint operation.

        Notes
        -----
        Increments a diagnostic counter; does not write/acknowledge state or
        deduplicate retries. A metric cannot establish durable completion.
        """
        self.checkpoint_saves_total.add(1, attributes=attributes)

    def flush(self) -> None:
        """Request the existing SDK metric/trace flush without an export receipt.

        Notes
        -----
        Retains legacy return behavior: SDK timeout booleans are not returned
        by this facade. This is neither durable publication nor proof that a
        remote collector persisted every observation; inspect SDK/backend health.
        """
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
        """Accept the legacy started-event signature without instrumentation.

        Parameters
        ----------
        attributes : dict or None, optional
            Ignored native labels. No provider, counter or run is created.
        """

    def mark_run_finished(self, attributes: dict[str, Any] | None = None) -> None:
        """Accept the legacy finished-event signature without instrumentation.

        Parameters
        ----------
        attributes : dict or None, optional
            Ignored native labels. No job is completed or acknowledged.
        """

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
        """Accept existing process samples without reading or exporting resources.

        Parameters
        ----------
        current_rss_mib, peak_rss_mib : float or None
            Ignored current/lifetime-peak MiB measurements, including zero/null.
        cpu_utilization_percent : float or None
            Ignored CPU/wall interval percentage, possibly above100.
        attributes : dict or None, optional
            Ignored native labels. No validation or retention is introduced.
        """

    def record_experiment_resources(
        self,
        wall_time_seconds: float,
        cpu_time_seconds: float,
        peak_rss_mib: float | None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Accept the retained final-resource signature with no side effects.

        Parameters
        ----------
        wall_time_seconds, cpu_time_seconds : float
            Ignored attempt-tracker cumulative seconds, never interpreted as fees.
        peak_rss_mib : float or None
            Ignored executable lifetime high-water MiB or unavailable reading.
        attributes : dict or None, optional
            Ignored native labels. No validation, provider or counters are added.
        """

    def record_checkpoint_save(self, attributes: dict[str, Any] | None = None) -> None:
        """Accept a checkpoint diagnostic without writing or acknowledging state.

        Parameters
        ----------
        attributes : dict or None, optional
            Ignored native labels. This does not establish checkpoint completion.
        """

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
            "Install the COLEMAN telemetry extra from a GitHub release wheel; see "
            "https://github.com/jacksonpradolima/coleman#installation. "
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
