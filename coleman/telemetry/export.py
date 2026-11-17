"""Opt-in bounded lifecycle gauges using an explicitly host-owned OTel meter.

Importing this module never loads OpenTelemetry or creates a provider, exporter,
thread, socket or global registration. The host owns exporter lifecycle/errors.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Protocol

from coleman.telemetry.lifecycle import LifecycleSnapshot, metric_catalog


class _Gauge(Protocol):  # pylint: disable=too-few-public-methods
    """Describe the standard OTel synchronous gauge without importing its SDK."""

    def set(self, amount: int | float, attributes: dict[str, str]) -> None:
        """Record an instantaneous gauge value using bounded attributes."""


class _Meter(Protocol):  # pylint: disable=too-few-public-methods
    """Describe only the explicitly requested standard OTel meter operation."""

    def create_gauge(self, name: str, *, unit: str, description: str) -> Any:
        """Acquire an instrument from the host's already configured provider."""


class LifecycleTelemetry:  # pylint: disable=too-few-public-methods
    """Record validated lifecycle snapshots through a host-owned meter.

    Parameters
    ----------
    meter : opentelemetry.metrics.Meter
        Explicit meter with synchronous gauges (qualified SDK1.45.2). The host
        chooses in-memory readers or network exporters and owns their shutdown.

    Notes
    -----
    Only route plus closed metric/status names become attributes: at most76
    numeric and304 availability combinations per resource. Outcomes, versions,
    fingerprints, user IDs, task IDs and timestamps stay in snapshot artifacts.
    Provider/resource cardinality and backend retention remain host responsibilities.
    Calls through one facade serialize. Distinct facades sharing instruments
    need coordinated writers; collector reads are not an atomic transaction.
    No exactly-once export is claimed. Gauge repeats overwrite the same series;
    they do not count deliveries. Read availability before using a numeric gauge:
    an unavailable reading leaves its previous numeric sample stale.
    """

    def __init__(self, meter: _Meter) -> None:
        """Acquire finite gauges without starting a provider or exporter.

        Parameters
        ----------
        meter : opentelemetry.metrics.Meter
            Trusted already configured host-owned meter.

        Raises
        ------
        AttributeError, RuntimeError
            Unsupported meter or provider/instrument failure; never retried.
        """
        self._values: dict[str, _Gauge] = {
            item.name: meter.create_gauge(
                f"coleman.lifecycle.{item.name}", unit=item.unit, description=item.interpretation
            )
            for item in metric_catalog()
        }
        self._availability: _Gauge = meter.create_gauge(
            "coleman.lifecycle.availability", unit="1", description="One-hot most recent metric availability"
        )
        self._lock = Lock()

    def record(self, snapshot: LifecycleSnapshot) -> None:
        """Export one snapshot after complete validation, without a host callback.

        Parameters
        ----------
        snapshot : LifecycleSnapshot
            Immutable v1 observation; revalidated even for unchecked instances.

        Raises
        ------
        ValueError
            Invalid snapshot, before any metric is emitted.
        RuntimeError, OSError
            Host meter/exporter failure. Export is not atomic or durable; record
            only after the caller's own scientific state commit. No host state
            is changed, acknowledged, rolled back or declared complete here.
        """
        current = LifecycleSnapshot.model_validate(snapshot)
        base: dict[str, str] = {"route": current.route}
        with self._lock:
            for item in current.points:
                for status in ("observed", "unknown", "unavailable", "not_applicable"):
                    self._availability.set(
                        float(status == item.status), {**base, "metric": item.name, "status": status}
                    )
                if item.value is not None:
                    self._values[item.name].set(item.value, base)


class NoOpLifecycleTelemetry:  # pylint: disable=too-few-public-methods
    """Validate the same contract without retaining observations or providers."""

    def record(self, snapshot: LifecycleSnapshot) -> None:
        """Check one snapshot with the same ingress semantics as instrumentation.

        Parameters
        ----------
        snapshot : LifecycleSnapshot
            Complete v1 observation; no clock, I/O, SDK or counters are touched.

        Raises
        ------
        ValueError
            Unsupported/invalid snapshot, including unchecked instance mutation.
        """
        LifecycleSnapshot.model_validate(snapshot)
