"""
coleman.telemetry - Optional OpenTelemetry Instrumentation.

This subpackage provides optional observability via OpenTelemetry metrics and
traces.  When the ``opentelemetry`` SDK is not installed (or telemetry is
disabled) all instrumentation resolves to lightweight no-op stubs.

Enable the telemetry extra from a COLEMAN GitHub release wheel; see
https://github.com/jacksonpradolima/coleman#installation.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coleman.telemetry.otel import NoOpTelemetry, Telemetry, get_telemetry

__all__ = ["Telemetry", "NoOpTelemetry", "get_telemetry"]


def __getattr__(name: str) -> Any:
    """Load the legacy optional facade only when explicitly requested.

    Parameters
    ----------
    name : str
        Existing public facade name. Pure lifecycle contracts never request it.

    Returns
    -------
    object
        Original legacy class/factory, retaining its initialization behavior.

    Raises
    ------
    AttributeError
        The requested name is not a supported public export.

    Notes
    -----
    A plain package/resources/contracts import initializes no SDK or exporter.
    Existing explicit facade imports still work; no artifact meaning changes.
    """
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("coleman.telemetry.otel"), name)
