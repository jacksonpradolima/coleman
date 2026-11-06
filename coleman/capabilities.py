"""Explicit optional installation checks at the application boundary.

This table does not discover plugins or load backends. Core mathematical calls
never consult it. Availability checks read distribution metadata only; installed
version compatibility is established by the resolver and runtime inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class InstallationCapability:
    """A named installation profile and its required distribution names.

    Attributes
    ----------
    name : str
        Closed, case-sensitive extra name.
    distributions : tuple[str, ...]
        Direct dependencies checked for presence, not imported or executed.
    """

    name: str
    distributions: tuple[str, ...]


_CAPABILITIES = (
    InstallationCapability("ci", ("duckdb", "numpy", "polars", "pyarrow", "pyyaml")),
    InstallationCapability("analytics", ("duckdb", "numpy", "polars", "pyarrow", "scipy", "scikit-posthocs")),
    InstallationCapability("mcp", ("mcp",)),
    InstallationCapability(
        "telemetry", ("opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http")
    ),
    InstallationCapability("clickhouse", ("clickhouse-connect",)),
)


class CapabilityUnavailableError(ImportError):
    """A selected installation capability lacks dependencies; no backend was loaded."""


def installation_capabilities() -> tuple[InstallationCapability, ...]:
    """Return the closed optional profile table without inspecting the environment.

    Returns
    -------
    tuple[InstallationCapability, ...]
        Immutable declarations. Importing and listing profiles cause no I/O,
        registry initialization, threads, network access or filesystem writes.
    """
    return _CAPABILITIES


def require_capability(name: str) -> None:
    """Require a selected capability before application effects or backend loading.

    Parameters
    ----------
    name : str
        Exact profile returned by ``installation_capabilities``.

    Raises
    ------
    ValueError
        Unknown capability, including unsupported provider/optimizer names.
    CapabilityUnavailableError
        At least one required distribution is absent. The message identifies
        the missing distributions and verified-wheel extra installation route.

    Notes
    -----
    Reads installed distribution metadata without importing dependency modules,
    creating workspaces, starting threads or connecting to services. Presence is
    not a version or behavioral certificate; the package resolver and explicit
    runtime inventory enforce those separate checks. Genuine metadata errors
    other than an absent distribution propagate without being masked.
    """
    profile = next((item for item in _CAPABILITIES if item.name == name), None)
    if profile is None:
        choices = ", ".join(item.name for item in _CAPABILITIES)
        raise ValueError(f"Unknown COLEMAN installation capability {name!r}; supported: {choices}")
    missing = []
    for distribution in profile.distributions:
        try:
            version(distribution)
        except PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise CapabilityUnavailableError(
            f"COLEMAN capability {name!r} requires missing distributions: {', '.join(missing)}. "
            f"Install the reviewed GitHub/local COLEMAN wheel with its [{name}] extra, "
            f"for example uv pip install './coleman-VERSION-py3-none-any.whl[{name}]'. "
            "The bare PyPI project named coleman has a different owner; see docs/core-installation.md."
        )
