"""Bounded transport-neutral application commands over native COLEMAN services.

Public aliases load only their selected implementation. Importing this namespace
performs no plugin discovery, configuration, I/O or optional backend import.
"""

from typing import TYPE_CHECKING, Any

from coleman._exports import resolve_export

if TYPE_CHECKING:
    from coleman.application.contracts import (
        ApplicationError,
        Caller,
        Handle,
        JobView,
        Limits,
        Parameters,
        PortableNativeProvenance,
        ResultBundle,
        ServerPolicy,
        Submit,
    )
    from coleman.application.registry import Configuration, default_registry, register_experiment
    from coleman.application.service import ApplicationService

_EXPORTS = {
    "ApplicationError": ("coleman.application.contracts", "ApplicationError"),
    "ApplicationService": ("coleman.application.service", "ApplicationService"),
    "Caller": ("coleman.application.contracts", "Caller"),
    "Configuration": ("coleman.application.registry", "Configuration"),
    "Handle": ("coleman.application.contracts", "Handle"),
    "JobView": ("coleman.application.contracts", "JobView"),
    "Limits": ("coleman.application.contracts", "Limits"),
    "Parameters": ("coleman.application.contracts", "Parameters"),
    "PortableNativeProvenance": ("coleman.application.contracts", "PortableNativeProvenance"),
    "ResultBundle": ("coleman.application.contracts", "ResultBundle"),
    "ServerPolicy": ("coleman.application.contracts", "ServerPolicy"),
    "Submit": ("coleman.application.contracts", "Submit"),
    "default_registry": ("coleman.application.registry", "default_registry"),
    "register_experiment": ("coleman.application.registry", "register_experiment"),
}

__all__ = [
    "ApplicationError",
    "ApplicationService",
    "Caller",
    "Configuration",
    "Handle",
    "JobView",
    "Limits",
    "Parameters",
    "PortableNativeProvenance",
    "ResultBundle",
    "ServerPolicy",
    "Submit",
    "default_registry",
    "register_experiment",
]


def __getattr__(name: str) -> Any:
    """Load a declared compatibility alias; unsupported names and import errors propagate."""
    value = resolve_export(name, _EXPORTS, __name__)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the unchanged public API without importing its implementations."""
    return sorted(set(globals()) | set(__all__))
