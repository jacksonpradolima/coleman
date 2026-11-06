"""Resolve explicitly listed public aliases without adapter discovery or setup."""

from importlib import import_module
from typing import Any


def resolve_export(name: str, exports: dict[str, tuple[str, str]], module_name: str) -> Any:
    """Import only the selected declared symbol; genuine module errors propagate."""
    target = exports.get(name)
    if target is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    module, symbol = target
    return getattr(import_module(module), symbol)
