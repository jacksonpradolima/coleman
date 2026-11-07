"""Canonical coefficient bytes shared by private controlled-state validators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coleman.core.primitives import canonical_json

if TYPE_CHECKING:
    from coleman.core.actions import Weight


def _encoded(value: object) -> bytes:
    """Compare complete typed bytes, retaining signed zero and versions."""
    return canonical_json(value).encode("utf-8")


def _weight_bytes(weights: tuple[Weight, ...]) -> bytes:
    """Retain the exact emitted coefficients rather than Python equality."""
    return _encoded([item.model_dump(mode="json") for item in weights])
