"""Retain the scientific v1 namespace over the shared bounded artifact codec."""

from __future__ import annotations

from typing import ClassVar

from coleman.core._wire import (
    HARD_ENVELOPE_BYTES,
    HARD_GRAPH_ITEMS,
    BoundedContract,
    _GraphBudget,
    _number_size,
    _quoted_size,
    _scalar_size,
    _visit_graph,
    _visit_mapping,
    guard_graph,
    model_storage,
    wire_size,
)

__all__ = [
    "HARD_ENVELOPE_BYTES",
    "HARD_GRAPH_ITEMS",
    "DesignContract",
    "_GraphBudget",
    "_number_size",
    "_quoted_size",
    "_scalar_size",
    "_visit_graph",
    "_visit_mapping",
    "guard_graph",
    "model_storage",
    "wire_size",
]


class DesignContract(BoundedContract):
    """Version new bounded experiment evidence independently of legacy artifacts.

    Notes
    -----
    Concrete roots require schema version 1, reject unknown fields and retain
    immutable typed values. The hard UTF-8/depth ceiling applies at every JSON
    ingress and serialization. A canonical body reserves one byte for a final LF
    in local artifacts; the LF is excluded from the model fingerprint. Concrete
    designs additionally enforce their smaller declared profile budgets.
    Construction and reading perform no migration, clock capture or file I/O.
    """

    _maximum_bytes: ClassVar[int] = HARD_ENVELOPE_BYTES
