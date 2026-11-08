"""Bound typed artifact graphs without importing a domain or transport."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any, ClassVar, Self
from zoneinfo import ZoneInfo

from pydantic import BaseModel, model_validator

from coleman.core.base import CoreContract
from coleman.core.primitives import canonical_json
from coleman.core.reading import bounded_json_bytes

HARD_ENVELOPE_BYTES = 64 * 1024**2
HARD_GRAPH_ITEMS = 2_000_000


class _GraphBudget:
    """Count every occurrence without serializing an unchecked model first."""

    def __init__(self, maximum_bytes: int, maximum_items: int) -> None:
        """Initialize local UTF-8 byte and graph-occurrence counters without I/O."""
        self.maximum_bytes = maximum_bytes
        self.maximum_items = maximum_items
        self.bytes = 0
        self.items = 0

    def charge(self, amount: int) -> None:
        """Charge bytes without permitting the current allowance to overflow."""
        self.bytes += amount
        if self.bytes > self.maximum_bytes:
            raise ValueError("Graph byte budget exceeded before model serialization")

    def item(self) -> None:
        """Count another occurrence, including references to a shared alias."""
        self.items += 1
        if self.items > self.maximum_items:
            raise ValueError("Graph item budget exceeded before model validation")


def _quoted_size(value: str, remaining: int) -> int:
    """Count exact unescaped-UTF8 JSON string bytes without a large temporary."""
    size = 2
    for character in value:
        code = ord(character)
        if character in '"\\\b\f\n\r\t':
            size += 2
        elif code < 32:
            size += 6
        elif code < 128:
            size += 1
        elif code < 2048:
            size += 2
        elif 0xD800 <= code <= 0xDFFF:
            raise ValueError("Graph strings require valid UTF-8 without lone surrogates")
        else:
            size += 3 if code < 65536 else 4
        if size > remaining:
            raise ValueError("Graph byte budget exceeded by a UTF-8/escaped string")
    return size


def _scalar_size(value: Any, remaining: int) -> int:
    """Bound only declared standard scalar forms, never an arbitrary encoder."""
    kind = type(value)
    if value is None:
        return 4
    if kind is str:
        return _quoted_size(value, remaining)
    if kind in {bool, int, float}:
        return _number_size(value, remaining)
    if kind is Decimal:
        if not value.is_finite():
            raise ValueError("Graph requires finite Decimal values")
        return _quoted_size(str(value), remaining)
    if kind in {datetime, date, time}:
        if kind in {datetime, time} and value.tzinfo is not None and type(value.tzinfo) not in {timezone, ZoneInfo}:
            raise TypeError("Graph timestamps require a standard explicit timezone")
        return _quoted_size(value.isoformat().replace("+00:00", "Z"), remaining)
    raise TypeError("Graph contains an unsupported opaque value before validation")


def _number_size(value: bool | int | float, remaining: int) -> int:
    """Measure exact built-in numeric JSON without executing conversion hooks."""
    if type(value) is bool:  # pylint: disable=unidiomatic-typecheck
        return 4 if value else 5
    if type(value) is int:  # pylint: disable=unidiomatic-typecheck
        if (value.bit_length() - 1) * 30103 // 100000 > remaining:
            raise ValueError("Graph byte budget exceeded by an integer")
        return len(str(value))
    if not math.isfinite(value):
        raise ValueError("Graph requires finite numeric values")
    return len(json.dumps(value, allow_nan=False))


def _visit_graph(value: Any, depth: int, active: set[int], budget: _GraphBudget) -> None:
    """Traverse with an ancestor set so aliases are charged per occurrence."""
    budget.item()
    if isinstance(value, BaseModel):
        identity = id(value)
        if identity in active:
            raise ValueError("Cannot validate a cyclic model graph")
        active.add(identity)
        _visit_graph(model_storage(value), depth, active, budget)
        active.remove(identity)
        return
    if type(value) not in {dict, list, tuple, set, frozenset}:
        budget.charge(_scalar_size(value, budget.maximum_bytes - budget.bytes))
        return
    if depth >= 64:
        raise ValueError("Graph container depth exceeds 64 before validation")
    identity = id(value)
    if identity in active:
        raise ValueError("Cannot validate a cyclic container graph")
    active.add(identity)
    budget.charge(2 + max(0, len(value) - 1))
    if type(value) is dict:  # pylint: disable=unidiomatic-typecheck
        _visit_mapping(value, depth, active, budget)
    else:
        for item in value:
            _visit_graph(item, depth + 1, active, budget)
    active.remove(identity)


def _visit_mapping(value: dict[Any, Any], depth: int, active: set[int], budget: _GraphBudget) -> None:
    """Bound each JSON key and value without expanding a complete child list."""
    for key, item in value.items():
        if type(key) is not str:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Graph object keys must be built-in strings")
        budget.item()
        budget.charge(_quoted_size(key, budget.maximum_bytes - budget.bytes) + 1)
        _visit_graph(item, depth + 1, active, budget)


def guard_graph(value: Any, *, maximum_bytes: int, maximum_items: int) -> None:
    """Reject oversized unchecked graphs before legacy validation/serialization.

    Parameters
    ----------
    value : object
        JSON-like built-in graph or Pydantic model. Models are read through their
        stored __dict__, without calling model_dump, to_json or model validators.
    maximum_bytes, maximum_items : int
        Strict positive bounds. UTF-8/escaping and every repeated alias occurrence
        are charged. Container depth is bounded at 64; ancestor cycles reject.

    Raises
    ------
    ValueError, TypeError
        Invalid bounds, excessive graph, cycles, unknown opaque values or invalid
        UTF-8/nonfinite data. No I/O, model conversion or callback is performed.

    Notes
    -----
    Model defaults/private stored fields can make this conservative relative to
    a custom serializer; actual encoded output is checked separately. It is an
    owned graph preflight, not arbitrary-Python authenticity or a resource sandbox.
    """
    if any(type(limit) is not int or limit <= 0 for limit in (maximum_bytes, maximum_items)):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Graph limits require strict positive integers")
    _visit_graph(value, 0, set(), _GraphBudget(maximum_bytes, maximum_items))


def model_storage(value: BaseModel) -> dict[str, Any]:
    """Inspect actual stored model fields before any legacy parser/serializer.

    Parameters
    ----------
    value : pydantic.BaseModel
        Caller-supplied model, possibly unchecked or hostile. Its model_dump,
        validators and a subclass __dict__ property must not execute here.

    Returns
    -------
    dict[str, object]
        Existing field storage for read-only boundary inspection, not an exported
        mutable application contract. Callers must not alter this mapping.

    Notes
    -----
    Uses BaseModel's actual storage descriptor. Performs no conversion, lookup,
    source access or state capture; arbitrary Python remains outside a sandbox.
    """
    return BaseModel.__dict__["__dict__"].__get__(value, type(value))  # pylint: disable=unnecessary-dunder-call


class BoundedContract(CoreContract):
    """Bound explicit versioned artifacts independently of legacy readers.

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

    @model_validator(mode="before")
    @classmethod
    def _guard_structure(cls, value: Any) -> Any:
        """Refuse hostile Python/decoded graphs before nested old model code."""
        guard_graph(value, maximum_bytes=cls._maximum_bytes, maximum_items=HARD_GRAPH_ITEMS)
        return value

    @model_validator(mode="after")
    def _bounded_wire(self) -> Self:
        """Admit only a complete envelope readable under the same byte/depth cap."""
        bounded_json_bytes(canonical_json(self.model_dump(mode="json")), self._maximum_bytes - 1)
        return self

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Read a bounded strict new envelope without upgrading legacy evidence.

        Parameters
        ----------
        payload : str or bytes
            Complete UTF-8 JSON; acquisition of a file must also be bounded.

        Returns
        -------
        Self
            Validated concrete v1 root with its original declared semantics.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Oversized/deep/invalid JSON, duplicate keys, unknown versions or
            violated concrete invariants. No migration or I/O occurs.
        """
        return super().from_json(bounded_json_bytes(payload, cls._maximum_bytes))

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Apply the same bounded ingress to Pydantic's public JSON route.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete new envelope; bytearray is copied once before validation.
        **kwargs : object
            Existing guarded Pydantic options. Relaxed strictness or unknown-field
            handling still rejects through CoreContract.

        Returns
        -------
        Self
            Concrete validated root; no side effects or legacy reinterpretation.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid input/limit/version or a forbidden validation relaxation.
        """
        if isinstance(json_data, bytearray):
            json_data = bytes(json_data)
        return super().model_validate_json(bounded_json_bytes(json_data, cls._maximum_bytes), **kwargs)


def wire_size(value: Any) -> int:
    """Measure actual structured canonical UTF-8 bytes, excluding LF framing.

    Parameters
    ----------
    value : object
        Validated model or JSON-compatible boundary value. Model field serializers
        run so original DomainRecord omission semantics remain effective.

    Returns
    -------
    int
        Encoded bytes including JSON escaping, never a character count.

    Raises
    ------
    ValueError, TypeError
        Content cannot be canonically encoded as finite JSON. Performs no I/O.
    """
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return len(canonical_json(payload).encode("utf-8"))
