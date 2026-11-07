"""Preserve original domain record identities and wire semantics in new evidence."""

from __future__ import annotations

import json
from typing import Any, Self

from pydantic import BaseModel, field_serializer, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.numeric import binary64
from coleman.core.primitives import Fingerprint, Identifier
from coleman.evaluation.design._wire import model_storage
from coleman.results.domain import DomainRecord


def _numeric_fields(value: Any, names: tuple[str, ...]) -> None:
    """Check supplied numeric fields before legacy coercion, preserving nulls."""
    if isinstance(value, BaseModel):
        value = model_storage(value)
    if not isinstance(value, dict):
        return
    for name in names:
        number = value.get(name)
        if number is not None:
            binary64(number)


def _numeric_record(record: Any) -> None:
    """Guard new raw JSON numeric tokens before the nested legacy float parser."""
    if isinstance(record, BaseModel):
        record = model_storage(record)
    if not isinstance(record, dict):
        return
    measurements = record.get("measurements", ())
    if isinstance(measurements, (list, tuple)):
        for measurement in measurements:
            _numeric_fields(measurement, ("value", "minimum"))
    _numeric_fields(record.get("learning_reward"), ("value",))


class OriginalRecordEvidence(CoreImmutable):
    """Retain an original #251 record without materializing absent defaults.

    Attributes
    ----------
    record_key : tuple of str
        Exact namespace/run/execution/decision identity of the original record.
    record_digest : str
        Digest of its original canonical DomainRecord.to_json representation.
    record : DomainRecord
        Original typed value. Its serializer preserves omitted optional fields
        versus explicit nulls, signed zero, statuses and supplied provenance.

    Raises
    ------
    ValueError
        Key/digest mismatch or a new JSON token requiring lossy integer promotion,
        nonzero underflow or nonfinite numeric interpretation.

    Notes
    -----
    The legacy DomainRecord reader is unchanged. Already parsed legacy values
    cannot recover their lost lexical input; old materialized enclosing artifacts
    require explicit imported provenance and cannot claim absent producer bytes.
    Construction does not read a file, transform a metric or grant admission.
    """

    record_key: tuple[Identifier, Identifier, Identifier, Identifier]
    record_digest: Fingerprint
    record: DomainRecord

    @model_validator(mode="before")
    @classmethod
    def _guard_numeric_ingress(cls, value: Any) -> Any:
        """Apply stronger token guards only at this new enclosing boundary."""
        fields = model_storage(value) if isinstance(value, BaseModel) else value
        if isinstance(fields, dict):
            _numeric_record(fields.get("record"))
        return value

    @field_serializer("record")
    def _original_wire(self, record: DomainRecord) -> dict[str, Any]:
        """Preserve actual legacy omission semantics, rather than expanding defaults."""
        return json.loads(record.to_json())

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Require the original content to agree with both supplied references."""
        if self.record.logical_key != self.record_key or self.record.digest != self.record_digest:
            raise ValueError("Original record differs from its exact key or digest")
        return self

    @property
    def record_bytes(self) -> int:
        """Return original canonical UTF-8 body bytes, excluding a framing LF."""
        return len(self.record.to_json().encode("utf-8"))

    @classmethod
    def from_record(cls, record: DomainRecord) -> Self:
        """Bind an explicitly supplied original record without capturing metadata.

        Parameters
        ----------
        record : DomainRecord
            Complete producer value; no physical/source evidence is invented.

        Returns
        -------
        Self
            Validated exact original identity, digest and omission-preserving data.

        Raises
        ------
        ValueError
            Invalid original data or an unchecked incompatible nested value.
        """
        _numeric_record(record)
        record = DomainRecord.model_validate(record)
        return cls(record_key=record.logical_key, record_digest=record.digest, record=record)
