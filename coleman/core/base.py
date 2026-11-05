"""Frozen shared bases and guarded readers for independently usable contracts."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from coleman.core.primitives import canonical_json, decode_json, fingerprint


class Immutable(BaseModel):
    """Strict immutable nested input with unchanged legacy validation semantics.

    Notes
    -----
    Unknown fields reject and nested model instances are revalidated. Concrete
    public boundaries declare immutable field types; arbitrary nested Python
    objects are not recursively frozen. Performs no I/O.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class VersionedContract(BaseModel):
    """Strict immutable v1 base retaining the existing artifact wire behavior.

    Attributes
    ----------
    schema_version : int
        Required integer 1. Missing, coerced and unknown versions reject.

    Notes
    -----
    Revalidates unchecked instances before serialization and performs no I/O.
    Legacy float projection and default-field serialization remain unchanged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]

    def to_json(self) -> str:
        """Serialize validated content as finite deterministic COLEMAN JSON v1.

        Returns
        -------
        str
            Canonical JSON including defaults, with array order retained.

        Raises
        ------
        ValueError, TypeError
            Invalid fields, versions, finite JSON or nested invariants.
        """
        validated = type(self).model_validate(self)
        return canonical_json(validated.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Read an explicit v1 artifact without migration or duplicate-key loss.

        Parameters
        ----------
        payload : str or bytes
            Complete JSON; acquisition code bounds bytes and nesting depth.

        Returns
        -------
        Self
            Frozen instance. Decimal fields retain exact tokens; legacy float
            fields retain their existing binary64 projection.

        Raises
        ------
        ValueError, TypeError
            Invalid JSON, version, unknown fields or typed invariants.
        """
        return cls.model_validate(decode_json(payload, exact_decimals=True))

    @property
    def fingerprint(self) -> str:
        """Return content SHA-256, distinct from logical identity or authenticity.

        Returns
        -------
        str
            Identified digest of canonical versioned content, without I/O.

        Raises
        ------
        ValueError, TypeError
            Revalidation or finite canonical serialization failed.
        """
        return fingerprint(json.loads(self.to_json()))


class _GuardedJSON(BaseModel):
    """Preserve numeric tokens and duplicate keys before new model validation."""

    @classmethod
    def model_validate(  # pylint: disable=too-many-arguments
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate a new boundary without relaxing its unknown-field contract.

        Parameters
        ----------
        obj : object
            Typed input or unchecked instance to validate and revalidate.
        strict : {True} or None, optional
            Default None retains declared strict fields and numeric guards.
            True adds Pydantic's strict Python-type requirements. False rejects
            because it would relax explicit version, count and boolean fields.
        extra : {'forbid'} or None, optional
            Unknown fields cannot be ignored or allowed at a core boundary.
        from_attributes : bool or None, optional
            Standard explicit host-controlled attribute extraction setting.
        context : object or None, optional
            Explicit host context passed only to validation.
        by_alias, by_name : bool or None, optional
            Standard Pydantic field-name selection.

        Returns
        -------
        Self
            Validated new contract with its declared immutable field types.

        Raises
        ------
        ValueError, TypeError
            Invalid inputs, unknown fields, or an attempted relaxation. This
            does not change retained legacy validation or perform I/O.
        """
        if strict is False:
            raise ValueError("Core contracts cannot relax declared strict fields")
        if extra not in {None, "forbid"}:
            raise ValueError("Core contracts forbid unknown fields")
        return super().model_validate(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_json(  # pylint: disable=too-many-arguments
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate new JSON without silently erasing nonzero numeric tokens.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete JSON; acquisition code bounds size and nesting depth.
        strict : {True} or None, optional
            Default None retains declared strict fields and numeric guards.
            False rejects. True applies strict Python validation after lossless
            token decoding, so JSON arrays cannot satisfy tuple fields; use the
            default for ordinary contract JSON with collection fields.
        extra : {'forbid'} or None, optional
            Unknown fields remain forbidden; relaxing this contract rejects.
        context : object or None, optional
            Explicit host validation context, never persisted or fabricated.
        by_alias, by_name : bool or None, optional
            Standard Pydantic field-name selection for typed validation.

        Returns
        -------
        Self
            Validated model using the same number guards as direct construction.

        Raises
        ------
        ValueError, TypeError
            Malformed JSON, duplicate/unknown keys, invalid typed content or an
            attempted unknown-field relaxation. No I/O or migration occurs.
        """
        if strict is False:
            raise ValueError("Core JSON contracts cannot relax declared strict fields")
        if extra not in {None, "forbid"}:
            raise ValueError("Core JSON contracts forbid unknown fields")
        if isinstance(json_data, bytearray):
            json_data = bytes(json_data)
        payload = decode_json(json_data, exact_decimals=True)
        return cls.model_validate(
            payload,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )


class CoreImmutable(_GuardedJSON, Immutable):
    """New nested boundary with guarded duplicate-key and numeric-token reading.

    Notes
    -----
    Separate from the legacy base, so stronger new ingress never silently
    changes old readers. Concrete boundaries still declare immutable fields.
    """


class CoreContract(_GuardedJSON, VersionedContract):
    """Versioned generic boundary with guarded JSON and retained canonical codecs.

    Notes
    -----
    Explicit version 1 and inherited strict/frozen invariants apply. The guarded
    Pydantic JSON route preserves Decimal tokens until each numeric role checks
    projection. No runtime, registration or storage is required.
    """
