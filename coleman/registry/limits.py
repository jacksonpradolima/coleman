"""Finite admission for optional registry contracts without runtime callbacks."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.reading import bounded_json_bytes


class RegistryLimits(CoreImmutable):
    """Select coupled count and encoded-byte ceilings for one population.

    Attributes
    ----------
    maximum_definitions : int, optional
        Retained immutable versions, default256, hard4096.
    maximum_facts : int, optional
        Accepted lifecycle facts including unknown knowledge, default/hard4096.
    maximum_producers : int, optional
        Retained compact opportunity seals, default4096, hard8192. A collection
        may therefore retain the public32 times95 opportunities when bytes fit.
    maximum_resources : int, optional
        Explicit ownership metadata entries, default256, hard4096.
    maximum_publications : int, optional
        Whole-checkpoint complete catalogue publications, default256, hard4096;
        separate from retained member definition versions within one registry.
    maximum_prefix_work : int, optional
        Logical fact-entry work: sum of distinct retained lifecycle-prefix
        lengths validated by one state read, default65536, hard1048576.
        Checked before prefix copies/hashes; each distinct prefix is cached
        once. This is not a summed encoded-byte or execution-time budget;
        complete encoded bytes are independently bounded by maximum_bytes.
    maximum_checkpoint_work : int, optional
        Logical whole-checkpoint origin/fact and publication/leaf comparison
        work, default65536, hard1048576. Separate from compact producer-prefix
        validation; it is not an encoded-byte or execution-time budget.
    maximum_bytes : int, optional
        Complete encoded contract ceiling, default2MiB, hard16MiB.

    Notes
    -----
    Counts are per supplied contract/population, never a Cartesian capacity
    promise. Existing collection32/128 owner and64MiB storage limits still
    apply externally. Refusal never evicts pending producers or tombstones.
    Construction performs no allocation of owners, callbacks or I/O.
    """

    maximum_definitions: Annotated[int, Field(strict=True, ge=1, le=4096)] = 256
    maximum_facts: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096
    maximum_producers: Annotated[int, Field(strict=True, ge=1, le=8192)] = 4096
    maximum_resources: Annotated[int, Field(strict=True, ge=1, le=4096)] = 256
    maximum_publications: Annotated[int, Field(strict=True, ge=1, le=4096)] = 256
    maximum_prefix_work: Annotated[int, Field(strict=True, ge=1, le=1048576)] = 65536
    maximum_checkpoint_work: Annotated[int, Field(strict=True, ge=1, le=1048576)] = 65536
    maximum_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024 * 1024)] = 2 * 1024 * 1024


class RegistryContract(CoreContract):
    """Reuse core v1 JSON with bounded registry ingress and complete output.

    Notes
    -----
    The16MiB hard ingress cap and core lexical depth64 check precede decoding;
    selected limits on containing records can be tighter. Inherited canonical
    JSON, explicit nulls, strict versions and duplicate-key rejection remain
    unchanged. No serializer, storage engine or migration is introduced.
    """

    def to_json(self) -> str:
        """Write core canonical v1 content within the same hard reader limits.

        Returns
        -------
        str
            Unchanged canonical JSON with explicit nulls, at most16MiB/depth64.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Core revalidation, encoded size, UTF-8 or depth admission fails.

        Notes
        -----
        Uses the existing core serializer; performs no mutation or I/O.
        """
        encoded = super().to_json()
        bounded_json_bytes(encoded, 16 * 1024 * 1024)
        return encoded

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Read bounded v1 content using the existing guarded core decoder.

        Parameters
        ----------
        payload : str or bytes
            Complete built-in UTF-8 JSON, at most16MiB and depth64.

        Returns
        -------
        Self
            Validated immutable contract without migration or registration.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Bounds, malformed/duplicate keys, future versions or invariants fail.

        Notes
        -----
        Performs no acquisition or I/O; callers also bound acquisition.
        """
        return super().from_json(bounded_json_bytes(payload, 16 * 1024 * 1024))

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
        """Bound the alternate core JSON ingress before guarded decoding.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete UTF-8 JSON at most16MiB and depth64. Bytearray is copied
            only after its byte count has been admitted.
        strict : {True} or None, optional
            Existing core strict-input mode; False cannot relax the contract.
        extra : {'forbid'} or None, optional
            Existing unknown-field policy; relaxation is rejected.
        context : object or None, optional
            Explicit caller validation context, never persisted.
        by_alias, by_name : bool or None, optional
            Standard field-name selection forwarded to core validation.

        Returns
        -------
        Self
            Validated immutable v1 content using unchanged core codecs.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Bounds, UTF-8, versions, duplicate keys or typed invariants fail.

        Notes
        -----
        Performs no acquisition, callbacks, migrations or I/O.
        """
        if type(json_data) is bytearray:  # pylint: disable=unidiomatic-typecheck
            if len(json_data) > 16 * 1024 * 1024:
                raise ValueError("Registry JSON exceeds the16MiB hard ingress cap")
            json_data = bytes(json_data)
        if not isinstance(json_data, str | bytes):
            raise TypeError("Registry reader requires built-in str, bytes or bytearray")
        encoded = bounded_json_bytes(json_data, 16 * 1024 * 1024)
        return super().model_validate_json(
            encoded, strict=strict, extra=extra, context=context, by_alias=by_alias, by_name=by_name
        )


def admit_bytes(value: CoreContract, limits: RegistryLimits) -> None:
    """Refuse complete encoded content above the selected finite ceiling.

    Parameters
    ----------
    value : CoreContract
        Already typed complete content; no factories are called.
    limits : RegistryLimits
        Explicit selected byte ceiling, revalidated before use.

    Raises
    ------
    ValueError
        Encoded content exceeds the selected ceiling or fails core validation.

    Notes
    -----
    Checks immutable model dumps without invoking ``to_json`` recursively.
    Performs no mutation, eviction, callbacks or I/O.
    """
    from coleman.core.primitives import canonical_json  # pylint: disable=import-outside-toplevel

    selected = RegistryLimits.model_validate(limits)
    if len(canonical_json(value.model_dump(mode="json")).encode("utf-8")) > selected.maximum_bytes:
        raise ValueError("Registry complete encoded content exceeds maximum_bytes")
