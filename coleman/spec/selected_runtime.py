"""Explicit selected-extra runtime generation; historical base inventories retain v1 meaning."""

from __future__ import annotations

import re
from importlib.metadata import Distribution, PackageNotFoundError, distributions
from typing import Annotated, Literal, Self, cast

from packaging.requirements import Requirement
from pydantic import Discriminator, Field, Tag, TypeAdapter, model_validator

from coleman.core.base import CoreContract
from coleman.core.primitives import decode_json
from coleman.spec.runtime_identity import (
    DistributionIdentity,
    InstalledRuntime,
    _capture_environment,
    _dependency_distributions,
    _RuntimeEnvironment,
)

_NAME = re.compile(r"\A[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_NATIVE_EXTRAS = ("ci",)
ExtraName = Annotated[str, Field(strict=True, min_length=1, max_length=256, pattern=_NAME)]
_ExtraSelection = Annotated[tuple[ExtraName, ...], Field(max_length=128)]


def _normalized(name: object) -> str:
    """Apply the Python Packaging separator/case rule without importing a provider."""
    if not isinstance(name, str):
        raise ValueError("Installed capability names must be strings")
    return re.sub(r"[-_.]+", "-", name.lower())


def _provided_extras(item: Distribution) -> frozenset[str]:
    """Validate metadata extra names before matching normalized requested extras."""
    names = tuple(_normalized(raw) for raw in item.metadata.get_all("Provides-Extra", ()))
    if any(not _NAME.fullmatch(name) for name in names) or len(names) != len(set(names)):
        raise ValueError("Installed capability has invalid or ambiguous extra metadata")
    return frozenset(names)


def _metadata_signature(item: Distribution) -> tuple[str, str, tuple[str, ...], frozenset[str], tuple[str, ...]]:
    """Compare relevant same-version duplicate metadata without claiming a wheel attestation."""
    return (
        _normalized(item.metadata.get("Name")),
        item.version,
        tuple(sorted(str(Requirement(raw)) for raw in item.requires or ())),
        _provided_extras(item),
        tuple(item.metadata.get_all("Default-Extra", ())),
    )


def _qualify_extras(item: Distribution, extras: set[str]) -> None:
    """Reject unknown selections and conflicting installed metadata before accepting their closure."""
    selected = {_normalized(name) for name in extras}
    if not selected <= _provided_extras(item):
        raise ValueError("Requested runtime capability extra is not declared by its installed distribution")
    signature = _metadata_signature(item)
    if any(_metadata_signature(other) != signature for other in distributions(name=item.metadata["Name"])):
        raise ValueError("Installed capability has conflicting dependency metadata")


class SelectedInstalledRuntime(CoreContract, _RuntimeEnvironment):
    """Recorded active base and explicitly selected COLEMAN extra dependency closure.

    Attributes
    ----------
    schema_version : int
        Required strict integer 2. This generation does not reinterpret v1.
    inventory_profile : {'selected-extras-v1'}
        Metadata-closure algorithm, independent of artifact schema generation.
    selected_extras : tuple[str, ...]
        Explicit canonical COLEMAN root extra names, sorted and unique; at most
        128. An empty selection records base dependencies only, never native CI.
    package_fingerprint : str
        Unchanged installed COLEMAN version/resource-byte fingerprint.
    python_implementation, python_version, operating_system, machine : str
        Recorded interpreter and platform identity; not a hardware attestation.
    distributions : tuple[DistributionIdentity, ...]
        At most 4096 unique canonical names and exact installed versions in the
        active selected closure, sorted by name. Unrelated packages are excluded.

    Notes
    -----
    Reading performs no I/O or installed-environment comparison. Capturing reads
    metadata/resources but never imports providers, installs packages or accesses
    the network. Name/version provenance does not attest all dependency binaries,
    ABI, toolchain state or upstream honesty. Historical v1 remains a distinct
    base-only contract. Execute only after separate native capability admission.
    """

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    inventory_profile: Literal["selected-extras-v1"]
    selected_extras: _ExtraSelection
    distributions: Annotated[tuple[DistributionIdentity, ...], Field(min_length=1, max_length=4096)]

    @model_validator(mode="before")
    @classmethod
    def _generation(cls, value):
        """Reject coerced generation tags even when an enclosing Pydantic reader relaxes strictness."""
        if not isinstance(value, dict):
            raise ValueError("Selected runtime requires a typed object")
        version = value.get("schema_version")
        if type(version) is not int or version != 2:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Selected runtime requires explicit integer schema_version 2")
        if value.keys() - cls.model_fields.keys():
            raise ValueError("Selected runtime forbids unknown fields")
        # Parent legacy readers may pass strict=False/extra='ignore'. Revalidate
        # these shared leaves independently so v2 never inherits that relaxation.
        environment = _RuntimeEnvironment.model_validate(
            {name: value[name] for name, _ in _RuntimeEnvironment.model_fields.items() if name in value}
        )
        extras = TypeAdapter(_ExtraSelection).validate_python(value.get("selected_extras"))
        return {**value, **environment.model_dump(), "selected_extras": extras}

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        """Reject ambiguous selections/inventories and canonicalize only their enumeration order."""
        names = [item.name for item in self.distributions]
        if len(self.selected_extras) != len(set(self.selected_extras)):
            raise ValueError("Selected runtime requires unique canonical root extras")
        if len(names) != len(set(names)) or any(not _NAME.fullmatch(name) for name in names) or "coleman" not in names:
            raise ValueError("Selected runtime requires unique canonical distribution names including coleman")
        object.__setattr__(self, "selected_extras", tuple(sorted(self.selected_extras)))
        object.__setattr__(self, "distributions", tuple(sorted(self.distributions, key=lambda item: item.name)))
        return self

    @classmethod
    def capture(cls, *, extras: tuple[str, ...]) -> SelectedInstalledRuntime:
        """Capture the actual installed closure of an explicit root selection.

        Parameters
        ----------
        extras : tuple[str, ...]
            Canonical unique COLEMAN extra names; no selection is inferred from
            packages that happen to be installed. Empty means base-only.

        Returns
        -------
        SelectedInstalledRuntime
            Generation 2 identity including the selection and active dependency
            closure. Equivalent selection/enumeration order produces equal bytes.

        Raises
        ------
        ValueError, TypeError, OSError, ImportError
            Invalid/unknown extras, missing or incompatible distributions,
            conflicting metadata, unsupported default extras, inventory bounds
            or unreadable installed resources. Nothing is installed or written.
        """
        selected = TypeAdapter(_ExtraSelection).validate_python(extras)
        if len(selected) != len(set(selected)):
            raise ValueError("Selected runtime requires unique canonical root extras")
        entries = _dependency_distributions(set(selected), qualify=_qualify_extras)
        return cls(
            schema_version=2,
            inventory_profile="selected-extras-v1",
            selected_extras=selected,
            **_capture_environment(entries).model_dump(),
        )


def _runtime_generation(value: object) -> str | None:
    """Dispatch exact known generation tags for dictionaries and revalidated model instances."""
    if isinstance(value, InstalledRuntime | SelectedInstalledRuntime):
        version = getattr(value, "schema_version", None)
    elif isinstance(value, dict):
        version = cast(dict[str, object], value).get("schema_version")
    else:
        return None
    if type(version) is not int:  # pylint: disable=unidiomatic-typecheck
        return None
    return {1: "runtime-v1", 2: "runtime-v2"}.get(version)


RuntimeIdentity = Annotated[
    Annotated[InstalledRuntime, Tag("runtime-v1")] | Annotated[SelectedInstalledRuntime, Tag("runtime-v2")],
    Discriminator(_runtime_generation),
]


def read_runtime(payload: str | bytes) -> RuntimeIdentity:
    """Read only explicit runtime generations 1 and 2 without migration or host inspection.

    Parameters
    ----------
    payload : str or bytes
        Complete JSON. Acquisition code bounds byte size and nesting depth.

    Returns
    -------
    InstalledRuntime or SelectedInstalledRuntime
        Historical base-only or explicit selected-extra identity, respectively.
        Canonical bytes/fingerprints retain their declared generation.

    Raises
    ------
    ValueError, TypeError
        Duplicate/unknown fields, malformed JSON, missing/coerced/unknown version
        or invalid inventory. Reading never qualifies native checkpoint reuse.
    """
    return TypeAdapter(RuntimeIdentity).validate_python(decode_json(payload, exact_decimals=True))


def capture_native_runtime() -> SelectedInstalledRuntime:
    """Capture generation 2 for the fixed native CI installation capability.

    Returns
    -------
    SelectedInstalledRuntime
        Exact active ``ci`` closure, excluding unrelated analytics/transports.

    Raises
    ------
    ValueError, ImportError, OSError
        Missing/invalid CI installation or unreadable identity. Install the
        verified COLEMAN wheel with its ``ci`` extra; no implicit install occurs.
    """
    try:
        return SelectedInstalledRuntime.capture(extras=_NATIVE_EXTRAS)
    except (ValueError, PackageNotFoundError) as error:
        raise ValueError(f"Native CI runtime unavailable; install the verified wheel with [ci]: {error}") from error


def require_native_runtime(runtime: RuntimeIdentity) -> None:
    """Require the fixed generation and actual native closure before executable state is opened.

    Parameters
    ----------
    runtime : InstalledRuntime or SelectedInstalledRuntime
        Frozen declared plan runtime. Historical v1 remains readable evidence;
        executing/resuming that generation requires its original pinned runtime.

    Raises
    ------
    ValueError, ImportError, OSError
        Historical/unknown profile, incomplete selection, changed installed
        closure or metadata/resource failure. No checkpoint, source outcome,
        storage mutation, provider import or native execution occurs here.
    """
    if not isinstance(runtime, SelectedInstalledRuntime) or runtime.selected_extras != _NATIVE_EXTRAS:
        raise ValueError("Saved runtime is incompatible with native v5; use pinned v4 or create a new CI plan")
    declared = SelectedInstalledRuntime.model_validate(runtime)
    if declared != capture_native_runtime():
        raise ValueError("Saved producer/runtime is incompatible with this native CI installation")
