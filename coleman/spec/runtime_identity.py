"""Recorded installed runtime identity, reusing the existing complete SDK resource fingerprint."""

from __future__ import annotations

import platform
import re
from collections.abc import Callable
from importlib.metadata import Distribution, distribution, distributions
from typing import Annotated, Self

from packaging.requirements import Requirement
from pydantic import Field, model_validator

from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.spec.producer_identity import installed_producer_identity

_DISTRIBUTION_SEPARATORS = r"[-_.]+"


class DistributionIdentity(Immutable):
    """One installed distribution's canonical project name/version, not a cryptographic wheel attestation."""

    name: Identifier
    version: Identifier


class _RuntimeEnvironment(Immutable):
    """Common recorded fields only; each concrete runtime retains its own generation and inventory meaning."""

    package_fingerprint: Fingerprint
    python_implementation: Identifier
    python_version: Identifier
    operating_system: Identifier
    machine: Identifier
    distributions: Annotated[tuple[DistributionIdentity, ...], Field(max_length=4096)]


def _capture_environment(entries: dict[str, str]) -> _RuntimeEnvironment:
    """Capture unchanged package/platform fields for one already qualified metadata closure."""
    return _RuntimeEnvironment(
        package_fingerprint=installed_producer_identity(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        operating_system=platform.system(),
        machine=platform.machine(),
        distributions=tuple(DistributionIdentity(name=name, version=value) for name, value in entries.items()),
    )


class InstalledRuntime(VersionedContract, _RuntimeEnvironment):
    """Recorded Python/platform/base-dependency closure and complete COLEMAN SDK identity.

    This conservative identity prevents checkpoint reuse across recorded runtime
    changes. Only active COLEMAN base requirements are inspected; unrelated host
    packages and optional dev/docs/MCP installations are outside this inventory.
    It does not attest all
    native library binaries, OS/toolchain state, hardware or upstream honesty.
    The legacy producer fingerprint algorithm remains unchanged and separately
    accessible. No network, subject execution or module/plugin loading occurs.
    """

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        """Reject noncanonical/ambiguous distribution identities and normalize enumeration order."""
        names = [item.name for item in self.distributions]
        if len(names) != len(set(names)) or any(
            re.sub(_DISTRIBUTION_SEPARATORS, "-", name.lower()) != name for name in names
        ):
            raise ValueError("Installed runtime requires unique canonical distribution names")
        object.__setattr__(self, "distributions", tuple(sorted(self.distributions, key=lambda item: item.name)))
        return self

    @classmethod
    def capture(cls) -> InstalledRuntime:
        """Read actual package resources/distribution metadata; fail on ambiguous or unreadable identity.

        Imported distributions are not executed. Identical duplicate metadata is
        deduplicated; conflicting versions fail rather than silently choosing one.
        Metadata/resource/platform errors propagate. Bytecode/hidden SDK files
        retain the existing producer identity exclusions.
        """
        entries = _base_distributions()
        return cls(schema_version=1, **_capture_environment(entries).model_dump())


def _installed_distribution(name: str) -> Distribution:
    """Read only a named dependency; conflicting installed versions cannot certify an unambiguous runtime."""
    item = distribution(name)
    variants = {other.version for other in distributions(name=name)}
    if variants != {item.version}:
        raise ValueError("Installed dependency has conflicting versions")
    actual = item.metadata["Name"]
    if not isinstance(actual, str) or re.sub(_DISTRIBUTION_SEPARATORS, "-", actual.lower()) != name:
        raise ValueError("Installed dependency metadata disagrees with its requested identity")
    if item.metadata.get_all("Default-Extra"):
        raise ValueError("Runtime default-extra metadata requires explicit qualification")
    return item


def _active_requirements(item: Distribution, extras: set[str]) -> tuple[Requirement, ...]:
    """Select actual metadata's active base/requested-extra requirements without installing or resolving packages."""
    requirements = tuple(Requirement(raw) for raw in item.requires or ())
    return tuple(
        req
        for req in requirements
        if req.marker is None or any(req.marker.evaluate({"extra": extra}) for extra in extras | {""})
    )


def _base_distributions() -> dict[str, str]:
    """Inventory active base dependencies and check their installed versions within bounded traversal."""
    return _dependency_distributions(set())


def _dependency_distributions(
    root_extras: set[str], *, qualify: Callable[[Distribution, set[str]], None] | None = None
) -> dict[str, str]:
    """Traverse the same bounded metadata graph; v1 supplies no root extras or additional qualification."""
    requested: dict[str, set[str]] = {"coleman": set(root_extras)}
    visited: dict[str, frozenset[str]] = {}
    entries: dict[str, str] = {}
    pending = ["coleman"]
    while pending:
        name = pending.pop()
        extras = requested[name]
        if visited.get(name) == frozenset(extras):
            continue
        item = _installed_distribution(name)
        if qualify is not None:
            qualify(item, extras)
        entries[name] = item.version
        visited[name] = frozenset(extras)
        for requirement in _active_requirements(item, extras):
            _include_requirement(requirement, requested, pending)
    return entries


def _include_requirement(requirement: Requirement, requested: dict[str, set[str]], pending: list[str]) -> None:
    """Validate one active requirement and merge its extras into the pending graph.

    Updates ``requested`` and appends to ``pending`` only for a newly required
    distribution or expanded extra set. Version/metadata errors propagate before
    those updates; the unchanged inventory bounds are checked after the merge.
    The caller retains metadata requirement order and marker selection.
    """
    dependency = _qualified_dependency(requirement)
    previous = requested.get(dependency)
    combined = set(requirement.extras) | (previous or set())
    if previous is None or previous != combined:
        requested[dependency] = combined
        pending.append(dependency)
    if len(requested) > 4096 or len(combined) > 128:
        raise ValueError("Installed base dependency inventory exceeds its supported bounds")


def _qualified_dependency(requirement: Requirement) -> str:
    """Check an active requirement against its named installed distribution without scanning unrelated packages."""
    dependency = re.sub(_DISTRIBUTION_SEPARATORS, "-", requirement.name.lower())
    installed = _installed_distribution(dependency)
    if installed.version not in requirement.specifier:
        raise ValueError("Installed dependency does not satisfy its active runtime requirement")
    return dependency
