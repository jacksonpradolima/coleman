"""Recorded installed runtime identity, reusing the existing complete SDK resource fingerprint."""

from __future__ import annotations

import platform
import re
from collections.abc import Callable, Iterator
from importlib.metadata import Distribution, distribution, distributions
from typing import Annotated, Self

from packaging.requirements import Requirement
from pydantic import Field, model_validator

from coleman.results.domain import Fingerprint, Identifier
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable
from coleman.spec.producer_identity import installed_producer_identity

_DISTRIBUTION_SEPARATORS = r"[-_.]+"
_ROOT_NAME = re.compile(r"\A[a-z0-9]+(?:-[a-z0-9]+)*\Z")


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


def _capture_environment(
    entries: dict[str, str], *, package_fingerprint: Fingerprint | None = None
) -> _RuntimeEnvironment:
    """Capture unchanged platform fields with optional independently bounded pins.

    Parameters
    ----------
    entries : dict of str to str
        Already qualified canonical distribution names and exact versions.
    package_fingerprint : str or None, optional
        New caller's actual bounded SDK resource digest. None retains the
        original package acquisition and argument evaluation order unchanged.

    Returns
    -------
    _RuntimeEnvironment
        Existing exact environment fields; no runtime generation is changed.
    """
    return _RuntimeEnvironment(
        package_fingerprint=installed_producer_identity() if package_fingerprint is None else package_fingerprint,
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


def _installed_distribution(name: str, *, before_read: Callable[[Distribution], None] | None = None) -> Distribution:
    """Read only a named dependency; conflicting installed versions cannot certify an unambiguous runtime."""
    item = distribution(name)
    if before_read is None:
        variants = {other.version for other in distributions(name=name)}
    else:
        before_read(item)
        variants = set()
        for other in distributions(name=name):
            before_read(other)
            variants.add(other.version)
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
    return {
        name: item.version for name, item in _walk_distribution_roots({"coleman": set(root_extras)}, qualify=qualify)
    }


def _dependency_distribution_roots(
    roots: tuple[tuple[str, frozenset[str]], ...],
    *,
    qualify: Callable[[Distribution, set[str]], None] | None = None,
    before_read: Callable[[Distribution], None] | None = None,
) -> tuple[Distribution, ...]:
    """Inspect a declared finite installed dependency closure without importing providers.

    Parameters
    ----------
    roots : tuple of (str, frozenset of str)
        One to 4096 unique canonical distribution names and at most 128 canonical
        extras each. Declared root order is retained; dependencies use the legacy
        last-in-first-out traversal and active metadata requirement order.
    qualify : callable or None, optional
        Trusted per-visit metadata check receiving the actual distribution and
        current selected-extra set. Expanded extras cause another visit; callers
        retaining that selection must copy it. No provider approval is inferred.
    before_read : callable or None, optional
        Trusted preflight called on each acquired item and each named variant
        before version/metadata/requirements properties are read. Use it to
        bound actual metadata bytes. A qualifier making its own acquisitions
        must separately preflight those objects; this is not a global hook.

    Returns
    -------
    tuple of importlib.metadata.Distribution
        Unique distributions in first-visit order, retaining the last acquired
        object after an extra-set revisit. Includes every declared root.

    Raises
    ------
    ValueError, TypeError, ImportError, OSError
        Invalid root declarations, missing/incompatible metadata, active closure
        growth beyond 4096 distributions/128 extras, or trusted qualifier failure.
        Does not install packages, resolve URLs or load entry points/resources.
    """
    requested = _root_requests(roots)
    if before_read is None:
        acquired = _walk_distribution_roots(requested, qualify=qualify)
    else:
        acquired = _walk_distribution_roots(requested, qualify=qualify, before_read=before_read)
    return tuple(dict(acquired).values())


def _root_requests(roots: tuple[tuple[str, frozenset[str]], ...]) -> dict[str, set[str]]:
    """Validate new root declarations before reading any installed metadata."""
    if not isinstance(roots, tuple) or not 1 <= len(roots) <= 4096:
        raise ValueError("Dependency roots require one to 4096 declared entries")
    requested: dict[str, set[str]] = {}
    for root in roots:
        if not isinstance(root, tuple) or len(root) != 2:
            raise ValueError("Dependency roots require name/extra pairs")
        name, extras = root
        if not isinstance(name, str) or not _ROOT_NAME.fullmatch(name):
            raise ValueError("Dependency root names must be canonical")
        if not isinstance(extras, frozenset) or len(extras) > 128:
            raise ValueError("Dependency root extras require a bounded frozenset")
        if any(not isinstance(extra, str) or not _ROOT_NAME.fullmatch(extra) for extra in extras):
            raise ValueError("Dependency root extras must be canonical")
        if name in requested:
            raise ValueError("Dependency root names must be unique")
        requested[name] = set(extras)
    return requested


def _walk_distribution_roots(
    requested: dict[str, set[str]],
    *,
    qualify: Callable[[Distribution, set[str]], None] | None,
    before_read: Callable[[Distribution], None] | None = None,
) -> Iterator[tuple[str, Distribution]]:
    """Yield acquired metadata at the unchanged legacy version-recording boundary."""
    visited: dict[str, frozenset[str]] = {}
    pending = list(reversed(requested))
    while pending:
        name = pending.pop()
        extras = requested[name]
        if visited.get(name) == frozenset(extras):
            continue
        if before_read is None:
            item = _installed_distribution(name)
        else:
            item = _installed_distribution(name, before_read=before_read)
        if qualify is not None:
            qualify(item, extras)
        yield name, item
        visited[name] = frozenset(extras)
        for requirement in _active_requirements(item, extras):
            if before_read is None:
                _include_requirement(requirement, requested, pending)
            else:
                _include_requirement(requirement, requested, pending, before_read=before_read)


def _include_requirement(
    requirement: Requirement,
    requested: dict[str, set[str]],
    pending: list[str],
    *,
    before_read: Callable[[Distribution], None] | None = None,
) -> None:
    """Validate one active requirement and merge its extras into the pending graph.

    Updates ``requested`` and appends to ``pending`` only for a newly required
    distribution or expanded extra set. Version/metadata errors propagate before
    those updates; the unchanged inventory bounds are checked after the merge.
    The caller retains metadata requirement order and marker selection.
    """
    if before_read is None:
        dependency = _qualified_dependency(requirement)
    else:
        dependency = _qualified_dependency(requirement, before_read=before_read)
    previous = requested.get(dependency)
    combined = set(requirement.extras) | (previous or set())
    if previous is None or previous != combined:
        requested[dependency] = combined
        pending.append(dependency)
    if len(requested) > 4096 or len(combined) > 128:
        raise ValueError("Installed base dependency inventory exceeds its supported bounds")


def _qualified_dependency(
    requirement: Requirement, *, before_read: Callable[[Distribution], None] | None = None
) -> str:
    """Check an active requirement against its named installed distribution without scanning unrelated packages."""
    dependency = re.sub(_DISTRIBUTION_SEPARATORS, "-", requirement.name.lower())
    if before_read is None:
        installed = _installed_distribution(dependency)
    else:
        installed = _installed_distribution(dependency, before_read=before_read)
    if installed.version not in requirement.specifier:
        raise ValueError("Installed dependency does not satisfy its active runtime requirement")
    return dependency
