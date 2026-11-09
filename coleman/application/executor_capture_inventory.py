"""Capture only declared directory-wheel closures through bounded owned reads."""

from __future__ import annotations

import csv
import io
import stat
from dataclasses import dataclass
from importlib.metadata import Distribution, PathDistribution, distributions
from pathlib import Path

from coleman.application.executor_capture_types import CaptureLimits, FileImage, read_owned
from coleman.core.primitives import decode_json, fingerprint
from coleman.core.reading import bounded_json_bytes
from coleman.spec.runtime_identity import _active_requirements
from coleman.spec.selected_runtime import _metadata_signature, _normalized, _provided_extras
from coleman.tasks.identity import DistributionInventory

_METADATA_NAMES = ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "direct_url.json")


@dataclass(frozen=True)
class DirectoryImage:
    """Keep actual local acquisition facts separate from portable identities."""

    root: Path
    directory: str
    metadata: tuple[tuple[str, FileImage | None], ...]


class CaptureCatalogue:
    """Preflight metadata before the existing walker can parse actual objects."""

    def __init__(self, limits: CaptureLimits) -> None:
        """Initialize one finite capture without discovering unrelated packages.

        Parameters
        ----------
        limits : CaptureLimits
            Already validated finite metadata/resource capacities for this call.
        """
        self.limits = limits
        self.images: dict[Path, DirectoryImage] = {}
        self.selections: dict[str, frozenset[str]] = {}
        self.locations: dict[str, Path] = {}
        self.metadata_bytes = 0

    def before_read(self, item: Distribution) -> None:
        """Bound actual directory metadata before version/requirement parsing.

        Parameters
        ----------
        item : importlib.metadata.Distribution
            Actual acquired object, including variants and transitive lookups.

        Raises
        ------
        ValueError, OSError
            Unsupported metadata backend, symlinks, missing wheel metadata,
            editability, changed data or finite metadata acquisition exceeded.
        """
        directory, root = _directory(item)
        key = root / directory
        previous = self.images.get(key)
        if previous is None and len(self.images) >= self.limits.max_distributions:
            raise ValueError("Executor metadata acquisition count exceeds capture limits")
        remaining = self.limits.max_metadata_total_bytes - self.metadata_bytes
        ceiling = min(self.limits.max_metadata_bytes, remaining) if previous is None else self.limits.max_metadata_bytes
        captured_records = []
        for name in _METADATA_NAMES:
            value = self._metadata_file(root, directory, name, ceiling)
            captured_records.append((name, value))
            if previous is None and value is not None:
                remaining -= value.size
                ceiling = min(self.limits.max_metadata_bytes, remaining)
        records = tuple(captured_records)
        image = DirectoryImage(root, directory, records)
        if previous is not None:
            if image != previous:
                raise ValueError("Installed executor metadata changed during capture")
            return
        self.metadata_bytes += sum(value.size for _, value in records if value is not None)
        if self.metadata_bytes > self.limits.max_metadata_total_bytes:
            raise ValueError("Executor aggregate metadata bytes exceed capture limits")
        _require_wheel(image)
        _metadata_expansion(image, self.limits)
        self.images[key] = image

    def _metadata_file(self, root: Path, directory: str, name: str, ceiling: int) -> FileImage | None:
        """Acquire one fixed metadata name without permitting an unbounded read."""
        try:
            return read_owned(root, f"{directory}/{name}", maximum=ceiling, chunk=self.limits.chunk_bytes, retain=True)
        except FileNotFoundError:
            if name in {"METADATA", "WHEEL", "RECORD"}:
                raise ValueError("Executor directory-wheel profile requires METADATA, WHEEL and RECORD") from None
            return None

    def qualify(self, item: Distribution, extras: set[str]) -> None:
        """Check bounded metadata selections and refuse ambiguous installed roots.

        Parameters
        ----------
        item : importlib.metadata.Distribution
            Actual preflighted installed metadata object.
        extras : set of str
            Existing walker's current extra selection; copied before retention.

        Raises
        ------
        ValueError, OSError
            Unknown extras, duplicate locations, conflicting metadata, active
            direct-URL requirements or configured distribution capacity.
        """
        self.before_read(item)
        selected = frozenset(_normalized(value) for value in extras)
        if not selected <= _provided_extras(item):
            raise ValueError("Executor requested an undeclared installed extra")
        name = _normalized(item.metadata.get("Name"))
        directory, root = _directory(item)
        key = root / directory
        previous = self.locations.get(name)
        if previous is not None and previous != key:
            raise ValueError("Executor distribution has ambiguous installed metadata locations")
        if previous is None and len(self.locations) >= self.limits.max_distributions:
            raise ValueError("Executor active distribution count exceeds capture limits")
        self._variants(item, name, key)
        if any(requirement.url is not None for requirement in _active_requirements(item, set(selected))):
            raise ValueError("Executor directory-wheel profile does not support active direct-URL requirements")
        self.locations[name] = key
        self.selections[name] = self.selections.get(name, frozenset()) | selected

    def _variants(self, item: Distribution, name: str, expected: Path) -> None:
        """Preflight every actual variant before reading its metadata signature."""
        signature = _metadata_signature(item)
        for ordinal, other in enumerate(distributions(name=name)):
            if ordinal >= self.limits.max_distributions:
                raise ValueError("Executor named metadata variants exceed capture limits")
            self.before_read(other)
            directory, root = _directory(other)
            if root / directory != expected or _metadata_signature(other) != signature:
                raise ValueError("Executor distribution has ambiguous or conflicting installed metadata")

    def inventory(self, items: tuple[Distribution, ...]) -> tuple[DistributionInventory, ...]:
        """Record exact bounded metadata after the existing dependency walk.

        Parameters
        ----------
        items : tuple of Distribution
            Actual final acquired objects returned by the shared metadata walker.

        Returns
        -------
        tuple of DistributionInventory
            Portable exact selected metadata identities, without provider loads.
        """
        values = []
        for item in items:
            name = _normalized(item.metadata.get("Name"))
            image = self.image_for(item)
            digest = fingerprint({name: None if body is None else body.sha256 for name, body in image.metadata})
            values.append(
                DistributionInventory(
                    name=name,
                    version=item.version,
                    extras=tuple(sorted(self.selections[name])),
                    metadata_fingerprint=digest,
                )
            )
        return tuple(values)

    def image_for(self, item: Distribution) -> DirectoryImage:
        """Return already preflighted acquisition facts, never capture implicitly.

        Parameters
        ----------
        item : Distribution
            Actual object already processed by the acquisition hook.

        Returns
        -------
        DirectoryImage
            Exact local metadata root and bounded original bodies.
        """
        directory, root = _directory(item)
        return self.images[root / directory]


def _directory(item: Distribution) -> tuple[str, Path]:
    """Restrict the versioned profile to actual standard directory wheels."""
    if type(item) is not PathDistribution:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Executor capture supports only standard directory-backed wheel metadata")
    raw_directory = item._path  # pylint: disable=protected-access
    if not isinstance(raw_directory, Path) or type(raw_directory) is not type(Path()):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Executor capture requires directory dist-info metadata, not zip/editable backends")
    directory = raw_directory
    if not directory.name.endswith(".dist-info"):
        raise ValueError("Executor capture requires directory dist-info metadata, not zip/editable backends")
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("Executor metadata directory must not be a symlink")
    location = item.locate_file("")
    if not isinstance(location, Path):
        raise ValueError("Executor wheel location must be a standard directory path")
    root = location.resolve(strict=True)
    if root != directory.parent.resolve(strict=True):
        raise ValueError("Executor metadata and owned resource roots disagree")
    return directory.name, root


def _require_wheel(image: DirectoryImage) -> None:
    """Reject editable metadata without resolving or following its origin URL."""
    bodies = dict(image.metadata)
    wheel = bodies["WHEEL"]
    if wheel is None or wheel.body is None:
        raise ValueError("Executor capture requires supported installed wheel metadata")
    versions = [
        line.split(b":", 1)[1].strip() for line in wheel.body.splitlines() if line.lower().startswith(b"wheel-version:")
    ]
    if versions != [b"1.0"]:
        raise ValueError("Executor capture requires supported installed wheel metadata version 1.0")
    direct = bodies["direct_url.json"]
    if direct is not None and direct.body is not None:
        value = decode_json(bounded_json_bytes(direct.body, direct.size))
        if not isinstance(value, dict):
            raise ValueError("Executor direct-origin metadata must be an object")
        source = value.get("dir_info")
        if source is not None and (not isinstance(source, dict) or source.get("editable", False) is not False):
            raise ValueError("Executor directory-wheel profile does not support editable installations")


def _metadata_expansion(image: DirectoryImage, limits: CaptureLimits) -> None:
    """Bound declaration rows before standard metadata/entry-point parsing."""
    for name, image_file in image.metadata:
        if image_file is None or image_file.body is None:
            continue
        lines = image_file.body.splitlines()
        if (
            name == "METADATA"
            and sum(line.lower().startswith(b"requires-dist:") for line in lines) > limits.max_requirements
        ):
            raise ValueError("Executor dependency declaration count exceeds capture limits")
        if name == "entry_points.txt" and sum(b"=" in line for line in lines) > limits.max_entry_points:
            raise ValueError("Executor entry-point declaration count exceeds capture limits")


def record_paths(item: Distribution, image: DirectoryImage, limits: CaptureLimits) -> tuple[str, ...]:
    """Read complete bounded RECORD rows without its lossy existing-file view.

    Parameters
    ----------
    item : Distribution
        Actual already preflighted directory metadata.
    image : DirectoryImage
        Original bounded metadata bodies.
    limits : CaptureLimits
        Finite row/file capacities.

    Returns
    -------
    tuple of str
        Exact declared RECORD paths; claims are not used as content hashes.

    Raises
    ------
    ValueError
        Missing, duplicate, malformed or excessive file inventory.
    """
    record = dict(image.metadata)["RECORD"]
    directory, root = _directory(item)
    if (directory, root) != (image.directory, image.root):
        raise ValueError("Executor RECORD and original metadata root do not match")
    if record is None or record.body is None:
        raise ValueError("Executor wheel requires a readable file inventory")
    paths = []
    for row in csv.reader(io.StringIO(record.body.decode("utf-8"), newline="")):
        if len(row) != 3 or not row[0] or len(row[0]) > 1024 or len(paths) >= limits.max_files:
            raise ValueError("Executor RECORD has invalid or excessive file declarations")
        paths.append(row[0])
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Executor RECORD requires unique nonempty paths")
    # Distribution.files silently filters missing paths through exists(), even
    # outside the installation. Raw rows retain missingness until scoped,
    # no-follow acquisition; RECORD sizes/hashes never substitute for reads.
    required = {f"{image.directory}/{name}" for name in ("METADATA", "WHEEL", "RECORD")}
    if not required <= set(paths):
        raise ValueError("Executor RECORD must declare its own complete wheel metadata")
    return tuple(paths)
