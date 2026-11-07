"""Bounded optional acquisition/publication over the existing protected audit.

This module imports storage deliberately. Pure design, selection and projection
modules must not import it. Limits bound mediated work; trusted local parents,
cooperative writers and the existing ArtifactWriter durability contract apply.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import decode_json
from coleman.core.reading import bounded_json_bytes
from coleman.evaluation.protected.audit import EvaluationAudit
from coleman.evaluation.protected.comparison import FamilyReport
from coleman.evaluation.protected.contracts import FrozenEvaluation

from ._wire import guard_graph, model_storage


class ProtocolCapacityError(RuntimeError):
    """A bounded operation exhausted its declared acquisition/publication capacity.

    It deliberately does not inherit ValueError: the legacy CAS loop catches
    ValueError for immutable-name conflicts, whereas a capacity refusal must
    propagate immediately and must never be mistaken for a losing publication.
    """


@dataclass(frozen=True, slots=True)
class AuditCapacity:  # pylint: disable=too-many-instance-attributes
    """Selected private audit bounds, in UTF-8 bytes and item counts.

    These values are derived from a validated versioned protocol profile. A
    readable history may contain exactly ``receipts`` entries; publication of
    the next ordinal is a separate refusal. No disk-quota or callback sandbox
    guarantee follows from bounded byte acquisition.
    """

    receipts: int
    receipt_bytes: int
    freeze_bytes: int
    report_bytes: int
    directory_entries: int
    directory_bytes: int
    configuration_bytes: int
    candidates: int = 64
    hypotheses: int = 64
    inputs: int = 4096
    graph_items: int = 4_194_304

    def __post_init__(self) -> None:
        """Reject invalid internal limits before creating or reading any files."""
        # Coercing bool here would silently change a persisted count contract.
        if any(type(getattr(self, item.name)) is not int or getattr(self, item.name) <= 0 for item in fields(self)):  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Audit capacities require positive integer limits")
        if self.directory_entries < self.receipts + 2:
            raise ValueError("Receipt scanning must reserve publication-name headroom")
        if self.directory_bytes < (self.receipts + 2) * self.receipt_bytes:
            raise ValueError("A full acknowledged prefix and publication headroom must remain readable")


def read_bounded(path: Path, maximum_bytes: int) -> bytes:
    """Acquire at most limit+1 bytes before JSON parsing, despite stat/read races.

    Parameters
    ----------
    path : pathlib.Path
        Known local immutable artifact; symbolic links are unsupported.
    maximum_bytes : int
        Positive UTF-8 byte ceiling, excluding no framing bytes.

    Returns
    -------
    bytes
        Complete original bytes, after UTF-8 and depth checks.

    Raises
    ------
    ProtocolCapacityError
        The file exceeds its declared size/depth or is empty.
    OSError, ValueError
        Unsupported file/link, malformed UTF-8 or invalid capacity.

    Notes
    -----
    This is bounded acquisition, not hostile-filesystem authentication. A
    concurrent immutable writer can only publish a complete file; unsupported
    external mutation can make a read fail but cannot request an unbounded read.
    """
    if type(maximum_bytes) is not int or maximum_bytes <= 0:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Artifact byte capacity must be a positive integer")
    if path.is_symlink():
        raise OSError("Protocol artifacts do not follow symbolic links")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError("Protocol artifacts support regular files only")
        content = stream.read(maximum_bytes + 1)
    try:
        return bounded_json_bytes(content, maximum_bytes)
    except ValueError as error:
        raise ProtocolCapacityError("Protocol artifact exceeds its bounded UTF-8/depth profile") from error


def read_contract[Contract: BaseModel](
    path: Path, contract: type[Contract], maximum_bytes: int, graph_items: int
) -> Contract:
    """Bound acquired bytes and the selected graph before nested model validation.

    Parameters
    ----------
    path : pathlib.Path
        Known immutable local artifact; regular files only.
    contract : type
        Exact expected typed contract, never inferred from arbitrary wire names.
    maximum_bytes : int
        Complete UTF-8 JSON acquisition ceiling, including framing.
    graph_items : int
        Selected graph occurrence ceiling before existing nested validators run.

    Returns
    -------
    object
        Validated instance of the requested concrete contract.

    Raises
    ------
    ValueError, TypeError, OSError, ProtocolCapacityError
        Acquisition, strict JSON, selected graph or typed contract failure.
    """
    value = decode_json(read_bounded(path, maximum_bytes), exact_decimals=True)
    guard_graph(value, maximum_bytes=maximum_bytes, maximum_items=graph_items)
    return contract.model_validate(value)


@dataclass(frozen=True, slots=True)
class _DirectoryView:
    """One bounded scan, including bytes/names which are not accepted receipts."""

    paths: tuple[Path, ...]
    entries: int
    byte_count: int


def _directory_entry_bytes(entry: os.DirEntry[str], receipt_bytes: int) -> int:
    """Charge a regular file or a vanished temporary's conservative byte budget."""
    try:
        metadata = entry.stat(follow_symlinks=False)
    except FileNotFoundError:
        if not entry.name.endswith(".tmp"):
            raise
        return receipt_bytes
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("Protocol receipt directories support regular files only")
    return metadata.st_size


def _directory_view(root: Path, capacity: AuditCapacity) -> _DirectoryView:
    """Bound all flat directory entries/bytes before collecting receipt paths.

    Non-JSON and temporary files consume scan capacity but are never accepted
    receipts or deleted. A temporary name removed by a concurrent writer still
    consumes one entry and, when its size is unavailable, one receipt byte budget.
    Other missing names fail closed. Expected receipts are parsed later by the
    unchanged legacy chain validator. Concurrent additions may yield a conservative
    read failure; this scan is not a transaction or a physical storage-quota service.
    """
    paths = []
    entries = total = 0
    try:
        listing = os.scandir(root)
    except FileNotFoundError:
        return _DirectoryView((), 0, 0)
    with listing:
        for entry in listing:
            entries += 1
            if entries > capacity.directory_entries:
                raise ProtocolCapacityError("Protocol receipt directory entry capacity exceeded")
            total += _directory_entry_bytes(entry, capacity.receipt_bytes)
            if total > capacity.directory_bytes:
                raise ProtocolCapacityError("Protocol receipt directory byte capacity exceeded")
            if entry.name.endswith(".json"):
                paths.append(Path(entry.path))
                if len(paths) > capacity.receipts:
                    raise ProtocolCapacityError("Protocol receipt history capacity exceeded")
    return _DirectoryView(tuple(sorted(paths)), entries, total)


def bounded_paths(root: Path, capacity: AuditCapacity) -> list[Path]:
    """Return bounded names for the existing validator without a second scan.

    Parameters
    ----------
    root : pathlib.Path
        Flat receipt directory; absent means no known local reservations.
    capacity : AuditCapacity
        Actual scan count/byte limits, including foreign and temporary names.

    Returns
    -------
    list of pathlib.Path
        Sorted JSON names only. Invalid entries or capacity failures propagate.
    """
    return list(_directory_view(root, capacity).paths)


def publish_bounded(
    writer: ArtifactWriter,
    path: Path,
    content: str,
    maximum_bytes: int,
    *,
    receipt_capacity: AuditCapacity | None = None,
) -> None:
    """Publish complete bounded JSON through the existing immutable writer.

    Parameters
    ----------
    writer : ArtifactWriter
        Existing local atomic publisher; its durability contract applies.
    path : pathlib.Path
        Owned immutable destination below the writer's root.
    content : str
        Complete JSON including any framing newline.
    maximum_bytes : int
        Complete encoded UTF-8 ceiling.
    receipt_capacity : AuditCapacity or None, optional
        For receipt directories, also reserve this publication's temporary and
        final names/bytes. Other concurrent writers are not a disk quota.

    Raises
    ------
    ProtocolCapacityError
        Encoded content or directory headroom exceeds the selected profile.
    ValueError, OSError
        Immutable conflict or a filesystem failure; no partial final is accepted.
    """
    try:
        encoded = bounded_json_bytes(content, maximum_bytes)
    except (ValueError, TypeError) as error:
        raise ProtocolCapacityError("Publication exceeds its bounded profile") from error
    try:
        destination_mode = path.lstat().st_mode
    except FileNotFoundError:
        destination_mode = None
    if destination_mode is not None and not stat.S_ISREG(destination_mode):
        raise OSError("Protocol publication supports regular final files only")
    if receipt_capacity is not None:
        view = _directory_view(path.parent, receipt_capacity)
        additions = 1 if path.exists() else 2
        if view.entries + additions > receipt_capacity.directory_entries:
            raise ProtocolCapacityError("Receipt publication has no temporary-name headroom")
        if view.byte_count + additions * len(encoded) > receipt_capacity.directory_bytes:
            raise ProtocolCapacityError("Receipt publication has no temporary-byte headroom")
    writer.write_immutable_stream(path, (encoded,), maximum_bytes=maximum_bytes)


def _fields(value: Any) -> dict[str, Any]:
    """Inspect an already acquired mapping/model without invoking its serializers."""
    if isinstance(value, BaseModel):
        return model_storage(value)
    if type(value) is dict:  # pylint: disable=unidiomatic-typecheck
        return value
    raise ValueError("Expected a protected contract object")


def _count(value: Any, name: str, maximum: int) -> tuple[Any, ...] | list[Any]:
    """Reject oversized nested collections before any legacy typed validation."""
    values = _fields(value).get(name)
    if type(values) not in {tuple, list}:
        raise ValueError(f"Protected contract requires a {name} collection")
    checked = cast(tuple[Any, ...] | list[Any], values)
    if len(checked) > maximum:
        raise ProtocolCapacityError(f"Protected {name} collection exceeds its declared capacity")
    return checked


def guard_freeze(value: Any, capacity: AuditCapacity) -> None:
    """Preflight the whole legacy freeze and each encoded configuration string.

    Parameters
    ----------
    value : object
        Acquired decoded mapping or typed freeze, possibly unchecked.
    capacity : AuditCapacity
        Whole graph, candidate/family and nested configuration limits.

    Raises
    ------
    ValueError, TypeError, ProtocolCapacityError
        Malformed graph, excess count/bytes or invalid nested encoded JSON.

    Notes
    -----
    No model_dump/to_json or legacy model validation runs before this guard.
    Configuration strings also receive their own byte/depth check because JSON
    nesting inside a quoted configuration is invisible to the outer JSON depth.
    """
    guard_graph(value, maximum_bytes=capacity.freeze_bytes, maximum_items=capacity.graph_items)
    _count(value, "comparison_family", capacity.hypotheses)
    for candidate in _count(value, "candidates", capacity.candidates):
        content = _fields(candidate).get("configuration_json")
        if not isinstance(content, (str, bytes)):
            raise ProtocolCapacityError("Protected configuration requires encoded JSON")
        try:
            bounded_json_bytes(content, capacity.configuration_bytes)
        except (ValueError, TypeError) as error:
            raise ProtocolCapacityError("Protected configuration exceeds its bounded profile") from error


def guard_report(value: Any, capacity: AuditCapacity) -> None:
    """Bound the complete old report before inference/serialization validation.

    Parameters
    ----------
    value : object
        Acquired decoded mapping or typed FamilyReport, possibly unchecked.
    capacity : AuditCapacity
        Complete graph, input, result and nested freeze bounds.

    Raises
    ------
    ValueError, TypeError, ProtocolCapacityError
        Invalid structure or exceeded declared count/byte limits. No I/O occurs.
    """
    guard_graph(value, maximum_bytes=capacity.report_bytes, maximum_items=capacity.graph_items)
    _count(value, "inputs", capacity.inputs)
    _count(value, "results", capacity.hypotheses)
    guard_freeze(_fields(value).get("frozen"), capacity)


class BoundedAuditWriter(ArtifactWriter):
    """Admit each real legacy CAS publication using its existing storage engine.

    Parameters
    ----------
    root : pathlib.Path
        Trusted owned artifact root; construction performs no I/O.
    capacity : AuditCapacity
        Selected strict acquisition and publication bounds.
    """

    def __init__(self, root: Path, capacity: AuditCapacity) -> None:
        """Configure a private selected-profile writer without filesystem access."""
        super().__init__(root)
        self.capacity = capacity

    def write_immutable(self, path: Path, content: str) -> None:
        """Guard ordinal/bytes on every CAS retry, then use the real immutable writer.

        Parameters
        ----------
        path : pathlib.Path
            Known legacy receipt/freeze/report namespace below this root.
        content : str
            Complete bounded JSON, including its framing newline.

        Raises
        ------
        ValueError, OSError, ProtocolCapacityError
            Immutable conflict, invalid namespace/capacity or filesystem failure.

        Notes
        -----
        Capacity errors propagate through the parent access loop immediately.
        Prepublication scans reserve this writer's own temporary/final-name work;
        concurrent or foreign writers can still consume physical disk space.
        """
        parts = path.relative_to(self.run_root).parts
        if not parts or ".." in parts:
            raise ProtocolCapacityError("Protected artifact is outside its selected namespace")
        maximum = self._maximum(path, parts)
        publish_bounded(
            ArtifactWriter(self.run_root),
            path,
            content,
            maximum,
            receipt_capacity=self.capacity if parts[0] == "evaluation-access" else None,
        )

    def _maximum(self, path: Path, parts: tuple[str, ...]) -> int:
        """Refuse max+1 while leaving exactly-max receipt history readable."""
        if parts[0] == "evaluation-access":
            stem = path.stem
            if len(stem) != 12 or not stem.isascii() or not stem.isdecimal():
                raise ProtocolCapacityError("Protected receipt requires its original ordinal filename")
            if not 1 <= int(stem) <= self.capacity.receipts:
                raise ProtocolCapacityError("Protected scope access slots exhausted")
            return self.capacity.receipt_bytes
        if parts[0] == "evaluation-contracts":
            return self.capacity.freeze_bytes
        if parts[0] == "evaluation-reports":
            return self.capacity.report_bytes
        raise ProtocolCapacityError("Unknown protected artifact namespace")


class BoundedEvaluationAudit(EvaluationAudit):
    """Reuse legacy reservation/chain semantics with bounded actual acquisition.

    Parameters
    ----------
    root : pathlib.Path
        Trusted cooperative local root; construction performs no I/O.
    capacity : AuditCapacity
        Selected actual read/write bounds.

    Notes
    -----
    This selected new route leaves EvaluationAudit's default readers unchanged.
    No snapshot, storage, inference or reservation algorithm is duplicated.
    """

    def __init__(self, root: Path, capacity: AuditCapacity) -> None:
        """Configure bounded storage paths only; construction performs no I/O."""
        super().__init__(root)
        self.capacity = capacity
        self.writer = BoundedAuditWriter(root, capacity)

    def _history_paths(self, holdout_id: str) -> list[Path]:
        """Acquire bounded receipt names before the unchanged chain validator."""
        return bounded_paths(self._scope_root(holdout_id), self.capacity)

    def _receipt_bytes(self, path: Path) -> bytes:
        """Read at most receipt-limit+1 bytes before legacy receipt decoding."""
        return read_bounded(path, self.capacity.receipt_bytes)

    def _freeze_bytes(self, path: Path) -> bytes:
        """Bound complete bytes/configurations before legacy frozen-model decoding."""
        raw = read_bounded(path, self.capacity.freeze_bytes)
        guard_freeze(decode_json(raw, exact_decimals=True), self.capacity)
        return raw

    def save_freeze(self, frozen: FrozenEvaluation) -> Path:
        """Guard the whole typed freeze before validation and publication.

        Parameters
        ----------
        frozen : FrozenEvaluation
            Exact supported legacy contract, potentially unchecked on entry.

        Returns
        -------
        pathlib.Path
            Immutable destination. Validation, I/O and capacity errors propagate.
        """
        guard_freeze(frozen, self.capacity)
        return super().save_freeze(frozen)

    def save_report(self, report: FamilyReport) -> Path:
        """Guard the typed report and preserve actual receipt/freeze checks.

        Parameters
        ----------
        report : FamilyReport
            Complete legacy report whose statistics and reservation must match.

        Returns
        -------
        pathlib.Path
            Immutable destination. Validation, I/O and capacity errors propagate.
        """
        guard_report(report, self.capacity)
        return super().save_report(report)
