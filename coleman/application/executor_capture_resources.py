"""Enumerate only exclusive owned wheel roots with bounded no-follow reads."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import Distribution
from importlib.util import cache_from_source
from pathlib import Path, PurePosixPath

from coleman.application.executor_capture_inventory import DirectoryImage
from coleman.application.executor_capture_types import CaptureLimits, file_pin, read_owned
from coleman.spec.selected_runtime import _normalized
from coleman.tasks.identity import InstalledFile


@dataclass
class ResourceBudget:
    """Count every scoped entry and actual resource byte during one capture."""

    limits: CaptureLimits
    entries: int = 0
    files: int = 0
    bytes: int = 0


def owned_files(
    item: Distribution, image: DirectoryImage, paths: tuple[str, ...], budget: ResourceBudget
) -> tuple[InstalledFile, ...]:
    """Hash exclusive recorded roots and existing standard source-cache siblings.

    Parameters
    ----------
    item : Distribution
        Actual preflighted metadata whose console launchers are not executed.
    image : DirectoryImage
        Trusted directory root, not a public source path.
    paths : tuple of str
        Complete bounded declared RECORD names.
    budget : ResourceBudget
        Shared actual-byte/entry/file capacities across all active distributions.

    Returns
    -------
    tuple of InstalledFile
        Actual stream hashes for owned resources and supported existing caches.

    Raises
    ------
    ValueError, OSError
        Unsafe/unlisted/shared/missing files, symlinks, changed metadata or caps.

    Notes
    -----
    Only exact generated console/gui launchers may be excluded under this
    not-executed-launcher profile. No arbitrary out-of-root paths are acquired.
    """
    name = _normalized(item.metadata.get("Name"))
    declared = _owned_paths(item, paths)
    permitted = set(declared)
    for relative in declared:
        if relative.endswith(".py"):
            permitted.add(Path(cache_from_source(str(image.root / relative))).relative_to(image.root).as_posix())
    pins = []
    for top in sorted({PurePosixPath(path).parts[0] for path in declared} - {"__pycache__"}):
        for relative in _walk_root(image.root, top, budget):
            if relative not in permitted and not relative.startswith(image.directory + "/"):
                raise ValueError("Executor owned root contains an unlisted resource or unsupported cache")
            if budget.files >= budget.limits.max_files:
                raise ValueError("Executor captured file count exceeds capture limits")
            maximum = min(budget.limits.max_file_bytes, budget.limits.max_total_bytes - budget.bytes)
            actual = read_owned(image.root, relative, maximum=maximum, chunk=budget.limits.chunk_bytes)
            budget.files += 1
            budget.bytes += actual.size
            pins.append(file_pin(name, relative, actual))
    direct = sorted(path for path in permitted if PurePosixPath(path).parts[0] == "__pycache__")
    pins.extend(_flat_caches(image.root, name, direct, budget))
    _require_complete_files(image, declared, pins)
    return tuple(pins)


def _flat_caches(root: Path, owner: str, paths: list[str], budget: ResourceBudget) -> tuple[InstalledFile, ...]:
    """Check the real cache parent before interpreting a missing selected file."""
    if not paths:
        return ()
    with ExitStack() as stack:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        try:
            cache = os.open("__pycache__", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        except FileNotFoundError:
            return ()
        stack.callback(os.close, cache)
        return _selected_caches(cache, root, owner, paths, budget)


def _selected_caches(
    cache: int, root: Path, owner: str, paths: list[str], budget: ResourceBudget
) -> tuple[InstalledFile, ...]:
    """Read exact files under a no-follow parent without scanning shared caches."""
    pins = []
    for relative in paths:
        parts = PurePosixPath(relative).parts
        if len(parts) != 2 or parts[0] != "__pycache__":
            raise ValueError("Executor flat cache paths require their exact standard parent")
        try:
            mode = os.stat(parts[-1], dir_fd=cache, follow_symlinks=False).st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Executor selected cache must be a regular owned file")
        if budget.files >= budget.limits.max_files or budget.entries >= budget.limits.max_entries:
            raise ValueError("Executor selected caches exceed file/entry capture limits")
        maximum = min(budget.limits.max_file_bytes, budget.limits.max_total_bytes - budget.bytes)
        actual = read_owned(root, relative, maximum=maximum, chunk=budget.limits.chunk_bytes)
        budget.entries += 1
        budget.files += 1
        budget.bytes += actual.size
        pins.append(file_pin(owner, relative, actual))
    return tuple(pins)


def _require_complete_files(image: DirectoryImage, declared: tuple[str, ...], pins: list[InstalledFile]) -> None:
    """Refuse missing resources or metadata changed since bounded parsing."""
    found = {pin.path for pin in pins}
    if not set(declared) <= found:
        raise ValueError("Executor recorded resources are missing from the installed image")
    by_path = {pin.path: pin for pin in pins}
    for metadata_name, original in image.metadata:
        current = by_path.get(f"{image.directory}/{metadata_name}")
        if (current is None) != (original is None) or (
            original is not None
            and current is not None
            and (original.sha256, original.size) != (current.sha256, current.size_bytes)
        ):
            raise ValueError("Executor metadata changed between parsing and resource capture")


def _owned_paths(item: Distribution, paths: tuple[str, ...]) -> tuple[str, ...]:
    """Exclude only known not-executed generated launchers, never read outside."""
    launchers = {entry.name for entry in item.entry_points if entry.group in {"console_scripts", "gui_scripts"}}
    owned = []
    for raw in paths:
        path = PurePosixPath(raw)
        if ".." in path.parts:
            parts = list(path.parts)
            while parts and parts[0] == "..":
                parts.pop(0)
            if len(parts) == 2 and parts[0] == "bin" and parts[1] in launchers:
                continue
            raise ValueError("Executor RECORD path escapes its owned installation root")
        if path.is_absolute() or path.as_posix() != raw or "\\" in raw or not path.parts or ":" in path.parts[0]:
            raise ValueError("Executor RECORD paths must be normalized owned relative paths")
        if raw.endswith(".pth"):
            raise ValueError("Executor directory-wheel profile does not support executable path files")
        owned.append(raw)
    return tuple(owned)


def _walk_root(root: Path, top: str, budget: ResourceBudget):
    """Enumerate one declared exclusive root using no-follow descriptors."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        mode = os.stat(top, dir_fd=descriptor, follow_symlinks=False).st_mode
        if stat.S_ISREG(mode):
            budget.entries += 1
            if budget.entries > budget.limits.max_entries:
                raise ValueError("Executor scoped entry count exceeds capture limits")
            yield top
        elif stat.S_ISDIR(mode):
            child = os.open(top, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                yield from _walk_directory(child, top, budget)
            finally:
                os.close(child)
        else:
            raise ValueError("Executor owned roots cannot be symlinks or special files")
    finally:
        os.close(descriptor)


def _walk_directory(descriptor: int, relative: str, budget: ResourceBudget):
    """Keep recursion scoped to live directory descriptors and finite entries."""
    if len(PurePosixPath(relative).parts) > budget.limits.max_depth:
        raise ValueError("Executor owned directory depth exceeds capture limits")
    names = []
    with os.scandir(descriptor) as entries:
        for entry in entries:
            budget.entries += 1
            if budget.entries > budget.limits.max_entries:
                raise ValueError("Executor scoped entry count exceeds capture limits")
            names.append(entry.name)
    for name in sorted(names):
        mode = os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode
        child_path = f"{relative}/{name}"
        if stat.S_ISREG(mode):
            yield child_path
        elif stat.S_ISDIR(mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                yield from _walk_directory(child, child_path, budget)
            finally:
                os.close(child)
        else:
            raise ValueError("Executor owned roots cannot contain symlinks or special files")
