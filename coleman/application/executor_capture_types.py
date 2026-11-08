"""Finite limits and no-follow file reads for the installed executor profile."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.tasks.identity import InstalledFile

CAPTURE_PROFILE = "directory-wheel-closure-v1"


class CaptureLimits(CoreImmutable):
    """Bound metadata acquisition and exclusive installed-resource inventories.

    Attributes
    ----------
    max_distributions, max_files, max_entries : int, optional
        At most4096 active distributions,32768 files and65536 scoped entries.
    max_depth, max_requirements, max_entry_points : int, optional
        At most64 nested owned directories,16384 requirements and4096 metadata
        entry-point rows; declaration defaults are4096 and1024 respectively.
    max_file_bytes, max_total_bytes : int, optional
        At most1GiB per file and8GiB across actual captured resource bytes.
    max_metadata_bytes, max_metadata_total_bytes : int, optional
        At most8MiB per metadata body and128MiB across preflighted metadata.
    max_identity_bytes : int, optional
        Complete encoded identity ceiling, at most16MiB.
    chunk_bytes : int, optional
        Stream read size, between4KiB and1MiB; never an unbounded read.

    Notes
    -----
    Count maxima need not fit simultaneously. Limits bound acquisition and
    declared bytes, not wall time, hostile-code behavior or free disk capacity.
    """

    max_distributions: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096
    max_files: Annotated[int, Field(strict=True, ge=1, le=32768)] = 32768
    max_entries: Annotated[int, Field(strict=True, ge=1, le=65536)] = 65536
    max_depth: Annotated[int, Field(strict=True, ge=1, le=64)] = 64
    max_requirements: Annotated[int, Field(strict=True, ge=1, le=16384)] = 4096
    max_entry_points: Annotated[int, Field(strict=True, ge=1, le=4096)] = 1024
    max_file_bytes: Annotated[int, Field(strict=True, ge=1, le=1024**3)] = 1024**3
    max_total_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * 1024**3)] = 8 * 1024**3
    max_metadata_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * 1024**2)] = 2 * 1024**2
    max_metadata_total_bytes: Annotated[int, Field(strict=True, ge=1, le=128 * 1024**2)] = 64 * 1024**2
    max_identity_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024**2)] = 16 * 1024**2
    chunk_bytes: Annotated[int, Field(strict=True, ge=4096, le=1024**2)] = 1024**2

    @model_validator(mode="after")
    def _coupled(self) -> Self:
        """Require each acquisition ceiling to fit its enclosing allowance."""
        if self.max_files > self.max_entries:
            raise ValueError("Executor file capacity cannot exceed scoped entry capacity")
        if max(self.max_file_bytes, self.max_metadata_total_bytes) > self.max_total_bytes:
            raise ValueError("Executor per-file/metadata capacity cannot exceed total bytes")
        if self.max_metadata_bytes > self.max_metadata_total_bytes:
            raise ValueError("Executor metadata body capacity cannot exceed metadata total")
        return self


def require_capture_profile() -> None:
    """Require the explicitly admitted immutable CPython no-bytecode profile.

    Raises
    ------
    ValueError
        Not CPython, bytecode writes enabled, optimization/cache prefix changed,
        or no POSIX no-follow directory access. No process flags are modified.
    """
    if sys.implementation.name != "cpython" or not sys.dont_write_bytecode or sys.flags.optimize != 0:
        raise ValueError("Executor capture requires CPython with bytecode writes disabled (-B) and optimization zero")
    if sys.pycache_prefix is not None or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("Executor capture requires the standard cache location and POSIX no-follow directory support")


@dataclass(frozen=True)
class FileImage:
    """Hold a bounded actual content digest and optional small metadata body."""

    size: int
    sha256: str
    body: bytes | None


def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Compare actual inode/content-clock facts during one immutable read."""
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def read_owned(root: Path, relative: str, *, maximum: int, chunk: int, retain: bool = False) -> FileImage:
    """Stream a regular owned file through no-follow directory descriptors.

    Parameters
    ----------
    root : pathlib.Path
        Actual resolved trusted installation root, not a request-selected path.
    relative : str
        Normalized POSIX path under that root, without traversal or symlinks.
    maximum, chunk : int
        Finite byte ceiling and fixed bounded stream size.
    retain : bool, optional
        Retain bounded small metadata bytes; resource hashing retains no body.

    Returns
    -------
    FileImage
        Actual length/hash and optional bytes, with stable inode/size/clocks.

    Raises
    ------
    ValueError, OSError
        Unsafe/nonregular/missing/changed file, exceeded bytes or unsupported OS.

    Notes
    -----
    No-follow checks narrow accidental races. The host still requires an
    immutable trusted installation; this is not a malicious-writer sandbox.
    """
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).as_posix() != relative or any(p in {".", ".."} for p in parts):
        raise ValueError("Executor resource requires a normalized owned relative path")
    if PurePosixPath(relative).is_absolute() or "\\" in relative or "\x00" in relative:
        raise ValueError("Executor resource path escapes its owned root")
    with ExitStack() as stack:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        for part in parts[:-1]:
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, parent)
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            initial = os.fstat(stream.fileno())
            if not stat.S_ISREG(initial.st_mode) or initial.st_size > maximum:
                raise ValueError("Executor resource must be a bounded regular file")
            image = _stream_image(stream, maximum, chunk, retain)
            final = os.fstat(stream.fileno())
            named = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if _signature(initial) != _signature(final) or _signature(final) != _signature(named):
                raise ValueError("Executor resource changed during capture")
            if image.size != initial.st_size:
                raise ValueError("Executor resource length changed during capture")
            return image


def _stream_image(stream, maximum: int, chunk: int, retain: bool) -> FileImage:
    """Hash finite chunks without retaining resource bodies."""
    digest = hashlib.sha256()
    size = 0
    pieces = []
    while part := stream.read(min(chunk, maximum - size + 1)):
        size += len(part)
        if size > maximum:
            raise ValueError("Executor resource exceeds its byte ceiling")
        digest.update(part)
        if retain:
            pieces.append(part)
    return FileImage(size, "sha256:" + digest.hexdigest(), b"".join(pieces) if retain else None)


def file_pin(distribution: str, relative: str, image: FileImage) -> InstalledFile:
    """Create the existing pure inventory leaf from verified actual stream facts.

    Parameters
    ----------
    distribution, relative : str
        Canonical declared owner and portable relative resource name.
    image : FileImage
        Actual bounded stream facts, not RECORD size/hash assertions.

    Returns
    -------
    InstalledFile
        Pure immutable portable identity, without absolute host paths.
    """
    return InstalledFile(distribution=distribution, path=relative, size_bytes=image.size, sha256=image.sha256)
