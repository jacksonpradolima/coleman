"""Measure an unchanged copied source tree without broad coverage path aliases."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import indent
from typing import Any
from uuid import uuid4

from coverage import Coverage, CoverageData

_RUN_OPTIONS = (
    "run:branch",
    "run:omit",
    "run:include",
    "run:concurrency",
    "run:patch",
    "run:core",
    "run:relative_files",
    "run:dynamic_context",
    "run:timid",
    "run:cover_pylib",
    "run:context",
    "run:plugins",
)


@dataclass(frozen=True)
class SourcePin:
    """Bind one complete non-cache source resource to its actual bytes."""

    relative: str
    size: int
    sha256: str


def _source_paths(root: Path) -> Iterator[Path]:
    """Enumerate complete regular resources without suppressing traversal errors."""
    directories = [root]
    while directories:
        for path in sorted(directories.pop().iterdir()):
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
                continue
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                directories.append(path)
            elif stat.S_ISREG(mode):
                yield path
            else:
                raise ValueError(f"Copied coverage refuses a non-regular source resource: {path.relative_to(root)}")


def _source_pin(root: Path, path: Path) -> SourcePin:
    """Stream one actual regular file into its complete size and SHA-256 pin."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return SourcePin(path.relative_to(root).as_posix(), size, digest.hexdigest())


def source_manifest(root: Path) -> tuple[SourcePin, ...]:
    """Hash every regular source resource, including non-Python resources.

    Parameters
    ----------
    root : pathlib.Path
        Actual source directory. Symlink resources and special files refuse.

    Returns
    -------
    tuple[SourcePin, ...]
        Complete sorted inventory. Only the existing copy recipe's bytecode
        caches are omitted; no source suffix filter or path alias is applied.

    Raises
    ------
    ValueError
        A source root/resource is not a real directory or regular file.
    OSError
        A resource cannot be read.
    """
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Copied coverage requires a real source directory")
    return tuple(sorted((_source_pin(root, path) for path in _source_paths(root)), key=lambda pin: pin.relative))


@dataclass(frozen=True)
class CopiedCoverage:
    """Retain exact source bindings and isolated raw data for one actual child."""

    original: Path
    copied: Path
    pins: tuple[SourcePin, ...]
    specification: Path
    parent_basename: Path
    branch: bool
    config_sha256: str

    @property
    def raw_basename(self) -> Path:
        """Keep raw data outside the parent's automatic parallel-file search."""
        return self.specification.parent / "raw-child"

    @property
    def config_file(self) -> Path:
        """Use the fixed source project's configuration rather than child cwd."""
        return self.original.parent / "pyproject.toml"

    def instrument(self, source: str, *, before: str) -> str:
        """Wrap the original child body in measurement with no new Python scope.

        Parameters
        ----------
        source : str
            Existing complete child program.
        before : str
            Unique exact boundary immediately before its first product import.

        Returns
        -------
        str
            Same original statements, with explicit startup and finally-save.

        Raises
        ------
        ValueError
            The product-import boundary is missing or ambiguous.
        """
        if source.count(before) != 1:
            raise ValueError("Copied coverage requires one exact child import boundary")
        prefix, boundary, remainder = source.partition(before)
        startup = (
            "\nfrom tests.copied_coverage import start_child_coverage, stop_child_coverage\n"
            f"_copied_collector = start_child_coverage({str(self.specification)!r})\ntry:\n"
        )
        return (
            prefix
            + startup
            + indent(boundary + remainder, "    ")
            + "\nfinally:\n    stop_child_coverage(_copied_collector)\n"
        )


def prepare_copied_coverage(original: Path, copied: Path, *, raw_parent: Path) -> CopiedCoverage | None:
    """Prepare measurement only when this parent actually has active coverage.

    Parameters
    ----------
    original, copied : pathlib.Path
        Complete current source tree and the actual independently imported copy.
    raw_parent : pathlib.Path
        Test-owned directory for a new raw-data/specification subdirectory.

    Returns
    -------
    CopiedCoverage or None
        Active exact-byte measurement, or no work at all for ``--no-cov``.

    Raises
    ------
    ValueError
        Inventories differ, sources overlap, or effective options are unsupported.
    OSError
        Source bytes, the fixed root config, or the test directory are unavailable.
    """
    active = Coverage.current()
    if active is None:
        return None
    pins = source_manifest(original)
    if not pins or source_manifest(copied) != pins:
        raise ValueError("Copied coverage source inventories or bytes differ")
    original, copied = original.resolve(), copied.resolve()
    if original == copied or original.is_relative_to(copied) or copied.is_relative_to(original):
        raise ValueError("Copied coverage requires separate source trees")
    if active.get_option("run:relative_files"):
        raise ValueError("Copied coverage requires absolute measured filenames")
    parent_basename = active.get_option("run:data_file")
    if not isinstance(parent_basename, str):
        raise ValueError("Copied coverage requires an actual parent data basename")
    config_file = original.parent / "pyproject.toml"
    config_sha256 = hashlib.sha256(config_file.read_bytes()).hexdigest()
    directory = Path(tempfile.mkdtemp(prefix="copied-coverage-", dir=raw_parent))
    raw_basename = directory / "raw-child"
    specification = directory / "specification.json"
    branch = bool(active.get_option("run:branch"))
    specification.write_text(
        json.dumps(
            {
                "source": [str(copied), str(original)],
                "config_file": str(config_file),
                "raw_basename": str(raw_basename),
                "run_options": {name: active.get_option(name) for name in _RUN_OPTIONS},
                "source_manifest": [asdict(pin) for pin in pins],
                "parent_basename": str(Path(parent_basename).absolute()),
                "config_sha256": config_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return CopiedCoverage(
        original,
        copied,
        pins,
        specification,
        Path(parent_basename).absolute(),
        branch,
        config_sha256,
    )


def start_child_coverage(specification: str) -> Coverage:
    """Start the public collector before any copied product module is imported.

    Parameters
    ----------
    specification : str
        Parent-authored exact source paths and effective public run options.

    Returns
    -------
    coverage.Coverage
        One started collector with a raw basename outside automatic combination.

    Notes
    -----
    The interpreter retains ``-I -S`` and its existing bytecode flags. This does
    not run site startup, call process_startup, or parse private serialized config.
    """
    settings: dict[str, Any] = json.loads(Path(specification).read_text(encoding="utf-8"))
    collector = Coverage(
        config_file=settings["config_file"],
        source=settings["source"],
        data_file=settings["raw_basename"],
        data_suffix=True,
    )
    for name, value in settings["run_options"].items():
        collector.set_option(name, value)
    collector.start()
    return collector


def stop_child_coverage(collector: Coverage) -> None:
    """Stop and save the same actual collector, including on exceptional exit.

    Parameters
    ----------
    collector : coverage.Coverage
        The collector started before the unchanged product imports.
    """
    collector.stop()
    collector.save()


def finish_copied_coverage(session: CopiedCoverage) -> Path:
    """Merge actual raw events only after complete before/after byte verification.

    Parameters
    ----------
    session : CopiedCoverage
        Exact source inventory retained before the actual child ran.

    Returns
    -------
    pathlib.Path
        Fresh active-parent parallel shard with root source names.

    Raises
    ------
    ValueError
        Source/config bytes changed, data is absent, mode differs, or a measured
        filename is outside the exact verified source inventory.

    Notes
    -----
    Public CoverageData.update preserves actual arcs, lines, contexts and tracer
    information. Raw data and manifest remain available; no event is synthesized.
    """
    if source_manifest(session.original) != session.pins or source_manifest(session.copied) != session.pins:
        raise ValueError("Copied coverage source inventory changed during execution")
    if hashlib.sha256(session.config_file.read_bytes()).hexdigest() != session.config_sha256:
        raise ValueError("Copied coverage root configuration changed during execution")
    mapping = {
        str(root / pin.relative): str(session.original / pin.relative)
        for root in (session.original, session.copied)
        for pin in session.pins
    }
    raw_paths = sorted(session.raw_basename.parent.glob(session.raw_basename.name + ".*"))
    if len(raw_paths) != 1:
        raise ValueError("Copied coverage requires exactly one actual child data file")
    raw = CoverageData(basename=raw_paths[0])
    raw.read()
    measured = raw.measured_files()
    if not measured or any(name not in mapping for name in measured):
        raise ValueError("Copied coverage contains an unexpected measured source file")
    if raw.has_arcs() != session.branch:
        raise ValueError("Copied coverage branch mode differs from the actual parent")
    if not any(raw.lines(name) for name in measured):
        raise ValueError("Copied coverage contains no actual executed source lines")
    # SQL update() does not populate coverage's automatic event-hash suffix.
    # A unique string suffix makes public combine() hash the actual file bytes.
    destination = CoverageData(basename=session.parent_basename, suffix=f"copied-{uuid4().hex}")
    destination.update(raw, map_path=mapping.__getitem__)
    destination.write()
    output = Path(destination.data_filename())
    (session.raw_basename.parent / "measurement.json").write_text(
        json.dumps(
            {
                "source_files": len(session.pins),
                "raw_file": str(raw_paths[0]),
                "raw_sha256": hashlib.sha256(raw_paths[0].read_bytes()).hexdigest(),
                "mapped_file": str(output),
                "mapped_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "measured_files": sorted(measured),
                "contexts": sorted(raw.measured_contexts()),
                "branch": raw.has_arcs(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output
