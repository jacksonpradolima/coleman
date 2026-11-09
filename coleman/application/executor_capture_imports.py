"""Qualify standard import origins without importing an approved provider's parents."""

from __future__ import annotations

import sys
import types
import zipimport
from dataclasses import dataclass
from importlib.machinery import (
    BYTECODE_SUFFIXES,
    EXTENSION_SUFFIXES,
    SOURCE_SUFFIXES,
    BuiltinImporter,
    ExtensionFileLoader,
    FileFinder,
    FrozenImporter,
    ModuleSpec,
    PathFinder,
    SourceFileLoader,
    SourcelessFileLoader,
)
from pathlib import Path

from coleman.tasks.identity import EntryPointSelection

_LOADERS = (
    (ExtensionFileLoader, EXTENSION_SUFFIXES),
    (SourceFileLoader, SOURCE_SUFFIXES),
    (SourcelessFileLoader, BYTECODE_SUFFIXES),
)


def require_standard_context() -> None:
    """Refuse unsupported meta/path/cache machinery before metadata or imports.

    Raises
    ------
    ValueError
        Nonstandard finders/path hooks/cache entries or unbounded search paths.

    Notes
    -----
    This qualifies the initial trusted host context, not arbitrary later
    mutation by approved Python. It does not import a provider or call a custom
    finder to ask whether that finder would intercept a selected module.
    """
    expected_meta = (BuiltinImporter, FrozenImporter, PathFinder)
    if len(sys.meta_path) != 3 or any(
        actual is not expected for actual, expected in zip(sys.meta_path, expected_meta, strict=True)
    ):
        raise ValueError(
            "Executor profile requires standard CPython meta-path finders; use a stdlib venv without hooks"
        )
    if len(sys.path) > 4096 or any(type(path) is not str for path in sys.path):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Executor profile requires a finite standard string search path")
    if (
        len(sys.path_hooks) != 2
        or sys.path_hooks[0] is not zipimport.zipimporter
        or not _standard_hook(sys.path_hooks[1])
    ):
        raise ValueError("Executor profile requires standard CPython path hooks")
    if len(sys.path_importer_cache) > 8192:
        raise ValueError("Executor importer cache exceeds the supported profile")
    for finder in sys.path_importer_cache.copy().values():
        if finder is None or type(finder) is zipimport.zipimporter:  # pylint: disable=unidiomatic-typecheck
            continue
        if type(finder) is not FileFinder or not _standard_cached_loaders(finder):  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Executor profile refuses a custom cached path finder")


def _standard_hook(hook: object) -> bool:
    """Recognize the standard closure without invoking a supplied hook."""
    reference = FileFinder.path_hook(*_LOADERS)
    if not isinstance(hook, types.FunctionType) or not isinstance(reference, types.FunctionType):
        return False
    if hook.__code__ is not reference.__code__:
        return False
    cells = tuple(cell.cell_contents for cell in hook.__closure__ or ())
    if len(cells) != 2 or cells[0] is not FileFinder or not _plain(cells[1], tuple):
        return False
    values = cells[1]
    if len(values) != len(_LOADERS):
        return False
    return all(
        _plain(actual, tuple) and len(actual) == 2 and actual[0] is expected[0] and _suffixes(actual[1], expected[1])
        for actual, expected in zip(values, _LOADERS, strict=True)
    )


def _suffixes(actual: object, expected: list[str]) -> bool:
    """Compare standard suffix values; CPython rebuilds extension suffix lists."""
    if not isinstance(actual, list) or not _plain(actual, list) or len(actual) != len(expected):
        return False
    return all(_plain(value, str) and value == target for value, target in zip(actual, expected, strict=True))


def _plain(value: object, kind: type) -> bool:
    """Recognize exact standard context containers without subclass callbacks."""
    return type(value) is kind  # pylint: disable=unidiomatic-typecheck


def _standard_cached_loaders(finder: FileFinder) -> bool:
    """Compare fixed loader classes without calling arbitrary equality hooks."""
    actual = vars(finder).get("_loaders")
    expected = [(suffix, loader) for loader, suffixes in _LOADERS for suffix in suffixes]
    if type(actual) is not list or len(actual) != len(expected):  # pylint: disable=unidiomatic-typecheck
        return False
    return all(
        _plain(entry, tuple) and len(entry) == 2 and _plain(entry[0], str) and entry[0] == suffix and entry[1] is loader
        for entry, (suffix, loader) in zip(actual, expected, strict=True)
    )


@dataclass(frozen=True)
class ModuleOrigin:
    """Retain the exact standard origin expected before provider import."""

    name: str
    path: Path
    package_paths: tuple[str, ...] | None
    loader_type: type


def resolve_origins(selection: EntryPointSelection, files: dict[Path, str]) -> tuple[ModuleOrigin, ...]:
    """Resolve every regular parent/leaf using PathFinder without parent imports.

    Parameters
    ----------
    selection : EntryPointSelection
        Exact independently approved metadata selector and static module path.
    files : dict of pathlib.Path to str
        Actual captured absolute file origins and their unique owning distribution.

    Returns
    -------
    tuple of ModuleOrigin
        Parent-to-leaf regular origins; no namespace or sourceless modules.

    Raises
    ------
    ValueError
        Missing/shadowed/unowned/ambiguous module, dynamic loader or preloaded
        origin/path mismatch. No selected parent or provider is imported.
    """
    return resolve_module_origins(selection.module, selection.distribution, files)


def resolve_module_origins(module: str, distribution: str, files: dict[Path, str]) -> tuple[ModuleOrigin, ...]:
    """Resolve an actual named module, also used for the preloaded SDK itself.

    Parameters
    ----------
    module, distribution : str
        Exact static module and expected leaf owner, not an invented entry point.
    files : dict of pathlib.Path to str
        Actual unambiguous captured origins.

    Returns
    -------
    tuple of ModuleOrigin
        Qualified regular parent/leaf locations, without importing them.

    Raises
    ------
    ValueError
        Unsupported import context or origin/owner mismatch.
    """
    require_standard_context()
    search = None
    values = []
    parts = module.split(".")
    if len(parts) > 64:
        raise ValueError("Executor module nesting exceeds the supported profile")
    for ordinal in range(len(parts)):
        name = ".".join(parts[: ordinal + 1])
        spec = PathFinder.find_spec(name, search)
        value = _origin(name, spec, files)
        if ordinal == len(parts) - 1 and files[value.path] != distribution:
            raise ValueError("Executor entry-point module is not owned by its approved distribution")
        _require_preloaded(value)
        values.append(value)
        search = None if value.package_paths is None else list(value.package_paths)
        if ordinal < len(parts) - 1 and search is None:
            raise ValueError("Executor parent module is not a regular package")
    return tuple(values)


def _origin(name: str, spec: ModuleSpec | None, files: dict[Path, str]) -> ModuleOrigin:
    """Join a standard regular loader to uniquely captured actual file bytes."""
    if spec is None or not isinstance(spec.origin, str) or not _plain(spec.origin, str):
        raise ValueError("Executor profile requires regular source/extension origins, not namespace/dynamic modules")
    if type(spec.loader) not in {SourceFileLoader, ExtensionFileLoader}:
        raise ValueError("Executor profile requires regular source/extension origins, not namespace/dynamic modules")
    origin = Path(spec.origin).resolve(strict=True)
    if origin not in files:
        raise ValueError("Executor import origin is shadowed or outside its approved captured image")
    paths = spec.submodule_search_locations
    if paths is not None and (type(paths) is not list or any(type(path) is not str for path in paths)):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Executor package search locations must be standard immutable installation paths")
    return ModuleOrigin(name, origin, None if paths is None else tuple(paths), type(spec.loader))


def _require_preloaded(origin: ModuleOrigin) -> None:
    """Reject preloaded modules whose current static origin declarations differ."""
    if origin.name not in sys.modules:
        return
    module = sys.modules[origin.name]
    if type(module) is not types.ModuleType:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Executor profile refuses unknown preloaded module objects")
    fields = vars(module)
    spec = fields.get("__spec__")
    if type(spec) is not ModuleSpec:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Executor preloaded module requires its standard original spec")
    filename = fields.get("__file__")
    if type(filename) is not str or Path(filename).resolve(strict=True) != origin.path:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Executor preloaded module file differs from its approved origin")
    if not isinstance(spec.origin, str) or not _plain(spec.origin, str):
        raise ValueError("Executor preloaded loader/spec differs from its approved origin")
    if Path(spec.origin).resolve(strict=True) != origin.path or not _plain(spec.loader, origin.loader_type):
        raise ValueError("Executor preloaded loader/spec differs from its approved origin")
    actual = fields.get("__path__")
    if not _package_paths_match(actual, origin.package_paths):
        raise ValueError("Executor preloaded package changed its declared search locations")


def _package_paths_match(actual: object, expected: tuple[str, ...] | None) -> bool:
    """Compare only plain standard package paths without custom equality calls."""
    if expected is None:
        return actual is None
    return (
        isinstance(actual, list)
        and _plain(actual, list)
        and all(_plain(path, str) for path in actual)
        and tuple(actual) == expected
    )


def require_loaded_origins(origins: tuple[ModuleOrigin, ...]) -> None:
    """Recheck actual loaded origins before factory/concrete codec callbacks.

    Parameters
    ----------
    origins : tuple of ModuleOrigin
        Exact previously resolved captured parent/leaf origin declarations.

    Raises
    ------
    ValueError
        Approved code changed the supported import context or loaded origins.

    Notes
    -----
    Normal imports remain normal Python. Approved package initialization is
    trusted to respect the declared static profile; this is not a sandbox for
    arbitrary imports or live in-memory mutation by approved code.
    """
    require_standard_context()
    for origin in origins:
        if origin.name not in sys.modules:
            raise ValueError("Executor bootstrap did not load its declared complete module path")
        _require_preloaded(origin)
