"""Capture and load an explicitly approved immutable directory-wheel executor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import Distribution
from pathlib import Path, PurePosixPath
from typing import Any, cast

from coleman.application.executor_capture_imports import (
    require_loaded_origins,
    require_standard_context,
    resolve_module_origins,
    resolve_origins,
)
from coleman.application.executor_capture_inventory import (
    CaptureCatalogue,
    DirectoryImage,
    record_paths,
)
from coleman.application.executor_capture_resources import ResourceBudget, _owned_paths, owned_files
from coleman.application.executor_capture_types import CAPTURE_PROFILE, CaptureLimits, require_capture_profile
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.core.reading import bounded_json_bytes
from coleman.objectives.bindings import DefinitionBinding
from coleman.spec.runtime_identity import _capture_environment, _dependency_distribution_roots
from coleman.spec.selected_runtime import SelectedInstalledRuntime, _normalized
from coleman.tasks.identity import (
    ApprovedExecutor,
    CodecDeclaration,
    CodecRole,
    ExecutorImplementationIdentity,
    ExecutorRecipe,
    InstalledFile,
)
from coleman.tasks.protocols import Executor

type LoadedExecutor = Executor[Any, Any, Any, Any, Any, Any]


@dataclass(frozen=True)
class _Captured:
    """Keep local origin checks outside the portable captured identity."""

    identity: ExecutorImplementationIdentity
    origins: dict[Path, str]


def capture_executor_identity(
    recipe: ExecutorRecipe, *, limits: CaptureLimits | None = None
) -> ExecutorImplementationIdentity:
    """Capture an approved recipe's declared installed closure without loading it.

    Parameters
    ----------
    recipe : ExecutorRecipe
        Independently operator-selected complete static recipe.
    limits : CaptureLimits or None, optional
        Finite acquisition capacities; defaults admit the initial CI/Hedge scope.

    Returns
    -------
    ExecutorImplementationIdentity
        Complete portable actual metadata/resource identity. Capturing does not
        approve a provider or construct its factory, learner or source loader.

    Raises
    ------
    ValueError, TypeError, OSError, ImportError
        Unsupported profile, missing/conflicting metadata, unsafe files or caps.

    Notes
    -----
    Requires CPython -B, optimization zero, standard import context and an
    immutable trusted directory installation. Existing source-cache bytes are
    captured; no caches are written or ignored. Known installed files do not
    attest undeclared dynamic imports, system libraries, ABI, hardware or code
    already mutated in memory. No URLs are fetched and no packages installed.
    Changed metadata/cache bytes, including relocation-dependent bytecode, can
    change this exact image identity even when source/version labels match.
    Create the environment with stdlib venv; uv/virtualenv helper meta-path
    finders are unsupported by this initial explicitly bounded profile.
    """
    return _capture(recipe, limits).identity


def require_executor_identity(
    expected: ExecutorImplementationIdentity, *, recipe: ExecutorRecipe, limits: CaptureLimits | None = None
) -> None:
    """Recapture exact current image before load/work/restore without migration.

    Parameters
    ----------
    expected : ExecutorImplementationIdentity
        Full retained capture, never a workspace-selected import approval.
    recipe : ExecutorRecipe
        Actual current operator-selected recipe, independently supplied.
    limits : CaptureLimits or None, optional
        Finite metadata/resource acquisition capacities.

    Raises
    ------
    ValueError, TypeError, OSError, ImportError
        Recipe mismatch, unsupported/missing/changed installed closure or caps.
    """
    retained = ExecutorImplementationIdentity.model_validate(expected)
    current_recipe = ExecutorRecipe.model_validate(recipe)
    if retained.recipe != current_recipe:
        raise ValueError("Retained executor capture differs from the current approved recipe")
    if _capture(current_recipe, limits).identity != retained:
        raise ValueError("Installed executor image differs from its retained complete capture")


def load_approved_executor(
    approval: ApprovedExecutor,
    *,
    expected: ExecutorImplementationIdentity | None = None,
    limits: CaptureLimits | None = None,
) -> LoadedExecutor:
    """Load a current operator-approved image and verify its concrete codec roles.

    Parameters
    ----------
    approval : ApprovedExecutor
        Current trusted host recipe and approved capture binding, never authority
        obtained from a plan, workspace settings or a public request.
    expected : ExecutorImplementationIdentity or None, optional
        Retained full plan capture. Its recipe/binding must match before effects.
    limits : CaptureLimits or None, optional
        Finite metadata/resource acquisition capacities.

    Returns
    -------
    Executor
        Actual frozen/read-only structural executor. Concrete codec declarations
        match the recipe before prepare/math/source/restore callbacks can run.

    Raises
    ------
    ValueError, TypeError, OSError, ImportError
        Approval/image/origin/entry-point mismatch or invalid bootstrap/codec.

    Notes
    -----
    Ordinary approved module initialization and its zero-argument bootstrap are
    trusted code. This is not a sandbox or live-memory attestation. Loading is
    outside metadata transactions; workers independently recapture before work.
    """
    current = ApprovedExecutor.model_validate(approval)
    if expected is not None:
        retained = ExecutorImplementationIdentity.model_validate(expected)
        if retained.recipe != current.recipe or retained.binding != current.implementation:
            raise ValueError("Retained executor identity does not match current host approval")
    captured = _capture(current.recipe, limits)
    if captured.identity.binding != current.implementation or (expected is not None and captured.identity != retained):
        raise ValueError("Actual installed executor identity does not match current host approval")
    selection = current.recipe.entry_point
    origins = resolve_origins(selection, captured.origins)
    module = import_module(selection.module)
    require_loaded_origins(origins)
    factory: Any = module
    for name in selection.attribute.split("."):
        factory = getattr(factory, name)
    if not callable(factory):
        raise TypeError("Approved executor entry point must be a zero-argument bootstrap")
    executor = cast(Callable[[], object], factory)()
    require_loaded_origins(origins)
    _require_executor(executor, current.recipe)
    return cast(LoadedExecutor, executor)


def _capture(recipe: ExecutorRecipe, limits: CaptureLimits | None) -> _Captured:
    """Reuse one metadata traversal and retain exact private origin locations."""
    require_capture_profile()
    require_standard_context()
    selected = ExecutorRecipe.model_validate(recipe)
    capacities = CaptureLimits.model_validate(CaptureLimits() if limits is None else limits)
    catalogue = CaptureCatalogue(capacities)
    roots = tuple((root.name, frozenset(root.extras)) for root in selected.roots)
    items = _dependency_distribution_roots(roots, qualify=catalogue.qualify, before_read=catalogue.before_read)
    resources = _resources(catalogue, items, capacities, selected)
    package_fingerprint = _package_fingerprint(catalogue, items, resources)
    runtime = _selected_runtime(catalogue, package_fingerprint)
    identity = ExecutorImplementationIdentity(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="executor_implementation",
        reference=Reference(id=selected.reference.id + "/installed-closure", version=CAPTURE_PROFILE),
        recipe=selected,
        runtime=runtime,
        distributions=catalogue.inventory(items),
        files=resources.files,
    )
    bounded_json_bytes(identity.to_json(), capacities.max_identity_bytes)
    return _Captured(identity, resources.origins)


def _selected_runtime(catalogue: CaptureCatalogue, package_fingerprint: str) -> SelectedInstalledRuntime:
    """Keep schema2's explicit COLEMAN-only closure despite provider roots."""
    extras = catalogue.selections["coleman"]
    runtime_items = _dependency_distribution_roots(
        (("coleman", extras),), qualify=catalogue.qualify, before_read=catalogue.before_read
    )
    environment = _capture_environment(
        {_normalized(item.metadata.get("Name")): item.version for item in runtime_items},
        package_fingerprint=package_fingerprint,
    )
    runtime = SelectedInstalledRuntime(
        schema_version=2,
        inventory_profile="selected-extras-v1",
        selected_extras=tuple(sorted(extras)),
        **environment.model_dump(),
    )
    return runtime


@dataclass(frozen=True)
class _Resources:
    """Keep actual file pins and unambiguous local origins together."""

    files: tuple[InstalledFile, ...]
    origins: dict[Path, str]


def _package_fingerprint(catalogue: CaptureCatalogue, items: tuple[Distribution, ...], resources: _Resources) -> str:
    """Reproduce the old SDK digest from already bounded actual resource pins."""
    origin = resolve_module_origins("coleman", "coleman", resources.origins)[0]
    item = next(value for value in items if _normalized(value.metadata.get("Name")) == "coleman")
    image = catalogue.image_for(item)
    prefix = origin.path.parent.relative_to(image.root).as_posix() + "/"
    modules = {}
    for pin in resources.files:
        if pin.distribution != "coleman" or not pin.path.startswith(prefix):
            continue
        relative = pin.path.removeprefix(prefix)
        if _old_sdk_inclusion(relative):
            modules["coleman/" + relative] = pin.sha256
    if "coleman/__init__.py" not in modules:
        raise ValueError("Executor capture must bind the actual complete preloaded COLEMAN package")
    return fingerprint({"package_version": item.version, "modules": modules})


def _old_sdk_inclusion(relative: str) -> bool:
    """Retain the original producer's hidden/cache/bytecode exclusions exactly."""
    return not relative.endswith((".pyc", ".pyo")) and not any(
        part.startswith(".") or part == "__pycache__" for part in PurePosixPath(relative).parts
    )


def _resources(
    catalogue: CaptureCatalogue, items: tuple[Distribution, ...], capacities: CaptureLimits, selected: ExecutorRecipe
) -> _Resources:
    """Join exclusive distribution roots before hashing their complete resources."""
    budget = ResourceBudget(capacities)
    pins = []
    origins = {}
    for declaration in _resource_declarations(catalogue, items, capacities, selected):
        current = owned_files(declaration.item, declaration.image, declaration.paths, budget)
        _merge_origins(origins, declaration.image.root, current)
        pins.extend(current)
    return _Resources(tuple(pins), origins)


@dataclass(frozen=True)
class _Declaration:
    """Retain a bounded complete RECORD before any resource acquisition."""

    item: Distribution
    image: DirectoryImage
    paths: tuple[str, ...]


def _resource_declarations(
    catalogue: CaptureCatalogue, items: tuple[Distribution, ...], capacities: CaptureLimits, recipe: ExecutorRecipe
) -> tuple[_Declaration, ...]:
    """Reject shared roots and total declared expansion before hashing sources."""
    claims = {}
    declarations = []
    count = 0
    for item in items:
        image = catalogue.image_for(item)
        paths = record_paths(item, image, capacities)
        count += len(paths)
        if count > capacities.max_files:
            raise ValueError("Executor aggregate declared file count exceeds capture limits")
        _claim_roots(claims, item, image, paths)
        _require_entry_point(item, recipe)
        declarations.append(_Declaration(item, image, paths))
    return tuple(declarations)


def _claim_roots(claims: dict[Path, str], item: Distribution, image: DirectoryImage, paths: tuple[str, ...]) -> None:
    """Reserve each exact root for one distribution, refusing shared namespaces."""
    owner = _normalized(item.metadata.get("Name"))
    for top in {PurePosixPath(path).parts[0] for path in _owned_paths(item, paths)}:
        if top == "__pycache__":
            continue  # Standard flat-module caches are checked as exact files, never a shared-root walk.
        key = image.root / top
        if key in claims and claims[key] != owner:
            raise ValueError("Executor profile refuses shared or ambiguous installed resource roots")
        claims[key] = owner


def _merge_origins(origins: dict[Path, str], root: Path, files: tuple[InstalledFile, ...]) -> None:
    """Preserve unique actual ownership for subsequent static module checks."""
    for pin in files:
        absolute = (root / pin.path).resolve(strict=True)
        if absolute in origins and origins[absolute] != pin.distribution:
            raise ValueError("Executor profile refuses ambiguous installed file ownership")
        origins[absolute] = pin.distribution


def _require_entry_point(item, recipe: ExecutorRecipe) -> None:
    """Select only the approved distribution's exact metadata pointer, unloaded."""
    selected = recipe.entry_point
    if _normalized(item.metadata.get("Name")) != selected.distribution:
        return
    values = [entry for entry in item.entry_points if (entry.group, entry.name) == (selected.group, selected.name)]
    if len(values) != 1:
        raise ValueError("Executor requires one exact approved distribution entry point")
    entry = values[0]
    if entry.value != selected.module + ":" + selected.attribute:
        raise ValueError("Installed executor entry-point pointer differs from its approved static recipe")


def _require_executor(value: object, recipe: ExecutorRecipe) -> None:
    """Validate actual selected structural capabilities, never call prepare."""
    definition = DefinitionBinding.model_validate(getattr(value, "definition", None))
    if definition != recipe.binding or not callable(getattr(value, "prepare", None)):
        raise ValueError("Loaded executor definition/capabilities differ from its approved recipe")
    roles: dict[CodecRole, str] = {
        "configuration": "configuration_codec",
        "inputs": "input_codec",
        "coordinate": "coordinate_codec",
        "coordinate_result": "coordinate_result_codec",
        "final_result": "final_result_codec",
        "progress_proof": "progress_proof_codec",
    }
    for role, name in roles.items():
        actual = getattr(value, name, None)
        expected = recipe.codec_for(role)
        if expected is None:
            if actual is not None:
                raise ValueError("Loaded executor exposes an undeclared optional progress capability")
        elif actual is None or CodecDeclaration.model_validate(actual.declaration) != expected:
            raise ValueError("Loaded executor concrete codec differs from its approved role declaration")
        elif not all(callable(getattr(actual, operation, None)) for operation in ("validate", "decode", "encode")):
            raise ValueError("Loaded executor codec requires all declared concrete operations")
