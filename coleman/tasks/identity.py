"""Declarative installed-executor identities without capture or provider imports."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.tasks._base import MAXIMUM_BYTES, TaskContract

CanonicalName = Annotated[str, Field(strict=True, min_length=1, max_length=256, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
DottedName = Annotated[
    str, Field(strict=True, min_length=1, max_length=512, pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
]
ShortText = Annotated[str, Field(strict=True, min_length=1, max_length=256)]
CodecRole = Literal[
    "configuration", "inputs", "coordinate", "coordinate_result", "final_result", "state", "progress_proof"
]


class CodecDeclaration(CoreImmutable):
    """Identify one selected concrete codec without importing or invoking it.

    Attributes
    ----------
    binding : DefinitionBinding
        Exact codec implementation/semantic definition.
    schema_fingerprint : str
        Digest of the selected concrete schema; not its import authority.
    maximum_bytes : int
        Positive UTF-8 body ceiling, at most 16 MiB.
    """

    binding: DefinitionBinding
    schema_fingerprint: Fingerprint
    maximum_bytes: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_BYTES)]


class CodecSelection(CoreImmutable):
    """Assign a codec to one explicit capability role, including optional state."""

    role: CodecRole
    codec: CodecDeclaration


class DistributionRoot(CoreImmutable):
    """Freeze a canonical installed distribution and its requested root extras."""

    name: CanonicalName
    extras: Annotated[tuple[CanonicalName, ...], Field(max_length=128)] = ()

    @model_validator(mode="after")
    def _extras(self) -> Self:
        """Reject duplicate extras and canonicalize only set enumeration."""
        if len(self.extras) != len(set(self.extras)):
            raise ValueError("Distribution root extras must be unique")
        object.__setattr__(self, "extras", tuple(sorted(self.extras)))
        return self


class EntryPointSelection(CoreImmutable):
    """Declare an operator-approved entry point; metadata never approves itself.

    Attributes
    ----------
    distribution, group, name : str
        Exact installed distribution and metadata entry-point selector.
    module, attribute : str
        Expected static Python module and attribute path, checked before load.
    """

    distribution: CanonicalName
    group: ShortText
    name: ShortText
    module: DottedName
    attribute: DottedName


class ExecutorRecipe(TaskContract):
    """Freeze the declared code/codec/frontier recipe before runtime capture.

    Attributes
    ----------
    reference, lineage : Reference, DefinitionLineage
        Versioned definition and supplied provenance, never authenticated.
    entry_point : EntryPointSelection
        Explicit operator-selected installed implementation.
    roots : tuple of DistributionRoot
        Complete declared dependency roots, including COLEMAN and the provider.
    codecs : tuple of CodecSelection
        Five required value roles and optional state/progress-proof roles.
    frontier : DefinitionBinding
        Exact work/frontier semantics, independent of native build ordinals.
    effects : {'none', 'local_state', 'reconcilable_external'}
        Declared effect profile; a label does not establish reconciliation.
    """

    artifact_type: Literal["executor_recipe"]
    reference: Reference
    lineage: DefinitionLineage
    entry_point: EntryPointSelection
    roots: Annotated[tuple[DistributionRoot, ...], Field(min_length=1, max_length=128)]
    codecs: Annotated[tuple[CodecSelection, ...], Field(min_length=5, max_length=7)]
    frontier: DefinitionBinding
    effects: Literal["none", "local_state", "reconcilable_external"]

    @model_validator(mode="after")
    def _catalogue(self) -> Self:
        """Require complete unique roles and explicitly selected root identities."""
        names = [root.name for root in self.roots]
        if len(names) != len(set(names)) or not {"coleman", self.entry_point.distribution} <= set(names):
            raise ValueError("Executor roots require unique COLEMAN and provider distributions")
        roles = [item.role for item in self.codecs]
        required = {"configuration", "inputs", "coordinate", "coordinate_result", "final_result"}
        if len(roles) != len(set(roles)) or not required <= set(roles):
            raise ValueError("Executor codecs require unique complete value roles")
        if self.effects == "local_state" and not {"state", "progress_proof"} <= set(roles):
            raise ValueError("Owned local-state executors require state and progress-proof codecs")
        object.__setattr__(self, "roots", tuple(sorted(self.roots, key=lambda item: item.name)))
        object.__setattr__(self, "codecs", tuple(sorted(self.codecs, key=lambda item: item.role)))
        return self

    @property
    def binding(self) -> DefinitionBinding:
        """Identify the full recipe without capture, import or registry lookup."""
        return DefinitionBinding(reference=self.reference, definition_fingerprint=self.fingerprint)

    def codec_for(self, role: CodecRole) -> CodecDeclaration | None:
        """Return the exact declared codec role, or explicit absent capability.

        Parameters
        ----------
        role : str
            Known codec capability role.

        Returns
        -------
        CodecDeclaration or None
            Selected immutable codec, never a dynamically loaded implementation.
        """
        return next((item.codec for item in self.codecs if item.role == role), None)


class DistributionInventory(CoreImmutable):
    """Retain exact bounded installed metadata and the selected active extras.

    Attributes
    ----------
    name, version : str
        Canonical distribution identity and exact installed version text.
    extras : tuple of str
        Actual union of selected active extras, not all installed capabilities.
    metadata_fingerprint : str
        Digest of the captured metadata recipe, not package authenticity.
    """

    name: CanonicalName
    version: ShortText
    extras: Annotated[tuple[CanonicalName, ...], Field(max_length=128)]
    metadata_fingerprint: Fingerprint

    @model_validator(mode="after")
    def _selection(self) -> Self:
        """Keep unique extras in deterministic catalogue order."""
        if len(self.extras) != len(set(self.extras)):
            raise ValueError("Captured distribution extras must be unique")
        object.__setattr__(self, "extras", tuple(sorted(self.extras)))
        return self


class InstalledFile(CoreImmutable):
    """Bind an owned distribution-relative file to actual captured bytes.

    Attributes
    ----------
    distribution : str
        Exact owner in the captured installed dependency catalogue.
    path : str
        Portable normalized relative path; no absolute path or traversal.
    size_bytes, sha256 : int, str
        Actual content length and content digest, not a RECORD assertion.
    """

    distribution: CanonicalName
    path: Annotated[str, Field(strict=True, min_length=1, max_length=1024)]
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=1024**3)]
    sha256: Fingerprint

    @model_validator(mode="after")
    def _relative(self) -> Self:
        """Reject ambiguous filesystem spellings without resolving any path."""
        path = PurePosixPath(self.path)
        if "\\" in self.path or "\x00" in self.path or path.is_absolute() or path.as_posix() != self.path:
            raise ValueError("Installed inventory paths must be normalized relative POSIX paths")
        if not path.parts or any(part in {"", ".", ".."} for part in path.parts) or ":" in path.parts[0]:
            raise ValueError("Installed inventory paths cannot traverse or name a drive")
        return self


class ExecutorImplementationIdentity(TaskContract):
    """Capture a bounded installed closure independently of its future plan.

    Attributes
    ----------
    reference, recipe : Reference, ExecutorRecipe
        Captured identity name and complete preceding recipe.
    runtime : SelectedInstalledRuntime
        Existing COLEMAN runtime generation 2, with unchanged wire semantics.
    distributions, files : tuple
        Complete declared closure metadata and captured owned resource bytes.

    Notes
    -----
    Construction only checks consistency. Capture/loading separately verifies
    actual metadata, origins and immutable installation; no authenticity or
    arbitrary dynamic-code completeness follows from these declarations.
    """

    artifact_type: Literal["executor_implementation"]
    reference: Reference
    recipe: ExecutorRecipe
    runtime: SelectedInstalledRuntime
    distributions: Annotated[tuple[DistributionInventory, ...], Field(min_length=1, max_length=4096)]
    files: Annotated[tuple[InstalledFile, ...], Field(min_length=1, max_length=32768)]

    @model_validator(mode="after")
    def _inventory(self) -> Self:
        """Join unique bounded files and required root metadata without I/O."""
        names = [item.name for item in self.distributions]
        if len(names) != len(set(names)) or not {root.name for root in self.recipe.roots} <= set(names):
            raise ValueError("Executor inventory requires unique complete root distributions")
        versions = {item.name: item.version for item in self.distributions}
        if any(versions.get(item.name) != item.version for item in self.runtime.distributions):
            raise ValueError("Captured executor closure must contain its exact COLEMAN runtime closure")
        selections = {item.name: set(item.extras) for item in self.distributions}
        if any(not set(root.extras) <= selections[root.name] for root in self.recipe.roots):
            raise ValueError("Captured root extras must include every declared selection")
        if selections["coleman"] != set(self.runtime.selected_extras):
            raise ValueError("Captured COLEMAN extras must match the selected runtime generation")
        keys = [(item.distribution, item.path) for item in self.files]
        if len(keys) != len(set(keys)) or any(item.distribution not in versions for item in self.files):
            raise ValueError("Captured files require unique paths and known distribution owners")
        if set(names) != {item.distribution for item in self.files}:
            raise ValueError("Every captured distribution requires captured owned files or metadata")
        if sum(item.size_bytes for item in self.files) > 8 * 1024**3:
            raise ValueError("Captured executor files exceed the 8 GiB declared byte ceiling")
        object.__setattr__(self, "distributions", tuple(sorted(self.distributions, key=lambda item: item.name)))
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda item: (item.distribution, item.path))))
        return self

    @property
    def binding(self) -> DefinitionBinding:
        """Identify the complete captured declaration without trusting its origin."""
        return DefinitionBinding(reference=self.reference, definition_fingerprint=self.fingerprint)


class ApprovedExecutor(CoreImmutable):
    """Keep an internal bootstrap selector small; inventory stays in the plan."""

    recipe: ExecutorRecipe
    implementation: DefinitionBinding


class HostBootstrapProfile(TaskContract):
    """Bound operator-selected bootstrap approval without workspace authority.

    Attributes
    ----------
    reference : Reference
        Explicit immutable host profile generation.
    executors : tuple of ApprovedExecutor
        Small recipes and approved identity bindings, without file inventories.

    Notes
    -----
    Only a host-owned internal argv channel supplies this profile. Reading
    stored equality metadata never grants import or execution authority.
    """

    artifact_type: Literal["host_bootstrap"]
    reference: Reference
    executors: Annotated[tuple[ApprovedExecutor, ...], Field(min_length=1, max_length=16)]
    _maximum_bytes = 16 * 1024

    @model_validator(mode="after")
    def _approved(self) -> Self:
        """Reject ambiguous recipe/implementation IDs and canonicalize a set."""
        recipes = [item.recipe.reference.id for item in self.executors]
        implementations = [item.implementation.reference.id for item in self.executors]
        if len(recipes) != len(set(recipes)) or len(implementations) != len(set(implementations)):
            raise ValueError("Host bootstrap requires unique recipe and implementation IDs")
        object.__setattr__(self, "executors", tuple(sorted(self.executors, key=lambda item: item.recipe.reference.id)))
        return self

    @property
    def binding(self) -> DefinitionBinding:
        """Identify the small explicit operator profile without importing providers."""
        return DefinitionBinding(reference=self.reference, definition_fingerprint=self.fingerprint)
