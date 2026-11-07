"""Immutable optional member catalogues and exact host-owned recipe bindings."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.identity import Reference, ordered_references
from coleman.core.primitives import Fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.registry.limits import RegistryContract, RegistryLimits, admit_bytes
from coleman.state.contracts import StateBinding

MemberKind = Literal["expert", "action"]


class MemberDefinition(RegistryContract):
    """Bind one immutable expert/action version to implementation and config.

    Attributes
    ----------
    kind : {'expert', 'action'}
        Local namespace; equal expert and action IDs do not collide.
    member : Reference
        Exact stable logical member and semantic version.
    implementation : DefinitionBinding
        Exact declared implementation content, without importing its code.
    configuration_fingerprint : str
        Digest of complete mathematical configuration, excluding performance.

    Notes
    -----
    Requires explicit schema1. Identity asserts supplied content, never source
    authenticity or method qualification. No runtime, registration or I/O.
    """

    kind: MemberKind
    member: Reference
    implementation: DefinitionBinding
    configuration_fingerprint: Fingerprint

    @property
    def binding(self) -> DefinitionBinding:
        """Return exact member/version and full immutable definition digest.

        Returns
        -------
        DefinitionBinding
            Binding usable by recipe and lifecycle contracts without lookup.

        Notes
        -----
        Revalidates through the inherited fingerprint codec; performs no I/O.
        """
        return DefinitionBinding(reference=self.member, definition_fingerprint=self.fingerprint)


class Catalogue(RegistryContract):
    """Retain bounded historical definitions in canonical namespace order.

    Attributes
    ----------
    reference : Reference
        Caller-owned publication identity, separate from child/task identity.
    definitions : tuple of MemberDefinition
        Exact historical versions ordered by kind, logical ID and version.
        Duplicate keys reject, including identical duplicates.

    Raises
    ------
    ValueError
        Duplicate definition keys, more than4096 leaves or more than16MiB.

    Notes
    -----
    Two historical versions of an ID may coexist; one active population may
    select only one. Performance is absent. No global catalogue or I/O exists.
    """

    reference: Reference
    definitions: Annotated[tuple[MemberDefinition, ...], Field(max_length=4096)]

    @model_validator(mode="after")
    def _canonical(self) -> Self:
        """Keep every historical leaf without collapsing conflicting keys."""
        keys = [(item.kind, item.member.id, item.member.version) for item in self.definitions]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate catalogue definition key or conflicting content")
        object.__setattr__(self, "definitions", tuple(sorted(self.definitions, key=_definition_key)))
        admit_bytes(self, RegistryLimits(maximum_bytes=16 * 1024 * 1024))
        return self

    def resolve(self, kind: MemberKind, member: Reference) -> MemberDefinition:
        """Resolve an exact supplied namespace/version without fallback.

        Parameters
        ----------
        kind : {'expert', 'action'}
            Required local namespace.
        member : Reference
            Exact logical member and historical version.

        Returns
        -------
        MemberDefinition
            Retained immutable matching leaf.

        Raises
        ------
        ValueError
            Unknown namespace/member/version. No newest-version inference.

        Notes
        -----
        Performs bounded linear lookup without mutation or I/O.
        """
        checked = Reference.model_validate(member)
        for definition in self.definitions:
            if definition.kind == kind and definition.member == checked:
                return definition
        raise ValueError("Unknown catalogue member kind or exact version")

    def register(self, definition: MemberDefinition, *, limits: RegistryLimits | None = None) -> Catalogue:
        """Return a prospective catalogue or reconcile an exact local retry.

        Parameters
        ----------
        definition : MemberDefinition
            Exact new leaf; one key can never change implementation/config.
        limits : RegistryLimits or None, optional
            Selected counts/bytes; defaults to the finite registry profile.

        Returns
        -------
        Catalogue
            New immutable value, or the validated original for an exact retry.

        Raises
        ------
        ValueError
            Conflicting immutable key or coupled count/byte capacity exhaustion.

        Notes
        -----
        Pure local retry reconciliation gives no durable delivery guarantee.
        No factories, state inheritance, eviction or I/O occur.
        """
        selected = RegistryLimits() if limits is None else RegistryLimits.model_validate(limits)
        current = Catalogue.model_validate(self)
        leaf = MemberDefinition.model_validate(definition)
        if len(current.definitions) > selected.maximum_definitions:
            raise ValueError("Registry definition count exceeds selected capacity")
        admit_bytes(current, selected)
        for previous in current.definitions:
            if _definition_key(previous) == _definition_key(leaf):
                if previous.fingerprint != leaf.fingerprint:
                    raise ValueError("Conflicting immutable catalogue definition")
                return current
        if len(current.definitions) >= selected.maximum_definitions:
            raise ValueError("Registry definition capacity exhausted")
        result = Catalogue(schema_version=1, reference=current.reference, definitions=(*current.definitions, leaf))
        admit_bytes(result, selected)
        return result


def _definition_key(value: MemberDefinition) -> tuple[str, str, str]:
    """Order namespace and every retained semantic version deterministically."""
    return value.kind, value.member.id, value.member.version


class RecipeBinding(RegistryContract):
    """Declare one recipe's exact mathematical meaning and state ownership.

    Attributes
    ----------
    child, task : Reference
        Fixed collection child and direct task identities.
    expert : DefinitionBinding
        Exact ``MemberDefinition.binding``; display labels never transfer state.
    state_epoch : Reference
        Explicit retained/restarted lifecycle epoch.
    owners : tuple of StateBinding
        Complete named owners, unique by logical owner ID and canonicalized.
    objective : Reference
        Exact objective meaning already used by the core request.
    target, loss_transform, controls : DefinitionBinding or None
        Explicit selected definition bindings, including explicit nulls. A
        complete controls binding can cover target/bounds/cadence/movement;
        these fields do not infer or recreate its mathematical semantics.

    Notes
    -----
    At most128 owners and16MiB. This declaration does not inspect runtime roots
    or certify independence; the optional ownership checker does that for its
    declared host protocol. No state copying, migration, factories or I/O.
    """

    child: Reference
    task: Reference
    expert: DefinitionBinding
    state_epoch: Reference
    owners: Annotated[tuple[StateBinding, ...], Field(max_length=128)]
    objective: Reference
    target: DefinitionBinding | None
    loss_transform: DefinitionBinding | None
    controls: DefinitionBinding | None

    @model_validator(mode="after")
    def _owners(self) -> Self:
        """Preserve exact owner/implementation/config/codec bindings."""
        ordered_references(tuple(owner.owner for owner in self.owners))
        object.__setattr__(self, "owners", tuple(sorted(self.owners, key=lambda owner: owner.owner.id)))
        admit_bytes(self, RegistryLimits(maximum_bytes=16 * 1024 * 1024))
        return self
