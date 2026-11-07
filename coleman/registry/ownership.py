"""Opt-in ownership declarations and bounded checks of declared live roots.

The checks establish declared roots/bindings checked. They do not discover
omitted, native, external or hidden aliases, sandbox trusted callbacks, or
establish statistical independence. Ordinary direct consumers need no registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint, canonical_json, decode_json
from coleman.core.reading import bounded_json_bytes
from coleman.registry.definitions import RecipeBinding
from coleman.registry.limits import RegistryContract, RegistryLimits, admit_bytes
from coleman.state.contracts import StateBinding

# Exact built-in shapes exclude custom iteration/conversion hooks at inventories.
# pylint: disable=unidiomatic-typecheck

ResourceRole = Literal[
    "owner",
    "weights",
    "statistics",
    "pending",
    "rng",
    "predictor",
    "transform",
    "cache",
    "environment",
    "path",
    "counters",
    "cadence",
    "movement",
    "input",
    "session",
]
ResourceAccess = Literal["owned", "borrowed_predictor", "immutable_bytes"]


class ResourceClaim(CoreImmutable):
    """Name one declared root and its exact mutable state owner or byte profile.

    Attributes
    ----------
    resource : Reference
        Stable resource identity and version, distinct from an object address.
    owner : StateBinding or None
        Exact sole mutable state owner; null only for immutable input bytes.
    child : Reference or None
        Exact consuming child; null identifies a collection-owned root.
    role : str
        Actual declared mathematical or enclosing owner role.
    access : {'owned', 'borrowed_predictor', 'immutable_bytes'}
        Exclusive mutable ownership, explicitly shared predictor consumption, or
        verified canonical JSON bytes. Frozen containers imply no byte profile.
    immutable_fingerprint : str or None, optional
        SHA-256 of exact canonical UTF-8 JSON bytes, required only for that profile.
    """

    resource: Reference
    owner: StateBinding | None
    child: Reference | None
    role: ResourceRole
    access: ResourceAccess
    immutable_fingerprint: Fingerprint | None = None

    @model_validator(mode="after")
    def _profile(self) -> Self:
        """Keep immutable input sharing separate from mutable ownership."""
        if self.access == "immutable_bytes":
            if self.owner is not None or self.role != "input" or self.immutable_fingerprint is None:
                raise ValueError("Immutable bytes require input role, no state owner and an exact byte fingerprint")
        elif self.owner is None or self.immutable_fingerprint is not None:
            raise ValueError("Mutable resources require one exact state owner and no immutable byte claim")
        if self.access == "borrowed_predictor" and (
            self.child is None or self.role not in {"predictor", "cache", "transform"}
        ):
            raise ValueError("Borrowing requires a child and an explicit predictor, transform or cache role")
        return self


class SharedPredictor(CoreImmutable):
    """Declare one writer followed by a versioned snapshot and ordered consumers.

    Attributes
    ----------
    resource : Reference
        Exact predictor, transform or cache root with one owned claim.
    writer : StateBinding
        Exact sole updater, included in the complete checkpoint inventory.
    consumers : tuple of Reference
        Exact borrowed children in required consumption order.
    operation : Reference
        Versioned supported update operation.
    barrier : Reference
        Versioned update-then-snapshot-then-consume discipline.
    snapshot_version : Reference
        Exact coherent snapshot consumed by every listed child.

    Notes
    -----
    The host implements this trusted discipline. Declaration and supplied event
    checks do not provide Python access control or inspect predictor methods.
    """

    resource: Reference
    writer: StateBinding
    consumers: tuple[Reference, ...] = Field(min_length=1, max_length=32)
    operation: Reference
    barrier: Reference
    snapshot_version: Reference

    @model_validator(mode="after")
    def _consumers(self) -> Self:
        """Retain meaningful order while refusing duplicate child identities."""
        if len({item.id for item in self.consumers}) != len(self.consumers):
            raise ValueError("Shared predictor consumers must have unique child identities")
        return self


class PredictorStep(CoreImmutable):
    """Supply a trusted observed operation for one shared predictor barrier.

    Attributes
    ----------
    actor : Reference
        Writer owner for update/snapshot, or exact consuming child.
    kind : {'update', 'snapshot', 'consume'}
        Explicit operation role; no implicit update or snapshot is inserted.
    operation, barrier, snapshot_version : Reference
        Exact declaration versions observed by the trusted host.
    """

    actor: Reference
    kind: Literal["update", "snapshot", "consume"]
    operation: Reference
    barrier: Reference
    snapshot_version: Reference


def check_predictor_barrier(sharing: SharedPredictor, steps: tuple[PredictorStep, ...]) -> None:
    """Check one complete supplied update/snapshot/consumption event sequence.

    Parameters
    ----------
    sharing : SharedPredictor
        Exact selected single-writer and coherent-snapshot discipline.
    steps : tuple of PredictorStep
        One writer update, its snapshot, then all consumers in declared order.

    Raises
    ------
    ValueError, TypeError
        Mutable/unbounded evidence, wrong actor/order/version, or missing or
        duplicate update, barrier, snapshot or consumer.

    Notes
    -----
    Performs no update. The host supplies truthful bounded observations; this
    cannot discover additional undeclared calls or enforce access restrictions.
    """
    checked = SharedPredictor.model_validate(sharing)
    if type(steps) is not tuple or len(steps) != 2 + len(checked.consumers):
        raise ValueError("Predictor barrier requires exactly one update, snapshot and each declared consumer")
    expected = ((checked.writer.owner, "update"), (checked.writer.owner, "snapshot")) + tuple(
        (child, "consume") for child in checked.consumers
    )
    for actual, (actor, kind) in zip(steps, expected, strict=True):
        step = PredictorStep.model_validate(actual)
        if (step.actor, step.kind, step.operation, step.barrier, step.snapshot_version) != (
            actor,
            kind,
            checked.operation,
            checked.barrier,
            checked.snapshot_version,
        ):
            raise ValueError("Predictor barrier actor, order, operation or coherent snapshot version differs")


class OwnershipDeclaration(RegistryContract):
    """Declare a finite complete participating recipe and resource inventory.

    Attributes
    ----------
    recipes : tuple of RecipeBinding
        At most 32 exact child/task/expert/epoch/control and state-owner bindings.
    shared_owners : tuple of StateBinding
        Collection owners, including any registry, domain or shared predictor.
    resources : tuple of ResourceClaim
        Complete host-declared roots, with one enclosing owner root per mutable
        state owner. Nested mutable roles must also be declared when present.
    shared_predictors : tuple of SharedPredictor
        Exact intentional mutable sharing clauses, each with a single writer.

    Notes
    -----
    Declaration completeness is a trusted host obligation. A missing hidden
    field cannot be found through this inventory. No callbacks, storage, runtime
    capture or automatic registration occur on this pure declaration path.
    Each actual mutable root has one resource identity. A predictor that is also
    Stateful uses a distinct enclosing owner wrapper in this narrow profile;
    duplicated role claims cannot rename the same mutable root.
    """

    recipes: tuple[RecipeBinding, ...] = Field(min_length=1, max_length=32)
    shared_owners: tuple[StateBinding, ...] = Field(default=(), max_length=128)
    resources: tuple[ResourceClaim, ...] = Field(min_length=1, max_length=4096)
    shared_predictors: tuple[SharedPredictor, ...] = Field(default=(), max_length=4096)

    @model_validator(mode="after")
    def _inventory(self) -> Self:
        """Canonicalize sets and validate exact ownership/sharing joins."""
        children = {recipe.child.id: recipe.child for recipe in self.recipes}
        if len(children) != len(self.recipes):
            raise ValueError("Ownership recipes require unique child identities")
        owners = _state_owners(self)
        groups: dict[str, list[ResourceClaim]] = {}
        keys = set()
        for claim in self.resources:
            key = _claim_key(claim)
            if key in keys:
                raise ValueError("Duplicate resource claim for one child")
            keys.add(key)
            _claim_owner(claim, children, owners)
            groups.setdefault(claim.resource.id, []).append(claim)
        _owner_roots(owners, self.resources)
        _sharing_groups(groups, self.shared_predictors)
        object.__setattr__(self, "recipes", tuple(sorted(self.recipes, key=lambda item: item.child.id)))
        object.__setattr__(self, "shared_owners", tuple(sorted(self.shared_owners, key=lambda item: item.owner.id)))
        object.__setattr__(self, "resources", tuple(sorted(self.resources, key=_claim_key)))
        object.__setattr__(
            self, "shared_predictors", tuple(sorted(self.shared_predictors, key=lambda item: item.resource.id))
        )
        admit_bytes(self, RegistryLimits(maximum_bytes=16 * 1024 * 1024))
        return self


def _state_owners(declaration: OwnershipDeclaration) -> dict[str, tuple[StateBinding, Reference | None]]:
    """Collect exact owned bindings without collapsing duplicate owner IDs."""
    values: list[tuple[StateBinding, Reference | None]] = [
        (owner, recipe.child) for recipe in declaration.recipes for owner in recipe.owners
    ]
    values.extend((owner, None) for owner in declaration.shared_owners)
    if not 1 <= len(values) <= 128 or len({owner.owner.id for owner, _ in values}) != len(values):
        raise ValueError("Ownership requires one through128 unique state owners")
    return {owner.owner.id: (owner, child) for owner, child in values}


def _claim_key(claim: ResourceClaim) -> tuple[str, str]:
    """Select logical claim identity without hashing any runtime root."""
    return (claim.resource.id, "" if claim.child is None else claim.child.id)


def _claim_owner(
    claim: ResourceClaim,
    children: dict[str, Reference],
    owners: dict[str, tuple[StateBinding, Reference | None]],
) -> None:
    """Require exact child/version and complete state-owner bindings."""
    if claim.child is not None and children.get(claim.child.id) != claim.child:
        raise ValueError("Resource claim requires an exact declared child/version")
    if claim.owner is None:
        return
    expected = owners.get(claim.owner.owner.id)
    if expected is None or expected[0] != claim.owner:
        raise ValueError("Resource claim requires an exact declared state owner binding")
    if claim.access == "owned" and expected[1] != claim.child:
        raise ValueError("Owned resource has the wrong child binding")


def _owner_roots(owners: dict[str, tuple[StateBinding, Reference | None]], claims: tuple[ResourceClaim, ...]) -> None:
    """Require the enclosing real mutable owner root for every state owner."""
    roots = [claim.owner.owner.id for claim in claims if claim.role == "owner" and claim.owner is not None]
    if len(roots) != len(owners) or set(roots) != set(owners):
        raise ValueError("Every state owner requires exactly one owned enclosing owner root")


def _sharing_groups(groups: dict[str, list[ResourceClaim]], sharing: tuple[SharedPredictor, ...]) -> None:
    """Check resource identities, sole writer and explicit exact consumers."""
    clauses = {item.resource.id: item for item in sharing}
    if len(clauses) != len(sharing):
        raise ValueError("Duplicate shared predictor resource identity")
    used = set()
    for resource_id, claims in groups.items():
        first = claims[0]
        if any(item.resource != first.resource or item.role != first.role for item in claims):
            raise ValueError("One resource identity has conflicting versions or roles")
        if first.access == "immutable_bytes":
            _immutable_claims(claims)
            continue
        if _mutable_claims(claims, clauses.get(resource_id)):
            used.add(resource_id)
    if used != set(clauses):
        raise ValueError("Shared predictor clause requires its owned root and exact borrowed consumers")


def _immutable_claims(claims: list[ResourceClaim]) -> None:
    """Require coherent immutable profile declarations for one resource."""
    first = claims[0]
    if any(item.access != first.access or item.immutable_fingerprint != first.immutable_fingerprint for item in claims):
        raise ValueError("One immutable resource requires an exact coherent byte profile")


def _mutable_claims(claims: list[ResourceClaim], clause: SharedPredictor | None) -> bool:
    """Validate one sole writer and return whether explicit borrowing occurs."""
    owned = [item for item in claims if item.access == "owned"]
    borrowed = [item for item in claims if item.access == "borrowed_predictor"]
    if len(owned) != 1 or len(owned) + len(borrowed) != len(claims):
        raise ValueError("Every mutable resource requires exactly one named writer")
    if not borrowed:
        return False
    if clause is None or clause.resource != claims[0].resource or clause.writer != owned[0].owner:
        raise ValueError("Borrowed predictor requires its exact declared sharing owner")
    if any(item.owner != clause.writer for item in borrowed):
        raise ValueError("Borrowed predictor claims must name the exact sole writer owner")
    if {(child.id, child.version) for child in clause.consumers} != {
        (item.child.id, item.child.version) for item in borrowed if item.child is not None
    }:
        raise ValueError("Shared predictor consumers differ from the exact borrowed child inventory")
    return True


@dataclass(frozen=True, slots=True, eq=False)
class RootBinding:
    """Pair one exact claim with its actual live root, retaining a strong reference.

    Attributes
    ----------
    claim : ResourceClaim
        Exact entry from the complete selected ownership declaration.
    root : object
        Actual enclosing or nested backing root. Equality/hash methods are never
        called. This in-process object is never part of canonical serialization.
    """

    claim: ResourceClaim
    root: object


@dataclass(frozen=True, slots=True, eq=False)
class RootInventory:
    """Supply exact current recipe bindings and actual declared root objects.

    Attributes
    ----------
    recipes : tuple of RecipeBinding
        Actual current child/epoch/target/transform/control bindings.
    roots : tuple of RootBinding
        Complete trusted callback result. Refresh after restore replaces roots.
    """

    recipes: tuple[RecipeBinding, ...]
    roots: tuple[RootBinding, ...]


class DeclaredRootGuard:
    """Check bounded declared roots against each other and retained live roots.

    Notes
    -----
    Construct and invoke inside accepted owner staging. Check each fresh partial
    inventory before restore, then complete refreshed inventories before and
    after every restore, before adaptive callbacks. Factories, inventories and
    restore remain trusted bounded code free of live/external effects. Checks
    cannot undo a malicious callback's side effects or discover omitted aliases.
    """

    def __init__(
        self,
        declaration: OwnershipDeclaration,
        live: RootInventory | None = None,
        *,
        limits: RegistryLimits | None = None,
    ) -> None:
        """Retain a validated declaration and strong live-root references.

        Parameters
        ----------
        declaration : OwnershipDeclaration
            Complete participating inventory; all mathematical roles present in
            the host must be declared explicitly.
        live : RootInventory or None, optional
            Retained actual owners to protect from staged mutable-root reuse.
        limits : RegistryLimits or None, optional
            Explicit admission limits, using pure registry defaults when null.

        Raises
        ------
        ValueError, TypeError
            Invalid declarations, oversized bytes/resources, incomplete live
            inventory, aliases or wrong actual recipe bindings.
        """
        self.declaration = OwnershipDeclaration.model_validate(declaration)
        self.limits = RegistryLimits() if limits is None else RegistryLimits.model_validate(limits)
        if len(self.declaration.resources) > self.limits.maximum_resources:
            raise ValueError("Ownership resource inventory exceeds its declared capacity")
        if len(self.declaration.to_json().encode("utf-8")) > self.limits.maximum_bytes:
            raise ValueError("Ownership declaration exceeds its declared byte capacity")
        self._claims = {_claim_key(claim): claim for claim in self.declaration.resources}
        self._live: tuple[RootBinding, ...] = ()
        if live is not None:
            self.check(live)
            self._live = live.roots

    def check_partial(self, roots: tuple[RootBinding, ...]) -> None:
        """Check a bounded fresh subset against peers and retained live roots.

        Parameters
        ----------
        roots : tuple of RootBinding
            Currently constructed staged roots, refreshed through trusted host
            inventories before any owner restore begins.

        Raises
        ------
        ValueError, TypeError
            Wrong/duplicate claim, aliases, conflicting roots, invalid immutable
            bytes or any mutable alias to a retained live root.
        """
        checked = self._checked_roots(roots)
        for index, actual in enumerate(checked):
            for prior in checked[:index]:
                _check_pair(prior, actual, retained=False)
            for retained in self._live:
                _check_pair(retained, actual, retained=True)

    def check(self, inventory: RootInventory) -> None:
        """Check complete actual recipe bindings and refreshed root inventory.

        Parameters
        ----------
        inventory : RootInventory
            Complete live or staged callback result, including replacements
            introduced by restore. No fields are inferred from owner objects.

        Raises
        ------
        ValueError, TypeError
            Missing/extra/wrong resource or exact recipe binding, root aliases,
            conflicting identities or an invalid immutable sharing profile.

        Notes
        -----
        Successful validation means declared roots/bindings checked. It does
        not establish isolation of omitted Python or external state.
        """
        if type(inventory) is not RootInventory or type(inventory.recipes) is not tuple:
            raise TypeError("Root inventory requires concrete bounded in-process tuples")
        if len(inventory.recipes) != len(self.declaration.recipes):
            raise ValueError("Root inventory requires every exact recipe binding")
        recipes = tuple(
            sorted((RecipeBinding.model_validate(item) for item in inventory.recipes), key=lambda x: x.child.id)
        )
        if recipes != self.declaration.recipes:
            raise ValueError("Actual child, task, expert, epoch, target or control recipe binding differs")
        self.check_partial(inventory.roots)
        if len(inventory.roots) != len(self._claims):
            raise ValueError("Root inventory requires every declared resource exactly once")

    def _checked_roots(self, roots: tuple[RootBinding, ...]) -> tuple[RootBinding, ...]:
        """Reject mutable/unbounded callback results before inspecting entries."""
        if type(roots) is not tuple or len(roots) > len(self._claims):
            raise ValueError("Root inventory requires a bounded tuple within the complete declared resources")
        keys = set()
        immutable_bytes = 0
        for item in roots:
            if type(item) is not RootBinding:
                raise TypeError("Root inventory entries require concrete RootBinding values")
            claim = ResourceClaim.model_validate(item.claim)
            key = _claim_key(claim)
            if key in keys or self._claims.get(key) != claim:
                raise ValueError("Root inventory contains a duplicate or different declared resource binding")
            keys.add(key)
            if claim.access == "immutable_bytes":
                if type(item.root) is bytes:
                    immutable_bytes += len(item.root)
                if immutable_bytes > self.limits.maximum_bytes:
                    raise ValueError("Declared immutable resource bytes exceed the selected byte capacity")
                _check_bytes(item, self.limits.maximum_bytes)
        return roots


def _check_bytes(item: RootBinding, maximum_bytes: int) -> None:
    """Verify the narrow exact immutable canonical JSON byte profile."""
    root = item.root
    if type(root) is not bytes or len(root) > maximum_bytes:
        raise ValueError("Immutable sharing requires bounded concrete canonical JSON bytes")
    if "sha256:" + sha256(root).hexdigest() != item.claim.immutable_fingerprint:
        raise ValueError("Immutable input bytes differ from their declared fingerprint")
    if canonical_json(decode_json(bounded_json_bytes(root, maximum_bytes))).encode("utf-8") != root:
        raise ValueError("Immutable shared bytes require exact canonical UTF-8 JSON content")


def _check_pair(prior: RootBinding, actual: RootBinding, *, retained: bool) -> None:
    """Use only object identity for real roots, including staged versus live."""
    immutable = prior.claim.access == actual.claim.access == "immutable_bytes"
    if retained:
        if prior.root is actual.root and not immutable:
            raise ValueError("Staged mutable resource aliases a retained live root before adaptive work")
        return
    same_resource = prior.claim.resource.id == actual.claim.resource.id
    if same_resource and prior.root is not actual.root:
        raise ValueError("One declared resource identity resolves to conflicting live roots")
    if prior.root is actual.root and not immutable and not same_resource:
        raise ValueError("Different mutable resource identities alias the same declared root")
