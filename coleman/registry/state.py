"""Opt-in declared-root checks inside the existing accepted owner staging.

This adapter wraps explicit component factories; it does not open a store or
replace its acceptance, publication, restore or concurrency protocol. Inventory
callbacks and custom state owners remain trusted bounded host code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from coleman.core.base import CoreContract
from coleman.core.protocols import Stateful
from coleman.registry.definitions import RecipeBinding
from coleman.registry.limits import RegistryLimits, admit_bytes
from coleman.registry.ownership import DeclaredRootGuard, OwnershipDeclaration, RootInventory
from coleman.state.components import ComponentBinding, ComponentCodec, StagedCollection


@dataclass(frozen=True, eq=False)
class OwnerInspection[S: CoreContract, H]:
    """Pair one exact component factory with its explicit root inspector.

    Parameters
    ----------
    binding : ComponentBinding
        Existing concrete codec, fresh factory and frontier inspector.
    inspect : callable
        Pure bounded callback receiving the actual owner and returning its
        recipe bindings and declared roots. Reports may be component subsets;
        their complete union must satisfy the selected declaration.

    Notes
    -----
    Construction invokes no callback. Inspect actual mutable backing roots,
    including owned host/session objects; omitted or hidden state is not found.
    """

    binding: ComponentBinding[S, H]
    inspect: Callable[[H], RootInventory]


@dataclass(frozen=True, eq=False)
class LiveOwner[S: CoreContract, H]:
    """Retain a live owner for comparison before accepted fresh restoration.

    Parameters
    ----------
    inspection : OwnerInspection
        Exact component and trusted inspector used for this live owner.
    owner : object
        Retained live object. Strong references survive the staging checks.

    Notes
    -----
    Neither the inspector nor the owner is called during construction. The
    application must hold its own mutation barrier while roots are inspected.
    """

    inspection: OwnerInspection[S, H]
    owner: H


def _inventory(reports: tuple[RootInventory, ...], maximum_resources: int) -> RootInventory:
    """Join bounded reports without hiding conflicting recipe declarations."""
    recipes: dict[str, RecipeBinding] = {}
    roots = []
    for report in reports:
        if type(report) is not RootInventory:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Owner inspection requires an exact RootInventory")
        if type(report.recipes) is not tuple or type(report.roots) is not tuple:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Owner inspection requires concrete tuple fields")
        if len(report.recipes) > 32 or len(roots) + len(report.roots) > maximum_resources:
            raise ValueError("Owner inventory exceeds its selected finite capacity")
        _merge_recipes(recipes, report.recipes)
        roots.extend(report.roots)
    return RootInventory(tuple(sorted(recipes.values(), key=lambda item: item.child.id)), tuple(roots))


def _merge_recipes(recipes: dict[str, RecipeBinding], offered: tuple[RecipeBinding, ...]) -> None:
    """Merge one admitted report's recipes while preserving exact join checks."""
    for recipe in offered:
        checked = RecipeBinding.model_validate(recipe)
        prior = recipes.get(checked.child.id)
        if prior is not None and prior.to_json() != checked.to_json():
            raise ValueError("Owner inventories disagree on the exact recipe binding")
        recipes[checked.child.id] = checked
        if len(recipes) > 32:
            raise ValueError("Owner inventory exceeds32 recipe bindings")


def _checked_inspection_binding(item: OwnerInspection[Any, Any]) -> ComponentBinding[Any, Any]:
    """Validate one static inspection descriptor without calling its callbacks."""
    if type(item) is not OwnerInspection:  # pylint: disable=unidiomatic-typecheck
        raise TypeError("Owned inspection requires exact OwnerInspection values")
    if type(item.binding) is not ComponentBinding or not callable(item.inspect):  # pylint: disable=unidiomatic-typecheck
        raise TypeError("Owned inspection requires an exact component and callable inspector")
    if type(item.binding.codec) is not ComponentCodec or not all(  # pylint: disable=unidiomatic-typecheck
        callable(callback) for callback in (item.binding.new_owner, item.binding.frontier)
    ):
        raise TypeError("Owned component requires an exact concrete codec and callable factory/frontier")
    return item.binding


class _Fence:
    """Hold one accepted staging's strong roots and lazy trusted callbacks."""

    def __init__(
        self,
        declaration: OwnershipDeclaration,
        inspections: tuple[OwnerInspection[Any, Any], ...],
        live: tuple[LiveOwner[Any, Any], ...],
        limits: RegistryLimits,
    ) -> None:
        """Retain static declarations and callbacks without invoking any of them."""
        self.declaration = declaration
        self.inspections = inspections
        self.live = live
        self.limits = limits
        self.guard: DeclaredRootGuard | None = None
        self.entries: list[tuple[OwnerInspection[Any, Any], Any]] = []

    def new_owner(self, inspection: OwnerInspection[Any, Any]) -> _CheckedOwner[Any, Any]:
        """Inspect live roots only after the existing store accepts staging."""
        if self.guard is None:
            reports = tuple(item.inspection.inspect(item.owner) for item in self.live)
            original = _inventory(reports, self.limits.maximum_resources) if self.live else None
            self.guard = DeclaredRootGuard(self.declaration, original, limits=self.limits)
        if any(item is inspection for item, _owner in self.entries):
            raise ValueError("OwnedBindings supports only one accepted staging operation")
        owner = inspection.binding.new_owner()
        self.entries.append((inspection, owner))
        reports = tuple(item.inspect(value) for item, value in self.entries)
        self.guard.check_partial(_inventory(reports, self.limits.maximum_resources).roots)
        return _CheckedOwner(inspection, owner, self)

    def check(self) -> None:
        """Refresh complete roots before restore, transition access and export."""
        if self.guard is None or len(self.entries) != len(self.inspections):
            raise ValueError("Complete owned factory inventory must precede restoration")
        reports = tuple(item.inspect(owner) for item, owner in self.entries)
        self.guard.check(_inventory(reports, self.limits.maximum_resources))


class _CheckedOwner[S: CoreContract, H]:
    """Expose the existing Stateful boundary with pre/post inventory checks."""

    def __init__(self, inspection: OwnerInspection[S, H], owner: H, fence: _Fence) -> None:
        """Retain a real fresh owner behind the shared accepted-staging fence."""
        self.inspection = inspection
        self.owner = owner
        self.fence = fence

    def restore_state(self, state: S, /) -> None:
        """Check every fresh/live root before trusted restore and replacements after."""
        self.fence.check()
        cast(Stateful[S], self.owner).restore_state(state)
        self.fence.check()

    def export_state(self) -> S:
        """Check refreshed roots around the trusted complete snapshot export."""
        self.fence.check()
        state = cast(Stateful[S], self.owner).export_state()
        self.fence.check()
        return state


class OwnedBindings:
    """Qualify declared roots during one accepted composed-state staging.

    Parameters
    ----------
    declaration : OwnershipDeclaration
        Exact recipe, owner, sharing and resource inventory.
    inspections : tuple of OwnerInspection
        Complete existing component inventory, at most128 owners. Every mutable
        participating component has one inspector; no factory runs here.
    live : tuple of LiveOwner, optional
        Complete live inventory, or empty when no live collection is retained.
        Inspectors run lazily inside the first accepted fresh factory.
    limits : RegistryLimits or None, optional
        Selected finite resource/count/byte limits; defaults apply when absent.

    Raises
    ------
    ValueError, TypeError
        Invalid declaration, incomplete/conflicting component bindings or
        malformed live inventory. Callback failures propagate only on staging.

    Notes
    -----
    Pass ``bindings`` to the existing store, then use ``owner`` in its accepted
    transition. Retained/stale requests retain the store's zero-factory and
    zero-inventory-callback behavior. Create a new wrapper for each factory
    staging attempt, including an attempt after a failed restore/transition.
    Exact acknowledged retries which run no factories may reuse it. No store is opened.

    Factories, inspectors, restore and export must be bounded and free of
    external or live-state side effects. These checks compare declared roots;
    they do not sandbox callbacks or roll back arbitrary Python effects.
    """

    def __init__(
        self,
        declaration: OwnershipDeclaration,
        inspections: tuple[OwnerInspection[Any, Any], ...],
        *,
        live: tuple[LiveOwner[Any, Any], ...] = (),
        limits: RegistryLimits | None = None,
    ) -> None:
        """Validate the class-documented inputs without factories or inspectors."""
        checked = OwnershipDeclaration.model_validate(declaration)
        selected = RegistryLimits() if limits is None else RegistryLimits.model_validate(limits)
        if len(checked.resources) > selected.maximum_resources:
            raise ValueError("Owned resource declaration exceeds its selected finite capacity")
        admit_bytes(checked, selected)
        self._validate(checked, inspections, live)
        self._fence = _Fence(checked, inspections, live, selected)
        self._pairs = tuple(
            (item.binding, ComponentBinding(item.binding.codec, self._factory(item), item.binding.frontier))
            for item in inspections
        )

    @staticmethod
    def _validate(
        declaration: OwnershipDeclaration,
        inspections: tuple[OwnerInspection[Any, Any], ...],
        live: tuple[LiveOwner[Any, Any], ...],
    ) -> None:
        """Check static inventory without calling any host-supplied function."""
        if type(inspections) is not tuple or not 1 <= len(inspections) <= 128:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Owned component inspection requires a tuple of one through128 owners")
        if type(live) is not tuple or len(live) > 128:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Live owned inventory requires a tuple of at most128 owners")
        expected = {owner.owner.id: owner for recipe in declaration.recipes for owner in recipe.owners}
        expected.update((owner.owner.id, owner) for owner in declaration.shared_owners)
        actual = {}
        for item in inspections:
            owner = _checked_inspection_binding(item).codec.binding
            if owner.owner.id in actual:
                raise ValueError("Duplicate owned component inspection")
            actual[owner.owner.id] = owner
        if actual != expected:
            raise ValueError("Owned inspections must match the complete exact state owner inventory")
        OwnedBindings._validate_live(inspections, live, set(expected))

    @staticmethod
    def _validate_live(
        inspections: tuple[OwnerInspection[Any, Any], ...],
        live: tuple[LiveOwner[Any, Any], ...],
        expected: set[str],
    ) -> None:
        """Require a complete live inventory from the selected exact inspectors."""
        if live:
            observed = []
            for item in live:
                if type(item) is not LiveOwner:  # pylint: disable=unidiomatic-typecheck
                    raise TypeError("Live inventory requires exact LiveOwner values")
                if not any(item.inspection is inspection for inspection in inspections):
                    raise ValueError("Live inventory must use the selected exact inspection")
                observed.append(item.inspection.binding.codec.binding.owner.id)
            if len(observed) != len(set(observed)) or set(observed) != expected:
                raise ValueError("Live inventory must contain every selected state owner exactly once")

    def _factory(self, item: OwnerInspection[Any, Any]) -> Callable[[], _CheckedOwner[Any, Any]]:
        """Capture one selected inspector without invoking its callback."""
        return lambda: self._fence.new_owner(item)

    @property
    def bindings(self) -> tuple[ComponentBinding[Any, Any], ...]:
        """Return existing-compatible wrappers, without factories or I/O.

        Returns
        -------
        tuple of ComponentBinding
            Exact original codecs/frontier inspectors with lazy fresh factories.
        """
        return tuple(wrapped for _original, wrapped in self._pairs)

    def owner[S: CoreContract, H](self, stage: StagedCollection, binding: ComponentBinding[S, H]) -> H:
        """Get a checked actual owner inside the existing accepted transition.

        Parameters
        ----------
        stage : StagedCollection
            Existing active stage supplied by the store to its callback.
        binding : ComponentBinding
            Exact original in-process binding, never a serialized selector.

        Returns
        -------
        H
            Actual fresh restored owner after complete declared-root checks.

        Raises
        ------
        ValueError
            Wrong/closed stage, unknown binding or invalid refreshed inventory.

        Notes
        -----
        The application must not retain owners past the stage or mutate roots
        concurrently. Export checks still run after the transition returns.
        """
        for original, wrapped in self._pairs:
            if original is binding:
                checked = stage.owner(wrapped)
                self._fence.check()
                return cast(H, checked.owner)
        raise ValueError("Requested owner is not in the selected owned bindings")
