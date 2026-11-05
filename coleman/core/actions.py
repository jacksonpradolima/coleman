"""Bounded, versioned action shapes independent of a runner or transport."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, Scalar, ScalarSpec, SimplexProfile, validate_simplex
from coleman.core.primitives import MAX_DIMENSIONS, MAX_MEMBERS, Identifier, canonical_json

ActionKind = Literal["ranking", "choice", "subset", "slate", "mixture", "vector", "acquisition"]


class _FiniteSpace(CoreImmutable):
    """Declare a versioned finite universe; empty universes permit abstention."""

    feasible_set: Reference
    members: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _members(self) -> Self:
        """Validate unique member versions before canonicalizing declaration order."""
        object.__setattr__(self, "members", ordered_references(self.members))
        return self


class RankingSpace(_FiniteSpace):
    """Declare members to rank once each; order is specified by the action.

    Attributes
    ----------
    kind : {'ranking'}
        Discriminator, never a probability or reward declaration.
    feasible_set : Reference
        Stable identity and version of the caller's feasible universe.
    members : tuple of Reference
        Unique versioned members; canonicalized by ID without discarding any.
    """

    kind: Literal["ranking"] = "ranking"


class ChoiceSpace(_FiniteSpace):
    """Declare a single-choice universe with explicit member and set versions.

    Attributes
    ----------
    kind : {'choice'}
        Single-member action discriminator.
    feasible_set : Reference
        Identity and semantic version of the feasible universe.
    members : tuple of Reference
        Unique bounded members, sorted by ID. Empty permits only abstention.
    """

    kind: Literal["choice"] = "choice"


class _SelectionSpace(_FiniteSpace):
    """Bound cardinality without silently selecting or repairing members."""

    minimum_selected: Annotated[int, Field(strict=True, ge=0)]
    maximum_selected: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def _cardinality(self) -> Self:
        """Reject impossible inclusive counts without repairing the universe."""
        if not self.minimum_selected <= self.maximum_selected <= len(self.members):
            raise ValueError("Selection bounds must satisfy 0 <= minimum <= maximum <= members")
        return self


class SubsetSpace(_SelectionSpace):
    """Declare unordered selection with inclusive explicit cardinality bounds.

    Attributes
    ----------
    kind : {'subset'}
        Unordered subset discriminator.
    feasible_set : Reference
        Exact versioned universe declaration.
    members : tuple of Reference
        Unique versioned members, canonically ordered by ID.
    minimum_selected, maximum_selected : int
        Required inclusive counts. Empty selection is valid only when minimum
        is zero; it is distinct from abstention.
    """

    kind: Literal["subset"] = "subset"


class SlateSpace(_SelectionSpace):
    """Declare ordered selection without repetition.

    Attributes
    ----------
    kind : {'slate'}
        Order-sensitive selection discriminator.
    feasible_set : Reference
        Exact versioned universe declaration.
    members : tuple of Reference
        Unique versioned feasible members, canonically ordered.
    minimum_selected, maximum_selected : int
        Required inclusive counts, bounded by the universe size.
    """

    kind: Literal["slate"] = "slate"


class MixtureSpace(_FiniteSpace):
    """Declare numerical simplex coordinates, without a sampling interpretation.

    Attributes
    ----------
    kind : {'mixture'}
        Allocation-weight discriminator.
    feasible_set : Reference
        Versioned allocation universe.
    members : tuple of Reference
        All coordinates, including those receiving zero weight.
    profile : {'binary64-simplex-v1'}
        Required numerical feasibility profile: exact stored-ratio residual at
        most 2**-49 from one. This is not an exact-arithmetic theorem.
    """

    kind: Literal["mixture"] = "mixture"
    profile: SimplexProfile


class AcquisitionSpace(_FiniteSpace):
    """Declare which member and fidelity may be acquired, without executing it.

    Attributes
    ----------
    kind : {'acquisition'}
        Acquisition proposal discriminator.
    feasible_set : Reference
        Versioned feasible acquisition universe.
    members : tuple of Reference
        Unique candidate identities.
    fidelities : tuple of Reference
        Explicit unique fidelity identities; no implicit cost or ordering.
    """

    kind: Literal["acquisition"] = "acquisition"
    fidelities: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _fidelities(self) -> Self:
        """Reject duplicate fidelity IDs and preserve their exact versions."""
        object.__setattr__(self, "fidelities", ordered_references(self.fidelities))
        return self


class VectorSpace(CoreImmutable):
    """Declare continuous action dimensions with explicit units and bounds.

    Attributes
    ----------
    kind : {'vector'}
        Continuous-vector discriminator.
    feasible_set : Reference
        Versioned coordinate/feasibility declaration.
    coordinates : tuple of ScalarSpec
        Unique action-role dimensions, sorted by ID; at most 64.
    signed : bool
        Required declaration allowing negative action values. Per-coordinate
        bounds still apply. False rejects negative values without clipping.

    Notes
    -----
    Describes coordinate bounds, not general joint feasibility or a density.
    An empty vector is explicit and distinct from abstention.
    """

    kind: Literal["vector"] = "vector"
    feasible_set: Reference
    coordinates: Annotated[tuple[ScalarSpec, ...], Field(max_length=MAX_DIMENSIONS)]
    signed: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def _coordinates(self) -> Self:
        """Require unique action dimensions and canonicalize coordinate order."""
        ordered_references(tuple(spec.dimension for spec in self.coordinates))
        if any(spec.role != "action" for spec in self.coordinates):
            raise ValueError("Vector coordinates require the action role")
        object.__setattr__(self, "coordinates", tuple(sorted(self.coordinates, key=lambda spec: spec.dimension.id)))
        return self


ActionSpace = Annotated[
    RankingSpace | ChoiceSpace | SubsetSpace | SlateSpace | MixtureSpace | VectorSpace | AcquisitionSpace,
    Field(discriminator="kind"),
]


def _check_members(values: tuple[Reference, ...], members: tuple[Reference, ...], *, complete: bool = False) -> None:
    """Reject repeated/unknown versions and optionally require complete coverage."""
    checked = ordered_references(values)
    if not set(checked).issubset(members) or (complete and checked != ordered_references(members)):
        raise ValueError("Action members do not match the versioned feasible universe")


class RankingAction(CoreContract):
    """Versioned exact permutation; scores, rewards and propensities are separate.

    Attributes
    ----------
    schema_version : int
        Required version 1; missing and unknown versions reject.
    kind : {'ranking'}
        Ranking discriminator.
    space : RankingSpace
        Versioned feasible universe.
    order : tuple of Reference
        Complete permutation with meaningful order, including the empty case.
    """

    kind: Literal["ranking"] = "ranking"
    space: RankingSpace
    order: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _permutation(self) -> Self:
        """Require each declared member exactly once without changing rank order."""
        _check_members(self.order, self.space.members, complete=True)
        return self


class ChoiceAction(CoreContract):
    """Versioned single choice with exact member-version membership.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    kind : {'choice'}
        Single-choice discriminator.
    space : ChoiceSpace
        Versioned feasible universe.
    chosen : Reference
        One member of the declared universe; no propensity is inferred.
    """

    kind: Literal["choice"] = "choice"
    space: ChoiceSpace
    chosen: Reference

    @model_validator(mode="after")
    def _chosen(self) -> Self:
        """Require exact versioned membership without sampling a replacement."""
        _check_members((self.chosen,), self.space.members)
        return self


def _check_selection(values: tuple[Reference, ...], space: SubsetSpace | SlateSpace) -> None:
    """Check versioned membership and explicit inclusive selection cardinality."""
    _check_members(values, space.members)
    if not space.minimum_selected <= len(values) <= space.maximum_selected:
        raise ValueError("Selection violates explicit cardinality bounds")


class SubsetAction(CoreContract):
    """Versioned unordered subset; canonical order does not encode a ranking.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    kind : {'subset'}
        Unordered-selection discriminator.
    space : SubsetSpace
        Versioned universe and inclusive cardinality bounds.
    selected : tuple of Reference
        Unique selected members, canonicalized by ID.
    """

    kind: Literal["subset"] = "subset"
    space: SubsetSpace
    selected: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _selected(self) -> Self:
        """Validate the subset and canonicalize its nonsemantic declaration order."""
        _check_selection(self.selected, self.space)
        object.__setattr__(self, "selected", ordered_references(self.selected))
        return self


class SlateAction(CoreContract):
    """Versioned ordered subset without repetition.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    kind : {'slate'}
        Ordered-selection discriminator.
    space : SlateSpace
        Versioned universe and cardinality bounds.
    selected : tuple of Reference
        Unique selected members in significant execution/display order.
    """

    kind: Literal["slate"] = "slate"
    space: SlateSpace
    selected: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _selected(self) -> Self:
        """Validate slate cardinality and uniqueness while retaining action order."""
        _check_selection(self.selected, self.space)
        return self


class Weight(CoreImmutable):
    """Attach one binary64 allocation weight to an explicit versioned member.

    Attributes
    ----------
    member : Reference
        Coordinate identity; not an observation or sampling mechanism.
    value : float
        Finite mathematical allocation. The containing mixture checks [0, 1].
    """

    member: Reference
    value: Binary64


class MixtureAction(CoreContract):
    """Versioned complete simplex allocation, without executing or sampling it.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    kind : {'mixture'}
        Allocation discriminator.
    space : MixtureSpace
        Explicit universe and numerical simplex profile.
    weights : tuple of Weight
        Every active member exactly once, including zeros, sorted by ID.

    Raises
    ------
    ValueError
        Missing/duplicate coordinates, nonfinite/out-of-range weights or
        residual exceeding the declared profile. No normalization occurs.
    """

    kind: Literal["mixture"] = "mixture"
    space: MixtureSpace
    weights: Annotated[tuple[Weight, ...], Field(max_length=MAX_MEMBERS)]

    @model_validator(mode="after")
    def _weights(self) -> Self:
        """Check complete coordinates and the named numerical simplex profile."""
        _check_members(tuple(weight.member for weight in self.weights), self.space.members, complete=True)
        validate_simplex(tuple(weight.value for weight in self.weights))
        object.__setattr__(self, "weights", tuple(sorted(self.weights, key=lambda weight: weight.member.id)))
        return self


class VectorAction(CoreContract):
    """Versioned complete observed vector with per-coordinate units and bounds.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    kind : {'vector'}
        Continuous-action discriminator.
    space : VectorSpace
        Versioned action dimensions, signedness and coordinate bounds.
    values : tuple of Scalar
        One observed value per coordinate with the exact declared spec.

    Notes
    -----
    Null is invalid for an executed/proposed numeric coordinate. No projection,
    joint-feasibility proof, unit conversion or external action occurs.
    """

    kind: Literal["vector"] = "vector"
    space: VectorSpace
    values: Annotated[tuple[Scalar, ...], Field(max_length=MAX_DIMENSIONS)]

    @model_validator(mode="after")
    def _values(self) -> Self:
        """Check exact observed dimensions and signedness without projection."""
        _check_members(tuple(value.spec.dimension for value in self.values), space_members(self.space), complete=True)
        expected = {spec.dimension: spec for spec in self.space.coordinates}
        for value in self.values:
            declared = canonical_json(expected[value.spec.dimension].model_dump(mode="json"))
            actual = canonical_json(value.spec.model_dump(mode="json"))
            if actual != declared or value.status != "observed":
                raise ValueError("Vector values require observed exact coordinate specifications")
            if not self.space.signed and value.value is not None and value.value < 0:
                raise ValueError("Unsigned vector rejects negative values")
        object.__setattr__(self, "values", tuple(sorted(self.values, key=lambda value: value.spec.dimension.id)))
        return self


class AcquisitionAction(CoreContract):
    """Versioned acquisition proposal with explicit member and fidelity.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    kind : {'acquisition'}
        Acquisition discriminator.
    space : AcquisitionSpace
        Exact candidate and fidelity declarations.
    chosen, fidelity : Reference
        Admitted versioned candidate and fidelity; no acquisition is performed.
    """

    kind: Literal["acquisition"] = "acquisition"
    space: AcquisitionSpace
    chosen: Reference
    fidelity: Reference

    @model_validator(mode="after")
    def _chosen(self) -> Self:
        """Require both chosen member and fidelity in their versioned catalogues."""
        _check_members((self.chosen,), self.space.members)
        _check_members((self.fidelity,), self.space.fidelities)
        return self


class AbstainAction(CoreContract):
    """Explicitly decline a decision without encoding it as zero or an empty set.

    Attributes
    ----------
    schema_version : int
        Required version 1.
    kind : {'abstain'}
        Explicit abstention discriminator.
    space : ActionSpace
        Original feasible-space declaration, including empty universes.
    reason : str
        Stable caller-declared reason code; no fallback is executed.
    """

    kind: Literal["abstain"] = "abstain"
    space: ActionSpace
    reason: Identifier


Action = Annotated[
    RankingAction
    | ChoiceAction
    | SubsetAction
    | SlateAction
    | MixtureAction
    | VectorAction
    | AcquisitionAction
    | AbstainAction,
    Field(discriminator="kind"),
]


def space_members(space: ActionSpace) -> tuple[Reference, ...]:
    """Return the declared universe without inferring observations or feasibility.

    Parameters
    ----------
    space : ActionSpace
        Validated finite universe or continuous coordinate declaration.

    Returns
    -------
    tuple of Reference
        Canonical members, or dimension identities for a vector. No I/O occurs.
    """
    if isinstance(space, VectorSpace):
        return tuple(spec.dimension for spec in space.coordinates)
    return space.members


def selected_members(action: Action) -> tuple[Reference, ...]:
    """Return explicitly selected members for action-wise feedback validation.

    Parameters
    ----------
    action : Action
        Validated actual action, not necessarily the original proposal.

    Returns
    -------
    tuple of Reference
        Chosen members, retaining slate order.

    Raises
    ------
    ValueError
        Ranking, mixture, vector or abstention has no selected-member semantics.
    """
    if isinstance(action, (ChoiceAction, AcquisitionAction)):
        return (action.chosen,)
    if isinstance(action, (SubsetAction, SlateAction)):
        return action.selected
    raise ValueError("Action has no selected-member feedback interpretation")
