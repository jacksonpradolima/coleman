"""Explicit subjective declarations and bounded pure preference arithmetic."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.actions import Weight
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, Scalar, ScalarSpec, SimplexProfile, validate_simplex
from coleman.core.primitives import MAX_DIMENSIONS, MAX_MEMBERS, canonical_json
from coleman.core.units import UnitSemantics
from coleman.objectives.arithmetic import PROFILE, checked_fraction
from coleman.objectives.bindings import DefinitionBinding, QuantityMeaning, SemanticDefinition, require_quantity

COEFFICIENT_SEMANTICS = UnitSemantics(unit="1", meaning="mixture-coefficient-fraction-v1")
CoefficientWeights = Annotated[tuple[Weight, ...], Field(min_length=1, max_length=MAX_MEMBERS)]
Nonnegative = Annotated[Binary64, Field(ge=0)]


def coefficient_weights(weights: tuple[Weight, ...], profile: SimplexProfile) -> tuple[Weight, ...]:
    """Validate complete coefficient fractions without normalizing or sampling.

    Parameters
    ----------
    weights : tuple of Weight
        One through 4096 unique versioned coordinates, including exact zeros.
    profile : {'binary64-simplex-v1'}
        Existing bounded stored-binary64 simplex convention.

    Returns
    -------
    tuple of Weight
        Revalidated coordinates in ascending logical member-ID order.

    Raises
    ------
    ValueError
        Unknown profile, duplicate IDs/versions or invalid simplex values.

    Notes
    -----
    Fractions describe coefficients, never sampling probabilities or evidence.
    No repair, empirical observation, state change or I/O occurs.
    """
    if not isinstance(weights, tuple) or profile != "binary64-simplex-v1":
        raise ValueError("Coefficient weights require an immutable tuple and a known simplex profile")
    checked = tuple(Weight.model_validate(weight) for weight in weights)
    ordered_references(tuple(weight.member for weight in checked))
    validate_simplex(tuple(weight.value for weight in checked))
    return tuple(sorted(checked, key=lambda weight: weight.member.id))


class _CoefficientDefinition(SemanticDefinition):
    """Share explicit fraction semantics without conflating declaration roles."""

    weights: CoefficientWeights
    profile: SimplexProfile
    semantics: UnitSemantics

    @model_validator(mode="after")
    def _coefficients(self) -> Self:
        """Validate canonical fractions and retain structural zero support."""
        if self.semantics != COEFFICIENT_SEMANTICS:
            raise ValueError("Coefficient declarations require canonical coefficient-fraction semantics")
        object.__setattr__(self, "weights", coefficient_weights(self.weights, self.profile))
        return self


class InitialPrior(_CoefficientDefinition):
    """Declare initial coefficients, independently from repeated preferences.

    Attributes
    ----------
    weights : tuple of Weight
        Complete initial simplex, including permitted structural zeros.
    profile : {'binary64-simplex-v1'}
        Numerical simplex contract, without implicit normalization.
    semantics : UnitSemantics
        Canonical coefficient fractions, distinct from probabilities.
    kind : {'initial_prior'}
        Initialization-only discriminator.

    Notes
    -----
    Inherited reference, lineage and required schema version bind all content.
    This declaration initializes nothing by itself. Methods needing positive
    support must separately call ``has_full_support`` or reject zero support.
    """

    kind: Literal["initial_prior"] = "initial_prior"


class DeclaredTargetShare(_CoefficientDefinition):
    """Declare a persistent share target, without executing a learner update.

    Attributes
    ----------
    weights : tuple of Weight
        Explicit target simplex with exact expert identities and versions.
    profile : {'binary64-simplex-v1'}
        Numerical coefficient convention.
    semantics : UnitSemantics
        Canonical coefficient fractions.
    alpha : float
        Finite share fraction in [0, 1], distinct from penalty strength.
    kind : {'declared_target_share'}
        Share-target discriminator, not a static penalty anchor.

    Notes
    -----
    Versioned lineage is subjective input, never predictive evidence. Applying
    the share recurrence and its cadence belongs to the consuming learner.
    Construction has no side effects and creates no feedback observations.
    """

    kind: Literal["declared_target_share"] = "declared_target_share"
    alpha: Annotated[Binary64, Field(ge=0, le=1)]


class StaticQuadraticAnchor(_CoefficientDefinition):
    """Specify a static subjective penalty ``strength * sum((p-a)**2)``.

    Attributes
    ----------
    weights : tuple of Weight
        Fixed anchor a, independent from the previous emitted vector.
    profile : {'binary64-simplex-v1'}
        Coefficient simplex convention.
    semantics : UnitSemantics
        Canonical coefficient fractions.
    strength : float
        Finite nonnegative objective units per squared coefficient fraction.
        This coefficient is not restricted to [0, 1] and is not a paid fee.
    output : ScalarSpec
        Explicit objective-role schema for the penalty quantity.
    kind : {'static_quadratic_anchor'}
        Static squared-distance penalty discriminator.

    Raises
    ------
    ValueError
        Invalid coefficients/strength or a non-objective output schema.

    Notes
    -----
    Each explicit evaluation adds one penalty quantity; initialization adds
    none. No repeated application, optimization, mutation or I/O is implicit.
    """

    kind: Literal["static_quadratic_anchor"] = "static_quadratic_anchor"
    strength: Nonnegative
    output: ScalarSpec

    @model_validator(mode="after")
    def _output(self) -> Self:
        """Require an objective quantity admitting a zero penalty."""
        if self.output.role != "objective":
            raise ValueError("Anchor output requires an objective schema admitting zero")
        Scalar(spec=self.output, value=0.0, status="observed")
        return self


class ObjectiveWeight(CoreImmutable):
    """Attach one subjective coefficient to a declared normalized loss.

    Attributes
    ----------
    quantity : QuantityMeaning
        Exact normalized-loss schema and horizon; probability/rank rejects.
    weight : float
        Finite nonnegative fraction. The outer definition checks the simplex.
    """

    quantity: QuantityMeaning
    weight: Annotated[Binary64, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _loss(self) -> Self:
        """Prevent scalarization of probability/ranking by relabeling."""
        if self.quantity.interpretation != "normalized_loss":
            raise ValueError("Objective weighting requires declared normalized losses")
        return self


class ObjectiveWeights(SemanticDefinition):
    """Declare explicit subjective weights, without inferring a tradeoff.

    Attributes
    ----------
    terms : tuple of ObjectiveWeight
        One through 64 unique loss dimensions, sharing the exact horizon.
    output : ScalarSpec
        Objective-role dimensionless subjective value bounded by [0, 1].
    kind : {'objective_weights'}
        Explicit scalarization discriminator.

    Raises
    ------
    ValueError
        Duplicate dimensions, mismatched horizons, invalid weight simplex or
        an output that could silently present subjective utility as feedback.

    Notes
    -----
    Missing preference is represented by absence of this definition: preserve
    the original baseline and do not call scalarization. No equal weighting,
    preference event, outcome or empirical claim is created automatically.
    """

    kind: Literal["objective_weights"] = "objective_weights"
    terms: Annotated[tuple[ObjectiveWeight, ...], Field(min_length=1, max_length=MAX_DIMENSIONS)]
    output: ScalarSpec

    @model_validator(mode="after")
    def _terms(self) -> Self:
        """Check bounded terms, exact horizons and explicitly subjective output."""
        ordered_references(tuple(term.quantity.spec.dimension for term in self.terms))
        validate_simplex(tuple(term.weight for term in self.terms))
        if any(term.quantity.horizon != self.terms[0].quantity.horizon for term in self.terms):
            raise ValueError("Objective weighting requires identical declared horizons")
        if (
            self.output.role != "objective"
            or self.output.semantics.unit != "1"
            or self.output.minimum != 0
            or self.output.maximum != 1
        ):
            raise ValueError("Subjective output requires objective role and explicit dimensionless [0,1] bounds")
        object.__setattr__(self, "terms", tuple(sorted(self.terms, key=lambda term: term.quantity.spec.dimension.id)))
        return self


class ReferencePoint(CoreImmutable):
    """Declare an observed target in its exact raw units and horizon.

    Attributes
    ----------
    quantity : QuantityMeaning
        Explicit target quantity interpretation, schema and horizon.
    value : Scalar
        Observed target value; missing targets cannot become fabricated zeros.

    Notes
    -----
    This target executes no utility, risk or optimization operation.
    """

    quantity: QuantityMeaning
    value: Scalar

    @model_validator(mode="after")
    def _target(self) -> Self:
        """Require a known value with exact expected quantity schema."""
        require_quantity(self.value, self.quantity)
        if self.value.status != "observed":
            raise ValueError("A reference point requires an observed declared target")
        return self


class ReferencePoints(SemanticDefinition):
    """Version separately named target quantities without inferring utility.

    Attributes
    ----------
    points : tuple of ReferencePoint
        One through 64 unique dimensions, canonicalized by ID.
    kind : {'reference_points'}
        Target-only discriminator, distinct from coefficients or evidence.

    Notes
    -----
    Construction retains units and lineage and performs no evaluation or I/O.
    """

    kind: Literal["reference_points"] = "reference_points"
    points: Annotated[tuple[ReferencePoint, ...], Field(min_length=1, max_length=MAX_DIMENSIONS)]

    @model_validator(mode="after")
    def _points(self) -> Self:
        """Check target identities before canonicalizing declaration order."""
        ordered_references(tuple(point.quantity.spec.dimension for point in self.points))
        object.__setattr__(
            self, "points", tuple(sorted(self.points, key=lambda point: point.quantity.spec.dimension.id))
        )
        return self


class HumanJudgment(SemanticDefinition):
    """Store one explicit subjective relation without cardinal inference.

    Attributes
    ----------
    left, right : Reference
        Distinct logical alternatives with exact semantic versions.
    relation : {'left', 'right', 'tie', 'incomparable'}
        Supplied preference, never a realized reward or probability.
    kind : {'human_judgment'}
        Subjective-event discriminator.

    Raises
    ------
    ValueError
        Same logical alternative appears on both sides or relation is unknown.

    Notes
    -----
    Source/time/version are inherited lineage. A missing judgment is no event;
    none is synthesized. No feedback history, learner or I/O is touched.
    """

    kind: Literal["human_judgment"] = "human_judgment"
    left: Reference
    right: Reference
    relation: Literal["left", "right", "tie", "incomparable"]

    @model_validator(mode="after")
    def _alternatives(self) -> Self:
        """Reject self-comparisons even when versions differ."""
        if self.left.id == self.right.id:
            raise ValueError("A judgment requires distinct logical alternatives")
        return self


class PreferenceEvaluation(CoreContract):
    """Retain subjective scalarization inputs and explicit noncomputability.

    Attributes
    ----------
    definition : DefinitionBinding
        Exact subjective definition content, not only its human reference.
    specification : ObjectiveWeights
        Complete typed definition, retained to recompute arithmetic on reading.
    inputs : tuple of Scalar
        Complete raw normalized-loss inputs, retaining each missingness state.
    output : Scalar or None
        Observed subjective objective, or null when any input is missing.
    status : {'evaluated', 'insufficient_evidence'}
        Operation result, independent from outcome evidence or learner updates.
    arithmetic_profile : {'binary64-rational-v1'}
        Rational intermediates with one guarded output projection.

    Raises
    ------
    ValueError
        Derived values, status or binding disagree with the complete definition
        and input quantities. Readers recompute rather than trusting a digest.

    Notes
    -----
    Reproducible arithmetic does not authenticate subjective source judgments,
    qualify empirical feedback or apply an optimizer/learner update. No I/O occurs.
    """

    definition: DefinitionBinding
    specification: ObjectiveWeights
    inputs: Annotated[tuple[Scalar, ...], Field(min_length=1, max_length=MAX_DIMENSIONS)]
    output: Scalar | None
    status: Literal["evaluated", "insufficient_evidence"]
    arithmetic_profile: Literal["binary64-rational-v1"] = PROFILE

    @model_validator(mode="after")
    def _status(self) -> Self:
        """Recompute subjective arithmetic instead of trusting a digest or label."""
        inputs, output = _objective_facts(self.specification, self.inputs)
        object.__setattr__(self, "inputs", inputs)
        expected_status = "evaluated" if output is not None else "insufficient_evidence"
        actual_output = None if self.output is None else self.output.model_dump(mode="json")
        expected_output = None if output is None else output.model_dump(mode="json")
        if (
            self.definition != self.specification.binding
            or self.status != expected_status
            or canonical_json(actual_output) != canonical_json(expected_output)
        ):
            raise ValueError("Preference receipt does not reproduce its declared specification and observed support")
        return self


class AnchorEvaluation(CoreContract):
    """Record one static penalty evaluation without altering the base objective.

    Attributes
    ----------
    definition : DefinitionBinding
        Exact anchor and strength definition.
    specification : StaticQuadraticAnchor
        Complete typed definition, allowing arithmetic to be checked on reading.
    weights : tuple of Weight
        Complete evaluated coefficient vector, canonicalized by ID.
    output : Scalar
        Observed penalty in the definition's objective units.
    arithmetic_profile : {'binary64-rational-v1'}
        Guarded rational arithmetic convention.

    Notes
    -----
    This receipt does not claim the penalty was added to an optimizer objective.
    Readers recompute its value from the complete supplied specification and
    vector; the supplied anchor's truth or desirability is not authenticated.
    """

    definition: DefinitionBinding
    specification: StaticQuadraticAnchor
    weights: CoefficientWeights
    output: Scalar
    arithmetic_profile: Literal["binary64-rational-v1"] = PROFILE

    @model_validator(mode="after")
    def _evaluation(self) -> Self:
        """Recompute the penalty and bind its complete versioned definition."""
        weights, output = _anchor_facts(self.specification, self.weights)
        object.__setattr__(self, "weights", weights)
        if self.definition != self.specification.binding or self.output.model_dump_json() != output.model_dump_json():
            raise ValueError("Anchor receipt does not reproduce its declared specification and observed objective")
        return self


def has_full_support(prior: InitialPrior) -> bool:
    """Inspect declared positive support without initializing a method.

    Parameters
    ----------
    prior : InitialPrior
        Revalidated initial coefficients; structural zeros remain valid data.

    Returns
    -------
    bool
        True exactly when every coordinate is strictly positive.

    Raises
    ------
    ValueError
        The declaration is invalid or is not an initial prior.

    Notes
    -----
    No normalization, epsilon, learning, state change or I/O occurs.
    """
    checked = InitialPrior.model_validate(prior)
    return all(weight.value > 0 for weight in checked.weights)


def weight_objectives(definition: ObjectiveWeights, values: tuple[Scalar, ...]) -> PreferenceEvaluation:
    """Evaluate one explicit subjective sum of normalized losses.

    Parameters
    ----------
    definition : ObjectiveWeights
        Complete versioned coefficient/schema declaration.
    values : tuple of Scalar
        Each declared loss exactly once. Missing values remain noncomputable,
        including a missing dimension whose subjective weight is zero.

    Returns
    -------
    PreferenceEvaluation
        Content-bound objective receipt or explicit insufficient support.

    Raises
    ------
    ValueError
        Missing/extra/conflicting dimensions, incompatible schema or output
        arithmetic overflow, nonzero underflow or violated output bounds.

    Notes
    -----
    No definition means no operation: retain the original baseline instead of
    calling this function or inventing equal weights. No empirical feedback,
    mutation, fitting, event or I/O is produced.
    """
    declared = ObjectiveWeights.model_validate(definition)
    inputs, output = _objective_facts(declared, values)
    return PreferenceEvaluation(
        schema_version=1,
        definition=declared.binding,
        specification=declared,
        inputs=inputs,
        output=output,
        status="evaluated" if output is not None else "insufficient_evidence",
    )


def _objective_facts(
    declared: ObjectiveWeights, values: tuple[Scalar, ...]
) -> tuple[tuple[Scalar, ...], Scalar | None]:
    """Derive typed inputs and subjective output without constructing a receipt."""
    if not isinstance(values, tuple) or len(values) != len(declared.terms):
        raise ValueError("Objective weighting requires each declared dimension exactly once")
    checked = tuple(Scalar.model_validate(value) for value in values)
    ordered_references(tuple(value.spec.dimension for value in checked))
    by_dimension = {(value.spec.dimension.id, value.spec.dimension.version): value for value in checked}
    ordered = []
    for term in declared.terms:
        key = (term.quantity.spec.dimension.id, term.quantity.spec.dimension.version)
        if key not in by_dimension:
            raise ValueError("Objective input dimension/version does not match its definition")
        ordered.append(require_quantity(by_dimension[key], term.quantity))
    inputs = tuple(ordered)
    output = None
    if all(value.status == "observed" for value in inputs):
        total = sum(
            (Fraction(term.weight) * Fraction(value.value) for term, value in zip(declared.terms, inputs, strict=True)),
            Fraction(),
        )
        output = Scalar(spec=declared.output, value=checked_fraction(total), status="observed")
    return inputs, output


def evaluate_static_anchor(definition: StaticQuadraticAnchor, weights: tuple[Weight, ...]) -> AnchorEvaluation:
    """Evaluate a static squared-distance penalty in declared objective units.

    Parameters
    ----------
    definition : StaticQuadraticAnchor
        Explicit fixed anchor and nonnegative strength.
    weights : tuple of Weight
        Complete coefficient vector with the same exact expert versions.

    Returns
    -------
    AnchorEvaluation
        One penalty receipt; the base objective remains unchanged.

    Raises
    ------
    ValueError
        Incompatible coordinates or nonrepresentable/out-of-schema result.

    Notes
    -----
    Scaling the base objective requires the same scaling of strength to retain
    its tradeoff. No fee, constraint repair, repeated update or I/O occurs.
    """
    declared = StaticQuadraticAnchor.model_validate(definition)
    checked, output = _anchor_facts(declared, weights)
    return AnchorEvaluation(
        schema_version=1, definition=declared.binding, specification=declared, weights=checked, output=output
    )


def _anchor_facts(declared: StaticQuadraticAnchor, weights: tuple[Weight, ...]) -> tuple[tuple[Weight, ...], Scalar]:
    """Derive a pure static penalty without recursively constructing its receipt."""
    checked = coefficient_weights(weights, declared.profile)
    if tuple(item.member for item in checked) != tuple(item.member for item in declared.weights):
        raise ValueError("Anchor evaluation requires identical expert IDs and versions")
    squared = sum(
        (
            (Fraction(value.value) - Fraction(anchor.value)) ** 2
            for value, anchor in zip(checked, declared.weights, strict=True)
        ),
        Fraction(),
    )
    output = Scalar(
        spec=declared.output, value=checked_fraction(Fraction(declared.strength) * squared), status="observed"
    )
    return checked, output
