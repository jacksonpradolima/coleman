"""Typed semantic envelopes above the unchanged v1 mathematical core records."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.core.actions import MixtureSpace, space_members
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.decisions import DecisionRequest
from coleman.core.identity import ordered_references
from coleman.core.primitives import MAX_DIMENSIONS, MAX_MEMBERS
from coleman.objectives.bindings import DefinitionBinding, QuantityMeaning, SemanticDefinition
from coleman.objectives.controls import L1MovementPenalty, MixtureControls
from coleman.objectives.feasibility import ScalarConstraint
from coleman.objectives.preferences import InitialPrior, ObjectiveWeights, ReferencePoints, StaticQuadraticAnchor
from coleman.objectives.transforms import AffineLoss, IdentityLoss
from coleman.objectives.volatility import ScaleSelection, SignedVolatilitySpec

TransformDefinition = Annotated[IdentityLoss | AffineLoss | SignedVolatilitySpec, Field(discriminator="kind")]


class ObjectiveTerm(CoreImmutable):
    """Bind one raw objective to its declared direction and exact loss definition.

    Attributes
    ----------
    quantity : QuantityMeaning
        Raw objective schema, interpretation and measurement horizon.
    direction : {'minimize', 'maximize'}
        Explicit raw-objective direction, independent of coefficient signs.
    transform : DefinitionBinding
        Exact loss definition reference and content, not a bare operation name.

    Notes
    -----
    The containing context checks the supplied definition; this leaf performs
    no lookup, transformation, utility inference or I/O.
    """

    quantity: QuantityMeaning
    direction: Literal["minimize", "maximize"]
    transform: DefinitionBinding


class ObjectiveDefinition(SemanticDefinition):
    """Declare a bounded vector of objectives without implicit scalar tradeoffs.

    Attributes
    ----------
    terms : tuple of ObjectiveTerm
        One through 64 unique dimension IDs, canonically ordered. Every term
        retains units, role, interpretation, horizon and transform content.

    Raises
    ------
    ValueError
        Empty/oversized terms or duplicate/conflicting dimension identities.

    Notes
    -----
    Different objectives remain a vector until explicit ObjectiveWeights are
    supplied. No ranking, scalarization, optimization or I/O occurs here.
    """

    terms: Annotated[tuple[ObjectiveTerm, ...], Field(min_length=1, max_length=MAX_DIMENSIONS)]

    @field_validator("terms")
    @classmethod
    def _terms(cls, values: tuple[ObjectiveTerm, ...]) -> tuple[ObjectiveTerm, ...]:
        """Reject ambiguous dimensions before canonicalizing set declarations."""
        ordered_references(tuple(term.quantity.spec.dimension for term in values))
        return tuple(sorted(values, key=lambda term: term.quantity.spec.dimension.id))


def _input(definition: IdentityLoss | AffineLoss | SignedVolatilitySpec) -> QuantityMeaning:
    """Get the exact operation input without inferring interpretation from shape."""
    return definition.quantity if isinstance(definition, IdentityLoss) else definition.input


def _output(definition: IdentityLoss | AffineLoss | SignedVolatilitySpec) -> QuantityMeaning:
    """Get the declared normalized-loss output of one qualified operation."""
    return definition.quantity if isinstance(definition, IdentityLoss) else definition.output


class SemanticContext(SemanticDefinition):
    """Supply content-bound objectives, transforms and independent control inputs.

    Attributes
    ----------
    objective : ObjectiveDefinition
        Exact raw vector/direction/transform contract.
    transforms : tuple of IdentityLoss or AffineLoss or SignedVolatilitySpec
        Complete unique definitions required by the objective terms.
    constraints : tuple of ScalarConstraint, optional
        Explicit hard scalar bounds, independently checked by finite feasibility.
    preferences : ObjectiveWeights or None, optional
        Declared tradeoffs over exactly the derived loss dimensions. None means
        no preference operation; equal weights are not fabricated.
    initial_prior : InitialPrior or None, optional
        Initial expert coefficients, distinct from ongoing objective operations.
    anchor : StaticQuadraticAnchor or None, optional
        Explicit static penalty, not the target-share or L1 movement profile.
    reference_points : ReferencePoints or None, optional
        Subjective target values in the exact raw objective schemas.
    controls : MixtureControls or None, optional
        Explicit coefficient bounds, origin, cadence and output-profile contract.
    scale_selections : tuple of ScaleSelection, optional
        Frozen producing-decision scale choices, not later refits. The request
        envelope checks their request fingerprints and original timing.

    Raises
    ------
    ValueError
        Conflicting identities/content, missing/extraneous transforms or
        incompatible quantity, preference, prior or scale definitions.

    Notes
    -----
    The context does not execute transforms, constraints or learners. Passing it
    does not authenticate provenance or qualify scientific efficacy. Existing
    bare v1 records remain readable and acquire no verified-context claim.
    """

    objective: ObjectiveDefinition
    transforms: Annotated[tuple[TransformDefinition, ...], Field(min_length=1, max_length=MAX_DIMENSIONS)]
    constraints: Annotated[tuple[ScalarConstraint, ...], Field(max_length=MAX_DIMENSIONS)] = ()
    preferences: ObjectiveWeights | None = None
    initial_prior: InitialPrior | None = None
    anchor: StaticQuadraticAnchor | None = None
    reference_points: ReferencePoints | None = None
    controls: MixtureControls | None = None
    scale_selections: Annotated[tuple[ScaleSelection, ...], Field(max_length=MAX_MEMBERS)] = ()

    @model_validator(mode="after")
    def _definitions(self) -> Self:
        """Check complete operation contents and canonical declaration ordering."""
        ordered_references(tuple(definition.reference for definition in self.transforms))
        ordered_references(tuple(constraint.reference for constraint in self.constraints))
        definitions = {definition.binding.model_dump_json(): definition for definition in self.transforms}
        if set(definitions) != {term.transform.model_dump_json() for term in self.objective.terms}:
            raise ValueError("Context must supply exactly the objective's bound transform definitions")
        ordered_references(tuple(_output(definition).spec.dimension for definition in self.transforms))
        for term in self.objective.terms:
            definition = definitions[term.transform.model_dump_json()]
            if _input(definition).model_dump_json() != term.quantity.model_dump_json():
                raise ValueError("Objective term does not match its transform's exact input meaning")
            direction = (
                definition.direction
                if isinstance(definition, AffineLoss)
                else ("minimize" if isinstance(definition, IdentityLoss) else "maximize")
            )
            if term.direction != direction:
                raise ValueError("Objective direction disagrees with its loss transformation")
        self._preference_inputs()
        self._selections(definitions)
        object.__setattr__(self, "transforms", tuple(sorted(self.transforms, key=lambda item: item.reference.id)))
        object.__setattr__(self, "constraints", tuple(sorted(self.constraints, key=lambda item: item.reference.id)))
        return self

    def _preference_inputs(self) -> None:
        """Keep subjective scalarization, raw targets and coefficient origins distinct."""
        if self.preferences is not None:
            expected = sorted(_output(item).model_dump_json() for item in self.transforms)
            actual = sorted(term.quantity.model_dump_json() for term in self.preferences.terms)
            if expected != actual:
                raise ValueError("Preference weights must name exactly the context's normalized-loss quantities")
        if self.reference_points is not None:
            allowed = {term.quantity.model_dump_json() for term in self.objective.terms}
            if any(point.quantity.model_dump_json() not in allowed for point in self.reference_points.points):
                raise ValueError("Reference points require exact declared raw objective meanings")
        if (
            self.controls is not None
            and self.initial_prior is not None
            and self.initial_prior.binding != self.controls.initial_prior.binding
        ):
            raise ValueError("Context and mixture controls cannot declare conflicting initial priors")

    def _selections(self, definitions: dict[str, IdentityLoss | AffineLoss | SignedVolatilitySpec]) -> None:
        """Validate supplied scale definitions and reject conflicting logical selections."""
        keys = []
        for selection in self.scale_selections:
            if selection.definition.binding.model_dump_json() not in definitions:
                raise ValueError("Scale selection does not belong to a supplied loss definition")
            member = None if selection.member is None else (selection.member.id, selection.member.version)
            keys.append((selection.definition.reference.id, selection.outcome.id, member))
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate or conflicting scale selections are not merged")
        ordered = tuple(sorted(self.scale_selections, key=lambda item: item.fingerprint))
        object.__setattr__(self, "scale_selections", ordered)


class SemanticRequest(CoreContract):
    """Bind a pure v1 request to full objective/control definition contents.

    Attributes
    ----------
    request : DecisionRequest
        Unchanged core input, including its logical task/opportunity identity.
    context : SemanticContext
        Full immutable semantic content. Changes affect the outer fingerprint
        even when a human-readable objective reference is unchanged.

    Raises
    ------
    ValueError
        Objective/member mismatch or a scale selected for another request/time.

    Notes
    -----
    This optional envelope supplies stronger content identity; it does not
    retrofit semantic verification into old records, execute a workflow or
    promise durable idempotency. Pure math still needs no request envelope.
    """

    request: DecisionRequest
    context: SemanticContext

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Bind definitions and frozen scales to the actual producing request."""
        if self.request.objective != self.context.objective.reference:
            raise ValueError("Request objective does not match the supplied definition")
        self._members()
        self._known_declarations()
        self._scale_coverage()
        for selection in self.context.scale_selections:
            if selection.decision_fingerprint != self.request.fingerprint or selection.timing != self.request.timing:
                raise ValueError("Scale selection is not bound to this producing request and its original timing")
        return self

    def _known_declarations(self) -> None:
        """Reject known post-cutoff configurations without fabricating unknown facts."""
        if self.request.timing is None:
            return
        definitions: list[SemanticDefinition | None] = [
            self.context,
            self.context.objective,
            *self.context.transforms,
            *self.context.constraints,
            self.context.preferences,
            self.context.initial_prior,
            self.context.anchor,
            self.context.reference_points,
        ]
        controls = self.context.controls
        if controls is not None:
            definitions.extend([controls, controls.initial_prior, controls.target, controls.bounds, controls.movement])
            if isinstance(controls.output, L1MovementPenalty):
                definitions.extend([controls.output, controls.output.origin])
        for selection in self.context.scale_selections:
            definitions.extend([selection.definition, selection.scale.definition])
        cutoff = self.request.timing.information_cutoff
        for definition in definitions:
            if definition is not None:
                instant = definition.lineage.declared_at
                if instant is not None and instant > cutoff:
                    raise ValueError("A known definition declaration follows the producing information cutoff")

    def _scale_coverage(self) -> None:
        """Require a frozen global scale or complete member-indexed choices per loss."""
        universe = {member.model_dump_json() for member in space_members(self.request.space)}
        for definition in self.context.transforms:
            if not isinstance(definition, SignedVolatilitySpec):
                continue
            selections = [
                item for item in self.context.scale_selections if item.definition.binding == definition.binding
            ]
            if not selections:
                raise ValueError("Signed-volatility semantic requests require preselected scale coverage")
            global_selections = [item for item in selections if item.member is None]
            if global_selections:
                if len(selections) != 1:
                    raise ValueError("A global scale choice cannot be mixed with additional member/outcome choices")
                continue
            members = [item.member.model_dump_json() for item in selections if item.member is not None]
            if len(members) != len(set(members)) or set(members) != universe:
                raise ValueError("Scale selections must cover the exact member universe once")
            if (
                selections[0].scale.definition.scope == "shared"
                and len({item.scale.fingerprint for item in selections}) != 1
            ):
                raise ValueError("Shared per-member scale selections must reuse the same frozen scale")

    def _members(self) -> None:
        """Reject expert controls applied to another universe or action capability."""
        if self.context.controls is not None and not isinstance(self.request.space, MixtureSpace):
            raise ValueError("Mixture controls require an explicit mixture action space")
        prior = self.context.initial_prior
        if self.context.controls is not None:
            prior = self.context.controls.initial_prior
        universe = {member.model_dump_json() for member in space_members(self.request.space)}
        if prior is not None and {weight.member.model_dump_json() for weight in prior.weights} != universe:
            raise ValueError("Initial prior does not match the exact requested member universe")
        if (
            self.context.anchor is not None
            and {weight.member.model_dump_json() for weight in self.context.anchor.weights} != universe
        ):
            raise ValueError("Static anchor does not match the exact requested member universe")
        if any(
            selection.member is not None and selection.member.model_dump_json() not in universe
            for selection in self.context.scale_selections
        ):
            raise ValueError("Scale selection member is outside the exact requested universe")

    @property
    def logical_identity(self) -> str:
        """Return the unchanged core task/opportunity identity.

        Returns
        -------
        str
            Stable logical digest, independent of parameter/content changes.

        Notes
        -----
        Equal logical identity is not proof of an idempotent retry. Compare the
        complete outer fingerprint before session acceptance; no I/O occurs.
        """
        return self.request.logical_identity


def deduplicate_semantic_requests(values: tuple[SemanticRequest, ...]) -> tuple[SemanticRequest, ...]:
    """Reconcile bounded local retries without accepting changed semantic content.

    Parameters
    ----------
    values : tuple of SemanticRequest
        At most 4096 attempted immutable envelopes, including exact retries.

    Returns
    -------
    tuple of SemanticRequest
        Unique envelopes in logical-identity order; inputs remain unchanged.

    Raises
    ------
    ValueError
        Invalid shape/count, unchecked content or conflicting context at one key.

    Notes
    -----
    Performs no durable acceptance, learner update, historical rewrite or I/O.
    Session-level concurrency/deduplication remains a separate responsibility.
    """
    if not isinstance(values, tuple) or len(values) > MAX_MEMBERS:
        raise ValueError("Semantic retries require at most 4096 attempted envelopes")
    unique: dict[str, SemanticRequest] = {}
    for value in values:
        checked = SemanticRequest.model_validate(value)
        previous = unique.setdefault(checked.logical_identity, checked)
        if previous.to_json() != checked.to_json():
            raise ValueError("Conflicting semantic content at one logical decision")
    return tuple(unique[key] for key in sorted(unique))
