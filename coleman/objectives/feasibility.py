"""Bounded finite constraint evidence and conservative fallback selection."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, Scalar
from coleman.core.primitives import MAX_DIMENSIONS, MAX_MEMBERS, Fingerprint, canonical_json
from coleman.objectives.bindings import DefinitionBinding, QuantityMeaning, SemanticDefinition, require_quantity

FeasibilityStatus = Literal["feasible", "infeasible", "insufficient_evidence"]


class ScalarConstraint(SemanticDefinition):
    """Declare inclusive scalar limits with exact units and interpretation.

    Attributes
    ----------
    quantity : QuantityMeaning
        Exact constrained schema, interpretation and horizon.
    lower, upper : float or None
        Required nullable finite limits in the quantity's declared units. At
        least one is required; equal limits express equality.

    Raises
    ------
    ValueError
        Both limits absent, a nonfinite/coerced limit or lower above upper.

    Notes
    -----
    Separately valid conflicting constraints remain valid declarations and
    produce typed infeasibility. No units, risk formula, nonlinear relation,
    optimizer, physical side effect or I/O is inferred.
    """

    quantity: QuantityMeaning
    lower: Binary64 | None
    upper: Binary64 | None

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Distinguish a malformed single interval from a conflicting set."""
        if self.lower is None and self.upper is None:
            raise ValueError("A scalar constraint requires at least one bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("A scalar constraint lower bound exceeds its upper bound")
        return self


class FiniteAlternative(CoreImmutable):
    """Supply known or explicitly missing quantities for one alternative.

    Attributes
    ----------
    member : Reference
        Exact alternative identity/version in the declared finite universe.
    values : tuple of Scalar
        At most 64 unique dimensions with explicit statuses. Extra unconstrained
        values are retained; no implicit conversion or imputation occurs.
    """

    member: Reference
    values: Annotated[tuple[Scalar, ...], Field(max_length=MAX_DIMENSIONS)]

    @model_validator(mode="after")
    def _values(self) -> Self:
        """Reject conflicting dimension versions before canonical ordering."""
        ordered_references(tuple(value.spec.dimension for value in self.values))
        object.__setattr__(self, "values", tuple(sorted(self.values, key=lambda value: value.spec.dimension.id)))
        return self


class FiniteFeasibilityRequest(CoreContract):
    """Bind a finite declared universe, observations and scalar constraints.

    Attributes
    ----------
    universe : tuple of Reference
        At most 4096 unique supplied alternative identities, sorted by ID.
    universe_complete : bool
        Strict caller attestation about modeled alternatives, not proof that all
        real-world possibilities were enumerated. False limits global claims.
    alternatives : tuple of FiniteAlternative
        At most one row per exact universe member; missing rows remain unknown.
    constraints : tuple of ScalarConstraint
        At most 64 unique named definitions. Multiple limits on one dimension
        must agree on its complete quantity meaning, schema and horizon.

    Raises
    ------
    ValueError
        Duplicate IDs, out-of-universe versions or inconsistent quantity schemas.

    Notes
    -----
    No alternative generation, catalogue completion, optimization, execution,
    registration or I/O occurs. This boundary preserves v1 core schemas.
    """

    universe: Annotated[tuple[Reference, ...], Field(max_length=MAX_MEMBERS)]
    universe_complete: Annotated[bool, Field(strict=True)]
    alternatives: Annotated[tuple[FiniteAlternative, ...], Field(max_length=MAX_MEMBERS)]
    constraints: Annotated[tuple[ScalarConstraint, ...], Field(max_length=MAX_DIMENSIONS)]

    @model_validator(mode="after")
    def _request(self) -> Self:
        """Canonicalize sets and validate exact versioned quantity boundaries."""
        universe = ordered_references(self.universe)
        members = ordered_references(tuple(alternative.member for alternative in self.alternatives))
        if not set((member.id, member.version) for member in members).issubset(
            (member.id, member.version) for member in universe
        ):
            raise ValueError("Alternatives require exact members of the declared universe")
        ordered_references(tuple(constraint.reference for constraint in self.constraints))
        meanings = _constraint_meanings(self.constraints)
        for alternative in self.alternatives:
            for value in alternative.values:
                if value.spec.dimension.id in meanings:
                    require_quantity(value, meanings[value.spec.dimension.id])
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "alternatives", tuple(sorted(self.alternatives, key=lambda row: row.member.id)))
        object.__setattr__(self, "constraints", tuple(sorted(self.constraints, key=lambda item: item.reference.id)))
        return self


class ConstraintEvidence(CoreImmutable):
    """Retain one alternative's actual scalar comparison or missing support.

    Attributes
    ----------
    constraint : DefinitionBinding
        Complete constraint identity, including its supplied lineage.
    value : Scalar or None
        Original scalar/status or null if the dimension is not supplied.
    status : {'pass', 'fail', 'unknown'}
        Literal finite comparison result, not statistical inference.
    reason : str
        Satisfied limit, lower/upper violation, absent dimension or missing value.
    """

    constraint: DefinitionBinding
    value: Scalar | None
    status: Literal["pass", "fail", "unknown"]
    reason: Literal["satisfied", "below_lower", "above_upper", "missing_dimension", "missing_value"]

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Keep completed comparisons distinct from missing scalar support."""
        expected = {
            "satisfied": "pass",
            "below_lower": "fail",
            "above_upper": "fail",
            "missing_dimension": "unknown",
            "missing_value": "unknown",
        }
        if self.status != expected[self.reason]:
            raise ValueError("Constraint evidence status and reason disagree")
        if self.reason == "missing_dimension":
            if self.value is not None:
                raise ValueError("Absent dimension evidence cannot contain a scalar")
        elif self.value is None or (self.value.status == "observed") != (self.status != "unknown"):
            raise ValueError("Constraint evidence disagrees with scalar support")
        return self


class AlternativeFeasibility(CoreImmutable):
    """Declare bounded per-alternative evidence, including missing rows.

    Attributes
    ----------
    member : Reference
        Exact alternative identity/version.
    present : bool
        Whether an input row was actually supplied.
    evidence : tuple of ConstraintEvidence
        Canonical named comparisons; missing rows have none.
    status : {'feasible', 'infeasible', 'insufficient_evidence'}
        A known violation suffices for infeasible; missing data alone never does.
    """

    member: Reference
    present: Annotated[bool, Field(strict=True)]
    evidence: Annotated[tuple[ConstraintEvidence, ...], Field(max_length=MAX_DIMENSIONS)]
    status: FeasibilityStatus

    @model_validator(mode="after")
    def _status(self) -> Self:
        """Require a classification justified by the retained comparisons."""
        ordered_references(tuple(item.constraint.reference for item in self.evidence))
        object.__setattr__(
            self, "evidence", tuple(sorted(self.evidence, key=lambda item: item.constraint.reference.id))
        )
        if not self.present and self.evidence:
            raise ValueError("An absent alternative cannot contain comparison evidence")
        expected = _alternative_status(self.present, self.evidence)
        if self.status != expected:
            raise ValueError("Alternative feasibility status disagrees with its evidence")
        return self


class FiniteFeasibilityResult(CoreContract):
    """Record finite feasibility without claiming an optimum or full catalogue.

    Attributes
    ----------
    request_fingerprint : str
        Exact supplied universe/observations/constraint content digest.
    request : FiniteFeasibilityRequest
        Full typed input, retained so guarded readers recheck actual comparisons
        and completeness instead of trusting a claimed status or source digest.
    universe_complete : bool
        Retained completeness attestation, not authenticity or global knowledge.
    alternatives : tuple of AlternativeFeasibility
        Complete per-member evidence for the supplied declared universe.
    conflicting_constraints : tuple of DefinitionBinding
        Valid individual limits whose intersection, including declared schema
        bounds, is empty. These prove modeled infeasibility without imputation.
    status : {'feasible', 'infeasible', 'insufficient_evidence'}
        Feasible means some supplied alternative passed, never globally optimal.

    Notes
    -----
    Unknown alternatives cannot complete a catalogue or justify an optimality
    claim. A digest identifies records and does not authenticate any attestation.
    Readers recompute all comparisons and completeness from the full retained
    request; source authenticity is distinct from correctly verified arithmetic.
    No fallback is executed or checkpoint/event state changed.
    """

    request_fingerprint: Fingerprint
    request: FiniteFeasibilityRequest
    universe_complete: Annotated[bool, Field(strict=True)]
    alternatives: Annotated[tuple[AlternativeFeasibility, ...], Field(max_length=MAX_MEMBERS)]
    conflicting_constraints: Annotated[tuple[DefinitionBinding, ...], Field(max_length=MAX_DIMENSIONS)]
    status: FeasibilityStatus

    @model_validator(mode="after")
    def _result(self) -> Self:
        """Recompute comparisons, conflicts and aggregate status from full input."""
        ordered_references(tuple(row.member for row in self.alternatives))
        ordered_references(tuple(binding.reference for binding in self.conflicting_constraints))
        object.__setattr__(self, "alternatives", tuple(sorted(self.alternatives, key=lambda row: row.member.id)))
        object.__setattr__(
            self,
            "conflicting_constraints",
            tuple(sorted(self.conflicting_constraints, key=lambda binding: binding.reference.id)),
        )
        alternatives, conflicts, status = _finite_facts(self.request)
        actual = [
            [row.model_dump(mode="json") for row in self.alternatives],
            [binding.model_dump(mode="json") for binding in self.conflicting_constraints],
        ]
        expected = [
            [row.model_dump(mode="json") for row in alternatives],
            [binding.model_dump(mode="json") for binding in conflicts],
        ]
        if (
            self.request_fingerprint != self.request.fingerprint
            or self.universe_complete != self.request.universe_complete
            or self.status != status
            or canonical_json(actual) != canonical_json(expected)
        ):
            raise ValueError("Finite feasibility receipt does not reproduce its complete request and retained support")
        return self

    @property
    def feasible_members(self) -> tuple[Reference, ...]:
        """Return explicitly qualified alternatives without selecting a winner.

        Returns
        -------
        tuple of Reference
            Exact member references whose retained evidence is feasible.

        Notes
        -----
        This is a filter on supplied evidence, not an optimizer or side effect.
        """
        if self.conflicting_constraints:
            return ()
        return tuple(row.member for row in self.alternatives if row.status == "feasible")


class FallbackDecision(CoreContract):
    """Declare a conservative fallback choice without executing an action.

    Attributes
    ----------
    evidence : str
        Exact finite-result fingerprint supporting this declaration.
    kind : {'abstain', 'baseline'}
        Only an explicitly feasible baseline can be retained.
    member : Reference or None
        Qualified baseline, or explicit null for abstention.
    reason : {'feasible_baseline', 'no_baseline', 'unqualified_baseline'}
        Why this declaration selected a baseline or abstained.

    Notes
    -----
    No random choice, replacement alternative, sampler preservation, execution,
    logging or I/O occurs. Hosts still own actual decision/sampling boundaries.
    """

    evidence: Fingerprint
    kind: Literal["abstain", "baseline"]
    member: Reference | None
    reason: Literal["feasible_baseline", "no_baseline", "unqualified_baseline"]

    @model_validator(mode="after")
    def _choice(self) -> Self:
        """Keep a qualified baseline distinct from an abstention declaration."""
        baseline = self.kind == "baseline"
        if baseline != (self.member is not None) or baseline != (self.reason == "feasible_baseline"):
            raise ValueError("Fallback kind, member and reason disagree")
        return self


def _constraint_meanings(constraints: tuple[ScalarConstraint, ...]) -> dict[str, QuantityMeaning]:
    """Require one complete quantity schema per logical dimension."""
    meanings: dict[str, QuantityMeaning] = {}
    for constraint in constraints:
        key = constraint.quantity.spec.dimension.id
        previous = meanings.setdefault(key, constraint.quantity)
        if canonical_json(previous.model_dump(mode="json")) != canonical_json(
            constraint.quantity.model_dump(mode="json")
        ):
            raise ValueError("Constraints disagree on exact quantity schema or horizon")
    return meanings


def _comparison(constraint: ScalarConstraint, value: Scalar | None) -> ConstraintEvidence:
    """Compare one observed finite scalar, retaining absence explicitly."""
    reason: Literal["satisfied", "below_lower", "above_upper", "missing_dimension", "missing_value"]
    if value is None:
        status, reason = "unknown", "missing_dimension"
    elif value.value is None:
        status, reason = "unknown", "missing_value"
    elif constraint.lower is not None and value.value < constraint.lower:
        status, reason = "fail", "below_lower"
    elif constraint.upper is not None and value.value > constraint.upper:
        status, reason = "fail", "above_upper"
    else:
        status, reason = "pass", "satisfied"
    return ConstraintEvidence(constraint=constraint.binding, value=value, status=status, reason=reason)


def _alternative_status(present: bool, evidence: tuple[ConstraintEvidence, ...]) -> FeasibilityStatus:
    """Known violations dominate unknown values without manufacturing evidence."""
    if any(item.status == "fail" for item in evidence):
        return "infeasible"
    if not present or any(item.status == "unknown" for item in evidence):
        return "insufficient_evidence"
    return "feasible"


def _conflicting_constraints(constraints: tuple[ScalarConstraint, ...]) -> tuple[DefinitionBinding, ...]:
    """Detect an empty intersection of individually valid declared limits."""
    groups: dict[str, list[ScalarConstraint]] = {}
    for constraint in constraints:
        groups.setdefault(constraint.quantity.spec.dimension.id, []).append(constraint)
    conflicts = []
    for items in groups.values():
        spec = items[0].quantity.spec
        lowers = [item.lower for item in items if item.lower is not None]
        uppers = [item.upper for item in items if item.upper is not None]
        if spec.minimum is not None:
            lowers.append(spec.minimum)
        if spec.maximum is not None:
            uppers.append(spec.maximum)
        if lowers and uppers and max(lowers) > min(uppers):
            conflicts.extend(item.binding for item in items)
    return tuple(conflicts)


def _result_status(
    alternatives: tuple[AlternativeFeasibility, ...], complete: bool, conflicts: bool
) -> FeasibilityStatus:
    """Limit infeasibility to a complete declared catalogue or hard proof."""
    if conflicts:
        return "infeasible"
    if any(row.status == "feasible" for row in alternatives):
        return "feasible"
    if complete and all(row.status == "infeasible" for row in alternatives):
        return "infeasible"
    return "insufficient_evidence"


def check_finite_feasibility(request: FiniteFeasibilityRequest) -> FiniteFeasibilityResult:
    """Filter supplied finite alternatives using exact declared scalar schemas.

    Parameters
    ----------
    request : FiniteFeasibilityRequest
        At most 4096 declared alternatives and 64 explicit scalar constraints.

    Returns
    -------
    FiniteFeasibilityResult
        Per-alternative evidence plus feasible, infeasible or insufficient
        support for the supplied model. Unknown data never becomes zero.

    Raises
    ------
    ValueError
        Malformed schemas, conflicting dimension versions, incompatible units
        or an unchecked input that fails full boundary revalidation.

    Notes
    -----
    Valid limits with an empty intersection produce typed infeasibility. An
    empty attested-complete universe is infeasible; incomplete coverage remains
    unknown. Completeness is a caller attestation, never proof about the world.
    No optimization, alternative generation, implicit unit conversion, side
    effect, learner update, catalogue completion or I/O occurs.
    """
    declared = FiniteFeasibilityRequest.model_validate(request)
    results, conflicts, status = _finite_facts(declared)
    return FiniteFeasibilityResult(
        schema_version=1,
        request_fingerprint=declared.fingerprint,
        request=declared,
        universe_complete=declared.universe_complete,
        alternatives=results,
        conflicting_constraints=conflicts,
        status=status,
    )


def _finite_facts(
    declared: FiniteFeasibilityRequest,
) -> tuple[tuple[AlternativeFeasibility, ...], tuple[DefinitionBinding, ...], FeasibilityStatus]:
    """Evaluate supplied constraints without constructing a recursive result."""
    rows = {(row.member.id, row.member.version): row for row in declared.alternatives}
    alternatives = []
    for member in declared.universe:
        row = rows.get((member.id, member.version))
        values = {} if row is None else {value.spec.dimension.id: value for value in row.values}
        evidence = (
            ()
            if row is None
            else tuple(
                _comparison(constraint, values.get(constraint.quantity.spec.dimension.id))
                for constraint in declared.constraints
            )
        )
        alternatives.append(
            AlternativeFeasibility(
                member=member,
                present=row is not None,
                evidence=evidence,
                status=_alternative_status(row is not None, evidence),
            )
        )
    results = tuple(alternatives)
    conflicts = _conflicting_constraints(declared.constraints)
    return results, conflicts, _result_status(results, declared.universe_complete, bool(conflicts))


def select_fallback(result: FiniteFeasibilityResult, baseline: Reference | None = None) -> FallbackDecision:
    """Retain only an explicitly feasible supplied baseline, otherwise abstain.

    Parameters
    ----------
    result : FiniteFeasibilityResult
        Full retained request and revalidated finite evidence, including actual
        scalar comparisons, exact identities/versions and completeness.
    baseline : Reference or None, optional
        Requested baseline, or absence requiring abstention. Existing identity
        alone does not establish compliance with hard constraints.

    Returns
    -------
    FallbackDecision
        Pure content-bound declaration; it executes or samples nothing.

    Raises
    ------
    ValueError
        Invalid evidence or baseline reference.

    Notes
    -----
    No substitute winner is selected. Missing or infeasible baseline evidence
    stays unqualified, and actual host filtering/overrides must still declare
    preserved/recomputed/invalidated sampling under the existing core contract.
    """
    checked = FiniteFeasibilityResult.model_validate(result)
    candidate = None if baseline is None else Reference.model_validate(baseline)
    accepted = candidate is not None and candidate in checked.feasible_members
    reason: Literal["feasible_baseline", "no_baseline", "unqualified_baseline"] = "unqualified_baseline"
    if accepted:
        reason = "feasible_baseline"
    elif candidate is None:
        reason = "no_baseline"
    return FallbackDecision(
        schema_version=1,
        evidence=checked.fingerprint,
        kind="baseline" if accepted else "abstain",
        member=candidate if accepted else None,
        reason=reason,
    )
