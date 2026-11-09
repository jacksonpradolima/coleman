"""Strict small continuous allocation declarations, independent of execution."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from coleman.core._wire import BoundedContract
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage

Point = Annotated[tuple[Binary64, ...], Field(min_length=1, max_length=4)]
Matrix = Annotated[tuple[Point, ...], Field(min_length=1, max_length=4)]
ALLOCATION_COEFFICIENT_SEMANTICS = UnitSemantics(
    unit="1", meaning="unit-reference-balance-noncash-allocation-coefficient"
)


class ContinuousContract(BoundedContract):
    """Keep strict v1 continuous records within 16MiB and lexical depth64.

    Notes
    -----
    Shared core graph, duplicate-key, unknown-field and revalidation guards
    apply. These mathematical records provide no I/O or execution authority.
    """

    _maximum_bytes = 16 * 1024**2

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Guard mutable input size before the shared reader makes a copy.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Built-in UTF8 input within the complete16MiB/depth64 envelope.
        **kwargs : object
            Existing strict Pydantic reader options, passed through unchanged.

        Returns
        -------
        Self
            Revalidated concrete known-version record.

        Raises
        ------
        ValueError, TypeError
            Wrong kind, oversized/deep content, unknown version or fields.
        """
        if type(json_data) not in {str, bytes, bytearray}:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Continuous JSON requires built-in immutable or mutable text bytes")
        if isinstance(json_data, bytearray) and len(json_data) > cls._maximum_bytes:
            raise ValueError("Continuous mutable JSON exceeds its ceiling before copying")
        return super().model_validate_json(json_data, **kwargs)


class CoefficientInterval(ContinuousContract):
    """Declare inclusive finite dimensionless coefficient endpoints.

    Attributes
    ----------
    lower, upper : float
        Exact stored binary64 endpoints, including signed values and zero.
        Reversed bounds are malformed declarations, not solver infeasibility.
    """

    lower: Binary64
    upper: Binary64

    @model_validator(mode="after")
    def _order(self) -> Self:
        """Reject reversed endpoints without clipping."""
        if self.lower > self.upper:
            raise ValueError("Coefficient lower bound exceeds upper bound")
        return self


class ContinuousVariable(CoefficientInterval):
    """Name one bounded dimensionless noncash allocation coefficient.

    Attributes
    ----------
    member : Reference
        Complete stable identity/version; display labels are not coordinates.
    """

    member: Reference


class ContinuousSet(ContinuousContract):
    """Bind one compact convex mathematical coefficient set.

    Attributes
    ----------
    reference, lineage : Reference, DefinitionLineage
        Caller-owned version and explicit source/declaration facts.
    profile : {'simplex-v1', 'signed-box-v1'}
        Simplex additionally requires x>=0 and sum(x)=1. Signed-box permits
        negative coordinates. Both keep every original finite variable box.
    variables : tuple of ContinuousVariable
        One through four coordinates in unique ascending logical-ID order.
    net, cash : CoefficientInterval
        Explicit bounds on sum(x) and on implied cash c=1-sum(x).
    gross_limit, leverage_limit : float or None
        Optional nonnegative bounds on sum(abs(x)) and sum(abs(x))+abs(c).
        Null means no additional bound, never a missing measured resource.
    coefficient_semantics : UnitSemantics
        Fixed dimensionless unit-reference-balance meaning. Other units or
        monetary/physical interpretations require a separate domain adapter.

    Notes
    -----
    All quantities here are dimensionless coefficients in a declared unit
    reference balance. These formulas do not establish financing, borrow,
    collateral, margin, monetary wealth or qualified financial short positions.
    Cash is an implied coordinate, not an independently optimized asset.
    Jointly contradictory well-formed bounds are returned as infeasible.
    """

    reference: Reference
    lineage: DefinitionLineage
    profile: Literal["simplex-v1", "signed-box-v1"]
    variables: Annotated[tuple[ContinuousVariable, ...], Field(min_length=1, max_length=4)]
    net: CoefficientInterval
    cash: CoefficientInterval
    gross_limit: Annotated[Binary64, Field(ge=0)] | None
    leverage_limit: Annotated[Binary64, Field(ge=0)] | None
    coefficient_semantics: UnitSemantics = ALLOCATION_COEFFICIENT_SEMANTICS

    @model_validator(mode="after")
    def _coordinates(self) -> Self:
        """Require exact canonical coordinate identities without reordering."""
        identities = tuple(item.member.id for item in self.variables)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("Continuous coordinates require unique ascending member IDs")
        if self.profile == "simplex-v1" and any(item.lower < 0 or item.upper > 1 for item in self.variables):
            raise ValueError("Simplex variable boxes must lie in[0,1]")
        if self.coefficient_semantics != ALLOCATION_COEFFICIENT_SEMANTICS:
            raise ValueError("Continuous allocations require their declared dimensionless coefficient semantics")
        return self


class QuadraticObjective(ContinuousContract):
    """Bind f(x)=x'Qx/2+linear'x+constant and its dimensionless score.

    Attributes
    ----------
    reference, lineage : Reference, DefinitionLineage
        Complete declared objective version and explicit lineage.
    matrix, linear : tuple
        Finite symmetric Q and matching coefficient vector, dimension at most4.
        The solver qualifies only strictly positive-definite Q; singular or
        indefinite objectives return unsupported instead of adding jitter.
    constant : float
        Finite offset, default0; it changes scores, not the minimizer.
    score : ScalarSpec
        Explicit dimensionless objective interpretation, unbounded in value.
        Variance, target shortfall and CVaR are not interchangeable names.
    """

    reference: Reference
    lineage: DefinitionLineage
    matrix: Matrix
    linear: Point
    constant: Binary64 = 0.0
    score: ScalarSpec

    @model_validator(mode="after")
    def _shape(self) -> Self:
        """Require exact symmetry, dimensions and a complete score meaning."""
        count = len(self.linear)
        if len(self.matrix) != count or any(len(row) != count for row in self.matrix):
            raise ValueError("Quadratic matrix and linear dimensions differ")
        if any(self.matrix[i][j] != self.matrix[j][i] for i in range(count) for j in range(i)):
            raise ValueError("Quadratic matrix must be exactly symmetric")
        if self.score.role != "objective" or self.score.semantics.unit != "1":
            raise ValueError("Quadratic score requires an explicit dimensionless objective schema")
        if self.score.minimum is not None or self.score.maximum is not None:
            raise ValueError("This quadratic profile requires an unbounded score schema")
        return self


class SolveLimits(ContinuousContract):
    """Bound deterministic face work and each derived rational integer.

    Attributes
    ----------
    maximum_faces : int
        One through65536 attempted active faces, default4096. Exhaustion is
        limited, never a proof of infeasibility or an extra online observation.
    maximum_integer_bits : int
        Between64 and32768 bits per derived numerator/denominator, default8192.
    """

    maximum_faces: Annotated[int, Field(strict=True, ge=1, le=65536)] = 4096
    maximum_integer_bits: Annotated[int, Field(strict=True, ge=64, le=32768)] = 8192


class SolverTolerances(ContinuousContract):
    """Name absolute binary64 emission qualification, not input repair.

    Attributes
    ----------
    primal, stationarity, complementarity : float
        Finite nonnegative absolute residual limits, at most1e-6. Defaults are
        1e-12,1e-10,1e-10. Zero requires exactly representable certificates.
    """

    primal: Annotated[Binary64, Field(ge=0, le=1e-6)] = 1e-12
    stationarity: Annotated[Binary64, Field(ge=0, le=1e-6)] = 1e-10
    complementarity: Annotated[Binary64, Field(ge=0, le=1e-6)] = 1e-10


class ContinuousProblem(ContinuousContract):
    """Retain the complete fixed feasible set, loss law and comparison context.

    Attributes
    ----------
    feasible_set, objective : ContinuousSet, QuadraticObjective
        Matching canonical coordinate dimension and complete declared content.
    comparator : DefinitionBinding
        Explicit comparator identity; no regret or domain guarantee is inferred.
    transition : DefinitionBinding or None
        Optional declared transition/accounting model. The solver does not
        execute it or turn objective penalties into realized charges.
    tolerances, limits : SolverTolerances, SolveLimits
        Fixed numerical qualification and finite deterministic work envelope.
    """

    feasible_set: ContinuousSet
    objective: QuadraticObjective
    comparator: DefinitionBinding
    transition: DefinitionBinding | None = None
    tolerances: SolverTolerances = Field(default_factory=lambda: SolverTolerances(schema_version=1))
    limits: SolveLimits = Field(default_factory=lambda: SolveLimits(schema_version=1))
    numerical_profile: Literal["small-spd-rational-active-face-binary64-v1"] = (
        "small-spd-rational-active-face-binary64-v1"
    )

    @model_validator(mode="after")
    def _dimension(self) -> Self:
        """Refuse a loss law that omits or adds a coordinate."""
        if len(self.feasible_set.variables) != len(self.objective.linear):
            raise ValueError("Continuous objective and feasible-set dimensions differ")
        return self


class KktResiduals(ContinuousContract):
    """Retain emitted primal, stationarity and complementarity absolute errors."""

    primal: Annotated[Binary64, Field(ge=0)]
    stationarity: Annotated[Binary64, Field(ge=0)]
    complementarity: Annotated[Binary64, Field(ge=0)]


class ContinuousCertificate(ContinuousContract):
    """Keep recheckable emitted-point KKT evidence in original constraint order.

    Attributes
    ----------
    value, gradient : float, tuple
        Actual emitted objective and analytic gradient under the retained law.
    duals : tuple of float
        Nonnegative multipliers of the deterministic compiled inequalities.
    equality_dual : float or None
        Simplex sum-one multiplier; null for signed-box without that equality.
    residuals : KktResiduals
        Independently recomputable emitted residuals under named tolerances.
    """

    value: Binary64
    gradient: Point
    duals: Annotated[tuple[Binary64, ...], Field(max_length=64)]
    equality_dual: Binary64 | None
    residuals: KktResiduals


class ContinuousSolution(ContinuousContract):
    """Retain proposed versus accepted output without silently repairing either.

    Attributes
    ----------
    problem : ContinuousProblem
        Full producing law, set, units, comparator and limits, not just a digest.
    status : str
        optimal, infeasible, unsupported, limited or numerical_failure.
        Only optimal carries an executed point and revalidated KKT certificate.
    reason : str
        Finite deterministic disposition category.
    proposal : tuple or None
        Unconstrained minimizer when representable; explicit null otherwise.
    executed, certificate : tuple or None, ContinuousCertificate or None
        One nearest binary64 rounding, checked without clipping/renormalizing.
    faces_examined : int
        Internal active faces tried, not samples, labels or online decisions.

    Notes
    -----
    Reading optimal results checks their bounded certificate, never enumerates
    faces. This is mathematical consistency, not execution authentication.
    """

    problem: ContinuousProblem
    status: Literal["optimal", "infeasible", "unsupported", "limited", "numerical_failure"]
    reason: Annotated[str, Field(min_length=1, max_length=256)]
    proposal: Point | None
    executed: Point | None
    certificate: ContinuousCertificate | None
    faces_examined: Annotated[int, Field(strict=True, ge=0, le=65536)]

    @model_validator(mode="after")
    def _evidence(self) -> Self:
        """Check closed output disposition and bounded mathematical joins."""
        # Deferred only to keep pure DTO and arithmetic module imports acyclic.
        from coleman.optimization._continuous_checks import validate_solution  # pylint: disable=import-outside-toplevel

        validate_solution(self)
        return self


class ProjectionRequest(ContinuousContract):
    """Bind a Euclidean projection proposal and its exact numerical context."""

    feasible_set: ContinuousSet
    proposal: Point
    comparator: DefinitionBinding
    tolerances: SolverTolerances = Field(default_factory=lambda: SolverTolerances(schema_version=1))
    limits: SolveLimits = Field(default_factory=lambda: SolveLimits(schema_version=1))

    @model_validator(mode="after")
    def _dimension(self) -> Self:
        """Require every original coordinate in the proposal."""
        if len(self.proposal) != len(self.feasible_set.variables):
            raise ValueError("Projection proposal and feasible-set dimensions differ")
        return self
