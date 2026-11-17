"""Small explicit coefficient laws for independent continuous qualification."""

from typing import Literal

from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.optimization.continuous_contracts import (
    CoefficientInterval,
    ContinuousProblem,
    ContinuousSet,
    ContinuousVariable,
    QuadraticObjective,
)


def ref(name):
    """Name one explicit fixture coordinate or semantic definition."""
    return Reference(id=name, version="1")


def lineage():
    """Keep source/clock facts unknown rather than inventing observations."""
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


def interval(lower, upper):
    """Create one strict known-version coefficient interval."""
    return CoefficientInterval(schema_version=1, lower=lower, upper=upper)


def geometry(
    profile: Literal["simplex-v1", "signed-box-v1"] = "simplex-v1",
    boxes=((0.0, 1.0), (0.0, 1.0)),
    *,
    gross=None,
    leverage=None,
):
    """Declare finite simplex or signed geometry with explicit cash/net context."""
    return ContinuousSet(
        schema_version=1,
        reference=ref("authored-continuous-set"),
        lineage=lineage(),
        profile=profile,
        variables=tuple(
            ContinuousVariable(schema_version=1, member=ref(chr(97 + i)), lower=lo, upper=hi)
            for i, (lo, hi) in enumerate(boxes)
        ),
        net=interval(-4.0, 4.0),
        cash=interval(-3.0, 5.0),
        gross_limit=gross,
        leverage_limit=leverage,
    )


def problem(feasible=None, *, matrix=((2.0, 1.0), (1.0, 2.0)), linear=(-2.5, -2.0)):
    """Declare the independently authored coupled quadratic and fixed comparator."""
    return ContinuousProblem(
        schema_version=1,
        feasible_set=geometry() if feasible is None else feasible,
        objective=QuadraticObjective(
            schema_version=1,
            reference=ref("authored-coupled-score"),
            lineage=lineage(),
            matrix=matrix,
            linear=linear,
            constant=0.0,
            score=ScalarSpec(
                dimension=ref("constructed-quadratic-score"),
                role="objective",
                semantics=UnitSemantics(unit="1", meaning="constructed-half-quadratic-loss"),
            ),
        ),
        comparator=DefinitionBinding(
            reference=ref("fixed-coefficient-comparator"),
            definition_fingerprint=fingerprint({"coefficients": [0.5, 0.5]}),
        ),
        transition=None,
    )
