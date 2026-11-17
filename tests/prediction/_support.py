"""Authored ridge declarations and an independent exact batch-fit oracle."""

from fractions import Fraction
from pathlib import Path

from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.prediction import AppendOnlyRidge, RidgeConfig, RidgeInput, RidgeLimits, RidgeSample

FIXTURES = Path(__file__).with_name("fixtures") / "v1"


def ref(name):
    """Return an authored public version-one identity."""
    return Reference(id=name, version="1")


def config(*, regularization=1.0, maximum_samples=65_536):
    """Declare two unitless features and a signed unitless synthetic target."""
    return RidgeConfig(
        schema_version=1,
        reference=ref("ci-ridge-reference"),
        lineage=DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
        features=tuple(
            ScalarSpec(
                dimension=ref(name), role="feature", semantics=UnitSemantics(unit="1", meaning="supplied-feature-v1")
            )
            for name in ("x0", "x1")
        ),
        target=ScalarSpec(
            dimension=ref("signed-ci-target"),
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="synthetic-ci-target-v1"),
        ),
        horizon=ref("one-ci-opportunity"),
        preprocessing=DefinitionBinding(
            reference=ref("frozen-dimensionless-features"),
            definition_fingerprint=fingerprint({"operation": "supplied identity; no learned preprocessing"}),
        ),
        regularization=regularization,
        limits=RidgeLimits(schema_version=1, maximum_samples=maximum_samples),
    )


def owner(configuration=None, *, name="ci-ridge-owner"):
    """Construct the real independent mathematical predictor."""
    return AppendOnlyRidge(config() if configuration is None else configuration, owner=ref(name))


def features(model, values):
    """Bind original ordered values to the complete actual configuration."""
    return RidgeInput(schema_version=1, config=model.config.binding, values=values)


def sample(model, values, target):
    """Supply a real observed label without a clock or delivery assertion."""
    return RidgeSample(
        schema_version=1,
        input=features(model, values),
        target=Scalar(spec=model.config.target, value=target, status="observed"),
    )


def batch_fit(rows, *, regularization: Fraction = Fraction(1)) -> tuple[Fraction, ...]:
    """Fit the identical prefix by independent rational Gaussian elimination."""
    size = 2
    matrix: list[list[Fraction]] = [
        [
            regularization * int(i == j) + sum((Fraction(x[i]) * Fraction(x[j]) for x, _ in rows), Fraction(0))
            for j in range(size)
        ]
        + [sum((Fraction(x[i]) * Fraction(y) for x, y in rows), Fraction(0))]
        for i in range(size)
    ]
    for pivot in range(size):
        divisor = matrix[pivot][pivot]
        matrix[pivot] = [value / divisor for value in matrix[pivot]]
        for row in range(size):
            if row != pivot:
                factor = matrix[row][pivot]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[pivot], strict=True)]
    return tuple(row[-1] for row in matrix)
