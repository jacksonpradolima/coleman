"""Public authored laws for a small funded synthetic portfolio experiment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from fractions import Fraction

from coleman.core.actions import Weight
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferenceConfig, ReferenceProfile
from coleman.learning.control_contracts import ControlledConfig
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage, QuantityMeaning
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExpertBound,
    ExpertBounds,
    IntervalCadence,
    L1MovementPenalty,
    MixtureControls,
    MovementAllowance,
    MovementLimit,
    ProjectionOutput,
    RoundCadence,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, DeclaredTargetShare, InitialPrior
from coleman.objectives.volatility import RollingScaleSpec, SignedVolatilitySpec
from coleman.optimization.continuous_contracts import (
    CoefficientInterval,
    ContinuousProblem,
    ContinuousSet,
    ContinuousVariable,
    QuadraticObjective,
)
from coleman.prediction import RidgeConfig
from coleman.problems.portfolio import AssetWeight, Book, Exact, Fees, Position, PricePanel, PriceQuote, Target

START = datetime(2026, 1, 1, tzinfo=UTC)


def ref(name: str) -> Reference:
    """Declare one explicit public synthetic logical identity, version1.

    Parameters
    ----------
    name : str
        Bounded identifier, not a source path or import instruction.

    Returns
    -------
    Reference
        Immutable exact identity with no discovery or registration.
    """
    return Reference(id="synthetic-portfolio-" + name, version="1")


ASSETS = (ref("asset-a"), ref("asset-b"))
RECIPES = tuple(ref("recipe-" + name) for name in ("a", "b", "trend"))
LINEAGE = DefinitionLineage(
    source=ref("authored-definition"),
    source_fingerprint=fingerprint({"authored": "funded-long-only-public-v1"}),
    declared_at=START,
    supersedes=None,
)
HORIZON = ref("one-calendar-period")
RAW = QuantityMeaning(
    spec=ScalarSpec(
        dimension=ref("signed-net-return"),
        role="objective",
        semantics=UnitSemantics(unit="1", meaning="signed-net-wealth-change-over-original-wealth-v1"),
    ),
    interpretation="raw_outcome",
    horizon=HORIZON,
)
LOSS = QuantityMeaning(
    spec=ScalarSpec(
        dimension=ref("bounded-surrogate-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
        semantics=UnitSemantics(unit="1", meaning="past-sample-volatility-clipped-signed-loss-v1"),
    ),
    interpretation="normalized_loss",
    horizon=HORIZON,
)
SCALE = RollingScaleSpec(
    schema_version=1,
    reference=ref("past-only-sample-sd"),
    lineage=LINEAGE,
    quantity=RAW,
    scale_spec=ScalarSpec(
        dimension=ref("sample-sd"),
        role="diagnostic",
        minimum=0.0,
        semantics=UnitSemantics(unit="1", meaning="past-net-return-sample-sd-ddof-one-v1"),
    ),
    window=2,
    scope="per_member",
    maximum_age_seconds=None,
)
TRANSFORM = SignedVolatilitySpec(
    schema_version=1,
    reference=ref("signed-volatility-transform"),
    lineage=LINEAGE,
    input=RAW,
    output=LOSS,
    scale_definition=SCALE.binding,
    c=2.0,
)

# Fully authored exogenous marks. Regimes and a reversal are illustrative rather
# than sampled market histories. Development ends at row6; protected rows follow.
PRICE_ROWS = (
    (100, 100),
    (102, 99),
    (101, 103),
    (105, 101),
    (108, 100),
    (110, 98),
    (111, 97),
    (106, 102),
    (101, 107),
    (97, 110),
    (98, 109),
    (102, 105),
    (100, 108),
    (104, 103),
    (103, 106),
    (105, 104),
)
CONTROLLED_MODES = (
    "plain",
    "anchored",
    "bounded",
    "movement",
    "cadence",
    "penalized",
    "anchored-bounded-interval-l1",
    "composed-online",
)
MODES = (
    "equal",
    "buy-and-hold",
    "periodic-rebalance",
    "hedge",
    "fixed-share",
    *(name for name in CONTROLLED_MODES if name != "composed-online"),
    "constrained-fixed",
    "composed-online",
)


def prices(index: int, rows: tuple[tuple[int, int], ...] = PRICE_ROWS) -> PricePanel:
    """Expose one authored exogenous mark without including future rows.

    Parameters
    ----------
    index : int
        Selected row, from0 through15 for the default bounded tape.
    rows : tuple of tuple of int, optional
        Explicit authored prices; the current row alone binds this source view.

    Returns
    -------
    PricePanel
        Complete same-instant observed credit/unit quotes. Its content identity
        does not fingerprint future or corrected later rows.
    """
    values = rows[index]
    return price_panel(index, values)


def price_panel(index: int, values: tuple[int, int]) -> PricePanel:
    """Bind one externally supplied authored current row, without a future table.

    Parameters
    ----------
    index : int
        Original source position and UTC calendar-day coordinate.
    values : tuple of int
        Two positive exact synthetic-credit/unit marks, in asset order.

    Returns
    -------
    PricePanel
        Canonical current-row source identity, timestamps and exact quote units.
        This pure call receives no past or future rows.
    """
    at = START + timedelta(days=index)
    return PricePanel(
        schema_version=1,
        source=ref("price-row-" + str(index)),
        source_fingerprint=fingerprint({"row": index, "prices": list(values), "unit": "synthetic-credit/asset-unit"}),
        at=at,
        quotes=tuple(
            PriceQuote(asset=asset, value=Exact.of(value), status="observed", event_at=at, available_at=at)
            for asset, value in zip(ASSETS, values, strict=True)
        ),
    )


def initial_book(name: str) -> Book:
    """Create an independently funded100-credit zero-holdings book.

    Parameters
    ----------
    name : str
        Candidate or virtual-recipe identity, each with its own ledger.

    Returns
    -------
    Book
        Virgin exact long-only path; no funding or effects occur externally.
    """
    return Book(
        schema_version=1,
        reference=ref("book-" + name),
        cash=Exact.of(100),
        positions=tuple(Position(asset=asset, units=Exact.of(0)) for asset in ASSETS),
    )


def fees(index: int, *, rate: Fraction = Fraction(1, 1000)) -> Fees:
    """Declare net-trade proportional and independent recurring period charges.

    Parameters
    ----------
    index : int
        Exact period, independent from delivery retries or solver updates.
    rate : Fraction, optional
        Explicit credit/absolute-traded-credit rate, default0.001.

    Returns
    -------
    Fees
        Zero recurring initialization, then0.001credit per period.
    """
    return Fees(
        schema_version=1,
        reference=ref("declared-fees"),
        period=ref("period-" + str(index)),
        declared_at=START,
        proportional=Exact.of(rate),
        recurring=Exact.of(0 if index == 0 else Fraction(1, 1000)),
        status="observed",
    )


def target(name: str, weights: tuple[Fraction, Fraction]) -> Target:
    """Map an example recipe into98% exposure with an explicit2% cash reserve.

    Parameters
    ----------
    name : str
        Stable producing target identity.
    weights : tuple of Fraction
        Nonnegative recipe fractions with total at most1, in exact asset order.
        This example adapter scales each by49/50 before funding; it is not
        a repair of the learner coefficients or the solver feasible set.

    Returns
    -------
    Target
        Validated funded-long-only target with at least2% cash; shorts and
        leverage reject. The reserve pays later recurring fees during holds.
    """
    return Target(
        schema_version=1,
        reference=ref(name),
        weights=tuple(
            AssetWeight(asset=asset, value=Exact.of(weight * Fraction(49, 50)))
            for asset, weight in zip(ASSETS, weights, strict=True)
        ),
    )


def reference_config(profile: ReferenceProfile) -> ReferenceConfig:
    """Declare bounded full-information Hedge or actual excluding-origin share.

    Parameters
    ----------
    profile : str
        One shipped reference profile, not a controlled-law alias.

    Returns
    -------
    ReferenceConfig
        Exact prior, loss transform and development-declared constant rate.
    """
    return ReferenceConfig(
        schema_version=1,
        reference=ref(profile),
        lineage=LINEAGE,
        profile=profile,
        prior=InitialPrior(
            schema_version=1,
            reference=ref("initial-prior"),
            lineage=LINEAGE,
            weights=tuple(
                Weight(member=member, value=value) for member, value in zip(RECIPES, (0.5, 0.25, 0.25), strict=True)
            ),
            profile="binary64-simplex-v1",
            semantics=COEFFICIENT_SEMANTICS,
        ),
        loss_spec=LOSS.spec,
        transform=TRANSFORM.binding,
        eta=0.5,
        alpha=0.1 if profile == "fixed-share-excluding-origin" else None,
        anytime_multiplier=None,
    )


def controlled_config(mode: str, *, strength: float = 0.05) -> ControlledConfig:
    """Select versioned control ablations with one fixed operation order.

    Parameters
    ----------
    mode : str
        Plain, anchored, bounded, movement, cadence, penalized or the complete
        anchored-bounded-interval-L1 combination. Composed-online uses the latter.
    strength : float, optional
        Nonnegative L1 coefficient-unit strength selected on development only.

    Returns
    -------
    ControlledConfig
        Actual shipped exponential/share/joint-output law. Interval modes use
        separate explicit movement periods with exact accumulated usage.
    """
    source = reference_config("hedge-constant")
    prior = source.prior
    full = mode in {"anchored-bounded-interval-l1", "composed-online"}
    anchored = mode == "anchored" or full
    bounded = mode in {"bounded", "movement", "cadence", "penalized"} or full
    movement = mode in {"movement", "cadence", "penalized"} or full
    penalty = mode == "penalized" or full
    origin = CoefficientOrigin(
        schema_version=1,
        reference=ref("initial-origin"),
        lineage=LINEAGE,
        weights=prior.weights,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    anchor = (
        DeclaredTargetShare(
            schema_version=1,
            reference=ref("approved-target"),
            lineage=LINEAGE,
            weights=tuple(
                Weight(member=member, value=value) for member, value in zip(RECIPES, (0.25, 0.25, 0.5), strict=True)
            ),
            profile="binary64-simplex-v1",
            semantics=COEFFICIENT_SEMANTICS,
            alpha=0.1,
        )
        if anchored
        else None
    )
    boxes = ExpertBounds(
        schema_version=1,
        reference=ref("expert-boxes"),
        lineage=LINEAGE,
        semantics=COEFFICIENT_SEMANTICS,
        bounds=tuple(
            ExpertBound(member=member, lower=0.1 if bounded else 0.0, upper=0.8 if bounded else 1.0)
            for member in RECIPES
        ),
    )
    allowance = (
        MovementAllowance(
            schema_version=1,
            reference=ref("movement-allowance"),
            lineage=LINEAGE,
            semantics=COEFFICIENT_SEMANTICS,
            scope="per_interval" if full else "per_release",
            measure="accumulated_absolute",
            limits=tuple(MovementLimit(member=member, maximum=0.125) for member in RECIPES),
        )
        if movement
        else None
    )
    output = (
        L1MovementPenalty(
            schema_version=1,
            reference=ref("l1-output"),
            lineage=LINEAGE,
            strength=strength,
            origin=origin,
            semantics=COEFFICIENT_SEMANTICS,
        )
        if penalty
        else ProjectionOutput()
    )
    controls = MixtureControls(
        schema_version=1,
        reference=ref("controls-" + mode),
        lineage=LINEAGE,
        initial_prior=prior,
        target=anchor,
        bounds=boxes,
        movement=allowance,
        cadence=IntervalCadence(calendar=ref("two-day-release-calendar"))
        if full
        else RoundCadence(panels=2 if mode in {"cadence", "penalized"} else 1),
        output=output,
    )
    return ControlledConfig(
        schema_version=1,
        reference=ref("law-" + mode),
        lineage=LINEAGE,
        profile="controlled-hedge-l1-movement" if penalty else "controlled-hedge-projection",
        controls=controls,
        loss_spec=LOSS.spec,
        transform=TRANSFORM.binding,
        eta=0.5,
        period_clock=ref("daily-movement-budget") if full else None,
    )


def ridge_config() -> RidgeConfig:
    """Declare an actual append-only ridge on lagged dimensionless features.

    Returns
    -------
    RidgeConfig
        Penalized intercept and lagged signed spread, regularization1. Labels
        are next-period return spreads and enter only after that period ends.
    """
    return RidgeConfig(
        schema_version=1,
        reference=ref("ridge"),
        lineage=LINEAGE,
        features=tuple(
            ScalarSpec(
                dimension=ref("feature-" + name),
                role="feature",
                minimum=-4.0,
                maximum=4.0,
                semantics=UnitSemantics(unit="1", meaning="explicit-lagged-synthetic-feature-v1"),
            )
            for name in ("constant", "lagged-spread")
        ),
        target=ScalarSpec(
            dimension=ref("return-spread-label"),
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="next-period-a-minus-b-signed-return-v1"),
        ),
        horizon=HORIZON,
        preprocessing=DefinitionBinding(
            reference=ref("causal-lag-preprocessing"),
            definition_fingerprint=fingerprint({"constant": 1, "lag": 1, "future": False}),
        ),
        regularization=1.0,
    )


def continuous_problem(mixture: tuple[Fraction, Fraction], forecast: float) -> ContinuousProblem:
    """Compose expert target and pre-label forecast in the real small QP solver.

    Parameters
    ----------
    mixture : tuple of Fraction
        Netted nonnegative recipe fractions, before constrained allocation.
    forecast : float
        Actual supplied or learned signed return-spread prediction. It is not
        a calibrated expected profit, risk estimate or executed cost.

    Returns
    -------
    ContinuousProblem
        Strictly convex squared-distance objective with declared forecast tilt,
        simplex and0.1..0.9asset boxes. Its penalty is never a paid charge.
    """

    def interval(low: float, high: float) -> CoefficientInterval:
        """Declare inclusive coefficient bounds without a domain interpretation."""
        return CoefficientInterval(schema_version=1, lower=low, upper=high)

    feasible = ContinuousSet(
        schema_version=1,
        reference=ref("long-only-math-set"),
        lineage=LINEAGE,
        profile="simplex-v1",
        variables=tuple(ContinuousVariable(schema_version=1, member=asset, lower=0.1, upper=0.9) for asset in ASSETS),
        net=interval(1.0, 1.0),
        cash=interval(0.0, 0.0),
        gross_limit=1.0,
        leverage_limit=1.0,
    )
    objective = QuadraticObjective(
        schema_version=1,
        reference=ref("forecast-target-distance"),
        lineage=LINEAGE,
        matrix=((2.0, 0.0), (0.0, 2.0)),
        linear=(-2 * float(mixture[0]) - forecast, -2 * float(mixture[1]) + forecast),
        score=ScalarSpec(
            dimension=ref("allocation-score"),
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="squared-target-distance-minus-declared-spread-tilt-v1"),
        ),
    )
    return ContinuousProblem(
        schema_version=1,
        feasible_set=feasible,
        objective=objective,
        comparator=DefinitionBinding(
            reference=ref("fixed-equal-funded-book"), definition_fingerprint=fingerprint({"target": ["1/2", "1/2"]})
        ),
        transition=DefinitionBinding(
            reference=ref("funded-exact-transition"),
            definition_fingerprint=fingerprint({"law": "W+c*sum(abs(wW-h))+F=V"}),
        ),
    )
