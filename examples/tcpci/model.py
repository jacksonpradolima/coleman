"""Public constructed CI inputs and explicit mathematical interpretations."""

from __future__ import annotations

from decimal import Decimal
from math import log

from coleman.core.actions import RankingSpace, Weight
from coleman.core.decisions import DecisionRequest
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.quantities import Quantity
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferenceConfig
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.context import ObjectiveDefinition, ObjectiveTerm, SemanticContext, SemanticRequest
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, InitialPrior
from coleman.objectives.transforms import IdentityLoss
from coleman.problems.tcpci import BudgetPrefix, TcpciProblem
from coleman.problems.tcpci import TestCase as TcpciTestCase

TABLE = (("A", "unit", "2"), ("B", "integration", "2"), ("C", "smoke", "1"))
EXPERT_ORDERS = (("fast-first", ("C", "A", "B")), ("risk-first", ("B", "A", "C")))
# Supplied only after each decision. This authored full source is not censored CI evidence.
PANELS = ((0, 1, 1), (1, 0, 1), (0, 1, 0))
DURATION = UnitSemantics(unit="s", meaning="tcpci-estimated-modeled-serial-duration-v1")
SOURCE_FINGERPRINT = fingerprint({"table": [list(row) for row in TABLE], "scope": "authored-planning-quotes-v1"})


def ref(name: str) -> Reference:
    """Return a supplied version-one identity, without registration or I/O.

    Parameters
    ----------
    name : str
        Logical identity within this constructed example.

    Returns
    -------
    Reference
        Exact supplied name and version.
    """
    return Reference(id=name, version="1")


def lineage() -> DefinitionLineage:
    """Retain unknown declaration clocks and sources without inventing them.

    Returns
    -------
    DefinitionLineage
        Explicit nullable provenance; this example is genuinely untimed.
    """
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


MEMBERS = tuple(ref(row[0]) for row in TABLE)
EXPERTS = tuple(ref(row[0]) for row in EXPERT_ORDERS)
HORIZON = ref("one-constructed-ci-build")
TASK = ref("constructed-ci-ranking")
LOSS = ScalarSpec(
    dimension=ref("missed-failure-fraction"),
    role="loss",
    semantics=UnitSemantics(unit="1", meaning="missed-synthetic-failures-divided-by-three-tests-v1"),
    minimum=0.0,
    maximum=1.0,
)
VERDICT = ScalarSpec(
    dimension=ref("synthetic-failure-verdict"),
    role="objective",
    semantics=UnitSemantics(unit="1", meaning="authored-binary-failure-verdict-v1"),
    minimum=0.0,
    maximum=1.0,
)
LOSS_MEANING = QuantityMeaning(spec=LOSS, interpretation="normalized_loss", horizon=HORIZON, risk_definition=None)
IDENTITY = IdentityLoss(
    schema_version=1, reference=ref("already-normalized-missed-failure-loss"), lineage=lineage(), quantity=LOSS_MEANING
)
OBJECTIVE = ObjectiveDefinition(
    schema_version=1,
    reference=ref("minimize-missed-synthetic-failures"),
    lineage=lineage(),
    terms=(ObjectiveTerm(quantity=LOSS_MEANING, direction="minimize", transform=IDENTITY.binding),),
)
CONTEXT = SemanticContext(
    schema_version=1,
    reference=ref("constructed-ci-objective-context"),
    lineage=lineage(),
    objective=OBJECTIVE,
    transforms=(IDENTITY,),
)


def problem(round_index: int, *, budget: str = "3") -> TcpciProblem:
    """Build a decision-time table containing quotes but no outcome labels.

    Parameters
    ----------
    round_index : int
        Nonnegative logical opportunity ordinal, unrelated to independent units.
    budget : str, optional
        Exact nonnegative decimal modeled serial-work seconds, default ``3``.

    Returns
    -------
    TcpciProblem
        Complete untimed ranking input with supplied quote identity and unknown
        availability/declaration clocks. Quotes are not physical measurements.

    Raises
    ------
    ValueError
        Invalid round, budget precision/range or incompatible typed content.

    Notes
    -----
    No clock, source outcome, native runner, workspace or learner is accessed.
    """
    duration = ScalarSpec(dimension=ref("planning-duration"), role="resource", semantics=DURATION, minimum=0.0)
    mapping = BudgetPrefix(
        schema_version=1,
        reference=ref("exact-estimated-prefix"),
        lineage=lineage(),
        quantity=QuantityMeaning(spec=duration, interpretation="resource", horizon=HORIZON, risk_definition=None),
    )
    request = DecisionRequest(
        schema_version=1,
        task=TASK,
        objective=OBJECTIVE.reference,
        decision_id=f"build-{round_index}",
        round=round_index,
        space=RankingSpace(members=MEMBERS, feasible_set=ref("three-authored-tests")),
    )
    return TcpciProblem(
        schema_version=1,
        semantic=SemanticRequest(schema_version=1, request=request, context=CONTEXT),
        mapping=mapping,
        budget=Quantity(value=Decimal(budget), semantics=DURATION),
        tests=tuple(
            TcpciTestCase(
                schema_version=1,
                member=ref(name),
                display_name=label,
                duration_quote=Quantity(value=Decimal(seconds), semantics=DURATION),
                duration_status="observed",
                quote_source=ref("authored-planning-table"),
                quote_source_fingerprint=SOURCE_FINGERPRINT,
                quote_available_at=None,
            )
            for name, label, seconds in TABLE
        ),
    )


def configuration() -> ReferenceConfig:
    """Declare actual constant Hedge on two complete synthetic expert losses.

    Returns
    -------
    ReferenceConfig
        Equal positive prior, eta=log(2), explicit bounded loss and transform.
        Coefficients are neither priorities nor calibrated probabilities.
    """
    prior = InitialPrior(
        schema_version=1,
        reference=ref("equal-two-expert-prior"),
        lineage=lineage(),
        weights=tuple(Weight(member=member, value=0.5) for member in EXPERTS),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    return ReferenceConfig(
        schema_version=1,
        reference=ref("constructed-ci-constant-hedge"),
        lineage=lineage(),
        profile="hedge-constant",
        prior=prior,
        loss_spec=LOSS,
        transform=IDENTITY.binding,
        eta=log(2),
        alpha=None,
    )
