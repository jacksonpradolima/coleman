"""Pure decision-time TCPCI inputs and independently checked prefix mappings."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Annotated, Literal, Self, cast

from pydantic import Field, model_validator

from coleman.core.actions import RankingAction, RankingSpace, SlateAction, SlateSpace
from coleman.core.decisions import Decision, ExecutedDecision
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import ScalarStatus
from coleman.core.primitives import MAX_MEMBERS, Fingerprint
from coleman.core.quantities import Quantity, decimal_value, ticks
from coleman.core.sampling import SamplingRecord
from coleman.core.temporal import Timestamp
from coleman.objectives.bindings import QuantityMeaning, SemanticDefinition
from coleman.objectives.context import SemanticRequest
from coleman.problems.tcpci._base import TcpciContract

DURATION_MEANING = "tcpci-estimated-modeled-serial-duration-v1"
PREFIX_PROFILE = "exact-decimal-estimated-serial-prefix-stop-v1"
INPUT_MAXIMUM = Decimal("1000000000000")
SAMPLING_REASON = "tcpci-prefix-shape-change-invalidates-sampling"


class TestCase(TcpciContract):
    """Retain one versioned test and its separate decision-time duration quote.

    Attributes
    ----------
    member : Reference
        Exact test identity/version; labels do not establish identity.
    display_name : str
        Verbatim label, including empty/Unicode/whitespace names; at most4096
        characters. Different members may have equal labels.
    duration_quote : Quantity or None
        Nonnegative exact decimal estimated serial work, never physical elapsed
        execution. Its units/meaning are checked against the enclosing rule.
    duration_status : str
        Existing observed/unknown/unavailable/not_applicable states. Exactly
        observed has a quantity, including genuinely observed zero.
    quote_source, quote_source_fingerprint : Reference or None, str or None
        Independently nullable supplied provenance; no authentication is inferred.
    quote_available_at : datetime or None
        Supplied original availability; unknown remains null. Timed mapping
        requires known availability at/before the original information cutoff.

    Notes
    -----
    Required schema_version1 and shared16MiB/depth64 guards apply. No future
    verdict, native build object, source access or execution is included.
    """

    member: Reference
    display_name: Annotated[str, Field(strict=True, max_length=4096)]
    duration_quote: Quantity | None
    duration_status: ScalarStatus
    quote_source: Reference | None
    quote_source_fingerprint: Fingerprint | None
    quote_available_at: Timestamp | None

    @model_validator(mode="after")
    def _quote(self) -> Self:
        """Keep missing states explicit and cap a known planning input."""
        if (self.duration_status == "observed") != (self.duration_quote is not None):
            raise ValueError("Exactly observed duration status requires a quantity")
        if self.duration_quote is not None and self.duration_quote.value > INPUT_MAXIMUM:
            raise ValueError("TCPCI duration quote exceeds 10^12 declared units")
        return self


class BudgetPrefix(SemanticDefinition, TcpciContract):
    """Declare exact modeled serial-work prefix stopping, with no execution.

    Attributes
    ----------
    profile : {'exact-decimal-estimated-serial-prefix-stop-v1'}
        Inclusive exact budget, stop at first unaffordable member, no skipping,
        complete observed quotes required. Always invalidates changed-shape
        sampling, even for a deterministic original ranking.
    quantity : QuantityMeaning
        Resource role/interpretation, unit s, meaning
        tcpci-estimated-modeled-serial-duration-v1, minimum0 and explicit horizon
        or null. Existing reference/lineage fields bind the complete definition.

    Notes
    -----
    Fixed-scale arithmetic uses10^-12 ticks. It does not certify actual test
    duration, concurrency, wall time, paid charges or budget compliance in a
    physical executor. Unknown lineage is not evidence of causal eligibility.
    """

    profile: Literal["exact-decimal-estimated-serial-prefix-stop-v1"] = PREFIX_PROFILE
    quantity: QuantityMeaning

    @model_validator(mode="after")
    def _meaning(self) -> Self:
        """Keep estimated planning work distinct from native quality or cost."""
        quantity = self.quantity
        spec = quantity.spec
        if quantity.interpretation != "resource" or spec.role != "resource":
            raise ValueError("TCPCI prefix requires a resource quantity interpretation")
        if (spec.semantics.unit, spec.semantics.meaning) != ("s", DURATION_MEANING):
            raise ValueError("TCPCI prefix requires estimated modeled serial-duration seconds")
        if spec.minimum is None or Fraction.from_float(spec.minimum) != 0:
            raise ValueError("TCPCI duration schema requires explicit minimum zero")
        return self


class TcpciProblem(TcpciContract):
    """Bind a complete ranking opportunity to explicit planning quantities.

    Attributes
    ----------
    semantic : SemanticRequest
        Complete core RankingSpace request and objective definitions. Eligible
        features/predictions live in its existing member_inputs/signals.
    tests : tuple of TestCase
        At most4096 exact unique references covering the entire ranking universe,
        canonically ordered by ID. Labels are retained without normalization.
    budget : Quantity
        Inclusive nonnegative estimated-work budget, at most10^12 seconds, with
        exact rule units/meaning. No implicit unit conversion is performed.
    mapping : BudgetPrefix
        Full immutable profile. Its horizon equals every objective-term horizon;
        this first profile describes one declared measurement opportunity.

    Notes
    -----
    Missing quotes and unknown provenance remain representable. Mapping refuses
    missing quotes/unknown timed availability. Known future quote or definition
    declarations reject. Input eligibility remains a producer declaration.
    """

    semantic: SemanticRequest
    tests: Annotated[tuple[TestCase, ...], Field(max_length=MAX_MEMBERS)]
    budget: Quantity
    mapping: BudgetPrefix

    @model_validator(mode="after")
    def _problem(self) -> Self:
        """Validate exact catalogue, quantities and original decision clocks."""
        request = self.semantic.request
        if not isinstance(request.space, RankingSpace):
            raise ValueError("TCPCI prefix problem requires a complete ranking space")
        members = ordered_references(tuple(test.member for test in self.tests))
        if members != request.space.members:
            raise ValueError("TCPCI catalogue must cover the exact ranking universe and versions")
        meaning = self.mapping.quantity
        _quantity(self.budget, meaning)
        if self.budget.value > INPUT_MAXIMUM:
            raise ValueError("TCPCI planning budget exceeds 10^12 declared units")
        if any(term.quantity.horizon != meaning.horizon for term in self.semantic.context.objective.terms):
            raise ValueError("TCPCI planning and objective horizons must agree")
        for test in self.tests:
            if test.duration_quote is not None:
                _quantity(test.duration_quote, meaning)
        _known_clocks(self)
        object.__setattr__(self, "tests", tuple(sorted(self.tests, key=lambda test: test.member.id)))
        return self


def _quantity(value: Quantity, meaning: QuantityMeaning) -> None:
    """Check exact decimal semantics/bounds without projecting through float."""
    if value.semantics != meaning.spec.semantics:
        raise ValueError("TCPCI quantity units and meaning must exactly match the prefix rule")
    maximum = meaning.spec.maximum
    if maximum is not None and Fraction(value.value) > Fraction.from_float(maximum):
        raise ValueError("TCPCI quantity exceeds the declared resource schema maximum")


def _known_clocks(problem: TcpciProblem) -> None:
    """Reject known future declarations without fabricating unknown clocks."""
    timing = problem.semantic.request.timing
    if timing is None:
        return
    clocks = (problem.mapping.lineage.declared_at, *(test.quote_available_at for test in problem.tests))
    if any(instant is not None and instant > timing.information_cutoff for instant in clocks):
        raise ValueError("TCPCI quote or rule declaration follows the information cutoff")


def _prefix(problem: TcpciProblem, proposal: Decision) -> tuple[SlateAction, Quantity, Quantity]:
    """Derive an ordered affordable prefix using exact bounded integer ticks."""
    ranking = _original_ranking(problem, proposal)
    quotes = _quote_ticks(problem)
    budget = ticks(problem.budget.value)
    used = 0
    selected = []
    for member in ranking.order:
        amount = quotes[member]
        if used + amount > budget:
            break
        selected.append(member)
        used += amount
    space = SlateSpace(
        feasible_set=ranking.space.feasible_set,
        members=ranking.space.members,
        minimum_selected=0,
        maximum_selected=len(problem.tests),
    )
    action = SlateAction(schema_version=1, space=space, selected=tuple(selected))
    semantics = problem.budget.semantics
    return (
        action,
        Quantity(value=decimal_value(used), semantics=semantics),
        Quantity(value=decimal_value(budget - used), semantics=semantics),
    )


def _original_ranking(problem: TcpciProblem, proposal: Decision) -> RankingAction:
    """Admit only the complete original request and its ranking action."""
    if proposal.request.to_json() != problem.semantic.request.to_json() or not isinstance(
        proposal.action, RankingAction
    ):
        raise ValueError("TCPCI mapping requires the exact original request and complete ranking proposal")
    return proposal.action


def _quote_ticks(problem: TcpciProblem) -> dict[Reference, int]:
    """Admit complete timely quotes before converting any bounded quantities."""
    if any(test.duration_quote is None for test in problem.tests):
        raise ValueError("TCPCI prefix profile requires complete observed duration quotes")
    if problem.semantic.request.timing is not None and any(test.quote_available_at is None for test in problem.tests):
        raise ValueError("Timed TCPCI mapping requires known original quote availability")
    # The complete-input guard above admits every quantity before arithmetic.
    return {test.member: ticks(cast(Quantity, test.duration_quote).value) for test in problem.tests}


def _execution(problem: TcpciProblem, proposal: Decision, action: SlateAction) -> ExecutedDecision:
    """Invalidate ranking mass instead of guessing induced slate probabilities."""
    sampling = SamplingRecord(
        mechanism=proposal.sampling.mechanism,
        mechanism_ref=problem.mapping.reference,
        status="unavailable",
        action_fingerprint=action.fingerprint,
        probability=None,
        reason=SAMPLING_REASON,
    )
    return ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=action,
        mapping="filtered",
        sampling_disposition="invalidated",
        sampling=sampling,
    )


class TcpciMapping(TcpciContract):
    """Retain the full problem and rederive a claimed ordered prefix result.

    Attributes
    ----------
    problem : TcpciProblem
        Complete original input/profile, sufficient to recheck this mapping.
    executed : ExecutedDecision
        Original ranking proposal and mapped ordered slate, with unavailable
        invalidated sampling. This envelope does not execute a physical action.
    estimated_cost, remaining_budget : Quantity
        Exact modeled serial-work quantities in the original budget semantics.

    Notes
    -----
    Reading rechecks arithmetic and the complete core proposal/action bytes.
    FeedbackBatch binds ``executed.fingerprint``, not this full domain mapping.
    Quotes, budget or full semantic definitions can change while those core
    bytes stay equal. Association with planning costs, treatment or the full
    domain context therefore requires the caller to retain this mapping body
    and check its expected ``fingerprint`` at that boundary. A core feedback
    fingerprint alone cannot establish that stronger association.

    Neither fingerprint grants execution attribution, source honesty, a storage
    ACK or a replay profile. No authentication or automatic adoption is supplied.
    """

    problem: TcpciProblem
    executed: ExecutedDecision
    estimated_cost: Quantity
    remaining_budget: Quantity

    @model_validator(mode="after")
    def _result(self) -> Self:
        """Reject altered action, sampling or accounting without repairing it."""
        action, cost, remaining = _prefix(self.problem, self.executed.proposal)
        expected = _execution(self.problem, self.executed.proposal, action)
        if self.executed.to_json() != expected.to_json():
            raise ValueError("TCPCI executed decision does not match the declared prefix profile")
        if self.estimated_cost != cost or self.remaining_budget != remaining:
            raise ValueError("TCPCI estimated cost or remaining budget does not match exact prefix arithmetic")
        return self
