"""Map complete test rankings without running, updating or persisting them."""

from coleman.core.decisions import Decision, DecisionRequest
from coleman.problems.tcpci.contracts import TcpciMapping, TcpciProblem, _execution, _prefix


def ranking_request(problem: TcpciProblem) -> DecisionRequest:
    """Return a revalidated complete ranking request from the domain input.

    Parameters
    ----------
    problem : TcpciProblem
        Complete bounded versioned catalogue, semantic request and prefix rule.

    Returns
    -------
    DecisionRequest
        Original RankingSpace request, retaining all eligible inputs and clocks.

    Raises
    ------
    ValueError, TypeError
        Invalid fields, versions, catalogue, quantities or known future clocks.

    Notes
    -----
    Missing quotes remain representable here. No learner, source or native
    implementation is called and no state or I/O is performed.
    """
    return TcpciProblem.model_validate(problem).semantic.request


def map_ranking(problem: TcpciProblem, proposal: Decision) -> TcpciMapping:
    """Retain an exact-budget ordered prefix and invalidate changed sampling.

    Parameters
    ----------
    problem : TcpciProblem
        Complete original estimated modeled serial-work input and fixed rule.
    proposal : Decision
        Complete RankingAction bound to the exact original core request.

    Returns
    -------
    TcpciMapping
        Independently rechecked slate/cost/remaining budget with the full problem
        and original proposal. Empty affordable prefixes are valid slates.

    Raises
    ------
    ValueError, TypeError
        Invalid content, wrong request/action, missing duration quote or unknown
        quote availability for a timed opportunity. No partial result is emitted.

    Notes
    -----
    Uses exact10^-12 integer ticks, no Decimal-context rounding or tolerance.
    Stops at the first unaffordable test without skipping. Core FeedbackBatch
    binds only the resulting complete ExecutedDecision content. Different
    quotes, budgets or full definitions can produce identical core actions;
    callers associating feedback with costs, treatment or the full domain
    context must retain the mapping body and check its expected fingerprint.
    No physical executor, learner update, replay extension, storage
    acknowledgement or I/O occurs.
    """
    checked = TcpciProblem.model_validate(problem)
    original = Decision.model_validate(proposal)
    action, cost, remaining = _prefix(checked, original)
    return TcpciMapping(
        schema_version=1,
        problem=checked,
        executed=_execution(checked, original, action),
        estimated_cost=cost,
        remaining_budget=remaining,
    )
