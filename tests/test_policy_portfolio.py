"""Unit tests for portfolio meta-policy."""

import polars as pl
import pytest

from coleman.agent import Agent
from coleman.policy import (
    Policy,
    PortfolioUCBPolicy,
)


def _build_agent() -> Agent:
    agent = Agent(policy=Policy())
    agent.actions = pl.DataFrame(
        {
            "Name": ["A1", "A2", "A3", "A4"],
            "ActionAttempts": [0.0, 2.0, 3.0, 1.0],
            "ValueEstimates": [0.0, 0.8, 1.2, 0.3],
            "Q": [0.0, 0.4, 0.4, 0.3],
        }
    )
    return agent


class _FixedOrderPolicy(Policy):
    def __init__(self, ordering):
        self.ordering = ordering

    def choose_all(self, agent):
        return list(self.ordering)

    def credit_assignment(self, agent):
        return None


def test_portfolio_ucb_explores_policies_and_delegates_choose_all():
    agent = _build_agent()
    p1 = _FixedOrderPolicy(["A1", "A2", "A3", "A4"])
    p2 = _FixedOrderPolicy(["A4", "A3", "A2", "A1"])
    portfolio = PortfolioUCBPolicy(policies=[p1, p2], c=1.0, window_size=3)

    first = portfolio.choose_all(agent)
    portfolio.credit_assignment(agent)

    # Second update covers non-initial observed reward branch.
    agent.actions = agent.actions.with_columns((pl.col("ValueEstimates") + 1.0).alias("ValueEstimates"))
    second = portfolio.choose_all(agent)
    portfolio.credit_assignment(agent)

    assert first == ["A1", "A2", "A3", "A4"]
    assert second == ["A4", "A3", "A2", "A1"]
    assert portfolio.active_policy in [p1, p2]


def test_portfolio_validations_and_active_policy_score_paths():
    p1 = _FixedOrderPolicy(["A1", "A2", "A3", "A4"])

    with pytest.raises(ValueError, match="at least one candidate policy"):
        PortfolioUCBPolicy(policies=[])
    with pytest.raises(ValueError, match="Exploration parameter c must be positive"):
        PortfolioUCBPolicy(policies=[p1], c=0)
    with pytest.raises(ValueError, match="window_size must be positive"):
        PortfolioUCBPolicy(policies=[p1], window_size=0)

    portfolio = PortfolioUCBPolicy(policies=[p1], c=1.0, window_size=2)
    assert "PortfolioUCB" in str(portfolio)
    assert portfolio.active_policy is p1
    # Cold-start branch (uses <= 0) must be optimistic.
    assert portfolio._policy_score(0, total_uses=0) == float("inf")


def test_public_namespace_exposes_portfolio_policy():
    import coleman.policy as policy_module

    assert hasattr(policy_module, "PortfolioUCBPolicy")
