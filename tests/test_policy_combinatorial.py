"""Unit tests for combinatorial bandit policies."""

import numpy as np
import polars as pl
import pytest

from coleman.agent import Agent
from coleman.policy import (
    CombinatorialThompsonPolicy,
    CombinatorialUCBPolicy,
    Policy,
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


def test_combinatorial_ucb_returns_full_order_with_subset_head():
    agent = _build_agent()
    policy = CombinatorialUCBPolicy(subset_size=2, c=1.0)

    ordered = policy.choose_all(agent)

    assert len(ordered) == agent.actions.height
    assert sorted(ordered) == sorted(agent.actions["Name"].to_list())
    assert ordered[0] == "A1"

    policy.credit_assignment(agent)
    assert np.isfinite(agent.actions["Q"].to_numpy()).all()


def test_combinatorial_thompson_updates_posteriors_and_q():
    agent = _build_agent()
    policy = CombinatorialThompsonPolicy(subset_size=3)

    first = policy.choose_all(agent)
    agent.last_prioritization = first
    agent.actions = agent.actions.with_columns((pl.col("ValueEstimates") + 0.5).alias("ValueEstimates"))

    policy.credit_assignment(agent)

    assert set(policy.alpha.keys()) == set(agent.actions["Name"].to_list())
    assert set(policy.beta.keys()) == set(agent.actions["Name"].to_list())
    assert np.isfinite(agent.actions["Q"].to_numpy()).all()


def test_combinatorial_validations_and_string_reprs():
    with pytest.raises(ValueError, match="subset_size must be positive"):
        CombinatorialUCBPolicy(subset_size=0)
    with pytest.raises(ValueError, match="Exploration parameter c must be positive"):
        CombinatorialUCBPolicy(subset_size=2, c=0)
    with pytest.raises(ValueError, match="subset_size must be positive"):
        CombinatorialThompsonPolicy(subset_size=0)

    assert "CombinatorialUCB" in str(CombinatorialUCBPolicy(subset_size=2, c=1.5))
    assert "Alpha=" in str(CombinatorialThompsonPolicy(subset_size=2, alpha_prior=2.0, beta_prior=3.0))


def test_public_namespace_exposes_combinatorial_policies():
    expected = {"CombinatorialUCBPolicy", "CombinatorialThompsonPolicy"}

    import coleman.policy as policy_module

    for name in expected:
        assert hasattr(policy_module, name)
