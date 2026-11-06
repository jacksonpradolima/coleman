"""Independent tiny witnesses of native ranking adaptations, never algorithm corrections."""

import copy
import math

import numpy as np
import polars as pl
import pytest

from coleman.agent import ContextualAgent, RewardAgent, RewardSlidingWindowAgent, SlidingWindowContextualAgent
from coleman.policy import (
    CombinatorialThompsonPolicy,
    CombinatorialUCBPolicy,
    DuelingUCBPolicy,
    EXP3IXPolicy,
    EXP3Policy,
    FRRMABPolicy,
    LinUCBPolicy,
    PairwiseThompsonRankingPolicy,
    PortfolioUCBPolicy,
    PursuitPolicy,
    RandomPolicy,
    SlMABPolicy,
    SoftmaxPolicy,
    SWContextualEpsilonGreedyPolicy,
    SWLinTSPolicy,
    SWLinUCBPolicy,
    ThompsonSamplingPolicy,
)
from coleman.policy import base as policy_base
from coleman.reward import RNFailReward
from coleman.runner import build_agents_from_config, normalize_and_validate_agent_build
from tests.native_semantics.support import cycle, prepare, rng_state


@pytest.fixture(autouse=True)
def owned_policy_rng(monkeypatch):
    monkeypatch.setattr(policy_base, "_rng", np.random.default_rng(2026))


@pytest.mark.parametrize("implicit", [False, True], ids=("EXP3", "EXP3IX"))
def test_adversarial_internal_denominators_are_not_actual_sampled_first_position_mass(implicit):
    policy = EXP3IXPolicy(eta=0.1, gamma=0.01) if implicit else EXP3Policy(gamma=0.2)
    agent = RewardAgent(policy, RNFailReward(), seed=17)
    cycle(agent, {"A": 0, "B": 0})
    before = copy.deepcopy(policy_base._rng.bit_generator.state)
    assert [agent.choose() for _ in range(5)] == [["A", "B"]] * 5
    assert policy_base._rng.bit_generator.state == before
    assert policy._probs(["A", "B"]) == {"A": 0.5, "B": 0.5}
    actual = cycle(agent, {"A": 1, "B": 1})
    assert actual["credits"] == [1.0, 1.0]
    exponent = 0.1 / (0.5 + 0.01) if implicit else 0.2 * (1 / 0.5) / 2
    expected = math.exp(exponent)
    assert math.isclose(policy.weights["A"], expected, rel_tol=1e-14)
    assert math.isclose(policy.weights["B"], expected, rel_tol=1e-14)
    # Separate sampled-action reference: u=.25 chooses A from (.5,.5),
    # estimates (2,0), and leaves unchosen B unchanged. It is not shipped here.
    sampled_b_weight = math.exp(0.2 * 0 / 2)
    assert sampled_b_weight == 1.0
    assert policy.weights["B"] > sampled_b_weight


@pytest.mark.parametrize("policy_type", [CombinatorialUCBPolicy, CombinatorialThompsonPolicy])
def test_combinatorial_subset_size_is_a_split_and_rejoin_of_the_same_complete_order(policy_type, monkeypatch):
    generator = copy.deepcopy(policy_base._rng.bit_generator.state)
    orders = []
    states = []
    for size in (1, 2, 3, 7):
        agent = RewardAgent(policy_type(subset_size=size), RNFailReward(), seed=17)
        agent.update_actions(["A", "B", "C"])
        # Explicit synthetic post-warmup state, not an alleged training trajectory.
        agent.actions = pl.DataFrame(
            {"Name": ["A", "B", "C"], "ActionAttempts": [1.0] * 3, "ValueEstimates": [0.9, 0.5, 0.1], "Q": [0.0] * 3}
        )
        agent.t = 1
        monkeypatch.setattr(policy_base._rng.bit_generator, "state", copy.deepcopy(generator))
        orders.append(agent.choose())
        states.append(copy.deepcopy(policy_base._rng.bit_generator.state))
    assert all(order == orders[0] for order in orders)
    assert sorted(orders[0]) == ["A", "B", "C"]
    assert all(state == states[0] for state in states)
    if policy_type is CombinatorialUCBPolicy:
        # Equal exposure gives a common sqrt(log4) bonus, so means fix the order.
        independent = {name: mean + math.sqrt(math.log(4)) for name, mean in (("A", 0.9), ("B", 0.5), ("C", 0.1))}
        assert orders[0] == sorted(independent, key=independent.__getitem__, reverse=True)


@pytest.mark.parametrize("policy_type", [SoftmaxPolicy, PursuitPolicy])
def test_internal_preferences_are_sorted_without_a_sampling_draw(policy_type):
    policy = policy_type()
    agent = RewardAgent(policy, RNFailReward(), seed=17)
    agent.update_actions(["A", "B"])
    agent.t = 1
    before = copy.deepcopy(policy_base._rng.bit_generator.state)
    own = rng_state(agent)
    expected = ["B", "A"] if policy_type is SoftmaxPolicy else ["A", "B"]
    assert [agent.choose() for _ in range(5)] == [expected] * 5
    assert policy_base._rng.bit_generator.state == before
    assert rng_state(agent) == own


@pytest.mark.parametrize("policy_type", [DuelingUCBPolicy, PairwiseThompsonRankingPolicy])
def test_pair_updates_follow_their_own_order_despite_opposite_actual_failure_feedback(policy_type, monkeypatch):
    snapshots = []
    generator = copy.deepcopy(policy_base._rng.bit_generator.state)
    for verdicts in ({"A": 1, "B": 0}, {"A": 0, "B": 1}):
        monkeypatch.setattr(policy_base._rng.bit_generator, "state", copy.deepcopy(generator))
        policy = policy_type()
        agent = RewardAgent(policy, RNFailReward(), seed=17)
        # Observe both a real warmup and the next actual policy-generated order.
        rounds = [cycle(agent, verdicts), cycle(agent, verdicts)]
        if isinstance(policy, DuelingUCBPolicy):
            pair = {"duels": dict(policy._duels), "wins_A": dict(policy._wins_first)}
        else:
            pair = {"alpha": dict(policy._alpha), "beta": dict(policy._beta)}
        snapshots.append((rounds, pair))
    assert [row["ranking"] for row in snapshots[0][0]] == [row["ranking"] for row in snapshots[1][0]]
    assert snapshots[0][0][0]["credits"] != snapshots[1][0][0]["credits"]
    assert snapshots[0][1] == snapshots[1][1]
    wins_a = sum(row["ranking"].index("A") < row["ranking"].index("B") for row in snapshots[0][0])
    if policy_type is DuelingUCBPolicy:
        assert snapshots[0][1]["duels"] == {("A", "B"): 2.0}
        assert snapshots[0][1]["wins_A"].get(("A", "B"), 0.0) == wins_a
    else:
        assert snapshots[0][1]["alpha"] == {("A", "B"): 1.0 + wins_a}
        assert snapshots[0][1]["beta"] == {("A", "B"): 1.0 + 2 - wins_a}


def test_portfolio_active_child_mutates_the_shared_table_and_consumes_preceding_child_credit():
    first = EXP3Policy(gamma=0.2)
    second = ThompsonSamplingPolicy(alpha_prior=1.0, beta_prior=1.0)
    policy = PortfolioUCBPolicy([first, second], c=1.0, window_size=3)
    agent = RewardAgent(policy, RNFailReward(), seed=17)
    cycle(agent, {"A": 1, "B": 0})
    before = agent.actions.to_dicts()
    current = cycle(agent, {"A": 0, "B": 0})
    assert policy.active_policy is second
    assert policy._policy_uses == [1, 1]
    assert current["credits"] == [0.0, 0.0]
    assert second.alpha == {"A": 2.0, "B": 1.0}
    assert second.beta == {"A": 1.0, "B": 2.0}
    assert [row["ValueEstimates"] for row in before] == [row["ValueEstimates"] for row in current["actions"]]
    assert [row["Q"] for row in before] != [row["Q"] for row in current["actions"]]


def test_contextual_targets_accumulate_prior_credits_and_unexecuted_members_receive_model_updates():
    policy = LinUCBPolicy(alpha=0.0)
    agent = ContextualAgent(policy, RNFailReward(), seed=17)
    first = cycle(agent, {"A": 1, "B": 1}, budget=1.0)
    second = cycle(agent, {"A": 0, "B": 1}, budget=1.0)
    assert first["scheduled"] == second["scheduled"] == ["A"]
    assert first["unscheduled"] == second["unscheduled"] == ["B"]
    assert first["credits"] == [1.0, 0.0]
    assert second["credits"] == [0.0, 0.0]
    # x=1 twice gives design3; sum of cumulative targets(1,1) gives b2.
    assert policy.context["A"]["A"].tolist() == [[3.0]]
    assert policy.context["b"]["A"].tolist() == [[2.0]]
    assert policy.context["A"]["B"].tolist() == [[3.0]]
    assert policy.context["b"]["B"].tolist() == [[0.0]]
    per_round_target = sum(round_["credits"][0] for round_ in (first, second))
    assert per_round_target == 1.0
    assert policy.context["b"]["A"][0, 0] != per_round_target


def test_unexecuted_member_zero_credit_is_not_an_observed_pass_but_updates_legacy_beta_and_exposure():
    policy = ThompsonSamplingPolicy()
    agent = RewardAgent(policy, RNFailReward(), seed=17)
    actual = cycle(agent, {"A": 1, "B": 1}, budget=1.0)
    hidden = actual["unscheduled"][0]
    assert hidden not in actual["scheduled"]
    assert policy.alpha[hidden] == 1.0
    assert policy.beta[hidden] == 2.0
    row = next(row for row in actual["actions"] if row["Name"] == hidden)
    assert row["ValueEstimates"] == 0.0
    assert math.isclose(row["ActionAttempts"], 1e-12, rel_tol=0, abs_tol=1e-24)


@pytest.mark.parametrize("name", ["FRRMAB", "SWLinUCB"])
def test_validated_multi_window_factory_shares_mutable_policy_state_between_real_agents(name):
    params = {"c": 0.0} if name == "FRRMAB" else {"alpha": 0.0}
    config = {name.lower(): {"window_sizes": [2, 3], "rnfail": params}}
    normalized, issues = normalize_and_validate_agent_build(
        algorithm_configs=config, policy_names=[name], rewards_names=["RNFail"], strict=True
    )
    assert issues == []
    first, second = build_agents_from_config(normalized, [name], ["RNFail"], seed=17)
    assert isinstance(first, RewardSlidingWindowAgent | SlidingWindowContextualAgent)
    assert isinstance(second, RewardSlidingWindowAgent | SlidingWindowContextualAgent)
    assert first.policy is second.policy
    cycle(first, {"A": 1, "B": 0})
    own_table, own_history = first.actions.clone(), first.history.clone()
    assert first.choose() == ["A", "B"]
    # Distinct panels intentionally expose direct-object coupling. This is not
    # an asserted Environment same-build sequence; the installed flow is separate.
    cycle(second, {"A": 0, "B": 1})
    assert first.actions.equals(own_table)
    assert first.history.equals(own_history)
    assert first.choose() == ["B", "A"]


def test_sliding_contextual_history_expires_but_inherited_matrices_do_not_refit_that_window():
    policy = SWLinUCBPolicy(alpha=0.0)
    agent = SlidingWindowContextualAgent(policy, RNFailReward(), 2, seed=17)
    for index in range(4):
        cycle(agent, {"A": int(index == 0), "B": 0})
    assert set(agent.history["T"].to_list()) == {3, 4}
    assert policy.context["A"]["A"].tolist() == [[5.0]]
    assert policy.context["b"]["A"].tolist() == [[1.0]]
    # The declared last-two-row reference would instead have design3,b0.
    retained = agent.history.filter(pl.col("Name") == "A")
    assert 1 + retained.height == 3
    assert retained["ValueEstimates"].sum() == 0.0
    assert policy.context["A"]["A"][0, 0] > 1 + retained.height


def test_reset_retains_policy_cache_and_rng_and_retired_actions_can_keep_policy_history():
    policy = ThompsonSamplingPolicy()
    agent = RewardAgent(policy, RNFailReward(), seed=17)
    cycle(agent, {"A": 1, "B": 0})
    own = rng_state(agent)
    expected_policy_state = copy.deepcopy((policy.alpha, policy.beta, policy._last_value_estimate))
    agent.reset()
    assert agent.t == 0
    assert rng_state(agent) == own
    actual_policy_state = (policy.alpha, policy.beta, policy._last_value_estimate)
    assert actual_policy_state == expected_policy_state
    cycle(agent, {"A": 0, "B": 0})
    projected_delta = 0.5 * (math.tanh(-1.0) + 1)
    assert math.isclose(policy.alpha["A"], 2 + projected_delta, rel_tol=1e-14)
    frr = RewardSlidingWindowAgent(FRRMABPolicy(c=0.0), RNFailReward(), 2, seed=17)
    cycle(frr, {"A": 1, "B": 0})
    prepare(frr, {"B": 0})
    assert frr.actions["Name"].to_list() == ["B"]
    assert frr.choose() == ["A", "B"]
    history = frr.history.clone()
    frr.reset()
    assert frr.history.equals(history)


def test_distinct_native_random_policies_share_global_rng_but_agent_warmup_owns_another_rng(monkeypatch):
    first = RewardAgent(RandomPolicy(), RNFailReward(), seed=17)
    other = RewardAgent(RandomPolicy(), RNFailReward(), seed=17)
    before = copy.deepcopy(policy_base._rng.bit_generator.state)
    for agent in (first, other):
        cycle(agent, {name: 0 for name in ("A", "B", "C", "D")})
    assert policy_base._rng.bit_generator.state == before
    assert first.policy is not other.policy
    local = rng_state(first)
    isolated = first.choose()
    monkeypatch.setattr(policy_base._rng.bit_generator, "state", copy.deepcopy(before))
    assert other.choose() == isolated
    assert first.choose() != isolated
    assert rng_state(first) == local


@pytest.mark.parametrize(
    ("name", "params", "failure", "message", "direct_type", "policy_type"),
    [
        ("SlMAB", {}, AttributeError, "history", RewardSlidingWindowAgent, SlMABPolicy),
        (
            "SWLinTS",
            {"alpha": 0.0},
            TypeError,
            "SlidingWindowContextualAgent",
            SlidingWindowContextualAgent,
            SWLinTSPolicy,
        ),
        (
            "SWContextualEpsilonGreedy",
            {"epsilon": 0.0},
            TypeError,
            "SlidingWindowContextualAgent",
            SlidingWindowContextualAgent,
            SWContextualEpsilonGreedyPolicy,
        ),
    ],
)
def test_three_preserved_factory_failures_and_their_actual_manual_window_controls(
    name, params, failure, message, direct_type, policy_type
):
    config = {name.lower(): {"window_sizes": [2], "rnfail": params}}
    normalized, issues = normalize_and_validate_agent_build(
        algorithm_configs=config, policy_names=[name], rewards_names=["RNFail"], strict=True
    )
    assert issues == []
    native = build_agents_from_config(normalized, [name], ["RNFail"], seed=17)[0]
    actual = prepare(native, {"A": 1, "B": 0})
    if name == "SlMAB":
        outcome = actual.pull(native.choose())
        with pytest.raises(failure, match=message):
            native.observe(outcome)
        assert native.t == 1  # Failure is not a transactional rollback.
    else:
        with pytest.raises(failure, match=message):
            native.choose()
    direct = direct_type(policy_type(**params), RNFailReward(), 2, seed=17)
    for verdicts in ({"A": 1, "B": 0}, {"A": 0, "B": 1}):
        assert sorted(cycle(direct, verdicts)["ranking"]) == ["A", "B"]
