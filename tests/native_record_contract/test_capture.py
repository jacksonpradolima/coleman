import copy
import json

import numpy as np
import polars as pl
import pytest

from coleman import policy as policies
from coleman.agent import ContextualAgent, RewardAgent, RewardSlidingWindowAgent
from coleman.artifacts import ArtifactWriter
from coleman.bandit import EvaluationMetricBandit
from coleman.evaluation import NAPFDVerdictMetric
from coleman.native.capture import choose_and_freeze_native
from coleman.native.decisions import NativeDecisionContext, native_binding
from coleman.native.publication import publish_native_decision, read_native_decision
from coleman.policy import EXP3Policy, FRRMABPolicy, GreedyPolicy, LinUCBPolicy, SlMABPolicy
from coleman.reward import RNFailReward, TimeRankReward


def _agent(policy=None):
    agent = RewardAgent(policy if policy is not None else EXP3Policy(gamma=0.2), RNFailReward(), seed=17)
    agent.update_actions(["C", "A", "B"])
    return agent


def test_snapshot_owns_one_real_choose_matches_direct_state_and_does_not_observe(context):
    agent = _agent()
    direct = copy.deepcopy(agent)
    expected = tuple(direct.choose())
    before_global = copy.deepcopy(policies._rng.bit_generator.state)
    record = choose_and_freeze_native(agent, context=context)
    assert record.ranking == expected == tuple(agent.last_prioritization)
    assert record.candidates == ("A", "B", "C")
    assert record.phase == "initial_shuffle"
    assert record.observation_round == agent.t == direct.t == 0
    assert agent.actions.equals(direct.actions)
    assert agent.policy.__dict__ == direct.policy.__dict__
    assert agent._rng.bit_generator.state == direct._rng.bit_generator.state
    assert policies._rng.bit_generator.state == before_global
    agent.last_prioritization.reverse()
    assert record.ranking == expected


def test_policy_phase_is_real_exp3_sort_without_exporting_internal_scores_as_mass(context):
    agent = _agent()
    agent.t = 1
    phase_context = NativeDecisionContext.model_validate(context.model_copy(update={"expected_round": 1}))
    old_rng = copy.deepcopy(policies._rng.bit_generator.state)
    record = choose_and_freeze_native(agent, context=phase_context)
    assert record.ranking == ("A", "B", "C")
    assert record.phase == "policy_ranking"
    assert agent.policy._probs(list(record.candidates)) == dict.fromkeys(record.candidates, 1 / 3)
    assert record.sampling.status == "not_recorded"
    assert record.sampling.value is None
    assert record.loss.status == "not_defined"
    assert policies._rng.bit_generator.state == old_rng


def test_contextual_agent_delegates_at_zero_instead_of_claiming_initial_shuffle(context):
    agent = ContextualAgent(LinUCBPolicy(alpha=0.1), RNFailReward(), seed=17)
    agent.update_features(["prior_score"])
    agent.update_context(pl.DataFrame({"Name": ["A", "B"], "prior_score": [1.0, -1.0]}))
    agent.update_actions(["A", "B"])
    current = NativeDecisionContext.model_validate(
        context.model_copy(update={"binding": native_binding("LinUCB", "RNFail")})
    )
    old_rng = copy.deepcopy(agent._rng.bit_generator.state)
    record = choose_and_freeze_native(agent, context=current)
    assert record.phase == "policy_ranking"
    assert set(record.ranking) == {"A", "B"}
    assert record.observation_round == 0
    assert agent._rng.bit_generator.state == old_rng


def test_audited_manual_window_is_distinct_from_refused_slmab_factory(context):
    agent = RewardSlidingWindowAgent(SlMABPolicy(), RNFailReward(), window_size=2, seed=17)
    agent.update_actions(["A", "B"])
    current = NativeDecisionContext.model_validate(
        context.model_copy(update={"binding": native_binding("SlMAB", "RNFail", route="manual-window")})
    )
    record = choose_and_freeze_native(agent, context=current)
    assert record.window_size == 2
    assert record.context.binding.route.origin == "manual"


@pytest.mark.parametrize("failure", ["count", "policy", "reward", "class", "window", "candidate", "definition"])
def test_metadata_preconditions_reject_before_stateful_choose(context, failure, monkeypatch):
    agent = _agent()
    if failure == "count":
        agent.t = 1
    elif failure == "policy":
        agent.policy = GreedyPolicy()
    elif failure == "reward":
        agent.reward_function = TimeRankReward()
    elif failure == "class":

        class Derived(RewardAgent):
            pass

        agent = Derived(EXP3Policy(), RNFailReward())
    elif failure == "window":
        agent.window_size = 2
    elif failure == "candidate":
        agent.actions = pl.DataFrame(
            {"Name": ["A", "A"], "ActionAttempts": [0.0, 0.0], "ValueEstimates": [0.0, 0.0], "Q": [0.0, 0.0]}
        )
    else:
        binding = context.binding.model_copy(update={"catalog_fingerprint": "sha256:" + "0" * 64})
        context = NativeDecisionContext.model_validate(context.model_copy(update={"binding": binding}))

    def must_not_choose():
        pytest.fail("invalid native metadata reached stateful choose")

    monkeypatch.setattr(agent, "choose", must_not_choose)
    with pytest.raises(ValueError):
        choose_and_freeze_native(agent, context=context)


@pytest.mark.parametrize("name", ["SlMAB", "SWLinTS", "SWContextualEpsilonGreedy"])
def test_new_route_refuses_known_factory_failures_before_backend_choice(context, name, monkeypatch):
    agent = _agent()
    current = NativeDecisionContext.model_validate(
        context.model_copy(update={"binding": native_binding(name, "RNFail")})
    )

    def must_not_choose():
        pytest.fail("unsupported factory route chose before refusal")

    monkeypatch.setattr(agent, "choose", must_not_choose)
    with pytest.raises(ValueError, match="Unsupported native route"):
        choose_and_freeze_native(agent, context=current)


def test_forecast_is_separate_stateful_choice_and_does_not_rewrite_evaluation_record(context, tmp_path):
    agent = _agent()
    evaluated = choose_and_freeze_native(agent, context=context)
    original = evaluated.to_json()
    forecast_context = NativeDecisionContext.model_validate(
        context.model_copy(update={"purpose": "forecast", "decision_id": "build-1.forecast"})
    )
    before_forecast_rng = copy.deepcopy(agent._rng.bit_generator.state)
    forecast = choose_and_freeze_native(agent, context=forecast_context)
    assert agent.t == 0
    assert agent._rng.bit_generator.state != before_forecast_rng
    assert forecast.logical_identity != evaluated.logical_identity
    assert evaluated.to_json() == original
    after_choice_rng = copy.deepcopy(agent._rng.bit_generator.state)
    path = publish_native_decision(evaluated, ArtifactWriter(tmp_path))
    for _ in range(2):
        assert read_native_decision(path.read_bytes()) == evaluated
        assert publish_native_decision(evaluated, ArtifactWriter(tmp_path)) == path
    assert agent._rng.bit_generator.state == after_choice_rng
    assert evaluated.to_json() == original


def test_no_snapshot_or_false_completed_record_when_actual_native_choice_raises(context, tmp_path):
    agent = ContextualAgent(LinUCBPolicy(alpha=0.1), RNFailReward(), seed=17)
    current = NativeDecisionContext.model_validate(
        context.model_copy(update={"binding": native_binding("LinUCB", "RNFail")})
    )
    # The actual native choice reads its missing context; no fake policy is used.
    old_rng = copy.deepcopy(agent._rng.bit_generator.state)
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        choose_and_freeze_native(agent, context=current)
    assert not list(tmp_path.iterdir())
    assert agent._rng.bit_generator.state == old_rng


@pytest.mark.parametrize("value", [-1, True, np.int64(0), 0.0])
def test_actual_counter_is_not_coerced_to_a_different_contract(context, value):
    agent = _agent()
    agent.t = value
    old_rng = copy.deepcopy(agent._rng.bit_generator.state)
    with pytest.raises(ValueError, match="agent.t"):
        choose_and_freeze_native(agent, context=context)
    assert agent._rng.bit_generator.state == old_rng


def test_exact_verbatim_names_survive_real_native_initial_choice(context):
    agent = _agent()
    names = ["", " test/α ", "../suite/B", 'quoted"[{\\test']
    agent.update_actions(names)
    record = choose_and_freeze_native(agent, context=context)
    assert set(record.ranking) == set(names)
    assert read_native_decision(json.dumps(json.loads(record.to_json()))).ranking == record.ranking


def test_actual_frrmab_retired_emission_is_refused_without_rewriting_legacy_policy(context, tmp_path):
    policy = FRRMABPolicy(c=0.0)
    agent = RewardSlidingWindowAgent(policy, RNFailReward(), window_size=2, seed=17)
    metric = NAPFDVerdictMetric()
    metric.update_available_time(2.0)
    arms = [
        {
            "Name": name,
            "Duration": 1.0,
            "Verdict": verdict,
            "CalcPrio": 0,
            "LastRun": "",
            "NumRan": 0,
            "NumErrors": verdict,
            "LastResults": "",
        }
        for name, verdict in (("A", 1), ("B", 0))
    ]
    bandit = EvaluationMetricBandit(arms, metric)
    agent.update_bandit(bandit)
    agent.observe(bandit.pull(agent.choose()))
    agent.update_actions(["B"])
    current = NativeDecisionContext.model_validate(
        context.model_copy(update={"binding": native_binding("FRRMAB", "RNFail"), "expected_round": 1})
    )
    with pytest.raises(ValueError, match="permutation"):
        choose_and_freeze_native(agent, context=current)
    assert agent.actions["Name"].to_list() == ["B"]
    assert "A" in agent.last_prioritization
    assert agent.t == 1
    assert not list(tmp_path.iterdir())
