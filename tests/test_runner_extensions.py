"""Tests for runner extension and compatibility validation APIs."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coleman.runner import (
    EnvironmentBuildConfig,
    RunnerExtension,
    build_agents_from_config,
    build_environment,
    effective_parallel_pool_size,
    normalize_and_validate_agent_build,
)


def test_effective_parallel_pool_size_public_helper_matches_internal_behavior():
    with patch.dict("os.environ", {"SCALENE_ALLOCATION_SAMPLING_WINDOW": "1024"}, clear=True):
        assert effective_parallel_pool_size(4) == 1


def test_build_environment_uses_extension_builder():
    expected_env = Mock()

    def _builder(config, runtime_metadata, seed):  # noqa: ARG001
        return expected_env, 9

    extension = RunnerExtension(build_environment_fn=_builder)
    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
        extension=extension,
    )

    env, trials = build_environment(config, {"execution_id": "e", "worker_id": "1", "parallel_mode": "s"})

    assert env is expected_env
    assert trials == 9


def test_build_environment_applies_extension_metric_to_custom_env():
    expected_env = Mock()
    expected_env.evaluation_metric = Mock()
    expected_metric = object()

    def _builder(config, runtime_metadata, seed):  # noqa: ARG001
        return expected_env, 3

    def _metric_builder(spec, dataset, budget_mode, budget_value):  # noqa: ARG001
        return expected_metric

    extension = RunnerExtension(build_environment_fn=_builder, build_metric_fn=_metric_builder)
    config = EnvironmentBuildConfig(
        datasets_dir="examples",
        dataset="fakedata",
        sched_time_ratio=0.5,
        use_hcs=False,
        use_context=False,
        context_config={},
        feature_groups={},
        results_config={},
        checkpoint_config={},
        telemetry_config={},
        algorithm_configs={},
        rewards_names=["RNFail"],
        policy_names=["Random"],
        extension=extension,
        resolved_spec=Mock(),
    )

    env, trials = build_environment(config, {"execution_id": "e", "worker_id": "1", "parallel_mode": "s"})

    assert env is expected_env
    assert env.evaluation_metric is expected_metric
    assert trials == 3


def test_normalize_and_validate_agent_build_portfolio_string_aliases():
    normalized, issues = normalize_and_validate_agent_build(
        algorithm_configs={
            "portfolioucb": {
                "rnfail": {
                    "policies": ["random", "greedy"],
                    "c": 1.0,
                }
            }
        },
        policy_names=["PortfolioUCB"],
        rewards_names=["RNFail"],
        strict=True,
    )

    nested = normalized["portfolioucb"]["rnfail"]["policies"]
    assert issues == []
    assert len(nested) == 2
    assert nested == ["Random", "Greedy"]


def test_build_agents_from_config_portfolio_builds_fresh_nested_instances():
    agents = build_agents_from_config(
        algorithm_configs={
            "portfolioucb": {
                "rnfail": {
                    "policies": ["Random", "Greedy"],
                    "c": 1.0,
                }
            }
        },
        policy_names=["PortfolioUCB"],
        rewards_names=["RNFail"],
        seed=123,
    )

    assert len(agents) == 1
    assert len(agents[0].policy.policies) == 2
    assert all(hasattr(p, "choose_all") for p in agents[0].policy.policies)


def test_normalize_and_validate_agent_build_raises_for_missing_window_sizes():
    with pytest.raises(ValueError, match="window_sizes"):
        normalize_and_validate_agent_build(
            algorithm_configs={"frrmab": {}},
            policy_names=["FRRMAB"],
            rewards_names=["RNFail"],
            strict=True,
        )


def test_normalize_and_validate_agent_build_collects_unknown_policy_when_not_strict():
    normalized, issues = normalize_and_validate_agent_build(
        algorithm_configs={},
        policy_names=["NoSuchPolicy"],
        rewards_names=["RNFail"],
        strict=False,
    )

    assert normalized == {}
    assert len(issues) == 1
    assert issues[0].code == "unknown_policy"


def test_normalize_and_validate_agent_build_portfolio_requires_reward_dict():
    _, issues = normalize_and_validate_agent_build(
        algorithm_configs={
            "portfolioucb": {
                "rnfail": [],
            }
        },
        policy_names=["PortfolioUCB"],
        rewards_names=["RNFail"],
        strict=False,
    )

    assert any(issue.code == "invalid_reward_policy_config" for issue in issues)


def test_normalize_and_validate_agent_build_portfolio_requires_non_empty_policy_list():
    _, issues = normalize_and_validate_agent_build(
        algorithm_configs={
            "portfolioucb": {
                "rnfail": {
                    "policies": [],
                }
            }
        },
        policy_names=["PortfolioUCB"],
        rewards_names=["RNFail"],
        strict=False,
    )

    assert any(issue.code == "portfolio_empty_policies" for issue in issues)


def test_normalize_and_validate_agent_build_portfolio_rejects_non_string_candidates():
    _, issues = normalize_and_validate_agent_build(
        algorithm_configs={
            "portfolioucb": {
                "rnfail": {
                    "policies": ["Random", 123],
                }
            }
        },
        policy_names=["PortfolioUCB"],
        rewards_names=["RNFail"],
        strict=False,
    )

    assert any(issue.code == "portfolio_invalid_policies_entry" for issue in issues)


def test_normalize_and_validate_agent_build_portfolio_reports_unknown_and_recursive_candidates():
    _, issues = normalize_and_validate_agent_build(
        algorithm_configs={
            "portfolioucb": {
                "rnfail": {
                    "policies": ["PortfolioUCB", "NotExisting"],
                }
            }
        },
        policy_names=["PortfolioUCB"],
        rewards_names=["RNFail"],
        strict=False,
    )

    codes = {issue.code for issue in issues}
    assert "unknown_portfolio_policy" in codes
    assert "portfolio_recursive_reference" in codes


def test_normalize_and_validate_agent_build_portfolio_init_error_is_collected():
    with patch("coleman.runner.load_class_from_module", side_effect=TypeError("boom")):
        _, issues = normalize_and_validate_agent_build(
            algorithm_configs={
                "portfolioucb": {
                    "rnfail": {
                        "policies": ["Random"],
                    }
                }
            },
            policy_names=["PortfolioUCB"],
            rewards_names=["RNFail"],
            strict=False,
        )

    assert any(issue.code == "portfolio_policy_init_error" for issue in issues)
