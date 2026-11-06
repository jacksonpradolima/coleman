"""Actual installed native routes and explicitly host-composed checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SEED = 20261106


def _dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _installed():
    import coleman

    assert Path(coleman.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert not Path(coleman.__file__).resolve().is_relative_to(Path.cwd())


def one_route(name, fixture_root, work):
    """Execute one preserved factory route through the actual native Environment."""
    import polars as pl

    from coleman.environment import Environment
    from coleman.evaluation import NAPFDVerdictMetric
    from coleman.runner import build_agents_from_config, get_scenario_provider, normalize_and_validate_agent_build

    _installed()
    cases = json.loads((fixture_root / "legacy-dispatch.json").read_text())
    case = next(row for row in cases["policies"] if row["name"] == name)
    inputs = json.loads((fixture_root / "legacy-ci-inputs.json").read_text())
    config = {name.lower(): {"rnfail": case["constructor_parameters"]}}
    if case["window_sizes"]:
        config[name.lower()]["window_sizes"] = case["window_sizes"]
    if name == "PortfolioUCB":
        config["ucb"] = {"rnfail": {"c": 1.0}}
    normalized, issues = normalize_and_validate_agent_build(
        algorithm_configs=config, policy_names=[name], rewards_names=["RNFail"], strict=True
    )
    assert issues == []
    agents = build_agents_from_config(normalized, [name], ["RNFail"], seed=SEED)
    assert [type(agent).__module__ + "." + type(agent).__qualname__ for agent in agents] == [
        row["agent_type"] for row in case["agents"]
    ]
    assert [getattr(agent, "window_size", None) for agent in agents] == [row["window_size"] for row in case["agents"]]
    groups = []
    for index, agent in enumerate(agents):
        if any(index in group for group in groups):
            continue
        groups.append([other for other, candidate in enumerate(agents) if candidate.policy is agent.policy])
    assert groups == case["policy_identity_groups"]
    data = work / "data/authored-ci"
    data.mkdir(parents=True)
    frame = pl.DataFrame(inputs["rows"])
    frame.write_parquet(data / "features-engineered.parquet")
    frame.write_parquet(data / "features-engineered-contextual.parquet")
    provider = get_scenario_provider(
        str(work / "data"),
        "authored-ci",
        0.5,
        False,
        case["context_required"],
        {"previous_build": []},
        {"feature_group_name": "authored-signal", "feature_group_values": ["prior_score"]},
    )
    environment = Environment(
        agents,
        provider,
        NAPFDVerdictMetric(),
        results_config={
            "enabled": True,
            "sink": "parquet",
            "out_dir": str(work / "results"),
            "batch_size": 1,
            "top_k_prioritization": 3,
        },
        checkpoint_config={"enabled": False},
        telemetry_config={"enabled": False},
        runtime_metadata={"execution_id": "authored-dispatch", "worker_id": "1", "parallel_mode": "sequential"},
    )
    failure = None
    try:
        environment.run_single(1, trials=4, restore=False, forecast_next=False)
    except (AttributeError, TypeError) as error:
        failure = {"type": type(error).__module__ + "." + type(error).__qualname__, "message": str(error)}
    finally:
        environment.monitor.close()
    rows = environment.monitor.rows_collected
    assert rows == case["expected_native_rows"], (name, rows)
    if case["factory_outcome"] == "failed":
        expected = case["expected_failure"]
        assert failure == {"type": expected["type"], "message": expected["message"]}
    else:
        assert failure is None, failure
        assert all(agent.t == 4 for agent in agents)
        shards = list((work / "results").rglob("*.parquet"))
        assert shards
        actual = pl.read_parquet(shards, hive_partitioning=True)
        assert actual.height == rows
        assert set(actual["step"].to_list()) == {1, 2, 3, 4}
        for row in actual.to_dicts():
            order = json.loads(row["prioritization_order_top_k"])
            assert sorted(order) == ["integration", "system", "unit"]
            assert hashlib.sha256(json.dumps(order).encode()).hexdigest() == row["prioritization_order_hash"]
            assert row["tests_ran"] + row["tests_not_ran"] == 3
            assert row["detected"] + row["missed"] == 1
    return {
        "name": name,
        "factory_outcome": case["factory_outcome"],
        "native_rows": rows,
        "single_trace_reproducibility": case["single_trace_reproducibility"],
        "failure": failure,
    }


def matrix(fixture_root, work):
    """Run all 37 actual names in fresh processes without a cross-platform trace claim."""
    from coleman.native.semantics import list_native_policies
    from coleman.runner import _available_policy_names

    _installed()
    cases = json.loads((fixture_root / "legacy-dispatch.json").read_text())["policies"]
    expected = {row["name"] for row in cases}
    assert {row.name for row in list_native_policies()} == expected
    assert set(_available_policy_names()) == expected
    records = []
    for case in cases:
        child = work / case["name"]
        child.mkdir()
        result = subprocess.run(
            [sys.executable, "-I", "-B", __file__, "one", str(fixture_root), str(child), case["name"]],
            cwd=child,
            env=os.environ,
            text=True,
            capture_output=True,
            timeout=40,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        records.append(json.loads((child / "receipt.json").read_text()))
    return {
        "records": records,
        "completed": sum(row["factory_outcome"] == "completed" for row in records),
        "failed": sum(row["factory_outcome"] == "failed" for row in records),
        "native_rows": sum(row["native_rows"] for row in records),
    }


def _host_bandit(rows):
    from coleman.bandit import EvaluationMetricBandit
    from coleman.evaluation import NAPFDVerdictMetric

    metric = NAPFDVerdictMetric()
    metric.update_available_time(3.0)
    return EvaluationMetricBandit(rows, metric)


def checkpoint(fixture_root, work, mode):
    """Compose the real new capture with the existing trusted native state store."""
    import numpy as np

    from coleman.agent import RewardAgent
    from coleman.application.workflow_checkpoint import WorkflowCheckpointStore
    from coleman.artifacts import ArtifactWriter
    from coleman.checkpoint.state import CheckpointPayload
    from coleman.core.identity import Reference
    from coleman.core.primitives import fingerprint
    from coleman.native.capture import choose_and_freeze_native
    from coleman.native.decisions import NativeDecisionContext, native_binding
    from coleman.native.publication import publish_native_decision, read_native_decision
    from coleman.policy import RandomPolicy
    from coleman.policy import base as policy_base
    from coleman.reward import RNFailReward
    from coleman.spec.selected_runtime import capture_native_runtime

    _installed()
    inputs = json.loads((fixture_root / "legacy-ci-inputs.json").read_text())
    configuration = {"policy": "Random", "reward": "RNFail", "seed": SEED, "budget_seconds": 3}
    producer = capture_native_runtime()
    binding = fingerprint({"inputs": inputs, "configuration": configuration, "producer": producer.fingerprint})
    store = WorkflowCheckpointStore(
        str(work / "checkpoints"), binding, minimum_step=2 if mode == "resume-checkpoint" else 0
    )
    writer = ArtifactWriter(work / "records")
    payload = store.load("authored-ci", 1)
    if payload is None:
        policy_base._rng = np.random.default_rng(SEED)
        agent = RewardAgent(RandomPolicy(), RNFailReward(), seed=SEED)
        start, mathematics = 1, []
    else:
        agent = payload.agents[0]
        start, mathematics = payload.step + 1, payload.extra["mathematics"]
        record = read_native_decision(payload.extra["last_native_decision"])
        before = json.dumps(policy_base._rng.bit_generator.state, sort_keys=True)
        assert publish_native_decision(record, writer).read_text() == record.to_json()
        assert json.dumps(policy_base._rng.bit_generator.state, sort_keys=True) == before
    for step in range(start, 5):
        rows = [row for row in inputs["rows"] if row["BuildId"] == step]
        actual = _host_bandit(rows)
        agent.update_bandit(actual)
        context = NativeDecisionContext(
            schema_version=1,
            run=Reference(id="authored-native-four-build", version="1"),
            execution_id="owned-random",
            decision_id=f"build-{step}",
            purpose="evaluation_ranking",
            expected_round=step - 1,
            binding=native_binding("Random", "RNFail"),
            configuration_fingerprint=fingerprint(configuration),
            input_fingerprint=fingerprint(rows),
            producer_fingerprint=producer.fingerprint,
            policy_state_group=Reference(id="owned-random-policy", version="1"),
        )
        record = choose_and_freeze_native(agent, context=context)
        before = json.dumps(policy_base._rng.bit_generator.state, sort_keys=True)
        own = json.dumps(agent._rng.bit_generator.state, sort_keys=True)
        path = publish_native_decision(record, writer)
        assert publish_native_decision(record, writer) == path
        assert json.dumps(policy_base._rng.bit_generator.state, sort_keys=True) == before
        assert json.dumps(agent._rng.bit_generator.state, sort_keys=True) == own
        assert agent.t == step - 1
        if mode == "crash-publication" and step == 2:
            os._exit(81)
        outcome = actual.pull(list(record.ranking))
        agent.observe(outcome)
        assert isinstance(agent.last_reward, list)
        mathematics.append(
            {
                "step": step,
                "ranking": list(record.ranking),
                "fitness": outcome.fitness,
                "cost": outcome.cost,
                "detected": outcome.detected_failures,
                "missed": outcome.undetected_failures,
                "credits": list(agent.last_reward),
            }
        )
        store.save(
            CheckpointPayload(
                run_id="authored-ci",
                experiment=1,
                step=step,
                agents=[agent],
                bandit=actual,
                extra={"last_native_decision": record.to_json(), "mathematics": mathematics},
            )
        )
        if mode == "crash-checkpoint" and step == 2:
            os._exit(82)
    snapshots = [read_native_decision(path.read_bytes()) for path in (work / "records").rglob("*.json")]
    snapshots.sort(key=lambda record: record.observation_round)
    assert [record.observation_round for record in snapshots] == [0, 1, 2, 3]
    return {
        "mathematics": mathematics,
        "decisions": [record.to_json() for record in snapshots],
        "agent_rng": agent._rng.bit_generator.state,
        "policy_rng": policy_base._rng.bit_generator.state,
        "agent_table": agent.actions.to_dicts(),
        "counter": agent.t,
        "qualification": (
            "explicit trusted host composition; no automatic Environment instrumentation or cross-store transaction"
        ),
    }


def main():
    mode, fixture, owned, *arguments = sys.argv[1:]
    fixture_root, work = Path(fixture), Path(owned)
    if mode == "one":
        result = one_route(arguments[0], fixture_root, work)
    elif mode == "matrix":
        result = matrix(fixture_root, work)
    elif mode == "checkpoint":
        result = checkpoint(fixture_root, work, arguments[0])
    else:
        raise ValueError("Unknown owned qualification mode")
    _dump(work / "receipt.json", result)


if __name__ == "__main__":
    main()
