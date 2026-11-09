"""Run unchanged native YAML/API/Parquet paths in a real installed environment.

The logical projection deliberately excludes measured process resources, wall
time and physical execution identifiers. Its exact digest is a compatibility
check on this bounded authored CI tape, not a universal stochastic trace claim.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

PROFILE = "legacy-native-numeric-v1"
NUMERIC_DISTRIBUTIONS = ("numpy", "polars", "pyarrow", "duckdb", "pydantic", "PyYAML", "packaging")
PROJECTED_FIELDS = (
    "scenario",
    "experiment",
    "step",
    "policy",
    "reward_function",
    "budget_mode",
    "budget_value",
    "total_build_duration",
    "detected",
    "missed",
    "tests_ran",
    "tests_not_ran",
    "ttf",
    "ttf_duration",
    "time_reduction",
    "fitness",
    "cost",
    "rewards",
    "avg_precision",
    "prioritization_order_hash",
    "prioritization_order_top_k",
    "variant",
)
SEED = 20261106


def canonical(value):
    """Encode only the selected finite JSON values with deterministic ordering."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _installed():
    """Require the actual noneditable package and report its selected numeric pins."""
    import coleman

    assert Path(coleman.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert not Path(coleman.__file__).resolve().is_relative_to(Path.cwd())
    return {name: importlib.metadata.version(name) for name in NUMERIC_DISTRIBUTIONS}


def run_case(inputs_path: Path, work: Path, policy: str, budget_mode: str, budget_value: float):
    """Exercise the real legacy loader, public API, runner and native result sink."""
    import polars as pl
    import yaml

    from coleman.api import load_spec, run
    from coleman.spec.run_id import compute_run_id

    versions = _installed()
    assert policy in {"Random", "UCB"}
    assert budget_mode in {"ratio", "fixed_time", "subset_size"}
    work.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(inputs_path.read_text())
    assert inputs["evidence_basis"] == "authored_synthetic_ci"
    assert inputs["duration_unit"] == "s"
    data = work / "data/compat-ci"
    data.mkdir(parents=True)
    pl.DataFrame(inputs["rows"]).write_parquet(data / "features-engineered.parquet")
    configuration = {
        "execution": {"parallel_pool_size": 1, "independent_executions": 1, "seed": SEED, "verbose": False},
        "experiment": {
            "budget": {"mode": budget_mode, "values": [budget_value]},
            "datasets_dir": str(work / "data"),
            "datasets": ["compat-ci"],
            "experiment_dir": str(work / "experiments") + "/",
            "rewards": ["RNFail"],
            "policies": [policy],
        },
        "algorithm": {"ucb": {"rnfail": {"c": 1.0}}},
        "results": {
            "enabled": True,
            "sink": "parquet",
            "out_dir": str(work / "runs"),
            "batch_size": 1,
            "top_k_prioritization": 3,
            "manifest_enabled": True,
        },
        "checkpoint": {"enabled": False},
        "telemetry": {"enabled": False},
    }
    configuration_path = work / "native.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration, sort_keys=True))
    spec = load_spec(configuration_path)
    outcome = run(spec)
    assert outcome.spec == spec and outcome.metrics == {}
    assert outcome.run_id == compute_run_id(spec)
    assert outcome.artifacts_dir is not None
    artifacts = Path(outcome.artifacts_dir)
    assert artifacts.is_relative_to(work)
    assert (artifacts / "spec.resolved.json").is_file()
    assert (artifacts / "provenance.json").is_file()
    assert (artifacts / "manifest.json").is_file()
    shards = sorted((artifacts / "results").rglob("*.parquet"))
    assert shards
    frame = pl.read_parquet(shards, hive_partitioning=True)
    assert frame.height == 4
    rows = sorted(frame.select(PROJECTED_FIELDS).to_dicts(), key=lambda row: (row["experiment"], row["step"]))
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    for row in rows:
        order = json.loads(row["prioritization_order_top_k"])
        assert sorted(order) == ["integration", "system", "unit"]
        assert hashlib.sha256(json.dumps(order).encode()).hexdigest() == row["prioritization_order_hash"]
        assert row["tests_ran"] + row["tests_not_ran"] == 3
        assert row["detected"] + row["missed"] == 1
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "policy": policy,
        "budget_mode": budget_mode,
        "budget_value": budget_value,
        "seed": SEED,
        "input_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "numeric_versions": versions,
        "producer": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
        "projected_fields": list(PROJECTED_FIELDS),
        "excluded_fields": sorted(set(frame.columns) - set(PROJECTED_FIELDS)),
        "api_metrics_empty": True,
        "rows": rows,
        "projection_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
    }


def main():
    """Write a receipt only after the complete real native result passes checks."""
    assert len(sys.argv) == 6
    inputs, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    result = run_case(inputs, work, sys.argv[3], sys.argv[4], float(sys.argv[5]))
    destination = work / "receipt.json"
    assert not destination.exists()
    destination.write_text(canonical(result) + "\n")
    print(canonical({"receipt": str(destination), "projection_sha256": result["projection_sha256"]}))


if __name__ == "__main__":
    main()
