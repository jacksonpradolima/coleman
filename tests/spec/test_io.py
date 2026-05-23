"""Tests for spec I/O (load_spec, save_resolved)."""

import json
import os
import tempfile

import yaml

from coleman.spec.io import load_spec, load_sweep_spec, save_resolved
from coleman.spec.models import ExecutionSpec, RunSpec


class TestLoadSpec:
    def test_load_minimal_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"verbose": True}}, fh)
            path = fh.name
        try:
            spec = load_spec(path)
            assert spec.execution.verbose is True
            assert spec.execution.parallel_pool_size == 10  # default
        finally:
            os.unlink(path)

    def test_load_full_config(self):
        data = {
            "execution": {"parallel_pool_size": 4},
            "experiment": {"datasets": ["a@b"], "policies": ["Random"]},
            "results": {"enabled": False},
            "checkpoint": {"enabled": False},
            "telemetry": {"enabled": False},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(data, fh)
            path = fh.name
        try:
            spec = load_spec(path)
            assert spec.execution.parallel_pool_size == 4
            assert spec.experiment.datasets == ["a@b"]
            assert spec.results.enabled is False
        finally:
            os.unlink(path)

    def test_load_with_packs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "packs", "results"))
            with open(os.path.join(tmpdir, "packs", "results", "parquet.yaml"), "w") as fh:
                yaml.dump({"results": {"sink": "parquet", "enabled": True}}, fh)

            cfg_path = os.path.join(tmpdir, "run.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {"packs": ["results/parquet"], "execution": {"verbose": True}},
                    fh,
                )

            spec = load_spec(cfg_path, packs_dir=os.path.join(tmpdir, "packs"))
            assert spec.results.sink == "parquet"
            assert spec.execution.verbose is True

    def test_load_packs_dir_from_config_location(self):
        """Default packs_dir is derived from the config file's directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "packs", "results"))
            with open(os.path.join(tmpdir, "packs", "results", "parquet.yaml"), "w") as fh:
                yaml.dump({"results": {"sink": "parquet", "enabled": True}}, fh)

            cfg_path = os.path.join(tmpdir, "run.yaml")
            with open(cfg_path, "w") as fh:
                yaml.dump(
                    {"packs": ["results/parquet"], "execution": {"verbose": True}},
                    fh,
                )

            # No explicit packs_dir — should resolve from config file location
            spec = load_spec(cfg_path)
            assert spec.results.sink == "parquet"
            assert spec.execution.verbose is True

    def test_load_spec_ignores_top_level_sweep(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(
                {
                    "execution": {"verbose": True},
                    "sweep": {
                        "axes": [
                            {
                                "mode": "grid",
                                "params": {
                                    "execution.seed": [0, 1],
                                },
                            }
                        ]
                    },
                },
                fh,
            )
            path = fh.name
        try:
            spec = load_spec(path)
            assert spec.execution.verbose is True
        finally:
            os.unlink(path)


class TestLoadSweepSpec:
    def test_returns_none_when_missing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"verbose": True}}, fh)
            path = fh.name
        try:
            assert load_sweep_spec(path) is None
        finally:
            os.unlink(path)

    def test_loads_sweep_section(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(
                {
                    "sweep": {
                        "axes": [
                            {
                                "mode": "grid",
                                "params": {
                                    "algorithm.ucb.rnfail.c": [0.1, 0.3, 0.5],
                                },
                            }
                        ],
                        "seeds": [10, 20],
                    }
                },
                fh,
            )
            path = fh.name
        try:
            sweep = load_sweep_spec(path)
            assert sweep is not None
            assert len(sweep.axes) == 1
            assert sweep.axes[0].mode == "grid"
            assert sweep.axes[0].params["algorithm.ucb.rnfail.c"] == [0.1, 0.3, 0.5]
            assert sweep.seeds == [10, 20]
        finally:
            os.unlink(path)


class TestSaveResolved:
    def test_creates_file(self):
        spec = RunSpec()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_resolved(spec, os.path.join(tmpdir, "sub", "spec.json"))
            assert out.exists()
            with open(out) as fh:
                data = json.load(fh)
            assert data["execution"]["parallel_pool_size"] == 10

    def test_roundtrip(self):
        spec = RunSpec(execution=ExecutionSpec(parallel_pool_size=7))
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_resolved(spec, os.path.join(tmpdir, "spec.json"))
            with open(out) as fh:
                data = json.load(fh)
            spec2 = RunSpec.model_validate(data)
            assert spec == spec2

    def test_redacts_sensitive_fields_by_default(self):
        spec = RunSpec.model_validate(
            {
                "telemetry": {
                    "enabled": True,
                    "otlp_endpoint": "https://user:pass@example.com:4318/v1/metrics?token=abc123",
                },
                "algorithm": {
                    "api_key": "secret-value",
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_resolved(spec, os.path.join(tmpdir, "spec.json"))
            with open(out) as fh:
                data = json.load(fh)

        assert data["algorithm"]["api_key"] == "<redacted>"
        endpoint = data["telemetry"]["otlp_endpoint"]
        assert "<redacted>:<redacted>@" in endpoint
        assert "token=%3Credacted%3E" in endpoint

    def test_allows_disabling_redaction(self):
        spec = RunSpec.model_validate(
            {
                "algorithm": {
                    "api_key": "secret-value",
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_resolved(spec, os.path.join(tmpdir, "spec.json"), redact_sensitive=False)
            with open(out) as fh:
                data = json.load(fh)

        assert data["algorithm"]["api_key"] == "secret-value"
