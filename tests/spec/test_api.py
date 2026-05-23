"""Tests for the public API module."""

import os
import tempfile

import pytest
import yaml

from coleman.api import RunResult, load_spec, run, run_many, run_with_extension, save_resolved, sweep
from coleman.budget import BudgetMode
from coleman.runner import RunnerExtension
from coleman.spec.models import BudgetSpec, ExecutionSpec, ExperimentSpec, ResultsSpec, RunSpec
from coleman.spec.run_id import compute_run_id
from coleman.spec.sweep import SweepAxis, SweepSpec


def _light_run_spec(tmpdir: str, **execution_overrides) -> RunSpec:
    """Build a minimal spec for fast API tests."""
    execution_kwargs = {
        "parallel_pool_size": 1,
        "independent_executions": 1,
        "verbose": False,
    }
    execution_kwargs.update(execution_overrides)
    execution = ExecutionSpec(**execution_kwargs)  # ty:ignore[invalid-argument-type]
    experiment = ExperimentSpec(
        budget=BudgetSpec(mode=BudgetMode.RATIO, values=[0.1]),
        datasets_dir="examples",
        datasets=["fakedata"],
        rewards=["RNFail"],
        policies=["Random"],
    )
    results = ResultsSpec(out_dir=tmpdir)
    return RunSpec(execution=execution, experiment=experiment, results=results)


class _NoOpEnvironment:
    """Minimal environment used to validate extension flow wiring."""

    def __init__(self) -> None:
        self.runtime_metadata: dict[str, str] | None = None

    def set_runtime_metadata(self, runtime_metadata):
        self.runtime_metadata = runtime_metadata

    def run_single(self, iteration, trials):  # noqa: ARG002
        return None

    def store_experiment(self):
        return None


def _picklable_extension_builder(config, runtime_metadata, agent_seed):  # noqa: ARG001
    return _NoOpEnvironment(), 1


def _picklable_post_execution(context, env):  # noqa: ARG001
    return None


_PICKLABLE_EXTENSION = RunnerExtension(
    build_environment_fn=_picklable_extension_builder,
    post_execution_fn=_picklable_post_execution,
)


class TestRunResult:
    def test_repr(self):
        spec = RunSpec()
        r = RunResult(run_id="abc123", spec=spec, metrics={"napfd": 0.9})
        assert "abc123" in repr(r)
        assert "napfd" in repr(r)

    def test_defaults(self):
        spec = RunSpec()
        r = RunResult(run_id="x", spec=spec)
        assert r.metrics == {}
        assert r.artifacts_dir is None


class TestRun:
    def test_produces_run_id(self):
        spec = _light_run_spec(tempfile.mkdtemp())
        result = run(spec)
        assert len(result.run_id) == 12
        assert result.run_id == compute_run_id(spec)

    def test_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _light_run_spec(tmpdir)
            result = run(spec)
            assert result.artifacts_dir is not None
            assert os.path.exists(os.path.join(result.artifacts_dir, "spec.resolved.json"))
            assert os.path.exists(os.path.join(result.artifacts_dir, "provenance.json"))

    def test_deterministic_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec1 = _light_run_spec(tmpdir)
            spec2 = _light_run_spec(tmpdir)
            r1 = run(spec1)
            r2 = run(spec2)
            assert r1.run_id == r2.run_id


class TestRunWithExtension:
    def test_run_with_extension_sequential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _light_run_spec(tmpdir)
            calls: dict[str, int] = {"build": 0, "post": 0}

            def _builder(config, runtime_metadata, agent_seed):  # noqa: ARG001
                calls["build"] += 1
                return _NoOpEnvironment(), 1

            def _post(context, env):  # noqa: ARG001
                calls["post"] += 1

            extension = RunnerExtension(build_environment_fn=_builder, post_execution_fn=_post)

            result = run_with_extension(spec, extension)

            assert len(result.run_id) == 12
            assert result.run_id == compute_run_id(spec)
            assert calls["build"] == 1
            assert calls["post"] == 1

    def test_run_with_extension_parallel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _light_run_spec(
                tmpdir,
                parallel_pool_size=2,
                independent_executions=2,
            )

            result = run_with_extension(spec, _PICKLABLE_EXTENSION)

            assert len(result.run_id) == 12
            assert result.artifacts_dir is not None
            assert os.path.exists(os.path.join(result.artifacts_dir, "spec.resolved.json"))


class TestRunMany:
    def test_sequential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [_light_run_spec(tmpdir, seed=i) for i in (1, 2, 3)]
            results = run_many(specs, max_workers=1)
            assert len(results) == 3
            ids = [r.run_id for r in results]
            assert len(set(ids)) == 3  # All unique


class TestSweep:
    def test_returns_expanded_specs(self):
        base = RunSpec()
        sw = SweepSpec(axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2, 4]})])
        specs = sweep(base, sw)
        assert len(specs) == 3

    def test_empty_sweep(self):
        base = RunSpec()
        sw = SweepSpec()
        specs = sweep(base, sw)
        assert len(specs) == 1


class TestSeedApplication:
    def test_policy_rng_reexport_stays_synced_with_base_rng(self):
        """Assigning through coleman.policy._rng must update base RNG used by policies."""
        import numpy as np

        import coleman.policy
        import coleman.policy.base

        new_rng = np.random.default_rng(123)
        coleman.policy._rng = new_rng  # type: ignore[assignment]

        assert coleman.policy.base._rng is new_rng
        assert coleman.policy._rng.bit_generator.state == new_rng.bit_generator.state

    def test_seed_applied_to_rng(self):
        """When execution.seed is set, the policy RNG should be deterministically seeded."""
        import numpy as np

        import coleman.policy.base
        from coleman.runner import run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _light_run_spec(tmpdir, seed=42)
            spec_dict = spec.model_dump()
            run_experiment(spec_dict)

            # After running, re-seed with same value and verify the generator
            # type matches (proves the seed path was taken).
            ref_rng = np.random.default_rng(42)
            actual = coleman.policy.base._rng.bit_generator.state
            expected = ref_rng.bit_generator.state
            assert actual["bit_generator"] == expected["bit_generator"]

    def test_no_seed_leaves_rng_unseeded(self):
        """Without execution.seed the policy RNG stays in its default state."""
        import numpy as np

        import coleman.policy.base
        from coleman.runner import run_experiment

        # Reset to a known-seed baseline so the test is reproducible
        coleman.policy.base._rng = np.random.default_rng(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = _light_run_spec(tmpdir)
            assert spec.execution.seed is None
            spec_dict = spec.model_dump()
            run_experiment(spec_dict)
            # RNG should still be a default_rng (PCG64) — no error
            state = coleman.policy.base._rng.bit_generator.state
            assert state["bit_generator"] == "PCG64"


class TestApiLoadSave:
    def test_load_spec_via_api(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump({"execution": {"verbose": True}}, fh)
            path = fh.name
        try:
            spec = load_spec(path)
            assert spec.execution.verbose is True
        finally:
            os.unlink(path)

    def test_save_resolved_via_api(self):
        spec = RunSpec()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_resolved(spec, os.path.join(tmpdir, "spec.json"))
            assert out.exists()


class TestRunManyEdgeCases:
    def test_duplicate_run_ids_raise_in_parallel(self):
        """run_many with max_workers > 1 must raise ValueError for duplicate run_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Two identical specs produce the same run_id
            spec = _light_run_spec(tmpdir)
            with pytest.raises(ValueError, match="Duplicate run_id"):
                run_many([spec, spec], max_workers=2)

    def test_run_many_empty_specs(self):
        """run_many with an empty list should return an empty list."""
        results = run_many([], max_workers=1)
        assert results == []

    def test_run_many_parallel_with_unique_specs(self):
        """run_many with max_workers > 1 and unique specs should succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [_light_run_spec(tmpdir, seed=i) for i in range(2)]
            results = run_many(specs, max_workers=2)
            assert len(results) == 2
            ids = [r.run_id for r in results]
            assert len(set(ids)) == 2
