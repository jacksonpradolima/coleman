"""Tests for the sweep engine."""

import pytest

from coleman.spec.models import RunSpec
from coleman.spec.run_id import compute_run_id
from coleman.spec.sweep import SweepAxis, SweepSpec, _expand_axis, _set_nested, expand_sweep


class TestSetNested:
    def test_simple_key(self):
        d: dict = {}
        _set_nested(d, "key", 42)
        assert d == {"key": 42}

    def test_nested_key(self):
        d: dict = {}
        _set_nested(d, "a.b.c", "val")
        assert d == {"a": {"b": {"c": "val"}}}

    def test_overwrite_existing(self):
        d = {"a": {"b": 1}}
        _set_nested(d, "a.b", 2)
        assert d["a"]["b"] == 2

    def test_non_dict_intermediate_raises(self):
        d = {"a": "not_a_dict"}
        with pytest.raises(ValueError, match="is not a mapping"):
            _set_nested(d, "a.b.c", 42)

    def test_initial_data_not_dict_with_multiple_parts_raises(self):
        """Covers the guard at the top of the for-loop (lines 73-79)."""
        with pytest.raises(ValueError, match="is not a mapping"):
            _set_nested("not_a_dict", "a.b", 42)

    def test_initial_data_not_dict_single_part_raises(self):
        """Covers the post-loop guard (lines 98-104)."""
        with pytest.raises(ValueError, match="is not a mapping"):
            _set_nested("not_a_dict", "key", 42)


class TestExpandAxis:
    def test_grid_single_param(self):
        axis = SweepAxis(mode="grid", params={"alpha": [0.1, 0.3, 0.5]})
        result = _expand_axis(axis)
        assert len(result) == 3
        assert result[0] == pytest.approx({"alpha": 0.1})
        assert result[2] == pytest.approx({"alpha": 0.5})

    def test_grid_two_params(self):
        axis = SweepAxis(mode="grid", params={"alpha": [0.1, 0.3], "seed": [0, 1, 2]})
        result = _expand_axis(axis)
        assert len(result) == 6  # 2 x 3

    def test_zip_mode(self):
        axis = SweepAxis(mode="zip", params={"window": [10, 50, 100], "decay": [0.9, 0.95, 0.99]})
        result = _expand_axis(axis)
        assert len(result) == 3
        assert result[0] == pytest.approx({"decay": 0.9, "window": 10})
        assert result[2] == pytest.approx({"decay": 0.99, "window": 100})

    def test_grid_stable_order(self):
        axis = SweepAxis(mode="grid", params={"b": [1, 2], "a": [10, 20]})
        r1 = _expand_axis(axis)
        r2 = _expand_axis(axis)
        assert r1 == r2
        # Sorted by key name: 'a' before 'b'
        assert r1[0] == {"a": 10, "b": 1}

    def test_zip_unequal_lengths_raises(self):
        axis = SweepAxis(mode="zip", params={"a": [1, 2, 3], "b": [10, 20]})
        with pytest.raises(ValueError, match="same length"):
            _expand_axis(axis)

    def test_unknown_mode_raises(self):
        """Covers the else-branch (lines 138-139) for an unknown sweep mode."""
        axis = SweepAxis.model_construct(mode="unknown", params={"key": [1, 2]})
        with pytest.raises(ValueError, match="Unknown sweep mode"):
            _expand_axis(axis)


class TestExpandSweep:
    def test_no_axes_returns_copy(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[])
        result = expand_sweep(base, sweep)
        assert len(result) == 1
        assert compute_run_id(result[0]) == compute_run_id(base)

    def test_grid_expansion(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[SweepAxis(mode="grid", params={"algorithm.ucb.timerank.c": [0.1, 0.3, 0.5]})])
        result = expand_sweep(base, sweep)
        assert len(result) == 3
        assert result[0].algorithm.model_dump()["ucb"]["timerank"]["c"] == pytest.approx(0.1)
        assert result[2].algorithm.model_dump()["ucb"]["timerank"]["c"] == pytest.approx(0.5)

    def test_deterministic_order(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2, 4]})])
        r1 = expand_sweep(base, sweep)
        r2 = expand_sweep(base, sweep)
        ids1 = [compute_run_id(s) for s in r1]
        ids2 = [compute_run_id(s) for s in r2]
        assert ids1 == ids2

    def test_seed_replication(self):
        base = RunSpec()
        seeds = [0, 1, 2]
        sweep = SweepSpec(
            axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2]})],
            seeds=seeds,
        )
        result = expand_sweep(base, sweep)
        # 2 specs x 3 seeds = 6
        assert len(result) == 6

        # Each spec should have an execution.seed taken from the provided seeds
        found_seeds = {spec.execution.seed for spec in result}
        assert found_seeds == set(seeds)

        # We expect the full Cartesian product: each parallel_pool_size with each seed
        combos = {(spec.execution.parallel_pool_size, spec.execution.seed) for spec in result}
        expected_combos = {(pool_size, seed) for pool_size in [1, 2] for seed in seeds}
        assert combos == expected_combos

        # Changing only the seed should change the run_id
        ids_by_pool_size: dict[int, set[str]] = {}
        for spec in result:
            pool_size = spec.execution.parallel_pool_size
            run_id = compute_run_id(spec)
            ids_by_pool_size.setdefault(pool_size, set()).add(run_id)

        for _pool_size, ids in ids_by_pool_size.items():
            # For each pool size, we should have one unique run_id per seed
            assert len(ids) == len(seeds)

    def test_multi_axis_cartesian(self):
        base = RunSpec()
        sweep = SweepSpec(
            axes=[
                SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2]}),
                SweepAxis(mode="grid", params={"results.batch_size": [100, 200]}),
            ]
        )
        result = expand_sweep(base, sweep)
        # 2 x 2 = 4
        assert len(result) == 4

    def test_each_spec_unique_run_id(self):
        base = RunSpec()
        sweep = SweepSpec(axes=[SweepAxis(mode="grid", params={"execution.parallel_pool_size": [1, 2, 4]})])
        result = expand_sweep(base, sweep)
        ids = [compute_run_id(s) for s in result]
        assert len(set(ids)) == 3
