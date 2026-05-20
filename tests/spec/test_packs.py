"""Tests for config pack resolution."""

import contextlib

import pytest
import yaml

from coleman.spec.packs import _deep_merge, resolve_packs


class TestDeepMerge:
    def test_simple_merge(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_scalar(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        over = {"a": {"c": 3, "d": 4}}
        result = _deep_merge(base, over)
        assert result == {"a": {"b": 1, "c": 3, "d": 4}}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        over = {"a": {"c": 2}}
        _deep_merge(base, over)
        assert base == {"a": {"b": 1}}


class TestResolvePacks:
    def test_no_packs(self):
        raw = {"execution": {"verbose": True}}
        result = resolve_packs(raw, packs_dir="/nonexistent")
        assert result == {"execution": {"verbose": True}}

    def test_pack_file_not_found(self, tmp_path):
        raw = {"packs": ["missing/pack"]}
        packs_dir = str(tmp_path / "no_such_dir")
        with pytest.raises(FileNotFoundError, match="Pack file not found"):
            resolve_packs(raw, packs_dir=packs_dir)

    def test_single_pack(self, tmp_path):
        policy_dir = tmp_path / "policy"
        policy_dir.mkdir()
        pack_path = policy_dir / "ucb.yaml"
        pack_path.write_text(yaml.dump({"experiment": {"policies": ["UCB"]}}))

        raw = {"packs": ["policy/ucb"], "experiment": {"rewards": ["RNFail"]}}
        result = resolve_packs(raw, packs_dir=str(tmp_path))

        assert result["experiment"]["policies"] == ["UCB"]
        assert result["experiment"]["rewards"] == ["RNFail"]
        assert "packs" not in result

    def test_multiple_packs_merge_order(self, tmp_path):
        (tmp_path / "policy").mkdir()
        (tmp_path / "results").mkdir()

        (tmp_path / "policy" / "greedy.yaml").write_text(yaml.dump({"experiment": {"policies": ["Greedy"]}}))
        (tmp_path / "results" / "parquet.yaml").write_text(yaml.dump({"results": {"sink": "parquet", "enabled": True}}))

        raw = {"packs": ["policy/greedy", "results/parquet"]}
        result = resolve_packs(raw, packs_dir=str(tmp_path))

        assert result["experiment"]["policies"] == ["Greedy"]
        assert result["results"]["sink"] == "parquet"

    def test_inline_overrides_win(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "parquet.yaml").write_text(
            yaml.dump({"results": {"sink": "parquet", "batch_size": 500}})
        )

        raw = {"packs": ["results/parquet"], "results": {"batch_size": 2000}}
        result = resolve_packs(raw, packs_dir=str(tmp_path))

        # Pack sets sink, but inline overrides batch_size
        assert result["results"]["sink"] == "parquet"
        assert result["results"]["batch_size"] == 2000

    def test_does_not_mutate_input(self, tmp_path):
        raw = {"packs": ["missing"], "execution": {"verbose": True}}
        original = dict(raw)
        # Even though the pack doesn't exist, the input should not be mutated
        # by the time the error is raised.
        packs_dir = str(tmp_path / "no_such_dir")
        with contextlib.suppress(FileNotFoundError):
            resolve_packs(raw, packs_dir=packs_dir)
        assert raw == original

    def test_packs_string_raises_type_error(self):
        raw = {"packs": "policy/linucb"}
        with pytest.raises(TypeError, match="must be a list of strings"):
            resolve_packs(raw, packs_dir="/nonexistent")

    def test_packs_none_raises_type_error(self):
        raw = {"packs": None}
        with pytest.raises(TypeError, match="must be a list of strings"):
            resolve_packs(raw, packs_dir="/nonexistent")

    def test_packs_non_string_item_raises_type_error(self):
        raw = {"packs": [123]}
        with pytest.raises(TypeError, match="must be a string"):
            resolve_packs(raw, packs_dir="/nonexistent")

    def test_packs_empty_string_raises_value_error(self):
        raw = {"packs": [""]}
        with pytest.raises(ValueError, match="non-empty string"):
            resolve_packs(raw, packs_dir="/nonexistent")
