"""Tests for the checkpoint store interface and implementations."""

import json
from unittest.mock import patch

from coleman.checkpoint.checkpoint_store import CheckpointStore, LocalCheckpointStore, NullCheckpointStore
from coleman.checkpoint.state import CheckpointPayload

# ============================================================================
# NullCheckpointStore
# ============================================================================


class TestNullCheckpointStore:
    def test_is_checkpoint_store(self):
        assert isinstance(NullCheckpointStore(), CheckpointStore)

    def test_save_is_noop(self):
        store = NullCheckpointStore()
        payload = CheckpointPayload(run_id="r1", experiment=1, step=10)
        store.save(payload)  # should not raise

    def test_load_returns_none(self):
        store = NullCheckpointStore()
        assert store.load("r1", 1) is None


# ============================================================================
# CheckpointPayload
# ============================================================================


class TestCheckpointPayload:
    def test_defaults(self):
        p = CheckpointPayload()
        assert p.run_id == ""
        assert p.experiment == 0
        assert p.step == 0
        assert p.agents is None
        assert p.monitor is None
        assert p.variant_monitors == {}
        assert p.bandit is None
        assert p.extra == {}

    def test_custom_values(self):
        p = CheckpointPayload(run_id="test", experiment=3, step=42, agents=["a1"])
        assert p.run_id == "test"
        assert p.experiment == 3
        assert p.step == 42
        assert p.agents == ["a1"]


# ============================================================================
# LocalCheckpointStore
# ============================================================================


class TestLocalCheckpointStore:
    def test_save_creates_files(self, tmp_path):
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        payload = CheckpointPayload(run_id="run1", experiment=1, step=100, agents=["agent1"])
        store.save(payload)

        run_dir = tmp_path / "ckpts" / "run1"
        assert run_dir.exists()
        assert (run_dir / "progress_ex1.json").exists()
        assert (run_dir / "ckpt_ex1_step100.pkl").exists()

    def test_progress_json_content(self, tmp_path):
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        payload = CheckpointPayload(run_id="run1", experiment=1, step=100)
        store.save(payload)

        progress_path = tmp_path / "ckpts" / "run1" / "progress_ex1.json"
        with open(progress_path) as f:
            progress = json.load(f)

        assert progress["run_id"] == "run1"
        assert progress["experiment"] == 1
        assert progress["step_committed"] == 100
        assert "checkpoint_path" in progress
        assert "timestamp" in progress

    def test_load_roundtrip(self, tmp_path):
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        payload = CheckpointPayload(run_id="run1", experiment=2, step=500, agents=["a1", "a2"], extra={"key": "value"})
        store.save(payload)

        loaded = store.load("run1", 2)
        assert loaded is not None
        assert loaded.run_id == "run1"
        assert loaded.experiment == 2
        assert loaded.step == 500
        assert loaded.agents == ["a1", "a2"]
        assert loaded.extra == {"key": "value"}

    def test_load_nonexistent_returns_none(self, tmp_path):
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        assert store.load("nonexistent", 1) is None

    def test_load_wrong_experiment_returns_none(self, tmp_path):
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        payload = CheckpointPayload(run_id="run1", experiment=1, step=100)
        store.save(payload)

        # Keep the same progress file path (progress_ex1.json), but force payload mismatch.
        progress_path = tmp_path / "ckpts" / "run1" / "progress_ex1.json"
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        progress["experiment"] = 2
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f)

        assert store.load("run1", 1) is None

    def test_overwrite_checkpoint(self, tmp_path):
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))

        store.save(CheckpointPayload(run_id="run1", experiment=1, step=100, agents=["old"]))
        store.save(CheckpointPayload(run_id="run1", experiment=1, step=200, agents=["new"]))

        loaded = store.load("run1", 1)
        assert loaded is not None
        assert loaded.step == 200
        assert loaded.agents == ["new"]

    def test_atomic_json_write(self, tmp_path):
        """Verify atomic write produces valid JSON."""
        path = str(tmp_path / "test.json")
        LocalCheckpointStore._atomic_json_write(path, {"key": "value"})

        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}

    def test_load_missing_pickle_returns_none(self, tmp_path):
        """Verify load returns None when progress.json exists but pickle file is missing."""
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        payload = CheckpointPayload(run_id="run1", experiment=1, step=100)
        store.save(payload)

        # Remove the pickle file
        run_dir = tmp_path / "ckpts" / "run1"
        for f in run_dir.glob("*.pkl"):
            f.unlink()

        assert store.load("run1", 1) is None

    def test_load_progress_with_missing_checkpoint_path_returns_none(self, tmp_path):
        """If progress points to a missing checkpoint, load() must return None."""
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        payload = CheckpointPayload(run_id="run1", experiment=1, step=100)
        store.save(payload)

        progress_path = tmp_path / "ckpts" / "run1" / "progress_ex1.json"
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        progress["checkpoint_path"] = str(tmp_path / "ckpts" / "run1" / "does_not_exist.pkl")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f)

        assert store.load("run1", 1) is None

    def test_load_missing_checkpoint_emits_warning(self, tmp_path):
        """Cover warning line for missing checkpoint file (line 188)."""
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))
        payload = CheckpointPayload(run_id="run1", experiment=1, step=100)
        store.save(payload)

        progress_path = tmp_path / "ckpts" / "run1" / "progress_ex1.json"
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        progress["checkpoint_path"] = str(tmp_path / "missing.pkl")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f)

        with patch("coleman.checkpoint.checkpoint_store.logger.warning") as warn:
            assert store.load("run1", 1) is None
        warn.assert_called_once()

    def test_multi_experiment_isolation(self, tmp_path):
        """Multiple experiments for the same run_id must not overwrite each other."""
        store = LocalCheckpointStore(base_dir=str(tmp_path / "ckpts"))

        store.save(CheckpointPayload(run_id="run1", experiment=1, step=50, agents=["a1"]))
        store.save(CheckpointPayload(run_id="run1", experiment=2, step=75, agents=["a2"]))

        loaded1 = store.load("run1", 1)
        loaded2 = store.load("run1", 2)

        assert loaded1 is not None
        assert loaded1.experiment == 1
        assert loaded1.step == 50
        assert loaded1.agents == ["a1"]

        assert loaded2 is not None
        assert loaded2.experiment == 2
        assert loaded2.step == 75
        assert loaded2.agents == ["a2"]

    def test_atomic_json_write_cleans_tmp_on_exception(self, tmp_path):
        """When os.replace fails, temporary file should be unlinked and error re-raised."""
        target = str(tmp_path / "target.json")

        with (
            patch("coleman.checkpoint.checkpoint_store.os.replace", side_effect=OSError("replace failed")),
            patch("coleman.checkpoint.checkpoint_store.Path.unlink") as unlink_mock,
        ):
            try:
                LocalCheckpointStore._atomic_json_write(target, {"ok": True})
            except OSError:
                pass
            else:
                raise AssertionError("Expected OSError from os.replace")

        unlink_mock.assert_called_once()
