"""
coleman.checkpoint.checkpoint_store - Checkpoint Store Interface and Implementations.

Provides ``CheckpointStore`` (abstract), ``LocalCheckpointStore`` (default,
file-system based) and ``NullCheckpointStore`` (no-op for when checkpoints are
disabled).
"""

from __future__ import annotations

import abc
import json
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

from coleman.checkpoint.state import CheckpointPayload

logger = logging.getLogger(__name__)


class CheckpointStore(abc.ABC):
    """Abstract interface for checkpoint persistence.

    Methods
    -------
    save(payload)
        Persist a checkpoint payload.
    load(run_id, experiment)
        Load the latest checkpoint for a run.
    """

    @abc.abstractmethod
    def save(self, payload: CheckpointPayload) -> None:
        """Persist a checkpoint payload.

        Parameters
        ----------
        payload : CheckpointPayload
            The state to persist.
        """

    @abc.abstractmethod
    def load(self, run_id: str, experiment: int) -> CheckpointPayload | None:
        """Load the latest checkpoint for a given run/experiment.

        Parameters
        ----------
        run_id : str
            Run identifier.
        experiment : int
            Experiment number.

        Returns
        -------
        CheckpointPayload or None
            The loaded payload, or ``None`` if no checkpoint exists.
        """


class NullCheckpointStore(CheckpointStore):
    """No-op checkpoint store used when checkpoints are disabled."""

    def save(self, payload: CheckpointPayload) -> None:
        """Discard the payload (no-op).

        Parameters
        ----------
        payload : CheckpointPayload
            Ignored.
        """

    def load(self, run_id: str, experiment: int) -> CheckpointPayload | None:
        """Return ``None`` (no checkpoint available).

        Parameters
        ----------
        run_id : str
            Ignored.
        experiment : int
            Ignored.

        Returns
        -------
        CheckpointPayload or None
            Always ``None``.
        """
        return None


class LocalCheckpointStore(CheckpointStore):
    """File-system based checkpoint store.

    Stores checkpoint payloads as pickle files and maintains an atomic
    ``progress.json`` marker for quick discovery.

    Parameters
    ----------
    base_dir : str
        Root directory for checkpoint storage.

    Attributes
    ----------
    base_dir : str
        Root directory.
    """

    def __init__(self, base_dir: str = "checkpoints") -> None:
        """Initialise the store and create *base_dir* if it does not exist.

        Parameters
        ----------
        base_dir : str
            Root directory for checkpoint storage.
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self, payload: CheckpointPayload) -> None:
        """Persist a checkpoint payload atomically.

        Writes the pickle payload first, then atomically updates
        ``progress.json``.

        Parameters
        ----------
        payload : CheckpointPayload
            The state to persist.
        """
        run_dir = os.path.join(self.base_dir, payload.run_id)
        os.makedirs(run_dir, exist_ok=True)

        ckpt_name = f"ckpt_ex{payload.experiment}_step{payload.step}.pkl"
        ckpt_path = os.path.join(run_dir, ckpt_name)

        # Write pickle payload
        with open(ckpt_path, "wb") as f:
            pickle.dump(payload, f)

        # Atomic progress.json update (write-rename)
        progress = {
            "run_id": payload.run_id,
            "experiment": payload.experiment,
            "step_committed": payload.step,
            "checkpoint_path": ckpt_path,
            "timestamp": time.time(),
        }
        progress_path = os.path.join(run_dir, f"progress_ex{payload.experiment}.json")
        self._atomic_json_write(progress_path, progress)

        logger.debug(
            "Checkpoint saved: run=%s exp=%s step=%s",
            payload.run_id,
            payload.experiment,
            payload.step,
        )

    def load(self, run_id: str, experiment: int) -> CheckpointPayload | None:
        """Load the latest checkpoint for a run/experiment.

        Parameters
        ----------
        run_id : str
            Run identifier.
        experiment : int
            Experiment number.

        Returns
        -------
        CheckpointPayload or None
            Loaded payload or ``None`` if not found.
        """
        run_dir = os.path.join(self.base_dir, run_id)
        progress_path = os.path.join(run_dir, f"progress_ex{experiment}.json")

        if not os.path.exists(progress_path):
            return None

        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)

        # Ensure the checkpoint is for the requested experiment
        if progress.get("experiment") != experiment:
            return None

        ckpt_path = progress.get("checkpoint_path", "")
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint file missing: %s", ckpt_path)
            return None

        with open(ckpt_path, "rb") as f:
            payload: CheckpointPayload = pickle.load(f)  # noqa: S301

        return payload

    @staticmethod
    def _atomic_json_write(path: str, data: dict[str, Any]) -> None:
        """Write JSON atomically via write-to-temp + rename.

        Parameters
        ----------
        path : str
            Target file path.
        data : dict
            JSON-serializable data.
        """
        dir_name = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            Path(tmp_path).unlink(missing_ok=True)
            raise
