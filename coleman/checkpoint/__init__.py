"""
coleman.checkpoint - Checkpoint and Recovery.

This subpackage provides checkpoint persistence so that long-running
experiments can be resumed from the last safe state after a crash.

Default: ``LocalCheckpointStore`` (JSON progress marker + pickle payload).
When disabled: ``NullCheckpointStore`` (no-op).
"""

from coleman.checkpoint.checkpoint_store import CheckpointStore, LocalCheckpointStore, NullCheckpointStore

__all__ = ["CheckpointStore", "LocalCheckpointStore", "NullCheckpointStore"]
