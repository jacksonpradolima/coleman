"""
coleman.checkpoint.state - Typed Payload Structures for Checkpoints.

Defines the data classes that capture the minimal state needed to resume an
experiment from a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckpointPayload:
    """Immutable snapshot of experiment state at a given step.

    Parameters
    ----------
    run_id : str
        Unique identifier for the experiment run.
    experiment : int
        Experiment number.
    step : int
        Last committed step.
    agents : Any
        Serializable agent state (list of agents).
    monitor : Any
        Serializable monitor state.
    variant_monitors : dict[str, Any]
        Serializable variant-monitor state.
    bandit : Any
        Serializable bandit state (may be ``None`` for the first step).
    extra : dict[str, Any]
        Arbitrary additional state.
    """

    run_id: str = ""
    experiment: int = 0
    step: int = 0
    agents: Any = None
    monitor: Any = None
    variant_monitors: dict[str, Any] = field(default_factory=dict)
    bandit: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
