"""Pure state update functions for agent numeric arrays.

These helpers isolate hot-path numeric transforms from agent classes so they
can be reused across different agent variants.
"""

from __future__ import annotations

import numpy as np


def map_names_to_indices(
    names: list[str],
    name_to_idx: dict[str, int],
    *,
    limit: int | None = None,
) -> np.ndarray:
    """Map action names to indices, using -1 for unknown names."""
    end = len(names) if limit is None else min(limit, len(names))
    if end <= 0:
        return np.empty(0, dtype=np.intp)

    return np.fromiter((name_to_idx.get(nm, -1) for nm in names[:end]), dtype=np.intp, count=end)


def add_attempt_weights(
    attempts: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Apply weighted attempt increments in-place for valid indices."""
    valid = indices >= 0
    if valid.any():
        np.add.at(attempts, indices[valid], weights[: len(indices)][valid])


def incremental_q_update(
    values: np.ndarray,
    attempts: np.ndarray,
    indices: np.ndarray,
    rewards: np.ndarray,
) -> None:
    """Apply in-place incremental Q update on selected indices."""
    valid = indices >= 0
    if not valid.any():
        return

    idx = indices[valid]
    rew = rewards[: len(indices)][valid]
    k = attempts[idx]
    positive = k > 0
    if not positive.any():
        return

    idx_pos = idx[positive]
    k_pos = k[positive]
    rew_pos = rew[positive]
    values[idx_pos] += (1.0 / k_pos) * (rew_pos - values[idx_pos])


def scatter_rewards(size: int, indices: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """Create a dense value-estimate array by scattering rewards to indices."""
    out = np.zeros(size, dtype=np.float64)
    valid = indices >= 0
    if valid.any():
        out[indices[valid]] = rewards[: len(indices)][valid]
    return out
