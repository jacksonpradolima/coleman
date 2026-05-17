"""Tests for pure state update helpers used by agent implementations."""

from __future__ import annotations

import numpy as np

from coleman.agent.state_ops import (
    add_attempt_weights,
    incremental_q_update,
    map_names_to_indices,
    scatter_rewards,
)


class TestMapNamesToIndices:
    def test_maps_known_names_and_uses_minus_one_for_unknowns(self):
        name_to_idx = {"a": 2, "b": 4}

        result = map_names_to_indices(["a", "x", "b"], name_to_idx)

        assert result.dtype == np.intp
        np.testing.assert_array_equal(result, np.array([2, -1, 4], dtype=np.intp))

    def test_respects_limit_and_can_return_empty_array(self):
        name_to_idx = {"a": 1}

        limited = map_names_to_indices(["a", "b", "c"], name_to_idx, limit=2)
        empty = map_names_to_indices(["a"], name_to_idx, limit=0)

        np.testing.assert_array_equal(limited, np.array([1, -1], dtype=np.intp))
        assert empty.size == 0


class TestAddAttemptWeights:
    def test_adds_only_for_valid_indices(self):
        attempts = np.array([0.0, 0.0, 0.0])
        indices = np.array([1, -1, 1], dtype=np.intp)
        weights = np.array([0.5, 10.0, 2.0])

        add_attempt_weights(attempts, indices, weights)

        np.testing.assert_allclose(attempts, np.array([0.0, 2.5, 0.0]))

    def test_no_valid_indices_leaves_attempts_unchanged(self):
        attempts = np.array([1.0, 2.0])
        indices = np.array([-1, -1], dtype=np.intp)
        weights = np.array([3.0, 4.0])

        add_attempt_weights(attempts, indices, weights)

        np.testing.assert_allclose(attempts, np.array([1.0, 2.0]))


class TestIncrementalQUpdate:
    def test_ignores_rows_with_only_invalid_indices(self):
        values = np.array([1.0, 2.0])
        attempts = np.array([1, 1], dtype=np.intp)
        indices = np.array([-1, -1], dtype=np.intp)
        rewards = np.array([10.0, 20.0])

        incremental_q_update(values, attempts, indices, rewards)

        np.testing.assert_allclose(values, np.array([1.0, 2.0]))

    def test_ignores_indices_without_positive_attempts(self):
        values = np.array([1.0, 2.0, 3.0])
        attempts = np.array([0, 0, 0], dtype=np.intp)
        indices = np.array([0, 2], dtype=np.intp)
        rewards = np.array([5.0, 7.0])

        incremental_q_update(values, attempts, indices, rewards)

        np.testing.assert_allclose(values, np.array([1.0, 2.0, 3.0]))

    def test_updates_values_incrementally_for_valid_indices(self):
        values = np.array([1.0, 2.0, 3.0])
        attempts = np.array([1, 2, 1], dtype=np.intp)
        indices = np.array([1, 2, -1], dtype=np.intp)
        rewards = np.array([4.0, 1.0, 9.0])

        incremental_q_update(values, attempts, indices, rewards)

        np.testing.assert_allclose(values, np.array([1.0, 3.0, 1.0]))


class TestScatterRewards:
    def test_scatters_valid_rewards_and_ignores_invalid_indices(self):
        result = scatter_rewards(4, np.array([0, -1, 2], dtype=np.intp), np.array([1.5, 9.0, 3.0]))

        np.testing.assert_allclose(result, np.array([1.5, 0.0, 3.0, 0.0]))

    def test_returns_zero_array_when_no_indices_are_valid(self):
        result = scatter_rewards(3, np.array([-1, -1], dtype=np.intp), np.array([1.0, 2.0]))

        np.testing.assert_allclose(result, np.zeros(3))
