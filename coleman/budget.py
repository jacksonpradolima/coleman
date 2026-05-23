"""Shared budget mode types used by specs, runner, and results."""

from __future__ import annotations

from enum import StrEnum


class BudgetMode(StrEnum):
    """Supported experiment budget semantics."""

    RATIO = "ratio"
    FIXED_TIME = "fixed_time"
    SUBSET_SIZE = "subset_size"
