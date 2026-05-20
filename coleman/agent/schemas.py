"""Schema constants for agent DataFrames."""

import polars as pl

#: Schema for the actions DataFrame shared by all agents.
#: Columns: Name (test-case id), ActionAttempts (weighted selection count),
#: ValueEstimates (accumulated reward), Q (policy quality estimate).
ACTIONS_SCHEMA: dict = {
    "Name": pl.Utf8,
    "ActionAttempts": pl.Float64,
    "ValueEstimates": pl.Float64,
    "Q": pl.Float64,
}

#: Schema for the sliding-window history DataFrame.
#: Extends ACTIONS_SCHEMA with T (time / build step).
HISTORY_SCHEMA: dict = {
    "Name": pl.Utf8,
    "ActionAttempts": pl.Float64,
    "ValueEstimates": pl.Float64,
    "Q": pl.Float64,
    "T": pl.Int64,
}
