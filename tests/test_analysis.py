"""Tests for built-in analysis reports."""

from __future__ import annotations

from pathlib import Path

import duckdb

from coleman.analysis import format_rows, run_report


def _build_parquet_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    file_path = root / "sample.parquet"

    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE t AS
        SELECT *
        FROM (
            VALUES
                (
                    's1', 1, 1, 'e1', '1', 'sequential', 'Random', 'RNFail',
                    'ratio', 0.5,
                    1.0, 0.1, 1.0, 1.0, 10.0, 1.0, 1.0,
                    1, 0, 10, 0, 1.0, 1.0, 0.0, 0.70, 0.30, 0.5, 0.5,
                    'h1', '', NULL
                ),
                (
                    's1', 1, 2, 'e2', '2', 'sequential', 'UCB1', 'RNFail',
                    'ratio', 0.5,
                    1.0, 0.2, 2.0, 2.0, 20.0, 2.0, 2.0,
                    1, 0, 10, 0, 1.0, 1.0, 0.0, 0.90, 0.10, 0.7, 0.7,
                    'h2', '', NULL
                )
        ) AS x(
            scenario, experiment, step, execution_id, worker_id, parallel_mode,
            policy, reward_function, budget_mode, budget_value,
            total_build_duration, prioritization_time, process_memory_rss_mib,
            process_memory_peak_rss_mib, process_cpu_utilization_percent,
            process_cpu_time_seconds, wall_time_seconds, detected, missed,
            tests_ran, tests_not_ran, ttf, ttf_duration, time_reduction,
            fitness, cost, rewards, avg_precision, prioritization_order_hash,
            prioritization_order_top_k, variant
        )
        """
    )
    con.execute("COPY t TO ? (FORMAT PARQUET)", [str(file_path)])


def test_run_report_quality_from_parquet(tmp_path):
    _build_parquet_fixture(tmp_path)

    columns, rows = run_report("quality", tmp_path)

    assert columns[:3] == ["policy", "reward_function", "avg_napfd"]
    assert rows[0][0] == "UCB1"


def test_format_rows_markdown():
    text = format_rows(["a", "b"], [(1, 2), (3, 4)], "markdown")
    assert "| a | b |" in text
    assert "| 1 | 2 |" in text
