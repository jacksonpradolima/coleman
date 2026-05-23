"""Tests for built-in analysis reports."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from coleman.analysis import _safe_attach_alias, _sql_quote_literal, format_rows, run_report


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


def test_run_report_pareto_from_parquet(tmp_path):
    _build_parquet_fixture(tmp_path)

    columns, rows = run_report("pareto", tmp_path)

    assert columns == ["policy", "reward_function", "avg_napfd", "avg_apfdc"]
    assert rows


def test_run_report_unsupported_report_raises(tmp_path):
    _build_parquet_fixture(tmp_path)
    with pytest.raises(ValueError, match="Unsupported report"):
        run_report("does-not-exist", tmp_path)


def test_format_rows_unsupported_output_format_raises():
    with pytest.raises(ValueError, match="Unsupported output format"):
        format_rows(["a"], [(1,)], "json")


def test_run_report_missing_input_path_raises(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Input path does not exist"):
        run_report("quality", missing)


def test_run_report_no_supported_files_raises(tmp_path):
    root = tmp_path / "empty"
    root.mkdir(parents=True)
    (root / "README.txt").write_text("no results", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No Parquet or DuckDB result files"):
        run_report("quality", root)


def test_run_report_quality_from_duckdb_with_quoted_path(tmp_path):
    duckdb_dir = tmp_path / "o'hara"
    duckdb_dir.mkdir(parents=True)
    db_path = duckdb_dir / "sample.duckdb"
    con = duckdb.connect(db_path.as_posix())
    con.execute(
        """
        CREATE TABLE coleman_results AS
        SELECT *
        FROM (
            VALUES
                (
                    's1', 1, 1, 'e1', '1', 'sequential', 'Random', 'RNFail',
                    'ratio', 0.5,
                    1.0, 0.1, 1.0, 1.0, 10.0, 1.0, 1.0,
                    1, 0, 10, 0, 1.0, 1.0, 0.0, 0.80, 0.20, 0.5, 0.5,
                    'h1', '', NULL
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
    con.close()

    columns, rows = run_report("quality", duckdb_dir)
    assert columns[:2] == ["policy", "reward_function"]
    assert rows[0][0] == "Random"


def test_format_rows_table_and_csv_render_expected_escaping():
    rows = [("a|b", 1.25, None), ('x,"y"', 2.0, "line\nnext")]
    markdown = format_rows(["name", "score", "extra"], rows, "markdown")
    csv_text = format_rows(["name", "score", "extra"], rows, "csv")
    table = format_rows(["name", "score", "extra"], rows, "table")

    assert "a\\|b" in markdown
    assert '"x,""y"""' in csv_text
    assert "line\nnext" in csv_text
    assert "1.250000" in table


def test_safe_attach_alias_and_sql_literal_helpers_cover_edge_cases():
    assert _safe_attach_alias(0) == "db_0"
    with pytest.raises(ValueError, match="Unsafe attach alias"):
        _safe_attach_alias(-1)

    assert _sql_quote_literal("o'hara") == "'o''hara'"
