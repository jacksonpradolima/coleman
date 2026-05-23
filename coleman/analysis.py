"""Built-in post-run analysis reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

_REPORT_QUERIES: dict[str, str] = {
    "quality": """
        SELECT
            policy,
            reward_function,
            AVG(fitness) AS avg_napfd,
            STDDEV_SAMP(fitness) AS std_napfd,
            COUNT(*) AS n
        FROM experiment_results
        GROUP BY policy, reward_function
        ORDER BY avg_napfd DESC, policy, reward_function
    """,
    "cost": """
        SELECT
            policy,
            reward_function,
            AVG(cost) AS avg_apfdc,
            STDDEV_SAMP(cost) AS std_apfdc,
            COUNT(*) AS n
        FROM experiment_results
        GROUP BY policy, reward_function
        ORDER BY avg_apfdc ASC, policy, reward_function
    """,
    "stability": """
        SELECT
            policy,
            reward_function,
            AVG(fitness) AS avg_napfd,
            STDDEV_SAMP(fitness) AS std_napfd,
            CASE
                WHEN AVG(fitness) = 0 THEN NULL
                ELSE STDDEV_SAMP(fitness) / AVG(fitness)
            END AS cv_napfd
        FROM experiment_results
        GROUP BY policy, reward_function
        ORDER BY cv_napfd ASC NULLS LAST, policy, reward_function
    """,
    "sensitivity": """
        SELECT
            scenario,
            policy,
            reward_function,
            AVG(fitness) AS avg_napfd,
            AVG(cost) AS avg_apfdc,
            AVG(prioritization_time) AS avg_prioritization_time
        FROM experiment_results
        GROUP BY scenario, policy, reward_function
        ORDER BY scenario, avg_napfd DESC, policy, reward_function
    """,
    "resources": """
        SELECT
            policy,
            reward_function,
            AVG(prioritization_time) AS avg_prioritization_time,
            AVG(process_memory_rss_mib) AS avg_rss_mib,
            AVG(process_cpu_utilization_percent) AS avg_cpu_pct,
            AVG(fitness) AS avg_napfd
        FROM experiment_results
        GROUP BY policy, reward_function
        ORDER BY avg_prioritization_time ASC, policy, reward_function
    """,
}


def run_report(report_name: str, input_path: str | Path) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Run a built-in analysis report and return tabular results."""
    report_name = report_name.lower()
    if report_name == "pareto":
        return _run_pareto(input_path)

    query = _REPORT_QUERIES.get(report_name)
    if query is None:
        msg = f"Unsupported report {report_name!r}."
        raise ValueError(msg)

    with duckdb.connect(":memory:") as con:
        _register_experiment_results(con, input_path)
        result = con.execute(query)
        columns = [d[0] for d in result.description]
        rows = result.fetchall()
    return columns, rows


def format_rows(columns: list[str], rows: list[tuple[Any, ...]], output_format: str) -> str:
    """Format report rows as table, CSV, or Markdown."""
    output_format = output_format.lower()
    if output_format == "csv":
        return _to_csv(columns, rows)
    if output_format == "markdown":
        return _to_markdown(columns, rows)
    if output_format == "table":
        return _to_table(columns, rows)
    msg = f"Unsupported output format {output_format!r}."
    raise ValueError(msg)


def _run_pareto(input_path: str | Path) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Compute non-dominated policy/reward pairs for quality-vs-cost trade-offs."""
    with duckdb.connect(":memory:") as con:
        _register_experiment_results(con, input_path)
        rows = con.execute(
            """
            WITH summary AS (
                SELECT
                    policy,
                    reward_function,
                    AVG(fitness) AS avg_napfd,
                    AVG(cost) AS avg_apfdc
                FROM experiment_results
                GROUP BY policy, reward_function
            ),
            dominated AS (
                SELECT
                    s1.policy,
                    s1.reward_function
                FROM summary s1
                JOIN summary s2
                  ON (s2.avg_napfd >= s1.avg_napfd AND s2.avg_apfdc <= s1.avg_apfdc)
                 AND (s2.avg_napfd > s1.avg_napfd OR s2.avg_apfdc < s1.avg_apfdc)
                GROUP BY s1.policy, s1.reward_function
            )
            SELECT
                s.policy,
                s.reward_function,
                s.avg_napfd,
                s.avg_apfdc
            FROM summary s
            LEFT JOIN dominated d
              ON s.policy = d.policy
             AND s.reward_function = d.reward_function
            WHERE d.policy IS NULL
            ORDER BY s.avg_apfdc ASC, s.avg_napfd DESC, s.policy, s.reward_function
            """
        ).fetchall()
    return ["policy", "reward_function", "avg_napfd", "avg_apfdc"], rows


def _register_experiment_results(con: duckdb.DuckDBPyConnection, input_path: str | Path) -> None:
    """Register ``experiment_results`` view from Parquet or DuckDB inputs."""
    root = Path(input_path)
    if not root.exists():
        msg = f"Input path does not exist: {root}"
        raise FileNotFoundError(msg)

    parquet_files = sorted(root.rglob("*.parquet")) if root.is_dir() else [root] if root.suffix == ".parquet" else []
    duckdb_files = sorted(root.rglob("*.duckdb")) if root.is_dir() else [root] if root.suffix == ".duckdb" else []

    if parquet_files:
        glob_path = (root / "**" / "*.parquet").as_posix() if root.is_dir() else root.as_posix()
        safe_glob_path = glob_path.replace("'", "''")
        con.execute(
            f"""
            CREATE OR REPLACE VIEW experiment_results AS
            SELECT *
            FROM read_parquet('{safe_glob_path}', hive_partitioning=1)
            """
        )
        return

    if duckdb_files:
        union_parts: list[str] = []
        for idx, db_path in enumerate(duckdb_files):
            alias = f"db_{idx}"
            con.execute(f"ATTACH '{db_path.as_posix()}' AS {alias}")
            union_parts.append(f"SELECT * FROM {alias}.coleman_results")
        con.execute(f"CREATE OR REPLACE VIEW experiment_results AS {' UNION ALL '.join(union_parts)}")
        return

    msg = f"No Parquet or DuckDB result files found under {root}"
    raise FileNotFoundError(msg)


def _to_csv(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Render tabular report output as CSV text."""
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_format_csv_value(value) for value in row))
    return "\n".join(lines) + "\n"


def _to_markdown(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Render tabular report output as Markdown table text."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(_format_value(value) for value in row) + " |" for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def _to_table(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Render tabular report output as fixed-width plain text table."""
    widths = [len(c) for c in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(_format_value(value)))

    def _fmt_line(values: list[str]) -> str:
        """Render one aligned table row using computed column widths."""
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    header = _fmt_line(columns)
    sep = "-+-".join("-" * w for w in widths)
    body = [_fmt_line([_format_value(value) for value in row]) for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def _format_value(value: Any) -> str:
    """Format one scalar value for report rendering."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_csv_value(value: Any) -> str:
    """Format one scalar value with CSV escaping when needed."""
    text = _format_value(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text
