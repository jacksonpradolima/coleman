"""Coleman workflow notebook.

Official marimo notebook covering configuration, code cost evaluation,
observability, resume/recovery, result export, and final analysis.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    """Load notebook dependencies and plotting defaults."""
    import json
    from pathlib import Path

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import yaml

    plt.style.use("ggplot")
    sns.set_style("whitegrid")
    path_class = Path
    return duckdb, json, mo, path_class, pd, plt, sns, yaml


@app.cell
def _(mo):
    """Introduce the end-to-end workflow covered by the notebook."""
    mo.md(
        """
        # Coleman Workflow

        This official notebook covers the full operational loop:

        1. Read active settings from `run.yaml`
        2. Evaluate code cost (structural, runtime, and energy)
        3. Inspect live behavior in Grafana
        4. Inspect checkpoint progress for resume and recovery
        5. Read final experiment results from Parquet
        6. Export compact summaries for reports or papers
        7. Compare policies by quality and resource cost
        """
    )


@app.cell
def _(mo, path_class, pd, yaml):
    """Read the active runtime configuration from run.yaml."""
    config_path = path_class("run.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    results_cfg = config.get("results", {})
    checkpoint_cfg = config.get("checkpoint", {})
    telemetry_cfg = config.get("telemetry", {})
    experiment_cfg = config.get("experiment", {})

    settings_df = pd.DataFrame(
        [
            {"setting": "datasets", "value": ", ".join(experiment_cfg.get("datasets", []))},
            {
                "setting": "scheduled_time_ratio",
                "value": ", ".join(str(value) for value in experiment_cfg.get("scheduled_time_ratio", [])),
            },
            {"setting": "results.enabled", "value": str(results_cfg.get("enabled", False))},
            {"setting": "results.sink", "value": results_cfg.get("sink", "parquet")},
            {"setting": "results.out_dir", "value": results_cfg.get("out_dir", "./runs")},
            {"setting": "checkpoint.enabled", "value": str(checkpoint_cfg.get("enabled", False))},
            {"setting": "checkpoint.base_dir", "value": checkpoint_cfg.get("base_dir", "checkpoints")},
            {"setting": "telemetry.enabled", "value": str(telemetry_cfg.get("enabled", False))},
            {"setting": "telemetry.otlp_endpoint", "value": telemetry_cfg.get("otlp_endpoint", "")},
        ]
    )

    mo.md("## Active Configuration")
    return checkpoint_cfg, mo, results_cfg, settings_df, telemetry_cfg


@app.cell
def _(settings_df):
    """Display the active configuration table."""
    return settings_df


@app.cell
def _(mo):
    """Introduce the code cost evaluation scorecard."""
    mo.md(
        """
        ## Code Cost Evaluation

        Coleman measures code cost as a **multi-dimensional scorecard**
        with four dimensions:

        | Dimension | What it measures | Tools |
        |-----------|-----------------|-------|
        | **Structural** | Maintainability, complexity, change risk | Radon, Xenon, Wily |
        | **Runtime** | CPU time, hotspots, memory pressure | Scalene, py-spy |
        | **Energy** | Estimated energy / carbon impact | CodeCarbon, pyRAPL |
        | **Operational** | Infrastructure effort proxy | All of the above |
        """
    )


def _parse_radon_mi(result):
    """Parse radon MI subprocess result into (scores, error)."""
    import json

    if result.returncode != 0:
        return {}, result.stderr.strip() or "radon exited with a non-zero status"
    try:
        mi_data = json.loads(result.stdout)
    except (ValueError, TypeError):
        return {}, "failed to parse radon MI output"
    parsed: dict[str, float] = {}
    for module, value in mi_data.items():
        if isinstance(value, (int, float)):
            parsed[module] = float(value)
        elif isinstance(value, dict) and "mi" in value:
            parsed[module] = float(value["mi"])
        else:
            return {}, f"unexpected radon MI format for {module}: {value!r}"
    return parsed, None


@app.cell
def _(mo, pd):
    """Run structural cost checks and display the results."""
    import subprocess
    import sys

    radon_mi_cmd = [sys.executable, "-m", "radon", "mi", "-s", "-j", "coleman/"]
    xenon_cmd = [
        sys.executable,
        "-m",
        "xenon",
        "--max-absolute",
        "C",
        "--max-modules",
        "B",
        "--max-average",
        "A",
        "coleman/",
    ]

    mi_result = subprocess.run(radon_mi_cmd, capture_output=True, text=True)
    xenon_result = subprocess.run(xenon_cmd, capture_output=True, text=True)

    mi_scores, mi_error = _parse_radon_mi(mi_result)
    mi_df = pd.DataFrame([{"module": m, "maintainability_index": s} for m, s in sorted(mi_scores.items())])

    xenon_status = "✅ pass" if xenon_result.returncode == 0 else "❌ fail"
    mi_passed = bool(mi_scores) and all(s >= 20 for s in mi_scores.values())
    if mi_error:
        mi_status = "⚠️ error"
    elif mi_passed:
        mi_status = "✅ pass"
    else:
        mi_status = "❌ fail"

    checks = [
        {
            "check": "Xenon complexity gate",
            "threshold": "max-absolute=C, max-modules=B, max-average=A",
            "status": xenon_status,
        },
        {
            "check": "Radon maintainability index",
            "threshold": "all modules ≥ A (MI ≥ 20)",
            "status": mi_status,
        },
    ]
    checks_df = pd.DataFrame(checks)

    mo.md("### Structural Cost — CI Gates")
    return checks_df, mi_df, mi_error, mi_result


@app.cell
def _(checks_df, mi_df, mi_error, mi_result, mo):
    """Display structural cost gate results and maintainability scores."""
    stderr = getattr(mi_result, "stderr", "")
    error_parts = [part.strip() for part in [mi_error, stderr] if part and part.strip()]
    error_text = "\n\n".join(error_parts)
    missing_dependency = any(
        token in error_text.lower()
        for token in [
            "no module named radon",
            "modulenotfounderror",
            "command not found",
            "not recognized as an internal or external command",
        ]
    )

    if not mi_df.empty:
        mi_content = mi_df
    elif error_text:
        message = f"Unable to compute maintainability index.\n\n```text\n{error_text}\n```"
        if missing_dependency:
            message += "\n\nRun `uv sync --extra dev` to install Radon."
        mi_content = mo.md(message)
    else:
        mi_content = mo.md("Run `uv sync --extra dev` to install Radon.")

    mo.vstack(
        [
            checks_df,
            mo.md("### Maintainability Index per Module"),
            mi_content,
        ]
    )


@app.cell
def _(mo):
    """Show code cost CLI commands for local evaluation."""
    mo.md(
        """
        ### Running Code Cost Checks Locally

        ```bash
        # All structural checks (complexity + maintainability + xenon gate)
        make cost-structural

        # Runtime profiling with Scalene
        make cost-profile-scalene

        # Energy estimation with CodeCarbon
        make cost-energy
        ```

        See [Code Cost Evaluation](code-cost.md) for full documentation.
        """
    )


@app.cell
def _(mo, telemetry_cfg):
    """Explain where live observability data is exposed."""
    grafana_url = "http://localhost:3000"
    collector_url = telemetry_cfg.get("otlp_endpoint", "http://localhost:4318")

    mo.md(
        f"""
        ## Live Observability

        Grafana is where you inspect live execution behavior:

        * Grafana: {grafana_url}
        * OTel collector endpoint: {collector_url}

        Use it for throughput, latency, CPU, memory, worker isolation,
        dataset slicing, and execution separation while the run is active.
        """
    )


@app.cell
def _(checkpoint_cfg, json, mo, path_class, pd):
    """Inspect checkpoint progress files used for resume and recovery."""
    checkpoint_root = path_class(checkpoint_cfg.get("base_dir", "checkpoints"))
    checkpoint_files = sorted(checkpoint_root.glob("**/progress_*.json"))

    rows = []
    for checkpoint_file in checkpoint_files:
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_dir": checkpoint_file.parent.name,
                "progress_file": str(checkpoint_file),
                "experiment": payload.get("experiment"),
                "step_committed": payload.get("step_committed"),
                "checkpoint_path": payload.get("checkpoint_path"),
                "timestamp": payload.get("timestamp"),
            }
        )

    checkpoint_df = pd.DataFrame(rows)
    mo.md("## Resume / Recovery State")
    return checkpoint_df, mo


@app.cell
def _(checkpoint_df, mo):
    """Display current checkpoint progress or explain its absence."""
    checkpoint_df if not checkpoint_df.empty else mo.md("No checkpoint progress files found.")


@app.cell
def _(mo, path_class, results_cfg):
    """Describe where final experiment results are stored and how reruns behave."""
    runs_root = path_class(results_cfg.get("out_dir", "./runs"))
    parquet_files = sorted(runs_root.glob("**/*.parquet"))

    mo.md(
        f"""
        ## Final Results Storage

        Final experiment facts are stored in the results sink, not in Grafana.

        * Parquet root: {runs_root}
        * Parquet files found: {len(parquet_files)}
        * ClickHouse sink: enabled only when `results.sink = "clickhouse"`

        Re-running experiments in the same `runs` directory appends new Parquet
        files by default. Existing result files are preserved.
        """
    )
    return parquet_files, runs_root


@app.cell
def _(duckdb, parquet_files, pd, runs_root):
    """Load raw Parquet rows and build an aggregated final-results summary."""
    if not parquet_files:
        raw_df = pd.DataFrame()
        summary_df = pd.DataFrame()
        return raw_df, summary_df

    parquet_glob = str(runs_root / "**" / "*.parquet")

    raw_df = duckdb.sql(
        f"""
        SELECT *
        FROM read_parquet('{parquet_glob}', hive_partitioning=1)
        """
    ).df()

    summary_df = duckdb.sql(
        f"""
        SELECT scenario,
               execution_id,
               experiment,
               policy,
               reward_function,
               AVG(fitness) AS avg_napfd,
               AVG(cost) AS avg_apfdc,
               AVG(prioritization_time) AS avg_prioritization_time,
               AVG(process_memory_rss_mib) AS avg_rss_mib,
               AVG(process_cpu_utilization_percent) AS avg_cpu_pct,
               MAX(wall_time_seconds) AS wall_time_seconds
        FROM read_parquet('{parquet_glob}', hive_partitioning=1)
        GROUP BY scenario, execution_id, experiment, policy, reward_function
        ORDER BY avg_napfd DESC, avg_apfdc DESC
        """
    ).df()
    return raw_df, summary_df


@app.cell
def _(mo, summary_df):
    """Preview the aggregated final-results summary."""
    mo.md("## Final Results Summary")
    summary_df.head(25) if not summary_df.empty else mo.md("No Parquet results found yet.")


@app.cell
def _(mo):
    """Show practical SQL snippets for DuckDB and ClickHouse users."""
    duckdb_example = """
SELECT scenario, execution_id, policy, AVG(fitness) AS avg_napfd
FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
GROUP BY scenario, execution_id, policy
ORDER BY avg_napfd DESC;
""".strip()

    clickhouse_example = """
SELECT scenario, execution_id, policy, AVG(fitness) AS avg_napfd
FROM coleman_results
GROUP BY scenario, execution_id, policy
ORDER BY avg_napfd DESC;
""".strip()

    mo.vstack(
        [
            mo.md("## Query Snippets"),
            mo.md("### DuckDB"),
            mo.md(f"```sql\n{duckdb_example}\n```"),
            mo.md("### ClickHouse"),
            mo.md(f"```sql\n{clickhouse_example}\n```"),
        ]
    )


@app.cell
def _(mo, runs_root, summary_df):
    """Export the current summary as a CSV artifact for reports."""
    export_dir = runs_root / "analysis"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "summary.csv"

    if not summary_df.empty:
        summary_df.to_csv(export_path, index=False)
        mo.md(f"## Export\nSummary exported to `{export_path}`.")
    else:
        mo.md("## Export\nNo summary exported because there are no result rows yet.")


@app.cell
def _(mo, plt, sns, summary_df):
    """Plot average NAPFD per policy from the persisted final results."""
    if summary_df.empty:
        mo.md("## Analysis Plot\nRun an experiment first to generate result plots.")
        return

    top_policies = (
        summary_df.groupby("policy", as_index=False)["avg_napfd"].mean().sort_values("avg_napfd", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_policies, x="policy", y="avg_napfd", ax=ax)
    ax.set_title("Average NAPFD by Policy")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average NAPFD")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


@app.cell
def _(mo):
    """Summarize the persistence semantics of results and checkpoints."""
    mo.md(
        """
        ## Result Persistence Semantics

        * Parquet appends new files under `./runs/`
        * ClickHouse appends new rows to `coleman_results`
        * `execution_id` is the safest way to isolate one run analytically
        * checkpoints update the latest durable state for the same run and experiment
        * if you want a fresh analytical space, clean `./runs/` and optionally `./checkpoints/`
        """
    )


@app.cell
def _(mo):
    """Close with recommended next steps after running the notebook."""
    mo.md(
        """
        ## Suggested Next Steps

        * Run `coleman run --config run.yaml` to generate fresh experiment data
        * Run `make cost-structural` to evaluate structural cost before and after changes
        * Run `make cost-energy` to compare energy impact of different implementations
        * Open Grafana to inspect live execution behavior while the run is active
        * Use the Parquet summary above for final comparisons and report export
        * Inspect `./checkpoints/` to verify resume and recovery progress
        * Switch to ClickHouse when you want a persistent analytical store instead of Parquet files
        """
    )


if __name__ == "__main__":
    app.run()
