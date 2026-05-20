# Optional Local Observability Stack

This directory contains an **optional** Docker Compose stack for local
observability when developing or profiling Coleman experiments.

> **Framework-first guarantee:** `coleman run --config run.yaml` works without Docker or
> any of these services.  Install optional extras only when you need them.

## Quick start

```bash
# Base stack (OTel Collector + Prometheus + Grafana)
docker compose up -d

# Use the telemetry/local pack in your run.yaml:
#   packs:
#     - telemetry/local

# Run your experiment
coleman run --config run.yaml

# Open Grafana (datasource + dashboard are auto-provisioned)
# http://localhost:3000
```

No manual datasource setup is required in Grafana.

The overview dashboard includes "Active Runs" and "Current Cycle By Active
Experiment" on the top row so optimization progress is visible in real time.

It also includes a `Sched Time Ratio` filter (`time_ratio` label) so you can
separate runs by CI budget percentage and compare behavior under tighter time
budgets.

For step-oriented analysis, prefer the snapshot panels (current iteration)
instead of timeline panels:

* `Current Cycle By Active Experiment`
* `Progress To Target Steps (%)`
* `Convergence Signal (Current NAPFD)`
* `Checkpoint Save Rate (last 5m)`

The main overview panels are configured as instant snapshots (table/gauge/stat)
so the dashboard emphasizes current step/stage rather than a time-axis history.

Snapshot table panels include a current-step column (derived from
`cycles_total`) beside the metric value.

Grafana is intended for live behavior during execution. Final experiment facts
remain in the configured results sink:

* `./runs/` when using the default Parquet sink
* `coleman_results` when using the ClickHouse sink
* `./checkpoints/` for resume/recovery state

For an end-to-end example covering setup, observability, checkpoints, export,
and analysis, see [docs/workflow.py](docs/workflow.py).

## With ClickHouse

```bash
OTEL_COLLECTOR_CONFIG=otel-collector-config-clickhouse.yaml \
  docker compose --profile clickhouse up -d
pip install coleman[telemetry,clickhouse]
```

When the ClickHouse profile is active:

* The OTel Collector exports **metrics, traces, and logs** to ClickHouse
  (`otel_metrics`, `otel_traces`, `otel_logs` tables, created automatically).
* Prometheus continues to receive metrics for live dashboards.
* Grafana ships with **two datasources**: Prometheus (default) and ClickHouse.
* Grafana Explore can query traces and logs in ClickHouse and supports
  logs ↔ traces correlation workflows.

### Persistent observability mode

Use the ClickHouse profile when you need:

* Post-run analysis and cross-run comparisons
* Long retention beyond Prometheus's local TSDB
* A single backend for both experiment results and telemetry
  (`coleman_results` + `otel_*` tables)

### Example ClickHouse queries

```sql
-- Recent traces
SELECT TraceId, SpanName, Duration
FROM otel_traces
ORDER BY Timestamp DESC
LIMIT 20;

-- Metric averages over the last hour
SELECT MetricName, avg(Value)
FROM otel_metrics
WHERE TimeUnix > now() - INTERVAL 1 HOUR
GROUP BY MetricName;

-- Log entries
SELECT Timestamp, SeverityText, Body
FROM otel_logs
ORDER BY Timestamp DESC
LIMIT 50;
```

## Tear down

```bash
docker compose --profile clickhouse down -v
```

## Metric names

| Metric | Type | Description |
|--------|------|-------------|
| `coleman.cycles_total` | Counter | Total experiment cycles processed |
| `coleman.bandit_update_latency` | Histogram (s) | Bandit arm-update latency |
| `coleman.prioritization_latency` | Histogram (s) | Test-case prioritization latency |
| `coleman.evaluation_latency` | Histogram (s) | Evaluation step latency |
| `coleman.napfd` | Histogram | NAPFD score distribution |
| `coleman.apfdc` | Histogram | APFDc score distribution |

## Endpoints

* Grafana: `http://localhost:3000`
* Prometheus UI: `http://localhost:9090`
* Collector exporter (scrape target): `http://localhost:8889/metrics`

### Cardinality rules

* **No `step` label** in metrics (would create unbounded cardinality).
* `run_id` is a resource attribute, not a metric label.
* Per-step detail is available in **traces** (span attributes).

## Example DuckDB queries over Parquet results

```sql
-- Average NAPFD per policy
SELECT policy, AVG(fitness) AS avg_napfd
FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
GROUP BY policy
ORDER BY avg_napfd DESC;

-- Cost distribution per reward function
SELECT reward_function,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cost) AS median_cost
FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
GROUP BY reward_function;
```
