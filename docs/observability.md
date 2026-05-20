# Observability

This page describes the **optional** local observability stack for
debugging and profiling Coleman experiments.

!!! note "Framework-first guarantee"
    `coleman run --config run.yaml` works without Docker or any of these services.
    The observability stack is **optional** for local installs, but
    **enabled automatically** in the DevContainer.

## DevContainer: zero-step setup

If you develop inside the DevContainer, **everything is already running**.
The container automatically:

1. Installs the `telemetry` and `clickhouse` pip extras
2. Starts OTel Collector + Prometheus + Grafana + ClickHouse via Docker Compose
3. Telemetry can be enabled via the `telemetry/local` pack in `run.yaml`

Just run your experiment:

```bash
coleman run --config run.yaml
# Open http://localhost:3000 → Grafana shows metrics in real-time
```

Grafana datasource and dashboard are provisioned automatically.
ClickHouse is also available at `http://localhost:8123`.

Parallel execution is also telemetry-safe now: each worker builds an isolated
environment and emits its own `execution_id` and `worker_id`, so concurrent
runs can be separated in Grafana and in persisted results.

The default dashboard now includes top-level filters for dataset, execution,
and policy, so you can slice one live run without visually mixing it with the
others.

The top row also includes a "Current Cycle By Active Experiment" panel next to
"Active Runs", so you can see optimization stage progression while runs are
still active.

The dashboard now has a dedicated `Sched Time Ratio` filter (label `time_ratio`)
so you can isolate runs by available CI budget percentage.

For operational analysis, use the snapshot panels that focus on current
iteration/stage instead of time-based curves:

* `Current Cycle By Active Experiment`
* `Progress To Target Steps (%)`
* `Convergence Signal (Current NAPFD)`
* `Checkpoint Save Rate (last 5m)`

The agent/system panels in the overview are also configured as instant
snapshots (table/gauge/stat), which removes the time-axis dependency when the
goal is to inspect current simulation stage.

Snapshot table panels now include an explicit current-step column (merged from
`cycles_total`) beside the metric value, so each metric row can be interpreted
in the context of the current simulation iteration.

## Live behavior vs. final results

Grafana answers: "what is happening right now?"

For final experiment results, use the persisted outputs instead:

* `./runs/` for partitioned Parquet datasets
* `coleman_results` in ClickHouse when the ClickHouse sink is enabled
* `./checkpoints/` to inspect resume/recovery progress files

The complete guided workflow is available in [workflow.py](workflow.py).

## Local setup (without DevContainer)

```bash
# Base stack (OTel Collector + Prometheus + Grafana)
cd examples/observability
docker compose up -d

# Install telemetry extras
uv pip install coleman[telemetry]

# Enable telemetry — use the telemetry/local pack in your run.yaml:
#   packs:
#     - telemetry/local

# Run your experiment
coleman run --config run.yaml
```

## Endpoints

* Grafana: `http://localhost:3000`
* Prometheus UI: `http://localhost:9090`
* Collector exporter (scrape target): `http://localhost:8889/metrics`

## With ClickHouse

```bash
docker compose --profile clickhouse up -d
pip install coleman[clickhouse]
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
| `coleman.process_memory_rss` | Histogram (MiB) | Resident memory sampled during execution |
| `coleman.process_memory_peak_rss` | Histogram (MiB) | Peak resident memory seen by the process |
| `coleman.process_cpu_utilization` | Histogram (%) | CPU utilization sampled during execution |
| `coleman.experiment_wall_time` | Histogram (s) | End-to-end elapsed time of one experiment |
| `coleman.experiment_cpu_time` | Histogram (s) | CPU time consumed by one experiment |

### Cardinality rules

* **No `step` label** in metrics (would create unbounded cardinality).
* `execution_id` and `worker_id` are metric labels on purpose so parallel runs stay separable.
* Per-step detail is available in **traces** (span attributes).

## Resource metrics and energy

The default stack now exposes CPU and memory cost directly in telemetry and in
persisted results.

Energy is intentionally not emitted by default because, in containers and
Codespaces, reliable hardware counters are often unavailable. If you want
energy metrics, the recommended next step is to integrate host-specific sources
such as Intel RAPL, NVIDIA NVML, or a node exporter running on the host.

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
