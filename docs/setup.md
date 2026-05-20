# Setup & Architecture

## Default run (no Docker required)

Coleman is framework-first: the default installation works without any
external services.

```bash
pip install coleman
```

Use the library API or the `coleman` CLI to run experiments:

```python
from coleman.api import run, load_spec

spec = load_spec("my-experiment.yaml")
result = run(spec)
print(result.run_id, result.artifacts_dir)
```

```bash
coleman run --config my-experiment.yaml
```

Results are written as **partitioned Parquet files** (zstd compressed) under
`./runs/` by default.  You can query them directly with DuckDB or Polars
without loading everything into RAM.

Repeated runs append new result files by default. They do not replace previous
results unless you explicitly remove `./runs/` or choose another `out_dir`.

## Architecture overview

Coleman separates three concerns:

| Layer | Purpose | Default | Optional |
|-------|---------|---------|----------|
| **Results** | Persist experiment facts (NAPFD, APFDc, …) | Partitioned Parquet | ClickHouse sink |
| **Checkpoints** | Crash-safe resume | Local filesystem (pickle + progress.json) | — |
| **Telemetry** | Observability (latency, throughput) | Disabled (NoOp) | OpenTelemetry + Collector |

### Null adapters

When a layer is disabled its module resolves to a **Null implementation**
with near-zero overhead:

* `NullSink` — discards result rows
* `NullCheckpointStore` — never saves/loads checkpoints
* `NoOpTelemetry` — all instrument calls are instant no-ops

## Configuration system

Coleman uses **YAML configuration files** validated against typed
Pydantic v2 models.  See the [Configuration guide](configuration.md) for
the full schema reference, config packs, sweep engine, and determinism
contract.

### Quick example

```yaml
# experiment.yaml
packs:
  - policy/linucb
  - reward/rnfail
  - results/parquet

experiment:
  datasets: ["alibaba@druid"]

execution:
  independent_executions: 30
```

```bash
coleman run --config experiment.yaml
```

### Key concepts

| Concept | Description |
|---------|-------------|
| **RunSpec** | Typed Pydantic v2 model describing one complete experiment run |
| **Config packs** | Composable YAML fragments (`packs/<category>/<name>.yaml`) merged left-to-right |
| **Sweep engine** | Grid (Cartesian) or zip (paired) expansion of parameter ranges into multiple specs |
| **Deterministic run_id** | `sha256(canonical_json(spec))[:12]` — same config always produces the same ID |
| **Provenance** | `spec.resolved.json` + `provenance.json` (git, Python version, `uv.lock` hash) per run |

## Optional extras

```bash
# Telemetry (OpenTelemetry SDK)
pip install coleman[telemetry]

# ClickHouse sink
pip install coleman[clickhouse]
```

## Optional observability stack

See the [Observability guide](observability.md) for a Docker Compose stack
with OTel Collector + Grafana (ClickHouse under a profile).

## How to query results

Results are written as Hive-partitioned Parquet files under `./runs/`.  You
can query them directly with DuckDB (already a project dependency):

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

For a guided end-to-end walkthrough, see the official [workflow notebook](workflow.py).
