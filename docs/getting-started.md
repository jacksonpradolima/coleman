# Getting Started

## Installation

### Option A: DevContainer (recommended)

The fastest way to start is with a [DevContainer](https://containers.dev/).
Open the repo in **VS Code** (or any DevContainer-compatible editor) and
select **"Reopen in Container"** — **everything works out of the box**,
including the full observability stack.

**What happens automatically:**

1. Python 3.14 + uv + all dependencies (including telemetry & ClickHouse extras) are installed
2. Pre-commit hooks are configured
3. The **observability stack** (OTel Collector + Grafana) starts via Docker-in-Docker
4. **Telemetry** can be enabled via the `telemetry/local` pack in `run.yaml`

After the container builds, just run your experiment:

```bash
coleman run --config run.yaml
# Open http://localhost:3000 → Grafana shows metrics in real-time
```

No manual steps required — skip to [Running Experiments](#running-experiments).

**What's included:**

| What | Why |
|------|-----|
| **Python 3.14 + uv** | Project package manager |
| **Docker-in-Docker** | Runs the observability stack automatically |
| **VS Code extensions** | Ruff, Pylance, Pyright, Copilot, TOML, Jupyter, and more |
| **Telemetry + ClickHouse extras** | Pre-installed — no extra `pip install` needed |
| **OTel Collector + Grafana** | Started automatically on container start |
| **Port forwarding** | All service ports mapped to your host browser |

### Option B: Local setup

#### Prerequisites

- Python 3.14+
- [UV](https://docs.astral.sh/uv/)

#### Install

```bash
git clone https://github.com/jacksonpradolima/coleman.git
cd coleman
uv sync
uv pip install -e .
```

#### Development Setup

```bash
# Install all development dependencies
make install

# Install pre-commit hooks
make pre-commit-install
```

## Configuration

Coleman uses YAML configuration files with typed Pydantic v2 models.
You can compose configs with **config packs** and override specific settings
inline.

### Quick config (YAML + packs)

Create a YAML config file:

```yaml
# my-experiment.yaml
packs:
  - policy/linucb
  - reward/rnfail
  - results/parquet
  - telemetry/off

experiment:
  datasets: ["alibaba@druid"]

execution:
  independent_executions: 30
```

Run with the `coleman` CLI:

```bash
coleman run --config my-experiment.yaml
```

Or use the library API:

```python
from coleman.api import run, load_spec

spec = load_spec("my-experiment.yaml")
result = run(spec)
print(result.run_id, result.artifacts_dir)
```

See the [Configuration guide](configuration.md) for the full schema
reference, config packs, sweep engine, and determinism contract.

## Running Experiments

### Using the `coleman` CLI (recommended)

```bash
# Single run
coleman run --config my-experiment.yaml

# Parameter sweep
coleman sweep --config my-experiment.yaml \
    --grid algorithm.ucb.rnfail.c=0.1,0.3,0.5 \
    --grid execution.seed=range(0,10) \
    --workers 4

# Dry-run (preview specs without executing)
coleman sweep --config my-experiment.yaml \
    --grid execution.seed=range(0,5) \
    --dry-run
```

Results are written to `./runs/<run_id>/` with `spec.resolved.json` and
`provenance.json` alongside the experiment data.

### DevContainer: just run it

If you're in the DevContainer, the observability stack is already running and
telemetry is enabled.  Just run:

```bash
coleman run --config my-experiment.yaml
# Results    → ./runs/       (Parquet)
# Checkpoints → ./checkpoints/
# Metrics    → http://localhost:3000 (Grafana, live)
```

## Checking Final Results

Grafana is for live execution behavior: throughput, latency, CPU, memory, and
run separation while the experiment is still running.

The final experiment facts are stored separately:

* Parquet, by default: `./runs/`
* Checkpoint progress for resume/recovery: `./checkpoints/`
* ClickHouse, when enabled: table `coleman_results`

### What happens if you run experiments again?

By default, Coleman **appends** new final results instead of replacing the
old ones.

For the default Parquet sink this means:

* new files are created under `./runs/`
* previous files are preserved
* old and new executions coexist in the same dataset
* the new `execution_id` lets you separate one run from another analytically

For the ClickHouse sink this means:

* new rows are inserted into `coleman_results`
* previous rows are preserved unless you explicitly delete them

Checkpoint behavior is different: for the same run and experiment, the latest
checkpoint progress is updated so recovery can continue from the most recent
durable step.

If you want a completely fresh analysis space, choose one of these options
before running again:

```bash
# Option 1: start from a clean Parquet directory
rm -rf ./runs

# Option 2: keep old runs, but write new ones elsewhere
# Set results.out_dir: "./runs-new" in your run.yaml

# Option 3: clean checkpoints too, if you do not want recovery state reused
rm -rf ./checkpoints
```

### Default Parquet output

After `coleman run --config run.yaml`, inspect the partitioned Parquet dataset:

```bash
find ./runs -name '*.parquet' | head
```

Prepare a reusable view in DuckDB, then query it directly:

```python
import duckdb
con = duckdb.connect("analysis.duckdb")
con.execute("""
    CREATE OR REPLACE VIEW experiment_results AS
    SELECT *
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
""")

con.sql("""
    SELECT scenario,
           execution_id,
           policy,
           reward_function,
           AVG(fitness) AS avg_napfd,
           AVG(cost) AS avg_apfdc,
           AVG(process_memory_rss_mib) AS avg_rss_mib,
           AVG(process_cpu_utilization_percent) AS avg_cpu_pct
    FROM experiment_results
    GROUP BY scenario, execution_id, policy, reward_function
    ORDER BY avg_napfd DESC
""")
```

### DuckDB, in practice

DuckDB is the easiest way to inspect final results deeply without moving data
out of the Parquet dataset.

#### Start an interactive session

```bash
uv run ipython
```

Then initialize once and reuse the same view across the remaining examples:

```python
import duckdb
con = duckdb.connect("analysis.duckdb")
con.execute("""
    CREATE OR REPLACE VIEW experiment_results AS
    SELECT *
    FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
""")
```

#### Inspect available columns

```python
con.sql("""
    DESCRIBE
    SELECT *
    FROM experiment_results
""").df()
```

#### See which executions are available

```python
con.sql("""
    SELECT scenario,
           execution_id,
           COUNT(*) AS rows,
           MIN(step) AS first_step,
           MAX(step) AS last_step
    FROM experiment_results
    GROUP BY scenario, execution_id
    ORDER BY scenario, execution_id
""").df()
```

#### Compare policies by final quality and resource cost

```python
con.sql("""
    SELECT scenario,
           policy,
           reward_function,
           AVG(fitness) AS avg_napfd,
           AVG(cost) AS avg_apfdc,
           AVG(prioritization_time) AS avg_prioritization_time,
           AVG(process_memory_rss_mib) AS avg_rss_mib,
           AVG(process_cpu_utilization_percent) AS avg_cpu_pct,
           MAX(wall_time_seconds) AS wall_time_seconds
    FROM experiment_results
    GROUP BY scenario, policy, reward_function
    ORDER BY avg_napfd DESC, avg_apfdc DESC
""").df()
```

#### Slice one specific execution

```python
execution_id = 'replace-with-real-execution-id'
con.sql(f"""
    SELECT experiment,
           step,
           policy,
           reward_function,
           fitness,
           cost,
           process_memory_rss_mib,
           process_cpu_utilization_percent
    FROM experiment_results
    WHERE execution_id = '{execution_id}'
    ORDER BY experiment, step, policy
""").df()
```

#### Export a filtered report

Before executte the following command, create the `analysis` directory first under the `runs/` directory.

```python
con.sql("""
    COPY (
        SELECT scenario,
               execution_id,
               policy,
               reward_function,
               AVG(fitness) AS avg_napfd,
               AVG(cost) AS avg_apfdc
        FROM experiment_results
        GROUP BY scenario, execution_id, policy, reward_function
    ) TO './runs/analysis/final-summary.csv' (HEADER, DELIMITER ',')
""")
print('exported ./runs/analysis/final-summary.csv')
```

### ClickHouse, in practice

If you want a long-lived analytical store instead of Parquet files, switch the
 results sink to ClickHouse.

#### Enable it

```yaml
# In your run.yaml, override the results section:
results:
  enabled: true
  sink: clickhouse
```

Run the service locally:

```bash
cd examples/observability
docker compose --profile clickhouse up -d
```

#### Initialize once and reuse

Start a Python session and connect once — then reuse `client` across all the
examples below:

```python
import clickhouse_connect

client = clickhouse_connect.get_client(host='localhost', port=8123, database='default')
```

#### Inspect stored schema

```python
print(client.query('DESCRIBE TABLE coleman_results').result_rows)
```

#### See which executions are available

```python
rows = client.query('''
    SELECT scenario,
           execution_id,
           COUNT(*) AS rows,
           MIN(step)  AS first_step,
           MAX(step)  AS last_step
    FROM coleman_results
    GROUP BY scenario, execution_id
    ORDER BY scenario, execution_id
''')
print(rows.result_rows)
```

#### Compare policies by final quality and resource cost

```python
rows = client.query('''
    SELECT scenario,
           policy,
           reward_function,
           AVG(fitness)                      AS avg_napfd,
           AVG(cost)                         AS avg_apfdc,
           AVG(prioritization_time)          AS avg_prioritization_time,
           AVG(process_memory_rss_mib)       AS avg_rss_mib,
           AVG(process_cpu_utilization_percent) AS avg_cpu_pct,
           MAX(wall_time_seconds)            AS wall_time_seconds
    FROM coleman_results
    GROUP BY scenario, policy, reward_function
    ORDER BY avg_napfd DESC, avg_apfdc DESC
''')
print(rows.result_rows)
```

#### Query a summary across all executions

```python
rows = client.query('''
    SELECT scenario,
           execution_id,
           policy,
           reward_function,
           AVG(fitness) AS avg_napfd,
           AVG(cost) AS avg_apfdc,
           AVG(process_memory_rss_mib) AS avg_rss_mib,
           AVG(process_cpu_utilization_percent) AS avg_cpu_pct
    FROM coleman_results
    GROUP BY scenario, execution_id, policy, reward_function
    ORDER BY avg_napfd DESC
''')
print(rows.result_rows)
```

#### Slice one specific execution

```python
execution_id = 'replace-with-real-execution-id'
rows = client.query(f'''
    SELECT experiment,
           step,
           policy,
           reward_function,
           fitness,
           cost,
           process_memory_rss_mib,
           process_cpu_utilization_percent
    FROM coleman_results
    WHERE execution_id = '{execution_id}'
    ORDER BY experiment, step, policy
''')
print(rows.result_rows)
```

#### Export a filtered report to CSV

```python
import csv

rows = client.query('''
    SELECT scenario,
           execution_id,
           policy,
           reward_function,
           AVG(fitness) AS avg_napfd,
           AVG(cost) AS avg_apfdc
    FROM coleman_results
    GROUP BY scenario, execution_id, policy, reward_function
''')

with open('./runs/analysis/final-summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(rows.column_names)
    writer.writerows(rows.result_rows)

print('exported ./runs/analysis/final-summary.csv')
```

#### Clean old data when needed

ClickHouse also accumulates results by default. If you need a fresh table:

```python
client.command('TRUNCATE TABLE coleman_results')
print('coleman_results truncated')
```

### Resume and recovery

If checkpoints are enabled, Coleman writes one directory per run under
`./checkpoints/`. Each experiment keeps a `progress_ex<N>.json` file pointing
to the last durable checkpoint.

To inspect recovery state:

```bash
find ./checkpoints -name 'progress_*.json' -maxdepth 3 -print
```

When `coleman run --config run.yaml` is started again with the same configuration,
the framework loads the last saved checkpoint and resumes from the next step
instead of replaying completed builds.

### Notebook workflow

For a guided end-to-end workflow covering configuration, observability,
checkpoints, export, and analysis, open the marimo notebook example in
[docs/workflow.py](docs/workflow.py).

## Port Reference

| Port | Service | URL |
|------|---------|-----|
| **3000** | Grafana | [http://localhost:3000](http://localhost:3000) |
| **4317** | OTel Collector (gRPC) | — (used by the framework) |
| **4318** | OTel Collector (HTTP) | — (used by the framework) |
| **8889** | Prometheus metrics | [http://localhost:8889/metrics](http://localhost:8889/metrics) |
| **8123** | ClickHouse (HTTP) | [http://localhost:8123](http://localhost:8123) *(profile: clickhouse)* |
| **9000** | ClickHouse (native) | — *(profile: clickhouse)* |

!!! note "DevContainer vs. local"
    In the DevContainer, the OTel Collector + Grafana start automatically
    and telemetry is enabled.  Locally, you start them manually with
    `docker compose up -d` and use the `telemetry/local` pack in your `run.yaml`.

## Quick Reference: Configuration

For the complete YAML schema, config packs, sweep engine, and determinism
contract, see the [Configuration guide](configuration.md).

See the [Setup & Architecture](setup.md) guide for a deeper explanation
of each layer and the [Observability guide](observability.md) for metric
names and cardinality rules.

---

See the [README](https://github.com/jacksonpradolima/coleman#readme) for
detailed usage instructions covering HCS strategies, contextual bandits, and
dataset preparation.
