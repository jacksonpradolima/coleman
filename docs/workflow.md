# Coleman Workflow

This page covers the full operational loop for running and analyzing
experiments.  Each section maps to a cell in the interactive
[marimo notebook](https://github.com/jacksonpradolima/coleman/blob/main/docs/workflow.py)
(`docs/workflow.py`) — open it locally with `marimo edit docs/workflow.py`
for a live, executable version.

---

## 1 — Active Configuration

Read the runtime configuration from `run.yaml`:

```python
import yaml
from pathlib import Path

config_path = Path("run.yaml")
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

results_cfg = config.get("results", {})
checkpoint_cfg = config.get("checkpoint", {})
telemetry_cfg = config.get("telemetry", {})
experiment_cfg = config.get("experiment", {})
```

Key settings you should inspect:

| Setting                  | Where it comes from                  |
|--------------------------|--------------------------------------|
| `datasets`               | `experiment.datasets`                |
| `scheduled_time_ratio`   | `experiment.scheduled_time_ratio`    |
| `results.enabled`        | `results.enabled`                    |
| `results.sink`           | `results.sink` (parquet / clickhouse)|
| `results.out_dir`        | `results.out_dir`                    |
| `checkpoint.enabled`     | `checkpoint.enabled`                 |
| `checkpoint.base_dir`    | `checkpoint.base_dir`               |
| `telemetry.enabled`      | `telemetry.enabled`                  |
| `telemetry.otlp_endpoint`| `telemetry.otlp_endpoint`            |

---

## 2 — Code Cost Evaluation

Coleman measures code cost as a **multi-dimensional scorecard**
with four dimensions:

| Dimension        | What it measures                      | Tools                  |
|------------------|---------------------------------------|------------------------|
| **Structural**   | Maintainability, complexity, change risk | Radon, Xenon, Wily  |
| **Runtime**      | CPU time, hotspots, memory pressure   | Scalene, py-spy        |
| **Energy**       | Estimated energy / carbon impact      | CodeCarbon, pyRAPL     |
| **Operational**  | Infrastructure effort proxy           | All of the above       |

### Structural cost — CI gates

Two gates run in CI on every pull request:

```bash
# Xenon complexity gate
python -m xenon --max-absolute C --max-modules B --max-average A coleman/

# Radon maintainability index (MI) — lists modules below the threshold;
# CI fails if this command reports such modules or exits with an error
python -m radon mi -s -n B coleman/
```

### Running code cost checks locally

```bash
# All structural checks (complexity + maintainability + xenon gate)
make cost-structural

# Runtime profiling with Scalene
make cost-profile-scalene

# Energy estimation with CodeCarbon
make cost-energy

# Complexity trend analysis with Wily
make cost-wily
```

See [Code Cost Evaluation](code-cost.md) for full documentation.

---

## 3 — Live Observability

Grafana is where you inspect live execution behavior:

- **Grafana:** [http://localhost:3000](http://localhost:3000)
- **OTel collector endpoint:** configured via `telemetry.otlp_endpoint`

Use it for throughput, latency, CPU, memory, worker isolation,
dataset slicing, and execution separation while the run is active.

---

## 4 — Resume / Recovery State

Inspect checkpoint progress files used for resume and recovery:

```python
import json
from pathlib import Path

checkpoint_root = Path(checkpoint_cfg.get("base_dir", "checkpoints"))
checkpoint_files = sorted(checkpoint_root.glob("**/progress_*.json"))

for f in checkpoint_files:
    payload = json.loads(f.read_text(encoding="utf-8"))
    print(f"  {f.parent.name}: step={payload.get('step_committed')}")
```

---

## 5 — Final Results Storage

Final experiment facts are stored in the results sink, not in Grafana.

- **Parquet root:** configured via `results.out_dir` (default `./runs`)
- **ClickHouse sink:** enabled only when `results.sink = "clickhouse"`
- Re-running experiments in the same `runs` directory appends new Parquet
  files by default. Existing result files are preserved.

### Loading results with DuckDB

```python
import duckdb

parquet_glob = "./runs/**/*.parquet"

summary_df = duckdb.sql(f"""
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
""").df()
```

---

## 6 — Export

Export the current summary as a CSV artifact for reports:

```python
export_dir = runs_root / "analysis"
export_dir.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(export_dir / "summary.csv", index=False)
```

---

## 7 — Analysis Plot

Plot average NAPFD per policy from the persisted final results:

```python
import matplotlib.pyplot as plt
import seaborn as sns

top_policies = (
    summary_df
    .groupby("policy", as_index=False)["avg_napfd"]
    .mean()
    .sort_values("avg_napfd", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=top_policies, x="policy", y="avg_napfd", ax=ax)
ax.set_title("Average NAPFD by Policy")
ax.set_xlabel("Policy")
ax.set_ylabel("Average NAPFD")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
```

---

## Query Snippets

### DuckDB

```sql
SELECT scenario, execution_id, policy, AVG(fitness) AS avg_napfd
FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
GROUP BY scenario, execution_id, policy
ORDER BY avg_napfd DESC;
```

### ClickHouse

```sql
SELECT scenario, execution_id, policy, AVG(fitness) AS avg_napfd
FROM coleman_results
GROUP BY scenario, execution_id, policy
ORDER BY avg_napfd DESC;
```

---

## Result Persistence Semantics

- Parquet appends new files under `./runs/`
- ClickHouse appends new rows to `coleman_results`
- `execution_id` is the safest way to isolate one run analytically
- Checkpoints update the latest durable state for the same run and experiment
- If you want a fresh analytical space, clean `./runs/` and optionally `./checkpoints/`

---

## Suggested Next Steps

- Run `coleman run --config run.yaml` to generate fresh experiment data
- Run `make cost-structural` to evaluate structural cost before and after changes
- Run `make cost-energy` to compare energy impact of different implementations
- Open Grafana to inspect live execution behavior while the run is active
- Use the Parquet summary above for final comparisons and report export
- Inspect `./checkpoints/` to verify resume and recovery progress
- Switch to ClickHouse when you want a persistent analytical store instead of Parquet files

!!! tip "Interactive version"
    For an executable version of this workflow, run the marimo notebook:
    ```bash
    marimo edit docs/workflow.py
    ```
