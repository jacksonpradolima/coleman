# Analysis Playbook

This guide is a complete reference for post-run analysis in Coleman, focused on
decision-quality, cost, stability, and extensibility impact.

Use this after one or more experiment runs have produced Parquet data under
your configured results directory.

## Data Access Patterns

### DuckDB (recommended)

```python
import duckdb

con = duckdb.connect("analysis.duckdb")
con.execute(
	"""
	CREATE OR REPLACE VIEW experiment_results AS
	SELECT *
	FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
	"""
)
```

### Basic sanity checks

```sql
SELECT COUNT(*) AS rows_total FROM experiment_results;

SELECT COUNT(DISTINCT execution_id) AS executions,
	   COUNT(DISTINCT policy) AS policies,
	   COUNT(DISTINCT reward_function) AS rewards
FROM experiment_results;
```

## Core Comparisons

### 1. Quality ranking (NAPFD)

```sql
SELECT policy,
	   reward_function,
	   AVG(fitness) AS avg_napfd,
	   STDDEV_SAMP(fitness) AS std_napfd,
	   COUNT(*) AS n
FROM experiment_results
GROUP BY policy, reward_function
ORDER BY avg_napfd DESC;
```

### 2. Cost ranking (APFDc)

```sql
SELECT policy,
	   reward_function,
	   AVG(cost) AS avg_apfdc,
	   STDDEV_SAMP(cost) AS std_apfdc,
	   COUNT(*) AS n
FROM experiment_results
GROUP BY policy, reward_function
ORDER BY avg_apfdc ASC;
```

### 3. Stability (coefficient of variation)

```sql
SELECT policy,
	   reward_function,
	   AVG(fitness) AS avg_napfd,
	   STDDEV_SAMP(fitness) AS std_napfd,
	   CASE
		 WHEN AVG(fitness) = 0 THEN NULL
		 ELSE STDDEV_SAMP(fitness) / AVG(fitness)
	   END AS cv_napfd
FROM experiment_results
GROUP BY policy, reward_function
ORDER BY cv_napfd ASC NULLS LAST;
```

## Budget and Scenario Sensitivity

Use scenario/time-ratio segmentation to find robust policies.

```sql
SELECT scenario,
	   policy,
	   AVG(fitness) AS avg_napfd,
	   AVG(cost) AS avg_apfdc,
	   AVG(prioritization_time) AS avg_prioritization_time
FROM experiment_results
GROUP BY scenario, policy
ORDER BY scenario, avg_napfd DESC;
```

If your scenario naming encodes time ratio, normalize it once and persist a
derived table for repeated analyses.

## Quality vs Cost Frontier (Pareto)

Goal: maximize NAPFD and minimize APFDc.

```python
import duckdb
import pandas as pd

df = duckdb.sql(
	"""
	SELECT policy,
		   reward_function,
		   AVG(fitness) AS avg_napfd,
		   AVG(cost) AS avg_apfdc
	FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
	GROUP BY policy, reward_function
	"""
).df()

def pareto_frontier(data: pd.DataFrame) -> pd.DataFrame:
	pts = data.sort_values(["avg_apfdc", "avg_napfd"], ascending=[True, False]).reset_index(drop=True)
	keep = []
	best_napfd = float("-inf")
	for _, row in pts.iterrows():
		if row["avg_napfd"] > best_napfd:
			keep.append(True)
			best_napfd = row["avg_napfd"]
		else:
			keep.append(False)
	return pts.loc[keep].copy()

front = pareto_frontier(df)
print(front)
```

Interpretation:

1. Points on the frontier are non-dominated trade-offs.
2. Off-frontier points are strictly worse in both metrics than at least one alternative.

## Win Rate Analysis (Pairwise)

Compare how often one policy beats another on the same execution slice.

```python
import duckdb
import pandas as pd

base = duckdb.sql(
	"""
	SELECT scenario,
		   execution_id,
		   policy,
		   AVG(fitness) AS score
	FROM read_parquet('./runs/**/*.parquet', hive_partitioning=1)
	GROUP BY scenario, execution_id, policy
	"""
).df()

pivot = base.pivot_table(index=["scenario", "execution_id"], columns="policy", values="score")
policies = list(pivot.columns)

rows = []
for a in policies:
	for b in policies:
		if a == b:
			continue
		valid = pivot[[a, b]].dropna()
		wins = (valid[a] > valid[b]).sum()
		total = len(valid)
		rows.append({"policy_a": a, "policy_b": b, "win_rate": wins / total if total else None, "n": total})

win_rate_df = pd.DataFrame(rows).sort_values(["policy_a", "win_rate"], ascending=[True, False])
print(win_rate_df.head(30))
```

## Runtime and Resource Footprint

```sql
SELECT policy,
	   AVG(prioritization_time) AS avg_prioritization_time,
	   AVG(process_memory_rss_mib) AS avg_rss_mib,
	   AVG(process_cpu_utilization_percent) AS avg_cpu_pct,
	   AVG(fitness) AS avg_napfd
FROM experiment_results
GROUP BY policy
ORDER BY avg_prioritization_time ASC;
```

Use this to identify policies that are fast enough for CI budget constraints.

## Extensions and Hook Artifacts

When using runner hooks and extensions, persist additional artifacts under the
run folder (for example, JSON/CSV derived from custom domain logic).

Recommended pattern:

1. Write one artifact per execution_id.
2. Include run_id, dataset_id, execution_id, and worker_id in each record.
3. Join artifact rows with experiment results in DuckDB for impact analysis.

Example join pattern (adapt paths to your artifact layout):

```sql
WITH base AS (
  SELECT run_id, execution_id, policy, fitness, cost
  FROM experiment_results
),
ext AS (
  SELECT run_id, execution_id, custom_signal, custom_bucket
  FROM read_json_auto('./runs/**/artifacts/*.json')
)
SELECT b.policy,
	   e.custom_bucket,
	   AVG(b.fitness) AS avg_napfd,
	   AVG(b.cost) AS avg_apfdc
FROM base b
JOIN ext e
  ON b.run_id = e.run_id AND b.execution_id = e.execution_id
GROUP BY b.policy, e.custom_bucket
ORDER BY avg_napfd DESC;
```

## Reproducibility Checklist

1. Keep spec.resolved.json and provenance.json for every run.
2. Compare only runs with consistent datasets and time-ratio sets.
3. Report both central tendency and variance.
4. Track run_id and execution_id in every exported table/chart.
5. Version-control analysis scripts and generated figures metadata.

## Suggested Reporting Bundle

For each experiment campaign, export:

1. Overall ranking by NAPFD and APFDc.
2. Pareto frontier table and scatter chart.
3. Stability table (std and CV).
4. Scenario sensitivity matrix.
5. Runtime/resource table.
6. Hook/extension impact table (if enabled).
