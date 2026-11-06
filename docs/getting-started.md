# Run your first CI experiment

Build a tiny test history, run Random and UCB under the same budget, inspect six
native result rows, then continue from checkpoint. This tutorial uses an empty
working directory and the installed package; no repository checkout, downloaded
benchmark, Docker or monitoring service is required.

The history is authored example data. It teaches the mechanics, rather than
establishing that either policy improves a real CI system.

## 1. Install in a virtual environment

Use Python 3.14 or later. The wheel URL below pins a published distribution and
its SHA-256; pip checks the complete download before installation.

```bash
mkdir coleman-first-run
cd coleman-first-run
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install "https://github.com/jacksonpradolima/coleman/releases/download/v4.19.0/coleman-4.19.0-py3-none-any.whl#sha256=070e21c19fb7e3f2e392bd68706c354ed5ab7f76b86db80f5f1bdc5721b767bd"
```

For the unreleased v5 source, install the reviewed wheel with `ci` using the
[component guide](core-installation.md). The pinned v4 command above retains its
original dependencies. Native durable workflows have their own Linux/macOS
process and filesystem requirements.

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. The experiment itself uses Python paths and works independently of
these shell activation commands. Optional integrations and developer installation
are covered in [setup](setup.md#installation-options) and
[contributing](contributing.md).

## 2. Create a history and run it

Save this complete program as `first_run.py`:

```python
from pathlib import Path

import polars as pl

from coleman.api import run
from coleman.spec import RunSpec

dataset = Path("data/example-ci")
dataset.mkdir(parents=True, exist_ok=True)
rows = [
    {"BuildId": build, "Name": name, "Duration": duration,
     "CalcPrio": 0, "LastRun": "2026-01-01", "Verdict": verdict}
    for build in (1, 2, 3)
    for name, duration, verdict in (
        ("unit", 1.0, 0), ("integration", 2.0, build % 2), ("system", 4.0, 0)
    )
]
pl.DataFrame(rows).write_parquet(dataset / "features-engineered.parquet")

spec = RunSpec.model_validate({
    "execution": {"independent_executions": 1, "parallel_pool_size": 1, "seed": 42},
    "experiment": {
        "datasets_dir": "data", "datasets": ["example-ci"],
        "policies": ["Random", "UCB"], "rewards": ["RNFail"],
        "budget": {"mode": "ratio", "values": [0.5]},
    },
    "algorithm": {"ucb": {"rnfail": {"c": 1.0}}},
    "results": {"out_dir": "runs"},
    "checkpoint": {"enabled": True, "interval": 1},
})
result = run(spec)
print(result.run_id, result.artifacts_dir)
```

Run it:

```bash
python first_run.py
```

Each build contains three tests costing one, two and four seconds. The `0.5`
ratio gives a modeled budget of 3.5 seconds per build. Recorded verdicts feed the
RNFail learning signal after scheduling. The policies share data, budget and seed.
The native runner simulates scheduling against those records; it does not launch
these test names as operating-system commands.

The program prints a deterministic run ID and its directory. The
[input guide](input-data.md) explains the columns; [concepts](concepts.md)
explains the feedback loop.

## 3. Inspect the result

The run directory contains resolved configuration, environment provenance,
partitioned results and enabled checkpoints:

```text
runs/<run_id>/
├── spec.resolved.json
├── provenance.json
├── results/          # partitioned Parquet
└── checkpoints/      # progress markers and checkpoint payloads
```

Save this as `inspect_results.py` and run `python inspect_results.py`:

```python
import duckdb

with duckdb.connect() as connection:
    print(connection.sql("""
        SELECT policy, step, fitness AS napfd, cost AS apfdc,
               tests_ran, tests_not_ran
        FROM read_parquet('runs/*/results/**/*.parquet', hive_partitioning=1)
        ORDER BY policy, step
    """))
```

In this fresh directory you should see **six rows**: two policies × three
builds. The legacy column `fitness` stores NAPFD; `cost` stores APFDc and is a
quality metric, not seconds. Read [metric interpretation](evaluation-guide.md#read-native-results)
before drawing conclusions from their values.

## 4. Run the same configuration through the CLI

Save `run.yaml` alongside the script. This uses explicit settings, so it does not
need a local `packs` directory:

```yaml
execution:
  independent_executions: 1
  parallel_pool_size: 1
  seed: 42
experiment:
  datasets_dir: data
  datasets: [example-ci]
  policies: [Random, UCB]
  rewards: [RNFail]
  budget:
    mode: ratio
    values: [0.5]
algorithm:
  ucb:
    rnfail:
      c: 1.0
results:
  out_dir: runs
checkpoint:
  enabled: true
  interval: 1
```

```bash
coleman validate --config run.yaml --resolve resolved.json
coleman run --config run.yaml
python inspect_results.py
```

These settings resolve to the same spec as the Python example, so the run ID and
checkpoint namespace match. A completed checkpoint lets the runner skip completed
work. After an interruption, work since the last durable checkpoint may need to
be repeated. Keep the input bytes, runtime and configuration fixed when resuming;
the native run ID fingerprints configuration, not the entire external dataset.
Do not load checkpoint pickle files from an untrusted source.

Changing a configuration field can change the run ID, including changing an output
path. For the complete hashing compatibility rules, see
[run identity](api/spec-run-id.md) and [configuration](configuration.md).

## Choose your next step

| You want to… | Continue with |
| --- | --- |
| Use actual public reports | [Dataset workflows](datasets.md), starting with [LRTS Karaf](lrts-parquet-etl.md) |
| Prepare your own data | [Input format and semantics](input-data.md) |
| Try contextual or changing-history policies | [Policies and rewards](policies.md) |
| Change budgets, packs or seeds | [Configuration](configuration.md) |
| Study software configurations | [WTS and VTS](highly-configurable.md) |
| Compare the policies rigorously | [Evaluation guide](evaluation-guide.md) |
| Query larger outputs or explore plots | [Analysis playbook](analysis-playbook.md), [workflow notebook](workflow.md) |
