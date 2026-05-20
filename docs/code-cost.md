# Code Cost Evaluation

Coleman treats code cost as a **multi-dimensional scorecard**, not a
single number.  This guide explains the tooling and commands available for
evaluating four complementary cost dimensions.

## Dimensions

| Dimension | What it measures | Primary tools |
|-----------|-----------------|---------------|
| **Structural cost** | Maintainability, complexity, and change risk | Radon, Xenon, Wily |
| **Runtime cost** | CPU time, execution hotspots, and memory pressure | Scalene, py-spy |
| **Energy cost** | Estimated energy or carbon impact | CodeCarbon, pyRAPL |
| **Operational cost proxy** | Basis for translating hotspots into infrastructure effort | All of the above |

---

## Quick start

Install all cost evaluation tools together with the rest of the dev
dependencies:

```bash
uv sync --extra dev
```

---

## 1 — Structural cost

### Cyclomatic complexity (Radon CC)

```bash
make cost-complexity
# or
uv run radon cc -s -a coleman/
```

**Example output:**

```text
coleman/runner.py
    F 175:0 get_scenario_provider - C (12)
    F 420:0 run_experiment - B (10)
    F 145:0 create_agents - B (6)
    ...
coleman/bandit.py
    M 110:4 Bandit.add_arms - B (8)
    C 47:0 Bandit - A (3)
    ...

Average complexity: A (2.1)
```

Each entry shows `<type> <line>:<col> <name> - <grade> (<score>)`.  The type
prefix is `M` (method), `C` (class), or `F` (function).  Grades map to
cyclomatic complexity scores: **A** (1–5 — excellent), **B** (6–10 — moderate),
**C** (11–15 — high), **D–F** (≥16 — critical).  The CI gate (Xenon) rejects
anything worse than **C** for a single block.

### Maintainability index (Radon MI)

```bash
make cost-maintainability
# or
uv run radon mi -s coleman/
```

**Example output (all modules pass):**

```text
✅ All modules meet MI ≥ 20
```

The Maintainability Index (MI) is a composite score (0–100) combining
Halstead volume, cyclomatic complexity, and lines of code.  Scores ≥ 20
are considered maintainable; scores below 20 indicate code that is
harder to understand and modify.  The CI gate fails if any module drops
below this threshold.

**Example output (failure case):**

```text
❌ Modules below maintainability threshold (MI < 20):
coleman/legacy.py - C (14.32)
```

### Raw metrics (Radon raw)

```bash
make cost-raw
# or
uv run radon raw coleman/
```

**Example output:**

```text
coleman/runner.py
    LOC: 524
    LLOC: 202
    SLOC: 290
    Comments: 5
    Single comments: 10
    Multi: 143
    Blank: 81
    - Comment Stats
        (C % L): 1%
        (C % S): 2%
        (C + M % L): 28%
```

Key metrics: **LOC** (total lines), **SLOC** (source lines, excluding blanks
and comments), **LLOC** (logical lines — statement count), and **Comment
Stats** showing the ratio of inline comments to code.  Low `C % S` (comments
as a share of source) may indicate under-documented code.

### Xenon complexity gate

Xenon wraps Radon and returns a non-zero exit code when complexity
thresholds are exceeded.  This is the first CI guardrail:

```bash
make cost-xenon
# or
uv run xenon --max-absolute C --max-modules B --max-average A coleman/
```

**Example output (all thresholds met):**

```text
# Exits silently with code 0 — all modules are within threshold
```

**Example output (threshold exceeded):**

```text
/path/to/file.py has a grade of D
```

Xenon exits with a non-zero code when any threshold is breached, which
causes CI to fail.  No output on success (exit code 0).

**Current CI thresholds:**

| Parameter | Threshold | Meaning |
|-----------|-----------|---------|
| `--max-absolute` | **C** | No single block worse than C |
| `--max-modules` | **B** | No module average worse than B |
| `--max-average` | **A** | Project-wide average must be A |

!!! note "Target thresholds"
    The long-term target is `--max-absolute B --max-modules A --max-average A`.
    Current thresholds are relaxed to match the existing codebase and should
    be tightened as complexity hotspots are refactored.

### Run all structural checks at once

```bash
make cost-structural
```

### Trend analysis with Wily

[Wily](https://wily.readthedocs.io/) tracks complexity metrics across
Git history.  It is included as a dev dependency:

```bash
make cost-wily
# custom file:
make cost-wily-file WILY_FILE=coleman/bandit.py

# or individually:
uv run wily build coleman
uv run wily index
uv run wily report coleman/runner.py
uv run wily diff coleman/runner.py -r HEAD^1
```

!!! note
    `wily` with the `git` archiver requires a clean repository. The Makefile
    targets auto-detect a dirty worktree and switch to the `filesystem`
    archiver so reports still work without forcing commit/stash.

**Example output:**

```text
Found 50 revisions from 'git' archiver.
Running operators - raw,cyclomatic,maintainability,halstead
Processing |################################| 200/200
Completed building wily history.

-----------History for coleman/runner.py-----------
╒════════════╤═══════════╤════════╤══════════╤══════╤══════════╕
│ Revision   │ Author    │ Date   │ Lines    │ CC   │ MI       │
╞════════════╪═══════════╪════════╪══════════╪══════╪══════════╡
│ abc1234    │ developer │ 2026-… │ 524      │ 10.2 │ 68.4     │
│ def5678    │ developer │ 2026-… │ 510      │  9.8 │ 69.1     │
╘════════════╧═══════════╧════════╧══════════╧══════╧══════════╛
```

Each row shows how **Lines of Code**, **Cyclomatic Complexity (CC)**, and
**Maintainability Index (MI)** evolved per commit.  Rising CC or falling MI
across revisions signals growing technical debt that should be addressed
before it compounds.

---

## 2 — Runtime cost

### Scalene (CPU + memory profiling)

[Scalene](https://github.com/plasma-umass/scalene) profiles CPU time,
memory allocation, and copy volume line-by-line:

```bash
make cost-profile-scalene
# or
uv run scalene run coleman/cli.py --- --help
```

Replace `--help` with a real workload for meaningful results.  For
example, profile an actual experiment run:

```bash
uv run scalene run coleman/cli.py --- run --config run.yaml
```

!!! note
    For profiling runs with Scalene, prefer sequential execution.
    Multiprocessing and heavy concurrency can produce incomplete attribution
    (for example, what each thread executed) and, on Python 3.14, may trigger
    intermittent spawn-pool serialization errors.

    Use this in your run config:

    ```yaml
    execution:
      parallel_pool_size: 4
      force_sequential_under_scalene: true
    ```
    uv run wily build coleman
    uv run wily index
    uv run wily report coleman/runner.py
    uv run wily diff coleman/runner.py -r HEAD^1
[py-spy](https://github.com/benfred/py-spy) is a sampling profiler that
attaches to a running process with minimal overhead:

```bash
make cost-profile-pyspy
# or
uv python install 3.13
uv venv .venv-pyspy --python 3.13
uv pip install --python .venv-pyspy/bin/python duckdb numpy polars pyarrow scipy scikit-posthocs "pydantic>=2.12.5" "pyyaml>=6.0.3"
PYTHONPATH=. .venv/bin/py-spy record --rate 20 --subprocesses -o profile.svg -- .venv-pyspy/bin/python -m coleman.cli run --config run.yaml
```

!!! note
    `py-spy` may require elevated privileges (`sudo`) on some systems.
    With Python 3.14 standalone runtimes, `py-spy` may fail with
    "Failed to find python version from target process".
    The workflow above uses a dedicated Python 3.13 profiling venv,
    while your main project environment can remain on Python 3.14.
    For this workload, `--rate 20` avoids severe sampler lag and
    `--subprocesses` captures worker processes spawned by the experiment runner.
    Some `py-spy` runs may still report a teardown error after writing
    `profile.svg`; when the flamegraph is written successfully, the generated
    profile can still be used.

    Prefer `py-spy record` with a real workload. Commands like `--help`
    end too quickly and can fail before samples are collected.
    If you want a live view instead of a saved report, run:

    ```bash
    PYTHONPATH=. .venv/bin/py-spy top --rate 20 --subprocesses -- .venv-pyspy/bin/python -m coleman.cli run --config run.yaml
    ```

---

## 3 — Energy cost

### CodeCarbon

[CodeCarbon](https://mlco2.github.io/codecarbon/) estimates the carbon
emissions of compute workloads.  A minimal example lives at
`scripts/measure_energy.py`:

```bash
make cost-energy
# or
uv run python scripts/measure_energy.py
```

**Example output:**

```text
[codecarbon INFO] [setup] RAM Tracking...
[codecarbon INFO] [setup] CPU Tracking...
[codecarbon INFO] CPU Model on constant consumption mode: AMD EPYC 7763 64-Core Processor
[codecarbon INFO] [setup] GPU Tracking...
[codecarbon INFO] No GPU found.
[codecarbon INFO] The below tracking methods have been set up:
                RAM Tracking Method: RAM power estimation model
                CPU Tracking Method: cpu_load
                GPU Tracking Method: Unspecified
[codecarbon INFO] >> Tracker's metadata:
[codecarbon INFO]   Platform system: Linux-6.8.0-x86_64
[codecarbon INFO]   Python version: 3.12.x
[codecarbon INFO]   Available RAM : 7.758 GB
[codecarbon INFO]   CPU count: 2 thread(s) in 1 physical CPU(s)
[codecarbon INFO]   CPU model: AMD EPYC 7763 64-Core Processor
[codecarbon INFO] Emissions data will be saved to emissions.csv
Estimated emissions (kg CO2eq): 0.000012
```

CodeCarbon logs setup details then prints the estimated CO₂-equivalent
emissions after the workload finishes.  Results are also saved to
`emissions.csv` for tracking over time.  On cloud environments (AWS,
Azure, GCP) CodeCarbon will attempt to use region-specific electricity
carbon intensity; when unavailable it falls back to a country average.

You can adapt the script to wrap any representative workload:

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
# ... your workload here ...
emissions = tracker.stop()
print(f"Estimated emissions (kg CO2eq): {emissions}")
```

### pyRAPL (Intel hardware only)

[pyRAPL](https://github.com/powerapi-ng/pyRAPL) reads Intel RAPL
counters for direct energy measurement.  It is included as a dev
dependency but only works on supported Intel hardware:

```python
import pyRAPL

pyRAPL.setup()
meter = pyRAPL.Measurement("my_workload")
meter.begin()
# ... your workload here ...
meter.end()
meter.result.export(pyRAPL.outputs.CSVOutput("energy_results.csv"))
```

!!! warning "Platform and Python version limitation"
    pyRAPL requires an Intel CPU with RAPL support **and** read access to
    `/sys/class/powercap/intel-rapl/`.  It does not work on AMD, ARM, or
    virtualised environments without RAPL passthrough.  Additionally, pyRAPL
    is not compatible with Python 3.14 (invalid escape sequences + missing
    `pymongo` dependency cause import errors on startup).  Use CodeCarbon
    (via `make cost-energy`) as the portable fallback on all platforms.

---

## 4 — CI integration

The GitHub Actions workflow **Code Cost — Structural checks**
(`.github/workflows/code-cost.yml`) runs automatically on every pull
request and enforces two gates:

1. **Xenon complexity gate** — fails when cyclomatic complexity exceeds
   the configured thresholds (see above).
2. **Radon maintainability index gate** — fails when any module scores
   below A (MI < 20).

If either gate fails, the PR cannot be merged until the structural
cost is addressed.

Radon cyclomatic complexity is also reported (blocks with rank C or
worse are highlighted) but does not fail the build — Xenon already
covers this dimension.

The heavier tools (Scalene, py-spy, CodeCarbon, pyRAPL) are intended
for **local or manual use** and are not part of the CI pipeline.

---

## Representative workloads

| Purpose | Command |
|---------|---------|
| Structural analysis | `make cost-structural` |
| Raw metrics | `make cost-raw` |
| Trend analysis | `make cost-wily` |
| Runtime profiling (Scalene) | `make cost-profile-scalene` |
| Runtime profiling (py-spy) | `make cost-profile-pyspy` |
| Energy estimation | `make cost-energy` |

For more realistic profiling, replace `--help` with an actual experiment
configuration:

```bash
uv run scalene run coleman/cli.py --- run --config run.yaml
```

---

## Risks and constraints

- Complexity does **not** directly equal energy use.
- Energy must be measured from **runtime execution**, not inferred
  purely from structure.
- Profiling quality depends on how realistic the workload is.
- `pyRAPL` is hardware-dependent and should stay optional.
- Over-enforcement too early may create friction; thresholds should be
  tuned after first real runs.
