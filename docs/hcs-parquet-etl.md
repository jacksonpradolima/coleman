# LibSSH: original CI data to variant-aware Parquet and native replay

Original CTest logs can additionally be [corroborated against the archived
CSV and job metadata](hcs-trace-evidence.md), preserving process states without
silently changing the old collapsed nonpass labels or temporal admission.

Use a pinned public historical LibSSH slice to exercise **Whole Test Set (WTS)**
and **Variant Test Set (VTS)** evaluation. Preserve pipeline/job identity, inspect
sparse configurations, compare their native result denominators and resume both
strategies through COLEMAN's existing runner and checkpoints.

The published **4.17 wheel** includes this workflow. Install it outside the
repository and supply the two reviewed original files below. Source integrity
and a successful replay do not establish scientific admission or a winning policy.

## Install the published wheel outside the repository

With Python 3.14+ and uv available:

```bash
uv venv /tmp/coleman-hcs-env --python 3.14
uv pip install --python /tmp/coleman-hcs-env/bin/python "https://github.com/jacksonpradolima/coleman/releases/download/v4.17.0/coleman-4.17.0-py3-none-any.whl#sha256=e5a272ddc0edf72bd909a743fa2b3bc0dfeb8a02871c487ab22683cddeea40ca"
mkdir -p /tmp/coleman-hcs-downloads
cd /tmp/coleman-hcs-downloads
```

Keep source files, cache and outputs outside Git. No Docker, compiler, MCP,
benchmark subject execution or upstream extractor installation is required.

## Download the two reviewed original files

The source is Prado Lima, Jackson (2021), *LIBSSH Mirror – Testing Results
(GitLab CI) – Version 2*, Harvard Dataverse, version 2.0,
[doi:10.7910/DVN/I1XAVR](https://doi.org/10.7910/DVN/I1XAVR).
The [released metadata](https://dataverse.harvard.edu/api/datasets/:persistentId/versions/2.0?persistentId=doi:10.7910/DVN/I1XAVR)
explicitly dedicates these unrestricted data to **CC0-1.0**. The package records
the metadata digest, review date, source/file version, exact byte counts and
SHA-256 pins. No third-party data or implementation is bundled with COLEMAN.
Data permission does not grant permission to copy or execute upstream code.

```bash
curl --fail --location --user-agent 'COLEMAN/corpus-integrity-v1' \
  'https://dataverse.harvard.edu/api/access/datafile/4378418?format=original' \
  --output executions.csv
curl --fail --location --user-agent 'COLEMAN/corpus-integrity-v1' \
  'https://dataverse.harvard.edu/api/access/datafile/4378387?format=original' \
  --output jobs.json
```

| Role | Original bytes | SHA-256 |
| --- | ---: | --- |
| Parsed execution CSV | 32,898,759 | `ba50a92a4571fec2b520f1155f1f8a8377236d44d95f3d6c683726bd9a2968c6` |
| Original job JSON | 33,004,397 | `19b8652f902c20706fb32b7e9bcf3cb4859c701b6a900cd3954152c563454cd0` |

The CLI verifies both hashes and sizes before conversion. Dataverse's default
ingested **TAB is different from the original semicolon CSV**: `format=original`
is required. The publisher's TAB MD5 is not a checksum of the original CSV.
The original JSON's published MD5 was independently verified. The CSV SHA-256
and byte count are explicit reproducibility pins, not a claimed publisher SHA.

## Convert, execute both strategies and resume

```bash
/tmp/coleman-hcs-env/bin/python -m coleman.transports.hcs_etl \
  --executions executions.csv --jobs jobs.json \
  --cache /tmp/coleman-hcs-cache --output /tmp/coleman-hcs-output --run

# Identical offline retry: reuse verified sources and existing native checkpoints.
/tmp/coleman-hcs-env/bin/python -m coleman.transports.hcs_etl \
  --cache /tmp/coleman-hcs-cache --output /tmp/coleman-hcs-output --run
```

Use `--strategy wts` or `--strategy vts` to execute only that strategy. Without
`--run`, the command exports and verifies Parquet only. `--recipe path.json`
accepts a compatible bounded typed source recipe; no URL is opened implicitly.
Native logs go to stderr and stdout contains one JSON summary with completion,
accounting, variant aliases and native artifact/run paths. Raw data stay in the
content-addressed source cache.

V1 accepts two raw files of at most 64 MiB each, 500,000 parsed CSV records,
16,000 selected records, 16,384 job records, sixteen pipelines, 64 observed
variant names and 1,000 tests per native suite. CSV counters/positions refer to
parsed records after the header. Recipe files have a 2 MiB wire read limit and a
1 MiB canonical metadata limit, reserving space for bounded completion receipts.
Oversized metadata rejects before acquisition or cache/output creation.

The frozen first-sixteen-pipeline slice has these descriptive counts:

| Evidence | Count |
| --- | ---: |
| Complete source CSV rows | 278,950 |
| Selected physical rows | 885 |
| Rows outside the slice, retained in original cached bytes | 278,065 |
| Other-attempt rows explicitly excluded | 29 |
| Retained native variant test executions | 856 |
| Surrogate union test rows | 320 |
| Labels with parsed CSV observations | 7 |
| All selected job labels, including metadata-only | 13 |
| Metadata-only labels without parsed observations | 6 |
| Declared pipeline × known-job-label cells | 208 |
| Cells with retained parsed tests | 44 |
| WTS result rows, including union and per-variant results | 240 |
| VTS result rows | 176 |

The two policies are Random and UCB/RNFail, at ratio budgets 0.5 and 1.0 with
seed 42. These are functional reproduction counts, not independent observations
or evidence of policy superiority. The complete raw file has **44 observed
variant names**, including all-pass variants. The publisher summary's 34 variants
that failed at least once are not the full universe; this bounded preset does not
claim to exercise all 44.

The six metadata-only labels are `CentOS7/libgcrypt`, `CentOS7/libgcrypt/x86-64`,
`Fedora/openssl 1.0.x/x86`, `Fedora/openssl 1.1.x/x86`, `FreeBSD/x86-64` and
`static-analysis`. They remain visible with metadata job counts and absent parsed
evidence across all declared pipelines. A known job label is not automatically an
observed test configuration: no test semantics are inferred for `static-analysis`.
Their native VTS tables are empty and they create no native executions. Missing
reported nonpass counts remain null. An independent raw-source reference checks
all 208 cells, including these uncovered rows, rather than only the active subset.

## Contract, identities and explicit costs

The execution reader joins each original `(pipeline_id, job_id, test_name)` to its
job's pipeline, commit and untouched variant label. A pipeline cannot have
contradictory commit bindings. Numeric pipeline order defines modeled rounds,
not verified chronological outcome availability. Repeated commits remain
different pipelines; shared variants, labels, trials and the earlier overlapping
LibSSH dataset are not independent projects or faults.

Variant aliases are portable digests with an explicit reverse mapping in the
receipt. `linux.x` and `linux_x`, case differences, slashes and original Unicode
names are preserved as distinct identities. No configuration feature or true
rename relationship is inferred from punctuation.

Identical repeated content at a source execution key gets an explicit duplicate
disposition; conflicting content rejects. For multiple parsed completed jobs with
one variant/pipeline, select the **lowest job ID**, independently of outcome or
duration, and retain the other rows as exclusions. Unsupported job status,
unavailable outcome and unavailable duration have separate primary dispositions.
Every physical row is conserved; an excluded attempt is not silently max-merged.

The new recipe requires the cost model
`additive-binary32-rounded-reported-test-seconds-v1`. Original decimal seconds
and their spelling remain in `observations.parquet`; native projected seconds
and their meaning are separate columns. The existing bandit stores float32
duration. Explicitly rounding all new native inputs to that precision before
deriving budgets aligns the loader, bandit and variant evaluator; the projected
values are carried exactly in binary64 Parquet columns. This changes neither
legacy inputs nor the existing bandit schema. It models test costs; it does not
measure CI occupancy or promise exact decimal arithmetic at arbitrary scales.
V1 also requires each projected suite's total coefficient on its finest binary
cost lattice to fit 53 bits. Then all nonnegative partial sums are representable
in binary64 regardless of ranking order. Unsupported dynamic ranges reject
before export completion; no tolerance silently omits or promotes a boundary test.

WTS learns on one declared surrogate per pipeline: maximum projected duration
and any reported nonpass for each shared test label. Its ranking projects onto
each actual variant suite. VTS uses separate policy state for each variant.
Per-variant budgets and outcome denominators use that variant's own tests.
Native result `step` counts processed scenarios. Resolve its source pipeline via
the sorted active `BuildId` values of the corresponding union or variant input,
then the receipt/cell mapping. Fully excluded rounds retain their original
modeled IDs in analytical evidence; they do not consume a native processed step.
The union is not another set of observed executions and its time is not the sum
of all variant work. Native RNFail is budget-censored: zero nondetection reward
for an unscheduled case is not an observed pass.

Observed zero duration remains zero. Empty source duration/outcome stays null
and is excluded from native input; negative/nonfinite time and unsupported units
reject. Upstream extraction interprets reported time as seconds, collapses all
nonpass statuses into binary one and omits whole jobs with zero total test time.
Its linked historical parser is methodology evidence, not a verified historical
producer revision. The native legacy result columns `detected` and `missed`
therefore count this source's **reported nonpass test executions**, including
potentially timeout/skipped/not-run/exception states already collapsed upstream.
They cannot be interpreted as unique faults or independently verified failures.

`cells.parquet` preserves absent and fully excluded cells. A missing parsed cell
does not become a passing test, verified missing configuration, zero setup cost
or completed run. Job start/finish labels remain source metadata; neither proves
when the prioritizer received a label or feature. Setup seconds, shared runner
occupancy, outcome/feature availability and upstream exclusion counts remain
unavailable. Use the existing [resource ledger/replay](resource-replay.md) for
separately supported observations; do not add dummy setup tests or equate summed
test seconds with physical work saved. Protected temporal admission is blocked.

## Files, compatibility and interrupted execution

```mermaid
flowchart LR
    CSV[Original execution CSV] --> Cache[Existing verified external cache]
    JSON[Original job JSON] --> Cache
    Cache --> Join[Typed pipeline/job/commit join]
    Join --> Evidence[Selected observations and conserved exclusions]
    Join --> Cells[Complete sparse cell matrix]
    Join --> Native[Explicit projected test seconds]
    Native --> WTS[Union and variant Parquet]
    Native --> VTS[Per-variant Parquet]
    WTS --> Runner[Existing RunSpec/API and native runner]
    VTS --> Runner
    Runner --> Pair[Verify persisted result/checkpoint pairing]
    Pair --> Recovery[Reconcile uncommitted Parquet tail on resume]
    Recovery --> Runner
    Runner --> Results[Per-policy/budget/variant results and checkpoints]
```

Export identity binds the full recipe, actual converter/storage code and Polars
writer version. Fixed-profile Parquet and a canonical versioned receipt permit
deterministic offline verification. Tables commit through the existing immutable
writer; `complete.json` is written last. Interrupted export can retain verified
individual tables, and identical concurrent retries converge. Validation rebuilds
expected bytes from the source; editing a Parquet file and re-signing its receipt
does not certify source truth.

The native API persists the exact receipt in its manifest. Multi-budget native
checkpoints use `checkpoints/budget-v1/<mode-value-digest>/<dataset>/`; close ratio
values that round to the same percent remain distinct. Single-budget checkpoint
paths and payloads are unchanged. An old unscoped multi-budget progress marker
is ambiguous and **rejects before loading it**, including nested run IDs such as
`team/project`. Detection covers the entire configured checkpoint tree. New
`budget-v1/<digest>` markers are distinguished by their original run ID in at
most 1 MiB of progress JSON; a legacy name resembling that namespace cannot
bypass detection. Malformed, oversized or mismatched scoped run-ID metadata
blocks continuation. No checkpoint pickle is read during this preflight.
Retain the original package and output; explicitly start a fresh output tree
rather than guessing a migration.

For built-in native scenarios using local checkpoints and the Parquet sink,
resume first verifies a persisted ownership binding between results and their
checkpoint directory, then removes rows beyond the last committed processed step for that
exact dataset, execution number and budget. This includes a crash after result
flush but before the first checkpoint: all rows from that uncommitted attempt
are replayed. Already committed observations and other coordinates are retained;
unchanged shards remain byte-identical. A progress marker without loadable state
rejects before deleting results. This does not repair corrupt or missing result
files that predate the checkpoint, or deduplicate previously committed evidence.

Before the first native row, version-one private metadata pairs each result
coordinate with a persistent checkpoint-directory token. A new or emptied
checkpoint directory cannot reclaim another run's result rows, even when dataset,
experiment number and budget match. Such reuse rejects before deletion; retain
the old data and use fresh paired output paths. Future or corrupt stored
ownership metadata rejects. Tokens are operational identities; they do not change
mathematical source/observation digests or claim independent observations.
Unbound completed legacy coordinates can be adopted only after loading their
checkpoint and checking that no result exceeds its committed step. This adds
ownership metadata while preserving original rows and checkpoint payloads.
Unbound legacy tails remain ambiguous and reject; no migration guesses a run.
New native shards also retain their owner token in versioned Parquet metadata.
An unmanaged writer's tail is never reclaimed as another run's work; it rejects
instead. One sink instance belongs to one checkpoint owner.
Successful native runs commit their final processed round even when shorter than
the configured checkpoint interval, so a completed run is not replayed as an
uncommitted tail. Fully restored completed runs preserve their result bytes.

Each native Parquet shard is completed under a temporary name before atomic
publication. A crash between shard publications leaves complete uncommitted
shards for reconciliation and temporary files that readers/manifests ignore.
Reconciliation itself uses atomic shard replacement and is safe to retry after
interruption. Errors block execution and manifest publication. Discard an instance
whose flush failed and resume with a fresh API call. Serialize attempts of the
same run; independent executions use distinct shards. Other sinks/custom scenario
providers keep their existing recovery semantics. Power-loss durability and
untrusted checkpoint files are outside this local process-interruption contract.

Missing/future schema versions or a missing explicit native cost declaration
reject. Old native Parquet, corpus profiles, source readers and their artifacts
retain their original APIs/meaning. Unknown producer or writer versions require
the recorded package/version; there is no silent conversion. In particular, the
publisher's prepared commit-factorized CSVs remain unchanged: their duplicate
identities are why this adapter uses raw pipeline/job records instead.

## Library API and inspection

After downloading the originals, the same workflow uses the core contracts and
existing application API without a CLI or network transport:

```python
from functools import partial
from importlib.resources import files
from pathlib import Path

import polars as pl

from coleman.application.hcs import run_hcs_replay
from coleman.artifacts import ArtifactWriter
from coleman.corpus.hcs_contracts import HCSRecipe, HCSReceipt
from coleman.corpus.hcs_etl import export_hcs_parquet, validate_hcs_etl
from coleman.corpus.materialization import SourceCache, materialize_source

recipe = HCSRecipe.from_json(
    files("coleman.corpus.resources").joinpath("libssh-first16-pipelines-v1.json").read_bytes()
)
cache = SourceCache("/tmp/coleman-hcs-cache")
downloads = Path("/tmp/coleman-hcs-downloads")
for package, name in ((recipe.executions, "executions.csv"), (recipe.jobs, "jobs.json")):
    materialize_source(package, cache, partial((downloads / name).open, "rb"))
completion = export_hcs_parquet(recipe, cache, ArtifactWriter("/tmp/coleman-hcs-output"))
receipt = HCSReceipt.from_json(completion.read_bytes())
validate_hcs_etl(receipt, cache, completion.parent)
print(pl.read_parquet(completion.parent / "cells.parquet"))
for strategy in ("wts", "vts"):
    result = run_hcs_replay(
        completion, cache, Path("/tmp/coleman-hcs-output/native") / strategy, strategy=strategy
    )
    rows = pl.read_parquet(str(Path(result.artifacts_dir) / "results/**/*.parquet"), hive_partitioning=True)
    print(rows.select("scenario", "variant", "step", "policy", "budget_value", "detected", "missed"))
```

Independent raw-source reference checks validate the retained cohort, exclusions,
surrogate union, projected budgets and per-cell denominators. The qualification
also checks differences against original decimal quantities; it is not a new
metric implementation or a protected policy comparison. From this checkout,
after populating the cache with the commands above, reproduce the installed-wheel
qualification and its independent reference:

```bash
COLEMAN_HCS_REAL_CACHE=/tmp/coleman-hcs-cache \
  uv run --extra ci --extra dev python -m pytest tests/corpus/test_installed_hcs.py --no-cov
```

The committed `tests/corpus/hcs_raw_reference.py` streams the original CSV and
job JSON without calling the converter, projection, metric or scheduler. It checks
every expected policy/budget/variant cell and compares projected prefixes with
exact rational original-decimal quantities. Authored mutation tests verify that
incorrect counts, duplicate orders and nonfinite totals fail; a declared modeled
cost boundary difference remains visible. No original benchmark bytes are bundled.
Scientific source
admission, repeated-measures inference and cross-corpus claims remain with the
[remaining public roadmap](roadmap.md).

The [original CTest evidence guide](hcs-trace-evidence.md#verify-a-log-from-the-original-licensed-release)
also verifies a selected job log against the complete released CC0 `logs.zip`,
with separate complete-last archive provenance. This does not grant historical admission.

## Continue the workflow

Review [WTS and VTS concepts](highly-configurable.md), corroborate [original CTest
outcomes](hcs-trace-evidence.md), then retain grouping in [repeated
observations](corpus-evaluation.md).
