# Plan, tune and recover a CI experiment

Compare Random with UCB, vary its exploration coefficient, leave the conversation
and return to the same experiment. COLEMAN saves the executable plan, runs its
existing TCPCI engine and retains native Parquet, checkpoints and evidence. An
MCP client translates a conversation into tools; a model never owns execution or
scientific interpretation.

Start with the small offline history below. It has three builds and three tests
per build, with durations in seconds and binary recorded verdicts. The default
plan has **four configurations**, **eight seed trials** and **24 native build
rows**. These are different quantities. Two seeds do not create two independent
repositories.

This guide describes the implementation in this source revision. The previously
published 4.19.1 wheel has the seven bounded application tools, without these
durable workflow tools. Build/install this revision to use the examples; the
installed-package acceptance test exercises the resulting noneditable wheel
outside the checkout.

## Install and initialize once

Use Python 3.14+ on Linux or macOS, with private local storage. Build the wheel
from the reviewed checkout, then install it into a separate environment:

```bash
coleman_workflow_dist="$(mktemp -d)"
uv build --wheel --out-dir "$coleman_workflow_dist"
python3.14 -m venv /tmp/coleman-workflows-env
for coleman_workflow_wheel in "$coleman_workflow_dist"/coleman-*.whl; do
  /tmp/coleman-workflows-env/bin/python -m pip install "${coleman_workflow_wheel}[ci,mcp]"
done
/tmp/coleman-workflows-env/bin/coleman workflow \
  --workspace /tmp/coleman-workspace init-fixture
```

Use the actual wheel produced by the build; the temporary directory contains
that build alone. `ci` supplies native execution and this workflow's bounded
protected comparison; its bounded-mean method does not require `analytics`.
Normal-model paired inference is a separate analytical capability. The `mcp`
extra is optional: library and CLI workflows run without that SDK. See
[component installation](core-installation.md). No model, API key, Docker or network collector is
needed after package installation and source preparation.

The workspace belongs to the host. Keep it outside Git. It contains private
learning checkpoints; trust the machine and filesystem that own it. Tools receive
registered IDs and retained handles, rather than arbitrary paths or commands.

## Connect a client

The executable accepts one trusted setup argument:

```bash
/tmp/coleman-workflows-env/bin/coleman-mcp --workspace /tmp/coleman-workspace
```

Stdout is reserved for the MCP protocol. Do not paste conversational text into
this process. A client starts it and invokes the discovered tools. A common
stdio client configuration is:

```json
{
  "mcpServers": {
    "coleman": {
      "command": "/tmp/coleman-workflows-env/bin/coleman-mcp",
      "args": ["--workspace", "/tmp/coleman-workspace"]
    }
  }
}
```

Client configuration formats vary. The verified client for this guide is the
pinned **MCP Python SDK 2.3.0**, negotiating protocol **2026-07-28**, over both
in-process and actual stdio transports.
No claim of testing every desktop assistant is implied. The following complete
client runs without a model:

```python
import asyncio
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

async def main():
    parameters = StdioServerParameters(
        command="/tmp/coleman-workflows-env/bin/coleman-mcp",
        args=["--workspace", "/tmp/coleman-workspace"],
    )
    async with Client(parameters) as client:
        discovery = await client.call_tool("workflow_discover", {})
        assert discovery.structured_content["error"] is None
        planned = await client.call_tool("plan_experiment", {
            "request": {
                "schema_version": 1,
                "dataset_id": "owned-ci-three-build-v1",
                "policies": ["Random", "UCB"],
                "ucb_exploration": [0.5, 1.0, 2.0],
                "seeds": [42, 43],
                "budget": {"mode": "ratio", "unit": "1", "values": [0.5]},
                "limits": {"max_configurations": 12, "max_trials": 8},
            }
        })
        plan = planned.structured_content["data"]
        assert (plan["configurations"], plan["trials"], plan["expected_build_rows"]) == (4, 8, 24)
        submitted = await client.call_tool("execute_plan", {
            "request": {"schema_version": 1, "plan_id": plan["plan_id"], "request_id": "first-comparison"}
        })
        job = submitted.structured_content["data"]
        print("Retain this job:", job["job_id"])
        while job["state"] in ("queued", "running"):
            await asyncio.sleep(0.1)
            response = await client.call_tool("experiment_status", {
                "request": {"schema_version": 1, "job_id": job["job_id"]}
            })
            job = response.structured_content["data"]
        assert job["state"] == "completed", job
        report = await client.call_tool("experiment_report", {
            "request": {"schema_version": 1, "job_id": job["job_id"]}
        })
        result = report.structured_content["data"]
        assert result["trial_count"] == 8
        assert result["native_build_rows"] == 24
        print(result["interpretation"])

asyncio.run(main())
```

The service knows declared defaults and reports their versioned origin through
discovery. It still requires a registered dataset. It rejects missing fields,
unknown/future schema versions, arbitrary configuration fields and work exceeding
the host's caps before execution. There is no additional COLEMAN confirmation
step between a valid execute request and its admission.

An incomplete MCP draft such as `{"request":{"schema_version":1}}` returns
`{"schema_version":1,"contract":"coleman-workflows-v1","data":null,"error":"invalid_input","missing_fields":["dataset_id"],"reason":null}`.
An explicit null dataset has the same response. The client can discover approved
IDs and ask the user to choose one. This transport draft is never saved or run;
the application still requires a strict, complete `PlanningRequest`.

## What a conversation maps to

The utterances below are **illustrative**. The accounting beneath them comes from
the actual SDK/native integration tests, not a model-generated demonstration.

> Which histories and policies can I use?

`workflow_discover` returns the approved dataset manifests, exact input hashes,
units, evidence restrictions, host limits and typed planning schema.

> Compare Random and UCB. Try c = 0.5, 1 and 2, with seeds 42 and 43 and half
> the modeled suite duration. Show the plan first.

`plan_experiment` returns four configuration IDs, eight trial IDs and the complete
coordinates. Random appears once per budget, with `exploration: null` meaning
**not applicable**. UCB choices are not multiplied into extra Random baselines.
The full retained plan includes every resolved native specification and the
producer/runtime identity; execution never asks a model to reconstruct it.
Local and portable readers recompute the deterministic choice from retained
native rows and require the matching same-budget Random fallback. Descriptive
selection uses all declared rows; protected selection uses validation rows only.
An alternate planned ID or null cannot silently replace either reported choice.

> Run that plan. I will come back later.

`execute_plan` returns a durable job ID. An identical request key returns that
same job across reconnects; the same key with another payload conflicts. Closing
the client detaches it. The fixed host-owned worker continues independently.

> Continue the experiment I started earlier.

`experiment_status` inspects the saved job. If physical ownership was lost,
`resume_experiment` checks compatibility and reuses completed trials while native
checkpoints continue incomplete ones. Successful offline crash/reconnect testing
retains eight logical trials and 24 rows with **nine physical trial invocations**:
one interrupted trial was continued, rather than counted as a new seed/configuration.

> Show the quality metrics, measured runtime and assumptions. Export the evidence.

`inspect_experiment` provides bounded scalar summaries and ranking digests;
`experiment_report` returns a compact completed overview.
`list_experiment_artifacts` exposes immutable plan/report/native file handles,
with the original source-projection receipt when available.
`read_experiment_artifact` downloads verified byte ranges up to 64 KiB.
`export_experiment` also supports a small self-contained portable bundle.
Importing such a bundle creates **read-only evidence**, never an executable plan,
registered history or resumable checkpoint.

Inspection includes unfinished coordinates and their typed physical attempt
events. Each native row's `attempt_id` matches its actual legacy `execution_id`;
its process samples belong to that attempt. A resumed table can retain earlier
committed rows and their original physical sampling origins. Complete rankings
remain in the original Parquet and full report artifacts, rather than entire
testcase histories being placed in model context.

## Verified tool response projections

These JSON projections are asserted by the real optional-SDK and native-runner
acceptance tests. They omit runtime-dependent IDs, timestamps and other response
fields; they are not complete wire payloads. Re-run the client above to obtain
those fields for your installed wheel and workspace.

| Tool | Selected actual response fields for the authored fixture |
| --- | --- |
| `plan_experiment` | `{"configurations":4,"trials":8,"expected_build_rows":24,"evaluation":"descriptive"}` |
| `experiment_status` after completion | `{"state":"completed","completed_trials":8,"total_trials":8,"attempts":8}` |
| `experiment_report` | `{"trial_count":8,"native_build_rows":24,"interpretation":"descriptive_only"}` |
| `experiment_status` after interrupted-trial recovery | `{"state":"completed","completed_trials":8,"total_trials":8,"attempts":9}` |

A separate protected fixture stops with `state: "awaiting_evaluation"` and 16
prefix rows. Its final report has `interpretation: "insufficient_evidence"`,
retains the Random fallback and includes the existing evaluator's
`too_few_independent_units` reason. The report does not describe its eight trial
coordinates as eight independent repositories.

## Read the numbers correctly

| Quantity | Meaning and unit |
| --- | --- |
| `napfd` | Original native NAPFD quality, dimensionless; larger is better |
| `apfdc` | Original native APFDc quality, dimensionless; the legacy Parquet column is named `cost` |
| `modeled_full_seconds` | Binary64 `math.fsum` of recorded source durations; modeled seconds, not measured replay wall time. Native testcase costs separately cross the existing binary32 boundary |
| `process_cpu_seconds` | Observed cumulative CPU sample in seconds for the physical native attempt, or null if unavailable |
| `process_wall_seconds` | Observed cumulative elapsed sample in seconds for that physical attempt, or null if unavailable |
| `tests_ran`, `tests_not_ran` | Simulated selected/unselected testcase counts; replay does not launch the subject tests |
| `attempts` | Physical native trial invocations; retries are not independent evidence |
| `eta_seconds` | Explicit unavailable measurement until an estimator is supported; no invented remaining time |

Zero is an observed value when the native contract permits it. Null is not zero.
Unavailable values retain their status. Random's missing UCB coefficient is not
applicable. Non-finite values reject, rather than becoming JSON NaN or a clipped
metric. A ranking/reward is not a calibrated failure probability.

## Download a full artifact without filling model context

A client can stream the complete report into a local file while the conversation
uses its compact overview. This verified SDK pattern checks both chunk and
whole-file digests; the server accepts handles, never arbitrary filesystem paths:

```python
import base64
import hashlib
import json
from pathlib import Path

async def download_report(client, job_id, destination):
    """Download a completed report through verified, host-issued artifact handles."""
    response = await client.call_tool("list_experiment_artifacts", {
        "request": {"schema_version": 1, "job_id": job_id, "limit": 2}
    })
    artifact = next(
        item for item in response.structured_content["data"]["artifacts"]
        if item["kind"] == "report"
    )
    parts, offset = [], 0
    while offset is not None:
        response = await client.call_tool("read_experiment_artifact", {
            "request": {
                "schema_version": 1, "job_id": job_id,
                "artifact_id": artifact["artifact_id"], "offset": offset,
                "maximum_bytes": 65536,
            }
        })
        chunk = response.structured_content["data"]
        part = base64.b64decode(chunk["content_base64"], validate=True)
        assert "sha256:" + hashlib.sha256(part).hexdigest() == chunk["chunk_sha256"]
        parts.append(part)
        offset = chunk["next_offset"]
    payload = b"".join(parts)
    assert len(payload) == artifact["size_bytes"]
    assert "sha256:" + hashlib.sha256(payload).hexdigest() == artifact["sha256"]
    destination.write_bytes(payload)
    report = json.loads(payload)
    return report

# Inside main(), while the SDK Client context is open:
# report = await download_report(client, job["job_id"], Path("report.json"))
```

This code runs in the client, without an LLM parsing every result row. Paginate
the artifact listing to retrieve original native Parquet and the approved source
receipt. Private learning checkpoints never appear in that membership. The
library exposes the same operations as `artifact_page`, `read_artifact` and
`report_overview` in `coleman.application.workflow_downloads`.

## Run the same saved plan from Python or CLI

```python
from pathlib import Path
from coleman.application.workflows import WorkflowService
from coleman.workflows.contracts import PlanningRequest
from coleman.workflows.lifecycle import ExecutePlan

service = WorkflowService(Path("/tmp/coleman-workspace"))
preview = service.plan(PlanningRequest(
    schema_version=1, dataset_id="owned-ci-three-build-v1",
))
complete_plan = service.saved_plan(preview.plan_id)
assert complete_plan.fingerprint == preview.digest
job = service.execute(ExecutePlan(
    schema_version=1, request_id="library-comparison", plan_id=preview.plan_id,
))
print(service.status(job.job_id).to_json())
service.close()  # Detaches; it does not cancel or remove the job.
```

```bash
coleman workflow --workspace /tmp/coleman-workspace discover
coleman workflow --workspace /tmp/coleman-workspace plan \
  --request '{"schema_version":1,"dataset_id":"owned-ci-three-build-v1"}'
# Use the actual retained IDs printed by plan/execute:
coleman workflow --workspace /tmp/coleman-workspace execute \
  --request '{"schema_version":1,"plan_id":"ACTUAL_PLAN_ID","request_id":"cli-comparison"}'
coleman workflow --workspace /tmp/coleman-workspace status --job ACTUAL_JOB_ID
coleman workflow --workspace /tmp/coleman-workspace inspect --job ACTUAL_JOB_ID --limit 2
coleman workflow --workspace /tmp/coleman-workspace export --job ACTUAL_JOB_ID > evidence.json
coleman workflow --workspace /tmp/coleman-workspace import --input evidence.json
```

`ACTUAL_PLAN_ID` and `ACTUAL_JOB_ID` are placeholders, not valid executable IDs.
CLI setup may read trusted local files; MCP requests never accept these paths.

Admission waits up to ten seconds for the coordinator to acknowledge observing
the queue in durable metadata. A busy process lock alone is insufficient: it
could belong to a retiring owner. If acknowledgement times out, the job remains
durably admitted; retry the same request and idempotency key instead of creating
a second experiment. The versioned, bounded coordination counters are operational
metadata, not independent scientific observations.

A newly spawned worker can see the earlier `queued` WAL snapshot while its
scheduler is committing the `running` claim. It reads only the immutable plan,
then synchronizes with that transaction and rereads the state within the attempt
deadline. A committed claim starts native work; rollback or cancellation does
not. The inherited worker lock remains held throughout this startup handshake.

The same coordinator keeps its kernel lock and retries transient SQLite
`BUSY`/`LOCKED` errors during startup and polling. A host's snapshot publication
may hold the write transaction longer than a connection's one-second wait; this
must not abandon an acknowledged queue. Other storage/schema failures remain
explicit failures. These retries do not create jobs or consume native attempts.

Opening an existing valid WAL workspace checks one consistent read snapshot,
without taking its writer lock. During execution, checkpoint progress, attempt
reservation and completion transactions retry numeric contention only within
the current attempt's remaining monotonic deadline. Each SQLite wait is shortened
to that remaining allowance. Only BEGIN/COMMIT commands retry: the body runs once,
so native reservations and immutable artifact writes are not repeated. Cancellation
is rechecked after acquiring the write lock. Exhausted attempts stop native work;
terminal failure bookkeeping retains its original bounded, one-second fail-fast
cleanup when no retry allowance remains. None of these operational waits changes
source clocks or the declared count of independent scientific observations.

An attempt interrupted before its first committed checkpoint can retry from the
beginning. Once durable job metadata acknowledges a native commit, a missing
checkpoint marker rejects continuation before another native invocation; it does
not silently reset learning. Restoring the exact original private files permits
compatible continuation. Portable evidence cannot supply those executable files.
An unsuccessful restoration retains that acknowledged frontier as well: a later
retry cannot erase the obligation to continue the original learned state.
Restoring an otherwise valid checkpoint/marker pair from an earlier build also
rejects before RNG restoration or native output reconciliation; it cannot rewind
work that durable job metadata already acknowledges.

Native checkpoints also retain the global Random policy generator through the
existing `CheckpointPayload.extra` extension, with a versioned PCG64 codec and
complete plan/trial binding. The native agents already retain their own RNG.
Before restoring that generator, the loader checks that the actual payload's
run, experiment and step exactly match its committed progress marker. A mismatch
rejects continuation without changing the generator state.
Ordinary older native checkpoints lack this workflow context and cannot be
silently adopted or reseeded. The existing native pickle/progress formats and
non-workflow checkpoint behavior are unchanged. The saved plan also records its
actual Polars thread pool; restart with that same host setting.

For the unreleased v5 source, new plans additionally bind the explicit
runtime-version-2 CI dependency selection. Runtime-version-1 historical plans
remain readable evidence and require their original pinned v4 environment for
execution/continuation. The outer workflow, SQLite store and existing PCG64
codec versions remain unchanged. See
[native runtime identities](core-installation.md#keep-native-runtime-identities-explicit)
for the exact boundary; installing extras does not migrate an old checkpoint.

## Keep tuning separate from protected evaluation

For the owned fixture, plan with `evaluation: "protected"`. Protected execution
requires `max_attempts` of at least two, covering the tuning worker and
the separate evaluation worker. A one-invocation protected plan rejects before
sweep expansion or admission. Descriptive replay can use one invocation.

Temporal decisions are joined by observation **and revision** identity back to
native build order. Alphabetical observation labels never change the learning
sequence, validation choice or holdout membership. Protected stages must form a
contiguous training/validation/test sequence in that native order. Before sweep
expansion, the planner validates the same repetition design used by evaluation:
at most **16,384 test-build/seed cells per candidate**, complete membership and
no duplicated protected source revisions or sibling groups. Seeds remain
repetitions; they do not create independent repository families.

The fixture's host-authored metadata declares train build 1, validation build 2
and test build 3. Native
source filtering happens **before** eager outcome materialization. Each trial
executes only its training/validation prefix, without a final forecast into the
holdout. The job stops at `awaiting_evaluation`: eight trials and **16 prefix rows**.

Call `evaluate_experiment` with its job ID. The application freezes the validation
choice, same-budget Random baseline, actual learning-state snapshots, seed/build
aggregation design and full bounded-mean hypothesis/protocol before the existing
audit ledger reserves test access. The best UCB challenger is compared with Random
even if Random won validation; ties prefer Random, then smaller UCB c, then
canonical identity. Nothing activates a production policy.

Local and portable report readers require exactly one frozen learning-state
entry for each planned seed of each selected configuration. Trial identities,
seed values and unique checkpoint references must agree with that plan. These
checks establish internal consistency; they do not authenticate an external
producer. Report interpretation and the exact ordered source/mode-specific
assumptions must agree with the declared plan; imports cannot remove the
constructed-source, repetition or no-production-promotion qualifications.

**Portable native run IDs are producer claims, not verified provenance.** The
native configuration hash includes private execution paths absent from a portable
bundle. Parquet bytes bind metrics and logical trial membership, but cannot prove
that path-derived ID. V1 receipts retain the original field and canonical bytes;
no migration silently upgrades its trust. `EvidenceBundle.native_run_ids_verified`
and read-only import handles explicitly return `False`, and inspection summaries
label `native_run_id_verification="producer_asserted_unverified"`. This also
qualifies native run IDs carried into derived mean records. A consumer needing
trusted producer identity must establish that trust separately.

**Native execution attribution also remains a producer assertion.** Legacy
Parquet does not carry an authoritative seed, treatment/configuration or source
binding; policy/budget Hive labels are supplied by its producer. Swapping two
same-configuration seed claims can leave every native byte unchanged. A valid
logical trial ID defines the declared coordinate; it does not prove that a file
was generated by that execution.

| The portable reader checks | Separate trust is still needed for |
| --- | --- |
| Canonical planned coordinate definitions and declared membership | Assigning native bytes to a particular trial/seed |
| Included metric values, file digests and internal arithmetic | Which policy, reward, budget and source produced those values |
| Exact source/mode qualifications and protected report consistency | Source authenticity, empirical admission and producer honesty |

`native_execution_bindings_verified` is therefore always `False` on bundle and
receipt objects, read-only import handles, trial inspection and report summaries.
It refers to independent verification from portable native bytes. A trusted local
producer may have the execution context, but an importer must establish that trust
separately or reproduce the run. Derived comparisons are conditional on the declared
attribution. The new qualifiers do not rewrite native Parquet or canonical v1
plan/receipt/report/bundle bytes; valid historical digests remain unchanged.

Readers reconstruct the unit evidence from unchanged native receipts through the
same producer used during execution. Observation/revision identity, source event,
target interval, complete mean record and provenance must match that projection.
All units retain one declared assembly timestamp. An eligible date or the right
numeric mean alone cannot substitute for the registered source chronology.

The selected challenger and Random continue their **original native checkpoints**
for both seeds. Four complete continuation receipts contain **12 cumulative rows**:
eight retained prefix rows plus four newly evaluated test rows. Other UCB choices
remain completed prefix evidence. Do not add these cumulative counts to the
prefix count and call the sum independent observations.

The existing repeated-corpus machinery computes equal seed means within each
build, then equal build means within each repository family. Metric assembly has
its actual current timestamp; it is never backdated to source label availability.
After a worker owns evaluation, it freezes a current bounded assembly window.
A long queue or a later resume therefore does not inherit an expired operational
cutoff. Each worker invocation retains its own immutable execution context.
The original source membership, stage boundaries, selected candidates, prefix
learning state and hypothesis family stay fixed. The original handoff bytes stay
unchanged, and every context keeps the same upstream holdout ID and access history.
Changing this assembly bookkeeping never creates a fresh confirmatory holdout.
Direct use of an expired context rejects before another protected-access receipt.

The fixture has one constructed family. Its successful evaluation returns
`insufficient_evidence`, with `too_few_independent_units`; seeds do not increase
that count. A second experiment on the same source retains the previous holdout
access in the shared ledger. Reading an already completed report does not reload
holdout outcomes or manufacture another access.

Descriptive full-history execution and imported outcome evidence also leave
durable source-exposure markers. Before protected access, those markers become
development-access receipts in the existing audit, retaining their original
operational timestamps separately. Import structurally scans the v1 JSON envelope
without decoding its report or file values, then validates only the source/version
header. It reserves exposure before interpreting outcome-bearing JSON, decoding
native Parquet or checking retention capacity. A later validation or quota failure therefore preserves conservative
exposure. Repeating the exact payload reuses that attempt; different encodings
can overcount accesses while retaining the same logical evidence handle.
At most 1024 distinct import attempts are retained globally per workspace,
including rejected attempts. Repeating an exact payload reuses its reservation,
even at capacity. Exhaustion rejects before outcome decoding; reservations never
expire or recycle silently. A crash before source marking can consume an unused
slot, but cannot authorize an unrecorded outcome read.
A crash during audit integration can
conservatively overcount exposure; it cannot erase it. The scope follows stable
upstream source identity, so a different Parquet encoding, candidate or protocol
does not make a previously exposed holdout appear new. The ledger audits this
workspace's API; it cannot detect an external reader or authenticate host source
declarations. Preserve exposure when moving workspaces, including by importing
previous outcome evidence, rather than claiming a fresh independent holdout.
The source lock covers exposure integration and the subsequent audit-capacity
check. If those receipts fill the 512-access ledger, evaluation rejects before
another holdout read; it does not discard earlier exposure to make room.

Evaluating an already descriptive job rejects with an explicit explanation;
create a separate eligible protected plan. The protected report preserves a
same-budget Random `fallback_configuration` when evidence is insufficient. Its
`selected_configuration` records the validation choice, without activating it.

This initial protected integration supports **constructed mechanisms**, one ratio
budget and Random/UCB RNFail. Host registration alone cannot grant empirical
source admission. An original source lacking reviewed admission, point-in-time
clocks or predecision scheduling-cost evidence remains descriptive; eligibility
errors are not repaired by inventing dates or independent units. Other native
budget modes remain available for descriptive replay.

## Register a real local history

Trusted host code prepares approved native Parquet in a temporary directory,
then admits and publishes it under the workspace's transactional quotas:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from coleman.application.workflow_registry import snapshot_history
from coleman.application.workflows import WorkflowService

service = WorkflowService(Path("/tmp/my-ci-workspace"))
with TemporaryDirectory(prefix="coleman-host-registration-") as temporary:
    staging = Path(temporary)
    manifest = snapshot_history(
        staging,
        Path("/approved/local/ci/features-engineered.parquet"),
        dataset_id="reviewed-ci-history",
        source_fingerprint="sha256:" + "a" * 64,
        projection_fingerprint="sha256:" + "b" * 64,
        evidence="descriptive_source",
        assumptions=("Recorded outcome replay; no empirical protected admission.",),
    )
    service.register(manifest, snapshot_root=staging)
```

Replace the illustrative path and hashes with the **actual reviewed source and
projection fingerprints**. The API cannot verify an upstream scientific claim
from a caller-provided label. It does verify exact native bytes, shape, finite
seconds, binary outcomes, unique build/test membership and immutable registration.
A friendly ID cannot later be rebound to different bytes or declarations.

CLI registration uses the same temporary preparation boundary. Rejected capacity,
identity or manifest checks leave no new snapshot in the workspace. An interrupted
publication can leave bytes before the metadata commit; those unregistered files
consume physical dataset slots and bytes. They cannot bypass subsequent quotas,
and retrying the same reviewed snapshot reuses its unchanged bytes. Cleanup never
guesses that an unregistered file is safe to delete. The older workspace-prepared
`register(manifest)` call remains available, but `snapshot_history` itself is a
preparation function and has no quota or admission authority. Keep original-source
ETL output outside the workflow workspace as well.

For a pinned original-source starting point, prepare the existing
[Karaf Parquet workflow](lrts-parquet-etl.md). Its receipt conserves 6,876 original
method observations: 6,567 runnable rows and 309 not-run exclusions across eight
builds. Preserve the receipt, source hashes, projection hash, exclusion counts,
assumed ordering and post-hoc duration limitations when registering that output.
Source acquisition remains a separate, explicit host setup operation; tools do
not download arbitrary benchmark URLs.

With its permitted cache already populated, register the approved preset offline:

```bash
# Use a fresh host workspace with an explicit 64 MiB output reservation ceiling.
python - <<'PY'
from pathlib import Path
from coleman.application.workflows import WorkflowService
from coleman.workflows.contracts import WorkflowLimits
from coleman.workflows.lifecycle import HostPolicy

WorkflowService(Path("/tmp/karaf-workspace"), policy=HostPolicy(
    schema_version=1,
    caps=WorkflowLimits(
        max_configurations=64, max_trials=128, max_output_bytes=67108864,
    ),
))
PY
coleman workflow --workspace /tmp/karaf-workspace register-karaf \
  --cache /tmp/coleman-lrts-source-cache --source-output /tmp/coleman-karaf-etl
coleman workflow --workspace /tmp/karaf-workspace plan \
  --request '{"schema_version":1,"dataset_id":"karaf-lrts-eight-v1","limits":{"max_trials":8,"max_output_bytes":67108864}}'
```

The host reuses the existing pinned ETL and validates its original source bytes,
receipt and output before registration. The complete source receipt remains in
`source-receipts/`; discovery exposes its projection fingerprint, exclusions and
eligibility restrictions. The default eight-trial plan expects **64 native build
rows**, rather than the owned fixture's 24. Missing cache entries fail without
network access. Acquisition still uses the separate explicit source workflow.

The installed-wheel acceptance run produced four configurations, eight trials and
64 native rows through both the library and a reconnected stdio client. Their
mathematical digest was
`sha256:e271c088528c47774542d23961a280edb5f9f3550f5313ed74ed7614db13d7e8`.
All 67 approved artifacts were listed; the full report and original source receipt
were downloaded in verified chunks. The report occupied approximately 3.94 MB;
native outputs, retained checkpoints and evidence occupied approximately 44.7 MB
per job. That is why this host setup explicitly reserves 64 MiB instead of silently
raising the normal 16 MiB default. Observed timing/artifact bytes can vary between
physical executions; the comparison digest excludes those operational samples.

To reproduce the complete installed-library/stdio comparison, use the
[executable Karaf acceptance example](https://github.com/jacksonpradolima/coleman/blob/main/examples/durable-experiments/verify_karaf.py).
Copy it from this checkout, then run it with the noneditable wheel's Python
outside the repository, using the same permitted pre-populated cache:

```bash
cp examples/durable-experiments/verify_karaf.py /tmp/verify_coleman_karaf.py
cd /tmp
/tmp/coleman-workflows-env/bin/python /tmp/verify_coleman_karaf.py \
  --cache /tmp/coleman-lrts-source-cache --output /tmp/coleman-karaf-proof
```

It runs both actual paths, disconnects/reconnects stdio, verifies all counts and
download digests, compares every mathematical row and writes `proof.json`, the
tool-response trace, downloaded artifacts and original source receipt under the
private output directory. An identical retry reuses retained jobs. Keep that
source-derived evidence outside Git; this example never downloads missing inputs.

This success demonstrates a pinned retrospective replay and transport parity.
It does not grant empirical source admission, independent repository observations
or prospective CI performance evidence.

## Recovery, limits and compatibility

| Situation | Behavior |
| --- | --- |
| Client disconnect / stdio restart | Reopen the same workspace and job ID; workers and completed evidence survive |
| Coordinator crash while a worker is alive | A retained kernel lock proves physical ownership; another coordinator does not duplicate execution |
| Worker crash / ownership loss | Running becomes interrupted; resume is explicit and compatibility checked |
| Cancel versus completion | The first committed lifecycle transition wins; resume waits until physical ownership is released |
| Changed dataset, configuration or producer/runtime | Continuation rejects; create an explicit new compatible plan rather than rewriting old evidence |
| Crash during publication | Temporary/orphan files do not certify completion; complete-last receipts and SQLite transitions govern handles |
| Expired result | Explicit cleanup removes terminal job bytes; the retained idempotency key returns the same expired job |
| Unknown schema / workspace version | Reject without silent migration or interpretation of future fields |
| Imported evidence | Read-only inspection/export; no checkpoint deserialization or execution |

The default host permits one running worker, eight queued jobs, 64 retained job
slots, 64 saved plans and 16 datasets. Retained tombstones and plans keep their
slots for the workspace lifetime. Use a new explicitly selected workspace when
those finite slots are exhausted; keys are never silently recycled.

Each default job reserves 16 MiB output and 1 GiB sampled worker memory; one
worker invocation has 300 seconds. The default allows four invocations per job
and at most four physical invocations per logical trial, including protected continuation.
Worst-case build work includes retries before expansion. Queue time/job age,
worker wall time and modeled scheduling seconds remain separate quantities.
The output reservation includes the job's input snapshot copy, native results,
checkpoints and execution contexts. Registration's 64 MiB input limit is separate
from this reservation. Planning rejects an input that alone fills or exceeds its
output allowance, before expansion; increase the allowance or reduce the history.
The remaining space still has to accommodate the actual produced artifacts.
Planning also reserves capacity under the worker's **32,768 directory/file entry**
ceiling, before sweep expansion. Its conservative bound is
`128 + trials × (64 + max_attempts × (5 × builds + 16))`: shared metadata,
per-trial directories, per-build native results/checkpoints, both stage evidence
copies, frozen learning and transient/attempt receipts. The complete tree is
reserved again for each possible invocation. Reduce trials, builds or retries
when that bound is exceeded; a larger byte allowance does not raise this ceiling.

Limits are sampled/cooperative local guards with possible transient overshoot,
not an untrusted-code sandbox, hard OS memory ceiling, distributed lock or
guarantee of exactly-once physical computation. Inline export is limited to 2 MiB
JSON, 1 MiB original native bytes and 64 MiB aggregate decoded Parquet row groups;
footer checks precede materialization. Larger evidence remains available through paged artifact handles and verified
64 KiB byte-range downloads. Trial inspection has a 512 KiB response cap; summaries
exclude full rankings, and a smaller page remains available for larger histories. Default terminal retention is one day;
`coleman workflow --workspace PATH cleanup` performs explicit expiry.

Without `--workspace`, `coleman-mcp` keeps its original **seven ephemeral tools**
and cleanup semantics. Their application-v1 contracts and Random-only adapter
are unchanged. Durable mode adds a separate `coleman-workflows-v1` namespace; it
does not relax an old reader or alter legacy native Parquet/run-ID meanings.

## Understand the method and evidence

The original TCPCI approach is described by Prado Lima and Vergilio,
[*A Multi-Armed Bandit Approach for Test Case Prioritization in Continuous
Integration Environments*](https://doi.org/10.1109/TSE.2020.2992428),
**IEEE Transactions on Software Engineering**. This integration adds durable
planning, reproducibility and an explicit evidence boundary around the native
loop; the original study does not establish superiority for every new workload.

The pinned Karaf source comes from Cheng et al.,
[*Revisiting Test-Case Prioritization on Long-Running Test Suites*](https://doi.org/10.1145/3650212.3680307),
**ISSTA 2024**. Its source/component terms and qualification limits remain
separate from COLEMAN's code license. A successful retrospective replay does not
establish prospective CI benefit or original execution/availability clocks.

Continue with [policies](policies.md), [experiment design](experiments.md),
[protected evaluation](protected-evaluation.md), [bounded paired
means](bounded-paired-evaluation.md), [corpus aggregation](corpus-evaluation.md)
and the [workflow contract and recovery diagrams](workflow-contracts.md).
