# Bounded CI application API and optional MCP

For configurable Random/UCB tuning, persistent jobs and conversational recovery,
start with [durable experiment workflows](durable-experiments.md). That opt-in
workspace uses a separate versioned contract. This page documents the original
seven ephemeral tools and their compatibility boundaries.

Expose bounded CI operations through shared application services. The application layer makes its existing
native services available through reusable, bounded commands. It does not introduce a new runner,
an LLM decision algorithm, or a hosted service.

## Install the release

Use Python 3.14 or later on Linux or macOS in an activated virtual environment:

```bash
python -m pip install 'https://github.com/jacksonpradolima/coleman/releases/download/v4.17.0/coleman-4.17.0-py3-none-any.whl#sha256=e5a272ddc0edf72bd909a743fa2b3bc0dfeb8a02871c487ab22683cddeea40ca'
```

For optional tools, install the same GitHub wheel with its extra:

```bash
python -m pip install 'coleman[mcp] @ https://github.com/jacksonpradolima/coleman/releases/download/v4.17.0/coleman-4.17.0-py3-none-any.whl#sha256=e5a272ddc0edf72bd909a743fa2b3bc0dfeb8a02871c487ab22683cddeea40ca'
```

Use this repository's release assets and verify their published digest and
attestations. The bare PyPI name belongs to a different project. Base CLI/library
execution requires no MCP dependency, model, credentials, network service, or
client. The optional adapter is tested against the pinned MCP Python SDK 2.3.0
and its negotiated MCP protocol; the independent COLEMAN application contract is
`coleman-application-v1`.
Contributor `make install` includes the optional SDK to exercise its tests and
type annotations; installing the base release wheel still excludes it.

## Architecture and ownership

```mermaid
flowchart TD
    Library[Typed library commands] --> Service[ApplicationService]
    CLI[coleman application] --> Commands[Versioned command boundary]
    MCP[Optional local MCP stdio tools] --> Commands
    Commands --> Service
    Registry[Host-approved immutable registry] --> Service
    Identity[Trusted caller context] --> Service
    Service --> Queue[Bounded queue and lifecycle supervisor]
    Queue --> Worker[Owned subprocess with deadline and resource guards]
    Worker --> Coverage[Native finite coverage solver]
    Worker --> Allocation[Native scenario allocation service]
    Worker --> Runner[Existing api.run and TCPCI runner]
    Coverage --> Publish[Existing ArtifactWriter: complete immutable JSON]
    Allocation --> Publish
    Runner --> Publish
    Publish --> Receipt[Schema-validated caller-owned result handle]
    Receipt --> Portable[Explicit portable export and import]
```

Domain packages never import CLI or MCP. Native `run`, sweeps, extensions, sinks,
checkpoints and unrestricted library APIs retain their existing behavior. The
application service owns a unique private session directory under a trusted
host-selected root. Clients select registered IDs; they cannot send a RunSpec,
plugin import, path, shell command, dataset URL or exporter endpoint.

The packaged registry contains three **synthetic smoke fixtures**:

| Spec ID | Native service | Meaning |
|---|---|---|
| `weighted-overlap-3` | Finite coverage | Overlapping A/B/C capabilities, modeled duration in seconds |
| `signed-ci-resource-3` | Scenario allocation | Declared joint scenario model with conservative evidence handling |
| `native-ci-smoke-v1` | TCPCI runner | Two synthetic builds, Random baseline, RNFail, seed 42, fixed-time budget 3 s |

These fixtures validate integration and mathematics; they are not independent
CI observations or evidence of practical time savings. Seeds do not create
independent observations. Native allocation retains its insufficient-evidence
and conservative fallback semantics.

## Complete library example

Run from outside the checkout after installing the wheel:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from coleman.application import (
    ApplicationService, Caller, Parameters, ResultBundle, Submit, default_registry,
)
from coleman.selection import CoverageResult

caller = Caller(subject="local-analysis")  # supplied by trusted host code
with TemporaryDirectory() as directory:
    with ApplicationService(default_registry(), root=Path(directory)) as service:
        print(service.capabilities().to_json())
        request = Submit(
            schema_version=1,
            request_id="ci-budget-3",
            spec_id="weighted-overlap-3",
            parameters=Parameters(budget=3),
        )
        initial = service.submit(request, caller=caller)
        final = service.wait(initial.job_id, caller=caller)
        assert final.state == "completed" and final.handle is not None
        bundle = service.result(final.handle, caller=caller)
        assert isinstance(bundle.result, CoverageResult)
        assert bundle.result.selected_ids == ("A", "C")
        assert bundle.result.cost.value == 3
        assert bundle.result.objective.value == 1
        assert service.submit(request, caller=caller).job_id == final.job_id
        exported = service.export_result(final.handle, caller=caller)
        Path(directory, "portable.json").write_text(exported, encoding="utf-8")
        assert ResultBundle.from_json(exported) == bundle
        imported = service.import_result(exported, caller=caller)
        assert imported.handle is not None
        assert imported.handle.digest == final.handle.digest
```

Save the export outside a temporary directory to retain it after shutdown.
Coverage budget overrides are exact nonnegative decimals, at most twelve decimal
places, in the registered cost unit and meaning. JSON serializes them as decimal
strings. Zero is valid; null means **no override**. Negative budgets, non-finite
values, unknown keys and implicit unit conversions are rejected. Allocation and
experiment parameters are frozen in their registrations in v1.
Application JSON readers preserve decimal number literals before precision
validation, so extra digits cannot disappear through intermediate binary64
rounding. Existing native readers retain their original compatibility behavior.

One-shot CLI execution uses the same command and services:

```bash
coleman application --request '{"schema_version":1,"request_id":"ci-budget-3","spec_id":"weighted-overlap-3","parameters":{"budget":3}}'
coleman application --request '{"schema_version":1,"request_id":"native-ci","spec_id":"native-ci-smoke-v1"}'
```

CLI output is a canonical JSON envelope. Success contains `ok: true` and `data`;
errors contain `ok: false`, a stable `error` and safe human `message`. A failed or
expired one-shot job returns the same error envelope, includes its typed state
in `data`, and exits with status 2. Existing
`coleman run --config ...` remains available.

## Native experiment registration by trusted hosts

For your own approved local dataset, create a normal RunSpec in host code, set
one sequential worker, one to sixteen executions and an explicit seed; use the
Random baseline, one or two RNFail/TimeRank rewards, and one to eight budgets.
Disable checkpoints, telemetry, hooks and extensions; use the local Parquet sink.
Call `register_experiment("approved-ci", spec)` and pass the returned immutable
`Configuration` to `ApplicationService`. Registration snapshots up to eight
`<datasets_dir>/<dataset>/features-engineered.parquet` files, totaling 1 MiB, with
at most 10,000 rows per dataset and 1,000 candidates per build. Dataset IDs use
the conservative application ID alphabet. This bounded adapter deliberately
supports a smaller surface than unrestricted native RunSpec execution.
The seed and every repetition offset (`seed + iteration_index`) must fit the
unsigned 64-bit range shared by the native NumPy/Polars execution path. Negative
seeds and overflowing repetition ranges are rejected during registration.

Inputs are copied into an owned worker directory; original files are never
reread during execution. Normalized relative input/output paths make the native
resolved run ID independent of temporary directories. Configuration fingerprints
bind complete input bytes and normalized RunSpec. Work units are total builds ×
executions × rewards × budget values. An over-limit experiment is rejected before
queue admission; work-limited decisions can return an explicitly limited native
result. V1 does not resume application jobs through native checkpoints. Existing
direct checkpoint APIs remain available separately.

Registration recomputes build/candidate dimensions from the snapshotted bytes.
The bounded adapter requires integer build IDs and binary verdicts, unique
testcase names per build, numeric finite nonnegative durations in seconds and
non-null native required columns. Invalid rows are rejected; this adapter never
invokes the legacy loader's null-duration-to-zero coercion. The direct legacy
loader remains compatible with its existing callers. Algorithm overrides are
outside the approved Random baseline registration surface in v1.
Build IDs must also survive the native loader's signed Int64 conversion; values
outside that range are rejected before execution.
Registration also checks the actual native duration conversion (Float64 loader
then the bandit's Float32 schema): overflow and positive-to-zero underflow are
rejected. Genuine zero and representable subnormal durations remain valid;
ordinary native rounding is preserved and native result bytes keep that meaning.
Budget values must produce unique native directory segments of at most 255 UTF-8
bytes. This also rejects different ratio budgets rounded to the same native label;
existing unrestricted native naming remains unchanged.
`CalcPrio` must be an integer column fitting the bandit's signed Int32 schema.
`LastRun` must survive the existing string conversion; unsupported nested values
are refused. Duration validation applies the loader's Float64 conversion first,
so supported decimal/integer source columns retain their native behavior.

Experiment bundles contain original resolved-specification and native result
bytes with logical filenames, byte counts, SHA-256 and canonical base64.
`provenance.json` in a produced bundle is explicitly a
`portable-native-provenance-v1` derivative: it records schema version, COLEMAN
producer/version, Python/platform runtime facts, the original local file's
SHA-256, and the redacted field names `cwd`, `git`, `uv_lock_hash`. Ambient host
directories, Git context and lock-file context are not exported. The original
native provenance file remains unchanged inside the worker until cleanup;
the source digest permits host-side audit and does not authenticate a producer.
Import preserves supplied evidence and never extracts files. Existing Parquet fields retain their
legacy meanings; unavailable metrics are not invented or converted to zero.
Runtime observations, timestamps and native shard filenames can differ across
equivalent executions. Mathematical results and approved-source provenance are
compared separately from runtime metadata and complete artifact digests.

## Limits, states and failures

`Limits` requests are reduced componentwise by `ServerPolicy.caps` before work.
Default caps are 30 elapsed seconds including queue/startup, 10,000 native work
units, 100 candidates, 512 MiB worker peak RSS/solver allocation ceiling and
256 KiB combined native output files plus portable result JSON. Native solver working-memory estimates cap at
1 GiB even if a host permits larger worker RSS. Native proof/dimension ceilings
still apply. The service defaults to one active worker, eight pending jobs,
64 retained job identities, 16 MiB reserved results and 2 GiB reserved active
worker memory. Limits count bytes, seconds and documented native work units;
they are not calibrated probabilities or scientific quality guarantees.

Admission reserves each job's maximum output until expiry and sums admitted
worker memory before starting processes. Registry snapshots are bounded to 32
configurations of 1.5 MiB each. Parent/interpreter overhead, input snapshots,
protocol frames and temporary serialization buffers add bounded local overhead
outside worker RSS. Linux worker peak RSS uses the current executable's kernel
`VmHWM`, because `getrusage` can include the parent's footprint before `exec`.
Other platforms retain the existing sampler; historical general telemetry and
stored artifacts keep their existing meaning. The memory ceiling is unchanged.
Worker RSS and native-file output are checked before/after
calculation and sampled every 20 ms; transient overshoot is possible. Unsupported
RSS measurement fails closed. This is a guard for trusted local native code,
not a hard memory allocator or hostile-code sandbox. The underlying SDK stdio
framing is for a trusted local client and is not an untrusted network ingress
limit. Hosted deployment needs its own authenticated, size/concurrency-limited
gateway and process/resource isolation; v1 provides none.
Pending admission counts actual capacity after active memory reservations:
unused concurrency slots do not expand the pending queue when memory prevents
their use. Jobs that fit an available slot start before `submit` returns;
otherwise the finite pending-queue cap applies. A zero queue cap permits only
immediate starts. Portable evidence imports require storage, not a worker slot.

Completion retains only the validated portable JSON bundle. Native work files,
input copies and private request files are removed after the worker exits, before
the completed receipt is issued; the bundle preserves native result/specification
bytes and the explicit redacted provenance derivative. Retained output reservations therefore cover the entire completed job
directory. Before final publication, the combined native-file and bundle byte
count must also fit the admitted output cap.

```mermaid
stateDiagram-v2
    [*] --> queued: validated and admitted
    queued --> running: worker and memory slot
    queued --> cancelled: cancel wins
    queued --> failed: worker startup fails
    queued --> expired: elapsed deadline
    running --> completed: exit zero and validated immutable JSON
    running --> failed: native error or resource refusal
    running --> cancelled: cancel wins
    running --> expired: elapsed deadline
    completed --> expired: retention TTL
    failed --> expired: retention TTL
    cancelled --> expired: retention TTL
```

Cancellation and completion commit under the same lock: the first terminal state
wins. Cancelling a completed job returns completion unchanged; cancelling queued
or running work terminates/reaps its worker and exposes no partial result. A
complete native incumbent with a work-limit status is valid evidence, not an
optimality claim. Native failures never become fabricated negative scientific
results. Only exit zero plus valid, matching native result JSON permits completion.

Retention starts at a terminal state, defaults to one hour, and deletes result
bytes on expiry. Status retains an `expired` tombstone; retrieval returns the
stable expiry error. Request IDs are scoped by trusted caller: identical retries
reuse the same job, changed commands conflict, and expired IDs never silently
start new work. Tombstones count toward the finite session job limit. After that
limit, start a new service session deliberately. `close()` cancels work and removes
session files. Restart loses job handles; orphaned crash directories require
trusted operator cleanup. Export/import is the explicit portable continuation,
not a claim to resume an interrupted native calculation.

| Error | Meaning |
|---|---|
| `invalid_input` | Invalid schema/version/parameter, changed request ID or wrong issued digest |
| `unknown_spec` | No such approved configuration |
| `not_found` | Job absent from this service session |
| `forbidden` | Existing job belongs to a different trusted caller |
| `expired` | Deadline or result retention expired |
| `resource_limit` | Input, native workload, queue, job, output or memory limit |
| `unavailable_result` | No completed result, corrupted artifact, worker failure or closed session |

Status is inspectable before a handle exists. A handle includes schema version,
job ID and content digest. Ownership is checked independently **before** exposing
its result. A digest proves integrity, not producer identity or authorization.
Imports recheck decision mathematics and opaque experiment-file hashes, then issue a fresh local
handle to the importing caller; they do not attest the claimed producer or run
any imported content. Imported experiment observations remain producer claims,
not independently rerun evidence. Bundle v1 requires integer version 1; missing, future and
unknown versions fail closed. Legacy native artifacts keep their existing readers;
there is no automatic migration or reinterpretation.

## Configure MCP once, then use tools

After installing the optional wheel extra, point an MCP-capable local client at
the absolute executable `/tmp/coleman-application-env/bin/coleman-mcp`, with no
arguments. This starts a local stdio server with the packaged registry. A trusted
custom host can instead call `create_server(service, caller=...)` with its own
approved registry. Each process has one host-owned local principal; caller fields
inside requests are rejected. There is no HTTP listener or hosted authentication.

Once configured, the client can discover configurations, submit a bounded CI
request, poll status, cancel, retrieve and export/import results through tools.
The user does not run a shell command for each request. No model is required to
exercise these operations; a natural-language client is only another consumer.

| Tool | Argument | Response |
|---|---|---|
| `discover` | none | IDs, kinds, fingerprints, budget units/meaning, policy and submit JSON schema |
| `submit` | `request_json` | Versioned `JobView` in success envelope |
| `status` / `cancel` | `command_json`: `{"schema_version":1,"job_id":"..."}` | Job state |
| `result` / `export_result` | `handle_json`: exact issued handle | Portable bundle in `data` |
| `import_result` | `bundle_json`: serialized exported `data` | New local completed job and handle |

Commands have a 16 KiB cap before JSON parsing; imports use the server output
cap. MCP JSON framing and escaping add overhead to the bundle byte cap. Unknown
fields, duplicate JSON keys, traversal/import/URL attempts and future versions
fail before execution. Standard SDK protocol/schema failures remain MCP errors;
valid tool invocations return the versioned COLEMAN success/error envelope.

The no-model tests use the actual optional SDK client both in process and over
stdio, then repeat the stdio lifecycle from an installed noneditable wheel outside
the checkout. They also verify base installation without MCP, real two-build
native execution, CLI/library/tool mathematics, caller isolation, races, immutable
publication, golden fixtures, portability, expiry, malicious inputs and saturation.

## Continue the workflow

Choose your integration boundary in the [integration guide](integration-guide.md), or return to the
[native experiment API](api/experiment-api.md) when a registered application session is unnecessary.
