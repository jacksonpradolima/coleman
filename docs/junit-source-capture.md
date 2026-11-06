# Reproduce a pinned Java fault pair

COLEMAN's optional JUnit3 producer collects source evidence before benchmark
admission. Its packaged Apache Commons CLI recipe compiles two pinned upstream
revisions against the same fixed-tree test fixture and records all 95 discovered
tests. It reuses the existing source cache, controlled-observation contracts and
immutable artifact writer. Native TCPCI experiments still use the existing runner.

The recipe contains integrity metadata only. Subject sources, dependencies,
container images and raw captures remain outside the repository. It does not
execute Defects4J or claim its complete reconstruction protocol. The generic
[controlled domain](controlled-faults.md) imports no Docker, CLI or MCP adapter.

For a predeclared multi-fix workflow, the published 4.10 wheel includes a separate
[controlled Java matrix](java-fault-matrix.md), reusing this producer unchanged.
It includes CLI-13, CLI-51 and an explicitly partial CODEC-65 fixture, with
negative/inconclusive results retained.

## Prepare and run

The published 4.17 wheel includes this single-pair producer and the matrix
command; use the verified [setup command](setup.md#default-run-no-docker-required).
The bundled preset requires POSIX,
Docker, an arm64 host and arm64 Docker engine. It pins Temurin `11.0.32.1` and
`America/Los_Angeles`. Other platforms require a separately reviewed recipe;
changing a platform label does not validate a different image.

Prepare the public image explicitly:

```bash
docker pull --platform linux/arm64 eclipse-temurin@sha256:e4b3ff0f1a427efb2deb02b092ff8316d58573480260720d767d6ecdd69999d8
```

From an external workspace, using the installed environment:

```bash
python -m coleman.transports.junit \
  --cache /tmp/coleman-junit-cache \
  --output /tmp/coleman-junit-results \
  --trial source-001 --fetch
```

`--fetch` permits only the individually pinned public source/JAR downloads from
`raw.githubusercontent.com` and `repo.maven.apache.org`. The HTTPS adapter verifies
TLS and exact entity lengths, and rejects redirects, credentials, queries, partial
responses and encoded transfers. It uses no ambient authentication or proxy
configuration. The producer never pulls an image. Once cached, omit `--fetch` for
offline acquisition; subject execution has no network even during the first run.
Source-cache and output directories must be separate and physically outside
Git worktrees. The producer reuses the cache's shared storage guard to reject
repository directories, linked-worktree marker files and symlink aliases. Provided
output roots are checked before runtime queries/source acquisition; concrete cache
blob, capture-session and staging paths are rechecked before content writes.
Staged members cannot introduce case-insensitive `.git` path components. The system
temporary directory must also be outside Git for source staging. Metadata-only
exports retain their existing publication rules.

The JSON summary includes dataset/state paths, role collection statuses, pair
counts, admission and generalization. Exit 0 means both role protocols completed;
assertion failures remain recorded outcomes. Incomplete collection publishes its
evidence and returns 2. Validation, infrastructure or publication exceptions exit
nonzero. Neither exit status certifies benchmark admission or algorithm efficacy.

To supply another frozen recipe, add `--recipe /absolute/path/recipe.json`.
Recipes are bounded to 2 MiB, require schema version 1 and pin the installed
COLEMAN-authored harness. Missing, future or unknown versions reject. Updating the
harness requires an explicit reviewed recipe update, never silent migration.
Existing native results, learning checkpoints and older source artifacts retain
their meaning and require no migration.

## What the recipe freezes

The [buggy revision](https://github.com/apache/commons-cli/tree/8f46f467a33ace44ccd9f394910940b6c48a3827)
and [fixed revision](https://github.com/apache/commons-cli/tree/b0e1b80b6d4a10a9c9f46539bc4c7a3cce55886e)
are the selected public CLI-13 pair. Each production view includes all `src/java`
blobs plus `LICENSE.txt` and `NOTICE.txt`; the common fixture includes all fixed-tree
`src/test` blobs plus those notices. Every raw file and derived ZIP is separately
size/digest pinned. The deterministic ZIP fixes member order, epoch, regular-file
permissions and stored bytes; it is a COLEMAN-derived view, not upstream archive
bytes. Its contract permits at most 128 homogeneous-license members, 8 MiB per
file, 16 MiB decoded total and 32 MiB output. It adds no TAR or ZIP64 support.

The dependencies are Commons Lang 2.1 and JUnit 3.8.1. Apache subject and Commons
Lang Apache-2.0 terms remain separate from JUnit's CPL-1.0 terms. The version-specific
[JUnit POM](https://repo.maven.apache.org/maven2/junit/junit/3.8.1/junit-3.8.1.pom)
declares CPL-1.0; the recipe records the reviewed [full terms](https://opensource.org/license/CPL-1.0).
Commons Lang's terms digest identifies `META-INF/LICENSE.txt` inside its independently
pinned JAR; its `NOTICE.txt` is retained. The COLEMAN MIT license does not replace
these terms. The recipe-only policy keeps foreign content in the external cache.

```mermaid
flowchart LR
    recipe[Pinned Java source recipe] --> raw[Reviewed raw source and dependency cache]
    raw --> views[Deterministic source views]
    views --> guards[Shared optional container guards]
    guards --> producer[Optional bounded Docker producer]
    harness[Packaged COLEMAN JUnit3 harness] --> producer
    producer --> packets[Raw v1 role packets and v2 ownership markers]
    packets --> report[Controlled report plus diagnostic members]
    report --> domain[Existing offline controlled conversion]
    domain --> artifact[Immutable controlled artifact]
```

Each role uses a fresh non-root, read-only container, four explicit read-only
input mounts, no network, no added capabilities, no privilege escalation, two
CPUs, 64 PIDs, a 128-MiB temporary filesystem and a 1-GiB memory default. The
parent bounds collection to 60 seconds and 1 MiB combined stdout/stderr by default;
recipes may select only the contract's bounded ranges. These guards support
trusted public-source reproduction; they are not a hostile-code security sandbox.

Diagnostic packaging has its own explicit budget: each complete role packet is
bounded to 16 MiB, the normalized report remains bounded to 8 MiB, and the decoded
envelope total is at most 40 MiB. The shared deterministic ZIP encoder preserves
stored bytes and existing ZIP64/entry policies; source-view limits remain
8 MiB per file and 16 MiB total. This accommodates the permitted 4-MiB output
profile without dropping raw streams or parsed diagnostics. Before acquisition
or execution, the producer computes a conservative normalized-report size bound
from declared test/trial labels and maximal field sizes. Unsupported combinations
reject before a source trial can create unpublishable completed role checkpoints.
The sizing records are never published as observations.

The filtered runtime receipt retains host/Python/package versions, Docker
client/server architecture and versions, kernel and target image config digest
and size. It excludes credentials, context names, unrelated containers, image
tags and hostnames. Actual producer module bytes, recipe and harness are input
receipts, so modified implementation bytes change context identity.

## Evidence and recovery

A fresh local execution collected 190 observations: 92 same-test pairs passed in
both revisions, two failed in both and CLI-13 failed then passed. The two other
failures (`BugsTest::test19383`, `BugsTest::test27635`) remain visible. No causal
explanation or extra unique-fault count is inferred. These are observations of
this runtime and fixture, not a promise that every future run has the same result.

Each case runs in the role's shared JVM in lexical test-ID order, including JUnit
setup/teardown. Independence from test order has not been established. Its strict
integer monotonic nanoseconds mean harness elapsed time, never removable runner
cost or wall-clock CI savings. Zero is observed; unavailable counters are null.
The separate parent timer includes startup and compilation. UTC capture instants
are explicitly quantized by the producer to microseconds; higher-precision input
strings reject rather than silently truncate. No collection clock becomes
historical CI chronology or a predecision feature.

An immutable start marker precedes each role execution. A completed raw packet
replays without executing that role again. A start marker without a committed
packet recovers as `interrupted`: streams and counters remain unavailable, no
partial cases are promoted. Use a new trial ID for a fresh execution. Both trials
remain in the same dependence group. A POSIX advisory lock rejects concurrent
producers for the same context/trial; a process crash releases it. Cleanup targets
only the producer's deterministic container name. A cleanup failure propagates
without committing a role packet; the next call retries cleanup and retains the
interrupted observation state. Source acquisition, staging and
publication failures never certify a completed learning checkpoint. New start
markers use schema version 2 and persist the original output namespace and exact
container name. Their name is validated against the logical context/trial/role
before cleanup. Version 1 markers remain readable and recover using their original
state-prefix container naming rule; unknown versions reject before cleanup. V2
markers are bounded to 8 KiB and legacy v1 markers retain their 4-KiB bound. Raw
role packets and controlled datasets retain their existing version 1 contracts.

Successful protocols require the exact discovered test sequence and a final
trailer. Discovery loads excluded classes without initializing them. Output from
test-class initializers and constructors is retained in the role's bounded raw
stderr, separately from protocol stdout and per-case diagnostics/timing. Discovery
errors restore the original streams and fail collection without inventing cases.
Each case declares exactly one run and at most one terminal failure or
error; contradictory counters cause a protocol error rather than a normalized
test outcome. Malformed/partial protocols, timeouts, output limits and process errors
retain bounded raw evidence without promoting partial cases into assertions.
On replay, exited process/protocol error labels are rederived from the stored
exit code and raw protocol; relabeling a valid complete packet cannot silently
discard its observations. Unavailable streams require an unknown exit code and
unavailable parent elapsed counter; observed streams require both observed process
facts. A timeout must reach its declared parent deadline, and an output-limit
packet must retain the complete bounded prefix. These checks reject contradictory
evidence; coherent timeout/output-limit labels remain producer attestations rather
than authenticated proof of which guard fired.
Both role packets are separately digest-pinned report members; normalized
observations reference their diagnostic member digest. An entirely unobserved
trial has no invented pairs. Negative, reversed and inconclusive comparisons
remain valid. All outputs retain `not_evaluated` and `insufficient_evidence`.

## Validate the installed flow

The normal suite exercises owned fixtures without external downloads. To repeat
the actual source flow from a wheel installed outside the checkout, first populate
the source cache with the command above and prepare the pinned image, then run:

```bash
COLEMAN_JUNIT_REAL_CACHE=/tmp/coleman-junit-cache \
  uv run --extra ci --extra dev pytest tests/corpus/test_installed_junit.py --no-cov -q
```

This explicit source reproduction checks all 190 observations and the observed
pair counts, then repeats the same trial to verify byte-identical recovery. It
mocks no source compiler, JVM, cache, converter or artifact writer. It establishes
functional reproduction only. Chronological benchmark admission, additional
languages/faults, independent observations, fair baselines and all seven diversity
regimes require additional source evidence and comparative evaluation.

New v2 container ownership includes the resolved output namespace. Equal logical
trials in separate output roots cannot collide or clean each other's containers;
observation identities retain their original meaning. Keep incomplete ownership
state together when moving an output tree: a version 2 marker still identifies
the original container after relocation. Legacy v1 names retain their global
context/trial scope; their missing output namespace cannot be inferred
retroactively. Retain the original reviewed producer
package when reopening older captures; changing package bytes changes the source
context and does not automatically discover an older producer's generation.

## Continue the workflow

Expand a pair through the [Java fault matrix](java-fault-matrix.md), or inspect how its evidence is
interpreted in [controlled observations](controlled-faults.md).
