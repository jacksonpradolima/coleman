# Moving existing experiments to v5

The v5 architecture keeps CI/TCPCI as a supported application while making the
decision components independently usable. An existing test-ranking experiment
does not become a generic bandit experiment merely by changing its package
version. Its action, observed feedback, reward and recorded units retain their
original meanings.

This inventory was checked against **4.20.0**, source revision
`1edbee874129aa85e26e592d9649826e85e1ac6b`. It records the compatibility contract
for the v5 work, rather than announcing that its adapters, migration tools or
lightweight core have shipped. The architecture change itself moves no public
Python modules and rewrites no retained data. See the
[architecture specification](v5-architecture.md) for the component boundaries.

The integration source now includes [versioned generic decision contracts](decision-contracts.md)
and shared primitive implementations below result/selection/temporal consumers.
Their old namespaces retain compatible reexports and existing wire semantics.
This step does not publish v5, migrate executable checkpoints or complete the
planned TCPCI adapter. [Component installation](core-installation.md) describes
the explicit base, CI and analytical dependency profiles in the integration source.

The integration source also provides a [complete native semantics catalog and explicit pre-evaluation capture](native-policy-semantics.md).
This additive evidence route preserves the original algorithms, names and raw
result readers. It does not automatically wrap `Environment`, backfill old rows,
turn rankings into sampled actions or qualify every policy's checkpoint state.

The [bounded replay API](causal-replay.md) adds separately versioned policies,
acceptance records and caller-owned continuation. Existing observations and core
feedback retain their wire meanings. This route does not turn a legacy native
checkpoint into generic state or provide a durable acknowledgement protocol.

The separate [owned-state foundation](state-recovery.md) adds pure typed
snapshots and an optional local acknowledged collection in a new root. Its
retained replay uses explicit generation2 and an opt-in ordinal/seal profile;
original replay generation1 readers and content remain unchanged. No native
pickle, workflow workspace or freeform historical trace is silently converted.

## Choose an execution route

Use a **pinned v4 environment** to continue an existing executable experiment or
workspace with its original producer, dependency and runtime identities. Retain
the original wheel, lock, data, configuration and trusted checkpoint directory.
Use a separate **v5 environment** for the qualified TCPCI adapter or a new generic
decision contract. Do not install the two versions into the same environment or
point both at one mutable workspace.

The v5 release must provide an installed-package TCPCI example that accepts the
unchanged supported native configuration and preserves its qualified behavior.
That is a migration acceptance requirement, not permission to load any old
pickle under a new runtime. An unsupported legacy policy, plugin or checkpoint
must produce an explicit incompatibility explanation and the pinned-v4 route;
it must not be silently replaced by a similarly named algorithm.

Implementation branches target `release/v5`. That integration branch does not
publish releases; the final qualified merge to `main` produces the single stable
v5 release. Prerelease version selection is exercised in isolated dry runs. See
[release integration](v5-release-integration.md) for the workflow and maintenance
routes. The current package version stays 4.20.0 during this architecture change.

The installation source remains this repository's verified **GitHub release
wheel**. The bare PyPI name `coleman` belongs to another project. Follow the
[installation guide](getting-started.md#1-install-in-a-virtual-environment) for
wheel verification and the [release history](release-history.md) to select an
existing version. A planned v5 release is not an installable release asset.

## How to read the inventory

Each row below has one primary disposition:

| Disposition | Meaning for the v5 delivery |
| --- | --- |
| **Retained** | Preserve the documented contract and existing import or serialized namespace. Moving its implementation behind that boundary requires conformance evidence. |
| **Shimmed/deprecated** | Preserve a bounded legacy facade inside v5, with explicit mapping into the TCPCI application. New generic code uses the new capabilities. A shim must not conceal a semantic correction. No removal date or runtime deprecation warning is introduced by this architecture document. |
| **Intentionally breaking** | A new contract cannot impersonate the v4 contract. Use an explicit new namespace/generation and a reviewed conversion, or retain the original execution in pinned v4. |
| **Readable-only** | Retain a validated inspection/import route for evidence. Reading does not authorize execution, resume, provenance upgrades or new scientific claims. |

These are required delivery dispositions. Where qualification or conversion is
still pending, **pinned v4 is the executable route available today**. Each v5
reader, shim and migration route must acquire its own release evidence before it
is advertised as supported. Existing v4 readers remain usable in their pinned
environment throughout that work.

The bounded public surface consists of the documented API modules and package
exports linked below, their documented configuration and transport commands, and
the published data contracts they consume. A row covering a package includes its
explicit public exports and documented source-extension classes. Underscore
helpers, private database tables, ambient module globals and incidental internal
imports are not additional stable extension APIs. Source-extension support does
not imply automatic binary or pickle compatibility.

## Python, configuration and transport surfaces

| Supported v4 surface | Disposition and v5 route | Original-behavior route and limits |
| --- | --- | --- |
| `coleman.api`: `RunResult`, `run`, `run_many`, `run_with_extension`, `sweep`, `load_spec`, `save_resolved` | **Shimmed/deprecated.** Keep the documented native facade and call shapes at the TCPCI boundary. New generic learners and solvers do not accept a RunSpec. | Pinned v4 executes the original engine. `RunResult.metrics` is currently empty; it must not acquire invented aggregate observations. Native files are the result source. See [experiment API](api/experiment-api.md). |
| `coleman.spec` exports: `RunSpec`; execution, experiment, budget, algorithm, HCS, contextual, results, checkpoint, telemetry and hook models; `SweepAxis`, `SweepSpec`, expansion/name/pack resolution and run ID helpers; documented `coleman.spec.io` functions | **Shimmed/deprecated.** Preserve native validation, defaults, config-pack merge order and grid/zip meaning in the legacy facade. Generic plans use a separately versioned contract. | Pinned v4 accepts the original resolved models/YAML. Most native model sections forbid extra keys; `AlgorithmSpec` and the declared extension options deliberately have different extension behavior. See [configuration](configuration.md). |
| Native YAML sections, `packs`, top-level `sweep`, seed repetitions, `extensions` and `hooks` | **Shimmed/deprecated.** Interpret these as native TCPCI configuration, never as implicit generic capability registration. Keep ratio, seconds and selected-test-count budget modes distinct. | Keep original files and packs with v4. A changed default/profile requires explicit versioned interpretation, not regeneration of old configurations with new defaults. Seeds repeat algorithm randomness, not independent source observations. |
| `coleman.runner.RunnerExtension`, `run_experiment`, `run_experiment_with_extension`, documented Scalene/parallelism helper and source extension points | **Shimmed/deprecated.** Retain the bounded native orchestration boundary. New problem, learner and executor protocols have separate responsibilities. | Exact callbacks, worker ownership and importable source subclasses remain available in pinned v4. A v5 adapter must document callback mappings and test sequential/process execution; no promise covers arbitrary monkey-patching. See [extensions](extensibility.md). |
| `coleman.hooks`: `HookContext`, `ExecutionResult`, `DatasetResult`, `RunResult`, `RunnerHook`, functional adapters and documented lifecycle dispatch | **Shimmed/deprecated.** Preserve native plugin call shapes, configured deterministic order, coordinator/worker ownership, path-based worker loading and `fail_fast` behavior. Hook payload `duration_seconds` remains seconds; context `budget_value` retains its declared budget mode. | Pinned v4 preserves the original lifecycle and trusted class/function loading. A v5 TCPCI shim must qualify these callbacks in sequential and process modes. New generic executor events use a separate typed contract; the hook `RunResult` is not `coleman.api.RunResult`. See [hook configuration](configuration.md#hook-context-and-payloads). |
| `coleman.agent`, `coleman.bandit`, `coleman.environment`/`environment_base`, `coleman.scenarios` and loader/virtual classes; action/history/bandit schemas | **Shimmed/deprecated.** Keep these as legacy ranking/application interfaces; do not make them generic algorithm base classes. | Pinned v4 is the route for custom subclasses whose state or constructor depends on the existing runtime. The v5 TCPCI adapter must publish its supported source-extension subset and reject unmapped capabilities. See [native API index](api-reference.md#extend-native-behavior). |
| `coleman.policy` and its documented subpackage classes/configuration names, including ranking adaptations of EXP3, contextual, combinatorial, dueling and PortfolioUCB policies | **Intentionally breaking** when introducing mathematically corrected generic semantics. The additive `coleman.native` catalog identifies preserved legacy recipes, exact factory/manual routes and explicit refusals. A new sampled-action or mixture learner gets its own capabilities and implementation identity. | Pinned v4 preserves the original mechanism and its actual RNG, tie and lifetime limits, including restricted FRRMAB repeatability. The [37-name inventory](native-policy-semantics.md#the-37-native-ranking-mechanisms) records 34 completing factory recipes and three failures, not a universal success or replay guarantee. Raw policy behavior is unchanged; the explicit capture facade refuses unsupported interpretations and routes. See [policies](policies.md). |
| `coleman.reward`: base `Reward`, RNFail, TimeRank, APFDc, top-k, discounted-failure and reciprocal-rank rewards; `coleman.evaluation` native metric exports | **Shimmed/deprecated.** Preserve native reward/metric meanings at the TCPCI boundary. New loss transforms and objectives require explicit units, bounds and transform identity. | Pinned v4 reproduces the original credit assignment and metric arithmetic. Reward, ranking score, probability and inference remain separate. See [reward API](api/reward.md) and [native evaluation API](api/evaluation.md). |
| `coleman.budget`, documented native selection resolution and duration/verdict/context input conventions | **Shimmed/deprecated.** Preserve explicitly qualified legacy semantics. Strict new adapters may reject inputs accepted or coerced by the unrestricted legacy loader, and must say so. | Pinned v4 preserves loader behavior, including its documented null-duration handling. Do not relabel zero substituted by a loader as a measured zero. Native `BuildId`, `Name`, `Duration`, `Verdict`, `CalcPrio` and `LastRun` are domain columns, not generic core fields. |
| `coleman` CLI: `run`, `sweep`, `validate`, `analyze`, `application`, `workflow`; documented `python -m coleman.corpus` source commands | **Shimmed/deprecated.** Keep existing commands at their application/adapter boundary; add generic routes explicitly. No transport-owned algorithm logic. | Pinned v4 supports its exact command grammar, file layouts and error/exit behavior. Generic requests must not be accepted as arbitrary native config or plugin imports. See [CLI API](api/cli.md), [datasets](datasets.md) and [durable workflows](durable-experiments.md). |
| `coleman.application` exports: `ApplicationService`, registry/configuration and experiment registration, caller, limits/policy, requests, jobs/handles and result bundle; `coleman-application-v1` JSON commands | **Retained.** Keep this bounded ephemeral application namespace and its approved registry semantics separate from generic durable execution. | Pinned v4 retains its seven tools and original limits. Registration is intentionally narrower than unrestricted RunSpec. Restart does not resume session jobs; export/import is portable evidence continuation. See [application API](application-api.md). |
| `coleman.workflows` exports; documented `WorkflowService`, host registration, planning/lifecycle/inspection/download and evidence import/export contracts; `coleman-workflows-v1` commands | **Retained** as versioned native contracts, not widened in place. Generic plans/executors require their own namespace/generation. Mutable state and old execution use the routes in the next table. | Pinned v4 executes the frozen Random/UCB, RNFail, NAPFD/APFDc and native Parquet workflow. Retaining its reader does not admit other algorithms, rewards, arbitrary paths or imported executable plans. See [workflow contracts](workflow-contracts.md). |
| `coleman.transports.mcp.create_server`, `coleman-mcp`, original seven tools and opt-in durable workflow tools | **Retained** as optional transport adapters over their versioned services. New generic operations require typed shared services and explicit discovery. | Pinned v4 preserves its supported SDK/protocol combination. The optional `mcp` extra remains unnecessary for direct library/CLI use. A natural-language client does not own execution, caller authority or evidence interpretation. |
| `coleman.telemetry`: `Telemetry`, `NoOpTelemetry`, `get_telemetry`; native monitor/resource observations, OTLP configuration and local dashboards | **Retained** at the outer integration layer. Keep native metric names/units and attempt scope; version new generic instruments independently. | Pinned v4 preserves existing exporters and dashboards. RSS/CPU/wall-time measurements are process observations, not physical subject-test cost. Core imports must not initialize telemetry. See [observability](observability.md). |
| `coleman.analysis` and documented statistical/plotting workflows | **Retained** for the native schemas they actually consume. Generic analysis must select the matching result contract explicitly. | Existing native queries/notebooks remain usable with pinned dependencies and data. A legacy `fitness` column is not silently renamed into a generic objective or treated as a calibrated probability. See [analysis playbook](analysis-playbook.md). |

## Results, artifacts, workspaces and resumable state

Persisted evidence and executable state need different routes. A valid content
digest checks bytes; it does not authenticate a producer, establish independent
observations or make a checkpoint safe to execute.

| Supported v4 surface | Disposition and v5 route | Preservation and migration rule |
| --- | --- | --- |
| Native Parquet rows, Hive partitions and ranking hashes/top-k; embedded DuckDB and ClickHouse `coleman_results` | **Retained.** Keep native columns, units, nullability and metric meaning; use explicit new tables/contracts for generic facts. | Old results remain readable with their native schema. Runtime metadata and shard filenames need not be byte-identical across reruns. No synthetic DomainRecord or missing provenance is generated by reading old rows. See [results and storage](observability.md#choose-result-storage). |
| `ResultsSink`, `NullSink`, `ParquetSink`, `DuckDBSink`, optional `ClickHouseSink`, catalog views, `ResultsWriter` and monitor collection | **Retained** at the persistence/integration boundary. New core algorithms do not consume sink row dictionaries. | Parquet local recovery has a specific checkpoint-ownership contract. DuckDB/ClickHouse append behavior does not acquire that guarantee. Null sinks discard native rows; both NullSink and ClickHouse explicitly refuse domain-record operations. |
| `DomainRecord`, `Measurement`, `LearningReward`, `ResultProvenance`, canonical JSON/fingerprint/comparability helpers and local domain sidecars | **Retained.** Keep version 1 readers, explicit units/semantics and logical-key/content-digest distinction. | Sidecars are additive to native facts, not joint transactions with them. Legacy runs without a sidecar remain absent. Unknown versions/corrupt keys reject. Zero, negative permitted values, null, unavailable and not-applicable retain distinct meanings. See [domain results](domain-results.md). |
| Additive `FrozenNativeDecision` evidence in the v5 integration source | **Retained as a new versioned contract**, separate from old native rows and generic decisions. Its explicit chooser wrapper retains one actual pre-evaluation ranking and complete selected semantic definitions. | Old results stay readable and absent metadata stays absent. The record declares sampling `not_recorded` and loss `not_defined`; no inferred probability, bounded loss, source clock or executable state is backfilled. Same-key identical publication is idempotent; changed bytes conflict. Choice itself is stateful and not retry-idempotent. See [native evidence](api/native.md). |
| `ArtifactWriter.path_for`, mutable atomic JSON/CSV exports, immutable text/stream publication and `manifest.json` | **Retained.** Preserve the distinct mutable and immutable contracts. | Mutable exports remain single-writer. Immutable identical publication is idempotent; conflicting bytes reject. Complete-last receipts are contract-specific; a run directory or operational manifest is not universal completion evidence. See [artifact API](api/artifacts.md). |
| `spec.resolved.json`, `provenance.json`, configuration-derived native run directories | **Readable-only** as historical evidence; the corresponding API remains the native shim. | Preserve original bytes and recorded redactions. A resolved spec is not a fingerprint of all input data. Re-execution requires the original dependency/input evidence or a new run identity. Portable redacted provenance remains a declared derivative. |
| `CheckpointStore`, `LocalCheckpointStore`, `NullCheckpointStore`; `CheckpointPayload`, `.pkl` files and progress markers | **Intentionally breaking** at a generic learner/session state boundary. No automatic unpickle, conversion or adoption into generic state. | Resume trusted native state in the pinned v4 producer/runtime and its original trusted root. A v5 native codec route needs explicit compatibility checks and interruption evidence; absence of that proof means refusal, not reseeding or a fresh run labeled resume. |
| Native Parquet checkpoint owner/binding metadata and committed-step reconciliation | **Retained** only with its matching native checkpoint codec and ownership rules. | Keep result/checkpoint pairs together. Unowned/ambiguous legacy tails reject unchanged. Concurrent attempts of the same native run remain unsupported; different coordinates do not justify taking over another attempt. See [native recovery source](https://github.com/jacksonpradolima/coleman/blob/1edbee874129aa85e26e592d9649826e85e1ac6b/coleman/results/recovery.py). |
| `WorkflowRandomState`, `WorkflowCheckpointStore` and the `coleman_workflow_rng_v1` native extra-state envelope | **Intentionally breaking** for generic owned streams; retain a separately qualified legacy isolated-worker wrapper where needed. | This envelope binds complete global PCG64 state and native trial/runtime identity. Ordinary Environment checkpoints without it are not workflow checkpoints. Old/stale/mismatched state rejects before RNG restoration; a seed alone cannot replace generator/cache state. |
| Mutable durable workspace: `workspace.sqlite3`, schema/user version 1, immutable host policy, retained registry/plans/jobs/imports, ownership, attempt state and job-owned checkpoint directories | **Intentionally breaking** for generic executor/workspace evolution. Use a distinct v5 workspace root and versioned storage profile. | Continue existing jobs in pinned v4. No automatic opening, schema change, policy replacement or mutation by a generic v5 service. A future explicit migrator must preserve original evidence, ownership, acknowledged frontiers, quotas and exposure lineage. See [workspace decision](#workspace-evolution). |
| Native `ExperimentPlan`, frozen defaults/executable snapshots, `WorkflowReport`, file pins, partial summaries, physical attempts and inspection artifacts | **Readable-only** when imported across generations; retained native readers are a v5 qualification target. | Verify declared schemas, file digests and report mathematics. Partial projections stay partial. Read/import does not qualify an executable runtime or upgrade a plan into a new generic plan. Pinned v4 remains the route to execute/resume its original plan. |
| `EvidenceBundle`, `ImportedEvidence`, portable application result bundles and caller-owned artifact downloads | **Readable-only** across generations, preserving declared producer claims and local import ownership. | Never extract imported checkpoints or execute embedded content. Retain `read_only` and the explicit unverified native execution-binding fields. Importing evidence does not reset a protected holdout's upstream exposure. |
| `CoverageProblem`, `CoverageResult`, `CoverageCheckpoint`, `solve_coverage`, exact quantities/semantics, limits and publication helpers | **Retained.** Reuse the actual finite-coverage solver and its versioned v1 inputs/results/verified incumbent checkpoint. | Missing/unknown versions, changed scope or incompatible units reject. Exactness remains limited to the documented method, input scope and certificate bounds. This is not a full-ranking or continuous learner. See [finite coverage](finite-coverage.md). |
| `AllocationProblem`, finite actions/scenarios/resource/risk/utility contracts, `AllocationResult`/checkpoint, forecasts, `select_allocation` and publication helpers | **Retained.** Reuse finite scenario-payoff allocation, exact decimal accounting and conservative evidence/fallback. | A new continuous allocation capability is separate. Scenario weights are declared model inputs, not automatically calibrated probabilities. Cached evaluations must bind the complete original problem; signs, units and choice status remain unchanged. See [scenario allocation](scenario-payoffs.md). |
| `ObservationHistory`, revisions/snapshots, `SignalRequest`, `DecisionSignal`, providers, quantity-mean training/model and publication helpers | **Retained.** Preserve point-in-time eligibility, receipt/revision lineage and signal provenance while moving lightweight temporal primitives below heavy application imports. | Corrections never rewrite what was available at an earlier decision. Generic replay adds typed causality and ownership; it does not reinterpret old clocks or infer missing ones. See [observations](point-in-time-observations.md). |
| Additive `coleman.replay` contracts, explicit processing schedule and concrete host continuation | **New versioned boundary.** Preserve original core/observation/objective v1 content inside the replay records. | Unknown receipt remains pending; no legacy clock, hidden alternative or propensity is invented. Restore requires the exact concrete host type, codec, implementation and complete matching state. New lineage is explicit; no native pickle or mutable-workspace adoption. See [delayed CI replay](causal-replay.md). |
| Pure `AlgorithmState[P]`, owned stdlib random state and explicit retained replay generation2 | **New explicit contracts.** Preserve original replay v1 readers/bytes; choose retained ordinal identity at origin. | No silent truncation, reseeding, freeform-ID rewriting or retained-profile fork. A known replay codec mapping requires an empty-tail quiescent seal; direct typed host codecs may retain pending data under their explicit mapping. See [owned state](state-recovery.md). |
| Local acknowledged component collections | **Separate new local root/profile**, not a modification of native workflow workspace1. | Explicit initialization, actual runtime/code/codec compatibility and complete owner inventory are required. SQLite COMMIT acknowledges all components; a pre-COMMIT manifest is an orphan. Missing/corrupt current state refuses instead of falling back. Preserve acceptance fences through known migrations. See [state API](api/state.md). |
| Protected `TemporalPlan`, freeze/candidates/protocols, observations, audit/access receipts, family reports, bounded paired inference and multiplicity helpers | **Retained.** Preserve existing methods and assumptions; generic prequential execution adds a distinct frozen protocol. | Retain upstream holdout and cumulative access identities. Do not reset exposure on import, alias or migration. Existing bounded-mean and legacy inference remain distinct procedures; a new report cannot silently select whichever result is favorable. See [protected evaluation](protected-evaluation.md) and [bounded comparisons](bounded-paired-evaluation.md). |
| Corpus manifests/data cards/registry/fixtures, source materialization/cache/receipts, profiles and source admission/completion contracts; source-specific analytical/native Parquet ETL and controlled-capture receipts | **Retained** for their published source-specific versions; historical producer validation may require pinned v4. | Keep source bytes/terms, producer closure, original units, missingness, exclusions and declared projection limits. A readable receipt is not a newly qualified source or a finance dataset. Re-capture under v5 gets a new producer identity. See [dataset index](datasets.md), [materialization](corpus-materialization.md) and [source admission](corpus-admission.md). |
| Repeated corpus evaluation, baseline matrices, stress cases and independent finite-model benchmark suites/reports/checkpoints | **Retained** as their native bounded experiments. Generic benchmarks add separate cases/protocols. | Frozen policy/input/model scope, independent-unit definitions, raw costs, limits and inconclusive outcomes remain visible. A new implementation identity cannot reuse an old qualification claim merely because mathematical results agree. See [corpus evaluation](corpus-evaluation.md), [baseline matrices](comparable-baselines.md) and [benchmarks](decision-benchmarks.md). |
| `coleman.resources` exact ledger/events, estimated selection, replay protocols/results/reports and versioned ledger restoration | **Retained** at its existing typed resource-accounting boundary. Generic constraints may wrap this capability without redefining costs. | Reserved, estimated, observed occupancy, queue, setup and elapsed amounts remain separate. Do not promote estimated feasibility to observed physical execution. See [resource replay](resource-replay.md). |
| `coleman.pilot` manifests/events/replay/adoption reports; `coleman.shadow` frozen protocols, admission/journal/state/checkpoint/report and host-clock/source-status contracts | **Retained** as bounded CI operational evidence and shadow execution contracts. Generic external effects require a separately qualified adapter. | Reuse the documented exact accounting and conservative findings, not an implied broker/order or arbitrary production control API. Shadow state remains bound to its source/protocol and latest acknowledged prefix. See [operational pilot](operational-pilot.md). |

The [dataset index](datasets.md) is the exhaustive entry point for supported
source-specific routes, including LRTS/stage views, RTPTorrent, LibSSH/HCS,
Westermo, JUnit and controlled Java/Python/C captures. Their individual schemas
are not collapsed into a new generic source schema by this decision. Notebook,
Docker, Grafana, DuckLake/ClickHouse analytical examples and contributor tools
remain optional outer integrations, with their own pinned environment and source
requirements. They are not dependencies of a mathematical learner call.

## Architecture decisions that govern migration

### One modular distribution first

Keep one distribution and the `coleman` import root initially. The implemented
`coleman.core` namespace supplies narrow mathematical contracts and protocols;
generic learners remain planned. The
existing source adapters, artifacts, services and transports remain separate
modules in that distribution. The integration metadata now keeps Pydantic and
packaging in the base installation; `ci` and `analytics` select the numerical,
Parquet, database and statistical capabilities used by those operations. Import
checks and actual clean dependency resolution qualify separate facts. Installed
regular-file bytes are disk-footprint observations, not process-memory evidence.
See [component installation](core-installation.md) for exact commands and limits.

New durable native workflow plans and baseline matrices use
`SelectedInstalledRuntime` schema version 2 with the fixed `ci` selection.
Historical `InstalledRuntime` version 1 retains its base-only meaning and reader
bytes; reading an old plan/matrix does not authorize v5 execution or continuation.
Use its original pinned v4 environment for that work. The outer workflow,
SQLite store and existing PCG64 codec stay at their own version 1. See
[native runtime identities](core-installation.md#keep-native-runtime-identities-explicit)
for capture, dependency closure and migration limits.

The [generic task route](typed-execution.md) uses the distinct `coleman-tasks-v1`
namespace and a separately captured installed-executor identity. New tasks and
compatible native jobs share the five-table metadata layout and bounded
coordinator in a fresh explicitly selected workspace. Stored profile equality
does not approve code; the current operator must independently supply it. The
new mixed reader is bounded at 64 MiB/depth 64, with a 16 MiB task ceiling;
native-only readers keep their original wire behavior. This adds no migration
of historical executable work or native pickle state.

Separate wheels are justified only by demonstrated dependency or support
isolation that a modular namespace and optional extras cannot provide. Evidence
must include package ownership/names, namespace/resource behavior, dependency
resolution, release/version coordination, supported Python/platform matrix and
installed examples. A different distribution name or split cannot appear first
as a changed install command. Existing `mcp`, `telemetry`, `clickhouse`, notebook,
docs and development extras keep their declared roles until explicitly revised.

### Keep the supported Python minimum

The audited metadata requires **Python 3.14 or later**; type checking and the
supported execution documentation target 3.14. The Ruff syntax target `py313`
does not establish Python 3.13 runtime support. Retain the actual minimum for
v5 architecture work. Lowering it requires evidence for interpreter syntax,
dependency wheels, process behavior, installed imports and all supported runtime
flows. A hypothetical lightweight wheel is not that evidence. See
[package metadata](https://github.com/jacksonpradolima/coleman/blob/1edbee874129aa85e26e592d9649826e85e1ac6b/pyproject.toml)
and [contributing](contributing.md).

### Workspace evolution

V4 durable workspaces have a versioned SQLite contract and retained host policy;
they are not ordinary output folders. The current store rejects unknown/corrupt
schemas and does not silently migrate. Preserve that refusal behavior. Opening
or restoring a generic v5 workspace must select its explicit format and trusted
root before any job ownership, checkpoint load or external work begins.

Use fresh roots for generic execution. A conversion, if implemented, is an
explicit operation on a consistent stopped/owned snapshot. It validates source
schema and complete artifact membership, writes a separate target, records a
versioned mapping and publishes completion last. Failure leaves the source
usable. It must preserve job/request identity, tombstones, protected-access
lineage and committed frontiers without turning imported evidence into runnable
state. Byte-copying a live SQLite database or renaming its directory is not this
operation. V4 jobs may instead finish under their original environment and
export their evidence for read-only inspection.

### Separate identity generations

There are several existing identities, and none substitutes for the others:

| Identity | Existing meaning | Rule for v5 |
| --- | --- | --- |
| Native run ID | First 12 hex characters of SHA-256 over normalized resolved RunSpec. Default Scalene protection, empty extensions/hooks and specified result options have compatibility normalization. | Preserve that computation for qualified legacy runs. Do not reuse it as a complete dataset/runtime identity or silently change its generation. |
| Domain logical key and content digest | Namespace/run/execution/decision identify a logical record; a separate digest binds its complete canonical content. | Preserve logical identity versus content distinction, conflict rejection and versioned schema semantics. |
| Source/producer/runtime identity | Some producers bind all installed package resource bytes and version; workflow runtime also binds its active installed dependency inventory. | Retain original pins for historical validation. Even an unrelated package-byte change can create a different producer. Any narrower closure requires a new reviewed generation. |
| Generic decision/plan/state identity | New capability, implementation, objective/transform, feasible set, input, seed/stream coordinates and codec identity. | Introduce an explicit generation and canonical payload. Never relabel a legacy ID as proof of these additional bindings. |
| Operational job, attempt and ownership IDs | Local request deduplication, physical attempt history and exclusive state/result ownership. | Keep distinct from scientific observations and mathematical identities. A new attempt must not reset request history, committed state or protected exposure. |

See [native run identity](api/spec-run-id.md),
[producer identity source](https://github.com/jacksonpradolima/coleman/blob/1edbee874129aa85e26e592d9649826e85e1ac6b/coleman/spec/producer_identity.py),
[runtime identity source](https://github.com/jacksonpradolima/coleman/blob/1edbee874129aa85e26e592d9649826e85e1ac6b/coleman/spec/runtime_identity.py) and
[architecture and provenance](setup.md#identity-and-reproducibility).

### External effects need an executor contract

Native TCPCI replay consumes recorded outcomes; it does not launch the subject's
tests. Source capture processes, network collectors, ClickHouse writes and OTLP
exports already live at explicit outer boundaries. Generic learner calls must
remain free of these effects. An optional executor must declare idempotency,
acceptance, cancellation, partial completion, reconciliation and recovery limits
for the actual effect it performs.

Checkpointing a learner is not an exactly-once guarantee for a remote operation.
An effect may complete while its acknowledgement is lost. The adapter must
reconcile an authoritative external identity or mark the outcome unresolved;
it cannot repeat irreversible work merely because no local result exists.
Broker execution and live financial orders are not supplied by this architecture
or by a historical replay adapter. Host-owned registration, credentials and
authorization stay outside the mathematical core.

## Evidence required before replacing a v4 route

A claimed v5 compatibility route must carry installed-package evidence outside
the checkout, frozen old fixtures and an explicit result for unsupported/future
formats. Native qualification must compare inputs, ranking/selection, reward and
metric semantics independently from wall-clock metadata. Checkpoint support
additionally requires interrupted/resumed execution with the exact state codec,
owned RNG streams or qualified legacy wrapper, committed frontier and result
ownership intact. Read-only imports must demonstrate that they neither execute
content nor reset access history.

Keep the original evidence when producing a migrated derivative: identify its
source digests, converter/version, changed semantics, unknown fields and target
schema. An unavailable fact stays unavailable. A failed or incomplete conversion
must not publish a completion receipt. Do not overwrite old artifacts, invent
units, infer missing provenance, or rewrite a negative/inconclusive outcome to
make a migration appear successful.

The native catalog and explicit capture/publication route document their own
bounded qualification; they do not complete the generic TCPCI adapter or all
native checkpoint proofs. The adapter and final release qualification must
retain that distinction. Continue executable historical work in its pinned v4
environment wherever the corresponding v5 execution/recovery route remains
unqualified.
