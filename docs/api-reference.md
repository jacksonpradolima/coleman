# Find the API you need

Start from a workflow when learning the platform. Use this index to find the
symbol and its contract once you know which boundary you need.

The unreleased v5 integration source also provides [generic decision contracts](api/core-contracts.md).
Use the [CI input → decision → feedback walkthrough](decision-contracts.md) to
understand their scope before composing a host-owned learner or solver.

## Run and configure experiments

| Entry point | Purpose | Working guide |
| --- | --- | --- |
| [Experiment API](api/experiment-api.md) | `run`, `run_many`, `run_with_extension`, `sweep` | [First experiment](getting-started.md) |
| [Typed replay](api/replay.md), v5 source | Eligible delayed feedback, explicit clocks and caller-owned continuation | [Replay late CI outcomes](causal-replay.md) |
| [State and recovery](api/state.md), v5 source | Pure owned state, bounded replay retention and complete locally acknowledged collections | [Resume complete CI decision state](state-recovery.md) |
| [Registry and ownership](api/registry.md), v5 source | Versioned eligibility, exact original recipe producers and declared-root checks | [Manage independent CI recipes](registry-lifecycle.md) |
| [Expert learning and constrained output](api/learning.md), v5 source | Pure full-information references, controlled coefficient output and optional direct/replay adapters | [Combine CI experts](expert-mixtures.md) |
| [Typed task execution](api/tasks.md), v5 source | Pure executor contracts, approved installed capture and the shared durable application lifecycle | [Run a durable adaptive task](typed-execution.md) |
| [Specification models](api/spec-models.md) | `RunSpec` and typed configuration sections | [Configuration](configuration.md) |
| [YAML I/O](api/spec-io.md) | `load_spec`, `load_sweep_spec`, `save_resolved` | [Configuration](configuration.md) |
| [Packs](api/spec-packs.md) | Resolve composable configuration fragments | [Configuration](configuration.md) |
| [Sweep expansion](api/spec-sweep.md) | Grid/zip candidate expansion | [Experiment design](experiments.md) |
| [Run identity](api/spec-run-id.md) | Compatibility-aware configuration hashing | [Architecture](setup.md#identity-and-reproducibility) |
| [Environment provenance](api/spec-provenance.md) | Persist run-environment facts | [Architecture](setup.md) |
| [CLI](api/cli.md) | Native command-line boundary | [First experiment](getting-started.md#4-run-the-same-configuration-through-the-cli) |

## Extend native behavior

[Agents](api/agent.md), [bandits](api/bandit.md), [environments](api/environment.md),
[policies](api/policy.md), [rewards](api/reward.md),
[scenarios](api/scenarios.md) and [native evaluation](api/evaluation.md) describe
the existing runtime. Use [extensions and parallelism](extensibility.md) for lifecycle
and worker-state guidance.

The integration source adds [native semantic definitions and frozen decision evidence](api/native.md)
without changing these runtime classes. Follow the
[native CI decision walkthrough](native-policy-semantics.md) to inspect all 37
policy names, capture one actual choice before evaluation, and interpret the
retained artifact's limits.

## Retain source and result evidence

| Contract/API | Purpose | Complete guide |
| --- | --- | --- |
| [Measurement records](api/results.md) | Explicit units, states, logical identity and provenance | [Domain results](domain-results.md) |
| [Artifact publication](api/artifacts.md) | Local mutable and immutable write contracts | [Architecture](setup.md) |
| [Source materialization](api/materialization.md) | Verified external source cache and immutable receipts | [Materialization workflow](corpus-materialization.md) |

The [dataset guide](datasets.md) indexes source-specific ETL/capture contracts.
The [evaluation guide](evaluation-guide.md) connects observations, repeated units
and protected-comparison APIs. [Integration](integration-guide.md) explains the
bounded application API and optional transport.
