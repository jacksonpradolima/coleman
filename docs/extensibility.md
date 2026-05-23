# Extensibility & Parallelism

This page is a complete implementation guide for extending Coleman while keeping
its native orchestration model.

## What You Can Extend

| Component | Typical goal | Current support |
|---|---|---|
| Policy | New action-selection strategy | Fully supported (source extension + config) |
| Reward | New credit assignment signal | Fully supported (source extension + config) |
| Hooks | Domain lifecycle logic and custom artifacts | Fully supported via `hooks.plugins` |
| Extensions | Namespaced custom domain config | Fully supported via `extensions` |
| EvaluationMetric | Alternate quality/cost metric logic | Supported via source extension (runner wiring) |
| Environment | Alternate orchestration/runtime behavior | Supported via source extension (runner wiring) |

## Public Runner Extension API

Coleman now provides a public extension layer that keeps orchestration in core
while allowing custom environment and metric wiring.

```python
from coleman.api import run_with_extension
from coleman.runner import RunnerExtension

extension = RunnerExtension(
  build_environment_fn=my_build_environment,
  build_metric_fn=my_build_metric,            # optional
  post_execution_fn=my_post_execution_hook,   # optional
)

result = run_with_extension(spec, extension)
```

Concrete example entrypoint:

1. [examples/custom-environment-template/advanced/my_project/extension_entrypoint.py](examples/custom-environment-template/advanced/my_project/extension_entrypoint.py)

Core-owned responsibilities remain unchanged:

1. execution plan generation
2. process-pool orchestration
3. hook event ordering
4. run_id and folder layout

Extension-owned responsibilities:

1. environment instantiation (`build_environment_fn`)
2. optional metric substitution (`build_metric_fn`)
3. optional per-execution post-processing (`post_execution_fn`)

## Parallelism Model (Complete)

Coleman has two independent parallel layers:

1. Intra-run process parallelism
   - Controlled by `execution.parallel_pool_size`.
   - Executes one run's `independent_executions` in worker processes.

2. Inter-spec sweep parallelism
   - Controlled by `coleman sweep --workers` or API `run_many(..., max_workers=...)`.
   - Executes multiple specs concurrently.

### Operational contract

1. Coordinator process events
   - `on_run_start`, `on_dataset_start`, `on_dataset_end`, `on_run_end`.
2. Worker process events
   - `on_execution_start`, `on_execution_end`.
3. Error event
   - `on_error` can be triggered in both contexts.
4. Profiling safety
   - `force_sequential_under_scalene: true` forces sequential worker mode under Scalene.

## Namespaced Custom Config (`extensions`)

Use `extensions` to pass domain-specific settings without relaxing strict RunSpec validation:

```yaml
extensions:
  my_domain:
    forecast_selection:
      policy: ThompsonSampling
      reward: Binary
      risk_budget: 0.15
```

Guidelines:

1. Keep keys namespaced (`my_domain`, `my_team`, etc.).
2. Keep values JSON-serializable.
3. Consume values from hook context (`context.extensions`).

## Hook Plugins (`hooks`)

Register plugins only in configuration:

```yaml
hooks:
  fail_fast: false
  plugins:
    - my_project.hooks.ForecastHook
    - my_project.hooks.audit_hook
```

Supported plugin forms:

1. Class plugin
   - Instantiated with no arguments.
   - Implement any lifecycle methods needed.
2. Function plugin
   - Signature: `(event_name, context, payload=None)`.

### Hook lifecycle

1. `on_run_start(context)`
2. `on_dataset_start(context)`
3. `on_execution_start(context)`
4. `on_execution_end(context, execution_result)`
5. `on_dataset_end(context, dataset_result)`
6. `on_run_end(context, run_result)`
7. `on_error(context, error)`

`on_error` is dispatched for startup failures as well, so hook-based cleanup
and telemetry can observe broken execution setup paths.

### Hook context fields

- `run_id`
- `dataset_id`
- `execution_id`
- `worker_id`
- `parallel_mode`
- `iteration`
- `trials`
- `sched_time_ratio`
- `extensions`

## Deterministic Hook Artifacts

Use `ArtifactWriter` to create deterministic, collision-safe paths and atomic
JSON/CSV files for hook artifacts.

```python
from coleman.artifacts import ArtifactWriter

writer = ArtifactWriter("./runs")
path = writer.path_for(
  run_id=context.run_id or "unknown",
  dataset_id=context.dataset_id,
  sched_time_ratio=context.sched_time_ratio,
  execution_id=context.execution_id,
  artifact_type="forecast",
  ext="json",
)
writer.write_json_atomic(path, payload)
```

Concrete plugin example (class + function hooks):

1. [examples/custom-environment-template/advanced/my_project/hooks.py](examples/custom-environment-template/advanced/my_project/hooks.py)

Default artifact layout:

`<runs>/<run_id>/artifacts/<artifact_type>/time_ratio_XX/<dataset_id>/<execution_id>/<stem>.<ext>`

## Public Scalene Compatibility Helper

For custom entrypoints, use `effective_parallel_pool_size(...)` from
`coleman.runner` to mirror the same Scalene safeguard behavior as the core runner.

## Adding a New Policy

Current loading contract:

1. Runner reads policy names from `experiment.policies`.
2. For each policy `X`, runner loads class `XPolicy` from `coleman.policy` module exports.

### Implementation steps

1. Create your policy class in `coleman/policy/...` extending base behavior from existing policy patterns.
2. Expose the class in `coleman/policy/__init__.py`.
3. Add algorithm hyperparameters under `algorithm.<policy_name_lower>` in YAML.
4. Add policy name (without suffix) in `experiment.policies`.
5. Add tests for deterministic behavior and credit-assignment updates.

Example config:

```yaml
experiment:
  policies: [MyPolicy]
  rewards: [RNFail]

algorithm:
  mypolicy:
    rnfail:
      temperature: 0.2
```

## Adding a New Reward

Current loading contract:

1. Runner reads reward names from `experiment.rewards`.
2. For each reward `Y`, runner loads class `YReward` from `coleman.reward` module exports.

### Implementation steps

1. Create your reward class in `coleman/reward/...` inheriting reward base contract.
2. Expose the class in `coleman/reward/__init__.py`.
3. Add reward name (without suffix) in `experiment.rewards`.
4. Add tests validating `evaluate(...)` against expected metric scenarios.

Example config:

```yaml
experiment:
  policies: [UCB]
  rewards: [MyReward]

algorithm:
  ucb:
    myreward:
      c: 0.4
```

## Custom EvaluationMetric (Source Extension)

Current default wiring uses `NAPFDVerdictMetric` in runner environment construction.

If you need a custom metric class today:

1. Implement metric class in `coleman/evaluation/...` extending evaluation base behavior.
2. Expose it in `coleman/evaluation/__init__.py`.
3. Wire it in runner environment construction where evaluation metric instance is created.
4. Add tests for metric math invariants and edge cases (no failures, full failures, partial budget).

## Custom Environment (Source Extension)

Current default wiring uses `Environment` in runner environment construction.

If you need a custom environment today:

1. Implement class inheriting `AbstractEnvironment`.
2. Preserve compatibility with monitor/checkpoint/telemetry contracts.
3. Wire your class in runner environment creation.
4. Add integration tests for sequential and parallel execution consistency.

## Recommended Strategy for Large Domain Workflows

1. Prefer `hooks + extensions` first.
2. Keep core runner orchestration untouched whenever possible.
3. Use source-level Environment/EvaluationMetric customization only when behavior cannot be expressed via hooks.
4. Persist custom artifacts with run/execution identifiers for later joins in analysis.

## Official Starter Template

An in-repo starter template is available under:

1. `examples/custom-environment-template/minimal/`
2. `examples/custom-environment-template/advanced/`

The advanced track includes hooks, extension-style wiring, and manifest-ready
artifact layout for parallel runs.

Quickstart for advanced track:

```bash
cd examples/custom-environment-template/advanced
python -m my_project.extension_entrypoint
```

## End-to-End Example (Parallel + Extensions + Hooks)

```yaml
packs:
  - execution/default
  - experiment/alibaba_druid
  - algorithm/defaults
  - reward/rnfail
  - results/parquet

execution:
  independent_executions: 20
  parallel_pool_size: 4
  force_sequential_under_scalene: true

experiment:
  policies: [UCB, MyPolicy]
  rewards: [RNFail, MyReward]

algorithm:
  ucb:
    rnfail:
      c: 0.3
  mypolicy:
    rnfail:
      temperature: 0.15

hooks:
  fail_fast: false
  plugins:
    - my_project.hooks.ForecastHook
    - my_project.hooks.audit_hook

extensions:
  my_domain:
    forecast_selection:
      policy: ThompsonSampling
      reward: Binary
      risk_budget: 0.15
```

CLI sweep with inter-spec parallelism:

```bash
coleman sweep --config run.yaml \
  --grid execution.seed=range(0,10) \
  --grid algorithm.ucb.rnfail.c=0.1,0.3,0.5 \
  --workers 4
```

## Testing Checklist for Extensions

1. Sequential and parallel should produce expected hook side effects.
2. Hook errors should obey `fail_fast` policy.
3. Custom artifacts should carry `run_id` and `execution_id`.
4. New policies/rewards should be covered by deterministic unit tests.
5. Any Environment/Metric source customization should include integration tests.
