# API Reference

## Experiment system

| Module | Description |
|--------|-------------|
| [`coleman.api`](api/experiment-api.md) | Library-first API: `run`, `run_many`, `sweep`, `load_spec`, `save_resolved` |
| [`coleman.spec.models`](api/spec-models.md) | Typed `RunSpec` and sub-spec Pydantic v2 models |
| [`coleman.spec.sweep`](api/spec-sweep.md) | Sweep engine: `SweepSpec`, `SweepAxis`, `expand_sweep` |
| [`coleman.spec.packs`](api/spec-packs.md) | Config pack resolution: `resolve_packs` |
| [`coleman.spec.run_id`](api/spec-run-id.md) | Deterministic `run_id`: `compute_run_id` |
| [`coleman.spec.io`](api/spec-io.md) | YAML I/O: `load_spec`, `save_resolved` |
| [`coleman.spec.provenance`](api/spec-provenance.md) | Provenance capture: `save_provenance` |
| [`coleman.cli`](api/cli.md) | CLI entry-point: `coleman run`, `coleman sweep`, `coleman validate` |

## Core framework

The core API is documented by module:

- [Agent](api/agent.md)
- [Bandit](api/bandit.md)
- [Environment](api/environment.md)
- [Evaluation](api/evaluation.md)
- [Policy](api/policy.md)
- [Reward](api/reward.md)
- [Scenarios](api/scenarios.md)
