# Synthetic funded-portfolio API

The domain layer contains exact typed accounting and local mathematical delivery
ownership. It imports neither application/CLI/MCP nor optional Arrow storage.

```python
from coleman.problems.portfolio import (
    AssetWeight, Book, BookSession, BookSnapshot, Exact, Execution, Fees,
    Position, PricePanel, PriceQuote, Target, Trade, Valuation, execute, mark,
)
```

`Exact.of("1/3")` retains a canonical reduced rational. Floats, noncanonical
fractions, nonfinite inputs and values exceeding the selected arithmetic capacity
reject. Units belong to the enclosing field: prices are synthetic credits per
asset unit, holdings are asset units, and wealth/charges are synthetic credits.

`mark` preserves price drift without trading. `execute` is pure exact mathematics;
it does not deduplicate repeated logical calls or publish a durable ACK.
`BookSession.deliver` serializes calls for one local owner and retains original
request receipts. `BookSnapshot` validates the complete transition chain before
continuation. An external store provides any cross-process durable publication.

Use the [worked guide](../synthetic-portfolio.md) for complete dated inputs,
independent accounting arithmetic, causal prediction/control composition,
JSON/Parquet output and optional shared task execution.

The optional installed provider publishes compact coordinate/final results.
Their typed outcomes and exact state/manifest references identify a complete
checkpoint in the acknowledged artifact. Use
`coleman_synthetic_portfolio.readback.read_full_task_result(service, job_id)` to
verify and recover that full evidence from trusted local storage. This performs
no mathematical update or remote file download. Explicit legacy result readers
remain in the provider's `recipe` module; reading them grants no execution
compatibility or automatic operational migration. The task adapter composes
opening with actual retained reconciliation or full result resolution within
one operation; each later library call reads and verifies the evidence again.

## Complete state, stored once

The direct `Experiment.export_state()` and `restore_state()` APIs retain the
original schema1 `ExperimentState`. Its `observations` array duplicates the
complete observations already attached to the accepted periods and, when
feedback precedes the next decision, `ready_observation`.

`Experiment.export_normalized_state()` returns a separate schema2
`ExperimentStateV2`. It stores those complete authoritative records once. Its
ordinary `observations` property reconstructs their ordered view without adding
a serialized field or caching a validation result. A missing period observation
rejects; it cannot disappear through filtering. All predictor statistics,
pending forecasts and scales, controller origins and movement usage, funded
books, price prefixes and complete outputs remain in the checkpoint. The task
owner selects `PortfolioCheckpointV2` through `STATE_V2` (codec reference version2)
in the explicitly selected `synthetic-funded-portfolio@3` recipe. The original
`PortfolioCheckpoint` and `STATE` remain separate schema1 readers.

After installing the matching SDK and example provider wheels as described in
the [worked guide](../synthetic-portfolio.md), run this complete local cycle:

```python
from coleman_synthetic_portfolio.experiment import Experiment
from coleman_synthetic_portfolio.flow_contracts import ExperimentStateV2

original = Experiment()
for index in range(2):
    original.step(index)

snapshot = original.export_normalized_state()
wire = snapshot.to_json()
decoded = ExperimentStateV2.from_json(wire)
assert "observations" not in decoded.model_dump(mode="json")
assert decoded.to_legacy().to_json() == original.export_state().to_json()

resumed = Experiment()
resumed.restore_normalized_state(decoded)
assert resumed.step(2).to_json() == original.step(2).to_json()
```

`ExperimentStateV2.from_legacy(state)` and `state.to_legacy()` perform explicit,
fully checked representation conversions. They retain the observation values
and original implementation/source bindings; the two representations have
different versioned bytes and content digests. Conversion is local reading work,
not an acknowledgement, a replayed decision or permission to resume a checkpoint
from another implementation. Unknown versions reject. Old v1 readers remain
available and reject v2 rather than silently interpreting it as v1. Operational
continuation still requires the selected owner, source, configuration and codec
to match; there is no automatic migration of an older installed runtime.

::: coleman.problems.portfolio.contracts

::: coleman.problems.portfolio.accounting

::: coleman.problems.portfolio.session
