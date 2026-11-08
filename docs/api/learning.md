# Expert-mixture learning API

Start with [Learn coefficients from complete expert losses](../expert-mixtures.md)
for the complete input table, worked reference arithmetic, required control
combinations, comparison assumptions and direct/replay/checkpoint boundaries.
These are unreleased v5 source APIs. No package import discovers implementations.

| Module | Responsibility |
| --- | --- |
| `contracts` | Fixed reference configuration, finite panel admission, typed failures and complete sufficient payload |
| `reference` | Constant Hedge, cumulative-current-rate anytime Hedge and excluding-origin Fixed Share |
| `control_contracts` | Fixed controlled law, required support/negative frontier, complete pending blocks, interval and movement state |
| `controlled` | One actual emitted-state exponential/target/output release operation |
| `direct` | Structural learner protocol, exact producer joins and complete typed direct continuation |
| `schedule` | Explicit immutable release/budget plan for the controlled direct application |
| `replay` | Optional actual ReplayApplication bridge, with no delivery machinery in pure kernels |
| `coleman.optimization.mixture` | Independently usable exact-geometry projection/joint L1 output with qualified binary64 emission |
| `coleman.objectives.controls` | Original typed target/bounds/movement/cadence declarations and additive exact geometry/path ledger |

The pure reference/control contracts and mathematical kernels import no replay
engine, local store, provider or CLI. Direct calls require no registry. Static
complete panels are the supported mathematical capability; sleeping members,
partial outcomes, catalogue changes and controls on the anytime/Fixed Share
references reject rather than silently changing the method.

The `Stateful` boundary returns complete concrete `AlgorithmState` values.
For untrusted mathematical-state bytes, use `coleman.state.codec.read_algorithm_state`
with the exact payload class and expected owner/method/configuration/codec binding.
Selected bytes cover the full state, not just the vector. Restore validates before
replacing the owned immutable payload. No raw mathematical call implies durable
acceptance, transaction fees or a synchronous theorem for delayed actions.

## Reference inputs, results and sufficient state

::: coleman.learning.contracts

## Three separately identified reference learners

::: coleman.learning.reference

## Controlled law, receipts and complete continuation

::: coleman.learning.control_contracts

::: coleman.learning.controlled

## Optional typed direct decision application

::: coleman.learning.direct

## Explicit controlled clocks and replay application

`ControlSchedule` requires unique logical interval identities and unique UTC
release instants. Adjacent intervals sharing a boundary cannot both release at
that instant. Invalid plans reject during construction or snapshot reading,
before the scheduled application invokes its learner; no release is silently
selected from a group of coincident points.

::: coleman.learning.schedule

::: coleman.learning.replay

## Independent constrained output

::: coleman.optimization.mixture
