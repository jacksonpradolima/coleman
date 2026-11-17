# Subset actions and acquisition API

Start with [Choose a subset, or buy one observation](../subset-acquisition.md)
for complete CI inputs, the purchased-observation cycle, metered resource
arithmetic and checkpoint recovery. These are unreleased v5 source APIs.

| Entry point | Responsibility |
| --- | --- |
| `coleman.selection.subsets` | Reuse the finite coverage solver and retain its complete certificate alongside a genuine subset or abstention |
| `coleman.acquisition.contracts` | Declare the complete terminal-allocation plan, resource units, fidelity evidence, purchases, producer receipts and verified ledger state |
| `coleman.acquisition.engine` | Inspect purchased knowledge or accept one original receipt with exact-content delivery deduplication |
| `coleman.acquisition.session` | Own complete in-memory terminal-allocation state without executing a provider or publishing a checkpoint |

The subset adapter keeps union coverage, additive modeled costs, restricted
scope and solver-specific certificates unchanged. Its action is not a ranking
and supplies no invented subset reward or inclusion probability.

The terminal allocator supports equal allocation or a declared deterministic
halving schedule. It sees purchased receipts only. Quotes control which work
starts; actual charges, unknown amounts and overruns remain separate evidence.
An empirical training-only fidelity gate does not guarantee held-out validity.
The recommendation concerns purchased terminal survivors, not a certified
full-pool optimum or a cumulative-reward result.

All mathematical components work with the base package alone. The
[acknowledged-store example](https://github.com/jacksonpradolima/coleman/blob/release/v5/examples/acquisition/checkpoint.py)
adds the existing concrete codec and state store explicitly. It does not make
remote execution exactly once. Identical complete plans identify continuation
of one logical run; independent repetitions require distinct plan references.

## Existing coverage as an actual subset

::: coleman.selection.subsets

## Plan, resource and purchased-observation inputs

::: coleman.acquisition.model

## Verified complete ledger state

::: coleman.acquisition.contracts.AllocationState

## Inspection and exact-content delivery

::: coleman.acquisition.engine

## Direct state ownership

::: coleman.acquisition.session
