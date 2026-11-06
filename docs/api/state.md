# State and recovery API

Start with [Resume the whole decision](../state-recovery.md) for the CI input
table, owner inventory, checkpoint cycle and output interpretation. These APIs
belong to the unreleased v5 integration source and use the base installation.
Package initializers have no eager exports; import the leaf boundary you need.

## Choose a state boundary

| Module | Contract |
| --- | --- |
| `coleman.state.contracts` | Concrete `AlgorithmState[P]` and exact owner/implementation/configuration/codec binding |
| `coleman.state.codec` | Bounded pure reading with an explicit concrete payload type and expected binding |
| `coleman.state.random_state` | Declarative complete state of a caller-owned stdlib Random |
| `coleman.state.components` | Finite trusted concrete codecs, component values and fresh-owner staging |
| `coleman.state.storage_contracts` | Versioned collection definitions, manifests, per-child frontiers, requests, receipts and limits |
| `coleman.state.local` | Optional local SQLite acknowledgement, recovery, known migration and cleanup |
| `coleman.replay.retained_contracts` | Explicit replay generation 2, quiescent seals and bounded codec rebinds |
| `coleman.replay.retained` | Shared causal engine with opt-in ordinal retention |
| `coleman.replay.retained_codec` | Dedicated generation 2 readers with concrete host-state types |

Direct state reading does not restore a host, draw RNG, capture runtime metadata,
open SQLite or decode pickle. The existing `Stateful` protocol owns mathematical
export/restore. Snapshot completeness and codec purity are qualified caller
contracts; arbitrary Python validators/factories are not sandboxed.

The new state wire roots require explicit schema 1. Retained replay roots explicitly
require schema 2; the original [replay readers](replay.md) remain generation 1 and
reject 2. Unknown/missing/future versions reject. There is no migration by field
name and no artifact-selected dynamic class lookup.

`RetainedReplay` refuses forks, including before a seal. Use the existing complete
v1 replay route when its matched-fork contract is required; no conversion is
inferred. Its restore signature follows the base protocol, but the runtime
requires an actual concrete generation 2 continuation and known host type.

## Interpret the local store's calls

| Call | Effect |
| --- | --- |
| `create` | Explicitly create a separate root with immutable local profile/header |
| `open` / `read_head` | Read metadata; no proof of current executable compatibility |
| `initialize` | Validate complete supplied states, stage/verify them and acknowledge generation 0 |
| `load` | Validate/decode the actual acknowledged collection under ownership; return prepared data |
| `commit_transition` | Validate actual compatibility/intent/predecessor, stage fresh owners, call once and acknowledge one complete successor |
| `migrate` | Apply one explicit known data mapping and publish a complete compatible target collection |
| `cleanup` | Retire bounded receipts and remove provably unreachable bytes without advancing child floors |

Every restore/commit operation requires explicitly captured actual compatibility.
Current compatibility is checked before returning even a retained operational
receipt. An exact accepted retry precedes stale-parent rejection but returns
`accepted_head` separately from `current_head`; it never rewinds progress.
For a new known migration, supply the actual target environment and all target
owner/mapping code bindings. Historical source codecs read declarative values;
the old implementation is not restored into an incompatible runtime. Subsequent
operational retries compare against the current acknowledged compatibility.
The profile allows 128 components and up to 256 declared implementation/mapping
identities; a migrator pin does not consume another component slot.

`BEGIN IMMEDIATE` owns the collection before callbacks. Decode all components
before any factory/restore, verify fresh distinct owners, and discard a failing
attempt. Immutable publication precedes the sole accepted point, SQLite COMMIT.
An ambiguous commit/response error requires reconciliation, not automatic
callback replay. Callback CPU is not forcibly preempted by acquisition limits.

## Direct typed state and reader

::: coleman.state.contracts

::: coleman.state.codec

## Owned stdlib random state

::: coleman.state.random_state

## Finite component codecs and staging

::: coleman.state.components

## Complete collection contracts

::: coleman.state.storage_contracts

## Optional local acknowledgement

::: coleman.state.local

## Explicit retained replay generation

::: coleman.replay.retained_contracts

::: coleman.replay.retained

::: coleman.replay.retained_codec
