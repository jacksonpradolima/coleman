# Replay API

Use [Replay late CI feedback](../causal-replay.md) for the complete input table,
decision/update timeline and installed example. These APIs belong to the
unreleased v5 integration source and run with the base installation.

The `coleman.replay` package initializer has no eager exports. Import the
contracts, engine and codec explicitly. Direct learners can keep using the
independent [core protocols](core-contracts.md); adopting a replay session is
optional.

## Choose the boundary you need

| Boundary | Responsibility |
| --- | --- |
| `coleman.replay.contracts` | Strict versioned policy, targets, opportunities, feedback/origin bindings, trace records and complete replay state |
| `coleman.replay.engine` | One host/session owner, eligible selection, explicit processing calls, logical retries and typed continuation |
| `coleman.replay.codec` | Bounded canonical reading with an explicitly supplied concrete host-state type |
| `coleman.replay.revisions` | Pure revision-graph, source-visibility and opportunity validation shared by readers and runtime |
| `coleman.replay.planning.PendingRound` | Inspection of admitted but unapplied logical feedback, with its blocking reason |

`read_replay_state`, `read_replay_feedback` and `read_continuation` accept built-in
UTF-8 strings/bytes, not filenames. Their default limit is 16 MiB and 64 nested
containers; callers also bound file/network acquisition before passing bytes.
The continuation reader requires an explicit concrete `CoreContract` subclass
for host state. It validates that supplied model but does not restore a host.

The host implements existing `Decider`, `Updater` and `Stateful` protocols. Its
optional replay updater receives a `ReplayApplication` containing the actual
application instant; an adapter can then pass the admitted mathematical feedback
to an independent learner. The complete host snapshot must include statistics,
pending work and any owned RNG/cache. An implementation or codec name is a
compatibility declaration, not a dynamic plugin lookup.

## Interpret calls and state correctly

| Call or property | What happens |
| --- | --- |
| `admit_history` / `admit_feedback` | Validate and retain bounded append-only facts without updating the host |
| `eligible_snapshot` / `prepare` | Select and bind the actual eligible input view without calling the host |
| `emit` | Drain eligible prior feedback, choose once and retain explicit offline identity execution |
| `advance` | Close an explicit eligibility/processing boundary and apply its selected complete rounds |
| `state` / `pending` | Inspect retained metadata; this is not verification of the live host snapshot |
| `export_continuation` | Verify and combine the live host snapshot with the exact replay frontier |
| `from_continuation` | Restore a matched complete envelope into a fresh compatible host |
| `fork_from_continuation` | Restore a matched prefix into a new child lineage, preserving parent records |

`eligible_through` and `apply_at` are independent monotonic clocks, with
eligibility no later than application. A decision's cutoff limits the feedback
that can affect it even when its actual decision time is later. The
`closed-drain-v1` schedule is explicit; changing drain boundaries is not a
promised equivalent execution.

Decision and completed-update counts are separate. Latest source-visible
revision selection happens before maturity/completeness checks. A visible
unready successor blocks the root. Under `preserve_live_v1`, a later correction
does not create another completed update for an accepted target.

Exact retained logical retries avoid a second host call. New work requires
matching current host state. Host failures invalidate the session without
claiming rollback of physical mutations. A complete caller-owned continuation
is separate from a durable acknowledged checkpoint; no native pickle migration
or exactly-once external effect is implied.

The separate [state and recovery API](state.md) provides optional local
acknowledgements and explicit retained replay generation2. Choose that profile
at origin; these v1 readers do not accept2 or silently truncate a full trace.
Forks and live-pending replay-codec migration are refused by the
retained profile. Direct complete v1 continuation/fork behavior above is unchanged.

The explicit offline identity-execution policy retains the proposal/executed
binding for simulation. It is not evidence that a real CI job ran. Source clocks,
origin declarations and fingerprints establish internal consistency of supplied
facts, not source authenticity or causal identification.

## Contracts

::: coleman.replay.contracts

## Engine and direct host protocol

::: coleman.replay.engine

## Pending feedback inspection

::: coleman.replay.planning.PendingRound

## Revision and eligibility validation

::: coleman.replay.revisions

## Bounded continuation reader

::: coleman.replay.codec
