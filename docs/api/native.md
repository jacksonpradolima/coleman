# Native ranking semantics and decision evidence

Use the [CI decision walkthrough](../native-policy-semantics.md) for complete
input, capture, evaluation and output examples. These APIs belong to the
unreleased v5 integration source, separately from the existing
[native policy classes](policy.md) and [generic core contracts](core-contracts.md).

`coleman.native` has a doc-only initializer. The catalog and frozen DTOs require
only the base installation. `coleman.native.capture` is the explicit CI edge and
requires the `ci` extra before loading native implementations. Publication uses
the existing `ArtifactWriter`; it does not run an algorithm or create a checkpoint.

## Inspect exact source-pinned definitions

The catalog covers all 37 native aliases and six reward definitions. A lookup
returns data, including unsupported factory routes; it does not instantiate or
repair a recipe. Native `theory_status="not_claimed"` is independent of route
support. A stronger generic use such as sampled choice is refused explicitly.

::: coleman.native.semantics

## Declare and retain the pre-evaluation decision

The host supplies stable IDs, expected observation round and fingerprints of
separately retained inputs, configuration and producer evidence. A pure DTO
constructor validates declarations; it is not an execution attestation. The
actual chooser wrapper below captures the call it performs.

::: coleman.native.decisions

## Choose once and freeze immediately

Call this in place of an ordinary `agent.choose()`, with exclusive host ownership
of the native agent and any shared policy/RNG. Evaluate `list(record.ranking)`
and observe through the original native interfaces afterwards. `Environment`
does not add this record automatically. Choice can mutate state even if later
validation fails; capture has no rollback or retry deduplication.

::: coleman.native.capture.choose_and_freeze_native

## Read and publish immutable evidence

Retain a returned record to retry publication. Identical publication is
idempotent; different bytes at the same logical key conflict. Reader byte limits,
version rejection and artifact completion do not imply a joint transaction with
native evaluation, observations or checkpoints.

::: coleman.native.publication
