# Registry and ownership API

Start with [Keep CI recipes eligible and independent](../registry-lifecycle.md)
for complete input/output examples, lifecycle and ownership diagrams, supported
profiles and the distinction between selecting over experts and invoking an
independently evolving recipe. This is unreleased v5 source functionality.

Import the leaf module you need; package import discovers nothing. The pure
definitions/lifecycle/envelope/ownership/composition modules perform no storage,
runtime capture, source loading, CLI/MCP activation or native execution. The
state/replay seams explicitly opt into existing component/replay dependencies.

| Module | Responsibility |
| --- | --- |
| `definitions` | Exact immutable expert/action versions, historical catalogue leaves and recipe bindings |
| `limits` | Strict selected capacities and bounded v1 readers using existing canonical serialization |
| `lifecycle` | Prospective facts, fixed clock policy, complete eligibility and compact registered-request seals |
| `envelopes` | Registered member-space requests, exact original execution/feedback joins and pure local sealing |
| `ownership` | Complete declared resources, intentional sharing, actual root inventories and identity checks |
| `composition` | Eligible recipe → unchanged inner action, full original origins and fixed-ownership checkpoints |
| `state` | Lazy root checks inside existing accepted component factory/restore staging |
| `replay` | Existing replay preflight, with conservative static/specialist/context/empty support |

`RegistryCheckpoint`'s `owned-recipes-v1` profile keeps the complete recipe binding
fixed. A changed epoch/control/configuration/layout requires an explicit compatible
new collection, preserving old original producers and old feedback receivers.
Its full recipe-execution frontier does not mix standalone registered-request
seals. Neither a raw mathematical update nor these pure helpers acquires durable
exactly-once guarantees; existing optional replay/store layers own acceptance.

## Definitions and finite ingress

::: coleman.registry.definitions

::: coleman.registry.limits

## Prospective eligibility and registered member decisions

::: coleman.registry.lifecycle

::: coleman.registry.envelopes

## Declared mutable ownership and sharing

::: coleman.registry.ownership

## Independently evolving recipe composition

::: coleman.registry.composition

## Optional accepted-staging and replay seams

::: coleman.registry.state

::: coleman.registry.replay
