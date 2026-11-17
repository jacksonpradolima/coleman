# Discrete chosen-feedback learning API

Start with [Learn from the CI recipe you actually run](../discrete-feedback.md)
for the complete input, decision, observed feedback and continuation example.
These are unreleased v5 source APIs, usable with the base package alone.

| Module | Responsibility |
| --- | --- |
| `contracts` | Fixed catalogue, explicit bounded reward semantics, method parameters, numerical compatibility and complete pending/RNG state |
| `sampling` | Deterministic finite-grid Exp3 proposal and integer-ticket selection with exact effective masses |
| `learners` | Owned sampled Exp3 or stationary UCB1, chosen-only updates and atomic direct continuation |

The public entry points are `DiscreteConfig`, `DiscreteBandit` and
`DiscreteState` in `coleman.learning.discrete`. Each instance owns its statistics,
pending original decision and, for Exp3, complete random stream. Calls require no
runner, registry, workspace, Parquet, CLI or application service.

The method profiles `exp3-dyadic53-v1` and `ucb1-binary64-v1` are distinct from
the existing TCPCI ranking policies. Both require fixed alternatives and
immediate, observed, chosen-only feedback with explicit dimensionless reward
bounds. Unsupported timing, contexts, changed catalogues and feedback revisions
reject. The guide explains the reference assumptions and the finite numerical
envelopes; a logged selection mass is not an outcome probability.

Reading state validates its version and typed statistics. Restoring additionally
checks the configured method, host compatibility and original pending proposal.
Neither operation acknowledges external work. Use the existing
[state infrastructure](../state-recovery.md) with an explicitly selected codec
and delivery adapter when durable publication is needed.

## Configuration, pending support and owned state

::: coleman.learning.discrete.contracts

## Exact finite-grid selection mechanics

::: coleman.learning.discrete.sampling

## Direct learner lifecycle

::: coleman.learning.discrete.learners
