# TCPCI problem mapping API

Start with the [complete input → ranking → slate → feedback guide](../tcpci-problems.md).
These pure records and functions require no native runner, workspace, CLI or MCP.
Keep physical execution and original native configuration at their separate edges.

## Decision-time input and modeled work

::: coleman.problems.tcpci.contracts.TestCase
    options:
      show_source: false

::: coleman.problems.tcpci.contracts.BudgetPrefix
    options:
      show_source: false

::: coleman.problems.tcpci.contracts.TcpciProblem
    options:
      show_source: false

## Ranking projection and complete mapping

::: coleman.problems.tcpci.mapping.ranking_request
    options:
      show_source: false

::: coleman.problems.tcpci.mapping.map_ranking
    options:
      show_source: false

::: coleman.problems.tcpci.contracts.TcpciMapping
    options:
      show_source: false

Feedback uses the existing [core contracts](core-contracts.md). A core feedback
batch identifies the executed action envelope. Retain and check the complete
mapping fingerprint as well when associating feedback with duration quotes,
budget, full objective definitions or modeled accounting. Neither digest
authenticates a source or certifies physical execution.

For existing experiment configurations and artifacts, follow the
[compatibility routes](../v5-compatibility.md). Reading evidence, continuing an
executable checkpoint and adopting a mutable workspace are separate operations.
