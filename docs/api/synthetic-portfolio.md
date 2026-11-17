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

::: coleman.problems.portfolio.contracts

::: coleman.problems.portfolio.accounting

::: coleman.problems.portfolio.session
