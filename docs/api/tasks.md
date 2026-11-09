# Typed task execution API

Start with the [complete input → execution → output guide](../typed-execution.md).
Use the pure declarations independently; installed capture, local storage,
workers and optional transports belong to the application boundary.

## Pure declarations

::: coleman.tasks.contracts
    options:
      show_source: false

::: coleman.tasks.identity
    options:
      show_source: false

::: coleman.tasks.requests
    options:
      show_source: false

::: coleman.tasks.results
    options:
      show_source: false

## Structural executor capabilities

Concrete codecs need not inherit a framework model. Complete mathematical state
stays distinct from execution compatibility and acknowledged operational work.

| Seed location | Contract |
| --- | --- |
| `TaskCoordinate.seed` | Frozen optional uint64 declaration; `None` differs from valid `0` |
| `CoordinateWork.seed` | Compiled value retained separately from the concrete body |
| `CoordinateContext.seed` | Same value on coordinate read, reconciliation, verification and publication |
| `AttemptContext.coordinate.seed` | Same declaration during mathematical advancement |

The two in-memory carriers add a trailing optional field; original five-argument
construction remains valid with `None`. Their equality, repr and field inspection
include the new value. Persisted v1 wire formats and identity laws are unchanged.
Providers own one-time RNG initialization and complete state restoration; the
platform does not reseed per attempt or infer independence from a seed. See the
[coordinate seed cycle](../typed-execution.md#carry-a-coordinate-seed-without-reseeding).

::: coleman.tasks.protocols
    options:
      show_source: false

## Explicit application execution

::: coleman.application.tasks.TaskService
    options:
      show_source: false

::: coleman.application.executor_capture.capture_executor_identity
    options:
      show_source: false

::: coleman.application.executor_capture.load_approved_executor
    options:
      show_source: false

::: coleman.application.executor_capture.require_executor_identity
    options:
      show_source: false

::: coleman.application.task_commands.TaskCommands
    options:
      show_source: false

::: coleman.transports.task_mcp.create_task_server
    options:
      show_source: false
