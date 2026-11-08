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
