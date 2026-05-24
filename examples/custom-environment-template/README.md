# Custom Environment Starter Template

This starter template provides two tracks for custom environment projects that extend Coleman without reimplementing runner orchestration.

## Tracks

- `minimal/`: single dataset, single policy, local sequential execution
- `advanced/`: parallel executions, hooks, custom artifact writing and runner extension callbacks

## Usage

1. Copy one track into your project.
2. Install Coleman as a dependency.
3. Run:

```bash
coleman run --config run.yaml
```

Advanced track entrypoint (public extension API):

```bash
python -m my_project.extension_entrypoint
```

Example plugin files in advanced track:

- `my_project/hooks.py`: class and function hook plugins using ArtifactWriter
- `my_project/extension_entrypoint.py`: minimal `run_with_extension(...)` usage

Both tracks preserve Coleman-owned responsibilities:

- execution plan generation
- process pool orchestration
- run_id and run folder layout
- hook event ordering

Your project owns only extension points (environment build, optional metric, optional post-execution hooks).
