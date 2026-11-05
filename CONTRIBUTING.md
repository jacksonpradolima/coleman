# Contribute to COLEMAN

Help make adaptive CI experiments understandable, reproducible and useful.
Contributions can improve a policy, a source adapter, an example, an explanation
or a contract. Begin with the user question and preserve the evidence behind the
answer.

For a tour of the product, start with the
[documentation map](https://jacksonpradolima.github.io/coleman/).
For current architectural findings and improvement priorities, read
[maintainability](https://jacksonpradolima.github.io/coleman/maintainability/).
The [quality and security guide](https://jacksonpradolima.github.io/coleman/quality-and-security/)
maps automated gates, local qualification and release evidence to their actual scope.

## Set up a development checkout

Use Python 3.14+, uv and Git. From the cloned repository:

```bash
git clone https://github.com/jacksonpradolima/coleman.git
cd coleman
uv sync --frozen --extra dev --extra docs --extra telemetry --extra mcp --extra release --group build
uv run --frozen --no-sync pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

This installs the locked development and documentation tools plus the extras used
by integration tests. Notebook, ClickHouse and controlled-capture work may need
additional extras or the exact runtime in that guide. `make install` provisions
the broader development environment; the DevContainer is an optional route with
its own monitoring services. Read its scripts before choosing that setup.

For v5 work, create a focused issue branch from updated `release/v5` and open its
PR against `release/v5`. Stable v4 maintenance continues from `main`. See the
[v5 integration guide](docs/v5-release-integration.md) for the reviewed release
route and how to record integrated but unreleased work.

Keep source archives, raw capture
outputs and large datasets outside the Git worktree. Normal tests use small owned
fixtures and run without network access.

## Choose the right boundary

Reuse the runner, RunSpec, source cache, sinks, artifact writer and checkpoints.
A new source adapter should retain original meaning and publish an explicit native
projection, rather than introducing another experiment engine. Keep CLI, HTTP,
process execution and optional MCP dependencies at their adapter boundaries.

Extend typed models for contracted state. Do not pass undocumented dictionaries
through the application when a domain contract is needed. Preserve old data
meanings, schema readers, deterministic identities and golden artifacts. Unknown
versions should fail according to their contract; a migration must not invent
missing units or provenance.

See [architecture](https://jacksonpradolima.github.io/coleman/setup/) and
[extensions](https://jacksonpradolima.github.io/coleman/extensibility/) for concrete paths.

## Write useful NumPy-style docstrings

Public modules, classes and functions need docstrings. Document each public
parameter using NumPy's `Parameters` section, and describe return values and
meaningful errors. Document constructor parameters on the class or constructor.
Include units, invariants, missingness, side effects, concurrency and compatibility
where they affect use. A one-line description is not evidence that these are covered.

```python
import math


def observed_seconds(value: float) -> float:
    """Validate an observed elapsed duration.

    Parameters
    ----------
    value : float
        Finite, nonnegative elapsed duration in seconds. Zero is observed;
        unavailable duration must use the caller's explicit missing state.

    Returns
    -------
    float
        Validated elapsed seconds, without unit conversion or rounding.

    Raises
    ------
    ValueError
        If the duration is negative or non-finite.

    Notes
    -----
    Performs no I/O and does not infer physical runner occupancy.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError("Duration must be finite and nonnegative")
    return value
```

This is a documentation pattern, not an additional SDK function.
Follow the [NumPy style guide](https://numpydoc.readthedocs.io/en/stable/format.html).
Run `make docstrings` for the currently enforced public entry points. To inspect
the broader backlog without treating it as a passing global gate:

```bash
uv run --frozen --no-sync python scripts/audit_docstrings.py > /tmp/coleman-docstring-audit.json
```

The audit checks signature coverage without importing inspected code. It does not
certify parameter-description accuracy, complete return/exception documentation or
which exported names constitute a stable API. `interrogate` separately checks
whether docstrings exist.

## Write documentation as a connected path

Lead with the reader's goal and a complete working example. Explain prerequisites,
inputs, expected output and what to do next. Use the central map and topic hubs
instead of adding another top-level navigation item for each feature.

- Put tutorials in the experiment, data, evaluation or integration path.
- Keep exact schemas and symbol details in reference pages.
- Link the relevant original paper where a method or source is explained.
- Explain a paper's role without claiming it validates later extensions.
- Keep release chronology in the compatibility archive.
- Update diagrams when relationships change, and give graph links a plain-text alternative.
- Add NumPy docstrings when a tutorial exposes a previously undocumented parameter.

Build the site strictly, inspect the generated pages, run the example outside the
checkout against a built package, and regenerate `llms.txt`. Do not edit that
generated file by hand.

## Validate a change

```bash
make lint
make format-check
make typecheck
make interrogate
make docstrings
make cost-structural
make docs
uv run --frozen --no-sync python scripts/generate_llms_txt.py
uv run --frozen --no-sync python -m pytest
uv build --no-build-isolation
git diff --check
```

Use focused tests while developing; the full relevant suite and installed-package
checks qualify the final change. Prefer `python -m pytest` from the project
environment. If intentionally skipping the whole-project coverage gate for a
focused run, add `--no-cov`. The full suite retains multiprocessing/thread coverage,
branch measurement and its 95% minimum; instrumentation timings are not performance
benchmark results.

The structural limits are Xenon C for an individual block, B for a module and A
for the project average, plus maintainability index at least 20. Inspect the
reason for a hotspot before extracting helpers solely to change a score.
[Code-cost guidance](https://jacksonpradolima.github.io/coleman/code-cost/)
explains structural and runtime measurements.

## Preserve scientific meaning

Keep negative, heterogeneous and inconclusive results. Seeds do not substitute for
independent observations. Preserve build/job/stage/variant/attempt grouping.
Distinguish raw outcomes, ranking scores, transformed rewards, calibrated
probabilities and inference. Compare sophisticated methods with simple baselines
under comparable source, budget, feedback and resource conditions.

Observed historical CI, controlled faults and synthetic mechanisms support
different claims. A successful conversion or a closed issue does not grant source
admission. When evidence is insufficient, report it and retain a conservative
fallback. The [evidence guide](https://jacksonpradolima.github.io/coleman/evaluation-guide/)
connects those rules to supported workflows.

## Submit a focused pull request

Use Conventional Commits, for example `docs: explain budgeted CI replay` or
`fix: reject conflicting source identities`. Explain the concrete problem,
resulting behavior, compatibility and validation in the PR description. Include
reproduction commands for source or behavior claims. Keep unrelated refactors out.

Before a maintainer push, review the whole diff from the merge-base. Use Alibaba
OpenCodeReview only in local delegation mode, read every applicable returned rule,
and manually review excluded tests and Markdown. Record file coverage and results;
fix confirmed critical, high and medium findings. Never configure an external
code-review endpoint for this workflow. Recheck the complete diff after material
changes, consolidate review fixes, and resolve valid threads with evidence.

A PR is ready to merge when required checks pass and blocking review findings are
resolved. Ordinary tests should not download external corpora or require a private
service. Installed-package tests should exercise the principal component outside
the checkout. Package and source identity checks matter when a producer fingerprints
its implementation; stage new package files before constructing the final wheel.

## Report a problem

For a bug, include the smallest reproducible input, command, expected/actual result,
OS, Python and package version, and relevant artifact identities. Preserve unknown
values and redact credentials. For a design proposal, explain the CI problem and
current limitation before suggesting a new abstraction.

Follow the [security policy](https://github.com/jacksonpradolima/coleman/blob/main/SECURITY.md)
for vulnerability reports. Contributions use the repository's
[MIT license](https://github.com/jacksonpradolima/coleman/blob/main/LICENSE).
