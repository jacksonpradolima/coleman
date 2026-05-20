# Contributing to Coleman

Thank you for your interest in contributing to **Coleman**! This document
provides guidelines and instructions for contributing.

## Getting Started

### Option A: DevContainer (recommended)

The easiest way to get a fully working development environment is with a
[DevContainer](https://containers.dev/):

1. Open the repo in **VS Code** (or any DevContainer-compatible editor).
2. When prompted, select **"Reopen in Container"** (or run the
   `Dev Containers: Reopen in Container` command).
3. Wait for the container to build — `make install` and pre-commit hooks
   are set up automatically.

Everything you need (Python, uv, Docker, extensions) is pre-configured.

### Option B: Local setup

#### Prerequisites

- [Python 3.14+](https://www.python.org/downloads/)
- [UV](https://docs.astral.sh/uv/) — fast Python package manager
- [GNU Make](https://www.gnu.org/software/make/)

#### Setup

```bash
# Clone the repository
git clone https://github.com/jacksonpradolima/coleman.git
cd coleman

# Install all dependencies (including dev extras)
make install

# Install pre-commit hooks
make pre-commit-install
```

## Development Workflow

### Running Tests

```bash
make test
```

### Linting and Formatting

```bash
# Run ruff linter and formatter
make lint
make format
```

### Type Checking

```bash
make typecheck
```

### Building Documentation

```bash
make docs
```

### Code Cost Evaluation

Every pull request is checked by CI gates for structural complexity. Before
submitting, verify your changes locally:

```bash
# Run all structural cost checks (cyclomatic complexity + maintainability + xenon gate)
make cost-structural
```

If the CI reports a failure, check specific dimensions:

```bash
make cost-complexity        # Radon cyclomatic complexity
make cost-maintainability   # Radon MI gate (fails if any module < 20)
make cost-xenon             # Xenon complexity gate (same thresholds as CI)
make cost-wily              # Wily trend analysis across Git history
```

The CI thresholds are:

| Gate | Threshold | Meaning |
|------|-----------|---------|
| Xenon `--max-absolute` | **C** | No single block worse than C |
| Xenon `--max-modules` | **B** | No module average worse than B |
| Xenon `--max-average` | **A** | Project-wide average must be A |
| Radon MI | **A** (MI ≥ 20) | All modules must score A or above |

See [Code Cost Evaluation](docs/code-cost.md) for the full guide including
runtime profiling and energy estimation.

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/)
- **Linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Docstrings**: [Numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style — all public modules, classes, and functions must have docstrings.
- **Type annotations**: Encouraged for all public APIs.
- **Line length**: 120 characters.

## Experiment Configuration

The project uses a **typed run specification system** based on Pydantic v2.
When working with experiment configuration:

- Models live in `coleman/spec/models.py` — every new config field should
  be added as a typed Pydantic field, not a raw dict key.
- Config packs live under `packs/<category>/<name>.yaml`.  When adding a new
  pack, follow the existing directory structure (policy, reward, runtime,
  results, telemetry).
- The sweep engine (`coleman/spec/sweep.py`) supports grid and zip modes.
  Zip-mode parameter lists **must** have equal length.
- `_set_nested()` validates intermediate path components — never silently
  overwrite a non-dict value.
- The deterministic `run_id` is `sha256(canonical_json(spec))[:12]`.
  Any change to models or serialisation **must** preserve backward
  compatibility of existing `run_id` values — the golden-determinism test
  guards this contract.
- Tests for the spec system are in `tests/spec/`.  Use `pytest.approx()` for
  floating-point comparisons and the `tmp_path` fixture (not `/tmp`) for
  temporary directories.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/). Commit
messages are enforced by `pre-commit`:

```
feat: add support for new policy
fix: correct NAPFD computation edge case
docs: update installation instructions
test: add tests for sliding window agent
refactor: simplify reward evaluation logic
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes following the code style guidelines above.
3. Add or update tests to cover your changes.
4. Ensure all checks pass: `make check-precommit`.
5. Open a pull request against `main`.
6. A maintainer will review your PR. Please be responsive to feedback.

## Reporting Bugs

Please open an issue on GitHub with:

- A clear title and description.
- Steps to reproduce the bug.
- Expected versus actual behavior.
- Python version and OS information.

## Suggesting Features

Feature requests are welcome! Open an issue describing:

- The problem you want to solve.
- Your proposed solution (if any).
- Any alternatives you considered.

## Security

If you discover a security vulnerability, please follow the process described in
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
