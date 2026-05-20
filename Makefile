SHELL := /bin/bash

# Resolve uv path: if not found in PATH, default to ~/.local/bin/uv (where installer places it)
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
PYTHON := .venv/bin/python
PYTHON_REAL := $(shell readlink -f $(PYTHON))
PYSPY := .venv/bin/py-spy
PROFILE_VENV := .venv-pyspy
PROFILE_PYTHON := $(PROFILE_VENV)/bin/python
WILY_FILE ?= coleman/runner.py
PYTEST := .venv/bin/pytest
PRE_COMMIT := .venv/bin/pre-commit

# Default link mode for uv to avoid hardlink warnings on some filesystems
export UV_LINK_MODE ?= copy

.DEFAULT_GOAL := help

.PHONY: help ensure-uv setup install pre-commit-install lint format format-check typecheck test test-cov docs docs-serve docs-export-workflow check-precommit clean interrogate build run cost-structural cost-complexity cost-maintainability cost-xenon cost-wily cost-wily-file cost-profile-scalene cost-profile-pyspy cost-energy

## —— Coleman Makefile ——————————————————————————————————

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ——— Setup ———————————————————————————————————————————————

ensure-uv: ## Ensure uv is installed; install if missing
	@if ! command -v $(UV) >/dev/null 2>&1; then \
	  echo "uv not found, installing..."; \
	  curl -Ls https://astral.sh/uv/install.sh | bash; \
	  echo "uv installed at $$HOME/.local/bin/uv"; \
	fi

setup: ensure-uv ## Setup Python 3.14 for local development
	$(UV) python install 3.14
	# Ensure project Python is pinned to a compatible version for uv (avoids reading an old .python-version)
	$(UV) python pin 3.14
	$(UV) venv .venv --python 3.14 --allow-existing

pre-commit-install: ## Install pre-commit hooks
	$(UV) run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push

install: ensure-uv setup pre-commit-install ## Install all dependencies from uv.lock (including dev, docs, notebook extras) into .venv
	$(UV) sync --frozen --extra dev --extra docs --extra notebook --extra telemetry --extra clickhouse
	# Ensure project is installed in editable mode
	$(UV) run --python $(PYTHON) --no-project pip install -e .

# ——— Quality ——————————————————————————————————————————————

lint: ensure-uv ## Run ruff linter
	$(UV) run ruff check .

format: ensure-uv ## Run ruff formatter
	$(UV) run ruff format .

format-check: ensure-uv ## Check ruff formatting without changing files
	$(UV) run ruff format --check .

typecheck: ensure-uv ## Run pyright and ty type checkers
	# Run both static type checkers
	$(UV) run --python $(PYTHON) --no-project pyright coleman
	$(UV) run --python $(PYTHON) --no-project ty check coleman tests

interrogate: ensure-uv ## Check docstring coverage with interrogate
	$(UV) run interrogate coleman/ -v

# ——— Testing ——————————————————————————————————————————————

test: ensure-uv install ## Run tests with pytest in parallel if xdist is available; otherwise, run serially
	@if $(UV) run --python $(PYTHON) --no-project python -c 'import xdist' >/dev/null 2>&1; then \
	  $(UV) run --python $(PYTHON) --no-project pytest -n auto; \
	else \
	  echo "pytest-xdist not found; running tests serially..."; \
	  $(UV) run --python $(PYTHON) --no-project pytest; \
	fi

test-cov: ensure-uv install ## Run tests with coverage report in parallel if xdist is available; otherwise, run serially
	@if $(UV) run --python $(PYTHON) --no-project python -c 'import xdist' >/dev/null 2>&1; then \
	  $(UV) run --python $(PYTHON) --no-project pytest -n auto \
	    --cov=coleman --cov-branch \
	    --cov-report=term-missing:skip-covered \
	    --cov-report=xml \
	    --cov-report=html; \
	else \
	  $(UV) run --python $(PYTHON) --no-project pytest \
	    --cov=coleman --cov-branch \
	    --cov-report=term-missing:skip-covered \
	    --cov-report=xml \
	    --cov-report=html; \
	fi

# ——— Documentation ————————————————————————————————————————

docs: ensure-uv ## Build documentation with Zensical
	$(UV) run --extra docs zensical build --strict

docs-serve: ensure-uv ## Serve documentation locally
	$(UV) run --extra docs zensical serve

docs-export-workflow: ensure-uv ## Export marimo workflow notebook to Markdown for Zensical
	$(UV) run --extra notebook marimo export md docs/workflow.py -o docs/workflow.md --force

# ——— Build ————————————————————————————————————————————————

build: ensure-uv ## Build the package
	$(UV) build

# ——— Run ————————————————————————————————————————————————

run: ensure-uv ## Remove runs/checkpoints and execute coleman run
	rm -rf runs/ checkpoints/
	$(UV) run coleman run --config run.yaml

# ——— Pre-commit ———————————————————————————————————————————

check-precommit: test typecheck interrogate ## Run tests, type checks, and docstring coverage (used by pre-commit)

# ——— Code Cost ————————————————————————————————————————————

cost-structural: cost-complexity cost-maintainability cost-xenon ## Run all structural cost checks

cost-complexity: ensure-uv ## Report cyclomatic complexity (Radon CC)
	$(UV) run --extra dev radon cc -s -a coleman/

cost-maintainability: ensure-uv ## Enforce maintainability index gate (Radon MI ≥ 20)
	@$(UV) run --extra dev radon mi -s -n B coleman/ > /tmp/mi_issues.txt; \
	if [ -s /tmp/mi_issues.txt ]; then \
		echo "❌ Modules below maintainability threshold (MI < 20):"; \
		cat /tmp/mi_issues.txt; \
		exit 1; \
	else \
		echo "✅ All modules meet MI ≥ 20"; \
	fi

cost-raw: ensure-uv ## Report raw source metrics (LOC, SLOC, comments) with Radon
	$(UV) run --extra dev radon raw coleman/

cost-xenon: ensure-uv ## Run Xenon complexity gate (CI threshold)
	$(UV) run --extra dev xenon --max-absolute C --max-modules B --max-average A coleman/

cost-wily: ensure-uv ## Build and report complexity trend with Wily
	@archiver=git; \
	if [ -n "$$(git status --porcelain 2>/dev/null)" ]; then \
		archiver=filesystem; \
		echo "INFO: Dirty repository detected; using Wily filesystem archiver."; \
	fi; \
	$(UV) run wily build -a $$archiver coleman
	$(UV) run wily index
	$(UV) run wily report $(WILY_FILE)

cost-wily-file: ensure-uv ## Build Wily history and report a specific file (use WILY_FILE=path)
	@archiver=git; \
	if [ -n "$$(git status --porcelain 2>/dev/null)" ]; then \
		archiver=filesystem; \
		echo "INFO: Dirty repository detected; using Wily filesystem archiver."; \
	fi; \
	$(UV) run wily build -a $$archiver coleman
	$(UV) run wily index
	$(UV) run wily report $(WILY_FILE)

cost-profile-scalene: ensure-uv ## Smoke-test Scalene against the CLI entrypoint
	$(UV) run scalene run coleman/cli.py --- run --config run.yaml

cost-profile-pyspy: ensure-uv ## Record a py-spy profile using a dedicated Python 3.13 profiling venv
	$(UV) python install 3.13
	$(UV) venv $(PROFILE_VENV) --python 3.13 --allow-existing
	$(UV) pip install --python $(PROFILE_PYTHON) duckdb numpy polars pyarrow scipy scikit-posthocs "pydantic>=2.12.5" "pyyaml>=6.0.3"
	@set -e; \
	rm -f profile.svg /tmp/pyspy-cost-profile.log; \
	status=0; \
	set +e; \
	set -o pipefail; \
	PYTHONPATH=. $(PYSPY) record --rate 20 --subprocesses -o profile.svg -- $(PROFILE_PYTHON) -m coleman.cli run --config run.yaml \
		2>&1 | tee /tmp/pyspy-cost-profile.log; \
	status=$${PIPESTATUS[0]}; \
	set -e; \
	if [ "$$status" -ne 0 ]; then \
		if [ -f profile.svg ] && grep -q "Wrote flamegraph data to 'profile.svg'" /tmp/pyspy-cost-profile.log; then \
			echo "INFO: py-spy returned a non-zero exit during teardown, but profile.svg was written successfully."; \
			exit 0; \
		fi; \
		exit $$status; \
	fi

cost-energy: ensure-uv ## Estimate energy/carbon for a representative workload
	$(UV) run python scripts/measure_energy.py

# ——— Cleanup ——————————————————————————————————————————————

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ site/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
