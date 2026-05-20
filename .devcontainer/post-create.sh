#!/usr/bin/env bash
# =============================================================================
# DevContainer post-create setup for Coleman
# =============================================================================
# Runs once after the container is created.  Installs uv, Python 3.14,
# project dependencies (including telemetry + clickhouse extras), pre-commit
# hooks, and prepares the environment so that observability works out of the
# box — zero manual steps.
# =============================================================================
set -e

echo "── Setting up Coleman DevContainer ──"

# 1. Install uv (the project's package manager)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | bash
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Install Python 3.14, create the venv, and install ALL dependencies
#    (dev + docs + notebook + telemetry + clickhouse extras)
make setup
uv sync --frozen --extra dev --extra docs --extra notebook --extra telemetry --extra clickhouse
uv run --python .venv/bin/python --no-project pip install -e .

# 3. Install pre-commit hooks (non-critical)
make pre-commit-install || echo "⚠  pre-commit install failed — you can run 'make pre-commit-install' manually."

# 4. Install recommended Copilot plugins (non-critical, opt-in)
#
# Set INSTALL_COPILOT_PLUGINS=1 to install:
# - polyglot-test-agent@awesome-copilot
# - testing-automation@awesome-copilot
# - context-engineering@awesome-copilot
# - security-best-practices@awesome-copilot
# - database-data-management@awesome-copilot
# - technical-spike@awesome-copilot
# - project-planning@awesome-copilot
# - structured-autonomy@awesome-copilot
if command -v copilot >/dev/null 2>&1 && [[ "${INSTALL_COPILOT_PLUGINS:-0}" == "1" ]]; then
  echo "Installing Copilot plugins from awesome-copilot..."
  copilot plugin install polyglot-test-agent@awesome-copilot || echo "⚠  Failed to install plugin: polyglot-test-agent"
  copilot plugin install testing-automation@awesome-copilot || echo "⚠  Failed to install plugin: testing-automation"
  copilot plugin install context-engineering@awesome-copilot || echo "⚠  Failed to install plugin: context-engineering"
  copilot plugin install security-best-practices@awesome-copilot || echo "⚠  Failed to install plugin: security-best-practices"
  copilot plugin install database-data-management@awesome-copilot || echo "⚠  Failed to install plugin: database-data-management"
  copilot plugin install technical-spike@awesome-copilot || echo "⚠  Failed to install plugin: technical-spike"
  copilot plugin install project-planning@awesome-copilot || echo "⚠  Failed to install plugin: project-planning"
  copilot plugin install structured-autonomy@awesome-copilot || echo "⚠  Failed to install plugin: structured-autonomy"
elif command -v copilot >/dev/null 2>&1; then
  echo "Skipping Copilot plugin installation (set INSTALL_COPILOT_PLUGINS=1 to enable)."
else
  echo "⚠  Copilot CLI not found; skipping plugin installation."
fi

# 5. Done — the post-start hook will start the observability stack.
#    Use packs/telemetry/local.yaml in your run.yaml to enable telemetry.

echo ""
echo "✅  DevContainer ready!"
echo ""
echo "The observability stack (OTel + Prometheus + Grafana + ClickHouse) starts automatically."
echo ""
echo "  Grafana          → http://localhost:3000"
echo "  Prometheus       → http://localhost:9090"
echo "  Prometheus metrics → http://localhost:8889/metrics"
echo "  ClickHouse HTTP  → http://localhost:8123"
echo ""
echo "Useful commands:"
echo "  make test          – run tests"
echo "  make lint          – run ruff linter"
echo "  make format        – run ruff formatter"
echo "  make docs-serve    – serve docs locally"
echo "  coleman run --config run.yaml  – run experiments (results + telemetry)"
echo ""
