#!/usr/bin/env bash
# =============================================================================
# DevContainer post-start hook for Coleman
# =============================================================================
# Runs every time the container starts (not just on creation).
# Brings up the observability stack so that `coleman run --config run.yaml`
# sends metrics to Grafana without any manual steps.
# =============================================================================
set -e

# Ensure uv is on PATH (installed by post-create.sh)
export PATH="$HOME/.local/bin:$PATH"

# ── 1. Start the observability stack (OTel + Prometheus + Grafana + ClickHouse) ──
#    Uses Docker-in-Docker.  If the containers are already running this is a
#    no-op thanks to `--wait`.
echo "── Starting observability stack ──"
if command -v docker >/dev/null 2>&1; then
  (cd examples/observability && docker compose --profile clickhouse up -d --wait) || \
    echo "⚠  Could not start observability stack. Check Docker status with 'docker ps' or restart the container."
else
  echo "⚠  Docker not available — skipping observability stack."
fi

# ── 2. Telemetry ─────────────────────────────────────────────────────────
#    Use packs/telemetry/local.yaml in your run.yaml to enable telemetry.
#    No config.toml patching needed.

echo ""
echo "✅  DevContainer started — observability is live."
echo "  Grafana           → http://localhost:3000"
echo "  Prometheus        → http://localhost:9090"
echo "  Prometheus metrics → http://localhost:8889/metrics"
echo "  ClickHouse HTTP   → http://localhost:8123"
echo ""
