"""Tests for run artifact manifest generation."""

from __future__ import annotations

import json

from coleman.manifest import generate_manifest


def test_generate_manifest_includes_relative_artifacts(tmp_path):
    run_dir = tmp_path / "rid123"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)

    (run_dir / "spec.resolved.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "provenance.json").write_text("{}\n", encoding="utf-8")
    (results_dir / "part-1.parquet").write_bytes(b"PAR1")

    manifest_path = generate_manifest(run_dir, run_id="rid123")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["manifest_version"] == 1
    assert data["run_id"] == "rid123"
    assert "generated_at" in data

    rel_paths = {item["relative_path"] for item in data["artifacts"]}
    assert "spec.resolved.json" in rel_paths
    assert "provenance.json" in rel_paths
    assert "results/part-1.parquet" in rel_paths


def test_generate_manifest_includes_duckdb_results(tmp_path):
    run_dir = tmp_path / "rid456"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "part-1.duckdb").write_bytes(b"DUCK")

    manifest_path = generate_manifest(run_dir, run_id="rid456")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    result_entries = [item for item in data["artifacts"] if item["type"] == "results"]
    assert result_entries
    assert any(item["format"] == "duckdb" for item in result_entries)
    assert any(item["relative_path"] == "results/part-1.duckdb" for item in result_entries)
