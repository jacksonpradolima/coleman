"""Tests for artifact writer utilities."""

from __future__ import annotations

import json
from pathlib import Path

from coleman.artifacts import ArtifactWriter


def test_artifact_writer_path_is_deterministic(tmp_path):
    writer = ArtifactWriter(tmp_path)

    path_a = writer.path_for(
        run_id="run123",
        dataset_id="my-dataset",
        sched_time_ratio=0.5,
        execution_id="exec-1",
        artifact_type="quality",
        ext="json",
    )
    path_b = writer.path_for(
        run_id="run123",
        dataset_id="my-dataset",
        sched_time_ratio=0.5,
        execution_id="exec-1",
        artifact_type="quality",
        ext="json",
    )

    assert path_a == path_b
    assert path_a.as_posix().endswith("run123/artifacts/quality/time_ratio_50/my-dataset/exec-1/artifact.json")


def test_artifact_writer_json_and_csv_are_atomic(tmp_path):
    writer = ArtifactWriter(tmp_path)

    json_path = Path(tmp_path) / "r1" / "artifacts" / "x" / "a.json"
    csv_path = Path(tmp_path) / "r1" / "artifacts" / "x" / "a.csv"

    writer.write_json_atomic(json_path, {"a": 1, "b": 2})
    writer.write_csv_atomic(csv_path, [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}])

    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == {"a": 1, "b": 2}

    assert "x,y" in csv_path.read_text(encoding="utf-8")


def test_artifact_writer_path_uses_budget_segment_for_non_ratio(tmp_path):
    writer = ArtifactWriter(tmp_path)

    path = writer.path_for(
        run_id="run123",
        dataset_id="my-dataset",
        sched_time_ratio=None,
        budget_mode="fixed_time",
        budget_value=30.0,
        execution_id="exec-1",
        artifact_type="quality",
        ext="json",
    )

    assert "budget_fixed_time_30" in path.as_posix()
