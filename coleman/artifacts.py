"""Deterministic artifact writing helpers for hooks and extensions."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


class ArtifactWriter:
    """Path builder and atomic writer for hook artifacts."""

    def __init__(self, run_root: str | Path) -> None:
        """Initialize writer with the base runs root path."""
        self.run_root = Path(run_root)

    def path_for(
        self,
        *,
        run_id: str,
        dataset_id: str | None,
        sched_time_ratio: float | None = None,
        budget_mode: str | None = None,
        budget_value: float | None = None,
        execution_id: str | None,
        artifact_type: str,
        ext: str,
        stem: str = "artifact",
    ) -> Path:
        """Build a deterministic artifact path under a run folder."""
        safe_dataset = _safe_segment(dataset_id or "dataset_unknown")
        safe_execution = _safe_segment(execution_id or "execution_unknown")
        safe_type = _safe_segment(artifact_type)
        safe_stem = _safe_segment(stem)
        ext = ext.lstrip(".")

        effective_budget_mode = (budget_mode or "ratio").lower()
        effective_budget_value = budget_value
        if effective_budget_value is None and sched_time_ratio is not None:
            effective_budget_value = float(sched_time_ratio)

        ratio_segment = "time_ratio_na"
        if effective_budget_mode == "ratio" and effective_budget_value is not None:
            ratio_segment = f"time_ratio_{int(round(float(effective_budget_value) * 100)):02d}"
        elif effective_budget_value is not None:
            value_segment = str(
                int(effective_budget_value)
                if int(effective_budget_value) == effective_budget_value
                else effective_budget_value
            ).replace(".", "_")
            ratio_segment = f"budget_{_safe_segment(effective_budget_mode)}_{_safe_segment(value_segment)}"

        base = self.run_root / run_id / "artifacts" / safe_type / ratio_segment / safe_dataset / safe_execution
        return base / f"{safe_stem}.{ext}"

    def write_json_atomic(self, path: Path, payload: Any, indent: int = 2) -> None:
        """Write JSON payload atomically (best effort across platforms)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def write_csv_atomic(self, path: Path, rows: list[dict[str, Any]]) -> None:
        """Write row-oriented CSV payload atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        fieldnames = sorted({key for row in rows for key in row})
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)


def _safe_segment(value: str) -> str:
    """Sanitize one filesystem path segment using conservative ASCII-safe chars."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "artifact"
