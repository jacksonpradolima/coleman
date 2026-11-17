"""Reproducible exact JSON/Parquet projections with complete-last publication."""

from __future__ import annotations

import io
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from coleman.artifacts import ArtifactWriter
from coleman.core.primitives import canonical_json
from coleman.objectives.volatility import clipping_statistics
from coleman.problems.portfolio import Exact

from . import model
from .flow_contracts import CandidateStep, ControlObjective, ExperimentState


def _exact(value: Exact | None) -> str:
    """Require complete original accounting before producing an exact column."""
    if value is None:
        raise ValueError("Complete artifact cannot fabricate missing settlement quantities")
    return str(value.fraction)


def _objective_columns(objective: ControlObjective | None) -> dict[str, str | None]:
    """Project separate mathematical diagnostic columns at the Arrow boundary."""
    fields = (
        "coefficient_solver_proximity_exact",
        "coefficient_l1_penalty_exact",
        "coefficient_solver_total_exact",
        "coefficient_control_full_l1_exact",
        "coefficient_l1_strength_exact",
    )
    values = (
        (None,) * 5
        if objective is None
        else (
            objective.proximity,
            objective.weighted_penalty,
            objective.total,
            objective.coefficient_l1,
            objective.strength,
        )
    )
    columns = {name: None if value is None else str(value.fraction) for name, value in zip(fields, values, strict=True)}
    columns["coefficient_objective_status"] = "not_applicable" if objective is None else objective.status
    columns["coefficient_objective_unit"] = "coefficient-fraction-squared"
    return columns


def _candidate_record(index: int, row: CandidateStep) -> dict[str, str | int | float | None]:
    """Project one complete typed settlement; no missing numerical values are filled."""
    return {
        "period": index,
        "candidate": row.name,
        "status": row.settlement.status,
        "currency": "synthetic-credit",
        "gross_value_exact": _exact(row.settlement.gross_value),
        "net_value_exact": _exact(row.settlement.net_value),
        "executed_full_l1_notional_exact": _exact(row.settlement.turnover),
        "proportional_paid_exact": _exact(row.settlement.proportional_paid),
        "recurring_paid_exact": _exact(row.settlement.recurring_paid),
        "coefficient_movement": row.coefficient_movement,
        "coefficient_release_disposition": None if row.control is None else row.control.disposition,
        **_objective_columns(row.control_objective),
        "settlement_fingerprint": row.settlement.fingerprint,
    }


def publish(state: ExperimentState, directory: Path, *, guard: Callable[[], None] = lambda: None) -> Path:
    """Publish a completed authored trajectory with exact quantities retained.

    Parameters
    ----------
    state : ExperimentState
        Complete16-period same-source result. A partial/in-flight state refuses
        before any output is written; checkpoints use the state store instead.
    directory : pathlib.Path
        Trusted caller-owned artifact root. Cooperative concurrent same-content
        writers use existing immutable publication; conflicting content rejects.
    guard : callable, optional
        Caller-owned cancellation/resource guard invoked before each publication.
        Exceptions propagate without a completion manifest; default permits work.

    Returns
    -------
    pathlib.Path
        Final completion manifest, written only after both data files exist.

    Raises
    ------
    ValueError, OSError, ImportError
        Partial state, refused settlement, conflicting/corrupt publication,
        unavailable filesystem or missing optional PyArrow installation.

    Notes
    -----
    Parquet columns preserve exact rational text, declared units and separate
    coefficient/turnover/paid-charge meanings. Pinned PyArrow23.0.1 emits the
    same deterministic bytes here; different writer versions need not. This
    publisher claims process-interruption atomicity, not power-loss durability.
    """
    current = ExperimentState.model_validate(state)
    if current.next_index != 16 or any(
        row.settlement.status != "executed" for period in current.periods for row in period.candidates
    ):
        raise ValueError("Complete synthetic artifacts require every executed original period")
    # Optional outer storage dependency never enters the accounting domain.
    import pyarrow as pa  # pylint: disable=import-outside-toplevel
    from pyarrow import parquet  # pylint: disable=import-outside-toplevel

    records = [_candidate_record(period.index, row) for period in current.periods for row in period.candidates]
    table = pa.Table.from_pylist(records).replace_schema_metadata(
        {b"contract": b"coleman-synthetic-funded-portfolio-v1", b"state_fingerprint": current.fingerprint.encode()}
    )
    sink = io.BytesIO()
    parquet.write_table(table, sink, version="2.6", compression="zstd", compression_level=9, use_dictionary=False)
    parquet_bytes = sink.getvalue()
    json_bytes = current.to_json().encode()
    writer = ArtifactWriter(directory)
    guard()
    writer.write_immutable_stream(directory / "trajectory.json", (json_bytes,), maximum_bytes=16 * 1024**2)
    guard()
    writer.write_immutable_stream(directory / "trajectory.parquet", (parquet_bytes,), maximum_bytes=16 * 1024**2)
    clipping = tuple(
        clipping_statistics(
            tuple(row.applications[member] for row in current.observations if (row.index < 6) == development),
            split=model.ref("labels-available-development" if development else "labels-available-protected"),
        ).model_dump(mode="json")
        for member in range(3)
        for development in (True, False)
    )
    manifest = canonical_json(
        {
            "schema_version": 1,
            "status": "complete",
            "interpretation": "authored-synthetic-descriptive-only",
            "state_fingerprint": current.fingerprint,
            "json_sha256": "sha256:" + sha256(json_bytes).hexdigest(),
            "parquet_sha256": "sha256:" + sha256(parquet_bytes).hexdigest(),
            "rows": len(records),
            "clipping_statistics": list(clipping),
        }
    )
    destination = directory / "complete.json"
    guard()
    writer.write_immutable(destination, manifest)
    return destination
