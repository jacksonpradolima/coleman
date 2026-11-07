"""Persist coupled finite experiment capacities before invoking outcome loaders."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.evaluation.design._wire import DesignContract

KIB = 1024
MIB = 1024**2
Count = Annotated[int, Field(strict=True, ge=0)]


class DesignCounts(CoreImmutable):
    """Describe actual finite membership counts without enumerating the grid.

    Attributes
    ----------
    candidates, source_members, protected_members, units, seeds : int
        Actual immutable catalogue dimensions; cells and aggregate inputs are
        derived products. Seeds are never independent units.
    hypotheses, metrics, trials, physical_attempts, output_references : int
        Distinct family/evidence counts. Physical attempts follow declared work
        grouping rather than one assumed job for every output cell.

    Notes
    -----
    Zero is meaningful for absent attempt ledgers, not an imputed physical cost.
    This leaf performs bounded integer arithmetic only and invokes no callback.
    """

    candidates: Count = 0
    source_members: Count = 0
    protected_members: Count = 0
    units: Count = 0
    seeds: Count = 0
    hypotheses: Count = 0
    metrics: Count = 0
    trials: Count = 0
    physical_attempts: Count = 0
    output_references: Count = 0

    @property
    def cells(self) -> int:
        """Return candidate/member/seed cells, without allocating their product."""
        return self.candidates * self.protected_members * self.seeds

    @property
    def aggregate_inputs(self) -> int:
        """Return one required derived input per candidate and declared unit."""
        return self.candidates * self.units


class ProtocolLimits(DesignContract):
    """Freeze independent ceilings plus a coupled complete-envelope reservation.

    Attributes
    ----------
    max_envelope_bytes, max_metadata_bytes : int
        Default 2 MiB complete artifact and 512 KiB repeated metadata; hard ceilings
        are 64 MiB and 8 MiB. Every serialized occurrence is charged.
    max_cells, max_aggregate_inputs : int
        Default 128 each; hard caps 8192 and 4096. Their independent maxima are not a
        promise that every Cartesian combination fits the byte budget.
    max_raw_record_bytes, max_raw_header_bytes : int
        Original canonical raw body, default 2 KiB and hard cap 8 KiB, and actual
        UTF-8 row header, 1 KiB. Identifier type alone supplies no length guarantee.
    max_source_receipts, max_audit_receipts, max_receipt_bytes : int
        Default 128 receipts per chain, hard cap 512, each at most 2 KiB including LF.
        Exact-full prefixes remain readable; appends require separate admission.
    result_reserve_bytes, admission_margin_bytes : int
        Explicit family/framing allowance and unused admission margin. Envelopes
        above 2 MiB require at least 1 MiB unused margin.

    Notes
    -----
    All counts/bytes are strict integers. The remaining max_* fields bound their
    named catalogues, ledgers, configurations and actual directory acquisition.
    These are data/owned-I/O limits, not RAM, CPU, free-disk or sandbox guarantees.
    Larger values require an explicit new declaration within fixed hard caps.
    No reader or existing artifact is silently modified.
    """

    max_candidates: Annotated[int, Field(strict=True, ge=2, le=64)] = 4
    max_source_members: Annotated[int, Field(strict=True, ge=1, le=4096)] = 256
    max_protected_members: Annotated[int, Field(strict=True, ge=1, le=2048)] = 64
    max_units: Annotated[int, Field(strict=True, ge=1, le=1024)] = 32
    max_seeds: Annotated[int, Field(strict=True, ge=1, le=64)] = 4
    max_cells: Annotated[int, Field(strict=True, ge=1, le=8192)] = 128
    max_aggregate_inputs: Annotated[int, Field(strict=True, ge=1, le=4096)] = 128
    max_hypotheses: Annotated[int, Field(strict=True, ge=1, le=64)] = 8
    max_metrics: Annotated[int, Field(strict=True, ge=1, le=32)] = 8
    max_trials: Annotated[int, Field(strict=True, ge=0, le=256)] = 32
    max_physical_attempts: Annotated[int, Field(strict=True, ge=0, le=4096)] = 32
    max_output_references: Annotated[int, Field(strict=True, ge=0, le=16384)] = 512
    max_source_receipts: Annotated[int, Field(strict=True, ge=1, le=512)] = 128
    max_audit_receipts: Annotated[int, Field(strict=True, ge=1, le=512)] = 128
    max_directory_entries: Annotated[int, Field(strict=True, ge=1, le=576)] = 192
    max_directory_bytes: Annotated[int, Field(strict=True, ge=1, le=2 * MIB)] = MIB
    max_graph_items: Annotated[int, Field(strict=True, ge=1, le=2_000_000)] = 65536
    max_raw_record_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * KIB)] = 2 * KIB
    max_raw_header_bytes: Annotated[int, Field(strict=True, ge=1, le=KIB)] = KIB
    max_aggregate_record_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * KIB)] = 2 * KIB
    max_input_header_bytes: Annotated[int, Field(strict=True, ge=1, le=KIB)] = KIB
    max_configuration_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * KIB)] = 4 * KIB
    max_freeze_bytes: Annotated[int, Field(strict=True, ge=1, le=2 * MIB)] = 128 * KIB
    max_design_bytes: Annotated[int, Field(strict=True, ge=1, le=2 * MIB)] = 256 * KIB
    max_selection_bytes: Annotated[int, Field(strict=True, ge=1, le=6 * MIB)] = 256 * KIB
    max_metadata_bytes: Annotated[int, Field(strict=True, ge=1, le=8 * MIB)] = 512 * KIB
    max_receipt_bytes: Annotated[int, Field(strict=True, ge=1, le=2 * KIB)] = 2 * KIB
    result_reserve_bytes: Annotated[int, Field(strict=True, ge=1, le=2 * MIB)] = 128 * KIB
    admission_margin_bytes: Annotated[int, Field(strict=True, ge=64 * KIB, le=MIB)] = 64 * KIB
    max_envelope_bytes: Annotated[int, Field(strict=True, ge=1, le=64 * MIB)] = 2 * MIB

    @model_validator(mode="after")
    def _profile(self) -> Self:
        """Require readable full histories and the larger profile's fixed margin."""
        if self.max_directory_entries < max(self.max_source_receipts, self.max_audit_receipts) + 2:
            raise ValueError("Receipt directory needs complete-prefix and publication headroom")
        if (
            self.max_directory_bytes
            < (max(self.max_source_receipts, self.max_audit_receipts) + 2) * self.max_receipt_bytes
        ):
            raise ValueError(
                "Directory byte budget cannot hold its complete declared receipt prefix and publication headroom"
            )
        if self.max_envelope_bytes > 2 * MIB and self.admission_margin_bytes < MIB:
            raise ValueError("Larger envelopes require one MiB unused admission margin")
        return self

    def require_counts(self, counts: DesignCounts) -> None:
        """Reject inadmissible dimensions before constructing any expected grid.

        Parameters
        ----------
        counts : DesignCounts
            Actual declared counts, not maxima multiplied together speculatively.

        Raises
        ------
        ValueError
            A count or bounded product exceeds this persisted profile. No
            enumeration, file access or outcome callback occurs.
        """
        counts = DesignCounts.model_validate(counts)
        dimensions = (*DesignCounts.model_fields, "cells", "aggregate_inputs")
        for name in dimensions:
            if getattr(counts, name) > getattr(self, "max_" + name):
                raise ValueError(f"Experiment {name} exceed the declared capacity")

    def reserve(  # pylint: disable=too-many-arguments
        self,
        cells: int,
        aggregate_inputs: int,
        *,
        metadata_bytes: int | None = None,
        result_bytes: int | None = None,
        source_receipts: int | None = None,
        audit_receipts: int | None = None,
    ) -> int:
        """Check a sufficient complete-envelope reservation before data loading.

        Parameters
        ----------
        cells, aggregate_inputs : int
            Actual bounded raw and derived input products.
        metadata_bytes, result_bytes : int or None, optional
            Actual fully charged repeated metadata and conservative result/frame
            reservation. None reserves this profile's complete named allowance.
        source_receipts, audit_receipts : int or None, optional
            Complete prefix lengths; None reserves each declared chain capacity.

        Returns
        -------
        int
            Reserved bytes excluding the separately required unused margin.

        Raises
        ------
        ValueError
            Invalid/coerced/negative counts, an individual limit, or an envelope
            whose reservation plus margin does not fit. Inputs are not mutated.
        """
        actual = (
            (cells, self.max_cells),
            (aggregate_inputs, self.max_aggregate_inputs),
            (self.max_metadata_bytes if metadata_bytes is None else metadata_bytes, self.max_metadata_bytes),
            (self.result_reserve_bytes if result_bytes is None else result_bytes, self.result_reserve_bytes),
            (self.max_source_receipts if source_receipts is None else source_receipts, self.max_source_receipts),
            (self.max_audit_receipts if audit_receipts is None else audit_receipts, self.max_audit_receipts),
        )
        raw, derived, metadata, framing, source, old = _checked_values(actual)
        total = (
            raw * (self.max_raw_record_bytes + self.max_raw_header_bytes)
            + derived * (self.max_aggregate_record_bytes + self.max_input_header_bytes)
            + metadata
            + (source + old) * self.max_receipt_bytes
            + framing
        )
        if total + self.admission_margin_bytes > self.max_envelope_bytes:
            raise ValueError("Complete envelope reservation and unused margin exceed the declared byte budget")
        return total


def _checked_values(pairs: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """Validate explicit named capacity values before reservation arithmetic."""
    for value, maximum in pairs:
        if type(value) is not int or not 0 <= value <= maximum:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Envelope reservation requires strict bounded nonnegative counts")
    return tuple(value for value, _ in pairs)
