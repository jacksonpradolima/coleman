"""Pure repetition cells retaining the original domain record and identity."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreImmutable
from coleman.core.primitives import Identifier
from coleman.evaluation._repetition import repetition_execution_id
from coleman.evaluation.design._wire import wire_size
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.results.domain import DomainRecord

__all__ = ["RawCell", "repetition_execution_id"]


class RawCell(CoreImmutable):
    """Bind one required candidate/member/seed cell to its unmodified record.

    Attributes
    ----------
    candidate_id, member_id : str
        Exact selected candidate and source member. The record's decision ID is
        the member ID; its execution ID binds the candidate/seed coordinate.
    seed : int
        Unsigned 64-bit repetition coordinate, never an independent unit or a
        physical-attempt count.
    state : {'completed', 'failed', 'aborted', 'skipped'}
        Supplied logical execution disposition. Completed negative outcomes are
        valid results; measurement status remains separate from this state.
    evidence : OriginalRecordEvidence
        Original record, logical key and digest. No source, configuration,
        timestamp, missingness or optional-field default is rewritten.
    reason : str or None, optional
        Supplied completion/applicability explanation. Omission remains omitted
        in the enclosing panel; no reason is inferred from a null outcome.

    Raises
    ------
    ValueError
        Invalid fields, seed, record identity or a cell relabeled with another
        candidate/member/repetition coordinate.

    Notes
    -----
    Construction does not run an experiment, read a source, infer independence
    or reserve protected access. Physical groups and their actual costs belong
    to the separate protected attempt ledger, not this row's seed count.
    """

    candidate_id: Identifier
    member_id: Identifier
    seed: Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
    state: Literal["completed", "failed", "aborted", "skipped"]
    evidence: OriginalRecordEvidence
    reason: Identifier | None = None

    @model_validator(mode="after")
    def _cell_binding(self) -> Self:
        """Reject moving an unchanged producer record to another grid cell."""
        if self.record.decision_id != self.member_id or self.record.execution_id != repetition_execution_id(
            self.candidate_id, self.seed
        ):
            raise ValueError("Original raw record identity does not bind its declared repetition cell")
        return self

    @property
    def record(self) -> DomainRecord:
        """Return the unmodified typed producer record, without I/O or projection."""
        return self.evidence.record

    @property
    def record_digest(self) -> str:
        """Return its verified original canonical digest, not the row's identity."""
        return self.evidence.record_digest

    @property
    def header_bytes(self) -> int:
        """Measure actual UTF-8 row framing and references, excluding its record.

        String escaping and original key/digest references count. This is a
        serialization measurement, not a fixed-size claim about identifiers.
        """
        return wire_size(self.model_dump(mode="json", exclude={"evidence": {"record"}}, exclude_unset=True))
