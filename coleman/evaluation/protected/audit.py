"""Append-only protected-data access receipts over COLEMAN's ArtifactWriter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import uuid4

from pydantic import Field

from coleman.artifacts import ArtifactWriter
from coleman.evaluation.protected.contracts import EvaluationContract, FrozenEvaluation
from coleman.evaluation.protected.temporal import Partition, Timestamp
from coleman.results.domain import Fingerprint, Identifier, fingerprint

if TYPE_CHECKING:
    from coleman.evaluation.protected.comparison import FamilyReport


class AccessReceipt(EvaluationContract):
    """One durable attempted access, including failures after reservation.

    Sequence counts all train/validation/test accesses within the same explicit
    holdout scope. ``prior_test_accesses`` counts every previous test access,
    including tuning or failed evaluations, so reuse cannot masquerade as virgin.
    Changing candidates/protocols/data does not reset a stable ``holdout_id``.
    Timestamp is audit request time (not outcome availability); access_id names
    a unique API invocation, not independent scientific evidence or an RNG seed.
    """

    holdout_id: Identifier
    sequence: Annotated[int, Field(strict=True, ge=1)]
    access_id: Identifier
    evaluation_fingerprint: Fingerprint
    data_fingerprint: Fingerprint
    partition: Partition
    purpose: Literal["development", "tuning", "confirmatory", "verification"]
    actor: Identifier
    requested_at: Timestamp
    prior_test_accesses: Annotated[int, Field(strict=True, ge=0)]
    previous_fingerprint: Fingerprint | None


class EvaluationAudit:
    """Persist frozen contracts and serialized access reservations without a new runner.

    Immutable ordinal files use atomic create-if-absent publication. Concurrent
    contenders reread the chain before retrying; no stale lock survives a crash.
    A reservation precedes outcome loading, and is counted even if loading or
    inference later fails. This is an audit of this API, not access control over
    externally readable files or proof that a caller did not inspect data elsewhere.
    Holdout IDs must remain stable across experiments on the same observations.
    Chain validation detects gaps/corruption; it is not an authenticated tamper seal.
    """

    def __init__(self, artifact_root: str | Path) -> None:
        """Configure paths only; filesystem side effects begin with save/access."""
        self.root = Path(artifact_root)
        self.writer = ArtifactWriter(self.root)

    def save_freeze(self, frozen: FrozenEvaluation) -> Path:
        """Publish a revalidated frozen contract before protected access.

        Identical publication is idempotent. Conflict, unknown versions or I/O
        failures propagate; the writer's local-filesystem durability limits apply.
        """
        frozen = FrozenEvaluation.model_validate(frozen)
        path = self.root / "evaluation-contracts" / f"{frozen.fingerprint.removeprefix('sha256:')}.json"
        self.writer.write_immutable(path, frozen.to_json() + "\n")
        return path

    def save_report(self, report: FamilyReport) -> Path:
        """Publish a versioned report only when its access receipt is in this audit.

        Identical report publication is idempotent. Unreserved/mismatched receipts
        fail before file writes; interrupted reservations remain attempts, not
        completed reports. Local writer errors propagate to the caller.
        """
        report = type(report).model_validate(report)
        history = self.history(report.access.holdout_id)
        index = report.access.sequence - 1
        if index >= len(history) or history[index] != report.access:
            raise ValueError("Report access receipt is not reserved in this audit")
        if self.load_freeze(report.frozen.fingerprint) != report.frozen:
            raise ValueError("Report frozen contract differs from persisted contract")
        path = self.root / "evaluation-reports" / f"{report.fingerprint.removeprefix('sha256:')}.json"
        self.writer.write_immutable(path, report.to_json() + "\n")
        return path

    def load_freeze(self, digest: str) -> FrozenEvaluation:
        """Load a v1 contract by identified fingerprint, failing on mismatched content."""
        if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
            raise ValueError("Expected an identified SHA-256 fingerprint")
        if any(character not in "0123456789abcdef" for character in digest[7:]):
            raise ValueError("Invalid SHA-256 digest")
        path = self.root / "evaluation-contracts" / f"{digest[7:]}.json"
        frozen = FrozenEvaluation.from_json(self._freeze_bytes(path))
        if frozen.fingerprint != digest:
            raise ValueError("Frozen evaluation fingerprint mismatch")
        return frozen

    def _scope_root(self, holdout_id: str) -> Path:
        """Hash explicit holdout identity so arbitrary names never become paths."""
        return self.root / "evaluation-access" / fingerprint([holdout_id]).removeprefix("sha256:")

    def _history_paths(self, holdout_id: str) -> list[Path]:
        """Acquire ordered receipt paths using the established legacy behavior.

        Optional bounded views override this acquisition boundary rather than
        scanning first and then falling back to an unbounded parent read. This
        default retains v1 glob ordering and its handling of temporary files.
        """
        return sorted(self._scope_root(holdout_id).glob("*.json"))

    def _receipt_bytes(self, path: Path) -> bytes:
        """Acquire one receipt with the unchanged legacy byte-reading behavior.

        Bounded views replace this operation with a finite read before parsing.
        Filesystem errors propagate; this default imposes no new v1 size limit.
        """
        return path.read_bytes()

    def _freeze_bytes(self, path: Path) -> bytes:
        """Acquire a frozen contract without changing the existing v1 reader.

        A bounded view can guard bytes, nested counts and configuration strings
        before legacy model validation. This default preserves prior semantics.
        """
        return path.read_bytes()

    def history(self, holdout_id: str) -> tuple[AccessReceipt, ...]:
        """Read and validate the complete immutable scope chain, ignoring .tmp files.

        Snapshot iteration is not a global transaction. Missing/corrupt/unknown
        entries fail closed; no repair invents past access events. External
        deletion is unsupported and detected by subsequent chain mismatches.
        """
        receipts = []
        previous = None
        test_accesses = 0
        for sequence, path in enumerate(self._history_paths(holdout_id), start=1):
            receipt = AccessReceipt.from_json(self._receipt_bytes(path))
            if (
                path.name != f"{sequence:012d}.json"
                or receipt.sequence != sequence
                or receipt.holdout_id != holdout_id
                or receipt.previous_fingerprint != previous
                or receipt.prior_test_accesses != test_accesses
            ):
                raise ValueError("Invalid protected-access audit chain")
            previous = receipt.fingerprint
            test_accesses += receipt.partition == "test"
            receipts.append(receipt)
        return tuple(receipts)

    def access(
        self,
        frozen: FrozenEvaluation,
        *,
        partition: Partition,
        purpose: Literal["development", "tuning", "confirmatory", "verification"],
        actor: str,
    ) -> AccessReceipt:
        """Reserve an access before any outcome loader runs, returning a durable receipt.

        Every invocation is a new access, including retries after interruption;
        no caller-supplied idempotency key can hide repeated confirmatory reads.
        Test tuning counts as test exposure. Native training/validation history
        is recorded without incrementing protected exposure. At most 1000
        publication conflicts are retried, then ``RuntimeError`` reports contention.
        Other validation/filesystem errors propagate immediately.
        """
        frozen = FrozenEvaluation.model_validate(frozen)
        requested_at = datetime.now(UTC)
        access_id = uuid4().hex
        # Validate the request before publishing even the frozen contract.
        request = AccessReceipt(
            schema_version=1,
            holdout_id=frozen.protocol.holdout_id,
            sequence=1,
            access_id=access_id,
            evaluation_fingerprint=frozen.fingerprint,
            data_fingerprint=frozen.data_fingerprint,
            partition=partition,
            purpose=purpose,
            actor=actor,
            requested_at=requested_at,
            prior_test_accesses=0,
            previous_fingerprint=None,
        )
        self.save_freeze(frozen)
        scope = self._scope_root(request.holdout_id)
        for _ in range(1000):
            history = self.history(request.holdout_id)
            payload = request.model_dump()
            payload.update(
                sequence=len(history) + 1,
                prior_test_accesses=sum(item.partition == "test" for item in history),
                previous_fingerprint=history[-1].fingerprint if history else None,
            )
            receipt = AccessReceipt.model_validate(payload)
            path = scope / f"{receipt.sequence:012d}.json"
            try:
                self.writer.write_immutable(path, receipt.to_json() + "\n")
            except ValueError:
                # A different complete receipt won this ordinal. Rereading the
                # entire validated chain accounts for it before the next attempt.
                if not path.exists():
                    raise
            else:
                return receipt
        raise RuntimeError("Protected-access reservation exceeded contention limit")
