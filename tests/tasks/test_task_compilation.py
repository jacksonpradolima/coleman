"""Concrete proof/value boundaries, separate from real installed ACK qualification."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from coleman.application.task_runtime import _Progress, _progress, _require_prepared
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.tasks.contracts import OwnedCommitProof, TaskPlan
from coleman.tasks.identity import CodecDeclaration
from coleman.tasks.protocols import CommittedProgress, CoordinateContext, PreparationContext

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"


@dataclass(frozen=True)
class _ProofCodec:
    """Use a real concrete proof model; these authored declarations are not ACKs."""

    @property
    def declaration(self):
        """Declare this concrete known model and its finite wire capacity."""
        schema = fingerprint(OwnedCommitProof.model_json_schema())
        return CodecDeclaration(
            binding=DefinitionBinding(
                reference=Reference(id="test-concrete-proof-codec", version="1"), definition_fingerprint=schema
            ),
            schema_fingerprint=schema,
            maximum_bytes=16384,
        )

    def validate(self, value, /):
        """Revalidate the complete declaration without establishing acceptance."""
        return OwnedCommitProof.model_validate(value)

    def encode(self, value, /):
        """Encode all concrete fields using the existing strict proof serializer."""
        return self.validate(value).to_json()

    def decode(self, body, /):
        """Reject unknown fields or versions through the real concrete reader."""
        return OwnedCommitProof.from_json(body)


@dataclass(frozen=True)
class _Verifier:
    """Record typed delegation only, without claiming a real storage acceptance."""

    validation: Callable[[CommittedProgress[Any], CoordinateContext], None]
    recovery: Callable[[CoordinateContext], CommittedProgress[Any] | None]

    def validate(self, progress, context, /):
        """Record selected validation delegation, with no storage claim."""
        self.validation(progress, context)

    def reconcile(self, context, /):
        """Return only the deliberately supplied unit recovery candidate."""
        return self.recovery(context)


def _values(tmp_path):
    plan = TaskPlan.from_json((FIXTURES / "plan.json").read_bytes())
    proof = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    coordinate = plan.coordinates[0]
    preparation = PreparationContext(
        plan.fingerprint, plan.source, plan.implementation.binding, proof.compatibility, tmp_path
    )
    context = CoordinateContext(
        preparation, coordinate.fingerprint, coordinate.reference, coordinate.target, tmp_path / "coordinate"
    )
    progress = CommittedProgress(plan.fingerprint, coordinate.fingerprint, proof.frontier, proof)
    return plan, progress, context


def test_progress_codec_rejects_incomplete_proof_before_selected_storage_callback(tmp_path):
    """Do not send malformed proofs into the selected storage verifier."""
    _, progress, context = _values(tmp_path)
    called = []
    verifier = _Verifier(lambda *_args: called.append("verify"), lambda _context: None)
    selected = _Progress(verifier, _ProofCodec())
    malformed = CommittedProgress(progress.plan_fingerprint, progress.coordinate_fingerprint, progress.frontier, 1)
    with pytest.raises(ValueError):
        selected.validate(malformed, context)
    assert not called
    selected.validate(progress, context)
    assert called == ["verify"]  # Typed delegation only; this stub never certifies actual storage.


@pytest.mark.parametrize("present", (False, True))
def test_reconcile_preserves_missing_or_complete_concrete_proof_without_fabricating_acceptance(tmp_path, present):
    """Preserve absent recovery and exact complete typed declarations separately."""
    _, progress, context = _values(tmp_path)
    callbacks = []

    def recover(selected_context):
        assert selected_context == context
        callbacks.append("actual-selected-recovery")
        return progress if present else None

    verifier = _Verifier(lambda *_args: None, recover)
    result = _Progress(verifier, _ProofCodec()).reconcile(context)
    assert callbacks == ["actual-selected-recovery"]
    assert result == (progress if present else None)
    if result is not None:
        assert isinstance(result.proof, OwnedCommitProof)
        assert result.proof.to_json() == progress.proof.to_json()


def test_recovered_malformed_declaration_is_not_promoted_to_typed_progress(tmp_path):
    """Reject an uncontracted recovery body before outer progress can use it."""
    _, progress, context = _values(tmp_path)
    malformed = CommittedProgress(progress.plan_fingerprint, progress.coordinate_fingerprint, progress.frontier, {})
    verifier = _Verifier(lambda *_args: None, lambda _context: malformed)
    selected = _Progress(verifier, _ProofCodec())
    with pytest.raises(ValueError):
        selected.reconcile(context)


def test_optional_progress_requires_both_concrete_codec_and_actual_verifier():
    """Keep stateless absence valid while refusing an undeclared proof codec."""
    # Deliberately partial prepared objects isolate this optional-capability guard.
    assert _progress(cast(Any, SimpleNamespace(progress_verifier=None)), None) is None
    selected = cast(
        Any, SimpleNamespace(progress_verifier=SimpleNamespace(validate=lambda *_args: None, reconcile=lambda _: None))
    )
    with pytest.raises(ValueError, match="concrete proof codec"):
        _progress(selected, None)
    assert isinstance(_progress(selected, _ProofCodec()), _Progress)


@pytest.mark.parametrize("missing", ("read_completed", "advance", "publish_coordinate", "boundary", "verifier"))
def test_prepared_local_state_capabilities_refuse_before_any_reservation(missing):
    """Refuse each missing declared capability before physical work is reserved."""
    plan = TaskPlan.from_json((FIXTURES / "plan.json").read_bytes())
    values: dict[str, object] = {
        "read_completed": lambda *_args: None,
        "advance": lambda *_args: None,
        "publish_coordinate": lambda *_args: None,
        "boundary": SimpleNamespace(produce=lambda _load: None),
        "progress_verifier": SimpleNamespace(validate=lambda *_args: None, reconcile=lambda _: None),
    }
    if missing == "verifier":
        values["progress_verifier"] = None
    else:
        values[missing] = None
    prepared = SimpleNamespace(**values)
    with pytest.raises((ValueError, TypeError)):
        _require_prepared(prepared, plan)
