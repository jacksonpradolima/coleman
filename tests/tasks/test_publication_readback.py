"""Optional publication/readback preserves the real concrete host boundary.

The small fixed file exercises the shared wrapper and immutable writer. It is
not a mathematical ACK, a captured external provider or a complete task job.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, cast

import pytest
from pydantic import Field

from coleman.application.task_runtime import _Prepared, _publication_readback, _require_prepared
from coleman.artifacts import ArtifactWriter
from coleman.core.base import VersionedContract
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.storage_contracts import ExecutionCompatibility
from coleman.tasks.contracts import TaskPlan
from coleman.tasks.identity import CodecDeclaration
from coleman.tasks.protocols import CoordinateContext, PreparationContext, PublicationReadbackV1


class _Result(VersionedContract):
    """Use a real finite typed codec, including observable signed zero."""

    value: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    unit: Literal["seconds"] = "seconds"


@dataclass(frozen=True)
class _Codec:
    """Keep concrete model validation and complete canonical bytes real."""

    @property
    def declaration(self):
        schema = fingerprint(_Result.model_json_schema())
        return CodecDeclaration(
            binding=DefinitionBinding(
                reference=Reference(id="publication-readback-fixture", version="1"), definition_fingerprint=schema
            ),
            schema_fingerprint=schema,
            maximum_bytes=1024,
        )

    def validate(self, value, /):
        return _Result.model_validate(value)

    def encode(self, value, /):
        return self.validate(value).to_json()

    def decode(self, body, /):
        return _Result.from_json(body)


class _Legacy:
    """Publish a fixed actual immutable file and perform a fresh bounded read."""

    def __init__(self):
        self.calls = []

    def publish_coordinate(self, coordinate, value, context, /):
        self.calls.append(("legacy-publish", coordinate, context))
        self._write(value, context)

    def read_completed(self, coordinate, context, /):
        self.calls.append(("legacy-read", coordinate, context))
        return self._read(context)

    @staticmethod
    def _write(value, context):
        ArtifactWriter(context.artifact_directory).write_immutable(
            context.artifact_directory / "coordinate.json", _Codec().encode(value)
        )

    @staticmethod
    def _read(context):
        with (context.artifact_directory / "coordinate.json").open("rb") as stream:
            payload = stream.read(1025)
        if len(payload) > 1024:
            raise ValueError("Actual publication readback exceeds its declared byte bound")
        return _Codec().decode(payload)


class _Combined(_Legacy):
    """Supply the typed optional operation while retaining legacy methods."""

    profile = Reference(id="coleman-coordinate-publication-readback", version="1")

    @property
    def publication_readback(self) -> PublicationReadbackV1[str, _Result]:
        return self

    def publish_and_read(self, coordinate: str, value: _Result, context: CoordinateContext, /) -> _Result:
        self.calls.append(("combined", coordinate, context))
        self._write(value, context)
        return self._read(context)


def _context(tmp_path):
    # Constructed metadata only, not a claim to have admitted this fixture image.
    path = Path(__file__).parents[1] / "fixtures/tasks/v1/plan.json"
    plan = TaskPlan.from_json(path.read_bytes())
    coordinate = plan.coordinates[0]
    preparation = PreparationContext(
        plan.fingerprint,
        plan.source,
        plan.implementation.binding,
        ExecutionCompatibility(
            schema_version=1,
            runtime=plan.implementation.runtime,
            owner_implementations=(plan.implementation.binding,),
        ),
        tmp_path,
    )
    return plan, CoordinateContext(
        preparation, coordinate.fingerprint, coordinate.reference, coordinate.target, tmp_path / "coordinate"
    )


def _wrapped(selected):
    # Publication unit only; no fake driver, owner, storage ACK or job completion.
    return _Prepared(cast(Any, selected), _Codec(), cast(Any, None), None)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_absent_or_none_capability_keeps_original_publish_then_read(explicit_none, tmp_path):
    selected = _Legacy()
    if explicit_none:
        cast(Any, selected).publication_readback = None
    _, context = _context(tmp_path)
    value = _Result(schema_version=1, value=0.0)
    _wrapped(selected).publish_coordinate("coordinate", value, context)
    assert [row[0] for row in selected.calls] == ["legacy-publish", "legacy-read"]
    assert _Codec().encode(selected.read_completed("coordinate", context)) == value.to_json()


def test_supported_handler_writes_and_reads_actual_bytes_without_legacy_duplicate(tmp_path):
    selected = _Combined()
    _, context = _context(tmp_path)
    value = _Result(schema_version=1, value=-1.0)
    wrapped = _wrapped(selected)
    wrapped.publish_coordinate("coordinate", value, context)
    assert selected.calls == [("combined", "coordinate", context)]
    assert (context.artifact_directory / "coordinate.json").read_text() == value.to_json()
    wrapped.publish_coordinate("coordinate", value, context)  # Actual immutable idempotent retry.
    assert [row[0] for row in selected.calls] == ["combined", "combined"]
    assert wrapped.read_completed("coordinate", context).to_json() == value.to_json()
    assert selected.calls[-1][0] == "legacy-read"  # Standalone reads remain fresh.


@pytest.mark.parametrize(
    "handler",
    [
        SimpleNamespace(profile=Reference(id="another-profile", version="1"), publish_and_read=lambda *_args: None),
        SimpleNamespace(
            profile=Reference(id="coleman-coordinate-publication-readback", version="2"),
            publish_and_read=lambda *_args: None,
        ),
        SimpleNamespace(profile=None, publish_and_read=lambda *_args: None),
        SimpleNamespace(
            profile=Reference.model_construct(id="invalid", version=True), publish_and_read=lambda *_args: None
        ),
        SimpleNamespace(profile=Reference(id="coleman-coordinate-publication-readback", version="1")),
        SimpleNamespace(
            profile=Reference(id="coleman-coordinate-publication-readback", version="1"), publish_and_read=1
        ),
    ],
    ids=["different-profile", "future-version", "missing-profile", "unchecked-profile", "missing-call", "not-callable"],
)
def test_bad_optional_profile_refuses_at_preparation_and_before_publication(handler, tmp_path):
    selected = _Legacy()
    cast(Any, selected).publication_readback = handler
    plan, context = _context(tmp_path)
    cast(Any, selected).advance = lambda *_args: None
    cast(Any, selected).boundary = SimpleNamespace(produce=lambda *_args: None)
    cast(Any, selected).progress_verifier = SimpleNamespace(validate=lambda *_args: None, reconcile=lambda *_args: None)
    with pytest.raises((ValueError, TypeError)):
        _require_prepared(selected, plan)
    value = _Result(schema_version=1, value=0.0)
    with pytest.raises((ValueError, TypeError)):
        _wrapped(selected).publish_coordinate("coordinate", value, context)
    assert selected.calls == []
    assert not context.artifact_directory.exists()


@pytest.mark.parametrize("damage", ["missing", "future", "nonfinite", "unit", "signed-zero", "different", "none"])
def test_combined_readback_must_preserve_complete_concrete_result(damage, tmp_path):
    selected = _Combined()
    _, context = _context(tmp_path)
    value = _Result(schema_version=1, value=0.0)
    original = selected.publish_and_read

    def damaged(coordinate, actual, selected_context):
        readback = original(coordinate, actual, selected_context)
        if damage == "missing":
            (selected_context.artifact_directory / "coordinate.json").unlink()
            return selected._read(selected_context)
        if damage == "none":
            return None
        changes = {
            "future": {"schema_version": 2},
            "nonfinite": {"value": float("inf")},
            "unit": {"unit": "milliseconds"},
            "signed-zero": {"value": -0.0},
            "different": {"value": 1.0},
        }
        return readback.model_copy(update=changes[damage])

    cast(Any, selected).publish_and_read = damaged
    with pytest.raises((ValueError, FileNotFoundError)):
        _wrapped(selected).publish_coordinate("coordinate", value, context)
    assert [row[0] for row in selected.calls] == ["combined"]
    assert not (context.preparation.job_root / "complete.json").exists()


def test_declared_handler_exception_preserves_written_artifact_and_never_falls_back(tmp_path):
    selected = _Combined()
    _, context = _context(tmp_path)
    value = _Result(schema_version=1, value=0.0)
    original = selected.publish_and_read

    def lost_response(coordinate, actual, selected_context):
        original(coordinate, actual, selected_context)
        raise RuntimeError("actual immutable publication response gap")

    cast(Any, selected).publish_and_read = lost_response
    with pytest.raises(RuntimeError, match="response gap"):
        _wrapped(selected).publish_coordinate("coordinate", value, context)
    assert (context.artifact_directory / "coordinate.json").read_text() == value.to_json()
    assert [row[0] for row in selected.calls] == ["combined"]
    assert not (context.preparation.job_root / "complete.json").exists()


def test_profile_is_rechecked_before_physical_publication(tmp_path):
    selected = _Combined()
    assert _publication_readback(selected) is selected
    selected.profile = Reference(id="coleman-coordinate-publication-readback", version="2")
    _, context = _context(tmp_path)
    value = _Result(schema_version=1, value=0.0)
    with pytest.raises(ValueError, match="profile"):
        _wrapped(selected).publish_coordinate("coordinate", value, context)
    assert selected.calls == []
    assert not context.artifact_directory.exists()
