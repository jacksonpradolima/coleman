"""Workflow RNG compatibility through the existing native checkpoint extra-state extension."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from coleman.checkpoint.checkpoint_store import LocalCheckpointStore
from coleman.checkpoint.state import CheckpointPayload
from coleman.corpus.materialization import validate_source_storage
from coleman.policy import base as policy_base
from coleman.results.domain import Fingerprint, decode_json
from coleman.workflows.contracts import WorkflowContract, WorkflowError

UInt128 = Annotated[int, Field(strict=True, ge=0, lt=2**128)]
UInt32 = Annotated[int, Field(strict=True, ge=0, lt=2**32)]
_KEY = "coleman_workflow_rng_v1"


class WorkflowRandomState(WorkflowContract):
    """Complete native global PCG64 state bound to one frozen workflow/trial context.

    Parameters
    ----------
    binding : str
        Complete plan/trial compatibility fingerprint, including data and runtime.
    algorithm : str
        Explicit NumPy PCG64 contract. Another generator requires a new reviewed codec.
    state, increment : int
        Unsigned 128-bit generator state/stream integers, not seeds or observations.
    has_uint32, uinteger : int
        Cached unsigned 32-bit state, required for faithful shuffle continuation.

    Notes
    -----
    Stored as canonical versioned JSON in native CheckpointPayload.extra, alongside
    the existing agents/bandit. No native payload/progress format is changed. Old
    checkpoints lacking this workflow binding reject rather than silently reseed.
    """

    binding: Fingerprint
    algorithm: Literal["PCG64"] = "PCG64"
    state: UInt128
    increment: UInt128
    has_uint32: Annotated[int, Field(strict=True, ge=0, le=1)]
    uinteger: UInt32

    @model_validator(mode="after")
    def _stream(self):
        """Require the odd PCG stream increment instead of silently accepting another recurrence."""
        if self.increment % 2 != 1:
            raise WorkflowError("PCG64 checkpoint requires an odd stream increment")
        return self


class WorkflowCheckpointStore(LocalCheckpointStore):
    """Extend actual local checkpoint saves/loads with compatible global policy RNG state.

    Parameters
    ----------
    base_dir : str
        Stable trusted job-owned native checkpoint root. Imported evidence can
        never supply executable checkpoint files or this path.
    binding : str
        Exact frozen workflow/trial fingerprint, verified before restoring RNG state.
    minimum_step : int
        Already acknowledged per-trial native build ordinal, zero through 512.
        Older or absent state rejects before pickle loading, RNG restoration or
        native output reconciliation. Zero permits a genuinely fresh trial.

    Notes
    -----
    Inherits the existing native checkpoint publication/recovery semantics. Only
    trusted host-owned pickle state is loaded; portable exports exclude all of it.
    Moving a native checkpoint to another root is unsupported and rejects its
    absolute progress-path binding. Native checkpoints outside workflows are unchanged.
    Uses the native process-global generator: callers must run in an isolated
    single native worker, as the workflow coordinator does, never concurrent threads.
    """

    def __init__(self, base_dir: str, binding: str, *, minimum_step: int = 0):
        """Configure the existing native store and its required workflow compatibility binding."""
        if type(minimum_step) is not int or not 0 <= minimum_step <= 512:
            raise ValueError("Acknowledged checkpoint frontier must be an integer from zero through 512")
        super().__init__(base_dir)
        self.binding = binding
        self.minimum_step = minimum_step

    def save(self, payload: CheckpointPayload) -> None:
        """Save native state and the actual global generator together through its existing extra field.

        Parameters
        ----------
        payload : CheckpointPayload
            Native agents/bandit/frontier after the actual committed build. The
            caller's payload object is not mutated. I/O/native errors propagate.
        """
        raw = policy_base._rng.bit_generator.state
        state = WorkflowRandomState(
            schema_version=1,
            binding=self.binding,
            algorithm=raw["bit_generator"],
            state=raw["state"]["state"],
            increment=raw["state"]["inc"],
            has_uint32=raw["has_uint32"],
            uinteger=raw["uinteger"],
        )
        super().save(replace(payload, extra={**payload.extra, _KEY: state.to_json()}))

    def load(self, run_id: str, experiment: int) -> CheckpointPayload | None:
        """Restore existing native state and compatible PCG64 state without replaying earlier outcomes.

        Parameters
        ----------
        run_id : str
            Owned native scenario ID, fixed by the registered dataset.
        experiment : int
            Native execution ordinal; this integration admits one execution per seed.

        Returns
        -------
        CheckpointPayload or None
            Actual native checkpoint; None only before any saved progress exists.

        Raises
        ------
        ValueError, OSError
            Missing/moved/corrupt or incompatible data/runtime/RNG state. No fallback
            silently resets learning or reinterprets an ordinary legacy checkpoint.
        """
        marker = Path(self.base_dir) / run_id / f"progress_ex{experiment}.json"
        if not marker.exists():
            if self.minimum_step:
                raise WorkflowError("Previously acknowledged native checkpoint is missing")
            return None
        validate_source_storage(marker)
        with marker.open("rb") as stream:
            raw = stream.read(16385)
        if len(raw) > 16384:
            raise WorkflowError("Native checkpoint progress exceeds 16 KiB")
        progress = decode_json(raw)
        step = progress.get("step_committed")
        if type(step) is not int or not 1 <= step <= 512:
            raise WorkflowError("Native checkpoint frontier is incompatible with this workflow")
        if step < self.minimum_step:
            raise WorkflowError("Native checkpoint is older than its acknowledged workflow frontier")
        expected = marker.parent / f"ckpt_ex{experiment}_step{step}.pkl"
        if (
            progress.get("checkpoint_path") != str(expected)
            or progress.get("experiment") != experiment
            or type(progress.get("experiment")) is not int
            or progress.get("run_id") != run_id
        ):
            raise WorkflowError("Native checkpoint is not owned by this exact workflow root/execution")
        validate_source_storage(expected)
        payload = _validated_payload(super().load(run_id, experiment), run_id, experiment, step)
        state = WorkflowRandomState.from_json(payload.extra[_KEY])
        if state.binding != self.binding:
            raise WorkflowError("Native checkpoint workflow/data/configuration/runtime binding is incompatible")
        policy_base._rng.bit_generator.state = {
            "bit_generator": state.algorithm,
            "state": {"state": state.state, "inc": state.increment},
            "has_uint32": state.has_uint32,
            "uinteger": state.uinteger,
        }
        return payload


def _validated_payload(payload, run_id: str, experiment: int, step: int) -> CheckpointPayload:
    """Require owned identity/frontier and RNG metadata before any global generator mutation."""
    if payload is None:
        raise WorkflowError("Native checkpoint lacks compatible workflow RNG state; implicit reseeding is forbidden")
    if (
        not isinstance(payload, CheckpointPayload)
        or type(payload.experiment) is not int
        or type(payload.step) is not int
        or (payload.run_id, payload.experiment, payload.step) != (run_id, experiment, step)
    ):
        raise WorkflowError("Native checkpoint payload identity/frontier differs from its progress marker")
    if not isinstance(payload.extra, dict) or not isinstance(payload.extra.get(_KEY), str):
        raise WorkflowError("Native checkpoint lacks compatible workflow RNG state; implicit reseeding is forbidden")
    return payload
