"""Typed workflow context inside the existing generic frozen-candidate contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from coleman.results.domain import Fingerprint
from coleman.selection.contracts import Immutable
from coleman.workflows.contracts import Seed
from coleman.workflows.lifecycle import FilePin


class FrozenLearningState(Immutable):
    """One seed's actual immutable native prefix state, without loading or exporting pickles.

    Parameters
    ----------
    trial_id : str
        Saved logical policy/configuration/seed coordinate.
    seed : int
        Explicit algorithmic repetition; never an inference unit.
    checkpoints : tuple[FilePin, ...]
        Complete bounded immutable prefix-state membership, including native RNG
        metadata. Paths are safe private relative names and are not tool arguments.
    """

    trial_id: Fingerprint
    seed: Seed
    checkpoints: Annotated[tuple[FilePin, ...], Field(min_length=1, max_length=4096)]


class FrozenWorkflowConfiguration(Immutable):
    """Strict workflow projection inside an existing FrozenCandidate's opaque JSON boundary.

    Parameters
    ----------
    configuration_id, workflow_plan : str
        Complete native coordinate and full data/runtime/resource/protocol plan binding.
    source_scope : str
        Stable upstream evidence scope; re-encoding/projection cannot reset exposure.
    learning_state : tuple[FrozenLearningState, ...]
        Every declared selected seed's actual native prefix state.
    corpus_repetition_design : str
        Existing whole build/seed/family/clock aggregation design, frozen before access.
    preprocessing : str
        Explicit identity preprocessing; no model, fitter or future feature is inferred.

    Notes
    -----
    Pure frozen validation, with no file reads, native execution or transport imports.
    Unknown fields reject; full generic evaluation contracts remain unchanged.
    """

    configuration_id: Fingerprint
    workflow_plan: Fingerprint
    source_scope: Fingerprint
    learning_state: Annotated[tuple[FrozenLearningState, ...], Field(min_length=1, max_length=64)]
    corpus_repetition_design: Fingerprint
    preprocessing: Literal["identity"] = "identity"
