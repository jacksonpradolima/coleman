"""Concrete typed optional task boundary for the same synthetic experiment."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.identity import Reference
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.objectives.bindings import DefinitionBinding, SemanticDefinition
from coleman.problems.portfolio.contracts import PortfolioContract
from coleman.state.contracts import StateBinding
from coleman.tasks.contracts import OwnedCommitProof

from . import model
from .experiment import ExperimentState, _implementation


class PortfolioConfiguration(PortfolioContract):
    """Freeze development-selected coefficient strength, not a paid cost."""

    strength: Annotated[Binary64, Field(ge=0, le=100)]
    periods_per_commit: Literal[8] = 8


class PortfolioTrajectory(PortfolioContract):
    """Retain one authored exogenous source and its exact virgin continuation."""

    reference: Reference
    initial_state: ExperimentState
    rows: Annotated[
        tuple[tuple[Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]], ...],
        Field(min_length=16, max_length=16),
    ]
    periods: tuple[Annotated[int, Field(strict=True, ge=0, le=15)], ...] = tuple(range(16))

    @model_validator(mode="after")
    def _virgin(self) -> Self:
        """Validate virgin typed data without constructing any executable owner."""
        if self.periods != tuple(range(16)):
            raise ValueError("Portfolio task requires every original period in order")
        initial = self.initial_state
        if initial.next_index != 0 or initial.implementation != _implementation():
            raise ValueError("Portfolio trajectory requires its exact virgin public host")
        if initial.tape_fingerprint != fingerprint([list(row) for row in self.rows]):
            raise ValueError("Portfolio trajectory initial source differs from its exogenous tape")
        if initial.ridge.payload.config != model.ridge_config() or initial.ridge.payload.sample_count != 0:
            raise ValueError("Portfolio trajectory requires an untrained configured predictor")
        if any(item.state.to_json() != model.initial_book(item.name).to_json() for item in initial.books):
            raise ValueError("Portfolio trajectory requires every independently funded virgin book")
        _virgin_references(initial)
        _virgin_controls(initial)
        return self


def _virgin_references(initial: ExperimentState) -> None:
    """Check original full-information law and zero counters using DTOs only."""
    if tuple(item.name for item in initial.references) != ("fixed-share", "hedge"):
        raise ValueError("Portfolio trajectory lost an exact reference learner")
    for item in initial.references:
        profile = "hedge-constant" if item.name == "hedge" else "fixed-share-excluding-origin"
        config = model.reference_config(profile)
        payload = item.state.payload
        if payload.configuration != config or payload.update_count != 0 or payload.weights != config.prior.weights:
            raise ValueError("Portfolio trajectory reference state is not its configured virgin prior")
        if item.state.binding.owner != model.ref(item.name + "-owner"):
            raise ValueError("Portfolio trajectory changed its original reference owner")


def _virgin_controls(initial: ExperimentState) -> None:
    """Check original control laws/frontiers without invoking a learner factory."""
    if tuple(item.name for item in initial.controlled) != tuple(sorted(model.CONTROLLED_MODES)):
        raise ValueError("Portfolio trajectory lost a controlled learner")
    for item in initial.controlled:
        payload = item.state.payload
        config = model.controlled_config(item.name, strength=initial.strength)
        if payload.config != config or payload.update_count != 0 or payload.release_count != 0:
            raise ValueError("Portfolio trajectory controls are not their configured virgin laws")
        if item.state.binding.owner != model.ref(item.name + "-owner"):
            raise ValueError("Portfolio trajectory changed its original controlled owner")


class PortfolioInputs(SemanticDefinition, PortfolioContract):
    """Bind exactly one original authored tape and all independently funded books."""

    trajectories: Annotated[tuple[PortfolioTrajectory, ...], Field(min_length=1, max_length=1)]


class PortfolioCheckpoint(PortfolioContract):
    """Wrap complete real composition state in original task/image bindings."""

    binding: StateBinding
    plan_fingerprint: Fingerprint
    coordinate_fingerprint: Fingerprint
    source: DefinitionBinding
    rows: tuple[tuple[int, int], ...]
    experiment: ExperimentState

    @model_validator(mode="after")
    def _source(self) -> Self:
        """Reject a state silently continued on changed exogenous prices."""
        if self.experiment.tape_fingerprint != fingerprint([list(row) for row in self.rows]):
            raise ValueError("Portfolio checkpoint tape and complete state disagree")
        return self


class PortfolioCoordinateResult(PortfolioContract):
    """Retain exact complete state and a real current durable-ACK proof."""

    checkpoint: PortfolioCheckpoint
    proof: OwnedCommitProof

    @model_validator(mode="after")
    def _joins(self) -> Self:
        """Keep work measured by original accepted periods, never publications."""
        state, proof = self.checkpoint, self.proof
        if (
            proof.plan_fingerprint,
            proof.coordinate_fingerprint,
            proof.frontier.coordinate,
            proof.frontier.position,
        ) != (state.plan_fingerprint, state.coordinate_fingerprint, state.binding.owner, state.experiment.next_index):
            raise ValueError("Portfolio result differs from its complete original ACK frontier")
        return self


class PortfolioReport(PortfolioContract):
    """Publish constructed descriptive mechanics without financial inference."""

    interpretation: Literal["constructed_mechanism_only"]
    results: Annotated[tuple[PortfolioCoordinateResult, ...], Field(min_length=1, max_length=1)]
