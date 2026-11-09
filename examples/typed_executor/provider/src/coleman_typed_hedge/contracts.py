"""Complete bounded fixture values; declarations import no local state store."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from coleman.core._wire import BoundedContract
from coleman.core.feedback import FullPanel
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint
from coleman.learning._math import normalize_positive_prior
from coleman.learning.contracts import ReferenceConfig, ReferencePayload, read_loss_panel
from coleman.objectives.bindings import DefinitionBinding, SemanticDefinition
from coleman.state.contracts import AlgorithmState, StateBinding
from coleman.tasks.contracts import OwnedCommitProof


class FixtureValue(BoundedContract):
    """Use this pinned SDK's bounded graph reader, with a separate namespace.

    Notes
    -----
    Version one is required. Every complete value fits two MiB/depth64, and
    selected codecs apply their own smaller caps before parsing or exporting.
    Unknown fields, nonfinite numbers and opaque/cyclic Python graphs refuse.
    """

    contract: Literal["coleman-typed-hedge-reference-v1"]
    _maximum_bytes = 2 * 1024**2

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Bound mutable input before delegating to the pinned strict reader.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete UTF-8 JSON, at most the concrete fixture profile capacity.
        **kwargs : object
            Existing strict core validation options; relaxation still refuses.

        Returns
        -------
        Self
            Complete concrete v1 artifact, with no implicit migration or I/O.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid/excessive data, unknown schema or forbidden reader options.
        """
        if type(json_data) in (bytearray,) and len(json_data) > cls._maximum_bytes:
            raise ValueError("Fixture bytearray exceeds its capacity before copying")
        return super().model_validate_json(json_data, **kwargs)


class HedgeConfiguration(FixtureValue):
    """Keep the complete constant-rate law and finite publication batch size.

    Attributes
    ----------
    reference : ReferenceConfig
        Shipped constant Hedge, at most eight fixed experts, loss units one.
    panels_per_commit : int
        One or two actual accepted panels per publication, not a learning rate.
    """

    reference: ReferenceConfig
    panels_per_commit: Annotated[int, Field(strict=True, ge=1, le=2)] = 2

    @model_validator(mode="after")
    def _reference(self) -> Self:
        """Refuse other algorithms rather than interpreting their parameters."""
        if self.reference.profile != "hedge-constant" or len(self.reference.prior.weights) > 8:
            raise ValueError("Reference executor requires constant Hedge with at most eight experts")
        return self


class HedgeTrajectory(FixtureValue):
    """Retain a virgin mathematical snapshot and the exact original panels.

    Attributes
    ----------
    reference : Reference
        Stable independent learner owner, unrelated to a storage generation.
    initial_state : AlgorithmState[ReferencePayload]
        Complete actual shipped Hedge state at U0, including its owner binding.
    panels : tuple of FullPanel
        One through eight ordered fully observed normalized-loss panels. Source
        labels and transforms survive; unavailable/null values never mean zero.
    """

    reference: Reference
    initial_state: AlgorithmState[ReferencePayload]
    panels: Annotated[tuple[FullPanel, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def _trajectory(self) -> Self:
        """Require one exact virgin owner and every bounded observed panel."""
        payload = self.initial_state.payload
        if self.initial_state.binding.owner != self.reference or payload.update_count != 0:
            raise ValueError("Hedge trajectory requires its exact virgin owner at U0")
        if payload.configuration.profile != "hedge-constant" or len(payload.configuration.prior.weights) > 8:
            raise ValueError("Hedge trajectory requires the constant reference profile")
        if payload.weights != normalize_positive_prior(payload.configuration.prior.weights) or any(
            value.as_integer_ratio()[0] != 0 for value in (*payload.cumulative_losses, *payload.compensations)
        ):
            raise ValueError("Hedge trajectory initial state must contain the unmodified prior")
        config = payload.configuration
        if len(self.panels) > config.limits.maximum_updates:
            raise ValueError("Hedge original trajectory exceeds its fixed accepted-panel capacity")
        for panel in self.panels:
            read_loss_panel(
                panel,
                members=tuple(item.member for item in config.prior.weights),
                loss_spec=config.loss_spec,
                transform=config.transform,
            )
        return self


class HedgeInputs(SemanticDefinition, FixtureValue):
    """Bind the finite complete constructed source independently of config.

    Attributes
    ----------
    reference, lineage : Reference, DefinitionLineage
        Explicit version and declared source facts, including unknown provenance.
    trajectories : tuple of HedgeTrajectory
        One through eight exact uniquely owned original trajectories, in declared
        execution order. The full content contributes to the source binding.
    """

    trajectories: Annotated[tuple[HedgeTrajectory, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        """Preserve declared order while refusing duplicate logical owners."""
        ids = [item.reference.id for item in self.trajectories]
        if len(ids) != len(set(ids)):
            raise ValueError("Hedge source requires unique independent learner owners")
        return self


class HedgeCheckpoint(FixtureValue):
    """Wrap unchanged inner math in exact surrounding continuation bindings.

    Attributes
    ----------
    binding : StateBinding
        Actual captured executor implementation and complete execution config.
    plan_fingerprint, coordinate_fingerprint : str
        Full immutable task identities, never local handles or filesystem paths.
    source : DefinitionBinding
        Complete exact finite input declaration.
    learner : AlgorithmState[ReferencePayload]
        Full shipped learner state; the direct/durable comparison uses its bytes.
    """

    binding: StateBinding
    plan_fingerprint: Fingerprint
    coordinate_fingerprint: Fingerprint
    source: DefinitionBinding
    learner: AlgorithmState[ReferencePayload]

    @model_validator(mode="after")
    def _owner(self) -> Self:
        """Prevent an outer envelope from relabelling the mathematical owner."""
        if self.binding.owner != self.learner.binding.owner:
            raise ValueError("Outer checkpoint and complete learner must retain the same owner")
        return self


class HedgeCoordinateResult(FixtureValue):
    """Keep complete math plus an actual rechecked current ACK declaration.

    Attributes
    ----------
    checkpoint : HedgeCheckpoint
        Full actual mathematical state and surrounding execution bindings.
    proof : OwnedCommitProof
        Exact command, actual receipt/current head and state-derived frontier.
        Reading this model checks joins; the prepared verifier checks storage.
    """

    checkpoint: HedgeCheckpoint
    proof: OwnedCommitProof

    @model_validator(mode="after")
    def _joins(self) -> Self:
        """Join actual logical work to inner U, not publication generation."""
        state, proof = self.checkpoint, self.proof
        if (
            proof.plan_fingerprint != state.plan_fingerprint
            or proof.coordinate_fingerprint != state.coordinate_fingerprint
            or proof.frontier.coordinate != state.binding.owner
            or proof.frontier.position != state.learner.payload.update_count
        ):
            raise ValueError("Hedge result must join complete state and its exact logical ACK frontier")
        return self


class HedgeReport(FixtureValue):
    """Report constructed mechanism checks without inference or adoption.

    Attributes
    ----------
    interpretation : {'constructed_mechanism_only'}
        Explicit scope, supplying no empirical source admission or promotion.
    results : tuple of HedgeCoordinateResult
        Complete unique coordinate results from the same immutable task plan.
    """

    interpretation: Literal["constructed_mechanism_only"]
    results: Annotated[tuple[HedgeCoordinateResult, ...], Field(min_length=1, max_length=8)]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Require unique coordinates under one immutable common plan."""
        ids = [item.checkpoint.coordinate_fingerprint for item in self.results]
        if len(ids) != len(set(ids)) or len({item.checkpoint.plan_fingerprint for item in self.results}) != 1:
            raise ValueError("Hedge report requires unique complete coordinates under one plan")
        return self
