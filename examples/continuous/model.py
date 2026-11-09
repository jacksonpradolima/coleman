"""Public synthetic forecast host using the existing causal replay boundary.

The mathematical predictor remains usable without this example. This bounded
host owns original features, forecasts and controls until eligible labels arrive;
the existing replay owns delivery deduplication and preserve-live corrections.
"""

from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.core.actions import VectorAction, VectorSpace
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.capabilities import Capability
from coleman.core.decisions import Decision, DecisionRequest
from coleman.core.feedback import AggregateFeedback
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, Scalar, ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.core.units import UnitSemantics
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.prediction.contracts import RidgeConfig, RidgeInput, RidgePayload, RidgePrediction, RidgeSample
from coleman.prediction.ridge import AppendOnlyRidge
from coleman.replay.contracts import ReplayApplication
from coleman.state.contracts import AlgorithmState, StateBinding


def reference(name: str) -> Reference:
    """Declare one public local identity.

    Parameters
    ----------
    name : str
        Explicit bounded logical name; no lookup occurs.

    Returns
    -------
    Reference
        Immutable version-one interpretation.
    """
    return Reference(id=name, version="1")


HORIZON = reference("one-synthetic-step")
OBJECTIVE = reference("synthetic-signed-response")
SOURCE = reference("constructed-continuous-inputs")
CODEC = reference("constructed-forecast-host-state")


def configuration() -> RidgeConfig:
    """Freeze the two-feature, no-intercept append-only ridge reference.

    Returns
    -------
    RidgeConfig
        Identity dimensionless preprocessing, lambda1, all coefficients
        penalized around zero, one-step signed dimensionless target.
    """
    return RidgeConfig(
        schema_version=1,
        reference=reference("constructed-ridge-law"),
        lineage=DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
        features=tuple(
            ScalarSpec(
                dimension=reference(f"feature-{index}"),
                role="feature",
                semantics=UnitSemantics(unit="1", meaning=f"constructed-input-{index}-v1"),
                minimum=-4.0,
                maximum=4.0,
            )
            for index in range(2)
        ),
        target=ScalarSpec(
            dimension=OBJECTIVE,
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="constructed-signed-response-v1"),
            minimum=-4.0,
            maximum=4.0,
        ),
        horizon=HORIZON,
        preprocessing=DefinitionBinding(
            reference=reference("identity-dimensionless-preprocessing"),
            definition_fingerprint="sha256:" + sha256(b"identity-dimensionless-v1").hexdigest(),
        ),
        regularization=1.0,
    )


def implementation() -> DefinitionBinding:
    """Capture this explicitly selected public host source, outside mathematics.

    Returns
    -------
    DefinitionBinding
        Actual source-byte identity; integrity does not authenticate execution.
    """
    return DefinitionBinding(
        reference=reference("constructed-forecast-host"),
        definition_fingerprint="sha256:" + sha256(Path(__file__).read_bytes()).hexdigest(),
    )


class PendingForecast(CoreImmutable):
    """Retain original producer features and pre-label controls exactly once."""

    request: DecisionRequest
    input: RidgeInput
    prediction: RidgePrediction
    mean_control: Binary64
    last_control: Binary64

    @model_validator(mode="after")
    def _original(self) -> Self:
        """Bind the emitted prediction to the original complete feature input."""
        if self.prediction.input.to_json() != self.input.to_json():
            raise ValueError("Pending prediction differs from its original feature values")
        if self.prediction.failure is not None:
            raise ValueError("The public host cannot retain a numerically unavailable emitted forecast")
        return self


class ForecastObservation(CoreImmutable):
    """Record one eligible signed label separately from forecasting errors."""

    pending: PendingForecast
    target: Scalar


class ForecastHostState(CoreContract):
    """Own complete predictor statistics, pending lineage and raw error trace.

    Notes
    -----
    Maximum128 opportunities. No seed, optimizer, execution fees or externally
    paid work is inferred. A reset requires a fresh model/session lineage.
    """

    binding: StateBinding
    ridge: AlgorithmState[RidgePayload]
    decisions: Annotated[int, Field(strict=True, ge=0, le=128)]
    pending: Annotated[tuple[PendingForecast, ...], Field(max_length=128)]
    observations: Annotated[tuple[ForecastObservation, ...], Field(max_length=128)]

    @model_validator(mode="after")
    def _counts(self) -> Self:
        """Reject lost or repeated original producer state before restore."""
        rows = self.pending + tuple(row.pending for row in self.observations)
        identities = tuple(row.request.logical_identity for row in rows)
        if self.decisions != len(rows) or len(set(identities)) != len(rows):
            raise ValueError("Complete forecast state requires every unique original producer")
        if any(row.pending.prediction.sample_count > index for index, row in enumerate(self.observations)):
            raise ValueError("Forecast cannot use its own or a future accepted label")
        return self


class ForecastHost:
    """Compose one real predictor with the existing optional replay protocol.

    Parameters
    ----------
    child : int, optional
        Independent ownership coordinate, zero through31; default0. Coordinates
        do not denote independent observations or pooled statistical learning.

    Notes
    -----
    Construction captures this public source for outer compatibility only.
    Direct mathematical append has no event deduplication. Use CausalReplay
    for original-producer joins, delayed eligibility and duplicate acceptance.
    """

    def __init__(self, child: int = 0):
        """Create fresh local state; perform no external execution."""
        # Exact built-in integers exclude bool and callback-bearing subclasses.
        if type(child) is not int or not 0 <= child < 32:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Forecast child must be an integer from zero through31")
        self.child = child
        self.config = configuration()
        self.task = reference(f"constructed-forecast-task-{child:02}")
        self.ridge = AppendOnlyRidge(self.config, owner=reference(f"constructed-ridge-{child:02}"))
        self.binding = StateBinding(
            owner=self.task,
            implementation=implementation(),
            state_codec=CODEC,
            configuration_fingerprint=self.config.fingerprint,
        )
        self._pending: tuple[PendingForecast, ...] = ()
        self._observations: tuple[ForecastObservation, ...] = ()

    @property
    def space(self) -> VectorSpace:
        """Return a signed forecast score vector, never a probability or holding."""
        return VectorSpace(
            feasible_set=reference("constructed-forecast-score-space"),
            signed=True,
            coordinates=(
                ScalarSpec(
                    dimension=reference("predicted-score"),
                    role="action",
                    semantics=self.config.target.semantics,
                ),
            ),
        )

    @property
    def capability(self) -> Capability:
        """Declare eligible aggregate supervised labels and required features."""
        return Capability(
            schema_version=1,
            implementation=self.binding.implementation.reference,
            action_kinds=("vector",),
            feedback_kinds=("aggregate",),
            required_values=(self.config.target,),
            required_inputs=self.config.features,
            origins=("synthetic",),
            timing_modes=("future",),
            supports_delayed_feedback=True,
            requires_known_receipt=True,
            requires_observed=True,
            state_owner="host",
            requires_sampling=False,
            sampling_requirement="not_required",
        )

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Predict using only captured eligible features before their label.

        Parameters
        ----------
        request : DecisionRequest
            Original timed opportunity with exact feature and score schemas.

        Returns
        -------
        Decision
            Actual signed prediction vector with no probability interpretation.

        Raises
        ------
        ValueError
            Unsupported task, schema, missing feature, duplicate producer or the
            bounded128-opportunity capacity. Rejected calls change no state.
        """
        checked = DecisionRequest.model_validate(request)
        if checked.task != self.task or checked.objective != OBJECTIVE or checked.space != self.space:
            raise ValueError("Forecast request differs from the fixed task/objective/space")
        values = {row.spec.dimension: row for row in checked.inputs}
        if set(values) != {spec.dimension for spec in self.config.features}:
            raise ValueError("Forecast requires exactly its declared complete feature schema")
        selected = tuple(values[spec.dimension] for spec in self.config.features)
        if any(
            row.spec != spec or row.status != "observed"
            for row, spec in zip(selected, self.config.features, strict=True)
        ):
            raise ValueError("Forecast requires observed features with exact units and semantics")
        previous = self._pending + tuple(row.pending for row in self._observations)
        if len(previous) >= 128 or any(row.request.logical_identity == checked.logical_identity for row in previous):
            raise ValueError("Forecast capacity or original logical producer already consumed")
        model_input = RidgeInput(
            schema_version=1,
            config=DefinitionBinding(reference=self.config.reference, definition_fingerprint=self.config.fingerprint),
            values=tuple(row.value for row in selected),
        )
        prediction = self.ridge.predict(model_input)
        if prediction.failure is not None or prediction.value.value is None:
            raise ValueError("Ridge prediction is numerically unavailable")
        targets = tuple(row.target.value for row in self._observations)
        pending = PendingForecast(
            request=checked,
            input=model_input,
            prediction=prediction,
            mean_control=math.fsum(targets) / len(targets) if targets else 0.0,
            last_control=targets[-1] if targets else 0.0,
        )
        action = VectorAction(
            schema_version=1,
            space=self.space,
            values=(Scalar(spec=self.space.coordinates[0], status="observed", value=prediction.value.value),),
        )
        result = Decision(
            schema_version=1,
            request=checked,
            implementation=self.capability.implementation,
            action=action,
            state_owner="host",
            sampling=SamplingRecord(
                mechanism="deterministic",
                mechanism_ref=self.capability.implementation,
                status="not_required",
                action_fingerprint=action.fingerprint,
                probability=None,
            ),
        )
        self._pending += (pending,)
        return result

    def update(self, application: ReplayApplication, /) -> object:
        """Append exactly the eligible original feature/label pair selected by replay.

        Parameters
        ----------
        application : ReplayApplication
            Exact producer, selected revision and independent eligibility/apply
            clocks. The enclosing replay has already validated this delivery.

        Returns
        -------
        object
            Actual predictor's typed accepted update.

        Raises
        ------
        ValueError
            The original producer, target schema or observed label is absent,
            or the numerical predictor refuses the append. No target is imputed.
        """
        checked = ReplayApplication.model_validate(application)
        request = checked.executed.proposal.request
        pending = next((row for row in self._pending if row.request.fingerprint == request.fingerprint), None)
        payload = checked.feedback.batch.feedback
        if pending is None or not isinstance(payload, AggregateFeedback) or len(payload.values) != 1:
            raise ValueError("Forecast update requires its original pending producer and one aggregate label")
        outcome = payload.values[0]
        target = Scalar(spec=outcome.spec, status=outcome.status, value=outcome.value)
        result = self.ridge.append(RidgeSample(schema_version=1, input=pending.input, target=target))
        if result.status != "accepted":
            raise ValueError("Ridge append refused the numerical update")
        self._observations += (ForecastObservation(pending=pending, target=target),)
        self._pending = tuple(row for row in self._pending if row is not pending)
        return result

    def export_state(self) -> ForecastHostState:
        """Export complete real state without I/O or pending-label duplication.

        Returns
        -------
        ForecastHostState
            Frozen predictor, pending original features and accepted label trace.
        """
        return ForecastHostState(
            schema_version=1,
            binding=self.binding,
            ridge=self.ridge.export_state(),
            decisions=len(self._pending) + len(self._observations),
            pending=self._pending,
            observations=self._observations,
        )

    def restore_state(self, state: ForecastHostState, /) -> None:
        """Restore a compatible complete host through the real predictor codec.

        Parameters
        ----------
        state : ForecastHostState
            Exact owner/source/configuration generation and complete state.

        Raises
        ------
        ValueError
            Invalid source, owner, configuration, feature versions or predictor
            statistics. No old checkpoint or preprocessing is silently migrated.
        """
        checked = ForecastHostState.model_validate(state)
        if checked.binding != self.binding:
            raise ValueError("Forecast host source/owner/configuration binding differs")
        if checked.ridge.payload.update_count != len(checked.observations):
            raise ValueError("Forecast label trace and actual predictor update count differ")
        for row in checked.pending + tuple(item.pending for item in checked.observations):
            if row.request.task != self.task or row.input.config.definition_fingerprint != self.config.fingerprint:
                raise ValueError("Forecast pending lineage differs from its task or feature/preprocessing generation")
            expected = tuple((spec, value) for spec, value in zip(self.config.features, row.input.values, strict=True))
            actual = tuple((value.spec, value.value) for value in row.request.inputs if value.status == "observed")
            if actual != expected or row.request.space != self.space or row.request.objective != OBJECTIVE:
                raise ValueError("Forecast original request differs from its pending feature/schema/score content")
        self._validate_prefixes(checked)
        self.ridge.restore_state(checked.ridge)
        self._pending, self._observations = checked.pending, checked.observations

    def _validate_prefixes(self, state: ForecastHostState) -> None:
        """Recheck bounded raw accepted prefixes before replacing any live state."""
        temporary = AppendOnlyRidge(self.config, owner=self.ridge.binding.owner)
        rows = state.pending + tuple(item.pending for item in state.observations)
        if any(row.prediction.sample_count > len(state.observations) for row in rows):
            raise ValueError("Forecast claims a future accepted-label prefix")
        for count in range(len(state.observations) + 1):
            targets = tuple(row.target.value for row in state.observations[:count])
            mean = math.fsum(targets) / count if count else 0.0
            last = targets[-1] if count else 0.0
            for row in rows:
                if row.prediction.sample_count == count and (
                    temporary.predict(row.input).to_json() != row.prediction.to_json()
                    or row.mean_control != mean
                    or row.last_control != last
                ):
                    raise ValueError("Forecast prediction or control differs from its actually accepted prefix")
            if count < len(state.observations):
                observed = state.observations[count]
                update = temporary.append(
                    RidgeSample(schema_version=1, input=observed.pending.input, target=observed.target)
                )
                if update.status != "accepted":
                    raise ValueError("Forecast trace cannot reproduce its accepted numerical update")
        if temporary.export_state().to_json() != state.ridge.to_json():
            raise ValueError("Forecast sufficient state differs from its complete accepted raw trace")
