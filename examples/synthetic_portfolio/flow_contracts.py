"""Complete typed original producers and continuation for the synthetic example.

All marks are authored exogenous synthetic observations. Virtual recipe books
own their own paths and modeled fees. Only a candidate's netted deployed book
owns that candidate's executed charges; virtual charges are never debited again.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Self

from pydantic import Field, model_validator

from coleman.core.actions import Weight
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, canonical_json
from coleman.learning.contracts import ReferencePayload
from coleman.learning.control_contracts import ControlledPayload, ControlOutcome
from coleman.objectives.bindings import DefinitionBinding
from coleman.optimization.continuous_contracts import ContinuousSolution
from coleman.prediction import RidgePayload
from coleman.problems.portfolio import Book, Execution, PricePanel, Target, mark
from coleman.problems.portfolio.contracts import PortfolioContract
from coleman.state.components import ComponentCodec
from coleman.state.contracts import AlgorithmState, StateBinding

from . import model
from .diagnostics import ControlObjective, _control_objective
from .feedback import Observation, Pending, _original_observation


def _validated_wire(value: PortfolioContract) -> str:
    """Compare complete already-validated children inside an after-validator.

    This private projection is never an input reader or exported serializer.
    Pydantic has fully validated every child before the containing after-validator
    runs, including constructed and copied ingress. Calling public ``to_json``
    here would recursively repeat that validation and every historical scale fit.
    """
    return canonical_json(value.model_dump(mode="json"))


class NamedBook(PortfolioContract):
    """Bind a candidate or virtual-recipe name to its entire funded path."""

    name: str
    state: Book


class NamedReference(PortfolioContract):
    """Retain an actual Hedge/FixedShare mathematical continuation."""

    name: str
    state: AlgorithmState[ReferencePayload]


class NamedControlled(PortfolioContract):
    """Retain every control origin, counter, pending block and exact usage ledger."""

    name: str
    state: AlgorithmState[ControlledPayload]


class CandidateStep(PortfolioContract):
    """Trace coefficients, moving origin, constrained target and own settlement."""

    name: str
    coefficients: tuple[Weight, ...] | None
    coefficient_movement: Annotated[Binary64, Field(ge=0)] | None
    control: ControlOutcome | None
    control_objective: ControlObjective | None
    proposed_target: Target | None
    solver: ContinuousSolution | None
    settlement: Execution

    @model_validator(mode="after")
    def _objective(self) -> Self:
        """Reject diagnostic amounts unrelated to their actual original controller."""
        expected = _control_objective(self.control)
        if self.control_objective != expected:
            raise ValueError("Control objective differs from its actual original emitted point")
        mixture = self.name in (*model.CONTROLLED_MODES, "hedge", "fixed-share")
        if mixture != (self.coefficients is not None) or mixture != (self.coefficient_movement is not None):
            raise ValueError("Coefficient applicability must match the original actual mixture route")
        if self.coefficients is not None and tuple(item.member for item in self.coefficients) != model.RECIPES:
            raise ValueError("Actual coefficient vectors require every original recipe/version")
        return self


class Period(PortfolioContract):
    """Retain one information/decision/output cycle without future labels."""

    index: Annotated[int, Field(strict=True, ge=0, le=127)]
    pending: Pending
    observation: Observation | None
    recipes: Annotated[tuple[Execution, ...], Field(min_length=3, max_length=3)]
    candidates: Annotated[tuple[CandidateStep, ...], Field(min_length=len(model.MODES), max_length=len(model.MODES))]

    @model_validator(mode="after")
    def _identities(self) -> Self:
        """Reject incomplete output catalogues and substituted producer ownership."""
        if tuple(row.name for row in self.candidates) != model.MODES:
            raise ValueError("Synthetic period requires every original candidate in declared order")
        if self.pending.index != self.index:
            raise ValueError("Synthetic period retains another original pending producer")
        for index, receipt in enumerate(self.recipes):
            if receipt.before.reference != model.ref("book-virtual-" + str(index)):
                raise ValueError("Synthetic virtual recipe output belongs to another funded path")
            if receipt.gross_value != self.pending.recipe_opening_values[index]:
                raise ValueError("Synthetic recipe return denominator differs from its original marked book")
            if receipt.request != model.ref("virtual-" + str(index) + "-decision-" + str(self.index)):
                raise ValueError("Synthetic recipe output belongs to another original producer")
        if any(row.settlement.before.reference != model.ref("book-" + row.name) for row in self.candidates):
            raise ValueError("Synthetic candidate output belongs to another funded path")
        if any(
            row.settlement.request != model.ref(row.name + "-decision-" + str(self.index)) for row in self.candidates
        ):
            raise ValueError("Synthetic candidate output belongs to another original producer")
        self._sources()
        return self

    def _sources(self) -> None:
        """Join actual source and controller calls without repeating past outputs."""
        price = self.recipes[0].valuation.prices
        if any(
            receipt.valuation.prices != price
            for receipt in (*self.recipes, *(row.settlement for row in self.candidates))
        ):
            raise ValueError("Synthetic period cannot substitute another exogenous mark across methods")
        if self.observation is not None and self.observation.prices != price:
            raise ValueError("Synthetic period source differs from its actual earlier applied observation")
        if self.observation is not None and self.observation.index != self.index:
            raise ValueError("Synthetic period retains another original observation index")
        receipts = () if self.observation is None else self.observation.controls
        for row in self.candidates:
            actual = next((receipt.outcome for receipt in reversed(receipts) if receipt.name == row.name), None)
            if row.control != actual:
                raise ValueError("Candidate control must be this period's actual call, never a stale last outcome")


class ExperimentState(PortfolioContract):
    """Capture all real component states and original in-flight label lineage.

    Notes
    -----
    This deterministic16-row example owns no RNG. It retains all predictor
    statistics, virtual and deployed books, original forecasts and scales,
    accepted feedback, control pending panels/cadence/period usage and outputs.
    A JSON artifact alone is not a durable ACK; use the existing state boundary
    for publication. Changing any configuration requires a fresh lineage.
    """

    strength: Annotated[Binary64, Field(ge=0, le=100)]
    implementation: DefinitionBinding
    tape_fingerprint: Fingerprint
    next_index: Annotated[int, Field(strict=True, ge=0, le=16)]
    books: tuple[NamedBook, ...]
    references: tuple[NamedReference, ...]
    controlled: tuple[NamedControlled, ...]
    ridge: AlgorithmState[RidgePayload]
    pending: Pending | None
    ready_observation: Observation | None
    observed_prices: Annotated[tuple[PricePanel, ...], Field(max_length=16)]
    observations: Annotated[tuple[Observation, ...], Field(max_length=15)]
    periods: Annotated[tuple[Period, ...], Field(max_length=16)]

    @model_validator(mode="after")
    def _frontier(self) -> Self:
        """Validate complete counters, original producers and funded inventory."""
        self._counts()
        self._originals()
        self._books()
        self._observation_origins()
        self._components()
        self._sources()
        self._coefficient_paths()
        return self

    def _counts(self: ExperimentState | ExperimentStateV2) -> None:
        """Bind contiguous accepted periods and the actual predictor frontier."""
        expected = max(0, self.next_index - 1) + int(self.ready_observation is not None)
        if self.next_index != len(self.periods) or len(self.observations) != expected:
            raise ValueError("Synthetic experiment lost chronological frontier")
        if self.ridge.payload.sample_count != expected:
            raise ValueError("Synthetic experiment lost accepted ridge labels")
        if (self.pending is None) != (self.next_index == 0):
            raise ValueError("Synthetic experiment must retain its original pending producer")
        if self.pending is not None and self.pending.index != self.next_index - 1:
            raise ValueError("Synthetic pending forecast belongs to another period")
        if self.ready_observation is not None and (
            self.next_index == 0 or self.ready_observation.index != self.next_index
        ):
            raise ValueError("Feedback-ready phase must belong to the next original decision")
        if len(self.observed_prices) != self.next_index + int(self.ready_observation is not None):
            raise ValueError("Synthetic source view must retain exactly its eligible price prefix")
        if tuple(item.index for item in self.periods) != tuple(range(self.next_index)):
            raise ValueError("Synthetic experiment period order is not contiguous")
        if any(
            period.pending.index != period.index or period.pending.prediction.sample_count != period.index
            for period in self.periods
        ):
            raise ValueError("Synthetic original prediction and period frontiers disagree")

    def _originals(self) -> None:
        """Reject substituted duplicated forecasts, labels or future initial labels."""
        if self.pending is not None and _validated_wire(self.pending) != _validated_wire(self.periods[-1].pending):
            raise ValueError("Synthetic pending forecast differs from its retained original producer")
        originals = tuple(period.observation for period in self.periods[1:])
        if self.ready_observation is not None:
            originals += (self.ready_observation,)
        if any(original is None for original in originals):
            raise ValueError("Synthetic experiment lost an original period observation")
        if tuple(_validated_wire(row) for row in self.observations) != tuple(
            _validated_wire(row) for row in originals if row is not None
        ):
            raise ValueError("Synthetic observations differ from their retained original period records")
        if self.periods and self.periods[0].observation is not None:
            raise ValueError("Initial synthetic decision cannot contain a future label")
        if tuple(row.index for row in self.observations) != tuple(range(1, len(self.observations) + 1)):
            raise ValueError("Synthetic observations must retain their original contiguous calendar")

    def _books(self: ExperimentState | ExperimentStateV2) -> None:
        """Keep every separately funded candidate and virtual path at one frontier."""
        names = tuple(item.name for item in self.books)
        if names != tuple(sorted((*model.MODES, *("virtual-" + str(index) for index in range(3))))):
            raise ValueError("Synthetic experiment must retain every independent funded book")
        if any(item.state.sequence != self.next_index for item in self.books):
            raise ValueError("Synthetic experiment has an incomplete funded-book frontier")
        previous = {name: model.initial_book(name) for name in names}
        for period in self.periods:
            rows = tuple((row.name, row.settlement) for row in period.candidates)
            rows += tuple(("virtual-" + str(index), row) for index, row in enumerate(period.recipes))
            for name, receipt in rows:
                if receipt.before != previous[name]:
                    raise ValueError("Synthetic book path differs from its previous original settlement")
                previous[name] = receipt.after
        if any(item.state != previous[item.name] for item in self.books):
            raise ValueError("Synthetic continuation book differs from its actual final settlement")

    def _observation_origins(self: ExperimentState | ExperimentStateV2) -> None:
        """Reconcile only retained original producer/mark joins, without refitting."""
        books = {item.name: item.state for item in self.books}
        for observation in self.observations:
            producer = self.periods[observation.index - 1].pending
            if observation.index < len(self.periods):
                marked = tuple(row.gross_value for row in self.periods[observation.index].recipes)
            else:
                marked = tuple(mark(books["virtual-" + str(index)], observation.prices).value for index in range(3))
            _original_observation(observation, producer, marked, self.observed_prices[observation.index - 1])

    def _components(self: ExperimentState | ExperimentStateV2) -> None:
        """Keep exactly one actual owner for every configured reference/control."""
        if tuple(item.name for item in self.references) != ("fixed-share", "hedge"):
            raise ValueError("Synthetic checkpoint requires each reference exactly once")
        if tuple(item.name for item in self.controlled) != tuple(sorted(model.CONTROLLED_MODES)):
            raise ValueError("Synthetic checkpoint requires each controlled owner exactly once")
        self._laws()
        accepted = sum(all(row.output.status == "observed" for row in item.applications) for item in self.observations)
        if any(item.state.payload.update_count != accepted for item in (*self.references, *self.controlled)):
            raise ValueError("Mixture continuation count differs from its actual accepted observed panels")
        if self.periods:
            self._emitted_coefficients()

    def _laws(self: ExperimentState | ExperimentStateV2) -> None:
        """Bind the complete declared profile, selected strength and logical owner.

        These pure configuration comparisons perform no learner construction or
        restore. Reading a complete artifact must not relabel another learner's
        law as this fixed reference profile, even before execution is requested.
        Content bindings establish internal consistency, not authentication.
        """
        references = tuple(
            (
                item.state,
                item.state.payload.configuration,
                model.reference_config("hedge-constant" if item.name == "hedge" else "fixed-share-excluding-origin"),
                model.ref(item.name + "-owner"),
            )
            for item in self.references
        )
        controlled = tuple(
            (
                item.state,
                item.state.payload.config,
                model.controlled_config(item.name, strength=self.strength),
                model.ref(item.name + "-owner"),
            )
            for item in self.controlled
        )
        for state, actual, configuration, owner in (
            *references,
            *controlled,
            (self.ridge, self.ridge.payload.config, model.ridge_config(), model.ref("ridge-owner")),
        ):
            expected = configuration.fingerprint
            if (
                actual.fingerprint,
                state.binding.configuration_fingerprint,
                state.binding.owner,
            ) != (expected, expected, owner):
                raise ValueError("Synthetic component differs from its declared original mathematical law or owner")

    def _emitted_coefficients(self: ExperimentState | ExperimentStateV2) -> None:
        """Join the next committed math origin to the actual last emitted trace."""
        emitted = {row.name: row.coefficients for row in self.periods[-1].candidates}
        # A feedback-ready checkpoint has advanced the actual learners but has
        # deliberately not emitted another candidate coefficient vector yet.
        if self.ready_observation is not None:
            return
        if any(item.state.payload.weights != emitted[item.name] for item in self.references):
            raise ValueError("Reference continuation differs from its actual last emitted coefficients")
        if any(item.state.payload.origin.weights != emitted[item.name] for item in self.controlled):
            raise ValueError("Controlled continuation differs from its actual last emitted coefficients")

    def _sources(self: ExperimentState | ExperimentStateV2) -> None:
        """Bind only the observed prefix, including the feedback-ready input."""
        for index, price in enumerate(self.observed_prices):
            if price.at != model.START + timedelta(days=index):
                raise ValueError("Synthetic source prefix has another original calendar")
            if index < len(self.periods):
                expected = self.periods[index].recipes[0].valuation.prices
            else:
                assert self.ready_observation is not None
                expected = self.ready_observation.prices
            if price != expected:
                raise ValueError("Synthetic source prefix differs from its retained original application")

    def _coefficient_paths(self: ExperimentState | ExperimentStateV2) -> None:
        """Recompute the declared binary64 full-L1 diagnostic along each actual path."""
        previous = None
        for period in self.periods:
            for index, candidate in enumerate(period.candidates):
                expected = None
                if candidate.coefficients is not None:
                    expected = 0.0
                    if previous is not None:
                        old = previous.candidates[index].coefficients
                        assert old is not None
                        expected = sum(
                            abs(new.value - before.value)
                            for new, before in zip(candidate.coefficients, old, strict=True)
                        )
                if candidate.coefficient_movement != expected:
                    raise ValueError("Reported coefficient movement differs from its actual original vectors")
            previous = period


class _ExperimentInvariantView:  # pylint: disable=too-few-public-methods
    """Reuse the exact legacy invariant functions on a complete typed view.

    This mixin owns no fields, reader, validation decision or cache. Each sibling
    DTO validates its own declared children before these same checks execute.
    """

    # These are the two sibling DTOs' own shared checks, not foreign owner state.
    # pylint: disable=protected-access
    _counts = ExperimentState._counts
    _originals = ExperimentState._originals
    _books = ExperimentState._books
    _observation_origins = ExperimentState._observation_origins
    _components = ExperimentState._components
    _laws = ExperimentState._laws
    _emitted_coefficients = ExperimentState._emitted_coefficients
    _sources = ExperimentState._sources
    _coefficient_paths = ExperimentState._coefficient_paths
    # pylint: enable=protected-access


class ExperimentStateV2(_ExperimentInvariantView, PortfolioContract):
    """Store each original observation once under its original period.

    Attributes
    ----------
    schema_version : int
        Required version2; legacy and unknown versions are not inferred.
    strength : float
        Fixed L1 strength in coefficient units, within[0,100]; not a paid fee.
    implementation : DefinitionBinding
        Exact selected host-source identity, not source authentication.
    tape_fingerprint : str
        Content identity of the complete frozen16-row exogenous source. A digest
        exposes no future prices to a decision or grants execution authority.
    next_index : int
        Accepted decision count in[0,16], also the next original period index.
    books : tuple of NamedBook
        Current independently funded candidate and virtual-recipe books. Cash
        is in synthetic credits; positions are exact divisible asset units.
    references : tuple of NamedReference
        Complete actual Hedge/FixedShare configurations, weights and counters.
    controlled : tuple of NamedControlled
        Complete control owners, emitted origins, pending cadence blocks and
        exact per-period coefficient-movement usage.
    ridge : AlgorithmState
        Complete causal predictor configuration and sufficient statistics.
    pending : Pending or None
        Most recent decision's original forecast, frozen scales and opening
        values. Retained lineage also accompanies an already-applied label.
    ready_observation : Observation or None
        Original accepted feedback awaiting the next decision, if present.
    observed_prices : tuple of PricePanel
        Exactly the eligible source prefix, with prices in synthetic credits
        per asset unit and explicit original clocks.
    periods : tuple of Period
        Every accepted decision, original observation and complete funded
        receipt in contiguous order; at most16 periods.
    observations : tuple of Observation
        Derived read-only chronological view, never a serialized field or cache.

    Notes
    -----
    All complete legacy state remains represented. ``observations`` is a derived
    chronological view of original period records plus the feedback-ready record;
    it is not a serialized field or a validation cache. The version and digest
    differ from v1. Reading/converting old evidence is not permission to resume
    another executable image or state codec.
    """

    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)]
    strength: Annotated[Binary64, Field(ge=0, le=100)]
    implementation: DefinitionBinding
    tape_fingerprint: Fingerprint
    next_index: Annotated[int, Field(strict=True, ge=0, le=16)]
    books: tuple[NamedBook, ...]
    references: tuple[NamedReference, ...]
    controlled: tuple[NamedControlled, ...]
    ridge: AlgorithmState[RidgePayload]
    pending: Pending | None
    ready_observation: Observation | None
    observed_prices: Annotated[tuple[PricePanel, ...], Field(max_length=16)]
    periods: Annotated[tuple[Period, ...], Field(max_length=16)]

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Return every original accepted label without filtering missing rows.

        Returns
        -------
        tuple of Observation
            Complete ordered labels, including an applied feedback-ready label.

        Raises
        ------
        ValueError
            An emitted later period lacks its original complete observation.
        """
        records: list[Observation] = []
        for period in self.periods[1:]:
            if period.observation is None:
                raise ValueError("Normalized state lost an original period observation")
            records.append(period.observation)
        if self.ready_observation is not None:
            records.append(self.ready_observation)
        return tuple(records)

    def _originals(self) -> None:
        """Bind original producers and calendar without comparing a view to itself."""
        if self.pending is not None and _validated_wire(self.pending) != _validated_wire(self.periods[-1].pending):
            raise ValueError("Synthetic pending forecast differs from its retained original producer")
        records = self.observations
        if self.periods and self.periods[0].observation is not None:
            raise ValueError("Initial synthetic decision cannot contain a future label")
        if tuple(row.index for row in records) != tuple(range(1, len(records) + 1)):
            raise ValueError("Synthetic observations must retain their original contiguous calendar")

    @model_validator(mode="before")
    @classmethod
    def _no_duplicate_observations(cls, value):
        """Refuse extra duplicate evidence even on copied/constructed ingress."""
        supplied = value if isinstance(value, dict) else getattr(value, "__dict__", {})
        if "observations" in supplied:
            raise ValueError("Normalized v2 has no serialized observations field")
        return value

    @classmethod
    def model_construct(cls, _fields_set=None, **values):
        """Keep unchecked construction from silently discarding duplicate input.

        Parameters
        ----------
        _fields_set : set of str or None, optional
            Standard Pydantic unchecked field-set metadata.
        **values : object
            Declared normalized fields. Their later reader still validates all.

        Returns
        -------
        ExperimentStateV2
            Unchecked assembly requiring the ordinary full reader before use.

        Raises
        ------
        ValueError
            The removed duplicate observation field is supplied.
        """
        if "observations" in values:
            raise ValueError("Normalized v2 has no serialized observations field")
        return super().model_construct(_fields_set=_fields_set, **values)

    @model_validator(mode="after")
    def _complete_view(self) -> Self:
        """Apply every unchanged legacy invariant to the authoritative view."""
        self._counts()
        self._originals()
        self._books()
        self._observation_origins()
        self._components()
        self._sources()
        self._coefficient_paths()
        return self

    @classmethod
    def from_legacy(cls, state: ExperimentState) -> Self:
        """Explicitly convert complete readable v1 evidence without resuming it.

        Parameters
        ----------
        state : ExperimentState
            Exact complete v1 state. Full legacy validation precedes conversion.

        Returns
        -------
        ExperimentStateV2
            New separately versioned representation of every original value.

        Raises
        ------
        ValueError, TypeError
            Invalid, incompatible or non-losslessly portable legacy evidence.

        Notes
        -----
        This pure read conversion grants no image, collection or execution
        compatibility. It is never called by the normalized decoder.
        """
        if type(state) is not ExperimentState:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Legacy conversion requires the exact v1 state type")
        checked = ExperimentState.model_validate(state)
        _portable_conversion(checked)
        fields = checked.model_dump(mode="python", exclude={"observations", "schema_version"})
        return cls(schema_version=2, **fields)

    def to_legacy(self) -> ExperimentState:
        """Expand the complete v1 read view explicitly, without migration.

        Returns
        -------
        ExperimentState
            Fully validated legacy representation, with all observations kept.

        Raises
        ------
        ValueError, TypeError
            Invalid state or values not preserved by the original wire reader.

        Notes
        -----
        The normalized reader never performs this expansion. No store, owner,
        execution acknowledgement or old-image continuation is created.
        """
        checked = type(self).model_validate(self)
        _portable_conversion(checked)
        fields = checked.model_dump(mode="python", exclude={"schema_version"})
        return ExperimentState(schema_version=1, observations=checked.observations, **fields)


def _portable_conversion(state: ExperimentState | ExperimentStateV2) -> None:
    """Use the public exact codec to refuse lossy typed read conversions."""
    binding = StateBinding(
        owner=model.ref("read-only-state-conversion"),
        implementation=state.implementation,
        state_codec=model.ref("read-only-state-schema-" + str(state.schema_version)),
        configuration_fingerprint=state.tape_fingerprint,
    )
    ComponentCodec(binding, type(state)).encode(state)
