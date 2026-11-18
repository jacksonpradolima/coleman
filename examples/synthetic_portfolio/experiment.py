"""Chronological real learners, causal ridge, constrained targets and funded books.

All marks are authored exogenous synthetic observations. Virtual recipe books
own their own paths and modeled fees. Only a candidate's netted deployed book
owns that candidate's executed charges; virtual charges are never debited again.
"""

from __future__ import annotations

from datetime import timedelta
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.numeric import Scalar
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.learning.control_contracts import MovementPeriod
from coleman.learning.controlled import ControlledHedge
from coleman.learning.reference import FixedShare, Hedge
from coleman.objectives.bindings import DefinitionBinding
from coleman.objectives.controls import IntervalCadence, ReleaseInterval
from coleman.objectives.volatility import (
    FitSample,
    FitSnapshot,
    apply_signed_volatility,
    fit_rolling_scale,
    select_scale,
)
from coleman.optimization.continuous import solve_continuous
from coleman.optimization.continuous_contracts import ContinuousSolution
from coleman.prediction import AppendOnlyRidge, RidgeInput, RidgeSample
from coleman.problems.portfolio import Book, PricePanel, Target, mark
from coleman.problems.portfolio.session import BookSession, BookSnapshot
from coleman.state.components import ComponentCodec
from coleman.state.contracts import StateBinding

from . import model
from .feedback import ControlReceipt
from .flow_contracts import (
    CandidateStep,
    ExperimentState,
    ExperimentStateV2,
    NamedBook,
    NamedControlled,
    NamedReference,
    Observation,
    Pending,
    Period,
    _control_objective,
)


def _rational_coefficients(weights: tuple[Weight, ...]) -> tuple[Fraction, ...]:
    """Map a qualified near-simplex to exact proportions in one declared adapter.

    The learner's emitted binary64 bytes remain unchanged in the trace. Dividing
    their exact ratios by their exact positive sum is solely the portfolio target
    adapter, not a repair of a learner or constrained solver output.
    """
    values = tuple(Fraction(item.value) for item in weights)
    total = sum(values, Fraction())
    return tuple(value / total for value in values)


def _implementation() -> DefinitionBinding:
    """Bind exact selected public host bytes, independently of filesystem paths."""
    digest = sha256()
    for name in ("model.py", "feedback.py", "diagnostics.py", "flow_contracts.py", "experiment.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return DefinitionBinding(
        reference=model.ref("exact-reference-host"), definition_fingerprint="sha256:" + digest.hexdigest()
    )


def _combine(
    coefficients: tuple[Weight, ...] | None, recipes: tuple[tuple[Fraction, Fraction], ...]
) -> tuple[Fraction, Fraction]:
    """Combine recipe targets exactly once; non-mixture baselines start equal."""
    if coefficients is None:
        return Fraction(1, 2), Fraction(1, 2)
    values = _rational_coefficients(coefficients)

    def combined(asset: int) -> Fraction:
        """Net recipe targets before any actual funded-book transaction."""
        return sum(
            (coefficient * recipe[asset] for coefficient, recipe in zip(values, recipes, strict=True)), Fraction()
        )

    return combined(0), combined(1)


def _known_value(book: Book, prices: PricePanel) -> Fraction:
    """Require an actual known mark; an unavailable return cannot become zero."""
    value = mark(book, prices).value
    if value is None:
        raise ValueError("Synthetic funded outcome valuation is unknown")
    return value.fraction


def _source_identity(rows: tuple[tuple[int, int], ...] | None, source_fingerprint: Fingerprint | None) -> Fingerprint:
    """Validate source metadata without exposing future observations to kernels."""
    if rows is not None and (
        type(rows) not in (tuple,)
        or len(rows) != 16
        or any(
            type(row) not in (tuple,)
            or len(row) != 2
            or any(type(value) not in (int,) or value <= 0 or value.bit_length() > 4096 for value in row)
            for row in rows
        )
    ):
        raise ValueError("Synthetic reference requires16positive integral two-asset marks")
    if rows is None and source_fingerprint is None:
        raise ValueError("Externally supplied prices require an explicit frozen source identity")
    expected = None if rows is None else fingerprint([list(row) for row in rows])
    if source_fingerprint is not None and expected is not None and source_fingerprint != expected:
        raise ValueError("Declared source fingerprint differs from the supplied complete source")
    declared = expected if source_fingerprint is None else source_fingerprint
    assert declared is not None
    return DefinitionBinding(
        reference=model.ref("authored-tape"), definition_fingerprint=declared
    ).definition_fingerprint


# Each actual independently checkpointed component stays explicit.
class Experiment:  # pylint: disable=too-many-instance-attributes
    """Own a bounded chronological authored tape and independently funded methods.

    Parameters
    ----------
    strength : float, optional
        Fixed nonnegative L1 strength. Development selects it before protected
        execution; changing it prevents continuation under the old law.
    rows : tuple of tuple of int, optional
        Exactly16 authored positive two-asset price rows. Decisions expose only
        the current/past prefix; exogenous prices are unaffected by actions.
        Null keeps all source values outside this consumer; each current quote
        must then be supplied through ``step`` or ``apply_feedback``.
    source_fingerprint : str or None, optional
        Required complete source content identity when rows is null. A digest
        identifies the frozen source; it supplies no future values or authority.

    Notes
    -----
    Calls are sequential and local. ``step`` stages a complete independent owner
    before assigning its accepted state. Duplicate already-retained period calls
    return the original period without repeating feedback, fees or penalties.
    It models no market microstructure, financing or statistical independence.
    """

    # Exact built-ins reject bool and callback-bearing int subclasses.
    def __init__(
        self,
        *,
        strength: float = 0.05,
        rows: tuple[tuple[int, int], ...] | None = model.PRICE_ROWS,
        source_fingerprint: Fingerprint | None = None,
    ) -> None:
        """Construct actual shipped component owners and virgin funded paths."""
        if type(strength) not in (float, int) or not 0 <= strength <= 100:
            raise ValueError("Synthetic declared strength must lie in[0,100]")
        self.strength, self.rows = strength, rows
        self.source_fingerprint = _source_identity(rows, source_fingerprint)
        names = (*model.MODES, *("virtual-" + str(index) for index in range(3)))
        self.books = {name: BookSession(model.initial_book(name)) for name in names}
        self.references = {
            "hedge": Hedge(model.reference_config("hedge-constant"), owner=model.ref("hedge-owner")),
            "fixed-share": FixedShare(
                model.reference_config("fixed-share-excluding-origin"), owner=model.ref("fixed-share-owner")
            ),
        }
        self.controlled = {
            name: ControlledHedge(model.controlled_config(name, strength=strength), owner=model.ref(name + "-owner"))
            for name in model.CONTROLLED_MODES
        }
        self.ridge = AppendOnlyRidge(model.ridge_config(), owner=model.ref("ridge-owner"))
        self.pending: Pending | None = None
        self.ready_observation: Observation | None = None
        self.observed_prices: tuple[PricePanel, ...] = ()
        self.observations: tuple[Observation, ...] = ()
        self.periods: tuple[Period, ...] = ()

    def _feedback(self, index: int) -> Observation | None:
        """Apply the actual prior producer's later labels, never a current fit."""
        if self.pending is None:
            return None
        quote = self.observed_prices[index]
        raw = tuple(
            Scalar(
                spec=model.RAW.spec,
                status="observed",
                value=float(_known_value(self.books["virtual-" + str(member)].book, quote) / opening.fraction - 1),
            )
            for member, opening in enumerate(self.pending.recipe_opening_values)
        )
        applications = tuple(
            apply_signed_volatility(selection, value)
            for selection, value in zip(self.pending.selections, raw, strict=True)
        )
        spread = self._spread(index)
        label = Scalar(spec=self.ridge.config.target, status="observed", value=float(spread))
        updated = self.ridge.append(RidgeSample(schema_version=1, input=self.pending.prediction.input, target=label))
        if updated.status != "accepted":
            raise ValueError("Synthetic ridge refused its eligible original label")
        controls: tuple[ControlReceipt, ...] = ()
        if all(application.output.status == "observed" for application in applications):
            # Keep this adapter's causal losses/provenance explicit; the similar
            # operational example instead constructs its own authored loss law.
            # pylint: disable=duplicate-code
            panel = FullPanel(
                rows=tuple(
                    FeedbackRow(
                        member=member,
                        values=(
                            OutcomeValue(
                                spec=model.LOSS.spec,
                                value=application.output.value,
                                status="observed",
                                origin="synthetic",
                                transform=model.TRANSFORM.reference,
                            ),
                        ),
                    )
                    for member, application in zip(model.RECIPES, applications, strict=True)
                )
            )
            # pylint: enable=duplicate-code
            for learner in self.references.values():
                if learner.observe(panel).status != "accepted":
                    raise ValueError("Synthetic reference update refused")
            controls = self._controlled_panel(panel, index, quote)
        controls += self._release_intervals(index, quote)
        observation = Observation(
            schema_version=1,
            index=index,
            prices=quote,
            raw=raw,
            applications=applications,
            ridge_label=label,
            controls=controls,
        )
        self.observations += (observation,)
        self.ready_observation = observation
        return observation

    def _controlled_panel(self, panel: FullPanel, index: int, quote: PricePanel) -> tuple[ControlReceipt, ...]:
        """Accept original available losses under the explicit movement clock."""
        receipts = []
        for name, learner in self.controlled.items():
            if learner.config.period_clock is not None:
                learner.begin_period(
                    MovementPeriod(
                        schema_version=1,
                        calendar=learner.config.period_clock,
                        interval=model.ref("movement-period-" + str(index)),
                        start=quote.at,
                        end=quote.at + timedelta(days=1),
                    ),
                    apply_at=quote.at,
                )
            result = learner.observe(panel, eligible_at=quote.at, apply_at=quote.at)
            if result.disposition in {"infeasible", "numerical_failure"}:
                raise ValueError("Synthetic control release refused: " + result.disposition)
            receipts.append(ControlReceipt(schema_version=1, name=name, outcome=result))
        return tuple(receipts)

    def _release_intervals(self, index: int, quote: PricePanel) -> tuple[ControlReceipt, ...]:
        """Release a closed explicit interval once, including an empty drain."""
        receipts = []
        if index % 2 == 0:
            for name, learner in self.controlled.items():
                if learner.config.period_clock is None:
                    continue
                if (
                    not learner.export_state().payload.periods
                    or learner.export_state().payload.periods[-1].start != quote.at
                ):
                    learner.begin_period(
                        MovementPeriod(
                            schema_version=1,
                            calendar=learner.config.period_clock,
                            interval=model.ref("movement-period-" + str(index)),
                            start=quote.at,
                            end=quote.at + timedelta(days=1),
                        ),
                        apply_at=quote.at,
                    )
                interval = ReleaseInterval(
                    interval=model.ref("release-" + str(index)),
                    start=quote.at - timedelta(days=2),
                    end=quote.at,
                    cutoff=quote.at,
                    release_at=quote.at,
                )
                cadence = learner.config.controls.cadence
                if not isinstance(cadence, IntervalCadence):
                    raise ValueError("Explicit interval control requires its actual calendar")
                result = learner.release(interval, calendar=cadence.calendar, apply_at=quote.at)
                if result.disposition in {"infeasible", "numerical_failure"}:
                    raise ValueError("Synthetic interval release refused: " + result.disposition)
                receipts.append(ControlReceipt(schema_version=1, name=name, outcome=result))
        return tuple(receipts)

    def _pending(self, index: int) -> Pending:
        """Freeze the actual pre-label forecast and past-only signed-loss scales."""
        quote = self.observed_prices[index]
        lagged = float(self._spread(index))
        prediction = self.ridge.predict(
            RidgeInput(schema_version=1, config=self.ridge.config.binding, values=(1.0, lagged))
        )
        if prediction.failure is not None:
            raise ValueError("Synthetic ridge prediction unavailable")
        timing = DecisionTiming(
            decision_at=quote.at,
            information_cutoff=quote.at,
            target_start=quote.at,
            target_end=quote.at + timedelta(days=1),
            eligible_input_fingerprint=quote.fingerprint,
            mode="future",
        )
        selections = []
        opening_values = []
        for member, recipe in enumerate(model.RECIPES):
            samples = tuple(
                FitSample(
                    observation=model.ref("outcome-" + str(row.index) + "-" + str(member)),
                    value=row.raw[member],
                    timing=FeedbackTiming(
                        event_at=model.START + timedelta(days=row.index),
                        available_at=model.START + timedelta(days=row.index),
                        mature_at=model.START + timedelta(days=row.index),
                        received_at=model.START + timedelta(days=row.index),
                    ),
                )
                for row in self.observations
            )
            frozen = fit_rolling_scale(
                model.SCALE,
                FitSnapshot(
                    schema_version=1,
                    source=model.ref("eligible-outcomes-" + str(index)),
                    quantity=model.RAW,
                    member=recipe,
                    training_cutoff=quote.at,
                    samples=samples,
                ),
                frozen_at=quote.at,
            )
            selections.append(
                select_scale(
                    model.TRANSFORM,
                    frozen,
                    decision_fingerprint=fingerprint(
                        {"period": index, "recipe": recipe.model_dump(mode="json"), "prices": quote.fingerprint}
                    ),
                    timing=timing,
                    outcome=model.ref("outcome-" + str(index + 1) + "-" + str(member)),
                    member=recipe,
                )
            )
            opening_values.append(mark(self.books["virtual-" + str(member)].book, quote).value)
        return Pending(
            schema_version=1,
            index=index,
            prediction=prediction,
            selections=tuple(selections),
            recipe_opening_values=tuple(opening_values),
        )

    def _candidate(
        self, name: str, index: int, pending: Pending, recipes: tuple[tuple[Fraction, Fraction], ...]
    ) -> CandidateStep:
        """Map coefficients once into a netted target, then fund one actual book."""
        learner = self.references.get(name) or self.controlled.get(name)
        receipts = () if self.ready_observation is None else self.ready_observation.controls
        control = next((receipt.outcome for receipt in reversed(receipts) if receipt.name == name), None)
        coefficients = learner.weights if learner is not None else None
        mixture = (Fraction(1, 2), Fraction(1, 2))
        if coefficients is not None:
            mixture = _combine(coefficients, recipes)
        proposed, solution = self._allocation(name, index, pending, mixture)
        settlement = self.books[name].deliver(
            self.observed_prices[index],
            None
            if index != 0 and (name == "buy-and-hold" or name == "periodic-rebalance" and index % 2 != 0)
            else proposed,
            model.fees(index),
            request=model.ref(name + "-decision-" + str(index)),
        )
        if settlement.status != "executed":
            raise ValueError("Synthetic candidate funded execution refused: " + name + ":" + settlement.reason)
        previous = (
            next((row for row in self.periods[-1].candidates if row.name == name), None) if self.periods else None
        )
        movement = None
        if coefficients is not None:
            movement = 0.0
            if previous is not None:
                if previous.coefficients is None:
                    raise ValueError("A mixture route cannot replace a non-mixture producer")
                movement = sum(
                    abs(new.value - old.value) for new, old in zip(coefficients, previous.coefficients, strict=True)
                )
        return CandidateStep(
            schema_version=1,
            name=name,
            coefficients=coefficients,
            coefficient_movement=movement,
            control=control,
            control_objective=_control_objective(control),
            proposed_target=proposed,
            solver=solution,
            settlement=settlement,
        )

    def _allocation(
        self, name: str, index: int, pending: Pending, mixture: tuple[Fraction, Fraction]
    ) -> tuple[Target, ContinuousSolution | None]:
        """Keep the original QP output distinct from its funded target adapter."""
        solution = None
        if name in {"equal", "buy-and-hold", "periodic-rebalance", "constrained-fixed"}:
            mixture = (Fraction(1, 2), Fraction(1, 2))
        if name in {"composed-online", "constrained-fixed", "periodic-rebalance"}:
            forecast = 0.0 if name == "constrained-fixed" else pending.prediction.value.value
            if forecast is None:
                raise ValueError("Numerically unavailable forecast cannot become a zero")
            solution = solve_continuous(model.continuous_problem(mixture, forecast))
            if solution.status != "optimal" or solution.executed is None:
                raise ValueError("Synthetic funded target refuses unqualified continuous solve")
            # A separate declared target adapter balances the last asset to exact
            # total1; the emitted solver coordinates and certificate stay intact.
            first = Fraction(solution.executed[0])
            mixture = (first, 1 - first)
        return model.target(name + "-target-" + str(index), mixture), solution

    def _step(self, index: int) -> Period:
        """Advance only the staged independent owner; the public call commits it."""
        observation = self.ready_observation if self.ready_observation is not None else self._feedback(index)
        pending = self._pending(index)
        lag = pending.prediction.input.values[1]
        trend = (Fraction(3, 4), Fraction(1, 4)) if lag > 0 else (Fraction(1, 4), Fraction(3, 4))
        recipes = ((Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 5), Fraction(4, 5)), trend)
        virtual = tuple(
            self.books["virtual-" + str(member)].deliver(
                self.observed_prices[index],
                model.target("recipe-target-" + str(index) + "-" + str(member), recipe),
                model.fees(index),
                request=model.ref("virtual-" + str(member) + "-decision-" + str(index)),
            )
            for member, recipe in enumerate(recipes)
        )
        if any(receipt.status != "executed" for receipt in virtual):
            raise ValueError("Synthetic virtual recipe funding refused")
        candidates = tuple(self._candidate(name, index, pending, recipes) for name in model.MODES)
        period = Period(
            schema_version=1,
            index=index,
            pending=pending,
            observation=observation,
            recipes=virtual,
            candidates=candidates,
        )
        self.pending = pending
        self.ready_observation = None
        self.periods += (period,)
        return period

    # Exact built-ins preserve the declared clock; same-class staging is owned.
    def step(self, index: int, *, prices: PricePanel | None = None) -> Period:
        """Atomically advance or deduplicate an original chronological period.

        Parameters
        ----------
        index : int
            Next index0..15, or an exact previously retained index retry.
        prices : PricePanel or None, optional
            Explicit current eligible price row, required for an external-source
            consumer. A retry cannot replace a retained row or past observation.

        Returns
        -------
        Period
            Complete original coefficients, forecasts, scale/label observations,
            independent virtual paths and funded candidate receipts.

        Raises
        ------
        ValueError
            Nonsequential index, unavailable/numerically refused component or
            exact arithmetic capacity. Any failure preserves this owner's state.

        Notes
        -----
        Uses a complete independently restored staging owner, not filesystem
        acknowledgement or external transaction semantics. Calls must be serial.
        """
        if type(index) not in (int,) or not 0 <= index < 16 or index > len(self.periods):
            raise ValueError("Synthetic period must be an original sequential index")
        if index < len(self.periods):
            if prices is not None and PricePanel.model_validate(prices) != self.observed_prices[index]:
                raise ValueError("A duplicate decision cannot replace its original eligible prices")
            return self.periods[index]
        staged = self._fork()
        staged._supply_prices(index, prices)  # pylint: disable=protected-access
        result = staged._step(index)  # pylint: disable=protected-access
        self._adopt(staged)
        return result

    def apply_feedback(self, index: int, *, prices: PricePanel | None = None) -> Observation:
        """Apply the original previous decision's later mark before a new decision.

        Parameters
        ----------
        index : int
            Next source index1..15, equal to the accepted decision count. This
            callback does not emit an action or execute a new funded rebalance.
        prices : PricePanel or None, optional
            Actual newly eligible current mark; required without an owned source.

        Returns
        -------
        Observation
            Original eligible label/scale applications. The complete feedback-ready
            phase survives checkpoints before the next decision. An exact retry
            of that ready index returns its original observation.

        Raises
        ------
        ValueError
            Wrong calendar, absent/conflicting mark, unavailable label or refused
            kernel. A failure preserves every original live component.

        Notes
        -----
        This is a domain delivery boundary, not an idempotent learner primitive.
        The optional existing replay owns outer producer/receipt deduplication.
        """
        if type(index) not in (int,) or not 1 <= index < 16 or index != len(self.periods):
            raise ValueError("Feedback requires the next original source index")
        staged = self._fork()
        staged._supply_prices(index, prices)  # pylint: disable=protected-access
        if staged.ready_observation is None:
            staged._feedback(index)  # pylint: disable=protected-access
        assert staged.ready_observation is not None
        result = staged.ready_observation
        self._adopt(staged)
        return result

    def _supply_prices(self, index: int, supplied: PricePanel | None) -> None:
        """Admit one current original row; never read future rows in a kernel."""
        if supplied is None:
            if index < len(self.observed_prices):
                return
            if self.rows is None:
                raise ValueError("Current eligible price input is unavailable")
            supplied = model.prices(index, self.rows)
        quote = PricePanel.model_validate(supplied)
        if (
            quote.at != model.START + timedelta(days=index)
            or tuple(item.asset for item in quote.quotes) != model.ASSETS
        ):
            raise ValueError("Eligible quote differs from the original source clock/catalogue")
        if index < len(self.observed_prices):
            if quote != self.observed_prices[index]:
                raise ValueError("An original applied mark cannot be replaced by a correction")
            return
        if index != len(self.observed_prices):
            raise ValueError("Eligible source prices must arrive in original order")
        if any(item.status != "observed" or item.value is None or item.event_at != quote.at for item in quote.quotes):
            raise ValueError("Synthetic reference waits for current observed price evidence")
        self.observed_prices += (quote,)

    def _spread(self, index: int) -> Fraction:
        """Derive lagged signed spread only from the retained eligible price prefix."""
        if index == 0:
            return Fraction()
        previous, current = self.observed_prices[index - 1], self.observed_prices[index]
        values = tuple(
            new.value.fraction / old.value.fraction
            for old, new in zip(previous.quotes, current.quotes, strict=True)
            if old.value is not None and new.value is not None
        )
        if len(values) != 2:
            raise ValueError("A missing price cannot fabricate a supervised return label")
        return values[0] - values[1]

    def _adopt(self, staged: Experiment) -> None:
        """Transfer only the completed independent stage's actual mutable owners."""
        self.books, self.references, self.controlled, self.ridge = (
            staged.books,
            staged.references,
            staged.controlled,
            staged.ridge,
        )
        self.pending, self.observations, self.periods = staged.pending, staged.observations, staged.periods
        self.ready_observation, self.observed_prices = staged.ready_observation, staged.observed_prices

    def _fork(self) -> Experiment:
        """Stage independent mutable owners, sharing only immutable valid records."""
        staged = Experiment(strength=self.strength, rows=self.rows, source_fingerprint=self.source_fingerprint)
        staged.books = {name: book.fork() for name, book in self.books.items()}
        for name, learner in self.references.items():
            staged.references[name].restore_state(learner.export_state())
        for name, learner in self.controlled.items():
            staged.controlled[name].restore_state(learner.export_state())
        staged.ridge.restore_state(self.ridge.export_state())
        staged.pending, staged.observations, staged.periods = self.pending, self.observations, self.periods
        staged.ready_observation, staged.observed_prices = self.ready_observation, self.observed_prices
        return staged

    def export_state(self) -> ExperimentState:
        """Export every actual owner and pending producer without changing state.

        Returns
        -------
        ExperimentState
            Complete deterministic known-version continuation. No RNG exists.
        """
        return ExperimentState.model_validate(self._capture_state())

    def _capture_state(self) -> ExperimentState:
        """Assemble a complete unchecked value for one immediate outer validation.

        This private projection performs no artifact reading or serialization;
        it reads local implementation source bytes solely for the code binding.
        Every owned field remains present. It is never a validated snapshot or
        cache entry until the caller's complete ExperimentState or enclosing
        replay/task checkpoint validator accepts it. Public export always takes
        that full path; caller-supplied values receive no reuse authority.
        """
        return ExperimentState.model_construct(
            schema_version=1,
            strength=self.strength,
            implementation=_implementation(),
            tape_fingerprint=self.source_fingerprint,
            next_index=len(self.periods),
            books=tuple(
                NamedBook(schema_version=1, name=name, state=self.books[name].book) for name in sorted(self.books)
            ),
            references=tuple(
                NamedReference(schema_version=1, name=name, state=learner.export_state())
                for name, learner in sorted(self.references.items())
            ),
            controlled=tuple(
                NamedControlled(schema_version=1, name=name, state=learner.export_state())
                for name, learner in sorted(self.controlled.items())
            ),
            ridge=self.ridge.export_state(),
            pending=self.pending,
            ready_observation=self.ready_observation,
            observed_prices=self.observed_prices,
            observations=self.observations,
            periods=self.periods,
        )

    def _capture_normalized_state(self) -> ExperimentStateV2:
        """Assemble a complete unchecked value for one immediate outer validation.

        This private projection performs no artifact I/O or publication. It reads
        local implementation source bytes solely for the code binding. Distinct
        observation joins use capture-local lossless serialization; all owned
        values remain represented. It is never a validated snapshot or
        cache entry until the caller's complete normalized state or enclosing
        task checkpoint validator accepts it. Public export always takes
        that full path; caller-supplied values receive no reuse authority.
        """
        captured = ExperimentStateV2.model_construct(
            schema_version=2,
            strength=self.strength,
            implementation=_implementation(),
            tape_fingerprint=self.source_fingerprint,
            next_index=len(self.periods),
            books=tuple(
                NamedBook(schema_version=1, name=name, state=self.books[name].book) for name in sorted(self.books)
            ),
            references=tuple(
                NamedReference(schema_version=1, name=name, state=learner.export_state())
                for name, learner in sorted(self.references.items())
            ),
            controlled=tuple(
                NamedControlled(schema_version=1, name=name, state=learner.export_state())
                for name, learner in sorted(self.controlled.items())
            ),
            ridge=self.ridge.export_state(),
            pending=self.pending,
            ready_observation=self.ready_observation,
            observed_prices=self.observed_prices,
            periods=self.periods,
        )

        # Live rolling fits still own this tuple. Validate it before comparing
        # with its single authoritative record view; do not silently drop a
        # divergent or unchecked live label during normalized export.
        originals = captured.observations
        if len(self.observations) != len(originals):
            raise ValueError("Normalized capture lost an actual owned observation")
        # Exact typed admission matters here: JSON equality alone can conceal a
        # differing datetime fold/zone or another representation-only value.
        # This capture-local public codec preserves every declared field and
        # refuses lossy serialization; it grants no store or execution authority.
        codec = ComponentCodec(
            StateBinding(
                owner=model.ref("live-observation-capture"),
                implementation=captured.implementation,
                state_codec=model.ref("lossless-observation-read"),
                configuration_fingerprint=self.source_fingerprint,
            ),
            Observation,
        )
        for actual, original in zip(self.observations, originals, strict=True):
            if actual is original:
                # Identity proves only this join, never validation. The complete
                # containing v2 reader still audits this authoritative record.
                continue
            if codec.encode(actual).content != codec.encode(original).content:
                raise ValueError("Normalized capture lost or changed an actual owned observation")
        return captured

    def export_normalized_state(self) -> ExperimentStateV2:
        """Export complete schema2 state with each original label stored once.

        Returns
        -------
        ExperimentStateV2
            Fully audited normalized state; every owned statistic, original
            label, producer, book path and pending phase remains represented.

        Raises
        ------
        ValueError, TypeError
            Invalid or divergent live original observations or continuation.

        Notes
        -----
        Performs no state mutation or publication. Legacy ``export_state`` is
        unchanged. Different representation versions have different digests.
        """
        return ExperimentStateV2.model_validate(self._capture_normalized_state())

    def restore_normalized_state(self, state: ExperimentStateV2, /) -> None:
        """Audit and atomically restore complete current-image schema2 state.

        Parameters
        ----------
        state : ExperimentStateV2
            Complete normalized state for this exact source, law and image.

        Raises
        ------
        ValueError, TypeError
            Invalid version, original evidence, codec, law or implementation.
            The live experiment remains unchanged on refusal.

        Notes
        -----
        Reading or converting v1 evidence does not grant compatibility with a
        new executable image. No automatic migration or acknowledgement occurs.
        """
        if type(state) is not ExperimentStateV2:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Normalized restore requires the exact schema2 state")
        checked = ExperimentStateV2.model_validate(state)
        self._restore_checked(checked)

    def restore_state(self, state: ExperimentState, /) -> None:
        """Restore complete compatible components before replacing any live owner.

        Parameters
        ----------
        state : ExperimentState
            Exact tape/law, independently funded paths, full learner/predictor
            payloads and original producer/feedback frontier.

        Raises
        ------
        ValueError
            Unknown version, different tape or strength, missing component,
            incompatible binding or inconsistent original frontier. No migration
            or partial live restore occurs.

        Notes
        -----
        Each component is revalidated through the concrete public snapshot type
        of its fresh staged owner. Pydantic may create distinct equivalent
        generic wrapper classes in different execution contexts; their Python
        object identity is not part of this example's wire format. This
        normalization retains every field and the existing strict component
        restore, without changing its configuration, codec or state binding.
        """
        checked = ExperimentState.model_validate(state)
        self._restore_checked(checked)

    def _restore_checked(self, checked: ExperimentState | ExperimentStateV2, /) -> None:
        """Stage every owned component after one immediate complete state audit.

        Parameters
        ----------
        checked : ExperimentState or ExperimentStateV2
            Complete state just validated by a public restore or the task
            owner's enclosing versioned checkpoint reader. No raw JSON,
            copied or unchecked caller state gains this private authority.

        Raises
        ------
        ValueError
            Incompatible tape, law, implementation, complete book history or
            mathematical continuation. Every live owner remains unchanged.

        Notes
        -----
        This private same-operation seam avoids repeating the inner model audit
        after the task boundary already validated every nested field. It keeps
        all binding checks, actual book replay, concrete-context normalization,
        strict mathematical restores and final atomic adoption. It retains no
        snapshot or validation cache and changes no public reader or schema.
        """
        if checked.strength != self.strength or checked.tape_fingerprint != self.source_fingerprint:
            raise ValueError("Synthetic checkpoint belongs to another tape or selected law")
        if checked.implementation != _implementation():
            raise ValueError("Synthetic checkpoint requires its exact selected host implementation")
        staged = Experiment(strength=self.strength, rows=self.rows, source_fingerprint=self.source_fingerprint)
        if {item.name for item in checked.references} != set(staged.references) or {
            item.name for item in checked.controlled
        } != set(staged.controlled):
            raise ValueError("Synthetic checkpoint lost an actual component")
        for item in checked.books:
            receipts = tuple(
                period.recipes[int(item.name[-1])]
                if item.name.startswith("virtual-")
                else next(row.settlement for row in period.candidates if row.name == item.name)
                for period in checked.periods
            )
            # Assemble every original field privately, then let the unchanged
            # public restore perform its complete exact transition replay once.
            history = BookSnapshot.model_construct(
                schema_version=1, initial=model.initial_book(item.name), receipts=receipts
            )
            staged.books[item.name].restore_state(history)
            if staged.books[item.name].book.to_json() != item.state.to_json():
                raise ValueError("Synthetic checkpoint book differs from its complete original path")
        for item in checked.references:
            learner = staged.references[item.name]
            concrete = type(learner.export_state()).model_validate(item.state.model_dump(mode="python"))
            learner.restore_state(concrete)
        for item in checked.controlled:
            controller = staged.controlled[item.name]
            controlled = type(controller.export_state()).model_validate(item.state.model_dump(mode="python"))
            controller.restore_state(controlled)
        predicted = type(staged.ridge.export_state()).model_validate(checked.ridge.model_dump(mode="python"))
        staged.ridge.restore_state(predicted)
        staged.pending, staged.observations, staged.periods = checked.pending, checked.observations, checked.periods
        staged.ready_observation, staged.observed_prices = checked.ready_observation, checked.observed_prices
        self._adopt(staged)
