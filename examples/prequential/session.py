"""Jointly acknowledged real replay and independent CI domain state owners."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from coleman.core.actions import MixtureAction
from coleman.core.primitives import fingerprint
from coleman.core.timing import FeedbackTiming
from coleman.evaluation.design.cells import repetition_execution_id
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.prequential.contracts import (
    ComponentObservation,
    StepEvidence,
)
from coleman.evaluation.prequential.execution import DomainStepPlan, PrequentialStep, execute_prequential_step
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance
from coleman.state.components import ComponentBinding, ComponentCodec, ComponentValue, child_frontier
from coleman.state.contracts import StateBinding
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import (
    CollectionDefinition,
    CommitIntent,
    CommitRequest,
    ComponentSpec,
    ExecutionCompatibility,
)

from . import model
from .profile import ObservationInput, decision_timing, execution_key, primary_component

MODEL = model


class Session:
    """Jointly acknowledge actual replay and complete domain/predictor/accounting.

    Parameters
    ----------
    root : pathlib.Path
        Missing store path for create, or existing path for explicit reopening.
    recipe : model.Recipe
        Immutable actual method, BASE source/configuration and independent world.
    environment : ExecutionCompatibility
        Actual outer capture, shared without recapturing it for every cell.
    create : bool, optional
        Explicit initialization versus reopening. Never falls back from corrupt
        or incompatible existing state to fresh learning.

    Notes
    -----
    Every actual transition uses fresh owners under LocalStateStore. No external
    CI action occurs. Source/protected exposure is a separate retained audit.
    """

    def __init__(self, root: Path, recipe, environment: ExecutionCompatibility, *, create: bool = False):
        """Create or reopen the selected joint store, preserving acknowledged state on errors."""
        self.recipe, self.environment = recipe, environment
        self._accepted_emissions: dict[int, tuple[PrequentialStep, float, float]] = {}
        self._accepted_observations: dict[int, ObservationInput] = {}
        replay = MODEL.ReplayOwner(recipe)
        domain = MODEL.DomainOwner(recipe)
        replay_binding = StateBinding(
            owner=MODEL.ref(recipe.child.id + "-replay"),
            implementation=MODEL.ADAPTER,
            state_codec=MODEL.ref("public-ci-retained-continuation"),
            configuration_fingerprint=fingerprint(
                {
                    "recipe": recipe.config.model_dump(mode="json"),
                    "schedule": recipe.schedule.model_dump(mode="json"),
                    "BASE": recipe.base_fingerprint,
                    "world": recipe.world,
                }
            ),
        )
        replay_codec = ComponentCodec(replay_binding, type(replay.export_state()))
        domain_codec = ComponentCodec(domain.binding, MODEL.DomainState)
        self.replay_binding = ComponentBinding(
            replay_codec, partial(MODEL.ReplayOwner, recipe), partial(child_frontier, child=recipe.child)
        )
        self.domain_binding = ComponentBinding(domain_codec, partial(MODEL.DomainOwner, recipe), lambda _: None)
        definition = CollectionDefinition(
            schema_version=1,
            reference=MODEL.ref(recipe.child.id + "-collection"),
            session=recipe.child,
            source_epoch=MODEL.EPOCH,
            children=(recipe.child,),
            components=(
                ComponentSpec(
                    binding=replay_binding,
                    child=recipe.child,
                    roles=("algorithm", "learner_pending", "delivery"),
                    frontier_profile="replay-ordinal-v2",
                ),
                ComponentSpec(
                    binding=domain.binding,
                    child=recipe.child,
                    roles=("domain", "predictor", "applied_path"),
                    frontier_profile="none",
                ),
            ),
        )
        if create:
            root.parent.mkdir(parents=True, exist_ok=True)
            self.store = LocalStateStore.create(root, definition, environment)
            self.store.initialize(
                self.command("initialize", None, None),
                (
                    ComponentValue(self.replay_codec, replay.export_state()),
                    ComponentValue(self.domain_codec, domain.export_state()),
                ),
                self.bindings,
                compatibility=environment,
            )
        else:
            self.store = LocalStateStore.open(root)
            if self.store.read_head() is None:
                self.store.initialize(
                    self.command("initialize", None, None),
                    (
                        ComponentValue(self.replay_codec, replay.export_state()),
                        ComponentValue(self.domain_codec, domain.export_state()),
                    ),
                    self.bindings,
                    compatibility=environment,
                )
            self.snapshots()

    @property
    def replay_codec(self):
        """Return the original concrete replay codec without mutating an owner."""
        return self.replay_binding.codec

    @property
    def domain_codec(self):
        """Return the original complete domain codec without reading source data."""
        return self.domain_binding.codec

    @property
    def bindings(self):
        """Return the fixed ordered pair of independently owned component bindings."""
        return self.replay_binding, self.domain_binding

    def snapshots(self):
        """Read and verify both actual accepted complete component states.

        Returns
        -------
        tuple
            Concrete retained replay continuation and complete DomainState, including
            original pending producers, raw expert history, forecasts and accounting.

        Raises
        ------
        ValueError, OSError
            Invalid/incompatible bytes, missing producer, cross-owner D/U mismatch or
            accounting differing from actual coefficients. No fresh-state fallback.

        Notes
        -----
        Acquires the existing bounded store read lock and rebuilds only in-memory
        immutable retry metadata. It performs no learning, new cost or publication."""
        current = self.store.load(self.bindings, compatibility=self.environment)
        replay, domain = current.component(self.replay_codec), current.component(self.domain_codec)
        if replay.state.decision_count != len(domain.receipts) or replay.state.update_count != domain.count:
            raise ValueError("Joint replay/domain counters differ in the accepted collection")
        producers = {row.fingerprint: row for row in replay.state.decisions}
        self._restore_emissions(domain, producers)
        self._restore_observations(replay, domain, producers)
        return replay, domain

    def _restore_emissions(self, domain, producers) -> None:
        """Rebuild immutable emission metadata after original producer/accounting checks."""
        previous = (0.5, 0.5)
        for ordinal, receipt in enumerate(domain.receipts):
            if receipt.coefficient_record_fingerprint not in producers:
                raise ValueError("Domain receipt has no actual retained coefficient producer")
            step = PrequentialStep(
                schema_version=1, coefficient=producers[receipt.coefficient_record_fingerprint], domain=receipt
            )
            if receipt.key != execution_key(self.recipe.mode, self.recipe.world, ordinal):
                raise ValueError("Accepted domain history changes its logical recipe or opportunity")
            if not isinstance(step.coefficient.executed.action, MixtureAction):
                raise ValueError("Accepted coefficient history changed its mixture action shape")
            weights = tuple(row.value for row in step.coefficient.executed.action.weights)
            movement = MODEL.fsum(abs(new - old) for new, old in zip(weights, previous, strict=True))
            rate = 0.0 if self.recipe.mode == "accounting_off" else 4.0
            if domain.movements[ordinal] != movement or domain.charges[ordinal] != rate * movement:
                raise ValueError("Accepted domain accounting differs from actual emitted coefficients")
            self._accepted_emissions[ordinal] = (step, domain.charges[ordinal], domain.movements[ordinal])
            previous = weights
        if domain.last_coefficients != previous:
            raise ValueError("Accepted last coefficients differ from the original replay history")

    def _restore_observations(self, replay, domain, producers) -> None:
        """Rebuild complete originals from actual selected applications and raw history."""
        applications = {row.application.executed.fingerprint: row.application for row in replay.state.updates}
        for step, seconds in zip(domain.observations, domain.expert_history, strict=True):
            ordinal = step.receipt.executed.proposal.request.round
            application = applications.get(step.receipt.coefficient_execution_fingerprint)
            if application is None:
                raise ValueError("Domain observation lacks its actual selected replay application")
            if any(component.applied_at != application.apply_at for component in step.components):
                raise ValueError(
                    "Accepted domain application clocks differ from the actual selected replay application"
                )
            self._accepted_observations[ordinal] = ObservationInput(
                schema_version=1,
                step=step,
                coefficient=producers[step.receipt.coefficient_record_fingerprint],
                feedback=application.feedback,
                expert_seconds=seconds,
            )

    def command(self, phase: str, ordinal: int | None, at: datetime | None) -> CommitRequest:
        """Create a complete CAS request before initialization or emission.

        Parameters
        ----------
        phase : {'initialize', 'emit'}
            Caller-selected command kind. Use observation_command for full outcomes.
        ordinal : int or None
            Logical round zero through two; null only for initialization.
        at : datetime or None
            Original zoned logical emission tick; null only for initialization. This
            is not a capture of the later physical callback completion time.

        Returns
        -------
        CommitRequest
            Actual expected accepted head and immutable next-generation intent.

        Raises
        ------
        ValueError, OSError
            Invalid metadata or failed bounded accepted-head acquisition.

        Notes
        -----
        Reading the head neither runs a callback nor advances learning. Retain this
        exact request to reconcile a lost response without inventing a new command."""
        head = self.store.read_head()
        return CommitRequest(
            schema_version=1,
            expected=head,
            intent=CommitIntent(
                schema_version=1,
                kind="initialize" if phase == "initialize" else "transition",
                generation=0 if head is None else head.generation + 1,
                content_fingerprint=fingerprint(
                    {
                        "phase": phase,
                        "ordinal": ordinal,
                        "at": None if at is None else at.isoformat(),
                        "BASE": self.recipe.base_fingerprint,
                        "world": self.recipe.world,
                    }
                ),
            ),
        )

    def emit(self, ordinal: int, at: datetime, *, request: CommitRequest | None = None) -> PrequentialStep:
        """Acknowledge actual coefficient and separate offline domain decisions.

        Parameters
        ----------
        ordinal : int
            Logical round zero through two in this independent recipe/world.
        at : datetime
            Original common UTC decision/cutoff tick, after earlier source application.
        request : CommitRequest or None, optional
            Original exact CAS request on retry; null constructs a new request.

        Returns
        -------
        PrequentialStep
            Complete actual unchanged coefficient record and feasible domain receipt.
            Decision checkpoints contain no not-yet-received outcome values.

        Raises
        ------
        ValueError, OSError
            Wrong clocks/profile/content, stale conflicting request, domain failure,
            incompatible state or publication failure; no false acknowledgement.

        Notes
        -----
        Mutates fresh staged owners, then publishes both under the existing SQL ACK.
        New ACKs may return immutable callback outputs; explicit retained requests
        load actual CURRENT accepted states and never rewind to an old receipt."""
        if request is None and ordinal in self._accepted_emissions:
            retained = self._accepted_emissions[ordinal][0]
            if decision_timing(retained.coefficient.opportunity.request).decision_at != at:
                raise ValueError("Logical replay retry changes its original source clock")
            return retained
        offered = self.command("emit", ordinal, at) if request is None else request
        expected_content = fingerprint(
            {
                "phase": "emit",
                "ordinal": ordinal,
                "at": at.isoformat(),
                "BASE": self.recipe.base_fingerprint,
                "world": self.recipe.world,
            }
        )
        if offered.intent.content_fingerprint != expected_content:
            raise ValueError("Emit request differs from its original complete logical input")
        produced: list[tuple[PrequentialStep, float, float]] = []

        def transition(stage):
            """Stage original replay/domain receipts and verify joint counters before ACK."""
            replay = stage.owner(self.replay_binding)
            domain = stage.owner(self.domain_binding)
            opportunity = replay.opportunity(ordinal, at)
            result = execute_prequential_step(
                replay.session,
                domain,
                opportunity,
                plan=DomainStepPlan(
                    schema_version=1,
                    key=execution_key(self.recipe.mode, self.recipe.world, ordinal),
                    adapter=MODEL.ADAPTER,
                    base_configuration_fingerprint=self.recipe.base_fingerprint,
                    data_fingerprint=self.recipe.data_fingerprint,
                ),
            )
            snapshot = domain.export_state()
            if (
                replay.session.state.decision_count != len(snapshot.receipts)
                or replay.session.state.update_count != snapshot.count
            ):
                raise ValueError("Staged coefficient/domain clocks disagree before acknowledgement")
            produced.append((result, snapshot.charges[ordinal], snapshot.movements[ordinal]))

        accepted = self.store.commit_transition(offered, self.bindings, transition, compatibility=self.environment)
        if accepted.status == "new":
            if len(produced) != 1 or accepted.accepted_head != accepted.current_head:
                raise RuntimeError("New acknowledgement differs from its staged immutable result")
            self._accepted_emissions[ordinal] = produced[0]
            return produced[0][0]
        if produced:
            raise RuntimeError("Retained acknowledgement unexpectedly invoked its transition")
        # Retained requests inspect actual current accepted state, never an old cache.
        replay_state, domain_state = self.snapshots()
        receipt = next(
            row
            for row in domain_state.receipts
            if row.key == execution_key(self.recipe.mode, self.recipe.world, ordinal)
        )
        coefficient = next(
            row for row in replay_state.state.decisions if row.fingerprint == receipt.coefficient_record_fingerprint
        )
        result = PrequentialStep(schema_version=1, coefficient=coefficient, domain=receipt)
        self._accepted_emissions[ordinal] = (result, domain_state.charges[ordinal], domain_state.movements[ordinal])
        return result

    def prepare_observation(
        self,
        ordinal: int,
        durations: tuple[float, float],
        evaluation_fingerprint: str,
        *,
        available_at: datetime | None = None,
        apply_at: datetime | None = None,
    ) -> ObservationInput:
        """Capture one complete original outcome without updating either owner.

        Parameters
        ----------
        ordinal : int
            Exact already emitted logical round zero through two.
        durations : tuple of float
            Full fast/full synthetic source panel, each in [0,8] modeled seconds.
        evaluation_fingerprint : str
            Original projected evaluation binding retained by produced raw records.
        available_at, apply_at : datetime or None, optional
            Actual supplied source-release and logical application ticks. Null captures
            current UTC for a NEW observation; reproduction supplies original records.

        Returns
        -------
        ObservationInput
            Complete original coefficient producer, expert feedback and separate five
            modeled/raw/diagnostic quantities with explicit units, origin and clocks.

        Raises
        ------
        ValueError, OSError
            Missing accepted producer, invalid values/units/clocks or incompatible state.

        Notes
        -----
        The output may be immutably retained before application. This reads accepted
        state or an already acknowledged immutable result; it changes no learner,
        forecast or accounting. Old clocks must not label newly captured outcomes."""
        emitted, charge, movement = self._retained_emission(ordinal)
        visible = datetime.now(UTC) if available_at is None else available_at
        source = self._source_record(
            ordinal,
            MODEL.modeled_components(emitted.domain, durations, charge, movement),
            evaluation_fingerprint,
            visible,
        )
        applied = datetime.now(UTC) if apply_at is None else apply_at
        timing = FeedbackTiming(
            event_at=decision_timing(emitted.domain.executed.proposal.request).target_end,
            available_at=visible,
            mature_at=visible,
            received_at=visible,
        )
        return ObservationInput(
            schema_version=1,
            step=StepEvidence(
                schema_version=1,
                receipt=emitted.domain,
                components=_observation_components(source, emitted.domain, timing, applied),
            ),
            coefficient=emitted.coefficient,
            feedback=MODEL.feedback(emitted.coefficient, self.recipe.config, durations, visible),
            expert_seconds=durations,
        )

    def _retained_emission(self, ordinal: int):
        """Load actual current accepted state only when the immutable emission is absent."""
        if ordinal not in self._accepted_emissions:
            replay_state, domain_state = self.snapshots()
            receipt = next(
                row
                for row in domain_state.receipts
                if row.key == execution_key(self.recipe.mode, self.recipe.world, ordinal)
            )
            record = next(
                row for row in replay_state.state.decisions if row.fingerprint == receipt.coefficient_record_fingerprint
            )
            self._accepted_emissions[ordinal] = (
                PrequentialStep(schema_version=1, coefficient=record, domain=receipt),
                domain_state.charges[ordinal],
                domain_state.movements[ordinal],
            )
        return self._accepted_emissions[ordinal]

    def _source_record(
        self, ordinal: int, values: MODEL.ModeledComponents, evaluation: str, visible: datetime
    ) -> DomainRecord:
        """Build explicit five-quantity evidence with an evaluation-scoped logical identity."""
        quantities = values.model_dump(mode="json")
        return DomainRecord(
            schema_version=1,
            namespace="constructed.prequential.ci",
            run_id="prequential-" + evaluation.removeprefix("sha256:"),
            execution_id=repetition_execution_id(self.recipe.mode, 0),
            decision_id=self.recipe.world if ordinal == 2 else self.recipe.world + "." + str(ordinal),
            data_fingerprint=self.recipe.data_fingerprint,
            available_at=visible,
            provenance=ResultProvenance(
                producer="public-prequential-ci-model",
                producer_version="1",
                spec_fingerprint=self.recipe.base_fingerprint,
                evaluation_fingerprint=evaluation,
            ),
            measurements=tuple(
                Measurement(
                    name=metric.name,
                    value=quantities[metric.name],
                    unit=metric.quantity.spec.semantics.unit,
                    meaning=metric.quantity.spec.semantics.meaning,
                    direction=metric.direction,
                    status="observed",
                )
                for metric in MODEL.metrics()
            ),
        )

    def observation_command(self, original: ObservationInput) -> CommitRequest:
        """Bind a CAS intent to every original feedback, quantity and clock byte.

        Parameters
        ----------
        original : ObservationInput
            Complete validated original source delivery, not only a net result.

        Returns
        -------
        CommitRequest
            Current expected head and next-generation full-payload identity.

        Raises
        ------
        ValueError, OSError
            Invalid input, virgin/corrupt collection or failed accepted-head acquisition.

        Notes
        -----
        No application occurs. Preserve the exact request across ambiguous responses;
        changing an unchosen expert's feedback is still a different logical payload."""
        checked = ObservationInput.model_validate(original)
        head = self.store.read_head()
        if head is None:
            raise ValueError("Observation requires an initialized state collection")
        return CommitRequest(
            schema_version=1,
            expected=head,
            intent=CommitIntent(
                schema_version=1,
                kind="transition",
                generation=head.generation + 1,
                content_fingerprint=checked.fingerprint,
            ),
        )

    def apply_original(self, original: ObservationInput, *, request: CommitRequest | None = None) -> StepEvidence:
        """Apply a complete retained source delivery once in both state owners.

        Parameters
        ----------
        original : ObservationInput
            Original expert panel, coefficient/domain producers and source/application
            clocks. Reproduction consumes these bytes rather than capturing new times.
        request : CommitRequest or None, optional
            Exact full-payload CAS request for a retry; null constructs a new request.

        Returns
        -------
        StepEvidence
            Actual accepted original domain evidence, including every quantity. An
            exact duplicate returns its retained original without another update/charge.

        Raises
        ------
        ValueError, OSError
            Conflicting full feedback, wrong recipe/clock/application, stale command or
            incompatible/corrupt state. Failed staging retains the old acknowledged head.

        Notes
        -----
        The real replay first selects and applies the original complete expert panel;
        only then does the separate domain owner update its past-only predictor. Both
        counters are checked before the existing joint ACK. No net scalar is substituted
        for the full expert losses; physical external execution is not performed."""
        original = ObservationInput.model_validate(original)
        if (
            original.step.receipt.key.cell.candidate_id != self.recipe.mode
            or original.step.receipt.key.cell.member_id != self.recipe.world
        ):
            raise ValueError("Original observation belongs to another independent recipe")
        ordinal = original.step.receipt.executed.proposal.request.round
        if request is None and ordinal in self._accepted_observations:
            retained = self._accepted_observations[ordinal]
            if retained.to_json() != original.to_json():
                raise ValueError("Logical observation retry differs from its complete retained original payload")
            return retained.step
        offered = self.observation_command(original) if request is None else request
        if offered.intent.content_fingerprint != original.fingerprint:
            raise ValueError("Observation request differs from its complete original payload")
        component = primary_component(original.step)
        if component.applied_at is None or component.timing.usable_at is None:
            raise ValueError("Original application requires known eligible observation clocks")
        produced: list[StepEvidence] = []

        def transition(stage):
            """Stage selected feedback and verify exact original domain retention before ACK."""
            self._apply_in_stage(stage, original, component)
            produced.append(original.step)

        accepted = self.store.commit_transition(offered, self.bindings, transition, compatibility=self.environment)
        return self._complete_observation(accepted, produced, original, ordinal)

    def _apply_in_stage(self, stage, original, component) -> None:
        """Apply actual replay feedback, then validate the domain successor before ACK."""
        replay = stage.owner(self.replay_binding)
        domain = stage.owner(self.domain_binding)
        replay.session.admit_feedback((original.feedback,))
        replay.session.advance(eligible_through=component.timing.usable_at, apply_at=component.applied_at)
        updates = replay.session.state.updates
        feedback_fingerprint = original.feedback.fingerprint if updates else None
        applications = tuple(
            row.application for row in updates if row.application.feedback.fingerprint == feedback_fingerprint
        )
        if len(applications) != 1 or applications[0].apply_at != component.applied_at:
            raise ValueError("Domain observation requires its actual selected replay application")
        snapshot = domain.observe(original.step, original.expert_seconds)
        self._require_applied_domain(replay, snapshot, original)

    def _require_applied_domain(self, replay, snapshot, original) -> None:
        """Require joint counters and exact retained original bytes in the staged owners."""
        if (
            replay.session.state.decision_count != len(snapshot.receipts)
            or replay.session.state.update_count != snapshot.count
        ):
            raise ValueError("Staged replay/domain observations disagree before acknowledgement")
        retained = tuple(row for row in snapshot.observations if row.receipt.key == original.step.receipt.key)
        if len(retained) != 1 or retained[0].to_json() != original.step.to_json():
            raise ValueError("Staged domain observation differs from the exact original applied evidence")
        if snapshot.expert_history[snapshot.observations.index(retained[0])] != original.expert_seconds:
            raise ValueError("Staged domain raw expert history differs from the actual original full panel")

    def _complete_observation(self, accepted, produced, original, ordinal) -> StepEvidence:
        """Handle actual new versus retained ACK without replaying or rewinding state."""
        if accepted.status == "new":
            if len(produced) != 1 or accepted.accepted_head != accepted.current_head:
                raise RuntimeError("Observation acknowledgement differs from its immutable staged result")
            self._accepted_observations[ordinal] = original
            return produced[0]
        if produced:
            raise RuntimeError("Retained observation unexpectedly invoked its callback")
        _, state = self.snapshots()
        return next(row for row in state.observations if row.receipt.key == original.step.receipt.key)

    def reproduce(self, originals: tuple[ObservationInput, ...]):
        """Replay original source clocks/values without calling them new observations.

        Parameters
        ----------
        originals : tuple of ObservationInput
            Actual retained receipts, complete original expert feedback and exact
            processing clocks from this same BASE recipe/world. The caller
            records reproduction/import exposure before reading this tape.

        Returns
        -------
        tuple
            Actual complete acknowledged replay and domain snapshots. New
            physical invocation metadata need not equal the earlier execution.

        Raises
        ------
        ValueError
            A source clock, producer, law or complete deterministic action
            differs. This method captures no replacement wall-clock timestamp.
        """
        for original in originals:
            checked = ObservationInput.model_validate(original)
            ordinal = checked.step.receipt.executed.proposal.request.round
            at = decision_timing(checked.step.receipt.executed.proposal.request).decision_at
            emitted = self.emit(ordinal, at)
            if (
                emitted.domain.fingerprint != checked.step.receipt.fingerprint
                or emitted.coefficient.to_json() != checked.coefficient.to_json()
            ):
                raise ValueError("Same-tape domain execution differs from its complete original receipt")
            self.apply_original(checked)
        return self.snapshots()


def _observation_components(source: DomainRecord, receipt, timing: FeedbackTiming, applied: datetime):
    """Bind every original modeled quantity to its exact producer and supplied clocks."""
    original = OriginalRecordEvidence.from_record(source)
    metrics = MODEL.metrics()
    if not metrics:
        return ()
    execution_fingerprint = receipt.fingerprint
    return tuple(
        ComponentObservation(
            schema_version=1,
            execution_fingerprint=execution_fingerprint,
            metric=metric,
            value=MODEL.OutcomeValue(
                spec=metric.quantity.spec,
                value=source.measurement(metric.name).value,
                status="observed",
                origin="synthetic" if metric.name in {"gross_seconds", "coefficient_l1"} else "modeled_counterfactual",
                transform=None,
            ),
            original=original,
            measurement_name=metric.name,
            timing=timing,
            applied_at=applied,
            physical_group_id=None,
        )
        for metric in metrics
    )
