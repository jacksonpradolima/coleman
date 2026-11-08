"""Optional typed decision/feedback bridge around independently usable learners.

This bridge owns bounded pending producer joins. It supplies no scheduler,
durable delivery, experiment service or implicit source/correction selection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Annotated, Protocol, Self, cast

from pydantic import Field, model_validator

from coleman.core.actions import Action, MixtureAction, MixtureSpace, Weight
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.capabilities import Capability, validate_decision, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch, FullPanel
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.reading import bounded_json_bytes
from coleman.core.sampling import SamplingRecord
from coleman.core.temporal import Timestamp
from coleman.learning.contracts import MAXIMUM_BYTES, _BoundedContract, read_loss_panel
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.contracts import AlgorithmState, StateBinding

Count = Annotated[int, Field(strict=True, ge=0, le=1048576)]
DIRECT_CODEC = Reference(id="direct-mixture-state", version="1")


class MixtureLearner[P: CoreContract](Protocol):
    """Describe the pure full-information state used by this optional bridge.

    Notes
    -----
    Shipped reference and controlled methods satisfy this structurally. The
    actual mathematical operation is independently consumable without this
    bridge. Inputs/outputs are coefficients, not selection probabilities.
    """

    @property
    def binding(self) -> StateBinding:
        """Return the exact immutable owner/implementation/configuration binding."""
        ...

    @property
    def members(self) -> tuple[Reference, ...]:
        """Return every fixed expert identity/version in canonical order."""
        ...

    @property
    def loss_spec(self) -> ScalarSpec:
        """Return the complete bounded dimensionless loss interpretation."""
        ...

    @property
    def transform(self) -> DefinitionBinding:
        """Return the pinned transform definition, without fitting it."""
        ...

    @property
    def weights(self) -> tuple[Weight, ...]:
        """Return actual currently emitted coefficient fractions."""
        ...

    @property
    def update_count(self) -> int:
        """Return accepted complete mathematical panels U, excluding decisions."""
        ...

    def observe(self, panel: FullPanel, /) -> object:
        """Admit one complete panel under the method's explicit failure policy.

        Parameters
        ----------
        panel : FullPanel
            Complete fixed expert vector with observed bounded losses.

        Returns
        -------
        object
            The method's typed update/hold/release/failure disposition.
        """
        ...

    def export_state(self) -> AlgorithmState[P]:
        """Return complete immutable sufficient state without I/O.

        Returns
        -------
        AlgorithmState
            Exact bound concrete snapshot, including pending mathematical work.
        """
        ...

    def restore_state(self, state: AlgorithmState[P], /) -> None:
        """Restore explicitly compatible complete mathematical state.

        Parameters
        ----------
        state : AlgorithmState
            Complete owner/configuration/profile-compatible snapshot.
        """
        ...


class DirectOptions(CoreImmutable):
    """Bind one task/objective/horizon and finite pending-producer limits.

    Attributes
    ----------
    task, objective : Reference
        Exact logical task and interpretation; no cross-task pooling is implied.
    horizon : Reference or None
        Exact expected feedback horizon, or explicit null for untimed algebra.
    maximum_pending : int
        One through 4096 complete unconsumed producers.
    maximum_bytes : int
        Complete bridge snapshot ceiling, at most 16 MiB.
    """

    task: Reference
    objective: Reference
    horizon: Reference | None
    maximum_pending: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096
    maximum_bytes: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_BYTES)] = MAXIMUM_BYTES


def _bridge_binding(method: StateBinding, options: DirectOptions) -> StateBinding:
    """Bind the complete fixed mathematical interpretation and join options."""
    return StateBinding(
        owner=method.owner,
        implementation=method.implementation,
        state_codec=DIRECT_CODEC,
        configuration_fingerprint=fingerprint(
            {"method": method.model_dump(mode="json"), "options": options.model_dump(mode="json")}
        ),
    )


class DirectMixtureState[P: CoreContract](_BoundedContract):
    """Capture the complete direct producer bridge and mathematical state.

    Attributes
    ----------
    binding : StateBinding
        Exact owner, method, bridge codec and complete configuration.
    options : DirectOptions
        Original task/objective/horizon and bounded admission choices.
    method : AlgorithmState
        Concrete complete learner snapshot, not only emitted coefficients.
    decision_count, update_count : int
        Bridge decisions D and observed completed producer panels U.
    initial_updates : int
        Mathematical U already present when the bridge was constructed.
    pending : tuple of ExecutedDecision
        Original offline-identity producers awaiting their one admitted panel.

    Notes
    -----
    Completed producer IDs are not a durable receipt inventory. Session/store
    deduplication and correction selection remain separate. Counts attest the
    supplied prefix; no discarded historical observations are authenticated.
    A generic payload need not expose its mathematical update counter as a
    top-level field. Its concrete reader owns those invariants; the adapter
    checks the actual learner counter on export and after trusted restoration.
    The shared strict version-one reader bounds complete UTF-8 content to
    16 MiB and lexical depth 64, checking bytearray length before copying.
    The selected options may impose a smaller complete-snapshot byte ceiling;
    both the hard reader envelope and that selected DTO limit apply.
    """

    binding: StateBinding
    options: DirectOptions
    method: AlgorithmState[P]
    decision_count: Count
    update_count: Count
    initial_updates: Count
    pending: Annotated[tuple[ExecutedDecision, ...], Field(max_length=4096)]

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Bind complete counters, original pending producers and fixed options."""
        if self.binding != _bridge_binding(self.method.binding, self.options):
            raise ValueError("Direct state differs from its complete method/options binding")
        if self.decision_count != self.update_count + len(self.pending):
            raise ValueError("Direct decisions must be completed or retained pending exactly once")
        if self.initial_updates + self.update_count > 1048576:
            raise ValueError("Direct mathematical update count exceeds its hard clock ceiling")
        if len(self.pending) > self.options.maximum_pending:
            raise ValueError("Direct pending producers exceed the selected capacity")
        identities = tuple(item.logical_identity for item in self.pending)
        if len(identities) != len(set(identities)):
            raise ValueError("Direct state cannot retain duplicate pending logical producers")
        self._pending_producers()
        if len(canonical_json(self.model_dump(mode="json")).encode("utf-8")) > self.options.maximum_bytes:
            raise ValueError("Direct complete state exceeds its selected byte capacity")
        return self

    def _pending_producers(self) -> None:
        """Keep the original producer profile without rewriting historical weights."""
        for item in self.pending:
            if (
                item.proposal.request.task != self.options.task
                or item.proposal.request.objective != self.options.objective
            ):
                raise ValueError("Direct state contains a foreign task/objective producer")
            if item.proposal.implementation != self.binding.implementation.reference:
                raise ValueError("Direct state contains another implementation's producer")
            if item.proposal.state_owner != "learner":
                raise ValueError("Direct pending state ownership must remain learner")
            if item.mapping != "unchanged":
                raise ValueError("Direct mixture state supports explicit offline unchanged execution only")


class LearningApplicationError(RuntimeError):
    """Refuse to acknowledge a panel that the mathematical learner did not accept."""


@dataclass(frozen=True, slots=True)
class _DirectFrontier:
    """Replace complete local bookkeeping together after a successful operation."""

    initial_updates: int
    decisions: int
    updates: int
    pending: tuple[ExecutedDecision, ...]
    expected: str
    failed: bool = False


class DirectMixture[P: CoreContract]:
    """Bridge a real learner to typed direct decision/feedback protocols.

    Parameters
    ----------
    learner : MixtureLearner
        Independently owned actual mathematical learner, never a shared mutable
        instance. Its complete snapshot is retained inside this bridge state.
    task, objective : Reference
        Exact permitted request task/objective. No registration is required.
    horizon : Reference or None, optional
        Expected batch horizon; default None is the untimed mathematical route.
    maximum_pending : int, optional
        Complete pending producer limit, default/hard 4096.
    maximum_bytes : int, optional
        Complete bridge snapshot byte ceiling, default/hard 16 MiB.

    Notes
    -----
    Decisions are explicit offline identity simulations, never physical test
    runs/trades. Pending exact retries return their original proposal. Once a
    panel is consumed, its duplicate is rejected; durable/revision delivery
    belongs to replay/store. External mutation of the owned learner invalidates
    this bridge. A host-call failure invalidates it without rollback; restore
    a compatible complete snapshot before continued use. No clock is read.
    Calls are sequential under one owner: immutable candidate replacement is
    not thread synchronization. Parallel work uses independent instances. The
    optional replay guard and owned-store CAS qualify their own concurrency.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        learner: MixtureLearner[P],
        *,
        task: Reference,
        objective: Reference,
        horizon: Reference | None = None,
        maximum_pending: int = 4096,
        maximum_bytes: int = MAXIMUM_BYTES,
    ):
        """Own one real learner under fixed, bounded producer-join options.

        Parameters
        ----------
        learner : MixtureLearner
            Independently owned complete mathematical state and direct API.
        task, objective : Reference
            Exact accepted request task and objective identities/versions.
        horizon : Reference or None, optional
            Expected feedback horizon; None retains untimed mathematics.
        maximum_pending : int, optional
            Selected complete pending-producer limit, at most 4096.
        maximum_bytes : int, optional
            Selected complete bridge-state byte limit, at most 16 MiB.

        Raises
        ------
        ValueError, TypeError
            Invalid options, state or insufficient selected capacity.
        Exception
            The actual learner's side-effect-free export fails.
        """
        self._method = learner
        self._options = DirectOptions(
            task=task,
            objective=objective,
            horizon=horizon,
            maximum_pending=maximum_pending,
            maximum_bytes=maximum_bytes,
        )
        snapshot = learner.export_state()
        payload_type = type(snapshot.payload)
        self._contract = cast(type[DirectMixtureState[P]], DirectMixtureState.__class_getitem__(payload_type))
        self._binding = _bridge_binding(learner.binding, self._options)
        self._frontier = _DirectFrontier(
            initial_updates=learner.update_count, decisions=0, updates=0, pending=(), expected=snapshot.fingerprint
        )
        self.export_state()

    @property
    def learner(self) -> MixtureLearner[P]:
        """Return the actual owned learner for inspection, not external updates.

        Returns
        -------
        MixtureLearner
            The real method. Mutating it outside this bridge breaks ownership.
        """
        return self._method

    @property
    def binding(self) -> StateBinding:
        """Return exact complete bridge codec/configuration ownership.

        Returns
        -------
        StateBinding
            Fixed task/horizon/method binding, without I/O.
        """
        return self._binding

    @property
    def state_type(self) -> type[DirectMixtureState[P]]:
        """Return the trusted concrete snapshot class for explicit reading.

        Returns
        -------
        type
            Parameterized complete state codec; not imported from an artifact.
        """
        return self._contract

    @property
    def capability(self) -> Capability:
        """Declare complete bounded feedback and coefficient-only actions.

        Returns
        -------
        Capability
            Direct untimed/timed complete-panel support, without a synchronous
            regret assertion for delayed decisions or sampling propensities.
        """
        return Capability(
            schema_version=1,
            implementation=self._binding.implementation.reference,
            action_kinds=("mixture",),
            feedback_kinds=("full_panel",),
            required_values=(self._method.loss_spec,),
            origins=("realized", "modeled_counterfactual", "synthetic"),
            timing_modes=("untimed", "future", "ongoing"),
            supports_delayed_feedback=True,
            requires_known_receipt=False,
            requires_observed=True,
            state_owner="learner",
            requires_sampling=False,
            sampling_requirement="not_required",
        )

    def _verify(self) -> AlgorithmState[P]:
        """Reject broken/external-mutated ownership before any new host operation."""
        if self._frontier.failed:
            raise LearningApplicationError("Direct mixture is invalidated; restore a complete compatible snapshot")
        state = self._method.export_state()
        if state.binding != self._method.binding or state.fingerprint != self._frontier.expected:
            self._frontier = replace(self._frontier, failed=True)
            raise LearningApplicationError("The exclusively owned mathematical state changed outside this bridge")
        if self._method.update_count != self._frontier.initial_updates + self._frontier.updates:
            self._frontier = replace(self._frontier, failed=True)
            raise LearningApplicationError("Direct and mathematical accepted-panel clocks disagree")
        return state

    @contextmanager
    def _operation(self) -> Iterator[None]:
        """Invalidate a possibly partial trusted mathematical host call, then raise."""
        try:
            yield
        except BaseException:
            self._frontier = replace(self._frontier, failed=True)
            raise

    def _snapshot(
        self, method: AlgorithmState[P], *, pending: tuple[ExecutedDecision, ...], decisions: int, updates: int
    ) -> DirectMixtureState[P]:
        """Construct complete bounded candidate state without assigning it.

        Notes
        -----
        The constructor revalidates all fields and the selected byte limit.
        Serialize only this fresh local DTO for the hard byte/depth guard;
        no host callback intervenes. Public writers still revalidate inputs.
        """
        if self._method.update_count != self._frontier.initial_updates + updates:
            raise ValueError("Direct snapshot and actual mathematical update clocks disagree")
        state = self._contract(
            schema_version=1,
            binding=self._binding,
            options=self._options,
            method=method,
            decision_count=decisions,
            update_count=updates,
            initial_updates=self._frontier.initial_updates,
            pending=pending,
        )
        content = bounded_json_bytes(canonical_json(state.model_dump(mode="json")), MAXIMUM_BYTES)
        if len(content) > self._options.maximum_bytes:
            raise ValueError("Direct complete state exceeds its selected byte capacity")
        return state

    def _request(self, request: DecisionRequest) -> DecisionRequest:
        """Preflight the actual task and fixed expert universe without a host call."""
        checked = DecisionRequest.model_validate(request)
        if checked.task != self._options.task or checked.objective != self._options.objective:
            raise ValueError("Direct request differs from the fixed task/objective")
        if not isinstance(checked.space, MixtureSpace) or checked.space.members != self._method.members:
            raise ValueError("Direct learner requires its complete fixed mixture catalogue and versions")
        return checked

    def _before_decision(self, request: DecisionRequest) -> None:
        """Let an explicit controlled subclass release inside a captured decision."""

    def _action(self, request: DecisionRequest) -> Action:
        """Build the complete actual coefficient action after any declared release."""
        if not isinstance(request.space, MixtureSpace):
            raise ValueError("Direct coefficient actions require a mixture space")
        return MixtureAction(schema_version=1, space=request.space, weights=self._method.weights)

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Return a typed coefficient proposal and retain its original producer.

        Parameters
        ----------
        request : DecisionRequest
            Exact fixed task/objective/catalogue and explicit optional timing.

        Returns
        -------
        Decision
            Actual learner coefficients with not-required sampling evidence.

        Raises
        ------
        ValueError
            Wrong schema/universe, logical conflict, exhausted pending/count/
            byte capacity. Preflight precedes the mathematical host call.
        LearningApplicationError
            Previously failed or externally mutated mathematical owner.
        Exception
            Trusted host failures propagate and invalidate this bridge.
        """
        before = self._verify()
        checked = self._request(request)
        for item in self._frontier.pending:
            if item.logical_identity == checked.logical_identity:
                if item.proposal.request.to_json() != checked.to_json():
                    raise ValueError("Conflicting direct request at one pending logical identity")
                return item.proposal
        if len(self._frontier.pending) >= self._options.maximum_pending or self._frontier.decisions >= 1048576:
            raise ValueError("Direct producer capacity exhausted")
        self._snapshot(
            before, pending=self._frontier.pending, decisions=self._frontier.decisions, updates=self._frontier.updates
        )
        with self._operation():
            self._before_decision(checked)
            action = self._action(checked)
            sampling = SamplingRecord(
                mechanism="deterministic",
                mechanism_ref=self._binding.implementation.reference,
                status="not_required",
                action_fingerprint=action.fingerprint,
                probability=None,
            )
            decision = Decision(
                schema_version=1,
                request=checked,
                implementation=self._binding.implementation.reference,
                action=action,
                sampling=sampling,
                state_owner="learner",
            )
            validate_decision(decision, self.capability)
            executed = ExecutedDecision(
                schema_version=1,
                proposal=decision,
                action=action,
                mapping="unchanged",
                sampling_disposition="preserved",
                sampling=sampling,
            )
            pending = (*self._frontier.pending, executed)
            state = self._method.export_state()
            self._snapshot(
                state, pending=pending, decisions=self._frontier.decisions + 1, updates=self._frontier.updates
            )
        self._frontier = replace(
            self._frontier, pending=pending, decisions=self._frontier.decisions + 1, expected=state.fingerprint
        )
        return decision

    def _observe(self, panel: FullPanel, *, eligible_at: Timestamp | None, apply_at: Timestamp | None) -> object:
        """Apply the pure method; controlled subclasses use supplied clocks only."""
        del eligible_at, apply_at
        return self._method.observe(panel)

    def update(
        self, feedback: FeedbackBatch, /, *, eligible_at: Timestamp | None = None, apply_at: Timestamp | None = None
    ) -> object:
        """Apply one complete original-producer panel, retaining no fake zeros.

        Parameters
        ----------
        feedback : FeedbackBatch
            Full observed bounded panel joined to an original pending execution.
            Direct callers own revision selection; replay supplies its selected
            complete batch rather than reusing a historical correction twice.
        eligible_at, apply_at : datetime or None, optional
            Explicit processing inputs for controlled subclasses. Neither is
            inferred from receipt; ordinary reference mathematics ignores them.

        Returns
        -------
        object
            The actual method's typed acceptance/hold/release disposition.

        Raises
        ------
        ValueError
            Unmatched/already consumed producer, incompatible horizon, units,
            transform, missing active loss or unsupported coverage.
        LearningApplicationError
            The method did not accept the panel or ownership became invalid.
        Exception
            Actual host errors propagate and invalidate this bridge; no rollback
            or durable exactly-once acknowledgement is supplied here.
        """
        self._verify()
        batch = FeedbackBatch.model_validate(feedback)
        matches = tuple(item for item in self._frontier.pending if item.fingerprint == batch.decision_fingerprint)
        if len(matches) != 1 or batch.horizon != self._options.horizon:
            raise ValueError("Direct feedback must match one original pending producer and exact horizon")
        executed = matches[0]
        validate_feedback(executed, batch, self.capability, apply_at=apply_at)
        if not isinstance(batch.feedback, FullPanel):
            raise ValueError("Direct expert learners require full-panel feedback")
        read_loss_panel(
            batch.feedback,
            members=self._method.members,
            loss_spec=self._method.loss_spec,
            transform=self._method.transform,
        )
        with self._operation():
            result = self._observe(batch.feedback, eligible_at=eligible_at, apply_at=apply_at)
            if self._method.update_count != self._frontier.initial_updates + self._frontier.updates + 1:
                raise LearningApplicationError("The mathematical method did not accept this complete panel")
            pending = tuple(item for item in self._frontier.pending if item is not executed)
            state = self._method.export_state()
            self._snapshot(
                state, pending=pending, decisions=self._frontier.decisions, updates=self._frontier.updates + 1
            )
        self._frontier = replace(
            self._frontier, pending=pending, updates=self._frontier.updates + 1, expected=state.fingerprint
        )
        return result

    def export_state(self) -> DirectMixtureState[P]:
        """Export complete actual bridge and mathematical state without I/O.

        Returns
        -------
        DirectMixtureState
            Concrete bounded schema1 state with original pending producers.

        Raises
        ------
        ValueError, LearningApplicationError
            Failed ownership, incompatible content or selected capacity.
        """
        return self._snapshot(
            self._verify(),
            pending=self._frontier.pending,
            decisions=self._frontier.decisions,
            updates=self._frontier.updates,
        )

    def restore_state(self, state: DirectMixtureState[P], /) -> None:
        """Restore a complete compatible bridge and independently owned learner.

        Parameters
        ----------
        state : DirectMixtureState
            Exact owner/method/task/objective/horizon/options snapshot.

        Raises
        ------
        ValueError
            Unknown schema, wrong concrete codec, binding/options or counters.
        Exception
            Actual mathematical restore/export failures invalidate this bridge.

        Notes
        -----
        Typed compatibility is checked before the trusted learner restore.
        Every retained producer must match the fixed task, expert versions and
        capability; its historical coefficients need not equal today's vector.
        A generic payload need not expose U in a particular field. Its concrete
        reader owns internal counter invariants, and the actual learner's U is
        checked after restore. A mismatch invalidates this bridge without a
        rollback claim, just like another trusted host-call failure.
        Host failures do not roll back arbitrary external effects. A successful
        complete restore replaces this bridge's pending and counter frontiers.
        """
        checked = self._contract.model_validate(state)
        if checked.binding != self._binding or checked.options.model_dump_json() != self._options.model_dump_json():
            raise ValueError("Direct snapshot differs from this exact owner/options binding")
        if checked.method.binding != self._method.binding:
            raise ValueError("Direct snapshot belongs to another mathematical owner or configuration")
        for pending in checked.pending:
            self._request(pending.proposal.request)
            validate_decision(pending.proposal, self.capability)
        with self._operation():
            self._method.restore_state(checked.method)
            current = self._method.export_state()
            if current.to_json() != checked.method.to_json():
                raise ValueError("Restored mathematical snapshot differs from the complete original")
            if self._method.update_count != checked.initial_updates + checked.update_count:
                raise ValueError("Restored direct/mathematical update clocks disagree")
        self._frontier = _DirectFrontier(
            initial_updates=checked.initial_updates,
            decisions=checked.decision_count,
            updates=checked.update_count,
            pending=checked.pending,
            expected=current.fingerprint,
        )
