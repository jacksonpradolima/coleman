"""Direct fixed-catalogue learners with atomic, complete in-memory continuation."""

from __future__ import annotations

import math
import platform
import sys
from random import Random
from threading import RLock

from coleman.core.actions import ChoiceAction
from coleman.core.capabilities import Capability, validate_decision, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import ChosenFeedback, FeedbackBatch
from coleman.core.identity import Reference
from coleman.core.sampling import SamplingRecord
from coleman.learning.discrete.contracts import (
    GRID_SIZE,
    DiscreteConfig,
    DiscreteState,
    NumericalRuntime,
    PendingChoice,
)
from coleman.learning.discrete.sampling import exp3_tickets, ticket_index
from coleman.state.random_state import capture_random, restore_random


def _runtime() -> NumericalRuntime:
    """Describe the compatible local interpreter and numerical host envelope."""
    host = platform.uname()
    return NumericalRuntime(
        python=sys.version.replace("\n", " "),
        implementation=sys.implementation.name,
        platform=sys.platform,
        machine=host.machine,
        os_release=host.release,
        os_version=host.version,
    )


def _positive_add(total: float, increment: float) -> float:
    """Refuse a positive observed quantity absorbed by binary64 accumulation."""
    updated = total + increment
    if increment > 0 and updated == total:
        raise ValueError("Positive reward or log-weight increment is absorbed by binary64 accumulation")
    return updated


def _ucb_index(state: DiscreteState) -> int:
    """Choose unobserved arms first, then UCB1 indices with canonical ties."""
    if 0 in state.counts:
        return state.counts.index(0)
    clock = sum(state.counts) + 1
    scores = tuple(
        reward / count + math.sqrt(2.0 * math.log(clock) / count)
        for reward, count in zip(state.rewards, state.counts, strict=True)
    )
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def _tickets(state: DiscreteState) -> tuple[int, ...]:
    """Return the actual post-policy finite distribution before feedback exists."""
    if state.config.gamma is not None:
        return exp3_tickets(state.log_weights, state.config.gamma)
    chosen = _ucb_index(state)
    return tuple(GRID_SIZE if index == chosen else 0 for index in range(len(state.counts)))


class DiscreteBandit:
    """Own a shipped Exp3 or UCB1 profile without runner or delivery machinery.

    Parameters
    ----------
    config : DiscreteConfig
        Versioned fixed-catalogue scientific and numerical contract.
    seed : int, optional
        Fresh owned Exp3 MT19937 initialization. Exported state, rather than the
        seed, is required for continuation. UCB1 ignores this initialization.

    Raises
    ------
    ValueError
        Invalid configuration or seed type.

    Notes
    -----
    In-memory calls are serialized per instance. Candidate state is validated
    before replacement, including RNG consumption; failed calls leave it intact.
    Raw updates consume one pending producer and reject repeats. External session
    delivery must supply its own durable acknowledgement/deduplication; this is
    no exactly-once physical execution service. Instances share no mutable state.
    """

    def __init__(self, config: DiscreteConfig, seed: int = 19):
        """Initialize independent, fully owned statistics and the declared stream."""
        checked = DiscreteConfig.model_validate(config)
        # Reject bool and conversion-bearing int subclasses at the seed boundary.
        # pylint: disable-next=unidiomatic-typecheck
        if type(seed) is not int:
            raise ValueError("Seed must be a built-in integer")
        size = len(checked.space.members)
        self._lock = RLock()
        self._state = DiscreteState(
            schema_version=1,
            config=checked,
            runtime=_runtime(),
            counts=(0,) * size,
            rewards=(0.0,) * size,
            log_weights=(0.0,) * size,
            generator=capture_random(Random(seed)) if checked.gamma is not None else None,
        )

    @property
    def capability(self) -> Capability:
        """Declare genuine choice, chosen feedback and exact mechanism mass.

        Returns
        -------
        Capability
            Immediate untimed, observed-only learner-owned support. Probability
            is selection mass conditional on the randomized mechanism, never an
            outcome forecast or statistical independence certificate.
        """
        config = self._state.config
        return Capability(
            schema_version=1,
            implementation=config.implementation,
            action_kinds=("choice",),
            feedback_kinds=("chosen",),
            required_values=(config.reward,),
            origins=config.origins,
            timing_modes=("untimed",),
            supports_delayed_feedback=False,
            requires_known_receipt=False,
            requires_observed=True,
            state_owner="learner",
            requires_sampling=True,
            sampling_requirement="exact",
        )

    @property
    def pending_execution(self) -> ExecutedDecision | None:
        """Return the original unchanged execution awaiting one observation.

        Returns
        -------
        ExecutedDecision or None
            Immutable producer binding for a chosen-only FeedbackBatch. The host
            must actually use this action; mapping or overriding it is unsupported.
        """
        with self._lock:
            return self._state.pending.executed if self._state.pending is not None else None

    def _request(self, request: DecisionRequest, state: DiscreteState) -> None:
        """Reject changed universes, context, clocks and skipped logical rounds."""
        config = state.config
        if request.task != config.task or request.objective != config.objective or request.space != config.space:
            raise ValueError("Request must retain the exact configured task, objective and catalogue")
        if request.timing is not None or request.inputs or request.member_inputs or request.signals:
            raise ValueError("Fixed discrete profiles do not support timing or contextual inputs")
        if request.round != sum(state.counts):
            raise ValueError("Synchronous request round must equal actual completed observations")
        if request.round >= config.maximum_updates:
            raise ValueError("Declared numerical horizon is exhausted")

    def _pending(self, request: DecisionRequest, tickets: tuple[int, ...], draw: int | None) -> PendingChoice:
        """Bind one actual selection to its complete immutable pre-feedback mass."""
        index = ticket_index(tickets, 0 if draw is None else draw)
        config = self._state.config
        action = ChoiceAction(schema_version=1, space=config.space, chosen=config.space.members[index])
        sampling = SamplingRecord(
            mechanism="deterministic" if draw is None else "sampled",
            mechanism_ref=Reference(id=config.profile, version="v1"),
            status="exact",
            action_fingerprint=action.fingerprint,
            probability=tickets[index] / GRID_SIZE,
        )
        proposal = Decision(
            schema_version=1,
            request=request,
            implementation=config.implementation,
            action=action,
            sampling=sampling,
            state_owner="learner",
        )
        validate_decision(proposal, self.capability)
        return PendingChoice(
            executed=ExecutedDecision(
                schema_version=1,
                proposal=proposal,
                action=action,
                mapping="unchanged",
                sampling_disposition="preserved",
                sampling=sampling,
            ),
            tickets=tickets,
            draw=draw,
        )

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Sample before feedback and atomically retain complete producer/RNG state.

        Parameters
        ----------
        request : DecisionRequest
            Fixed untimed catalogue and next zero-based logical observation round.

        Returns
        -------
        Decision
            Actual choice and exact selection mass. An identical pending request
            returns identical bytes without consuming another draw.

        Raises
        ------
        ValueError
            Unsupported request, conflicting outstanding producer, exhausted
            horizon or numerical support failure. No state is committed on error.
        """
        request = DecisionRequest.model_validate(request)
        with self._lock:
            state = self._state
            self._request(request, state)
            if state.pending is not None:
                if state.pending.executed.proposal.request.to_json() == request.to_json():
                    return state.pending.executed.proposal
                raise ValueError("Only one original producer may await feedback")
            tickets = _tickets(state)
            generator = state.generator
            draw = None
            if generator is not None:
                owned = Random(0)
                restore_random(owned, generator)
                draw = owned.getrandbits(53)
                generator = capture_random(owned)
            pending = self._pending(request, tickets, draw)
            candidate = DiscreteState.model_validate(
                state.model_copy(update={"pending": pending, "generator": generator})
            )
            self._state = candidate
            return pending.executed.proposal

    def update(self, batch: FeedbackBatch, /) -> None:
        """Commit only the original chosen observation, using its effective mass.

        Parameters
        ----------
        batch : FeedbackBatch
            Immediate root chosen-feedback batch bound to pending_execution, with
            the exact declared reward, units, origin and transform; horizon null.

        Raises
        ------
        ValueError
            Missing/conflicting producer, unsupported feedback or transform,
            nonobserved/nonfinite reward, or numerical-envelope exhaustion.

        Notes
        -----
        Exp3 estimates only the chosen reward as x/q and increases only its log
        weight by gamma*x/(K*q), then centers all logs by their maximum. Centering
        preserves weight ratios. UCB1 increments only the actual chosen count.
        No alternative outcome is accepted; zero is a genuine observation.
        Repeated direct delivery rejects after consumption without another update.
        """
        batch = FeedbackBatch.model_validate(batch)
        with self._lock:
            state = self._state
            if state.pending is None:
                raise ValueError("No original chosen producer awaits feedback")
            validate_feedback(state.pending.executed, batch, self.capability)
            if (
                not isinstance(batch.feedback, ChosenFeedback)
                or batch.supersedes is not None
                or batch.horizon is not None
                or batch.timing is not None
            ):
                raise ValueError("Only immediate untimed root chosen feedback with null horizon is supported")
            value = next(value for value in batch.feedback.row.values if value.spec == state.config.reward)
            reward = value.value
            if value.transform != state.config.transform or reward is None:
                raise ValueError("Reward requires its exact declared transform and observed value")
            index = state.config.space.members.index(batch.feedback.row.member)
            counts = list(state.counts)
            rewards = list(state.rewards)
            logs = list(state.log_weights)
            counts[index] += 1
            rewards[index] = _positive_add(rewards[index], reward)
            if state.config.gamma is not None:
                probability = state.pending.tickets[index] / GRID_SIZE
                increment = (state.config.gamma / len(counts)) * (reward / probability)
                if reward > 0 and increment == 0:
                    raise ValueError("Positive chosen update underflows the binary64 numerical envelope")
                logs[index] = _positive_add(logs[index], increment)
                center = max(logs)
                logs = [value - center for value in logs]
                if reward > 0 and len(counts) > 1 and tuple(logs) == state.log_weights:
                    raise ValueError("Positive chosen update is absorbed after log-weight centering")
            candidate = DiscreteState.model_validate(
                state.model_copy(
                    update={
                        "counts": tuple(counts),
                        "rewards": tuple(rewards),
                        "log_weights": tuple(logs),
                        "pending": None,
                    }
                )
            )
            self._state = candidate

    def export_state(self) -> DiscreteState:
        """Export complete immutable continuation state without drawing or I/O.

        Returns
        -------
        DiscreteState
            Canonical schema1-compatible statistics, configuration, owned RNG and
            pending original producer. No artifact completion is advertised.
        """
        with self._lock:
            return DiscreteState.model_validate(self._state)

    def restore_state(self, state: DiscreteState, /) -> None:
        """Validate full configuration/runtime/producer before atomic replacement.

        Parameters
        ----------
        state : DiscreteState
            Concrete schema1 snapshot, including pending state after a draw.

        Raises
        ------
        ValueError
            Invalid state, incompatible configuration/runtime or inconsistent
            original sampler/proposal. Failed restoration leaves live state intact.

        Notes
        -----
        Explicit restore rewinds in-memory mathematics. It neither rolls back
        external work nor acknowledges delivery. Older legacy ranking snapshots
        cannot resume these distinct methods and are never silently migrated.
        """
        candidate = DiscreteState.model_validate(state)
        with self._lock:
            if candidate.config.to_json() != self._state.config.to_json() or candidate.runtime != _runtime():
                raise ValueError("Snapshot configuration or Python RNG runtime is incompatible")
            if candidate.pending is not None:
                pending = candidate.pending
                self._request(pending.executed.proposal.request, candidate)
                if (candidate.config.gamma is not None) != (pending.draw is not None):
                    raise ValueError("Pending draw does not match the configured sampling mechanism")
                tickets = _tickets(candidate)
                expected = self._pending(pending.executed.proposal.request, tickets, pending.draw)
                if expected != pending:
                    raise ValueError("Pending producer does not match its original policy distribution")
            self._state = candidate
