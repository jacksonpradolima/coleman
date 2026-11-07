"""Thin optional ReplayApplication adapter, leaving all mathematics independent."""

from __future__ import annotations

from typing import Protocol

from coleman.core.base import CoreContract
from coleman.core.capabilities import Capability
from coleman.core.decisions import Decision, DecisionRequest
from coleman.core.feedback import FeedbackBatch
from coleman.core.temporal import Timestamp
from coleman.replay.contracts import ReplayApplication
from coleman.state.contracts import StateBinding


class MixtureApplication[S: CoreContract](Protocol):
    """Describe one complete typed direct/scheduled application, without stores."""

    @property
    def capability(self) -> Capability:
        """Declare the complete original-producer feedback capability."""
        ...

    @property
    def binding(self) -> StateBinding:
        """Bind the actual complete original-producer and mathematical state."""
        ...

    @property
    def state_type(self) -> type[S]:
        """Supply a trusted concrete codec, never imported from artifacts."""
        ...

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Apply one original decision request at its declared clocks.

        Parameters
        ----------
        request : DecisionRequest
            Complete original typed opportunity.

        Returns
        -------
        Decision
            Actual deterministic coefficient or explicit abstention proposal.
        """
        ...

    def update(
        self, feedback: FeedbackBatch, /, *, eligible_at: Timestamp | None = None, apply_at: Timestamp | None = None
    ) -> object:
        """Apply a full panel at the supplied actual processing instant.

        Parameters
        ----------
        feedback : FeedbackBatch
            Selected complete original-producer panel.
        eligible_at, apply_at : datetime or None, optional
            Explicit eligibility and actual application clocks.

        Returns
        -------
        object
            Actual method's typed mathematical disposition.
        """
        ...

    def export_state(self) -> S:
        """Return complete immutable actual application state."""
        ...

    def restore_state(self, state: S, /) -> None:
        """Restore only explicitly compatible complete application state.

        Parameters
        ----------
        state : CoreContract
            Concrete complete versioned application continuation.
        """
        ...


class ReplayMixture[S: CoreContract]:
    """Adapt an actual direct mixture to the existing causal replay protocol.

    Parameters
    ----------
    application : MixtureApplication
        Actual separately owned direct/scheduled stateful learner application.

    Notes
    -----
    No clocks, corrections, delivery acknowledgements, schedulers or stores are
    invented. The enclosing existing replay chooses the revision and supplies
    apply_at; this adapter unwraps only that validated application envelope.
    Its wrapper is an additional actual live root in an ownership inventory.
    """

    def __init__(self, application: MixtureApplication[S]):
        """Retain the actual exclusively owned application without invoking it."""
        self.application = application

    @property
    def capability(self) -> Capability:
        """Return the actual application's declared full-panel capability."""
        return self.application.capability

    @property
    def binding(self) -> StateBinding:
        """Return the unchanged complete application/codec ownership binding."""
        return self.application.binding

    @property
    def state_type(self) -> type[S]:
        """Return the trusted concrete complete application snapshot class."""
        return self.application.state_type

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Delegate the captured post-drain request without another release call.

        Parameters
        ----------
        request : DecisionRequest
            Actual closed-drain opportunity supplied by existing causal replay.

        Returns
        -------
        Decision
            Actual application proposal, with original identity and clocks.
        """
        return self.application.decide(request)

    def update(self, feedback: ReplayApplication, /) -> object:
        """Unwrap the selected original batch and explicit actual replay clocks.

        Parameters
        ----------
        feedback : ReplayApplication
            Complete selected revision, actual execution and processing clocks.

        Returns
        -------
        object
            Actual method's typed disposition, not an external acknowledgement.

        Raises
        ------
        ValueError, TypeError
            Malformed application, unsupported panel/units or original join.
        Exception
            Actual host failures propagate; the enclosing replay invalidates
            its session without pretending to roll back arbitrary host effects.
        """
        checked = ReplayApplication.model_validate(feedback)
        return self.application.update(
            checked.feedback.batch, eligible_at=checked.eligible_at, apply_at=checked.apply_at
        )

    def export_state(self) -> S:
        """Export the actual complete application, without a second state cache."""
        return self.application.export_state()

    def restore_state(self, state: S, /) -> None:
        """Restore through the actual explicitly compatible application codec.

        Parameters
        ----------
        state : CoreContract
            Concrete complete compatible application continuation.

        Raises
        ------
        ValueError, TypeError
            Incompatible version, owner, fixed law, schedule or pending content.
        """
        self.application.restore_state(state)
