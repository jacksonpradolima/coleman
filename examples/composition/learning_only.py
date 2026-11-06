"""A tiny user-owned full-information learner; no shipped v5 method is implied.

Run this copied file with the installed minimal wheel. It learns only from
explicit complete synthetic CI loss panels. It is an integration example,
without a regret, efficacy, calibration or independent-observation claim.
"""

import json
from typing import Annotated, cast

from pydantic import Field, model_validator

from coleman.core.actions import ChoiceAction, ChoiceSpace
from coleman.core.base import CoreContract
from coleman.core.capabilities import Capability, ProblemCapabilities, validate_decision, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.core.units import UnitSemantics
from coleman.testing.decision_contracts import ConformanceCase, check_decision_contracts


def reference(name: str) -> Reference:
    """Return one public synthetic v1 identity."""
    return Reference(id=name, version="v1")


MEMBERS = (reference("recipe-a"), reference("recipe-b"))
SPACE = ChoiceSpace(feasible_set=reference("synthetic-ci-recipes"), members=MEMBERS)
IMPLEMENTATION = reference("user-owned-cumulative-loss-example")
LOSS = ScalarSpec(
    dimension=reference("ci-surrogate-loss"),
    semantics=UnitSemantics(unit="1", meaning="synthetic-bounded-ci-loss-v1"),
    role="loss",
    minimum=0,
    maximum=1,
)


def capability() -> Capability:
    """Declare exactly this user-owned immediate full-panel example."""
    return Capability(
        schema_version=1,
        implementation=IMPLEMENTATION,
        action_kinds=("choice",),
        feedback_kinds=("full_panel",),
        required_values=(LOSS,),
        origins=("synthetic",),
        timing_modes=("untimed",),
        supports_delayed_feedback=False,
        requires_known_receipt=False,
        requires_observed=True,
        state_owner="host",
        requires_sampling=False,
        sampling_requirement="not_required",
    )


class LearnerState(CoreContract):
    """Keep this example's complete immutable direct state, without a runtime.

    Attributes
    ----------
    updates : int
        Accepted complete panels, at most 64 in this bounded example.
    totals : tuple of float
        Two cumulative dimensionless losses in fixed MEMBERS order.
    pending : ExecutedDecision or None
        At most one actual pending decision; delivery/dedup is caller-owned.
    """

    updates: Annotated[int, Field(strict=True, ge=0, le=64)] = 0
    totals: tuple[Binary64, Binary64] = (0.0, 0.0)
    pending: ExecutedDecision | None = None

    @model_validator(mode="after")
    def _bounds(self):
        """Check totals before exposing the replacement direct state."""
        if any(not 0 <= total <= self.updates for total in self.totals):
            raise ValueError("Example cumulative loss exceeds its update support")
        return self


class HostLearner:
    """Choose the least cumulative observed loss, with stable recipe-ID ties.

    This class belongs to the user's copied script, not the installed library.
    Direct update mutates its own state and is not implicitly idempotent. It
    requires one complete panel after each decision and no external artifacts.
    """

    def __init__(self):
        self._state = LearnerState(schema_version=1)

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Create one proposal, retaining its actual pending feedback binding."""
        request = DecisionRequest.model_validate(request)
        if self._state.updates >= 64:
            raise ValueError("Example supports at most 64 completed opportunities")
        if self._state.pending is not None or request.round != self._state.updates or request.space != SPACE:
            raise ValueError("Example requires its fixed universe and one update per opportunity")
        index = min(range(2), key=lambda coordinate: (self._state.totals[coordinate], MEMBERS[coordinate].id))
        action = ChoiceAction(schema_version=1, space=SPACE, chosen=MEMBERS[index])
        sampling = SamplingRecord(
            mechanism="deterministic",
            mechanism_ref=IMPLEMENTATION,
            status="not_required",
            action_fingerprint=action.fingerprint,
            probability=None,
        )
        proposal = Decision(
            schema_version=1,
            request=request,
            implementation=IMPLEMENTATION,
            action=action,
            sampling=sampling,
            state_owner="host",
        )
        validate_decision(proposal, capability())
        executed = ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=action,
            sampling=sampling,
            mapping="unchanged",
            sampling_disposition="preserved",
        )
        self._state = LearnerState.model_validate(self._state.model_copy(update={"pending": executed}))
        return proposal

    def update(self, batch: FeedbackBatch, /) -> None:
        """Consume a complete compatible synthetic loss panel, once pending.

        Notes
        -----
        Capability validation checks the required loss dimension/version and its
        unit/role/bounds. Select that dimension explicitly; canonical ordering
        may place an admitted unrelated diagnostic before the loss.
        """
        pending = self._state.pending
        if pending is None:
            raise ValueError("No example decision is pending; raw updates are not delivery retries")
        validate_feedback(pending, batch, capability())
        if not isinstance(batch.feedback, FullPanel) or batch.supersedes is not None:
            raise ValueError("Example consumes only root full-information panels")
        values = {
            row.member: next(value.value for value in row.values if value.spec.dimension == LOSS.dimension)
            for row in batch.feedback.rows
        }
        additions = tuple(values[member] for member in MEMBERS)
        if any(value is None for value in additions):
            raise ValueError("Unknown losses cannot become zeros")
        totals = tuple(old + cast(float, value) for old, value in zip(self._state.totals, additions, strict=True))
        self._state = LearnerState(schema_version=1, updates=self._state.updates + 1, totals=(totals[0], totals[1]))

    def export_state(self) -> LearnerState:
        """Return revalidated complete immutable state without storage or a clock."""
        return LearnerState.model_validate(self._state)

    def restore_state(self, state: LearnerState, /) -> None:
        """Validate the complete replacement before changing this host instance."""
        self._state = LearnerState.model_validate(state)


def cases() -> tuple[ConformanceCase, ...]:
    """Provide two independently specified complete synthetic loss panels."""
    problem = ProblemCapabilities(
        schema_version=1,
        space=SPACE,
        feedback_kind="full_panel",
        values=(LOSS,),
        origins=("synthetic",),
        timing_mode="untimed",
        delayed_feedback=False,
        state_owners=("host",),
        sampling_statuses=("not_required",),
        receipt_status="unknown",
    )
    return tuple(
        ConformanceCase(
            problem=problem,
            request=DecisionRequest(
                schema_version=1,
                task=reference("synthetic-ci-choice"),
                decision_id=f"build-{index}",
                round=index,
                space=SPACE,
                objective=reference("minimize-declared-loss"),
            ),
            outcomes=tuple(
                FeedbackRow(
                    member=member,
                    values=(
                        OutcomeValue(
                            spec=LOSS,
                            value=value,
                            status="observed",
                            origin="synthetic",
                            transform=reference("synthetic-loss-identity"),
                        ),
                    ),
                )
                for member, value in zip(MEMBERS, losses, strict=True)
            ),
        )
        for index, losses in enumerate(((1.0, 0.0), (0.0, 1.0)))
    )


def main() -> None:
    """Show a feedback-driven choice change and real restored protocol paths."""
    report = check_decision_contracts(HostLearner, capability(), cases())
    print(
        json.dumps(
            {
                "implementation": "user-owned-example",
                "coverage": "full_panel",
                "choices": [cast(ChoiceAction, decision.action).chosen.id for decision in report.decisions],
                "accepted_panels": len(report.feedback),
                "builtin_methods": "pending-368",
            }
        )
    )


if __name__ == "__main__":
    main()
