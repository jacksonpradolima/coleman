"""A sampled, user-owned chosen-feedback example with complete owned RNG state.

This four-bin epsilon-greedy demonstration is not shipped UCB or Exp3. It uses
Python Random.randrange(4): three bins choose the largest empirical mean and
one chooses the other recipe. Its declared .75/.25 masses are those of this
explicit uniform-bin mechanism, not calibrated outcome probabilities or claims
of independent PRNG draws conditional on a publicly known deterministic state.
"""

import json
from random import Random
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from coleman.core.actions import ChoiceAction, ChoiceSpace
from coleman.core.base import CoreContract
from coleman.core.capabilities import Capability, ProblemCapabilities, validate_decision, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import ChosenFeedback, FeedbackBatch, FeedbackRow, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.core.units import UnitSemantics
from coleman.testing.decision_contracts import ConformanceCase, check_decision_contracts

Word = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
Count = Annotated[int, Field(strict=True, ge=0, le=64)]


def reference(name: str) -> Reference:
    """Return an explicit public synthetic v1 identity."""
    return Reference(id=name, version="v1")


MEMBERS = (reference("recipe-a"), reference("recipe-b"))
SPACE = ChoiceSpace(feasible_set=reference("synthetic-ci-recipes"), members=MEMBERS)
IMPLEMENTATION = reference("user-owned-four-bin-choice-example")
REWARD = ScalarSpec(
    dimension=reference("detected-fault-credit"),
    semantics=UnitSemantics(unit="1", meaning="synthetic-detected-fault-credit-v1"),
    role="reward",
    minimum=0,
    maximum=1,
)


def capability() -> Capability:
    """Declare chosen-only feedback and the actual four-bin sampling evidence."""
    return Capability(
        schema_version=1,
        implementation=IMPLEMENTATION,
        action_kinds=("choice",),
        feedback_kinds=("chosen",),
        required_values=(REWARD,),
        origins=("synthetic",),
        timing_modes=("untimed",),
        supports_delayed_feedback=False,
        requires_known_receipt=False,
        requires_observed=True,
        state_owner="host",
        requires_sampling=True,
        sampling_requirement="exact",
    )


class MABState(CoreContract):
    """Keep the complete bounded example state as non-executable typed values.

    Attributes
    ----------
    generator : {'python-random-mt19937-v3'}
        Explicit Python state profile; no cross-runtime portability is promised.
    words : tuple of int
        All 624 MT words plus its final index, retaining buffered generator state.
        Gaussian cache is absent because this example calls only randrange.
    counts, rewards : tuple
        Per-recipe observed counts and cumulative dimensionless rewards. No
        unchosen observations are inferred.
    pending : ExecutedDecision or None
        At most one actual pending decision. Raw updates require caller delivery
        discipline and provide no durable exactly-once guarantee.
    """

    generator: Literal["python-random-mt19937-v3"] = "python-random-mt19937-v3"
    words: Annotated[tuple[Word, ...], Field(min_length=625, max_length=625)]
    counts: tuple[Count, Count] = (0, 0)
    rewards: tuple[Binary64, Binary64] = (0.0, 0.0)
    pending: ExecutedDecision | None = None

    @model_validator(mode="after")
    def _bounds(self):
        """Validate complete count/reward/RNG shape before state replacement."""
        if self.words[-1] > 624 or sum(self.counts) > 64:
            raise ValueError("Example RNG index or feedback support is invalid")
        if any(not 0 <= reward <= count for reward, count in zip(self.rewards, self.counts, strict=True)):
            raise ValueError("Example reward totals exceed chosen observations")
        return self


class HostMAB:
    """Use one owned uniform-bin stream and only its actually chosen feedback.

    This copied host implementation is not part of COLEMAN's shipped algorithms.
    A seed initializes a fresh stream; restoration uses the whole exported state.
    The component owns no global RNG, registry, workspace or executable pickle.
    """

    def __init__(self, seed: int = 19):
        self._state = MABState(schema_version=1, words=Random(seed).getstate()[1])

    def decide(self, request: DecisionRequest, /) -> Decision:
        """Sample one actual recipe and atomically retain its local RNG state."""
        request = DecisionRequest.model_validate(request)
        if sum(self._state.counts) >= 64:
            raise ValueError("Example supports at most 64 completed opportunities")
        if self._state.pending is not None or request.round != sum(self._state.counts) or request.space != SPACE:
            raise ValueError("Example requires fixed recipes and one update per opportunity")
        means = tuple(
            reward / count if count else 0.0
            for reward, count in zip(self._state.rewards, self._state.counts, strict=True)
        )
        best = max(range(2), key=lambda index: (means[index], -index))
        generator = Random(0)
        generator.setstate((3, self._state.words, None))
        chosen = best if generator.randrange(4) < 3 else 1 - best
        action = ChoiceAction(schema_version=1, space=SPACE, chosen=MEMBERS[chosen])
        sampling = SamplingRecord(
            mechanism="sampled",
            mechanism_ref=reference("python-randrange-four-bin-v1"),
            status="exact",
            action_fingerprint=action.fingerprint,
            probability=0.75 if chosen == best else 0.25,
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
        self._state = MABState.model_validate(
            self._state.model_copy(update={"words": generator.getstate()[1], "pending": executed})
        )
        return proposal

    def update(self, batch: FeedbackBatch, /) -> None:
        """Update one chosen recipe without receiving another oracle row.

        Notes
        -----
        Capability validation checks the required reward dimension/version and
        its unit/role/bounds. Lookup uses that dimension, so admitted unrelated
        diagnostics never become reward merely through canonical ordering.
        """
        pending = self._state.pending
        if pending is None:
            raise ValueError("No sampled decision is pending")
        validate_feedback(pending, batch, capability())
        if not isinstance(batch.feedback, ChosenFeedback) or batch.supersedes is not None:
            raise ValueError("Example accepts root chosen-only feedback")
        index = MEMBERS.index(batch.feedback.row.member)
        value = next(
            outcome.value for outcome in batch.feedback.row.values if outcome.spec.dimension == REWARD.dimension
        )
        if value is None:
            raise ValueError("Unknown rewards cannot become zeros")
        counts = tuple(count + (coordinate == index) for coordinate, count in enumerate(self._state.counts))
        rewards = tuple(
            reward + (value if coordinate == index else 0.0) for coordinate, reward in enumerate(self._state.rewards)
        )
        self._state = MABState(
            schema_version=1, words=self._state.words, counts=(counts[0], counts[1]), rewards=(rewards[0], rewards[1])
        )

    def export_state(self) -> MABState:
        """Return complete immutable counts/rewards/RNG/pending state, without I/O."""
        return MABState.model_validate(self._state)

    def restore_state(self, state: MABState, /) -> None:
        """Reject unknown/malformed state before replacing any live host state."""
        self._state = MABState.model_validate(state)


def cases() -> tuple[ConformanceCase, ...]:
    """Supply a bounded hidden oracle; the harness exposes only the chosen row."""
    problem = ProblemCapabilities(
        schema_version=1,
        space=SPACE,
        feedback_kind="chosen",
        values=(REWARD,),
        origins=("synthetic",),
        timing_mode="untimed",
        delayed_feedback=False,
        state_owners=("host",),
        sampling_statuses=("exact",),
        receipt_status="unknown",
    )
    return tuple(
        ConformanceCase(
            problem=problem,
            request=DecisionRequest(
                schema_version=1,
                task=reference("synthetic-chosen-ci-feedback"),
                decision_id=f"build-{index}",
                round=index,
                space=SPACE,
                objective=reference("maximize-declared-credit"),
            ),
            outcomes=tuple(
                FeedbackRow(
                    member=member,
                    values=(
                        OutcomeValue(
                            spec=REWARD,
                            value=value,
                            status="observed",
                            origin="synthetic",
                            transform=reference("synthetic-credit-identity"),
                        ),
                    ),
                )
                for member, value in zip(MEMBERS, (0.0, 1.0), strict=True)
            ),
        )
        for index in range(8)
    )


def main() -> None:
    """Exercise real draws and restored continuation through public protocols."""
    report = check_decision_contracts(HostMAB, capability(), cases())
    print(
        json.dumps(
            {
                "implementation": "user-owned-example",
                "coverage": "chosen",
                "choices": [cast(ChoiceAction, decision.action).chosen.id for decision in report.decisions],
                "sampling_masses": [decision.sampling.probability for decision in report.decisions],
                "delivered_rows": len(report.feedback),
                "builtin_methods": "pending-370",
            }
        )
    )


if __name__ == "__main__":
    main()
