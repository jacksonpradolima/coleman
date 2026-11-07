"""Independent causal control regressions using actual shipped components."""

from datetime import UTC, datetime, timedelta

from coleman.core.actions import AbstainAction, MixtureAction, MixtureSpace
from coleman.core.decisions import DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.learning.control_contracts import ControlOutcome
from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectOptions
from coleman.learning.schedule import ControlledDirect, ControlledDirectState, ControlSchedule
from coleman.objectives.controls import ReleaseInterval
from tests.learning._controlled_cases import configuration, panel, ref


def test_failed_interval_remains_abstained_after_held_feedback_until_successful_recovery():
    """The later cancelling panel restores representability, not prior admission."""
    start = datetime(2026, 11, 7, tzinfo=UTC)
    calendar = ref("independent-release-clock")
    config = configuration(prior=(0.5, 0.25, 0.25), eta=1e308, strength=0.05, cadence=calendar)
    points = tuple(
        ReleaseInterval(
            interval=ref(f"interval-{index}"),
            start=start + timedelta(seconds=begin),
            end=start + timedelta(seconds=end),
            cutoff=start + timedelta(seconds=cutoff),
            release_at=start + timedelta(seconds=release),
        )
        for index, (begin, end, cutoff, release) in enumerate(((0, 10, 2, 3), (10, 20, 12, 13)))
    )
    schedule = ControlSchedule(schema_version=1, calendar=calendar, intervals=points)
    options = DirectOptions(task=ref("independent-ci-task"), objective=ref("independent-loss"), horizon=None)
    actual = ControlledHedge(config, owner=ref("independent-control-owner"))
    app = ControlledDirect(actual, schedule, options)

    def request(index, at, cutoff):
        decision_at = start + timedelta(seconds=at)
        timing = DecisionTiming(
            decision_at=decision_at,
            information_cutoff=start + timedelta(seconds=cutoff),
            target_start=decision_at,
            target_end=decision_at + timedelta(milliseconds=100),
            eligible_input_fingerprint="sha256:" + "a" * 64,
            mode="future",
        )
        return DecisionRequest(
            schema_version=1,
            task=options.task,
            objective=options.objective,
            decision_id=f"build-{index}",
            round=index,
            timing=timing,
            space=MixtureSpace(
                feasible_set=ref("all-ci-experts"), members=actual.members, profile="binary64-simplex-v1"
            ),
        )

    def feedback(proposal, losses, at):
        executed = ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=proposal.action,
            mapping="unchanged",
            sampling_disposition="preserved",
            sampling=proposal.sampling,
        )
        clock = start + timedelta(seconds=at)
        timing = FeedbackTiming(event_at=clock, available_at=clock, mature_at=clock, received_at=clock)
        return FeedbackBatch(
            schema_version=1,
            decision_fingerprint=executed.fingerprint,
            feedback_id=f"panel-{at}",
            revision_id="root",
            supersedes=None,
            horizon=None,
            feedback=panel(config, losses),
            timing=timing,
            delay_rounds=1,
        )

    first = app.decide(request(0, 0, 0))
    delayed = app.decide(request(1, 0.5, 0.5))
    assert isinstance(first.action, MixtureAction) and isinstance(delayed.action, MixtureAction)
    first_result = app.update(
        feedback(first, (0.0, 1.0, 1.0), 1),
        eligible_at=start + timedelta(seconds=1),
        apply_at=start + timedelta(seconds=1),
    )
    assert isinstance(first_result, ControlOutcome) and first_result.disposition == "held"
    failed = app.decide(request(2, 3, 2))
    assert isinstance(failed.action, AbstainAction)
    failure = actual.last_outcome
    assert failure is not None and failure.disposition == "numerical_failure"
    assert failure.failure is not None and failure.failure.code == "positive_underflow"
    assert actual.update_count == 1 and actual.release_count == 0
    second_result = app.update(
        feedback(delayed, (1.0, 0.0, 0.0), 4),
        eligible_at=start + timedelta(seconds=4),
        apply_at=start + timedelta(seconds=4),
    )
    assert isinstance(second_result, ControlOutcome) and second_result.disposition == "held"
    assert actual.update_count == 2 and actual.release_count == 0
    assert len(actual.export_state().payload.pending) == 2
    continuation = app.export_state()
    restored_kernel = ControlledHedge(config, owner=ref("independent-control-owner"))
    restored = ControlledDirect(restored_kernel, schedule, options)
    restored.restore_state(ControlledDirectState.from_json(continuation.to_json()))
    assert restored.export_state().to_json() == continuation.to_json()
    for candidate in (app, restored):
        still_failed = candidate.decide(request(3, 5, 5))
        assert isinstance(still_failed.action, AbstainAction), "Held feedback cannot erase an unresolved failed release"
        assert candidate.learner.unresolved_failure == failure
        recovered = candidate.decide(request(4, 13, 12))
        assert isinstance(recovered.action, MixtureAction)
        assert tuple(weight.value for weight in recovered.action.weights) == (0.5, 0.25, 0.25)
        assert candidate.learner.update_count == 2 and candidate.learner.release_count == 1
        payload = candidate.learner.export_state().payload
        assert payload.unresolved_failure is None and not payload.pending
        assert payload.last_outcome is not None
        assert tuple(item.value for item in payload.last_outcome.summed_losses) == (1.0, 1.0, 1.0)
    assert restored.export_state().to_json() == app.export_state().to_json()
