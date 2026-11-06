"""Independent CI arithmetic, causal clocks and observed-coverage oracles."""

import json

import pytest

from coleman.core.feedback import FeedbackRow
from coleman.core.numeric import ScalarSpec
from coleman.core.units import UnitSemantics
from tests.replay.support import FIXTURES, chosen, emit, rows, selected, session, set_clocks, with_rows


def test_public_chosen_example_runs_the_real_replay(example):
    assert example.run() == {
        "evidence": "authored_synthetic_ci",
        "execution": "offline_identity_v1",
        "choices": ["suite-a", "suite-a", "suite-b"],
        "decisions": 3,
        "accepted_rounds": 1,
        "counts": (1, 0, 0),
        "loss_sums": (1.0, 0.0, 0.0),
    }


def test_full_panel_hand_worked_delayed_timeline(example):
    oracle = json.loads((FIXTURES / "timeline-oracles.json").read_text())["full_panel"]
    replay, host = session(example, "full_panel")
    records = []
    for item in oracle["decisions"]:
        record = emit(
            example, replay, item["index"], profile="full_panel", at=item["at"], cutoff=item["cutoff"], end=item["end"]
        )
        records.append(record)
        assert chosen(record) == item["choice"]
        assert host.export_state().update_count == item["updates_before"]
        if item["index"] < 3:
            outcome = oracle["outcomes"][item["index"]]
            feedback = example.feedback(
                record,
                tuple(outcome["values"]),
                profile="full_panel",
                visible=outcome["receipt"],
                event=outcome["event"],
                mature=outcome["mature"],
                delay=outcome["delay"],
            )
            feedback = set_clocks(feedback, available_at=example.instant(outcome["available"]))
            replay.admit_feedback((feedback,))
    replay.advance(eligible_through=example.instant(9), apply_at=example.instant(9))
    state = host.export_state()
    assert state.decision_count == oracle["final_decisions"]
    assert state.update_count == oracle["final_updates"]
    assert list(state.observed_counts) == oracle["final_counts"]
    assert list(state.loss_sums) == oracle["final_sums"]
    assert len(replay.state.updates) == 3
    assert [row.application.apply_at for row in replay.state.updates] == [
        example.instant(5),
        example.instant(8),
        example.instant(9),
    ]
    assert tuple(record.to_json() for record in records) == tuple(record.to_json() for record in replay.state.decisions)


def test_chosen_feedback_does_not_expose_other_suite_losses(example):
    outputs = []
    for hidden in ((0.0, 0.0), (1.0, 0.75), (0.5, 1.0)):
        replay, host = session(example)
        first = emit(example, replay, 0, at=0, end=1)
        second = emit(example, replay, 1, at=1, end=2)
        root = example.feedback(first, (1.0, *hidden), visible=2, delay=1)
        replay.admit_feedback((root,))
        third = emit(example, replay, 2, at=3, end=4)
        next_feedback = example.feedback(second, (0.0, *hidden), visible=4, delay=1)
        replay.admit_feedback((next_feedback,))
        replay.advance(eligible_through=example.instant(4), apply_at=example.instant(4))
        last_feedback = example.feedback(third, (hidden[0], 0.0, hidden[1]), visible=5, delay=0)
        replay.admit_feedback((last_feedback,))
        fourth = emit(example, replay, 3, at=6, end=7)
        state = host.export_state()
        assert state.observed_counts == (2, 1, 0)
        assert state.loss_sums == (1.0, 0.0, 0.0)
        assert [chosen(row) for row in (first, second, third, fourth)] == ["suite-a", "suite-a", "suite-b", "suite-c"]
        assert all(len(application.feedback.origins) == 1 for application in state.applications)
        outputs.append((state.to_json(), replay.state.causal_prefix_fingerprint, root.to_json()))
    assert outputs == [outputs[0]] * 3


def test_component_panel_updates_selected_members_once_and_keeps_other_outcomes_hidden(example):
    outputs = []
    for hidden in (0.0, 1.0):
        replay, host = session(example, "component")
        first = emit(example, replay, 0, profile="component", at=0, end=1)
        replay.admit_feedback((example.feedback(first, (1.0, 0.0, hidden), profile="component", visible=2, delay=0),))
        second = emit(example, replay, 1, profile="component", at=3, end=4)
        replay.admit_feedback((example.feedback(second, (hidden, 0.5, 0.5), profile="component", visible=4, delay=0),))
        third = emit(example, replay, 2, profile="component", at=5, end=6)
        state = host.export_state()
        assert [selected(row) for row in (first, second, third)] == [
            ("suite-a", "suite-b"),
            ("suite-b", "suite-c"),
            ("suite-b", "suite-c"),
        ]
        assert state.decision_count == 3
        assert state.update_count == 2
        assert state.observed_counts == (1, 2, 1)
        assert state.loss_sums == (1.0, 0.5, 0.5)
        outputs.append((state.to_json(), replay.state.causal_prefix_fingerprint))
    assert outputs[0] == outputs[1]


def test_partial_components_wait_for_one_explicit_complete_successor(example):
    replay, host = session(example, "component")
    first = emit(example, replay, 0, profile="component", at=0, end=2)
    partial = example.feedback(
        first, (0.5, 0.0, 1.0), profile="component", visible=2, delay=1, completeness="partial", revision="z-root"
    )
    complete = example.feedback(
        first, (0.5, 0.0, 1.0), profile="component", visible=4, delay=1, revision="a-complete", supersedes="z-root"
    )
    replay.admit_feedback((complete, partial))
    emit(example, replay, 1, profile="component", at=3, cutoff=3, end=4)
    assert host.export_state().update_count == 0
    third = emit(example, replay, 2, profile="component", at=6, cutoff=4, end=7)
    state = host.export_state()
    assert state.update_count == 1
    assert state.observed_counts == (1, 1, 0)
    assert state.loss_sums == (0.5, 0.0, 0.0)
    assert selected(third) == ("suite-b", "suite-c")
    assert state.applications[0].feedback.batch.revision_id == "a-complete"
    replay.admit_feedback((partial, complete, partial))
    assert host.export_state().fingerprint == state.fingerprint


@pytest.mark.parametrize("blocked", ["unavailable", "immature", "partial"])
def test_latest_source_visible_unready_successor_blocks_a_ready_root(example, blocked):
    profile = "component" if blocked == "partial" else "chosen"
    replay, host = session(example, profile)
    first = emit(example, replay, 0, profile=profile, at=0, end=1)
    root = example.feedback(first, (0.0, 0.0, 0.0), profile=profile, visible=4, delay=0)
    child = example.feedback(
        first,
        (None if blocked == "unavailable" else 0.0, 0.0, 0.0),
        profile=profile,
        visible=5,
        mature=20 if blocked == "immature" else 5,
        delay=0,
        revision="correction",
        supersedes="root",
        completeness="partial" if blocked == "partial" else "complete",
    )
    replay.admit_feedback((child, root))
    replay.advance(eligible_through=example.instant(6), apply_at=example.instant(10))
    assert host.export_state().update_count == 0
    assert host.export_state().loss_sums == (0.0, 0.0, 0.0)
    assert len(replay.state.updates) == 0


def test_actual_prior_root_acceptance_is_preserved_when_later_correction_arrives(example):
    replay, host = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    root = example.feedback(first, (1.0, 0.0, 0.0), visible=4, delay=0)
    replay.admit_feedback((root,))
    replay.advance(eligible_through=example.instant(4), apply_at=example.instant(4))
    before = host.export_state().to_json()
    accepted = replay.state.updates[0].to_json()
    child = example.feedback(first, (None, 0.0, 0.0), visible=5, delay=0, revision="corrected", supersedes="root")
    replay.admit_feedback((child,))
    replay.advance(eligible_through=example.instant(6), apply_at=example.instant(10))
    assert host.export_state().to_json() == before
    assert replay.state.updates[0].to_json() == accepted


def test_lagging_information_clock_is_independent_of_actual_application_time(example):
    replay, host = session(example)
    first = emit(example, replay, 0, at=2, end=4)
    emit(example, replay, 1, at=6, end=7)
    feedback = example.feedback(first, (1.0, 0.0, 0.0), visible=8.5, event=4, mature=7, delay=2)
    feedback = set_clocks(feedback, available_at=example.instant(6))
    replay.admit_feedback((feedback,))
    emit(example, replay, 2, cutoff=8, at=10, end=12)
    assert host.export_state().update_count == 0
    emit(example, replay, 3, cutoff=9, at=11, end=12)
    state = host.export_state()
    assert state.update_count == 1
    assert state.applications[0].eligible_at == example.instant(8.5)
    assert state.applications[0].apply_at == example.instant(11)
    assert state.applications[0].feedback.batch.delay_rounds == 2


@pytest.mark.parametrize("null", [False, True])
def test_zero_is_a_real_update_but_unavailable_loss_stays_pending(example, null):
    replay, host = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    feedback = example.feedback(first, (None if null else 0.0, 1.0, 1.0), visible=2, delay=0)
    replay.admit_feedback((feedback,))
    replay.advance(eligible_through=example.instant(2), apply_at=example.instant(2))
    assert host.export_state().update_count == (0 if null else 1)
    assert host.export_state().loss_sums == (0.0, 0.0, 0.0)
    assert host.export_state().observed_counts == ((0, 0, 0) if null else (1, 0, 0))


@pytest.mark.parametrize("diagnostic", [-0.5, None])
@pytest.mark.parametrize("dimension", ["aaa-diagnostic", "zzz-diagnostic"])
def test_unused_negative_or_missing_diagnostic_preserves_stats_and_actual_next_decision(example, diagnostic, dimension):
    outputs = []
    full_hashes = []
    for include in (False, True):
        replay, host = session(example)
        first = emit(example, replay, 0, at=0, end=1)
        feedback = example.feedback(first, (1.0, 0.0, 0.0), visible=2, delay=0)
        if include:
            spec = ScalarSpec(
                dimension=example.reference(dimension),
                semantics=UnitSemantics(unit="1", meaning="optional-ci-diagnostic"),
                role="diagnostic",
                minimum=-1,
                maximum=1,
            )
            diagnostic_value = type(rows(feedback)[0].values[0])(
                spec=spec,
                value=diagnostic,
                status="unavailable" if diagnostic is None else "observed",
                origin="synthetic",
                transform=None,
            )
            modified = tuple(
                FeedbackRow(member=row.member, values=(*row.values, diagnostic_value)) for row in rows(feedback)
            )
            feedback = with_rows(example, feedback, modified)
        replay.admit_feedback((feedback,))
        second = emit(example, replay, 1, at=3, end=4)
        state = host.export_state()
        outputs.append((state.loss_sums, state.observed_counts, state.update_count, chosen(second)))
        full_hashes.append((state.fingerprint, replay.state.causal_prefix_fingerprint))
    assert outputs[0] == outputs[1]
    assert full_hashes[0] != full_hashes[1]


def test_multiplex_capability_selects_a_compatible_feedback_shape_before_actual_component_execution(example):
    capability = type(example.capability("component")).model_validate(
        example.capability("component").model_copy(
            update={"action_kinds": ("choice", "subset"), "feedback_kinds": ("chosen", "component")}
        )
    )
    policy = type(example.policy("component")).model_validate(
        example.policy("component").model_copy(update={"capability": capability})
    )
    replay, host = session(example, "component", policy=policy)
    first = emit(example, replay, 0, profile="component", at=0, end=1)
    outcome = example.feedback(first, (1.0, 0.0, 0.5), profile="component", visible=2, delay=0)
    replay.admit_feedback((outcome,))
    second = emit(example, replay, 1, profile="component", at=3, end=4)
    assert selected(first) == ("suite-a", "suite-b")
    assert selected(second) == ("suite-b", "suite-c")
    assert host.export_state().update_count == 1
    assert host.export_state().observed_counts == (1, 1, 0)
    assert host.export_state().loss_sums == (1.0, 0.0, 0.0)


def test_feedback_from_a_same_instant_new_producer_enters_only_the_later_causal_phase(example):
    replay, host = session(example)
    first = emit(example, replay, 0, at=0, end=0)
    assert first.update_count_before == 0
    outcome = example.feedback(first, (0.0, 1.0, 1.0), visible=0, delay=0)
    replay.admit_feedback((outcome,))
    replay.advance(eligible_through=example.instant(0), apply_at=example.instant(0))
    assert host.export_state().update_count == 1
    assert replay.state.updates[0].sequence > first.sequence
    assert replay.state.updates[0].application.apply_at == first.accepted_at
    assert replay.advance(eligible_through=example.instant(0), apply_at=example.instant(0)) == ()


def test_every_ready_application_is_preflighted_before_any_actual_update(example):
    replay, host = session(example)
    first = emit(example, replay, 0, at=0, end=1)
    second = emit(example, replay, 1, at=1, end=2)
    valid = example.feedback(first, (1.0, 0.0, 0.0), visible=3, delay=1)
    wrong_delay = example.feedback(second, (0.0, 1.0, 0.0), visible=3, delay=1)
    replay.admit_feedback((valid, wrong_delay))
    before = replay.export_continuation().to_json()
    eligible, actual = example.instant(3), example.instant(4)
    with pytest.raises(ValueError, match="delay"):
        replay.advance(eligible_through=eligible, apply_at=actual)
    assert host.export_state().update_count == 0
    assert replay.export_continuation().to_json() == before


def test_ongoing_event_may_precede_decision_but_target_completion_still_gates_application(example):
    replay, host = session(example)
    opportunity = example.opportunity(0, at=0, end=5)
    timing = type(opportunity.request.timing).model_validate(
        opportunity.request.timing.model_copy(update={"mode": "ongoing", "target_start": example.instant(-2)})
    )
    request = type(opportunity.request).model_validate(opportunity.request.model_copy(update={"timing": timing}))
    opportunity = type(opportunity).model_validate(opportunity.model_copy(update={"request": request}))
    first = replay.emit(replay.prepare(opportunity), execution="offline_identity_v1")
    outcome = example.feedback(first, (1.0, 0.0, 0.0), event=-1, visible=1, mature=1, delay=0)
    replay.admit_feedback((outcome,))
    replay.advance(eligible_through=example.instant(2), apply_at=example.instant(4))
    assert host.export_state().update_count == 0
    replay.advance(eligible_through=example.instant(6), apply_at=example.instant(7))
    assert host.export_state().update_count == 1
    application = replay.state.updates[0].application
    assert application.feedback.batch.timing is not None
    assert application.feedback.batch.timing.event_at == example.instant(-1)
    assert application.eligible_at == example.instant(5)
    assert application.apply_at == example.instant(7)
