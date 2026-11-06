"""Real incremental admission rejects ambiguous or reinterpreted CI evidence."""

import itertools

import pytest

from coleman.core.feedback import AbsentFeedback, FeedbackRow
from coleman.core.identity import Reference
from coleman.core.splits import TemporalObservation
from coleman.observations.contracts import ObservationHistory, ObservationRevision
from coleman.replay.contracts import ReplayFeedback
from coleman.results.domain import Measurement
from tests.replay.support import changed_value, emit, rows, session, set_clocks, with_rows


def test_mixed_realized_modeled_and_synthetic_values_remain_distinct_in_the_accepted_full_panel(example):
    replay, host = session(example, "full_panel")
    first = emit(example, replay, 0, profile="full_panel", at=0, end=1)
    outcome = example.feedback(first, (0.0, 0.5, 1.0), profile="full_panel", visible=2, delay=0)
    origins = ("realized", "modeled_counterfactual", "synthetic")
    modified = tuple(
        FeedbackRow(
            member=row.member,
            values=(type(row.values[0]).model_validate(row.values[0].model_copy(update={"origin": origin})),),
        )
        for row, origin in zip(rows(outcome), origins, strict=True)
    )
    outcome = with_rows(example, outcome, modified)
    replay.admit_feedback((outcome,))
    replay.advance(eligible_through=example.instant(2), apply_at=example.instant(3))
    assert host.export_state().loss_sums == (0.0, 0.5, 1.0)
    application = host.export_state().applications[0]
    assert tuple(item.origin for item in application.feedback.origins) == origins
    assert application.feedback.origins[0].producer is None
    assert application.feedback.origins[1].producer == example.GENERATOR
    assert application.feedback.origins[2].producer == example.GENERATOR
    assert tuple(row.values[0].origin for row in rows(application.feedback)) == origins


def _feedback(example, replay):
    record = emit(example, replay, 0, at=0, end=1)
    return record, example.feedback(record, (0.0, 0.0, 0.0), visible=2, delay=0)


def _batch(feedback, **changes):
    batch = type(feedback.batch).model_validate(feedback.batch.model_copy(update=changes))
    return ReplayFeedback.model_validate(feedback.model_copy(update={"batch": batch}))


def _unchanged(replay, host):
    return replay.state.to_json(), host.export_state().to_json()


@pytest.mark.parametrize("fault", ["missing_parent", "fork", "cycle", "alias"])
def test_complete_revision_graph_preflight_rejects_before_any_host_effect(example, fault):
    replay, host = session(example)
    _, root = _feedback(example, replay)
    first_child = example.feedback(
        replay.state.decisions[0], (1.0, 0.0, 0.0), visible=3, delay=0, revision="r2", supersedes="root"
    )
    second_child = example.feedback(
        replay.state.decisions[0], (0.5, 0.0, 0.0), visible=4, delay=0, revision="r3", supersedes="root"
    )
    if fault == "missing_parent":
        submitted = (root, _batch(first_child, supersedes="missing"))
    elif fault == "fork":
        submitted = (root, first_child, second_child)
    elif fault == "cycle":
        submitted = (
            root,
            _batch(first_child, supersedes="r3"),
            _batch(
                set_clocks(second_child, available_at=example.instant(3), received_at=example.instant(3)),
                supersedes="r2",
            ),
        )
    else:
        submitted = (root, _batch(first_child, feedback_id="different-round"))
    before = _unchanged(replay, host)
    with pytest.raises(ValueError):
        replay.admit_feedback(submitted)
    assert _unchanged(replay, host) == before


@pytest.mark.parametrize("fault", ["value", "signed_zero", "receipt"])
def test_same_revision_changed_content_conflicts_instead_of_becoming_another_round(example, fault):
    replay, host = session(example)
    _, root = _feedback(example, replay)
    replay.admit_feedback((root,))
    if fault == "receipt":
        changed = set_clocks(root, received_at=example.instant(3))
    else:
        changed = changed_value(example, root, -0.0 if fault == "signed_zero" else 1.0)
    before = _unchanged(replay, host)
    with pytest.raises(ValueError, match="Conflicting|conflict"):
        replay.admit_feedback((changed,))
    assert _unchanged(replay, host) == before


def test_parent_receipt_can_follow_child_receipt_in_a_structurally_complete_chain(example):
    replay, host = session(example)
    first, root = _feedback(example, replay)
    root = set_clocks(root, received_at=example.instant(7))
    child = example.feedback(first, (0.5, 0.0, 0.0), visible=5, delay=0, revision="a-child", supersedes="root")
    replay.admit_feedback((child, root))
    replay.advance(eligible_through=example.instant(6), apply_at=example.instant(6))
    state = host.export_state()
    assert state.update_count == 1
    assert state.loss_sums == (0.5, 0.0, 0.0)
    assert state.applications[0].feedback.batch.revision_id == "a-child"


def test_exact_retry_and_enumeration_permutations_have_one_acceptance_and_one_identity(example):
    outputs = []
    for order in itertools.permutations((0, 1, 0)):
        replay, host = session(example)
        first, root = _feedback(example, replay)
        child = example.feedback(first, (0.5, 0.0, 0.0), visible=3, delay=0, revision="a-child", supersedes="root")
        revisions = (root, child)
        replay.admit_feedback(tuple(revisions[index] for index in order))
        replay.advance(eligible_through=example.instant(4), apply_at=example.instant(4))
        before = host.export_state().to_json()
        replay.admit_feedback((child, root, child))
        assert host.export_state().to_json() == before
        assert replay.state.update_count == 1
        assert len(replay.state.feedback) == 2
        outputs.append((replay.state.to_json(), before))
    assert outputs == [outputs[0]] * 6


def test_maximum_bounded_retry_cohort_is_idempotent_even_when_its_revision_is_already_retained(example):
    replay, host = session(example)
    _, root = _feedback(example, replay)
    replay.admit_feedback((root,))
    before = _unchanged(replay, host)
    replay.admit_feedback((root,) * 4096)
    assert _unchanged(replay, host) == before
    with pytest.raises(ValueError, match="4096"):
        replay.admit_feedback((root,) * 4097)
    assert _unchanged(replay, host) == before


@pytest.mark.parametrize(
    "fault", ["execution", "lineage", "target", "horizon", "member_version", "dimension_version", "unit"]
)
def test_actual_execution_and_exact_quantity_and_target_versions_are_required(example, fault):
    replay, host = session(example)
    _, original = _feedback(example, replay)
    if fault == "execution":
        bad = _batch(original, decision_fingerprint="sha256:" + "0" * 64)
    elif fault == "lineage":
        bad = ReplayFeedback.model_validate(original.model_copy(update={"lineage": example.reference("foreign")}))
    elif fault in {"target", "horizon"}:
        target = original.target.model_copy(
            update={"reference" if fault == "target" else "horizon": example.reference("foreign")}
        )
        batch = original.batch if fault == "target" else original.batch.model_copy(update={"horizon": target.horizon})
        bad = ReplayFeedback.model_validate(original.model_copy(update={"target": target, "batch": batch}))
    else:
        row = rows(original)[0]
        value = row.values[0]
        member = row.member
        if fault == "member_version":
            member = Reference(id=member.id, version="v2")
        elif fault == "dimension_version":
            spec = value.spec.model_copy(update={"dimension": Reference(id=value.spec.dimension.id, version="v2")})
            value = type(value).model_validate(value.model_copy(update={"spec": spec}))
        else:
            units = value.spec.semantics.model_copy(update={"unit": "s"})
            spec = value.spec.model_copy(update={"semantics": units})
            value = type(value).model_validate(value.model_copy(update={"spec": spec}))
        bad = with_rows(example, original, (FeedbackRow(member=member, values=(value,)),))
    before = _unchanged(replay, host)
    with pytest.raises(ValueError):
        replay.admit_feedback((bad,))
    assert _unchanged(replay, host) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_unchecked_nonfinite_or_boolean_outcome_is_revalidated_at_actual_admission(example, value):
    replay, host = session(example)
    _, root = _feedback(example, replay)
    row = rows(root)[0]
    unchecked_value = row.values[0].model_copy(update={"value": value})
    unchecked_row = row.model_copy(update={"values": (unchecked_value,)})
    unchecked_payload = root.batch.feedback.model_copy(update={"row": unchecked_row})
    unchecked_batch = root.batch.model_copy(update={"feedback": unchecked_payload})
    unchecked = root.model_copy(update={"batch": unchecked_batch})
    before = _unchanged(replay, host)
    with pytest.raises(ValueError):
        replay.admit_feedback((unchecked,))
    assert _unchanged(replay, host) == before


@pytest.mark.parametrize("fault", ["model_version", "model_digest", "origin"])
def test_revision_cannot_silently_change_one_value_producer_or_origin(example, fault):
    replay, host = session(example)
    first, root = _feedback(example, replay)
    child = example.feedback(first, (1.0, 0.0, 0.0), visible=3, delay=0, revision="r2", supersedes="root")
    if fault == "origin":
        row = rows(child)[0]
        value = type(row.values[0]).model_validate(row.values[0].model_copy(update={"origin": "realized"}))
        child = with_rows(example, child, (FeedbackRow(member=row.member, values=(value,)),))
    else:
        evidence = child.origins[0]
        producer = evidence.producer
        assert producer is not None
        if fault == "model_version":
            producer = producer.model_copy(update={"reference": Reference(id=producer.reference.id, version="v2")})
        else:
            producer = producer.model_copy(update={"definition_fingerprint": "sha256:" + "a" * 64})
        evidence = type(evidence).model_validate(evidence.model_copy(update={"producer": producer}))
        child = ReplayFeedback.model_validate(child.model_copy(update={"origins": (evidence,)}))
    before = _unchanged(replay, host)
    with pytest.raises(ValueError, match="origin|model"):
        replay.admit_feedback((root, child))
    assert _unchanged(replay, host) == before


def test_absent_notification_never_creates_a_numeric_zero_or_update(example):
    replay, host = session(example)
    _, root = _feedback(example, replay)
    batch = type(root.batch).model_validate(
        root.batch.model_copy(update={"feedback": AbsentFeedback(reason="not-delivered")})
    )
    absence = ReplayFeedback(schema_version=1, lineage=root.lineage, batch=batch, target=root.target, origins=())
    replay.admit_feedback((absence,))
    replay.advance(eligible_through=example.instant(4), apply_at=example.instant(4))
    assert host.export_state().update_count == 0
    assert host.export_state().observed_counts == (0, 0, 0)


def observation(example, available, *, value=1.0, revision="root", supersedes=None):
    return ObservationRevision(
        schema_version=1,
        metadata=TemporalObservation(
            observation_id="authored-build-duration",
            revision_id=revision,
            group_id="authored-family",
            event_at=example.instant(0),
            available_at=example.instant(available),
            target_end=example.instant(0),
        ),
        entity_id="suite-a",
        received_at=example.instant(available),
        supersedes=supersedes,
        measurement=Measurement(
            name="duration",
            value=value,
            status="observed",
            unit="s",
            meaning="authored-completed-ci-duration-v1",
            direction="minimize",
            minimum=0,
        ),
    )


def source(example, *revisions):
    return ObservationHistory(
        schema_version=1, source_id=example.SOURCE.id, source_version=example.SOURCE.version, revisions=revisions
    )


@pytest.mark.parametrize("available", [5, 10])
def test_old_history_backfill_rejects_without_rewriting_an_emitted_eligible_snapshot(example, available):
    replay, host = session(example)
    emit(example, replay, 0, at=12, cutoff=10, end=13)
    backfilled = source(example, observation(example, available))
    before = _unchanged(replay, host)
    with pytest.raises(ValueError, match="backfill|eligible|snapshot"):
        replay.admit_history(backfilled)
    assert _unchanged(replay, host) == before


def test_future_history_append_correction_and_perturbation_preserve_earlier_full_host_and_prefix(example):
    outcomes = []
    for value in (1.0, 1234.0):
        replay, host = session(example)
        first = emit(example, replay, 0, at=0, end=1)
        early = host.export_state().to_json(), first.to_json(), replay.state.causal_prefix_fingerprint
        root = observation(example, 100, value=value)
        child = observation(example, 101, value=value + 1, revision="r2", supersedes="root")
        appended = source(example, child, root)
        old_inventory = replay.state.fingerprint
        replay.admit_history(appended)
        assert replay.state.fingerprint != old_inventory
        actual_prefix = host.export_state().to_json(), first.to_json(), replay.state.causal_prefix_fingerprint
        assert actual_prefix == early
        record = replay.emit(
            replay.prepare(example.opportunity(1, at=2, end=3, source=appended)), execution="offline_identity_v1"
        )
        outcomes.append((host.export_state().to_json(), record.to_json(), replay.state.causal_prefix_fingerprint))
    assert outcomes[0] == outcomes[1]


def test_future_feedback_values_do_not_enter_current_decisions_or_causal_prefix(example):
    outputs = []
    inventories = []
    for value in (0.0, 1.0):
        replay, host = session(example)
        first = emit(example, replay, 0, at=0, end=1)
        future = example.feedback(first, (value, 0.0, 0.0), visible=100, delay=1)
        replay.admit_feedback((future,))
        second = emit(example, replay, 1, at=2, end=3)
        outputs.append((second.to_json(), host.export_state().to_json(), replay.state.causal_prefix_fingerprint))
        inventories.append(replay.state.fingerprint)
    assert outputs[0] == outputs[1]
    assert inventories[0] != inventories[1]


def test_model_binding_cannot_change_after_coordinate_is_omitted_by_an_intermediate_partial_revision(example):
    replay, host = session(example, "component")
    first = emit(example, replay, 0, profile="component", at=0, end=1)
    root = example.feedback(first, (0.0, 0.5, 1.0), profile="component", visible=2, delay=0, completeness="partial")
    middle = example.feedback(
        first,
        (0.0, 0.5, 1.0),
        profile="component",
        visible=3,
        delay=0,
        completeness="partial",
        revision="r2",
        supersedes="root",
    )
    unit_value = rows(middle)[0].values[0]
    middle = with_rows(example, middle, (FeedbackRow(member=example.MEMBERS[1], values=(unit_value,)),))
    complete = example.feedback(
        first, (0.0, 0.5, 1.0), profile="component", visible=4, delay=0, revision="r3", supersedes="r2"
    )
    evidence = list(complete.origins)
    target = next(i for i, item in enumerate(evidence) if item.member == example.MEMBERS[0])
    producer = example.GENERATOR.model_copy(
        update={"reference": Reference(id=example.GENERATOR.reference.id, version="v2")}
    )
    evidence[target] = type(evidence[target]).model_validate(evidence[target].model_copy(update={"producer": producer}))
    complete = ReplayFeedback.model_validate(complete.model_copy(update={"origins": tuple(evidence)}))
    before = _unchanged(replay, host)
    with pytest.raises(ValueError, match="origin|model"):
        replay.admit_feedback((root, middle, complete))
    assert _unchanged(replay, host) == before
