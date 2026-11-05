"""Independently exercise feedback coverage, evidence origin and local retries."""

from itertools import permutations
from typing import Any, cast

import pytest

from coleman.core.feedback import (
    AbsentFeedback,
    AggregateFeedback,
    ChosenFeedback,
    ComponentFeedback,
    FeedbackBatch,
    FeedbackRow,
    FullPanel,
    Origin,
    OutcomeValue,
    deduplicate_feedback,
)
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarRole, ScalarSpec, ScalarStatus
from coleman.core.primitives import MAX_MEMBERS
from coleman.core.timing import FeedbackTiming
from coleman.core.units import UnitSemantics

EXECUTED = "sha256:" + "a" * 64
GOLDEN_ABSENT_V1 = (
    '{"decision_fingerprint":"' + EXECUTED + '","delay_rounds":0,'
    '"feedback":{"kind":"absent","reason":"not-yet-observed"},'
    '"feedback_id":"feedback-1","horizon":null,"revision_id":"initial",'
    '"schema_version":1,"supersedes":null,"timing":null}'
)


def _ref(name: str, version: str = "v1") -> Reference:
    return Reference(id=name, version=version)


def _value(
    name: str = "quality",
    value: float | None = 0.0,
    *,
    role: ScalarRole = "objective",
    status: ScalarStatus = "observed",
    origin: Origin = "realized",
    version: str = "v1",
) -> OutcomeValue:
    return OutcomeValue(
        spec=ScalarSpec(
            dimension=_ref(name, version),
            semantics=UnitSemantics(unit="1", meaning="public-synthetic-quantity-v1"),
            role=role,
        ),
        value=value,
        status=status,
        origin=origin,
        transform=_ref("identity-loss") if role in {"loss", "reward"} else None,
    )


def _row(member: str = "A", *values: OutcomeValue) -> FeedbackRow:
    return FeedbackRow(member=_ref(member), values=values or (_value(),))


def _batch(feedback=None, **changes) -> FeedbackBatch:
    fields = {
        "schema_version": 1,
        "decision_fingerprint": EXECUTED,
        "feedback_id": "feedback-1",
        "revision_id": "initial",
        "supersedes": None,
        "horizon": None,
        "feedback": feedback if feedback is not None else AbsentFeedback(reason="not-yet-observed"),
    }
    fields.update(changes)
    return FeedbackBatch.model_validate(fields)


def test_versioned_golden_absence_round_trip_without_fake_values():
    record = FeedbackBatch.from_json(GOLDEN_ABSENT_V1)
    assert record.to_json() == GOLDEN_ABSENT_V1
    assert isinstance(record.feedback, AbsentFeedback)
    assert record.horizon is None and record.timing is None
    assert "value" not in type(record.feedback).model_fields
    assert record.fingerprint == FeedbackBatch.model_validate_json(GOLDEN_ABSENT_V1).fingerprint


@pytest.mark.parametrize("version", [0, 2, 99, True, "1"])
def test_unknown_or_coerced_version_cannot_read_a_v1_feedback_record(version):
    payload = _batch().model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValueError):
        FeedbackBatch.model_validate(payload)


def test_missing_version_and_duplicate_nested_keys_reject():
    payload = _batch().model_dump(mode="json")
    del payload["schema_version"]
    with pytest.raises(ValueError):
        FeedbackBatch.model_validate(payload)
    duplicate = GOLDEN_ABSENT_V1.replace('"reason":"not-yet-observed"', '"reason":"first","reason":"second"')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        FeedbackBatch.from_json(duplicate)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        FeedbackBatch.model_validate_json(duplicate)


def test_multiple_measurements_stay_chosen_only_and_keep_per_value_origin():
    observed_quality = _value("quality", 0.7)
    modeled_cost = _value("cost", -2.0, role="resource", origin="modeled_counterfactual")
    chosen = ChosenFeedback(row=_row("A", observed_quality, modeled_cost))
    record = FeedbackBatch.from_json(_batch(chosen).to_json())
    assert isinstance(record.feedback, ChosenFeedback)
    assert record.feedback.row.member.id == "A"
    values = {item.spec.dimension.id: item for item in record.feedback.row.values}
    assert values["quality"].origin == "realized"
    assert values["cost"].origin == "modeled_counterfactual"
    assert values["cost"].value == -2.0
    assert record.feedback.kind != FullPanel(rows=(_row("A"), _row("B"))).kind


def test_panel_and_dimension_declaration_permutations_have_one_identity():
    values = (_value("quality", 0.7), _value("latency", 2.0, role="resource"))
    identities = set()
    for value_order in permutations(values):
        rows = (_row("A", *value_order), _row("B", *reversed(value_order)))
        for row_order in permutations(rows):
            record = _batch(FullPanel(rows=row_order))
            assert FeedbackBatch.from_json(record.to_json()).to_json() == record.to_json()
            identities.add(record.fingerprint)
    assert len(identities) == 1


@pytest.mark.parametrize("container", [FullPanel, ComponentFeedback])
def test_duplicate_or_conflicting_member_versions_are_not_deduplicated(container):
    rows = (_row("A"), FeedbackRow(member=_ref("A", "v2"), values=(_value(),)))
    extras = {"completeness": "partial"} if container is ComponentFeedback else {}
    with pytest.raises(ValueError, match="member IDs must be unique"):
        container.model_validate({"rows": rows} | extras)


@pytest.mark.parametrize("container", [FeedbackRow, AggregateFeedback])
def test_duplicate_dimension_versions_reject_without_merging(container):
    values = (_value("quality"), _value("quality", version="v2"))
    extras = {"member": _ref("A")} if container is FeedbackRow else {}
    with pytest.raises(ValueError, match="dimension IDs must be unique"):
        container(values=values, **extras)


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_null_states_are_retained_in_structurally_complete_panels(status):
    panel = FullPanel(rows=(_row("A", _value(value=None, status=status)),))
    restored = FeedbackBatch.from_json(_batch(panel).to_json())
    assert isinstance(restored.feedback, FullPanel)
    assert restored.feedback.rows[0].values[0].status == status
    assert restored.feedback.rows[0].values[0].value is None
    assert _batch(panel).fingerprint != _batch(FullPanel(rows=(_row("A", _value(value=0.0)),))).fingerprint


@pytest.mark.parametrize("role", ["loss", "reward"])
def test_loss_and_reward_require_explicit_transform_identity(role):
    fields = _value(role=role).model_dump()
    fields["transform"] = None
    with pytest.raises(ValueError, match="transform reference"):
        OutcomeValue.model_validate(fields)
    assert _value(role=role).transform == _ref("identity-loss")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "0.5"])
def test_feedback_numeric_guards_apply_before_any_update(value):
    with pytest.raises(ValueError):
        _value(value=value)


def test_revisions_are_explicit_and_local_parent_presence_is_not_invented():
    correction = _batch(revision_id="corrected", supersedes="initial")
    assert correction.supersedes == "initial"
    assert deduplicate_feedback((correction,)) == (correction,)
    with pytest.raises(ValueError, match="supersede itself"):
        _batch(supersedes="initial")
    for field in ("supersedes", "horizon"):
        fields = _batch().model_dump()
        del fields[field]
        with pytest.raises(ValueError):
            FeedbackBatch.model_validate(fields)


@pytest.mark.parametrize("delay", [-1, True, "1", 0.5])
def test_logical_delay_rejects_negative_or_coerced_values(delay):
    with pytest.raises(ValueError):
        _batch(delay_rounds=delay)


def test_unknown_receipt_stays_unknown_in_optional_timing():
    timing = FeedbackTiming.model_validate(
        {
            "event_at": "2026-01-01T10:00:00Z",
            "available_at": "2026-01-01T10:01:00Z",
            "mature_at": "2026-01-01T10:02:00Z",
            "received_at": None,
        }
    )
    restored = FeedbackBatch.from_json(_batch(timing=timing).to_json())
    assert restored.timing is not None
    assert restored.timing.received_at is None
    assert restored.timing.usable_at is None


def test_aggregate_and_component_coverage_remain_distinct():
    aggregate = _batch(AggregateFeedback(values=(_value("total-cost", 3.0, role="resource"),)))
    components = _batch(ComponentFeedback(rows=(_row("A"),), completeness="partial"))
    assert isinstance(FeedbackBatch.from_json(aggregate.to_json()).feedback, AggregateFeedback)
    assert isinstance(FeedbackBatch.from_json(components.to_json()).feedback, ComponentFeedback)
    assert aggregate.fingerprint != components.fingerprint


def test_duplicate_retries_are_idempotent_and_canonical_by_logical_key():
    original = _batch()
    correction = _batch(revision_id="next", supersedes="initial")
    result = deduplicate_feedback((correction, original, original, correction))
    assert result == (original, correction)
    assert deduplicate_feedback(result) == result
    assert deduplicate_feedback(()) == ()


def test_changed_content_and_signed_zero_at_one_revision_conflict():
    positive = _batch(ChosenFeedback(row=_row("A", _value(value=0.0))))
    negative = _batch(ChosenFeedback(row=_row("A", _value(value=-0.0))))
    assert positive == negative  # Python numeric equality is weaker than canonical content equality.
    assert positive.to_json() != negative.to_json()
    with pytest.raises(ValueError, match="Conflicting feedback content"):
        deduplicate_feedback((positive, negative))


def test_unchecked_nested_copies_are_revalidated_on_retry_check():
    original = _batch(ChosenFeedback(row=_row()))
    forged = original.model_copy(update={"schema_version": 2})
    with pytest.raises(ValueError):
        deduplicate_feedback((forged,))
    with pytest.raises(TypeError, match="typed FeedbackBatch"):
        deduplicate_feedback((cast(Any, original.model_dump()),))


def test_retry_bound_counts_attempts_including_identical_duplicates():
    record = _batch()
    consumed = 0

    def attempts():
        nonlocal consumed
        while True:
            consumed += 1
            yield record

    with pytest.raises(ValueError, match="4096 attempted"):
        deduplicate_feedback(attempts())
    assert consumed == MAX_MEMBERS + 1
