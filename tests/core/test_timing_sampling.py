"""Explicit clock eligibility and selection-evidence truth-table regressions."""

import math
from datetime import UTC, datetime
from fractions import Fraction

import pytest

from coleman.core.actions import (
    AbstainAction,
    AcquisitionAction,
    ChoiceAction,
    MixtureAction,
    RankingAction,
    SlateAction,
    SubsetAction,
    VectorAction,
)
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.core.sampling import SamplingRecord, validate_sampling
from coleman.core.timing import DecisionTiming, FeedbackTiming


def ref(name):
    return Reference(id=name, version="1")


def action_for(kind):
    members = (ref("a"), ref("b"))
    space = {"kind": kind, "feasible_set": ref("universe"), "members": members}
    common = {"schema_version": 1, "space": space}
    if kind == "choice":
        return ChoiceAction.model_validate({**common, "chosen": members[0]})
    if kind == "ranking":
        return RankingAction.model_validate({**common, "order": members})
    if kind in {"subset", "slate"}:
        space.update({"minimum_selected": 0, "maximum_selected": 2})
        model = SubsetAction if kind == "subset" else SlateAction
        return model.model_validate({**common, "selected": (members[0],)})
    if kind == "mixture":
        space["profile"] = "binary64-simplex-v1"
        return MixtureAction.model_validate(
            {**common, "weights": ({"member": members[0], "value": 0.5}, {"member": members[1], "value": 0.5})}
        )
    if kind == "vector":
        return VectorAction.model_validate(
            {
                "schema_version": 1,
                "space": {"kind": "vector", "feasible_set": ref("vector"), "coordinates": (), "signed": True},
                "values": (),
            }
        )
    if kind == "acquisition":
        space["fidelities"] = (ref("terminal"),)
        return AcquisitionAction.model_validate({**common, "chosen": members[0], "fidelity": ref("terminal")})
    if kind == "abstain":
        space["kind"] = "choice"
        return AbstainAction.model_validate({**common, "reason": "insufficient-input"})
    raise AssertionError(kind)


def evidence(action, **changes):
    return SamplingRecord.model_validate(
        {
            "mechanism": "sampled",
            "mechanism_ref": ref("owned-categorical-mechanism"),
            "status": "exact",
            "action_fingerprint": action.fingerprint,
            "probability": 0.25,
            **changes,
        }
    )


def timing(**changes):
    return DecisionTiming.model_validate(
        {
            "decision_at": "2026-01-02T10:00:00Z",
            "information_cutoff": "2026-01-02T09:00:00Z",
            "target_start": "2026-01-02T10:00:00Z",
            "target_end": "2026-01-02T11:00:00Z",
            "eligible_input_fingerprint": fingerprint({"owned": "eligible-past-view"}),
            "mode": "future",
            **changes,
        }
    )


def feedback_timing(**changes):
    return FeedbackTiming.model_validate(
        {
            "event_at": "2026-01-02T10:00:00Z",
            "available_at": "2026-01-02T10:01:00Z",
            "mature_at": "2026-01-02T10:03:00Z",
            "received_at": "2026-01-02T10:02:00Z",
            **changes,
        }
    )


@pytest.mark.parametrize("kind", ["choice", "ranking", "subset", "slate"])
@pytest.mark.parametrize("mechanism,probability", [("sampled", 0.25), ("deterministic", 1.0)])
def test_exact_action_mass_is_supported_only_for_declared_finite_shapes(kind, mechanism, probability):
    action = action_for(kind)
    record = evidence(action, mechanism=mechanism, probability=probability)
    validate_sampling(action, record)
    assert SamplingRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("kind", ["mixture", "vector", "acquisition", "abstain"])
@pytest.mark.parametrize("status", ["exact", "qualified_approximation"])
def test_allocation_weights_density_and_acquisition_are_not_action_mass(kind, status):
    action = action_for(kind)
    extras = {"error_bound": 0.01, "qualification": ref("finite-error-bound")} if status != "exact" else {}
    record = evidence(action, status=status, **extras)
    with pytest.raises(ValueError):
        validate_sampling(action, record)


@pytest.mark.parametrize(
    "kind", ["choice", "ranking", "subset", "slate", "mixture", "vector", "acquisition", "abstain"]
)
@pytest.mark.parametrize("mechanism", ["deterministic", "sampled"])
@pytest.mark.parametrize("status", ["not_required", "unavailable"])
def test_unavailable_and_not_required_remain_explicit_null_states(kind, mechanism, status):
    action = action_for(kind)
    reason = "host-cannot-reconstruct-mechanism" if status == "unavailable" else None
    record = evidence(action, mechanism=mechanism, status=status, probability=None, reason=reason)
    validate_sampling(action, record)
    assert record.probability is None and record.status == status


@pytest.mark.parametrize(
    "changes",
    [
        {"probability": 0.0},
        {"probability": -0.0},
        {"probability": -0.1},
        {"probability": 1.1},
        {"probability": None},
        {"probability": True},
        {"mechanism": "deterministic", "probability": 0.9},
        {"status": "unavailable", "probability": None},
        {"status": "not_required", "probability": None, "reason": "not-needed"},
        {"status": "exact", "error_bound": 0.01},
        {"status": "exact", "qualification": ref("unused")},
        {"status": "qualified_approximation"},
        {"status": "qualified_approximation", "error_bound": 0.0, "qualification": ref("bound")},
    ],
)
def test_sampling_status_field_truth_table_rejects_ambiguity(changes):
    with pytest.raises(ValueError):
        evidence(action_for("choice"), **changes)


def test_qualified_error_cannot_exclude_known_deterministic_mass():
    with pytest.raises(ValueError):
        evidence(
            action_for("choice"),
            mechanism="deterministic",
            status="qualified_approximation",
            probability=0.1,
            error_bound=0.01,
            qualification=ref("bound-that-does-not-cover-one"),
        )


def test_deterministic_error_boundary_uses_exact_stored_probability_and_bound():
    probability = 0.1
    accepted_bound = 0.9
    rejected_bound = math.nextafter(accepted_bound, 0.0)
    exact_error = 1 - Fraction.from_float(probability)
    assert Fraction.from_float(rejected_bound) < exact_error <= Fraction.from_float(accepted_bound)
    action = action_for("choice")
    accepted = evidence(
        action,
        mechanism="deterministic",
        status="qualified_approximation",
        probability=probability,
        error_bound=accepted_bound,
        qualification=ref("known-mass-bound"),
    )
    validate_sampling(action, accepted)
    with pytest.raises(ValueError):
        evidence(
            action,
            mechanism="deterministic",
            status="qualified_approximation",
            probability=probability,
            error_bound=rejected_bound,
            qualification=ref("known-mass-bound"),
        )


def test_qualified_sampled_mass_declares_error_and_binding_without_authenticating_it():
    action = action_for("choice")
    approximate = evidence(
        action, status="qualified_approximation", error_bound=0.01, qualification=ref("declared-error-profile")
    )
    validate_sampling(action, approximate)
    changed = action.model_copy(update={"chosen": ref("b")})
    with pytest.raises(ValueError, match="content"):
        validate_sampling(changed, approximate)
    corrupted = approximate.model_copy(update={"error_bound": -0.1})
    with pytest.raises(ValueError):
        validate_sampling(action, corrupted)


@pytest.mark.parametrize(
    "changes",
    [
        {"information_cutoff": "2026-01-02T10:00:00.000001Z"},
        {"target_start": "2026-01-02T09:59:59.999999Z"},
        {"target_end": "2026-01-02T09:59:59Z"},
        {"mode": "ongoing", "target_start": "2026-01-02T10:00:00.000001Z"},
        {"mode": "ongoing", "target_start": "2026-01-02T09:00:00Z", "target_end": "2026-01-02T09:59:59Z"},
    ],
)
def test_target_clock_modes_reject_wrong_information_and_interval_order(changes):
    with pytest.raises(ValueError):
        timing(**changes)


def test_future_and_ongoing_target_boundaries_are_closed_and_explicit():
    future = timing(information_cutoff="2026-01-02T10:00:00Z")
    assert future.information_cutoff == future.decision_at == future.target_start
    ongoing = timing(mode="ongoing", target_start="2026-01-02T09:00:00Z")
    assert ongoing.target_start < ongoing.decision_at < ongoing.target_end
    endpoint = timing(mode="ongoing", target_start="2026-01-02T09:00:00Z", target_end="2026-01-02T10:00:00Z")
    assert endpoint.target_end == endpoint.decision_at
    assert DecisionTiming.model_validate_json(ongoing.model_dump_json()) == ongoing


@pytest.mark.parametrize("field", ["decision_at", "information_cutoff", "target_start", "target_end"])
@pytest.mark.parametrize("value", [0, "1767348000", "2026-01-02T10:00:00"])
def test_no_epoch_scale_or_timezone_is_inferred(field, value):
    with pytest.raises(ValueError):
        timing(**{field: value})


def test_feedback_usable_time_is_latest_required_clock_and_unknown_receipt_stays_unknown():
    sample = feedback_timing()
    assert sample.usable_at == datetime(2026, 1, 2, 10, 3, tzinfo=UTC)
    late = feedback_timing(received_at="2026-01-02T11:00:00Z")
    assert late.usable_at == late.received_at
    unknown = feedback_timing(received_at=None)
    assert unknown.usable_at is None and unknown.received_at is None
    payload = unknown.model_dump(mode="json")
    del payload["received_at"]
    with pytest.raises(ValueError):
        FeedbackTiming.model_validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"available_at": "2026-01-02T09:59:59Z"},
        {"mature_at": "2026-01-02T09:59:59Z"},
        {"received_at": "2026-01-02T10:00:59.999999Z"},
    ],
)
def test_feedback_clocks_cannot_rewrite_event_or_availability(changes):
    with pytest.raises(ValueError):
        feedback_timing(**changes)


def test_timezone_offsets_normalize_without_consulting_host_clock():
    first = timing()
    equivalent = timing(decision_at="2026-01-02T11:00:00+01:00")
    assert equivalent == first
    assert equivalent.decision_at.tzinfo is UTC


def test_unchanged_execution_cannot_change_selection_evidence():
    action = action_for("choice")
    request = DecisionRequest.model_validate(
        {
            "schema_version": 1,
            "task": ref("owned-decision-task"),
            "decision_id": "decision-1",
            "round": 0,
            "space": action.space,
            "objective": ref("objective"),
        }
    )
    record = evidence(action)
    proposal = Decision(
        schema_version=1,
        request=request,
        implementation=ref("host-owned-method"),
        action=action,
        sampling=record,
        state_owner="host",
    )
    accepted = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=record,
    )
    assert ExecutedDecision.from_json(accepted.to_json()) == accepted
    with pytest.raises(ValueError):
        ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=action,
            mapping="unchanged",
            sampling_disposition="recomputed",
            sampling=evidence(action, probability=0.9),
        )
