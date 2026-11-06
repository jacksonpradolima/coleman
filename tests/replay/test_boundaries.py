"""Public recovery and rejection seams preserve a real host before unsafe work."""

from typing import Literal, cast

import pytest

from coleman.core.base import CoreContract
from coleman.core.feedback import AbsentFeedback
from coleman.core.splits import TemporalObservation
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.context import ObjectiveDefinition, ObjectiveTerm, SemanticContext
from coleman.objectives.transforms import IdentityLoss
from coleman.observations.contracts import ObservationHistory, ObservationRevision
from coleman.replay.contracts import (
    PreparedDecision,
    ReplayDecisionRecord,
    ReplayDrain,
    ReplayFeedback,
    ReplayOpportunity,
    ValueOriginEvidence,
)
from coleman.replay.engine import CausalReplay, ReplayInvalidatedError
from coleman.replay.revisions import feedback_chains, latest_visible
from coleman.results.domain import Measurement
from tests.replay.support import emit, session, set_clocks


def history_with_duration(example, *, value=1.0, available=100):
    revision = ObservationRevision(
        schema_version=1,
        metadata=TemporalObservation(
            observation_id="boundary-ci-duration",
            revision_id="root",
            group_id="boundary-ci-family",
            event_at=example.instant(0),
            available_at=example.instant(available),
            target_end=example.instant(0),
        ),
        entity_id="suite-a",
        received_at=example.instant(available),
        supersedes=None,
        measurement=Measurement(
            name="duration",
            value=value,
            status="observed",
            unit="s",
            meaning="authored-ci-duration",
            direction="minimize",
            minimum=0,
        ),
    )
    return ObservationHistory(
        schema_version=1, source_id=example.SOURCE.id, source_version=example.SOURCE.version, revisions=(revision,)
    )


def semantic_context(example):
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    quantity = QuantityMeaning(spec=example.LOSS, interpretation="normalized_loss", horizon=example.HORIZON)
    transform = IdentityLoss(
        schema_version=1, reference=example.reference("authored-ci-loss-identity"), lineage=lineage, quantity=quantity
    )
    objective = ObjectiveDefinition(
        schema_version=1,
        reference=example.reference("minimize-authored-ci-loss"),
        lineage=lineage,
        terms=(ObjectiveTerm(quantity=quantity, direction="minimize", transform=transform.binding),),
    )
    return SemanticContext(
        schema_version=1,
        reference=example.reference("boundary-ci-context"),
        lineage=lineage,
        objective=objective,
        transforms=(transform,),
    )


@pytest.mark.parametrize("state_type", [CoreContract, dict, None])
def test_constructor_requires_an_explicit_complete_snapshot_class_before_retaining_the_real_host(example, state_type):
    host = example.HostConsumer()
    before = host.export_state().to_json()
    policy = example.policy()
    codec = example.CODEC
    selected = cast(type[CoreContract], state_type)
    with pytest.raises(TypeError, match="concrete"):
        CausalReplay(host, policy, state_type=selected, state_codec=codec)
    assert host.export_state().to_json() == before


def test_child_policy_cannot_be_constructed_as_an_unverified_independent_origin(example):
    parent, _ = session(example)
    child = CausalReplay.fork_from_continuation(
        example.HostConsumer(),
        parent.export_continuation(),
        lineage=example.reference("boundary-child"),
        state_type=example.HostConsumerState,
        state_codec=example.CODEC,
    )
    fresh = example.HostConsumer()
    before = fresh.export_state().to_json()
    policy, state_type, codec = child.state.policy, example.HostConsumerState, example.CODEC
    with pytest.raises(ValueError, match="complete matched parent"):
        CausalReplay(fresh, policy, state_type=state_type, state_codec=codec)
    assert fresh.export_state().to_json() == before


@pytest.mark.parametrize("operation", ["prepare", "eligible_snapshot"])
def test_missing_history_rejects_before_a_source_view_or_decision_is_fabricated(example, operation):
    host = example.HostConsumer()
    replay = CausalReplay(host, example.policy(), state_type=example.HostConsumerState, state_codec=example.CODEC)
    argument = example.opportunity(0) if operation == "prepare" else example.instant(0)
    principal = getattr(replay, operation)
    with pytest.raises(ValueError, match="history"):
        principal(argument)
    assert host.export_state().decision_count == 0
    assert replay.state.history is None


def test_eligible_snapshot_selects_actual_history_without_exposing_future_duration_or_advancing_host(example):
    replay, host = session(example)
    source = history_with_duration(example)
    replay.admit_history(source)
    cutoff = example.instant(1)
    before = replay.export_continuation().to_json()
    selected = replay.eligible_snapshot(cutoff)
    assert selected.to_json() == source.as_of(cutoff).to_json()
    assert selected.observations == ()
    assert replay.export_continuation().to_json() == before
    assert host.export_state().decision_count == 0


def test_non_simulated_execution_profile_rejects_before_the_actual_decider_is_called(example):
    replay, host = session(example)
    prepared = replay.prepare(example.opportunity(0))
    unsupported = cast(Literal["offline_identity_v1"], "physical_execution")
    before = replay.export_continuation().to_json()
    with pytest.raises(ValueError, match="offline_identity_v1"):
        replay.emit(prepared, execution=unsupported)
    assert replay.export_continuation().to_json() == before
    assert host.export_state().decision_count == 0


def test_internally_bound_forged_prepared_view_still_must_match_actual_admitted_history(example):
    replay, host = session(example)
    prepared = replay.prepare(example.opportunity(0))
    snapshot = type(prepared.snapshot).model_validate(
        prepared.snapshot.model_copy(update={"source_id": "other-source"})
    )
    timing = prepared.opportunity.request.timing
    assert timing is not None
    timing = type(timing).model_validate(timing.model_copy(update={"eligible_input_fingerprint": snapshot.fingerprint}))
    request = type(prepared.opportunity.request).model_validate(
        prepared.opportunity.request.model_copy(update={"timing": timing})
    )
    opportunity = ReplayOpportunity.model_validate(prepared.opportunity.model_copy(update={"request": request}))
    forged = PreparedDecision.model_validate(
        prepared.model_copy(update={"opportunity": opportunity, "snapshot": snapshot})
    )
    before = replay.export_continuation().to_json()
    with pytest.raises(ValueError, match="actual source history"):
        replay.emit(forged, execution="offline_identity_v1")
    assert replay.export_continuation().to_json() == before
    assert host.export_state().decision_count == 0


def test_concrete_snapshot_class_drift_after_real_decide_invalidates_the_session(example):
    class AnotherSnapshot(example.HostConsumerState):
        """A different host codec despite identical field names."""

    class Drifting(example.HostConsumer):
        def export_state(self):
            result = super().export_state()
            if result.decision_count:
                return AnotherSnapshot.model_validate(result.model_dump())
            return result

    host = Drifting()
    replay, _ = session(example, host=host)
    prepared = replay.prepare(example.opportunity(0))
    with pytest.raises(TypeError, match="different concrete"):
        replay.emit(prepared, execution="offline_identity_v1")
    assert host.export_state().decision_count == 1
    with pytest.raises(ReplayInvalidatedError):
        replay.export_continuation()


def test_changed_returned_request_after_a_real_decide_is_not_recorded_as_the_producing_request(example):
    class Changing(example.HostConsumer):
        def decide(self, request, /):
            proposal = super().decide(request)
            changed = type(request).model_validate(request.model_copy(update={"objective": example.reference("other")}))
            return type(proposal).model_validate(proposal.model_copy(update={"request": changed}))

    host = Changing()
    replay, _ = session(example, host=host)
    prepared = replay.prepare(example.opportunity(0))
    with pytest.raises(ValueError, match="changed the exact producing request"):
        replay.emit(prepared, execution="offline_identity_v1")
    assert host.export_state().decision_count == 1
    with pytest.raises(ReplayInvalidatedError):
        replay.export_continuation()


@pytest.mark.parametrize("fault", ["source_id", "source_version", "deletion", "changed_value"])
def test_history_source_replacement_or_retained_revision_rewrite_never_resets_the_owner(example, fault):
    replay, host = session(example)
    original = history_with_duration(example)
    replay.admit_history(original)
    if fault == "deletion":
        changed = example.history()
    elif fault == "changed_value":
        changed = history_with_duration(example, value=2.0)
    else:
        changed = ObservationHistory.model_validate(original.model_copy(update={fault: "another"}))
    before = replay.export_continuation().to_json()
    with pytest.raises(ValueError, match="identity/version|byte-for-byte"):
        replay.admit_history(changed)
    assert replay.export_continuation().to_json() == before
    assert host.export_state().decision_count == 0


def test_backfill_can_leave_old_decisions_unchanged_but_still_cross_the_closed_processing_frontier(example):
    replay, host = session(example)
    emit(example, replay, 0, at=0, end=1)
    replay.advance(eligible_through=example.instant(10), apply_at=example.instant(10))
    backfill = history_with_duration(example, available=5)
    before = replay.export_continuation().to_json()
    with pytest.raises(ValueError, match="closed replay information frontier"):
        replay.admit_history(backfill)
    assert replay.export_continuation().to_json() == before
    assert host.export_state().decision_count == 1


def test_future_inventory_keeps_the_eligible_view_but_invalidates_a_prepared_owner_binding(example):
    replay, host = session(example)
    prepared = replay.prepare(example.opportunity(0))
    future = history_with_duration(example)
    replay.admit_history(future)
    cutoff = example.instant(0)
    assert replay.eligible_snapshot(cutoff).to_json() == prepared.snapshot.to_json()
    with pytest.raises(ValueError, match="stale"):
        replay.emit(prepared, execution="offline_identity_v1")
    assert host.export_state().decision_count == 0


@pytest.mark.parametrize("change", ["objective", "semantic_definition"])
def test_later_objective_or_semantic_definition_cannot_reinterpret_the_existing_live_lineage(example, change):
    replay, host = session(example)
    first = example.opportunity(0)
    context = semantic_context(example) if change == "semantic_definition" else None
    if context is not None:
        first = ReplayOpportunity.model_validate(first.model_copy(update={"context": context}))
    replay.emit(replay.prepare(first), execution="offline_identity_v1")
    second = example.opportunity(1)
    if change == "objective":
        request = type(second.request).model_validate(
            second.request.model_copy(update={"objective": example.reference("other-objective")})
        )
        second = ReplayOpportunity.model_validate(second.model_copy(update={"request": request}))
    else:
        assert context is not None
        context = SemanticContext.model_validate(
            context.model_copy(update={"reference": example.reference("other-semantic-definition")})
        )
        second = ReplayOpportunity.model_validate(second.model_copy(update={"context": context}))
    before = replay.export_continuation().to_json()
    message = "task/objective" if change == "objective" else "semantic-definition"
    with pytest.raises(ValueError, match=message):
        replay.prepare(second)
    assert replay.export_continuation().to_json() == before
    assert host.export_state().decision_count == 1


@pytest.mark.parametrize(
    "reason",
    [
        "not_drained",
        "not_visible",
        "unknown_receipt",
        "partial_components",
        "missing_required_value",
        "absent",
        "awaiting_target_or_maturity",
    ],
)
def test_pending_reason_never_becomes_an_adaptive_update_or_imputed_zero(example, reason):
    profile = "component" if reason == "partial_components" else "chosen"
    replay, host = session(example, profile)
    first = emit(example, replay, 0, profile=profile, at=0, end=0 if reason == "not_drained" else 1)
    feedback = example.feedback(
        first,
        (None if reason == "missing_required_value" else 0.0, 0.5, 1.0),
        profile=profile,
        visible=0 if reason == "not_drained" else (10 if reason == "not_visible" else 2),
        mature=20 if reason == "awaiting_target_or_maturity" else None,
        delay=0,
        completeness="partial" if reason == "partial_components" else "complete",
    )
    if reason == "unknown_receipt":
        feedback = set_clocks(feedback, received_at=None)
    elif reason == "absent":
        batch = type(feedback.batch).model_validate(
            feedback.batch.model_copy(update={"feedback": AbsentFeedback(reason="no-result")})
        )
        feedback = ReplayFeedback.model_validate(feedback.model_copy(update={"batch": batch, "origins": ()}))
    replay.admit_feedback((feedback,))
    if reason != "not_drained":
        replay.advance(eligible_through=example.instant(3), apply_at=example.instant(3))
    pending = replay.pending
    assert len(pending) == 1
    assert pending[0].reason == reason
    assert host.export_state().update_count == 0
    assert host.export_state().observed_counts == (0, 0, 0)
    assert host.export_state().loss_sums == (0.0, 0.0, 0.0)


def test_empty_feedback_graph_and_latest_visible_selection_do_not_fabricate_a_round(example):
    cutoff = example.instant(0)
    assert feedback_chains(()) == ()
    assert latest_visible((), cutoff) is None


@pytest.mark.parametrize("fault", ["clock", "counts", "empty_host_transition"])
def test_serialized_empty_drain_cannot_assert_impossible_progress(example, fault):
    replay, _ = session(example)
    emit(example, replay, 0)
    payload = replay.state.drains[0].model_dump()
    if fault == "clock":
        payload["eligible_through"] = example.instant(2)
    elif fault == "counts":
        payload["update_count_after"] = 1
    else:
        payload["host_after"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError):
        ReplayDrain.model_validate(payload)


@pytest.mark.parametrize("fault", ["request", "counter", "acceptance_time", "snapshot", "mapping"])
def test_isolated_decision_record_rejects_changed_producing_facts_before_restoration(example, fault):
    replay, _ = session(example)
    record = emit(example, replay, 0)
    payload = record.model_dump()
    if fault == "request":
        payload["executed"]["proposal"]["request"]["decision_id"] = "another-decision"
    elif fault == "counter":
        payload["decision_index"] = 1
    elif fault == "acceptance_time":
        payload["accepted_at"] = example.instant(1)
    elif fault == "snapshot":
        payload["snapshot"]["source_id"] = "another-source"
    else:
        payload["executed"]["mapping"] = "overridden"
    with pytest.raises(ValueError):
        ReplayDecisionRecord.model_validate(payload)


@pytest.mark.parametrize("fault", ["missing_model", "realized_with_model"])
def test_origin_annotation_requires_the_appropriate_explicit_producer_binding(example, fault):
    replay, _ = session(example)
    first = emit(example, replay, 0, end=1)
    feedback = example.feedback(first, (0.0, 0.5, 1.0), visible=2, delay=0)
    payload = feedback.origins[0].model_dump()
    if fault == "missing_model":
        payload["producer"] = None
    else:
        payload["origin"] = "realized"
    with pytest.raises(ValueError, match="model or generator binding"):
        ValueOriginEvidence.model_validate(payload)


def test_declared_semantic_objective_horizon_cannot_be_replaced_by_a_replay_target_horizon(example):
    context = semantic_context(example)
    valid = ReplayOpportunity.model_validate(example.opportunity(0).model_copy(update={"context": context}))
    target = valid.target.model_copy(update={"horizon": example.reference("different-horizon")})
    payload = valid.model_dump()
    payload["target"] = target.model_dump()
    with pytest.raises(ValueError, match="objective horizons"):
        ReplayOpportunity.model_validate(payload)


def test_delayed_delivery_declaration_cannot_override_a_synchronous_only_capability(example):
    capability = type(example.capability("chosen")).model_validate(
        example.capability("chosen").model_copy(update={"supports_delayed_feedback": False})
    )
    payload = example.policy().model_dump()
    payload["capability"] = capability.model_dump()
    policy_type = type(example.policy())
    with pytest.raises(ValueError, match="explicit delayed capability"):
        policy_type.model_validate(payload)
