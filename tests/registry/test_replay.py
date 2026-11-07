"""Registered preflight preserves real replay's fixed semantics and empty limit."""

import json

import pytest

from coleman.core.actions import ChoiceAction, space_members
from coleman.core.base import CoreContract
from coleman.core.decisions import DecisionRequest
from coleman.core.feedback import AbsentFeedback, FeedbackBatch
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.objectives.bindings import DefinitionLineage, QuantityMeaning
from coleman.objectives.context import ObjectiveDefinition, ObjectiveTerm, SemanticContext
from coleman.objectives.transforms import IdentityLoss
from coleman.observations.contracts import ObservationHistory
from coleman.registry.envelopes import RegistryRequest
from coleman.registry.replay import RegisteredReplayOpportunity
from coleman.replay.contracts import ReplayFeedback, ReplayOpportunity, ReplayPolicy, ReplayTarget
from coleman.replay.engine import CausalReplay
from tests.registry.test_envelopes import capability, execution, loss_spec, registered_request
from tests.registry.test_lifecycle import START, fact, history, ref


def source():
    return ObservationHistory(schema_version=1, source_id="public-ci-input", source_version="v1", revisions=())


def registered(state, round_number=0):
    original = registered_request(state, round_number, decision_id=f"round-{round_number}")
    timing = DecisionTiming.model_validate(
        original.request.timing.model_dump() | {"eligible_input_fingerprint": source().as_of(START).fingerprint}
    )
    request = DecisionRequest.model_validate(original.request.model_dump() | {"timing": timing})
    return RegistryRequest.model_validate(original.model_dump() | {"request": request})


def opportunity(request, context=None):
    return ReplayOpportunity(
        schema_version=1,
        request=request.request,
        context=context,
        target=ReplayTarget(reference=ref("one-ci-target"), horizon=ref("one-ci-round"), groups=()),
    )


def policy(state, **changes):
    return ReplayPolicy.model_validate(
        dict(
            schema_version=1,
            stream=state.policy.task,
            lineage=ref("ci-lineage"),
            parent=None,
            source=ref("public-ci-input"),
            capability=capability(timing_modes=("future",)),
            delivery="delayed",
            specialists=False,
        )
        | changes
    )


class HostState(CoreContract):
    calls: int


class FixedHost:
    """Actual host-owned deterministic protocol consumer for one opportunity."""

    def __init__(self, request):
        self.proposal = execution(request).executed.proposal
        self.state = HostState(schema_version=1, calls=0)

    def export_state(self):
        return self.state

    def restore_state(self, state):
        self.state = HostState.model_validate(state)

    def decide(self, request):
        assert request.to_json() == self.proposal.request.to_json()
        self.state = HostState(schema_version=1, calls=self.state.calls + 1)
        return self.proposal

    def update(self, _application):
        raise AssertionError("This emission test offers no outcome/update")


def test_actual_existing_replay_prepares_and_emits_the_unchanged_registered_core_request():
    state = history(count=2, clock="timed")
    request = registered(state)
    host = FixedHost(request)
    session = CausalReplay(host, policy(state), state_type=HostState, state_codec=ref("fixed-host-state"))
    session.admit_history(source())
    bound = RegisteredReplayOpportunity(
        schema_version=1,
        registered=request,
        opportunity=opportunity(request),
        policy=session.state.policy,
        first=None,
    )
    assert host.state.calls == 0
    record = session.emit(session.prepare(bound.opportunity), execution="offline_identity_v1")
    assert record.executed.proposal.request.to_json() == request.request.to_json()
    assert isinstance(record.executed.action, ChoiceAction)
    assert record.executed.action.chosen == ref("A")
    assert host.state.calls == 1
    assert RegisteredReplayOpportunity.from_json(bound.to_json()) == bound
    data = json.loads(bound.to_json())
    data["schema_version"] = 2
    serialized = json.dumps(data)
    with pytest.raises(ValueError):
        RegisteredReplayOpportunity.from_json(serialized)


@pytest.mark.parametrize("delivery", ["synchronous", "delayed"])
def test_empty_direct_view_is_explicitly_rejected_for_sync_and_only_pending_compatible_for_delay(delivery):
    state = history(count=0, clock="timed")
    request = registered(state)
    host = FixedHost(request)
    selected = policy(state, delivery=delivery)
    values = dict(schema_version=1, registered=request, opportunity=opportunity(request), policy=selected, first=None)
    if delivery == "synchronous":
        with pytest.raises(ValueError, match="cannot complete an empty"):
            RegisteredReplayOpportunity.model_validate(values)
        assert host.state.calls == 0
    else:
        bound = RegisteredReplayOpportunity.model_validate(values)
        session = CausalReplay(host, selected, state_type=HostState, state_codec=ref("fixed-host-state"))
        session.admit_history(source())
        record = session.emit(session.prepare(bound.opportunity), execution="offline_identity_v1")
        feedback = ReplayFeedback(
            schema_version=1,
            lineage=selected.lineage,
            target=bound.opportunity.target,
            origins=(),
            batch=FeedbackBatch(
                schema_version=1,
                decision_fingerprint=record.executed.fingerprint,
                feedback_id="explicit-absence",
                revision_id="initial",
                supersedes=None,
                horizon=bound.opportunity.target.horizon,
                timing=FeedbackTiming(event_at=START, available_at=START, mature_at=START, received_at=START),
                feedback=AbsentFeedback(reason="no-active-expert"),
            ),
        )
        session.admit_feedback((feedback,))
        session.advance(eligible_through=START, apply_at=START)
        assert session.state.decision_count == 1
        assert session.state.update_count == 0
        assert session.state.feedback[0].batch.feedback.kind == "absent"
        assert host.state.calls == 1


@pytest.mark.parametrize("specialists", [False, True])
def test_active_change_uses_existing_specialist_gate_without_conveying_a_theorem(specialists):
    original = history(count=2, clock="timed")
    first = opportunity(registered(original))
    sleeping = original.append(fact(original, "B", "inactive", 1, known_at=START))
    changed = registered(sleeping, 1)
    values = dict(
        schema_version=1,
        registered=changed,
        opportunity=opportunity(changed),
        policy=policy(original, specialists=specialists),
        first=first,
    )
    if specialists:
        actual = RegisteredReplayOpportunity.model_validate(values)
        assert space_members(actual.opportunity.request.space) == (ref("A"),)
    else:
        with pytest.raises(ValueError, match="specialist"):
            RegisteredReplayOpportunity.model_validate(values)


def test_unchanged_active_contract_survives_new_round_and_full_snapshot_identity():
    state = history(count=2, clock="timed")
    first, second = registered(state), registered(state, 1)
    assert first.snapshot.fingerprint != second.snapshot.fingerprint
    assert first.request.space == second.request.space
    bound = RegisteredReplayOpportunity(
        schema_version=1,
        registered=second,
        opportunity=opportunity(second),
        policy=policy(state),
        first=opportunity(first),
    )
    assert bound.opportunity.request.round == 1


def test_wrong_inner_request_rejects_before_any_replay_or_host_call():
    state = history(count=2, clock="timed")
    first, second = registered(state), registered(state, 1)
    second_opportunity = opportunity(second)
    selected_policy = policy(state)
    with pytest.raises(ValueError, match="exact original core request"):
        RegisteredReplayOpportunity(
            schema_version=1, registered=first, opportunity=second_opportunity, policy=selected_policy, first=None
        )


def context(version="v1"):
    meaning = QuantityMeaning(spec=loss_spec(), interpretation="normalized_loss", horizon=ref("one-ci-round"))
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    transform = IdentityLoss(schema_version=1, reference=ref("identity-loss"), lineage=lineage, quantity=meaning)
    objective = ObjectiveDefinition(
        schema_version=1,
        reference=ref("toy-loss"),
        lineage=lineage,
        terms=(ObjectiveTerm(quantity=meaning, direction="minimize", transform=transform.binding),),
    )
    return SemanticContext(
        schema_version=1,
        reference=ref("ci-context", version),
        lineage=lineage,
        objective=objective,
        transforms=(transform,),
    )


def test_specialist_declaration_keeps_existing_semantic_context_freeze():
    state = history(count=2, clock="timed")
    first, second = registered(state), registered(state, 1)
    second_opportunity = opportunity(second, context("v2"))
    selected_policy = policy(state, specialists=True)
    first_opportunity = opportunity(first, context())
    with pytest.raises(ValueError, match="semantic-definition"):
        RegisteredReplayOpportunity(
            schema_version=1,
            registered=second,
            opportunity=second_opportunity,
            policy=selected_policy,
            first=first_opportunity,
        )
