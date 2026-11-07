"""Producing joins use old snapshots and unchanged core feedback semantics."""

import json
from datetime import timedelta

import pytest

from coleman.core.actions import AbstainAction, ChoiceAction, ChoiceSpace
from coleman.core.capabilities import Capability
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import AbsentFeedback, FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.numeric import ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming
from coleman.core.units import UnitSemantics
from coleman.registry.definitions import RecipeBinding
from coleman.registry.envelopes import RegisteredExecution, RegisteredFeedback, RegistryRequest, seal_request
from coleman.registry.lifecycle import RegistryState
from tests.registry.test_lifecycle import FIXTURES, START, fact, history, ref


def registered_request(state, round_number=0, decision_id="opportunity-0", **changes):
    snapshot = state.snapshot(round_number, START if state.policy.clock == "timed" else None)
    timing = None
    if snapshot.cutoff is not None:
        timing = DecisionTiming(
            decision_at=START,
            information_cutoff=START,
            target_start=START,
            target_end=START,
            eligible_input_fingerprint="sha256:" + "e" * 64,
            mode="future",
        )
    request = DecisionRequest(
        schema_version=1,
        task=state.policy.task,
        decision_id=decision_id,
        round=round_number,
        space=ChoiceSpace(feasible_set=snapshot.feasible_set("public-space"), members=snapshot.active),
        objective=ref("toy-loss"),
        timing=timing,
    )
    return RegistryRequest.model_validate(dict(schema_version=1, request=request, snapshot=snapshot) | changes)


def execution(request):
    space = request.request.space
    action = (
        ChoiceAction(schema_version=1, space=space, chosen=space.members[0])
        if space.members
        else AbstainAction(schema_version=1, space=space, reason="empty-active-population")
    )
    sampling = SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=ref("public-first-canonical"),
        status="not_required",
        action_fingerprint=action.fingerprint,
        probability=None,
    )
    decision = Decision(
        schema_version=1,
        request=request.request,
        implementation=ref("public-toy-consumer"),
        action=action,
        sampling=sampling,
        state_owner="host",
    )
    actual = ExecutedDecision(
        schema_version=1,
        proposal=decision,
        action=action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=sampling,
    )
    return RegisteredExecution(schema_version=1, request=request, executed=actual)


def loss_spec():
    return ScalarSpec(
        dimension=ref("loss"),
        semantics=UnitSemantics(unit="1", meaning="public-bounded-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
    )


def capability(**changes):
    return Capability.model_validate(
        dict(
            schema_version=1,
            implementation=ref("public-toy-consumer"),
            action_kinds=("choice",),
            feedback_kinds=("full_panel", "absent"),
            required_values=(loss_spec(),),
            origins=("synthetic",),
            timing_modes=("untimed",),
            supports_delayed_feedback=True,
            requires_known_receipt=False,
            requires_observed=True,
            state_owner="host",
            requires_sampling=False,
            sampling_requirement="not_required",
        )
        | changes
    )


def feedback(producer, values=None):
    rows = []
    for index, member in enumerate(producer.request.snapshot.active):
        value = (0.25, 0.75)[index] if values is None else values[index]
        outcome = OutcomeValue(
            spec=loss_spec(),
            value=value,
            status="unavailable" if value is None else "observed",
            origin="synthetic",
            transform=ref("identity-loss"),
        )
        rows.append(FeedbackRow(member=member, values=(outcome,)))
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=producer.executed.fingerprint,
        feedback_id="public-feedback",
        revision_id="initial",
        supersedes=None,
        horizon=None,
        feedback=FullPanel(rows=tuple(rows)) if rows else AbsentFeedback(reason="empty-active"),
    )
    return RegisteredFeedback(schema_version=1, producer_fingerprint=producer.fingerprint, batch=batch)


def test_outer_envelopes_roundtrip_without_changing_core_bytes_or_requiring_metadata_for_direct_callers():
    request = registered_request(history(count=2))
    producer = execution(request)
    observation = feedback(producer)
    observation.validate_for(producer, capability())
    assert RegistryRequest.from_json(request.to_json()) == request
    assert RegisteredExecution.from_json(producer.to_json()) == producer
    assert RegisteredFeedback.from_json(observation.to_json()) == observation
    assert DecisionRequest.from_json(request.request.to_json()).to_json() == request.request.to_json()
    assert ExecutedDecision.from_json(producer.executed.to_json()).to_json() == producer.executed.to_json()
    assert FeedbackBatch.from_json(observation.batch.to_json()).to_json() == observation.batch.to_json()


def test_sleeping_member_has_no_row_but_active_unavailable_remains_null_and_rejects_observed_update():
    sleeping = execution(registered_request(history(), 1))
    observation = feedback(sleeping, (0.5,))
    observation.validate_for(sleeping, capability())
    assert [row.member.id for row in observation.batch.feedback.rows] == ["A"]
    active = execution(registered_request(history(), 2))
    missing = feedback(active, (0.25, None))
    assert missing.batch.feedback.rows[1].values[0].value is None
    assert missing.batch.feedback.rows[1].values[0].status == "unavailable"
    selected_capability = capability()
    with pytest.raises(ValueError):
        missing.validate_for(active, selected_capability)
    missing.validate_for(active, capability(requires_observed=False))
    omitted = FullPanel(rows=(missing.batch.feedback.rows[0],))
    incomplete = RegisteredFeedback.model_validate(
        missing.model_dump()
        | {
            "batch": missing.batch.model_dump() | {"feedback": omitted},
        }
    )
    with pytest.raises(ValueError, match="Full panel"):
        incomplete.validate_for(active, selected_capability)


def test_retired_and_restarted_delayed_feedback_always_joins_original_outer_producer_and_epoch():
    state = history()
    original = execution(registered_request(state, 2))
    replacement = execution(registered_request(state, 4))
    delayed = feedback(original)
    delayed.validate_for(original, capability())
    assert next(row.state_epoch for row in original.request.snapshot.members if row.member.id == "B") == ref("B:e0")
    assert next(row.state_epoch for row in replacement.request.snapshot.members if row.member.id == "B") == ref("B:e1")
    selected_capability = capability()
    with pytest.raises(ValueError, match="outer producer"):
        delayed.validate_for(replacement, selected_capability)
    bad_inner = RegisteredFeedback.model_validate(
        delayed.model_dump()
        | {
            "batch": delayed.batch.model_dump() | {"decision_fingerprint": replacement.executed.fingerprint},
        }
    )
    with pytest.raises(ValueError, match="executed decision"):
        bad_inner.validate_for(original, selected_capability)


def test_empty_active_uses_existing_abstain_and_no_fabricated_full_panel_or_update():
    producer = execution(registered_request(history(), 5))
    assert isinstance(producer.executed.action, AbstainAction)
    assert not producer.request.snapshot.active
    observation = feedback(producer)
    assert isinstance(observation.batch.feedback, AbsentFeedback)
    observation.validate_for(producer, capability())
    with pytest.raises(ValueError):
        FullPanel(rows=())


def test_exact_retry_and_conflicting_round_outer_recipes_and_sealed_fact_admission():
    state = history(count=1)
    original = registered_request(state)
    sealed = seal_request(state, original)
    before = sealed.to_json()
    assert seal_request(sealed, RegistryRequest.from_json(original.to_json())) == sealed
    assert sealed.snapshot(0) == original.snapshot
    later_same_id = registered_request(sealed, 1)
    assert original.request.logical_identity == later_same_id.request.logical_identity
    with pytest.raises(ValueError, match="Conflicting registered retry"):
        seal_request(sealed, later_same_id)
    retrospective = fact(sealed, name="B")
    with pytest.raises(ValueError, match="prospectively"):
        sealed.append(retrospective)
    next_state = sealed.append(fact(sealed, name="B", round_number=1))
    assert seal_request(next_state, original) == next_state
    with pytest.raises(ValueError, match="sealed producing snapshot"):
        next_state.snapshot(0)
    assert sealed.to_json() == before
    recipe = RecipeBinding(
        schema_version=1,
        child=ref("child-A"),
        task=state.policy.task,
        expert=state.catalogue.resolve("expert", ref("A")).binding,
        state_epoch=ref("A:e0"),
        owners=(),
        objective=original.request.objective,
        target=None,
        loss_transform=None,
        controls=None,
    )
    changed_outer = RegistryRequest.model_validate(original.model_dump() | {"recipes": (recipe,)})
    with pytest.raises(ValueError, match="Conflicting registered retry"):
        seal_request(sealed, changed_outer)


def test_wrong_task_epoch_space_or_snapshot_rejects_before_host_work():
    state = history(count=2)
    original = registered_request(state)
    for changes in (
        {"task": ref("wrong-child")},
        {"round": 1},
        {"space": ChoiceSpace(feasible_set=ref("public-space"), members=original.snapshot.active)},
        {"space": ChoiceSpace(feasible_set=original.snapshot.feasible_set("public-space"), members=(ref("A"),))},
    ):
        candidate = original.model_dump() | {"request": original.request.model_dump() | changes}
        with pytest.raises(ValueError):
            RegistryRequest.model_validate(candidate)
    timed = registered_request(history(count=1, clock="timed"))
    without_timing = timed.model_dump() | {"request": timed.request.model_dump() | {"timing": None}}
    with pytest.raises(ValueError, match="timing"):
        RegistryRequest.model_validate(without_timing)
    changed = execution(registered_request(history(count=2), 1, "other-opportunity"))
    with pytest.raises(ValueError, match="exact original core request"):
        RegisteredExecution(schema_version=1, request=original, executed=changed.executed)


def test_recipe_binding_matches_active_expert_and_never_conflates_action_epoch():
    state = history(count=1)
    original = registered_request(state)
    recipe = RecipeBinding(
        schema_version=1,
        child=ref("child-A"),
        task=state.policy.task,
        expert=state.catalogue.resolve("expert", ref("A")).binding,
        state_epoch=ref("A:e0"),
        owners=(),
        objective=original.request.objective,
        target=None,
        loss_transform=None,
        controls=None,
    )
    assert registered_request(state, recipes=(recipe,)).recipes == (recipe,)
    for change in (
        {"state_epoch": ref("A:stale")},
        {"task": ref("wrong-task")},
        {"objective": ref("wrong-objective")},
        {"expert": state.catalogue.resolve("expert", ref("B")).binding},
    ):
        candidate_recipe = RecipeBinding.model_validate(recipe.model_dump() | change)
        candidate = original.model_dump() | {"recipes": (candidate_recipe,)}
        with pytest.raises(ValueError, match="exact active expert"):
            RegistryRequest.model_validate(candidate)
    values = state.model_dump(mode="json")
    for leaf in values["catalogue"]["definitions"]:
        leaf["kind"] = "action"
    values["facts"] = []
    values["policy"]["kind"] = "action"
    action_state = RegistryState.model_validate(values)
    action_state = action_state.append(fact(action_state))
    assert not registered_request(action_state).recipes
    with pytest.raises(ValueError, match="external ownership"):
        registered_request(action_state, recipes=(recipe,))


def test_timed_sealed_cutoffs_cannot_move_backward_with_a_new_opportunity():
    state = history(count=1, clock="timed")
    original = registered_request(state)
    state = seal_request(state, original)
    earlier = START - timedelta(microseconds=1)
    snapshot = state.snapshot(1, earlier)
    timing = DecisionTiming.model_validate(original.request.timing.model_dump() | {"information_cutoff": earlier})
    request = DecisionRequest.model_validate(
        original.request.model_dump()
        | {
            "round": 1,
            "decision_id": "new-opportunity",
            "timing": timing,
            "space": ChoiceSpace(feasible_set=snapshot.feasible_set("public-space"), members=snapshot.active),
        }
    )
    envelope = RegistryRequest(schema_version=1, request=request, snapshot=snapshot)
    before = state.to_json()
    with pytest.raises(ValueError, match="cutoffs must be nondecreasing"):
        seal_request(state, envelope)
    assert state.to_json() == before


def test_restored_compact_seals_reject_rewritten_policy_history_and_retrospective_suffix_facts():
    literal = json.loads((FIXTURES / "sealed-history.json").read_text(encoding="utf-8"))
    original = RegistryState.model_validate(literal)
    for key in ("reason", "event_id"):
        changed = json.loads(json.dumps(literal))
        changed["facts"][0][key] = "altered-sealed-prefix"
        with pytest.raises(ValueError, match="exact causal prefix"):
            RegistryState.model_validate(changed)
    changed = json.loads(json.dumps(literal))
    changed["policy"]["population"]["id"] = "other-population"
    with pytest.raises(ValueError, match="exact causal prefix"):
        RegistryState.model_validate(changed)
    changed = json.loads(json.dumps(literal))
    changed["facts"][2]["admitted_before_round"] = 0
    with pytest.raises(ValueError, match="after a sealed prefix"):
        RegistryState.model_validate(changed)
    request = RegistryRequest.from_json((FIXTURES / "request.json").read_bytes())
    assert seal_request(original, request) == original
    assert original.to_json() == (FIXTURES / "sealed-history.json").read_text(encoding="utf-8").rstrip("\n")


def test_restored_seals_cannot_rewind_an_already_accepted_scheduled_future_prefix():
    state = history(count=1)
    state = state.append(fact(state, name="B", round_number=10))
    sealed = seal_request(state, registered_request(state))
    prior = sealed.producers[0]
    for round_number in (0, 1):
        rewind = prior.model_dump() | {
            "logical_identity": "sha256:" + "f" * 64,
            "round": round_number,
            "fact_count": 1,
            "history_fingerprint": history(count=1).history_fingerprint,
        }
        candidate = sealed.model_dump() | {"producers": (*sealed.producers, rewind)}
        with pytest.raises(ValueError, match="fact_count|Same-round"):
            RegistryState.model_validate(candidate)
