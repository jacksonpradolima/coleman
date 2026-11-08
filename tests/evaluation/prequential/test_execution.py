"""Real causal mixture → distinct finite action, retained retry and refusal cases."""

import json
from pathlib import Path
from typing import Any

import pytest

from coleman.core.actions import ChoiceAction, MixtureAction
from coleman.core.base import CoreContract
from coleman.evaluation.prequential.execution import (
    DomainExecutionContext,
    DomainStepPlan,
    PrequentialStep,
    execute_prequential_step,
)
from tests.evaluation.prequential.conformance_support import binding, changed
from tests.evaluation.prequential.execution_support import DriverDomainState, FiniteDomain, session, step_plan


def test_actual_mixture_drives_distinct_projected_action_with_complete_prior_state():
    ci, direct, replay = session()
    domain = FiniteDomain()
    prior = domain.export_state().fingerprint
    result = execute_prequential_step(replay, domain, ci.opportunity(direct, 0, 0), plan=step_plan())
    assert result.coefficient.execution == "offline_identity_v1"
    assert result.coefficient.executed.action.kind == "mixture"
    assert isinstance(result.domain.executed.proposal.action, ChoiceAction)
    assert isinstance(result.domain.executed.action, ChoiceAction)
    assert result.domain.executed.proposal.action.chosen.id == "full"
    assert result.domain.executed.action.chosen.id == "fast"
    assert result.domain.executed.mapping == "projected"
    assert {member.id for member in result.domain.feasibility.feasible_members} == {"fast"}
    assert result.domain.prior_domain_state_fingerprint == prior
    assert result.domain.coefficient_record_fingerprint == result.coefficient.fingerprint
    assert result.domain.coefficient_execution_fingerprint == result.coefficient.executed.fingerprint
    state = direct.export_state()
    assert (state.decision_count, state.update_count, domain.state.calls) == (1, 0, 1)
    assert PrequentialStep.from_json(result.to_json()).to_json() == result.to_json()
    retained = domain.execution(step_plan().key)
    assert retained is not None
    assert retained.to_json() == result.domain.to_json()


def test_retained_retry_after_progress_preserves_original_prior_and_all_complete_bytes():
    ci, direct, replay = session()
    domain = FiniteDomain()
    first_opportunity = ci.opportunity(direct, 0, 0)
    original = execute_prequential_step(replay, domain, first_opportunity, plan=step_plan())
    execute_prequential_step(replay, domain, ci.opportunity(direct, 1, 1), plan=step_plan(1))
    expected_replay = replay.export_continuation().to_json()
    expected_domain = domain.export_state().to_json()
    repeated = execute_prequential_step(replay, domain, first_opportunity, plan=step_plan())
    assert repeated.to_json() == original.to_json()
    assert replay.export_continuation().to_json() == expected_replay
    assert domain.export_state().to_json() == expected_domain
    assert (direct.export_state().decision_count, domain.state.calls) == (2, 2)
    assert original.domain.prior_domain_state_fingerprint != domain.export_state().fingerprint


@pytest.mark.parametrize("field", ["adapter", "data_fingerprint", "base_configuration_fingerprint"])
def test_changed_retained_law_or_source_is_rejected_before_any_new_decision(field):
    ci, direct, replay = session()
    domain = FiniteDomain()
    opportunity = ci.opportunity(direct, 0, 0)
    execute_prequential_step(replay, domain, opportunity, plan=step_plan())
    expected_replay = replay.export_continuation().to_json()
    expected_domain = domain.export_state().to_json()
    value = binding("changed-adapter") if field == "adapter" else "sha256:" + "9" * 64
    incompatible_plan = changed(step_plan(), **{field: value})
    with pytest.raises(ValueError, match="predeclared"):
        execute_prequential_step(replay, domain, opportunity, plan=incompatible_plan)
    assert replay.export_continuation().to_json() == expected_replay
    assert domain.export_state().to_json() == expected_domain


def test_existing_domain_key_cannot_turn_unrelated_new_opportunity_into_a_retry():
    ci, direct, replay = session()
    domain = FiniteDomain()
    execute_prequential_step(replay, domain, ci.opportunity(direct, 0, 0), plan=step_plan())
    expected_replay = replay.export_continuation().to_json()
    expected_domain = domain.export_state().to_json()
    unrelated = ci.opportunity(direct, 1, 1)
    original_plan = step_plan()
    with pytest.raises(ValueError, match="matching coefficient"):
        execute_prequential_step(replay, domain, unrelated, plan=original_plan)
    assert replay.export_continuation().to_json() == expected_replay
    assert domain.export_state().to_json() == expected_domain


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("source", "predeclared"),
        ("base", "predeclared"),
        ("adapter", "predeclared"),
        ("producer", "producer"),
        ("execution", "producer"),
        ("prior", "pre-execution"),
        ("clock", "calendar"),
        ("target", "calendar"),
        ("input", "calendar"),
        ("round", "calendar"),
        ("unretained", "retain"),
    ],
)
def test_real_host_join_failure_propagates_after_coefficient_without_false_atomicity(fault, reason):
    ci, direct, replay = session()
    domain = FiniteDomain(fault=fault)
    opportunity = ci.opportunity(direct, 0, 0)
    plan = step_plan()
    with pytest.raises(ValueError, match=reason):
        execute_prequential_step(replay, domain, opportunity, plan=plan)
    # The direct helper documents ordered mutations; the accepted store boundary
    # separately discards fresh staged owners on failure rather than faking undo.
    assert direct.export_state().decision_count == 1
    assert domain.state.calls == 1


def test_domain_snapshot_capacity_refuses_before_coefficient_mutation():
    ci, direct, replay = session()
    domain = FiniteDomain(state=DriverDomainState(schema_version=1, padding="x" * (16 * 1024**2)))
    opportunity = ci.opportunity(direct, 0, 0)
    plan = step_plan()
    with pytest.raises(ValueError, match="budget"):
        execute_prequential_step(replay, domain, opportunity, plan=plan)
    assert (direct.export_state().decision_count, domain.state.calls) == (0, 0)


@pytest.mark.parametrize("state_kind", ["opaque", "base", "unbound"])
def test_missing_concrete_domain_codec_refuses_before_any_coefficient_or_domain_action(state_kind):
    class UnboundState[T](CoreContract):
        value: T

    class InvalidExporter:
        def execution(self, opportunity):
            return None

        def execute(self, context):
            pytest.fail("An invalid state cannot authorize a domain action")

        def export_state(self) -> Any:
            return {
                "opaque": {"schema_version": 1},
                "base": CoreContract(schema_version=1),
                "unbound": UnboundState(schema_version=1, value=0),
            }[state_kind]

        def restore_state(self, value):
            pytest.fail("The direct driver must not restore or create another owner")

    ci, direct, replay = session()
    invalid = InvalidExporter()
    opportunity = ci.opportunity(direct, 0, 0)
    plan = step_plan()
    with pytest.raises(TypeError, match="concrete"):
        execute_prequential_step(replay, invalid, opportunity, plan=plan)
    assert direct.export_state().decision_count == 0


def test_new_context_and_exchange_reject_future_versions_and_conflicting_full_producer():
    ci, direct, replay = session()
    result = execute_prequential_step(replay, FiniteDomain(), ci.opportunity(direct, 0, 0), plan=step_plan())
    context = DomainExecutionContext(
        schema_version=1,
        plan=step_plan(),
        coefficient=result.coefficient,
        prior_domain_state_fingerprint=result.domain.prior_domain_state_fingerprint,
    )
    for value in (step_plan(), context, result):
        wire = json.loads(value.to_json())
        wire["schema_version"] = 2
        serialized = json.dumps(wire)
        reader = type(value).from_json
        with pytest.raises(ValueError):
            reader(serialized)
    conflicting_domain = changed(result.domain, coefficient_record_fingerprint="sha256:" + "9" * 64)
    with pytest.raises(ValueError, match="producer"):
        changed(result, domain=conflicting_domain)
    assert DomainStepPlan.from_json(step_plan().to_json()).fingerprint == step_plan().fingerprint


def test_complete_v1_wire_goldens_and_independent_finite_ci_hand_fixture():
    root = Path(__file__).parent / "fixtures/execution-v1"
    expected = json.loads((root / "hand.json").read_bytes())
    ci, direct, replay = session()
    domain = FiniteDomain()
    plan = step_plan()
    result = execute_prequential_step(replay, domain, ci.opportunity(direct, 0, 0), plan=plan)
    context = DomainExecutionContext(
        schema_version=1,
        plan=plan,
        coefficient=result.coefficient,
        prior_domain_state_fingerprint=result.domain.prior_domain_state_fingerprint,
    )
    for name, value in (("plan", plan), ("context", context), ("step", result)):
        wire = (root / (name + ".json")).read_bytes()
        assert (value.to_json() + "\n").encode() == wire
        assert type(value).from_json(wire).fingerprint == value.fingerprint
    executed = result.domain.executed
    assert isinstance(executed.proposal.action, ChoiceAction)
    assert isinstance(executed.action, ChoiceAction)
    assert executed.proposal.action.chosen.id == expected["proposal"]
    assert executed.action.chosen.id == expected["execution"]
    assert [member.id for member in result.domain.feasibility.feasible_members] == expected["feasible_members"]
    request = result.domain.feasibility.request
    assert request.constraints[0].upper == expected["upper_bound"]
    assert all(alternative.values[0].spec.semantics.unit == expected["unit"] for alternative in request.alternatives)
    assert {alternative.member.id: alternative.values[0].value for alternative in request.alternatives} == expected[
        "predicted_durations"
    ]
    assert isinstance(result.coefficient.executed.action, MixtureAction)
    assert [weight.value for weight in result.coefficient.executed.action.weights] == expected["initial_coefficients"]
    state = direct.export_state()
    assert (state.decision_count, state.update_count, domain.state.calls) == (
        expected["decisions"],
        expected["updates"],
        expected["domain_calls"],
    )
