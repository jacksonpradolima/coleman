"""Principal reader tests for separate finite execution and later evidence."""

import json
import math
import subprocess
import sys
import sysconfig
from datetime import timedelta
from pathlib import Path

import pytest

from coleman.core.actions import AbstainAction, ChoiceAction, ChoiceSpace, RankingSpace, SubsetAction, SubsetSpace
from coleman.core.decisions import Decision, ExecutedDecision
from coleman.core.sampling import SamplingRecord
from coleman.evaluation.prequential.conformance import ReferenceConformancePlan, ReferenceConformanceReport
from coleman.evaluation.prequential.contracts import (
    ComponentObservation,
    ConformanceLimits,
    ConformanceTrace,
    DomainExecutionReceipt,
    StepEvidence,
)
from coleman.objectives.feasibility import check_finite_feasibility
from tests.evaluation.prequential.conformance_support import (
    NOW,
    changed,
    key,
    metric,
    observation,
    receipt,
    ref,
    trace,
)

FIXTURES = Path(__file__).parent / "fixtures" / "conformance-v1"
ROOTS = (
    ("receipt", DomainExecutionReceipt),
    ("observation", ComponentObservation),
    ("trace", ConformanceTrace),
    ("plan", ReferenceConformancePlan),
    ("report", ReferenceConformanceReport),
)


@pytest.mark.parametrize(("name", "contract"), ROOTS)
def test_versioned_canonical_fixture_roundtrip_preserves_original_wire(name, contract):
    raw = (FIXTURES / (name + ".json")).read_bytes()
    restored = contract.from_json(raw)
    assert restored.to_json().encode() + b"\n" == raw
    assert contract.model_validate_json(raw).to_json() == restored.to_json()
    assert contract.from_json(restored.to_json()).fingerprint == restored.fingerprint


@pytest.mark.parametrize(("name", "contract"), ROOTS)
@pytest.mark.parametrize("version", (0, 2, "1", True))
def test_new_readers_reject_future_unknown_or_coerced_versions(name, contract, version):
    fields = json.loads((FIXTURES / (name + ".json")).read_text())
    fields["schema_version"] = version
    raw = json.dumps(fields)
    with pytest.raises(ValueError, match="schema_version"):
        contract.from_json(raw)


def test_receipt_keeps_nonidentity_domain_mapping_separate_from_prior_coefficient():
    result = receipt()
    assert isinstance(result.executed.proposal.action, ChoiceAction)
    assert isinstance(result.executed.action, ChoiceAction)
    assert result.executed.proposal.action.chosen == ref("full")
    assert result.executed.action.chosen == ref("fast")
    assert result.executed.mapping == "projected"
    assert result.executed.sampling_disposition == "recomputed"
    assert result.feasibility.feasible_members == (ref("fast"),)
    assert result.coefficient_execution_fingerprint != result.executed.fingerprint
    assert "components" not in result.model_dump()
    assert result.prior_domain_state_fingerprint != result.fingerprint


@pytest.mark.parametrize("field", ("decision_id", "task"))
def test_receipt_rejects_a_different_producing_opportunity(field):
    original = receipt()
    updates = {field: "different" if field == "decision_id" else ref("other-task")}
    invalid = original.model_copy(update={"key": changed(original.key, **updates)})
    with pytest.raises(ValueError, match="logical opportunity"):
        DomainExecutionReceipt.model_validate(invalid)


def test_execution_cannot_precede_its_proposal():
    invalid = receipt().model_copy(update={"executed_at": NOW - timedelta(microseconds=1)})
    with pytest.raises(ValueError, match="cannot predate"):
        DomainExecutionReceipt.model_validate(invalid)


def test_executed_choice_requires_actual_finite_feasibility():
    feasible = receipt(selected="full", proposed="full", upper=8.0)
    invalid = feasible.model_copy(update={"feasibility": receipt().feasibility})
    with pytest.raises(ValueError, match="lacks retained feasible"):
        DomainExecutionReceipt.model_validate(invalid)


def test_sampled_execution_is_not_silently_qualified_as_deterministic():
    original = receipt()
    sampling = changed(original.executed.sampling, mechanism="sampled")
    executed = changed(original.executed, sampling=sampling)
    invalid = original.model_copy(update={"executed": executed})
    with pytest.raises(ValueError, match="deterministic"):
        DomainExecutionReceipt.model_validate(invalid)


def test_untimed_domain_execution_cannot_receive_a_fabricated_clock():
    original = receipt()
    request = changed(original.executed.proposal.request, timing=None)
    proposal = changed(original.executed.proposal, request=request)
    executed = changed(original.executed, proposal=proposal)
    invalid = original.model_copy(update={"executed": executed})
    with pytest.raises(ValueError, match="requires timing"):
        DomainExecutionReceipt.model_validate(invalid)


def test_nonchoice_proposal_is_representable_in_core_but_not_in_this_executor_profile():
    original = receipt()
    space = SubsetSpace(
        feasible_set=ref("subsets"), members=(ref("fast"), ref("full")), minimum_selected=0, maximum_selected=2
    )
    action = SubsetAction(schema_version=1, space=space, selected=(ref("fast"),))
    request = changed(original.executed.proposal.request, space=space)
    sampling = changed(original.executed.sampling, action_fingerprint=action.fingerprint)
    proposal = Decision(
        schema_version=1,
        request=request,
        implementation=ref("subset-adapter"),
        action=action,
        sampling=sampling,
        state_owner="host",
    )
    executed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=sampling,
    )
    invalid = original.model_copy(update={"executed": executed})
    with pytest.raises(ValueError, match="proposal must be a finite choice"):
        DomainExecutionReceipt.model_validate(invalid)


@pytest.mark.parametrize("shape", ("ranking_abstention", "changed_choice_universe"))
def test_core_mapping_does_not_implicitly_qualify_unadmitted_domain_action_spaces(shape):
    original = receipt()
    if shape == "ranking_abstention":
        space = RankingSpace(feasible_set=ref("rankings"), members=(ref("fast"), ref("full")))
        action = AbstainAction(schema_version=1, space=space, reason="unsupported")
    else:
        space = ChoiceSpace(feasible_set=ref("different-universe"), members=(ref("fast"), ref("other")))
        action = ChoiceAction(schema_version=1, space=space, chosen=ref("fast"))
    sampling = SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=ref("finite-choice"),
        status="unavailable",
        action_fingerprint=action.fingerprint,
        probability=None,
        reason="changed-action",
    )
    executed = ExecutedDecision(
        schema_version=1,
        proposal=original.executed.proposal,
        action=action,
        mapping="overridden",
        sampling_disposition="invalidated",
        sampling=sampling,
    )
    invalid = original.model_copy(update={"executed": executed})
    with pytest.raises(ValueError, match="choice space|choice universe"):
        DomainExecutionReceipt.model_validate(invalid)


def test_retained_finite_result_from_another_universe_does_not_certify_an_execution():
    original = receipt()
    first = next(item for item in original.feasibility.request.alternatives if item.member == ref("fast"))
    other = changed(first, member=ref("other"))
    request = changed(original.feasibility.request, universe=(ref("fast"), ref("other")), alternatives=(first, other))
    feasibility = check_finite_feasibility(request)
    invalid = original.model_copy(update={"feasibility": feasibility})
    with pytest.raises(ValueError, match="feasibility universe"):
        DomainExecutionReceipt.model_validate(invalid)


def test_explicit_domain_abstention_is_retained_without_fake_zero_action():
    original = receipt()
    action = AbstainAction(schema_version=1, space=original.executed.action.space, reason="operator-hold")
    sampling = SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=ref("finite-choice"),
        status="unavailable",
        action_fingerprint=action.fingerprint,
        probability=None,
        reason="no-selection",
    )
    executed = ExecutedDecision(
        schema_version=1,
        proposal=original.executed.proposal,
        action=action,
        mapping="overridden",
        sampling_disposition="invalidated",
        sampling=sampling,
    )
    retained = changed(original, executed=executed)
    assert retained.executed.action.kind == "abstain"
    assert retained.executed.sampling.probability is None


@pytest.mark.parametrize(
    ("value", "status"),
    ((0.0, "observed"), (-2.0, "observed"), (None, "unknown"), (None, "unavailable"), (None, "not_applicable")),
)
def test_later_observations_preserve_signed_values_and_distinct_missingness(value, status):
    item = observation(value=value, status=status)
    restored = ComponentObservation.from_json(item.to_json())
    assert restored.value.value == value
    assert restored.value.status == status
    assert restored.original.record.measurements[0].status == status


def test_signed_zero_and_original_omitted_optionals_survive_nested_trace():
    item = observation(value=-0.0)
    restored = ConformanceTrace.from_json(trace(components=(item,)).to_json())
    component = restored.steps[0].components[0]
    assert component.value.value is not None
    assert math.copysign(1, component.value.value) == -1
    assert component.original.record.to_json() == item.original.record.to_json()
    assert "source_revision" not in json.loads(component.original.record.to_json())["provenance"]
    assert "learning_reward" not in json.loads(component.original.record.to_json())


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), True))
def test_nonfinite_or_coerced_component_tokens_fail_before_legacy_projection(value):
    invalid = observation().model_dump(mode="json")
    invalid["value"]["value"] = value
    with pytest.raises(ValueError):
        ComponentObservation.model_validate(invalid)


def test_component_units_and_original_value_cannot_be_relabelled():
    original = observation()
    incompatible = changed(original.metric, quantity=metric(unit="ms").quantity)
    bad_units = original.model_copy(update={"metric": incompatible})
    bad_value = original.model_copy(update={"value": changed(original.value, value=3.0)})
    with pytest.raises(ValueError, match="exact declared schema"):
        ComponentObservation.model_validate(bad_units)
    with pytest.raises(ValueError, match="differs from the original"):
        ComponentObservation.model_validate(bad_value)


@pytest.mark.parametrize("field", ("available_at", "applied_at", "received_at"))
def test_later_observation_does_not_invent_early_availability_or_application(field):
    original = observation()
    if field == "available_at":
        invalid = original.model_copy(
            update={"timing": changed(original.timing, available_at=NOW + timedelta(seconds=10))}
        )
    elif field == "applied_at":
        invalid = original.model_copy(update={"applied_at": NOW})
    else:
        invalid = original.model_copy(update={"timing": changed(original.timing, received_at=None)})
    with pytest.raises(ValueError, match="availability|application"):
        ComponentObservation.model_validate(invalid)


def test_realized_physical_group_is_explicit_but_modeled_charge_is_not_measured_work():
    physical = observation(quantity=metric(role="physical_cost"), origin="realized")
    assert physical.physical_group_id == "physical-group-1"
    modeled = observation(origin="modeled_counterfactual")
    assert modeled.physical_group_id is None
    invalid = physical.model_copy(update={"physical_group_id": None})
    mislabeled = physical.model_copy(update={"value": changed(physical.value, origin="modeled_counterfactual")})
    with pytest.raises(ValueError, match="realized physical costs"):
        ComponentObservation.model_validate(invalid)
    with pytest.raises(ValueError, match="Modeled or synthetic"):
        ComponentObservation.model_validate(mislabeled)


@pytest.mark.parametrize("field", ("execution_fingerprint", "data_fingerprint", "configuration", "event"))
def test_step_verifies_the_actual_original_producer_and_causal_order(field):
    execution = receipt()
    item = observation(execution)
    if field == "execution_fingerprint":
        execution = changed(execution, prior_domain_state_fingerprint="sha256:" + "2" * 64)
    elif field == "data_fingerprint":
        execution = changed(execution, data_fingerprint="sha256:" + "2" * 64)
        item = changed(item, execution_fingerprint=execution.fingerprint)
    elif field == "configuration":
        execution = changed(execution, base_configuration_fingerprint="sha256:" + "2" * 64)
        item = changed(item, execution_fingerprint=execution.fingerprint)
    else:
        execution = changed(execution, executed_at=NOW + timedelta(seconds=11))
        item = changed(item, execution_fingerprint=execution.fingerprint)
    fields = {"schema_version": 1, "receipt": execution, "components": (item,)}
    with pytest.raises(ValueError, match="producing|source|BASE|predate"):
        StepEvidence.model_validate(fields)


def test_trace_keeps_prefix_and_terminal_steps_but_refuses_duplicate_delivery():
    first, second = receipt(), receipt(key("build-2"))
    steps = (trace(first).steps[0], trace(second).steps[0])
    result = ConformanceTrace(schema_version=1, recipe=trace().recipe, steps=steps)
    assert len(result.steps) == 2
    duplicate = {"schema_version": 1, "recipe": trace().recipe, "steps": (steps[0], steps[0])}
    with pytest.raises(ValueError, match="duplicate logical"):
        ConformanceTrace.model_validate(duplicate)


@pytest.mark.parametrize("limit", ({"max_steps": 0}, {"max_components": True}, {"max_bytes": 2**24 + 1}))
def test_selected_capacities_reject_malformed_or_above_hard_profile(limit):
    with pytest.raises(ValueError):
        ConformanceLimits.model_validate(limit)


def test_new_reader_guards_unknown_fields_duplicate_keys_depth_and_relaxations():
    raw = receipt().to_json()
    fields = json.loads(raw)
    fields["unexpected"] = 1
    unknown = json.dumps(fields)
    duplicate = raw[:-1] + ',"schema_version":1}'
    deep = '{"deep":' + "[" * 65 + "0" + "]" * 65 + "}"
    with pytest.raises(ValueError, match="unexpected"):
        DomainExecutionReceipt.from_json(unknown)
    with pytest.raises(ValueError, match="Duplicate"):
        DomainExecutionReceipt.from_json(duplicate)
    with pytest.raises(ValueError, match="depth"):
        DomainExecutionReceipt.model_validate_json(deep)
    with pytest.raises(ValueError, match="cannot relax"):
        DomainExecutionReceipt.model_validate_json(raw, strict=False)


def test_public_mutable_reader_refuses_above_hard_bytes_before_copy():
    raw = bytearray(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match="before copying"):
        DomainExecutionReceipt.model_validate_json(raw)


def test_pure_contract_imports_do_not_load_replay_store_transports_or_heavy_providers():
    script = """
import importlib.abc
import sys
sys.path[:0] = sys.argv[1:]
blocked = ('coleman.replay', 'coleman.state', 'coleman.cli', 'coleman.mcp',
           'coleman.native', 'numpy', 'pandas', 'pyarrow', 'scipy', 'duckdb')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Pure contract imported ' + fullname)
sys.meta_path.insert(0, Guard())
from coleman.evaluation.prequential.contracts import DomainExecutionReceipt
from coleman.evaluation.prequential.conformance import ReferenceConformancePlan
assert DomainExecutionReceipt.__module__.endswith('.contracts')
assert ReferenceConformancePlan.__module__.endswith('.conformance')
print('pure-boundary-ok')
"""
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(repository), sysconfig.get_path("purelib")],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "pure-boundary-ok"
