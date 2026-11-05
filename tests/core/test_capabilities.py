"""Check declared compatibility against concrete action, quantity and time evidence."""

from datetime import UTC, datetime
from itertools import combinations, permutations
from typing import Any, cast

import pytest

from coleman.core.actions import (
    AbstainAction,
    AcquisitionAction,
    AcquisitionSpace,
    ChoiceAction,
    ChoiceSpace,
    MixtureAction,
    MixtureSpace,
    RankingAction,
    RankingSpace,
    SlateAction,
    SlateSpace,
    SubsetAction,
    SubsetSpace,
    VectorAction,
    VectorSpace,
    Weight,
)
from coleman.core.capabilities import (
    Capability,
    CompatibilityResult,
    Incompatibility,
    ProblemCapabilities,
    validate_composition,
    validate_decision,
    validate_feedback,
)
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision, MemberInputs
from coleman.core.feedback import (
    AbsentFeedback,
    AggregateFeedback,
    ChosenFeedback,
    ComponentFeedback,
    FeedbackBatch,
    FeedbackRow,
    FullPanel,
    OutcomeValue,
)
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics


def _ref(name, version="v1"):
    return Reference(id=name, version=version)


def _spec(name="bounded-loss", **changes):
    fields = {
        "dimension": _ref(name),
        "semantics": UnitSemantics(unit="1", meaning="public-bounded-loss-v1"),
        "role": "loss",
        "minimum": 0.0,
        "maximum": 1.0,
    }
    fields.update(changes)
    return ScalarSpec.model_validate(fields)


def _value(spec=None, **changes):
    fields = {
        "spec": spec or _spec(),
        "value": 0.25,
        "status": "observed",
        "origin": "realized",
        "transform": _ref("bounded-loss-transform"),
    }
    fields.update(changes)
    return OutcomeValue.model_validate(fields)


def _row(member="A", *values):
    return FeedbackRow(member=_ref(member), values=values or (_value(),))


def _space(kind="choice"):
    common = {"feasible_set": _ref("public-feasible-universe"), "members": tuple(_ref(n) for n in "ABC")}
    match kind:
        case "choice":
            return ChoiceSpace.model_validate(common)
        case "ranking":
            return RankingSpace.model_validate(common)
        case "subset":
            return SubsetSpace.model_validate(common | {"minimum_selected": 0, "maximum_selected": 2})
        case "slate":
            return SlateSpace.model_validate(common | {"minimum_selected": 1, "maximum_selected": 2})
        case "mixture":
            return MixtureSpace.model_validate(common | {"profile": "binary64-simplex-v1"})
        case "acquisition":
            return AcquisitionSpace.model_validate(common | {"fidelities": (_ref("full"), _ref("partial"))})
        case "vector":
            coordinate = _spec("exposure", role="action", minimum=-1.0, maximum=1.0)
            return VectorSpace(feasible_set=common["feasible_set"], coordinates=(coordinate,), signed=True)
        case _:
            raise AssertionError(kind)


def _action(space=None, chosen="A"):
    space = space or _space()
    match space:
        case ChoiceSpace():
            return ChoiceAction(schema_version=1, space=space, chosen=_ref(chosen))
        case RankingSpace():
            return RankingAction(schema_version=1, space=space, order=tuple(reversed(space.members)))
        case SubsetSpace():
            return SubsetAction(schema_version=1, space=space, selected=(_ref("A"), _ref("B")))
        case SlateSpace():
            return SlateAction(schema_version=1, space=space, selected=(_ref("B"), _ref("A")))
        case MixtureSpace():
            weights = tuple(Weight(member=member, value=0.0 if member.id == "C" else 0.5) for member in space.members)
            return MixtureAction(schema_version=1, space=space, weights=weights)
        case AcquisitionSpace():
            return AcquisitionAction(schema_version=1, space=space, chosen=_ref(chosen), fidelity=_ref("full"))
        case VectorSpace():
            values = (Scalar(spec=space.coordinates[0], value=-0.25, status="observed"),)
            return VectorAction(schema_version=1, space=space, values=values)
        case _:
            raise AssertionError(type(space))


def _sampling(action, status="not_required"):
    fields = {
        "mechanism": "deterministic" if status == "exact" else "sampled",
        "mechanism_ref": _ref("public-selection"),
        "action_fingerprint": action.fingerprint,
        "status": status,
        "probability": 1.0 if status == "exact" else None,
    }
    if status == "qualified_approximation":
        fields.update(probability=0.5, error_bound=0.01, qualification=_ref("public-approximation"))
    if status == "unavailable":
        fields["reason"] = "host-mapping-without-qualified-mass"
    return SamplingRecord.model_validate(fields)


def _decision(action=None, sampling="not_required", **request_changes):
    action = action or _action()
    request = DecisionRequest(
        schema_version=1,
        task=_ref("public-tcpc-test-decision"),
        decision_id="build-1",
        round=0,
        space=action.space,
        objective=_ref("bounded-loss-objective"),
        **request_changes,
    )
    return Decision(
        schema_version=1,
        request=request,
        implementation=_ref("public-method"),
        action=action,
        sampling=_sampling(action, sampling),
        state_owner="host",
    )


def _executed(proposal=None, actual=None, sampling="not_required"):
    proposal = proposal or _decision()
    action = actual or proposal.action
    unchanged = action.fingerprint == proposal.action.fingerprint
    return ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=action,
        mapping="unchanged" if unchanged else "overridden",
        sampling_disposition="preserved" if unchanged else "recomputed",
        sampling=proposal.sampling if unchanged else _sampling(action, sampling),
    )


def _cap(**changes):
    fields = {
        "schema_version": 1,
        "implementation": _ref("public-method"),
        "action_kinds": ("choice",),
        "feedback_kinds": ("full_panel", "chosen", "component", "aggregate", "absent"),
        "required_values": (_spec(),),
        "origins": ("realized",),
        "timing_modes": ("untimed",),
        "supports_delayed_feedback": False,
        "requires_known_receipt": False,
        "requires_observed": True,
        "state_owner": "host",
        "requires_sampling": False,
        "sampling_requirement": "not_required",
    }
    fields.update(changes)
    return Capability.model_validate(fields)


def _problem(**changes):
    fields = {
        "schema_version": 1,
        "space": _space(),
        "feedback_kind": "full_panel",
        "values": (_spec(),),
        "origins": ("realized",),
        "timing_mode": "untimed",
        "delayed_feedback": False,
        "sampling_statuses": ("not_required",),
        "receipt_status": "unknown",
        "state_owners": ("host",),
    }
    fields.update(changes)
    return ProblemCapabilities.model_validate(fields)


def _batch(executed, feedback, **changes):
    fields = {
        "schema_version": 1,
        "decision_fingerprint": executed.fingerprint,
        "feedback_id": "build-outcomes",
        "revision_id": "initial",
        "supersedes": None,
        "horizon": None,
        "feedback": feedback,
    }
    fields.update(changes)
    return FeedbackBatch.model_validate(fields)


def _decision_timing():
    return DecisionTiming.model_validate(
        {
            "decision_at": "2026-01-01T10:00:00Z",
            "information_cutoff": "2026-01-01T09:59:00Z",
            "target_start": "2026-01-01T10:01:00Z",
            "target_end": "2026-01-01T10:05:00Z",
            "eligible_input_fingerprint": "sha256:" + "a" * 64,
            "mode": "future",
        }
    )


def _feedback_timing(received_at: str | None = "2026-01-01T10:06:00Z"):
    return FeedbackTiming.model_validate(
        {
            "event_at": "2026-01-01T10:05:00Z",
            "available_at": "2026-01-01T10:06:00Z",
            "mature_at": "2026-01-01T10:10:00Z",
            "received_at": received_at,
        }
    )


def test_profiles_and_compatibility_result_round_trip_in_canonical_order():
    fingerprints = set()
    for kinds in permutations(("choice", "ranking")):
        capability = _cap(action_kinds=kinds, origins=("synthetic", "realized"))
        restored = Capability.from_json(capability.to_json())
        assert restored == capability
        assert restored.action_kinds == ("choice", "ranking")
        fingerprints.add(capability.fingerprint)
    assert len(fingerprints) == 1
    problem = _problem()
    assert ProblemCapabilities.from_json(problem.to_json()) == problem
    assert validate_composition(problem, _cap()) == CompatibilityResult(compatible=True, reasons=())


@pytest.mark.parametrize("model", [Capability, ProblemCapabilities])
@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
def test_profile_versions_never_silently_migrate(model, version):
    fields = (_cap() if model is Capability else _problem()).model_dump()
    if version is None:
        del fields["schema_version"]
    else:
        fields["schema_version"] = version
    with pytest.raises(ValueError):
        model.model_validate(fields)


@pytest.mark.parametrize(
    "changes",
    [
        {"origins": ()},
        {"origins": ("realized", "realized")},
        {"action_kinds": ("choice", "choice")},
        {"feedback_kinds": ()},
        {"timing_modes": ("untimed", "untimed")},
        {"required_values": (_spec(), _spec(dimension=_ref("bounded-loss", "v2")))},
        {"required_inputs": (_spec(),)},
        {"supports_delayed_feedback": 1},
        {"requires_sampling": True},
        {"sampling_requirement": "exact"},
    ],
)
def test_capability_declarations_reject_duplicates_coercions_and_contradictions(changes):
    with pytest.raises(ValueError):
        _cap(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"state_owners": ("host", "host")},
        {"inputs": (_spec(),)},
        {"values": (_spec(), _spec())},
        {"delayed_feedback": "false"},
        {"sampling_statuses": ()},
        {"sampling_statuses": ("exact", "exact")},
        {"sampling_statuses": ("not-recorded",)},
        {"receipt_status": True},
    ],
)
def test_problem_profile_has_the_same_explicit_schema_and_boolean_guards(changes):
    with pytest.raises(ValueError):
        _problem(**changes)


@pytest.mark.parametrize("field", ["sampling_statuses", "receipt_status"])
def test_composition_evidence_guarantees_are_required_without_inferred_defaults(field):
    fields = _problem().model_dump()
    del fields[field]
    with pytest.raises(ValueError):
        ProblemCapabilities.model_validate(fields)


@pytest.mark.parametrize("requirement", ["not_required", "exact", "qualified_approximation"])
@pytest.mark.parametrize(
    "offered",
    [
        offered
        for count in range(1, 5)
        for offered in combinations(("exact", "qualified_approximation", "unavailable", "not_required"), count)
    ],
)
def test_composition_sampling_truth_matrix_checks_every_possible_actual_evidence_status(requirement, offered):
    allowed = {
        "not_required": {"exact", "qualified_approximation", "unavailable", "not_required"},
        "exact": {"exact"},
        "qualified_approximation": {"exact", "qualified_approximation"},
    }[requirement]
    profile = _cap(requires_sampling=requirement != "not_required", sampling_requirement=requirement)
    result = validate_composition(_problem(sampling_statuses=offered), profile)
    expected = set(offered) <= allowed
    assert result.compatible is expected
    assert ("sampling" in {reason.code for reason in result.reasons}) is not expected


@pytest.mark.parametrize("known_required", [False, True])
@pytest.mark.parametrize("receipt_status", ["known", "unknown"])
def test_composition_receipt_truth_matrix_never_infers_a_known_receipt(known_required, receipt_status):
    result = validate_composition(_problem(receipt_status=receipt_status), _cap(requires_known_receipt=known_required))
    expected = not known_required or receipt_status == "known"
    assert result.compatible is expected
    assert ("receipt" in {reason.code for reason in result.reasons}) is not expected


def test_offered_evidence_status_order_is_canonical_but_duplicates_are_not_repaired():
    first = _problem(sampling_statuses=("unavailable", "exact"))
    second = _problem(sampling_statuses=("exact", "unavailable"))
    assert first.to_json() == second.to_json()
    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    ("problem_changes", "cap_changes", "code"),
    [
        ({"space": _space("ranking")}, {}, "action_kind"),
        ({"feedback_kind": "chosen"}, {"feedback_kinds": ("full_panel",)}, "feedback_kind"),
        ({"origins": ("modeled_counterfactual",)}, {}, "origin"),
        ({"timing_mode": "ongoing"}, {}, "timing"),
        ({"delayed_feedback": True}, {}, "delay"),
        ({"state_owners": ("stateless",)}, {}, "state_owner"),
    ],
)
def test_each_unsupported_contract_axis_returns_a_stable_reason(problem_changes, cap_changes, code):
    result = validate_composition(_problem(**problem_changes), _cap(**cap_changes))
    assert not result.compatible
    assert code in {reason.code for reason in result.reasons}


@pytest.mark.parametrize(
    "changes",
    [
        {"dimension": _ref("bounded-loss", "v2")},
        {"semantics": UnitSemantics(unit="ms", meaning="public-bounded-loss-v1")},
        {"semantics": UnitSemantics(unit="1", meaning="different-objective-v1")},
        {"role": "reward"},
        {"minimum": -1.0},
        {"maximum": 2.0},
        {"minimum": None},
        {"maximum": None},
    ],
)
def test_schema_compatibility_requires_identity_units_meaning_role_and_interval_containment(changes):
    result = validate_composition(_problem(values=(_spec(**changes),)), _cap())
    assert not result.compatible
    assert result.reasons[0].code == "value_schema"


def test_tighter_intervals_and_extra_diagnostics_are_compatible_without_converting_values():
    narrower = _spec(minimum=0.0, maximum=0.5)
    diagnostic = _spec("latency", role="diagnostic", minimum=None, maximum=None)
    assert validate_composition(_problem(values=(diagnostic, narrower)), _cap()).compatible
    assert validate_composition(_problem(), _cap(required_values=(_spec(minimum=None, maximum=None),))).compatible


def test_signed_zero_bounds_describe_the_same_admissible_interval_without_rewriting_record_content():
    positive = _spec(minimum=0.0)
    negative = _spec(minimum=-0.0)
    assert validate_composition(_problem(values=(positive,)), _cap(required_values=(negative,))).compatible
    assert validate_composition(_problem(values=(negative,)), _cap(required_values=(positive,))).compatible
    assert positive.model_dump_json() != negative.model_dump_json()


@pytest.mark.parametrize(
    ("kind", "coverage"),
    [("ranking", "chosen"), ("choice", "component"), ("mixture", "component"), ("vector", "full_panel")],
)
def test_coverage_is_independent_of_outcome_dimensionality_and_requires_a_relevant_action(kind, coverage):
    result = validate_composition(_problem(space=_space(kind), feedback_kind=coverage), _cap(action_kinds=(kind,)))
    assert "coverage_shape" in {reason.code for reason in result.reasons}


@pytest.mark.parametrize("kind", ["mixture", "vector", "acquisition"])
def test_selection_mass_is_not_inferred_from_continuous_weights_or_acquisition_coordinates(kind):
    profile = _cap(action_kinds=(kind,), requires_sampling=True, sampling_requirement="exact")
    result = validate_composition(_problem(space=_space(kind), feedback_kind="aggregate"), profile)
    assert "sampling" in {reason.code for reason in result.reasons}


@pytest.mark.parametrize("kind", ["ranking", "choice", "subset", "slate", "mixture", "vector", "acquisition"])
@pytest.mark.parametrize(
    "offered",
    [
        offered
        for count in range(1, 5)
        for offered in combinations(("exact", "qualified_approximation", "unavailable", "not_required"), count)
    ],
)
def test_offered_mass_must_fit_the_action_family_even_without_a_consumer_requirement(kind, offered):
    profile = _cap(action_kinds=(kind,))
    problem = _problem(space=_space(kind), feedback_kind="aggregate", sampling_statuses=offered)
    reordered = _problem(space=_space(kind), feedback_kind="aggregate", sampling_statuses=tuple(reversed(offered)))
    result = validate_composition(problem, profile)
    has_mass = bool(set(offered) & {"exact", "qualified_approximation"})
    expected = kind not in {"mixture", "vector", "acquisition"} or not has_mass
    assert result.compatible is expected
    assert ("sampling" in {reason.code for reason in result.reasons}) is not expected
    assert validate_composition(reordered, profile) == result
    assert reordered.fingerprint == problem.fingerprint


def test_incompatibilities_are_sorted_and_contradictory_or_duplicate_results_reject():
    reasons = (Incompatibility(code="z", detail="last"), Incompatibility(code="a", detail="first"))
    result = CompatibilityResult(compatible=False, reasons=reasons)
    assert result.reasons == tuple(reversed(reasons))
    for compatible, reported in [(True, reasons), (False, ()), (False, reasons + reasons)]:
        with pytest.raises(ValueError):
            CompatibilityResult(compatible=compatible, reasons=reported)


def test_global_required_input_cannot_be_satisfied_by_member_context_or_a_different_unit():
    feature = _spec("duration", role="feature", semantics=UnitSemantics(unit="ms", meaning="previous-duration-v1"))
    observed = Scalar(spec=feature, value=0.5, status="observed")
    profile = _cap(required_inputs=(feature,))
    member_only = _decision(member_inputs=(MemberInputs(member=_ref("A"), values=(observed,)),))
    with pytest.raises(ValueError, match="Missing or incompatible dimension duration"):
        validate_decision(member_only, profile)
    validate_decision(_decision(inputs=(observed,)), profile)
    assert not validate_composition(_problem(), profile).compatible
    assert validate_composition(_problem(inputs=(feature,)), profile).compatible
    seconds = _spec("duration", role="feature", semantics=UnitSemantics(unit="s", meaning="previous-duration-v1"))
    with pytest.raises(ValueError, match="incompatible dimension"):
        validate_decision(_decision(inputs=(Scalar(spec=seconds, value=0.5, status="observed"),)), profile)
    missing = Scalar(spec=feature, value=None, status="unknown")
    with pytest.raises(ValueError, match="observed values"):
        validate_decision(_decision(inputs=(missing,)), profile)


@pytest.mark.parametrize(
    "changes",
    [{"implementation": _ref("public-method", "v2")}, {"state_owner": "learner"}],
)
def test_actual_decision_checks_implementation_version_and_state_owner(changes):
    fields = _decision().model_dump()
    fields.update(changes)
    with pytest.raises(ValueError, match="implementation/version or state owner"):
        validate_decision(Decision.model_validate(fields), _cap())


def test_actual_decision_checks_action_kind_and_time_mode():
    with pytest.raises(ValueError, match="action shape"):
        validate_decision(_decision(_action(_space("ranking"))), _cap())
    with pytest.raises(ValueError, match="temporal mode"):
        validate_decision(_decision(timing=_decision_timing()), _cap())


def test_feedback_rejects_an_actual_mapped_shape_the_consumer_does_not_support():
    executed = _executed(_decision(), _action(_space("ranking")))
    with pytest.raises(ValueError, match="Executed action shape is unsupported"):
        validate_feedback(executed, AggregateFeedback(values=(_value(),)), _cap())


def test_full_panel_requires_the_exact_actual_universe_and_version_without_imputation():
    executed = _executed()
    validate_feedback(executed, FullPanel(rows=tuple(_row(member) for member in "ABC")), _cap())
    for rows in [(_row("A"), _row("B")), (_row("A"), _row("B"), _row("D"))]:
        with pytest.raises(ValueError, match="exact active members"):
            validate_feedback(executed, FullPanel(rows=rows), _cap())
    wrong_version = FeedbackRow(member=_ref("C", "v2"), values=(_value(),))
    with pytest.raises(ValueError, match="exact active members"):
        validate_feedback(executed, FullPanel(rows=(_row("A"), _row("B"), wrong_version)), _cap())


@pytest.mark.parametrize("coverage", ["chosen", "full_panel"])
@pytest.mark.parametrize("objective_count", [1, 2])
def test_required_objective_truth_matrix_keeps_coverage_distinct_from_dimensionality(coverage, objective_count):
    objectives = (
        _spec(
            "failure-detection-rate",
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="observed-ci-failure-detection-fraction-v1"),
        ),
        _spec(
            "early-detection-rate",
            role="objective",
            semantics=UnitSemantics(unit="1", meaning="observed-ci-early-failure-detection-fraction-v1"),
        ),
    )[:objective_count]
    values = tuple(_value(spec, value=0.75, transform=None) for spec in objectives)
    executed = _executed()
    profile = _cap(required_values=objectives, feedback_kinds=(coverage,))
    problem = _problem(values=objectives, feedback_kind=coverage)
    assert validate_composition(problem, profile).compatible
    if coverage == "chosen":
        feedback = ChosenFeedback(row=_row("A", *values))
    else:
        feedback = FullPanel(rows=tuple(_row(member, *values) for member in "ABC"))
    batch = _batch(executed, feedback)
    restored = FeedbackBatch.from_json(batch.to_json())
    assert restored.to_json() == batch.to_json()
    assert restored.fingerprint == batch.fingerprint
    assert restored.feedback.kind == coverage
    validate_feedback(executed, restored, profile)
    if isinstance(restored.feedback, ChosenFeedback):
        rows = (restored.feedback.row,)
    else:
        assert isinstance(restored.feedback, FullPanel)
        rows = restored.feedback.rows
    assert len(rows) == (1 if coverage == "chosen" else 3)
    assert all(len(row.values) == objective_count for row in rows)
    assert all(value.spec.role == "objective" for row in rows for value in row.values)

    # The same dimension, units and number of measurements cannot hide a role change.
    resource = ScalarSpec.model_validate(objectives[-1].model_dump() | {"role": "resource"})
    substituted = (*values[:-1], _value(resource, value=0.75, transform=None))
    if coverage == "chosen":
        incorrect = ChosenFeedback(row=_row("A", *substituted))
    else:
        incorrect = FullPanel(rows=tuple(_row(member, *substituted) for member in "ABC"))
    incorrect_batch = FeedbackBatch.from_json(_batch(executed, incorrect).to_json())
    with pytest.raises(ValueError, match="Missing or incompatible dimension"):
        validate_feedback(executed, incorrect_batch, profile)

    if coverage == "full_panel":
        incomplete = FullPanel(rows=tuple(_row(member, *values) for member in "AB"))
        incomplete_batch = FeedbackBatch.from_json(_batch(executed, incomplete).to_json())
        with pytest.raises(ValueError, match="exact active members"):
            validate_feedback(executed, incomplete_batch, profile)


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_structurally_complete_panels_with_null_losses_cannot_update_observed_loss_consumers(status):
    panel = FullPanel(rows=(_row("A"), _row("B"), _row("C", _value(value=None, status=status))))
    with pytest.raises(ValueError, match="observed values"):
        validate_feedback(_executed(), panel, _cap())
    validate_feedback(_executed(), panel, _cap(requires_observed=False))


def test_valid_zero_negative_resource_and_unavailable_unused_diagnostic_remain_distinct():
    spec = _spec("energy-delta", role="resource", minimum=-10.0, maximum=10.0)
    diagnostic = _value(_spec("unreported", role="diagnostic"), value=None, status="unavailable")
    feedback = ChosenFeedback(row=_row("A", _value(value=0.0), _value(spec, value=-2.0), diagnostic))
    validate_feedback(_executed(), feedback, _cap(required_values=(_spec(), spec)))


@pytest.mark.parametrize(
    "spec",
    [
        _spec(dimension=_ref("bounded-loss", "v2")),
        _spec(role="reward"),
        _spec(semantics=UnitSemantics(unit="ms", meaning="public-bounded-loss-v1")),
        _spec(minimum=None),
        _spec("different-loss"),
    ],
)
def test_actual_feedback_cannot_substitute_another_schema_despite_valid_numeric_values(spec):
    feedback = ChosenFeedback(row=_row("A", _value(spec)))
    with pytest.raises(ValueError, match="Missing or incompatible dimension"):
        validate_feedback(_executed(), feedback, _cap())


def test_chosen_feedback_and_batch_bind_actual_execution_instead_of_proposed_member():
    proposal = _decision()
    executed = _executed(proposal, _action(chosen="B"))
    actual = ChosenFeedback(row=_row("B"))
    validate_feedback(executed, actual, _cap())
    validate_feedback(executed, _batch(executed, actual), _cap())
    with pytest.raises(ValueError, match="actual selected choice"):
        validate_feedback(executed, ChosenFeedback(row=_row("A")), _cap())
    wrong_binding = _batch(executed, actual, decision_fingerprint=proposal.fingerprint)
    with pytest.raises(ValueError, match="executed decision content"):
        validate_feedback(executed, wrong_binding, _cap())


@pytest.mark.parametrize("kind", ["subset", "slate", "acquisition"])
def test_component_coverage_checks_actual_selection_and_explicit_completeness(kind):
    executed = _executed(_decision(_action(_space(kind))))
    profile = _cap(action_kinds=(kind,))
    partial = ComponentFeedback(rows=(_row("A"),), completeness="partial")
    validate_feedback(executed, partial, profile)
    complete_rows = (_row("A"),) if kind == "acquisition" else (_row("A"), _row("B"))
    validate_feedback(executed, ComponentFeedback(rows=complete_rows, completeness="complete"), profile)
    with pytest.raises(ValueError, match="actual selected members"):
        validate_feedback(executed, ComponentFeedback(rows=(_row("C"),), completeness="partial"), profile)
    if kind != "acquisition":
        with pytest.raises(ValueError, match="completeness"):
            validate_feedback(executed, ComponentFeedback(rows=(_row("A"),), completeness="complete"), profile)


@pytest.mark.parametrize("kind", ["ranking", "choice", "subset", "slate", "mixture", "vector", "acquisition"])
def test_aggregate_stays_a_whole_action_outcome_without_member_fanout(kind):
    executed = _executed(_decision(_action(_space(kind))))
    aggregate = AggregateFeedback(values=(_value(),))
    validate_feedback(executed, aggregate, _cap(action_kinds=(kind,)))
    assert not hasattr(aggregate, "rows")


def test_vector_coordinates_are_not_an_arm_loss_panel():
    executed = _executed(_decision(_action(_space("vector"))))
    panel = FullPanel(rows=(_row("exposure"),))
    with pytest.raises(ValueError, match="cannot describe this actual action shape"):
        validate_feedback(executed, panel, _cap(action_kinds=("vector",)))


@pytest.mark.parametrize("kind", ["choice", "ranking", "mixture", "vector"])
def test_direct_component_validation_cannot_bypass_action_coverage_negotiation(kind):
    executed = _executed(_decision(_action(_space(kind))))
    component = ComponentFeedback(rows=(_row("A"),), completeness="partial")
    with pytest.raises(ValueError, match="cannot describe this actual action shape"):
        validate_feedback(executed, component, _cap(action_kinds=(kind,)))


def test_absence_is_admitted_only_explicitly_and_never_fabricates_required_values():
    absent = AbsentFeedback(reason="not-yet-observed")
    validate_feedback(_executed(), absent, _cap())
    assert validate_composition(_problem(feedback_kind="absent", values=()), _cap()).compatible
    with pytest.raises(ValueError, match="coverage is unsupported"):
        validate_feedback(_executed(), absent, _cap(feedback_kinds=("chosen",)))
    action = AbstainAction(schema_version=1, space=_space(), reason="insufficient-evidence")
    executed = _executed(_decision(action))
    validate_feedback(executed, absent, _cap())


def test_full_information_remains_observable_after_abstention_without_inventing_a_choice():
    action = AbstainAction(schema_version=1, space=_space(), reason="insufficient-evidence")
    executed = _executed(_decision(action))
    panel = FullPanel(rows=tuple(_row(member) for member in "ABC"))
    validate_feedback(executed, panel, _cap())
    with pytest.raises(ValueError, match="actual selected choice"):
        validate_feedback(executed, ChosenFeedback(row=_row()), _cap())


def test_origin_is_per_value_and_modeled_values_are_not_promoted_to_realized():
    diagnostic = _spec("estimated-cost", role="diagnostic")
    chosen = ChosenFeedback(row=_row("A", _value(), _value(diagnostic, origin="modeled_counterfactual")))
    with pytest.raises(ValueError, match="unsupported outcome origin"):
        validate_feedback(_executed(), chosen, _cap())
    validate_feedback(_executed(), chosen, _cap(origins=("realized", "modeled_counterfactual")))
    assert chosen.row.values[1].origin == "modeled_counterfactual"


@pytest.mark.parametrize("coverage", ["full_panel", "component"])
def test_a_required_dimension_cannot_quietly_mix_transform_versions_across_rows(coverage):
    row_a = _row("A")
    row_b = _row("B", _value(transform=_ref("bounded-loss-transform", "v2")))
    kind = "choice" if coverage == "full_panel" else "subset"
    if coverage == "full_panel":
        feedback = FullPanel(rows=(row_a, row_b, _row("C")))
    else:
        feedback = ComponentFeedback(rows=(row_a, row_b), completeness="complete")
    with pytest.raises(ValueError, match="cannot mix transform identities or versions"):
        validate_feedback(_executed(_decision(_action(_space(kind)))), feedback, _cap(action_kinds=(kind,)))


@pytest.mark.parametrize(
    ("requirement", "status", "accepted"),
    [
        ("exact", "exact", True),
        ("exact", "qualified_approximation", False),
        ("exact", "unavailable", False),
        ("exact", "not_required", False),
        ("qualified_approximation", "qualified_approximation", True),
        ("qualified_approximation", "exact", True),
        ("qualified_approximation", "unavailable", False),
    ],
)
def test_sampling_evidence_is_explicit_and_approximation_is_never_exact(requirement, status, accepted):
    profile = _cap(requires_sampling=True, sampling_requirement=requirement)
    decision = _decision(sampling=status)
    if accepted:
        validate_decision(decision, profile)
        validate_feedback(_executed(decision), ChosenFeedback(row=_row()), profile)
    else:
        with pytest.raises(ValueError, match="sampling evidence"):
            validate_decision(decision, profile)


def test_host_mapping_that_invalidates_sampling_cannot_satisfy_an_exact_consumer():
    proposal = _decision(sampling="exact")
    actual = _action(chosen="B")
    executed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=actual,
        mapping="overridden",
        sampling_disposition="invalidated",
        sampling=_sampling(actual, "unavailable"),
    )
    with pytest.raises(ValueError, match="sampling evidence"):
        validate_feedback(
            executed, ChosenFeedback(row=_row("B")), _cap(requires_sampling=True, sampling_requirement="exact")
        )


def test_feedback_uses_recomputed_actual_mass_even_when_the_proposal_mass_was_unavailable():
    proposal = _decision(sampling="unavailable")
    executed = _executed(proposal, _action(chosen="B"), sampling="exact")
    profile = _cap(requires_sampling=True, sampling_requirement="exact")
    with pytest.raises(ValueError, match="sampling evidence"):
        validate_decision(proposal, profile)
    validate_feedback(executed, ChosenFeedback(row=_row("B")), profile)


def test_feedback_matches_explicitly_mapped_actual_shape_instead_of_proposal_shape():
    proposal = _decision(_action(_space("ranking")))
    executed = _executed(proposal, _action(chosen="B"))
    with pytest.raises(ValueError, match="action shape is unsupported"):
        validate_decision(proposal, _cap())
    validate_feedback(executed, ChosenFeedback(row=_row("B")), _cap())


def test_timed_decisions_need_a_batch_and_explicit_eligible_application_time():
    executed = _executed(_decision(timing=_decision_timing()))
    profile = _cap(timing_modes=("future",))
    feedback = ChosenFeedback(row=_row())
    with pytest.raises(ValueError, match="explicit batch timing"):
        validate_feedback(executed, feedback, profile)
    with pytest.raises(ValueError, match="explicit batch timing"):
        validate_feedback(executed, _batch(executed, feedback), profile)
    batch = _batch(executed, feedback, timing=_feedback_timing())
    with pytest.raises(ValueError, match="explicit apply_at"):
        validate_feedback(executed, batch, profile)
    with pytest.raises(ValueError, match="before known"):
        validate_feedback(executed, batch, profile, apply_at=datetime(2026, 1, 1, 10, 9, tzinfo=UTC))
    validate_feedback(executed, batch, profile, apply_at=datetime(2026, 1, 1, 10, 10, tzinfo=UTC))


def test_unknown_receipt_stays_unknown_and_is_accepted_only_without_a_receipt_requirement():
    executed = _executed(_decision(timing=_decision_timing()))
    feedback = ChosenFeedback(row=_row())
    batch = _batch(executed, feedback, timing=_feedback_timing(received_at=None))
    apply_at = datetime(2026, 1, 1, 10, 10, tzinfo=UTC)
    validate_feedback(executed, batch, _cap(timing_modes=("future",)), apply_at=apply_at)
    assert batch.timing is not None
    assert batch.timing.received_at is None
    assert batch.timing.usable_at is None
    with pytest.raises(ValueError, match="known feedback receipt"):
        validate_feedback(
            executed, batch, _cap(timing_modes=("future",), requires_known_receipt=True), apply_at=apply_at
        )
    with pytest.raises(ValueError, match="known feedback receipt"):
        validate_feedback(_executed(), feedback, _cap(requires_known_receipt=True))


@pytest.mark.parametrize("mode", ["ongoing", "future"])
@pytest.mark.parametrize("known_receipt", [False, True])
@pytest.mark.parametrize("apply_hour", [9, 10, 11])
def test_feedback_application_cannot_predate_its_originating_decision(mode, known_receipt, apply_hour):
    decision_timing = DecisionTiming.model_validate(
        {
            "decision_at": "2026-01-01T10:00:00Z",
            "information_cutoff": "2026-01-01T09:00:00Z",
            "target_start": "2026-01-01T08:00:00Z" if mode == "ongoing" else "2026-01-01T10:00:00Z",
            "target_end": "2026-01-01T12:00:00Z",
            "eligible_input_fingerprint": "sha256:" + "a" * 64,
            "mode": mode,
        }
    )
    earlier = datetime(2026, 1, 1, 8, tzinfo=UTC)
    feedback_timing = FeedbackTiming(
        event_at=earlier,
        available_at=earlier,
        mature_at=earlier,
        received_at=earlier if known_receipt else None,
    )
    executed = _executed(_decision(timing=decision_timing))
    batch = _batch(executed, ChosenFeedback(row=_row()), timing=feedback_timing)
    profile = _cap(timing_modes=(mode,))
    apply_at = datetime(2026, 1, 1, apply_hour, tzinfo=UTC)
    if apply_hour < 10:
        with pytest.raises(ValueError, match="before known decision"):
            validate_feedback(executed, batch, profile, apply_at=apply_at)
    else:
        validate_feedback(executed, batch, profile, apply_at=apply_at)
        if not known_receipt:
            assert feedback_timing.received_at is None
            assert feedback_timing.usable_at is None
            receipt_required = _cap(timing_modes=(mode,), requires_known_receipt=True)
            with pytest.raises(ValueError, match="known feedback receipt"):
                validate_feedback(executed, batch, receipt_required, apply_at=apply_at)


def test_any_supplied_timing_is_honored_even_when_decision_is_untimed():
    executed = _executed()
    batch = _batch(executed, ChosenFeedback(row=_row()), timing=_feedback_timing())
    with pytest.raises(ValueError, match="explicit apply_at"):
        validate_feedback(executed, batch, _cap())
    with pytest.raises(ValueError):
        validate_feedback(executed, batch, _cap(), apply_at=datetime(2026, 1, 1, 10, 10))
    validate_feedback(executed, batch, _cap(), apply_at=datetime(2026, 1, 1, 10, 10, tzinfo=UTC))


def test_later_known_receipt_remains_a_gate_even_after_maturity():
    executed = _executed()
    timing = _feedback_timing(received_at="2026-01-01T10:12:00Z")
    batch = _batch(executed, ChosenFeedback(row=_row()), timing=timing)
    with pytest.raises(ValueError, match="before known"):
        validate_feedback(executed, batch, _cap(), apply_at=datetime(2026, 1, 1, 10, 10, tzinfo=UTC))
    validate_feedback(executed, batch, _cap(), apply_at=datetime(2026, 1, 1, 10, 12, tzinfo=UTC))


def test_logical_delay_is_distinct_from_waiting_for_maturity_and_needs_explicit_support():
    executed = _executed()
    batch = _batch(executed, ChosenFeedback(row=_row()), delay_rounds=1)
    with pytest.raises(ValueError, match="logical feedback rounds"):
        validate_feedback(executed, batch, _cap())
    validate_feedback(executed, batch, _cap(supports_delayed_feedback=True))


def test_unchecked_models_and_untyped_payloads_do_not_bypass_validation():
    executed = _executed()
    feedback = ChosenFeedback(row=_row())
    with pytest.raises(ValueError):
        validate_feedback(executed, feedback.model_copy(update={"kind": "full_panel"}), _cap())
    with pytest.raises(TypeError, match="typed feedback"):
        validate_feedback(executed, cast(Any, feedback.model_dump()), _cap())
    with pytest.raises(ValueError):
        validate_composition(_problem().model_copy(update={"schema_version": 2}), _cap())
