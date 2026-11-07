"""Restore-ingress refusals on actual controlled owners and retained receipts."""

from datetime import timedelta

import pytest

from coleman.core.actions import Weight
from coleman.learning.contracts import LearningFailure
from coleman.learning.control_contracts import (
    AcceptedPanel,
    AggregateLoss,
    ClosedInterval,
    ControlledConfig,
    ControlledLimits,
    ControlledPayload,
    ControlOutcome,
    MovementPeriod,
)
from coleman.learning.controlled import ControlledHedge
from coleman.objectives.controls import MovementLimit, MovementUsage, exact_mixture_intervals, initial_exact_path_usage
from coleman.optimization.mixture import solve_mixture_output
from coleman.state.codec import read_algorithm_state
from tests.learning._controlled_cases import configuration, panel, ref
from tests.learning.test_controlled import START, clone, interval, period


def forged_state(learner, **changes):
    state = learner.export_state()
    return state.model_copy(update={"payload": state.payload.model_copy(update=changes)})


@pytest.fixture(scope="module")
def held_owner():
    config = configuration(cadence=3)
    learner = ControlledHedge(config, owner=ref("contract-held-owner"))
    learner.observe(panel(config, (0.0, 1.0, 0.5)))
    learner.observe(panel(config, (1.0, 0.0, 0.5)))
    return learner


@pytest.fixture(scope="module")
def released_owner():
    config = configuration()
    learner = ControlledHedge(config, owner=ref("contract-released-owner"))
    learner.observe(panel(config, (0.0, 1.0, 0.5)))
    return learner


@pytest.fixture(scope="module")
def interval_owner():
    calendar = ref("contract-release-calendar")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("contract-interval-owner"))
    learner.observe(panel(config, (0.0, 1.0, 0.5)), eligible_at=START, apply_at=START)
    first, second = interval(0), interval(1)
    learner.release(first, calendar=calendar, apply_at=first.release_at)
    learner.release(second, calendar=calendar, apply_at=second.release_at)
    return learner


@pytest.fixture(scope="module")
def period_owner():
    config = configuration(movement=(0.4, 0.4, 0.4), scope="per_interval", strength=0.0)
    learner = ControlledHedge(config, owner=ref("contract-period-owner"))
    learner.begin_period(period(config, 0), apply_at=START)
    learner.observe(panel(config, (0.0, 1.0, 1.0)), apply_at=START)
    return learner


@pytest.mark.parametrize("support", [(), (True,), (False, True, True), (1, 1, 1)])
def test_required_support_refuses_forgery_without_live_assignment(released_owner, support):
    before = released_owner.export_state().to_json()
    candidate = forged_state(released_owner, support=support)
    with pytest.raises(ValueError, match="support|Support"):
        released_owner.restore_state(candidate)
    assert released_owner.export_state().to_json() == before


def test_support_is_required_in_the_reader_without_inference(released_owner):
    raw = released_owner.export_state().payload.model_dump()
    del raw["support"]
    with pytest.raises(ValueError, match="support|Support"):
        ControlledPayload.model_validate(raw)


@pytest.mark.parametrize("change", ["member", "box"])
def test_current_origin_exact_members_and_static_boxes_before_assignment(released_owner, change):
    state = released_owner.export_state()
    weights = list(state.payload.origin.weights)
    weights[0] = weights[0].model_copy(update={"member": ref("other-expert")})
    if change == "box":
        config = configuration(upper=(0.6, 1.0, 1.0), strength=0.0)
        learner = ControlledHedge(config, owner=ref("box-owner"))
        weights = tuple(
            Weight(member=item.member, value=value)
            for item, value in zip(learner.weights, (0.7, 0.2, 0.1), strict=True)
        )
    else:
        learner = released_owner
    before = learner.export_state().to_json()
    origin = learner.export_state().payload.origin.model_copy(update={"weights": tuple(weights)})
    candidate = forged_state(learner, origin=origin, last_outcome=None)
    with pytest.raises(ValueError, match="exact configured member|static box"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["transform", "spec", "unavailable", "dimension"])
def test_pending_full_panel_is_revalidated_against_fixed_law(held_owner, change):
    state = held_owner.export_state()
    pending = state.payload.pending[0]
    rows = list(pending.panel.rows)
    value = rows[0].values[0]
    if change == "transform":
        value = value.model_copy(update={"transform": ref("other-transform")})
    elif change == "spec":
        value = value.model_copy(update={"spec": value.spec.model_copy(update={"unit": "other-unit"})})
    elif change == "unavailable":
        value = value.model_copy(update={"value": None, "status": "unavailable"})
    values = (value, value) if change == "dimension" else (value,)
    rows[0] = rows[0].model_copy(update={"values": values})
    item = pending.model_copy(update={"panel": pending.panel.model_copy(update={"rows": tuple(rows)})})
    candidate = forged_state(held_owner, pending=(item, state.payload.pending[1]))
    before = state.to_json()
    with pytest.raises(ValueError, match="transform|spec|loss|scalar|observed"):
        held_owner.restore_state(candidate)
    assert held_owner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["reference", "sequence", "frontier"])
def test_pending_identity_sequence_and_count_frontiers_are_exact(held_owner, change):
    state = held_owner.export_state()
    first, second = state.payload.pending
    fields = {"reference": first.reference} if change == "reference" else {"sequence": first.sequence}
    if change == "frontier":
        fields = {"sequence": 10}
    candidate = forged_state(held_owner, pending=(first, second.model_copy(update=fields)))
    before = state.to_json()
    with pytest.raises(ValueError, match="unique original|contiguous"):
        held_owner.restore_state(candidate)
    assert held_owner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["law", "counter", "weights", "origin", "strength"])
def test_latest_outcome_cannot_change_law_counters_or_current_origin(released_owner, change):
    state = released_owner.export_state()
    outcome = state.payload.last_outcome
    if change == "law":
        altered = outcome.model_copy(update={"law": outcome.law.model_copy(update={"reference": ref("other-law")})})
    elif change == "counter":
        altered = outcome.model_copy(update={"update_count": outcome.update_count + 1})
    elif change == "strength":
        altered = outcome.model_copy(update={"output": outcome.output.model_copy(update={"strength": 0.2})})
    elif change == "origin":
        emitted = outcome.emitted_origin.model_copy(update={"reference": ref("other-emitted-origin")})
        altered = outcome.model_copy(update={"emitted_origin": emitted})
    else:
        weights = tuple(
            Weight(member=item.member, value=value)
            for item, value in zip(outcome.weights, (0.5, 0.25, 0.25), strict=True)
        )
        altered = outcome.model_copy(update={"weights": weights})
    candidate = forged_state(released_owner, last_outcome=altered)
    before = state.to_json()
    with pytest.raises(ValueError, match="law|counter|origin|coefficients"):
        released_owner.restore_state(candidate)
    assert released_owner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["identity", "overlap", "law", "rewind"])
def test_closed_intervals_keep_original_identities_clock_order_and_law(interval_owner, change):
    state = interval_owner.export_state()
    first, second = state.payload.intervals
    if change == "identity":
        second = second.model_copy(
            update={"interval": second.interval.model_copy(update={"interval": first.interval.interval})}
        )
    elif change == "overlap":
        second = second.model_copy(
            update={"interval": second.interval.model_copy(update={"start": first.interval.start})}
        )
    elif change == "law":
        changed = second.outcome.law.model_copy(update={"reference": ref("other-historical-law")})
        second = second.model_copy(update={"outcome": second.outcome.model_copy(update={"law": changed})})
    else:
        counters = {"update_count": 0, "release_count": 0, "consumed_panels": 0}
        second = second.model_copy(update={"outcome": second.outcome.model_copy(update=counters)})
    candidate = forged_state(interval_owner, intervals=(first, second))
    before = state.to_json()
    with pytest.raises(ValueError, match="repeat|overlap|law|rewind|counter"):
        interval_owner.restore_state(candidate)
    assert interval_owner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["movement", "period_origin", "current_origin", "interval", "underreported"])
def test_exact_path_requires_fixed_law_and_both_original_origin_joins(period_owner, change):
    state = period_owner.export_state()
    path = state.payload.exact_path
    if change == "movement":
        altered = path.model_copy(
            update={"movement": path.movement.model_copy(update={"reference": ref("other-movement")})}
        )
    elif change == "period_origin":
        altered = path.model_copy(
            update={"period_origin": path.period_origin.model_copy(update={"reference": ref("other-start")})}
        )
    elif change == "current_origin":
        altered = path.model_copy(update={"through_origin": path.period_origin.binding})
    elif change == "interval":
        altered = path.model_copy(update={"interval": ref("other-period")})
    else:
        used = tuple(item.model_copy(update={"atoms_decimal": "0"}) for item in path.used)
        altered = path.model_copy(update={"used": used})
    candidate = forged_state(period_owner, exact_path=altered)
    before = state.to_json()
    with pytest.raises(ValueError, match="law|origin|period|usage|displacement"):
        period_owner.restore_state(candidate)
    assert period_owner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["duplicate", "overlap", "clock", "missing_origin"])
def test_movement_period_history_is_exact_and_monotone(period_owner, change):
    state = period_owner.export_state()
    first = state.payload.periods[0]
    second = period(period_owner.config, 1)
    changes: dict[str, object] = {"periods": (first, second)}
    if change == "duplicate":
        changes["periods"] = (first, first)
    elif change == "overlap":
        changes["periods"] = (first, second.model_copy(update={"start": first.start}))
    elif change == "clock":
        changes["periods"] = (first.model_copy(update={"calendar": ref("other-clock")}),)
    else:
        changes = {"period_origin": None}
    candidate = forged_state(period_owner, **changes)
    before = state.to_json()
    with pytest.raises(ValueError, match="unique identities|overlap|present together"):
        period_owner.restore_state(candidate)
    assert period_owner.export_state().to_json() == before


def test_nonperiod_law_cannot_import_movement_period_state(released_owner, period_owner):
    retained = period_owner.export_state().payload
    candidate = forged_state(released_owner, periods=retained.periods, period_origin=retained.period_origin)
    before = released_owner.export_state().to_json()
    with pytest.raises(ValueError, match="without per-interval"):
        released_owner.restore_state(candidate)
    assert released_owner.export_state().to_json() == before


def test_new_period_can_retain_original_previous_outcome_and_partial_block():
    config = configuration(cadence=2, movement=(0.4, 0.4, 0.4), scope="per_interval")
    learner = ControlledHedge(config, owner=ref("period-carry-owner"))
    learner.begin_period(period(config, 0), apply_at=START)
    learner.observe(panel(config, (0.0, 1.0, 0.5)), apply_at=START)
    learner.observe(panel(config, (0.0, 1.0, 0.5)), apply_at=START)
    old_outcome = learner.export_state().payload.last_outcome
    assert old_outcome is not None
    original = old_outcome.to_json()
    second = period(config, 1)
    learner.begin_period(second, apply_at=second.start)
    new_outcome = learner.export_state().payload.last_outcome
    assert new_outcome is not None
    assert new_outcome.to_json() == original
    assert clone(learner).export_state().to_json() == learner.export_state().to_json()
    learner.observe(panel(config, (1.0, 0.0, 0.5)), apply_at=second.start)
    assert learner.release_count == 1
    assert len(learner.export_state().payload.pending) == 1


def test_well_formed_empty_exact_geometry_and_staged_none_are_valid():
    config = configuration(prior=(1 / 3, 1 / 3, 1 / 3), movement=(0.0, 0.0, 0.0))
    learner = ControlledHedge(config, owner=ref("exact-empty-contract-owner"))
    geometry = exact_mixture_intervals(
        config.controls.bounds, learner.export_state().payload.origin, movement=config.controls.movement
    )
    assert geometry.status == "infeasible"
    outcome = learner.observe(panel(config, (0.0, 0.5, 1.0)))
    assert outcome.disposition == "infeasible"
    state = learner.export_state()
    staged = ControlledPayload.model_validate(state.payload.model_copy(update={"last_outcome": None}))
    assert staged.pending == state.payload.pending
    assert clone(learner).export_state().to_json() == state.to_json()


@pytest.mark.parametrize("field", ["pending", "intervals", "periods", "support"])
def test_raw_selected_counts_reject_before_nested_content_is_read(held_owner, field):
    raw = held_owner.export_state().payload.model_dump()
    raw[field] = (None,) * 4097
    with pytest.raises(ValueError, match="count capacity"):
        ControlledPayload.model_validate(raw)


@pytest.mark.parametrize(
    "dto",
    [
        ControlledConfig,
        ControlledLimits,
        ControlledPayload,
        MovementPeriod,
        AcceptedPanel,
        AggregateLoss,
        ControlOutcome,
        ClosedInterval,
    ],
)
@pytest.mark.parametrize("container", [str, bytes, bytearray])
def test_all_new_control_json_routes_bound_lexical_depth_before_decode(dto, container):
    raw = "[" * 1200 + "0" + "]" * 1200
    payload = raw if container is str else container(raw.encode())
    with pytest.raises(ValueError, match="depth.*64"):
        dto.model_validate_json(payload)


def test_interval_pending_requires_both_original_clocks(interval_owner):
    config = interval_owner.config
    learner = ControlledHedge(config, owner=ref("clocked-pending-owner"))
    learner.observe(panel(config, (0.0, 1.0, 0.5)), eligible_at=START, apply_at=START)
    state = learner.export_state()
    pending = state.payload.pending[0].model_copy(update={"eligible_at": None})
    candidate = forged_state(learner, pending=(pending,))
    with pytest.raises(ValueError, match="explicit eligibility"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == state.to_json()


def test_retained_application_cannot_follow_processing_frontier(held_owner):
    state = held_owner.export_state()
    first, second = state.payload.pending
    first = first.model_copy(update={"apply_at": START + timedelta(days=1)})
    candidate = forged_state(held_owner, pending=(first, second), last_apply_at=START)
    with pytest.raises(ValueError, match="last processing clock"):
        held_owner.restore_state(candidate)
    assert held_owner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("change", ["duplicate_panels", "missing_output", "half_solver", "missing_sums", "large_sum"])
def test_attempt_receipt_requires_complete_original_math_inputs(released_owner, change):
    state = released_owner.export_state()
    outcome = state.payload.last_outcome
    assert outcome is not None
    changes = {}
    if change == "duplicate_panels":
        changes = {"panels": (outcome.panels[0], outcome.panels[0])}
    elif change == "missing_output":
        changes = {"output": None}
    elif change == "half_solver":
        changes = {"solver_inputs": None}
    elif change == "missing_sums":
        changes = {"summed_losses": outcome.summed_losses[:-1]}
    else:
        sums = (outcome.summed_losses[0].model_copy(update={"value": 2.0}),) + outcome.summed_losses[1:]
        changes = {"summed_losses": sums}
    candidate = outcome.model_copy(update=changes)
    with pytest.raises(ValueError, match="panel identity|input binding|present together|Block sums|block sum"):
        ControlOutcome.model_validate(candidate)
    assert released_owner.export_state().to_json() == state.to_json()


def test_held_receipt_cannot_hide_a_solver_attempt(held_owner):
    state = held_owner.export_state()
    outcome = state.payload.last_outcome
    assert outcome is not None
    candidate = outcome.model_copy(update={"solver_profile": "binary64-grid-breakpoint-kkt-v1"})
    with pytest.raises(ValueError, match="hidden solver attempt"):
        ControlOutcome.model_validate(candidate)
    assert held_owner.export_state().to_json() == state.to_json()


def test_selected_pending_count_is_checked_before_malformed_nested_panels():
    limits = ControlledLimits(maximum_pending=1)
    config = configuration(limits=limits)
    owner = ControlledHedge(config, owner=ref("small-count-owner"))
    raw = owner.export_state().payload.model_dump()
    raw["pending"] = (None, None)
    with pytest.raises(ValueError, match="pending panels exceeds the selected count"):
        ControlledPayload.model_validate(raw)
    assert owner.update_count == 0


def test_coupled_schema_byte_floor_rejects_before_recursive_panel_validation():
    limits = ControlledLimits(maximum_bytes=20000, maximum_pending=64)
    config = configuration(limits=limits, cadence=64)
    owner = ControlledHedge(config, owner=ref("schema-budget-owner"))
    raw = owner.export_state().payload.model_dump()
    raw["pending"] = (None,) * 64
    with pytest.raises(ValueError, match="schemas and coordinates.*byte capacity"):
        ControlledPayload.model_validate(raw)
    assert owner.update_count == 0


def test_selected_member_capacity_precedes_nested_control_validation():
    raw = configuration().model_dump()
    raw["limits"] = ControlledLimits(maximum_members=2)
    with pytest.raises(ValueError, match="Controlled members exceeds the selected count"):
        ControlledConfig.model_validate(raw)


def test_complete_outer_state_must_fit_selected_bytes_not_just_payload():
    base = configuration()
    owner = ControlledHedge(base, owner=ref("outer-byte-owner"))
    state = owner.export_state()
    ceiling = len(state.payload.to_json().encode("utf-8")) + 100
    assert ceiling < len(state.to_json().encode("utf-8"))
    config = configuration(limits=ControlledLimits(maximum_bytes=ceiling))
    with pytest.raises(ValueError, match="complete state.*byte capacity"):
        ControlledHedge(config, owner=ref("outer-byte-owner"))


@pytest.fixture(scope="module")
def failed_interval_owner():
    calendar = ref("negative-frontier-calendar")
    config = configuration(cadence=calendar, eta=1e308, strength=0.0)
    learner = ControlledHedge(config, owner=ref("negative-frontier-owner"))
    learner.observe(panel(config, (0.0, 1.0, 1.0)), eligible_at=START, apply_at=START)
    first = interval(0)
    failed = learner.release(first, calendar=calendar, apply_at=first.release_at)
    assert failed.disposition == "numerical_failure"
    learner.observe(panel(config, (1.0, 0.0, 0.0)), eligible_at=START, apply_at=first.release_at)
    return learner


@pytest.mark.parametrize("change", ["missing", "nonnegative", "law", "origin", "counter", "future", "panels"])
def test_unresolved_negative_frontier_survives_held_receipts_and_rejects_forgery(failed_interval_owner, change):
    state = failed_interval_owner.export_state()
    failed = state.payload.unresolved_failure
    assert failed is not None
    assert state.payload.last_outcome is not None
    assert state.payload.last_outcome.disposition == "held"
    if change == "missing":
        altered = None
    elif change == "nonnegative":
        altered = state.payload.last_outcome
    elif change == "law":
        altered = failed.model_copy(
            update={"law": failed.law.model_copy(update={"reference": ref("false-negative-law")})}
        )
    elif change == "origin":
        origin = failed.origin.model_copy(update={"reference": ref("false-origin")})
        assert failed.output is not None
        output = failed.output.model_copy(update={"origin": origin})
        altered = failed.model_copy(update={"origin": origin, "output": output})
    elif change == "counter":
        altered = failed.model_copy(update={"release_count": 1, "consumed_panels": 1})
    elif change == "future":
        altered = failed.model_copy(update={"update_count": 3})
    else:
        altered = failed.model_copy(update={"panels": (ref("unowned-failed-panel"),)})
    candidate = forged_state(failed_interval_owner, unresolved_failure=altered)
    with pytest.raises(
        ValueError, match="unresolved|negative|fixed controlled law|current origin|counter frontier|failed panel"
    ):
        failed_interval_owner.restore_state(candidate)
    assert failed_interval_owner.export_state().to_json() == state.to_json()


def test_unresolved_failure_field_is_required_even_when_null(held_owner):
    raw = held_owner.export_state().payload.model_dump()
    del raw["unresolved_failure"]
    with pytest.raises(ValueError, match="(?s)unresolved_failure.*Field required"):
        ControlledPayload.model_validate(raw)


def test_successful_nonempty_release_clears_explicit_frontier_and_refuses_its_restore(failed_interval_owner):
    learner = clone(failed_interval_owner)
    old = learner.export_state().payload.unresolved_failure
    assert old is not None
    second = interval(1)
    result = learner.release(second, calendar=learner.config.controls.cadence.calendar, apply_at=second.release_at)
    assert result.disposition == "released"
    state = learner.export_state()
    assert state.payload.unresolved_failure is None
    assert state.payload.release_count == 1
    assert state.payload.consumed_panels == 2
    candidate = forged_state(learner, unresolved_failure=old)
    with pytest.raises(ValueError, match="later successful release"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize(
    "change", ["profile", "initial_box", "period_clock", "cadence_capacity", "bytes", "initial_origin"]
)
def test_controlled_law_rejects_semantically_incompatible_complete_inputs(change):
    config = configuration()
    raw = config.model_dump()
    if change == "profile":
        raw["profile"] = "controlled-hedge-projection"
    elif change == "initial_box":
        raw["controls"]["bounds"]["bounds"][0]["lower"] = 0.6
    elif change == "period_clock":
        raw["period_clock"] = ref("undeclared-budget-calendar")
    elif change == "cadence_capacity":
        raw["controls"]["cadence"]["panels"] = 2
        raw["limits"] = ControlledLimits(maximum_pending=1)
    elif change == "bytes":
        raw["limits"] = ControlledLimits(maximum_bytes=1)
    else:
        values = (0.25, 0.5, 0.25)
        for weight, value in zip(raw["controls"]["output"]["origin"]["weights"], values, strict=True):
            weight["value"] = value
    with pytest.raises(
        ValueError, match="discriminator|static box|period clock|cadence block|byte capacity|initial emitted"
    ):
        ControlledConfig.model_validate(raw)


@pytest.mark.parametrize("model", [ControlledConfig, ControlledPayload, ControlOutcome, AcceptedPanel])
def test_control_contracts_refuse_nonobject_roots(model):
    with pytest.raises(ValueError, match="dictionary|instance"):
        model.model_validate([])


def test_controlled_config_refuses_missing_nested_prior_without_attribute_callbacks():
    raw = configuration().model_dump()
    raw["controls"] = None
    with pytest.raises(ValueError, match="finite tuple or list"):
        ControlledConfig.model_validate(raw)


def test_movement_period_and_panel_clocks_require_original_order(held_owner):
    config = configuration(movement=(0.1, 0.1, 0.1), scope="per_interval")
    raw_period = period(config, 0).model_dump() | {"end": START}
    with pytest.raises(ValueError, match="start before end"):
        MovementPeriod.model_validate(raw_period)
    item = held_owner.export_state().payload.pending[0]
    raw_panel = item.model_dump() | {"eligible_at": START + timedelta(seconds=1), "apply_at": START}
    with pytest.raises(ValueError, match="eligibility cannot follow"):
        AcceptedPanel.model_validate(raw_panel)


@pytest.mark.parametrize("shape", ["missing_rows", "zero_values", "oversized_values"])
def test_retained_panel_raw_shapes_are_bounded_before_nested_feedback_validation(held_owner, shape):
    raw = held_owner.export_state().payload.pending[0].model_dump()
    if shape == "missing_rows":
        raw["panel"]["rows"] = None
    elif shape == "zero_values":
        raw["panel"]["rows"][0]["values"] = ()
    else:
        raw["panel"]["rows"][0]["values"] = (None, None)
    with pytest.raises(ValueError, match="finite tuple or list|exactly one loss|count capacity"):
        AcceptedPanel.model_validate(raw)


@pytest.mark.parametrize(
    "change", ["missing_failure", "missing_emission", "empty_block", "missing_proposal", "unavailable_proposal"]
)
def test_released_and_negative_receipts_cannot_conflate_presence_laws(released_owner, change):
    state = released_owner.export_state()
    outcome = state.payload.last_outcome
    assert outcome is not None
    if change == "missing_failure":
        fields = {"disposition": "numerical_failure", "emitted_origin": None}
    elif change == "missing_emission":
        fields = {"emitted_origin": None}
    elif change == "empty_block":
        fields = {"panels": ()}
    elif change == "missing_proposal":
        fields = {"proposal": None}
    else:
        fields = {"proposal": None, "proposal_status": "unavailable"}
    candidate = outcome.model_copy(update=fields)
    with pytest.raises(
        ValueError,
        match="explicit failure|committed emitted origin|complete panel block|availability|observed proposal",
    ):
        ControlOutcome.model_validate(candidate)
    assert released_owner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("change", ["release", "availability", "output", "proposal_members"])
def test_held_receipt_and_proposal_coordinates_keep_their_distinct_meanings(held_owner, released_owner, change):
    state = held_owner.export_state()
    outcome = state.payload.last_outcome
    assert outcome is not None
    if change == "release":
        fields = {"release": ref("hidden-release")}
    elif change == "availability":
        fields = {"proposal_status": "unavailable"}
    elif change == "output":
        fields = {"output": held_owner.config.controls.output}
    else:
        released = released_owner.export_state().payload.last_outcome
        assert released is not None
        assert released.proposal is not None
        first = released.proposal[0].model_copy(update={"member": ref("other-proposal-member")})
        fields = {"proposal": (first, *released.proposal[1:])}
        outcome = released
    candidate = outcome.model_copy(update=fields)
    with pytest.raises(
        ValueError, match="hidden mathematical|unavailable proposal|hidden output|exact coefficient member"
    ):
        ControlOutcome.model_validate(candidate)
    assert held_owner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("change", ["unconsumed", "too_many_releases", "zero_release"])
def test_complete_controlled_counter_frontiers_reject_without_assignment(released_owner, change):
    state = released_owner.export_state()
    fields = {"update_count": 2}
    if change == "too_many_releases":
        fields = {"release_count": 2}
    elif change == "zero_release":
        fields = {"release_count": 0}
    candidate = forged_state(released_owner, **fields)
    with pytest.raises(ValueError, match="consumed or retained|release counters"):
        released_owner.restore_state(candidate)
    assert released_owner.export_state().to_json() == state.to_json()


def test_round_cadence_refuses_imported_interval_history(released_owner, interval_owner):
    state = released_owner.export_state()
    candidate = forged_state(released_owner, intervals=interval_owner.export_state().payload.intervals)
    with pytest.raises(ValueError, match="Round cadence cannot retain host"):
        released_owner.restore_state(candidate)
    assert released_owner.export_state().to_json() == state.to_json()


def test_round_cadence_refuses_a_consumed_partial_block_even_with_consistent_totals(held_owner):
    state = held_owner.export_state()
    first, second = state.payload.pending
    pending = (first.model_copy(update={"sequence": 2}), second.model_copy(update={"sequence": 3}))
    candidate = forged_state(
        held_owner, update_count=3, release_count=1, consumed_panels=1, pending=pending, last_outcome=None
    )
    with pytest.raises(ValueError, match="cannot flush or merge"):
        held_owner.restore_state(candidate)
    assert held_owner.export_state().to_json() == state.to_json()


def test_interval_pending_sequence_is_checked_independently_of_round_block_contiguity():
    calendar = ref("sequence-clock")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("sequence-owner"))
    learner.observe(panel(config, (0.0, 0.5, 1.0)), eligible_at=START, apply_at=START)
    learner.observe(panel(config, (1.0, 0.5, 0.0)), eligible_at=START, apply_at=START)
    state = learner.export_state()
    first, second = state.payload.pending
    candidate = forged_state(learner, pending=(first, second.model_copy(update={"sequence": 1})))
    with pytest.raises(ValueError, match="unique original accepted sequence"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("change", ["last_application", "missing_ledger", "missing_period"])
def test_period_geometry_requires_the_complete_current_frontier(period_owner, change):
    state = period_owner.export_state()
    if change == "last_application":
        fields = {"last_apply_at": None}
    elif change == "missing_ledger":
        fields = {"exact_path": None}
    else:
        fields = {"periods": (), "period_origin": None, "exact_path": None, "last_apply_at": None}
    candidate = forged_state(period_owner, **fields)
    with pytest.raises(ValueError, match="processing frontier|complete exact ledger|admitted current period"):
        period_owner.restore_state(candidate)
    assert period_owner.export_state().to_json() == state.to_json()


def test_last_receipt_cannot_replace_both_weight_and_emitted_snapshot_while_current_state_is_unchanged(released_owner):
    state = released_owner.export_state()
    outcome = state.payload.last_outcome
    assert outcome is not None
    assert outcome.emitted_origin is not None
    weights = tuple(
        Weight(member=item.member, value=value) for item, value in zip(outcome.weights, (0.5, 0.25, 0.25), strict=True)
    )
    origin = outcome.emitted_origin.model_copy(update={"weights": weights})
    forged = outcome.model_copy(update={"weights": weights, "emitted_origin": origin})
    candidate = forged_state(released_owner, last_outcome=forged)
    with pytest.raises(ValueError, match="current counters and emitted coefficients"):
        released_owner.restore_state(candidate)
    assert released_owner.export_state().to_json() == state.to_json()


@pytest.fixture(params=["future_release", "unknown_processing"])
def invalid_closed_clock(request):
    calendar = ref("closed-clock-calendar")
    config = configuration(cadence=calendar)
    learner = ControlledHedge(config, owner=ref("closed-clock-owner"))
    first = interval(0)
    learner.release(first, calendar=calendar, apply_at=first.release_at)
    state = learner.export_state()
    assert state.payload.last_apply_at == first.release_at
    assert first.end > first.release_at
    if request.param == "future_release":
        boundaries = {
            name: getattr(first, name) + timedelta(days=1) for name in ("start", "end", "cutoff", "release_at")
        }
        closed = state.payload.intervals[0].model_copy(update={"interval": first.model_copy(update=boundaries)})
        changes = {"intervals": (closed,)}
    else:
        changes = {"last_apply_at": None}
    payload = state.payload.model_copy(update=changes)
    candidate = state.model_copy(update={"payload": payload})
    return learner, state, candidate


def test_closed_release_clock_cannot_pass_the_restored_processing_frontier(invalid_closed_clock):
    learner, before, candidate = invalid_closed_clock
    with pytest.raises(ValueError, match="Closed release.*processing frontier"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == before.to_json()


def test_closed_release_clock_is_rejected_by_the_actual_bounded_artifact_reader(invalid_closed_clock):
    learner, before, candidate = invalid_closed_clock
    # model_dump does not revalidate an unchecked model_copy; the reader must.
    raw = candidate.model_dump_json()
    expected = learner.binding
    with pytest.raises(ValueError, match="Closed release.*processing frontier"):
        read_algorithm_state(raw, payload_type=ControlledPayload, expected=expected)
    assert learner.export_state().to_json() == before.to_json()


@pytest.fixture(scope="module", params=["net_displacement", "accumulated_absolute"])
def historical_period_owner(request):
    calendar = ref("historical-release-calendar")
    config = configuration(
        cadence=calendar,
        movement=(0.4, 0.4, 0.4),
        scope="per_interval",
        measure=request.param,
        lower=(0.1, 0.1, 0.1),
        upper=(0.8, 0.8, 0.8),
        strength=0.05,
    )
    learner = ControlledHedge(config, owner=ref("historical-context-" + request.param))
    first_period = period(config, 0).model_copy(update={"end": START + timedelta(days=2)})
    learner.begin_period(first_period, apply_at=START)
    learner.observe(panel(config, (0.0, 1.0, 1.0)), eligible_at=START, apply_at=START)
    first_interval = interval(0)
    first = learner.release(first_interval, calendar=calendar, apply_at=first_interval.release_at)
    assert first.disposition == "released"
    assert first.weights != first.origin.weights
    assert first.proposal is not None
    assert first.output is not None
    original = solve_mixture_output(
        proposal=first.proposal,
        bounds=config.controls.bounds,
        origin=first.origin,
        output=first.output,
        movement=config.controls.movement,
        usage=first.movement_usage,
        exact_path=first.exact_path_before,
    )
    assert original.status == "optimal"
    assert original.certificate is not None
    learner.observe(
        panel(config, (1.0, 0.0, 0.0)), eligible_at=START + timedelta(days=1), apply_at=START + timedelta(days=1)
    )
    second_interval = interval(1)
    second = learner.release(second_interval, calendar=calendar, apply_at=second_interval.release_at)
    assert second.disposition == "released"
    next_period = period(config, 2)
    learner.begin_period(next_period, apply_at=next_period.start)
    return learner, first, original


def test_historical_positive_penalty_solve_reproduces_after_later_release_and_period_reset(historical_period_owner):
    learner, first, original = historical_period_owner
    state = learner.export_state()
    read = read_algorithm_state(state.to_json(), payload_type=ControlledPayload, expected=learner.binding)
    receipt = read.payload.intervals[0].outcome
    assert receipt.to_json() == first.to_json()
    assert len(read.payload.periods) == 2
    assert read.payload.release_count == 2
    assert receipt.proposal is not None
    assert receipt.output is not None
    restored = solve_mixture_output(
        proposal=receipt.proposal,
        bounds=read.payload.config.controls.bounds,
        origin=receipt.origin,
        output=receipt.output,
        movement=read.payload.config.controls.movement,
        usage=receipt.movement_usage,
        exact_path=receipt.exact_path_before,
    )
    assert restored.status == "optimal"
    assert restored.weights == first.weights
    assert restored.geometry.inputs_fingerprint == receipt.solver_inputs
    assert restored.certificate == original.certificate
    if receipt.exact_path_before is not None:
        assert read.payload.exact_path is not None
        assert receipt.exact_path_before.interval != read.payload.exact_path.interval
    assert clone(learner).export_state().to_json() == state.to_json()


@pytest.mark.parametrize("change", ["missing", "period", "period_origin", "solver_inputs", "wrong_profile"])
def test_historical_release_context_refuses_missing_or_forged_geometry_before_assignment(
    historical_period_owner, change
):
    learner, _, _ = historical_period_owner
    state = learner.export_state()
    closed = state.payload.intervals[0]
    receipt = closed.outcome
    net = receipt.movement_usage
    path = receipt.exact_path_before
    if change == "solver_inputs":
        altered = receipt.model_copy(update={"solver_inputs": "sha256:" + "f" * 64})
    elif change == "missing":
        altered = receipt.model_copy(update={"movement_usage": None, "exact_path_before": None})
    elif change == "wrong_profile":
        if net is not None:
            movement = learner.config.controls.movement.model_copy(update={"measure": "accumulated_absolute"})
            extra = initial_exact_path_usage(interval=net.interval, movement=movement, origin=net.origin)
            fields = {"exact_path_before": extra}
        else:
            extra = MovementUsage(
                interval=path.interval,
                origin=path.period_origin,
                used=tuple(MovementLimit(member=item.member, maximum=0.0) for item in path.period_origin.weights),
            )
            fields = {"movement_usage": extra}
        altered = receipt.model_copy(update=fields)
    else:
        context = net if net is not None else path
        assert context is not None
        if change == "period":
            changed = context.model_copy(update={"interval": ref("never-admitted-period")})
        else:
            origin = net.origin if net is not None else path.period_origin
            weights = tuple(
                item.model_copy(update={"value": value})
                for item, value in zip(origin.weights, (0.0, 0.5, 0.5), strict=True)
            )
            origin = origin.model_copy(update={"weights": weights})
            fields = (
                {"origin": origin} if net is not None else {"period_origin": origin, "through_origin": origin.binding}
            )
            changed = context.model_copy(update=fields)
        key = "movement_usage" if net is not None else "exact_path_before"
        altered = receipt.model_copy(update={key: changed})
    replacement = closed.model_copy(update={"outcome": altered})
    candidate = forged_state(learner, intervals=(replacement, *state.payload.intervals[1:]))
    with pytest.raises(
        ValueError, match="original movement usage|original exact path|admitted period|static box|fingerprint"
    ):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("field", ["movement_usage", "exact_path_before"])
def test_receipt_context_fields_are_required_even_when_null(released_owner, field):
    receipt = released_owner.export_state().payload.last_outcome
    assert receipt is not None
    raw = receipt.model_dump()
    del raw[field]
    with pytest.raises(ValueError, match="Field required"):
        ControlOutcome.model_validate(raw)


def test_held_receipt_cannot_embed_a_hidden_period_solve_context(held_owner, historical_period_owner):
    learner, first, _ = historical_period_owner
    state = held_owner.export_state()
    held = state.payload.last_outcome
    assert held is not None
    candidate = held.model_copy(
        update={"movement_usage": first.movement_usage, "exact_path_before": first.exact_path_before}
    )
    with pytest.raises(ValueError, match="hidden pre-attempt movement context"):
        ControlOutcome.model_validate(candidate)
    assert held_owner.export_state().to_json() == state.to_json()
    assert learner.release_count == 2


def test_historical_usage_cannot_understate_the_original_preattempt_displacement(historical_period_owner):
    learner, _, _ = historical_period_owner
    state = learner.export_state()
    closed = state.payload.intervals[1]
    outcome = closed.outcome
    if outcome.movement_usage is not None:
        usage = outcome.movement_usage
        used = tuple(item.model_copy(update={"maximum": 0.0}) for item in usage.used)
        altered = outcome.model_copy(update={"movement_usage": usage.model_copy(update={"used": used})})
    else:
        path = outcome.exact_path_before
        assert path is not None
        used = tuple(item.model_copy(update={"atoms_decimal": "0"}) for item in path.used)
        altered = outcome.model_copy(update={"exact_path_before": path.model_copy(update={"used": used})})
    replacement = closed.model_copy(update={"outcome": altered})
    candidate = forged_state(learner, intervals=(state.payload.intervals[0], replacement))
    with pytest.raises(ValueError, match="usage.*origins|below known net displacement"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("historical_period_owner", ["accumulated_absolute"], indirect=True)
def test_historical_exact_context_cannot_change_the_complete_fixed_movement_law(historical_period_owner):
    learner, _, _ = historical_period_owner
    state = learner.export_state()
    closed = state.payload.intervals[0]
    path = closed.outcome.exact_path_before
    assert path is not None
    movement = path.movement.model_copy(update={"reference": ref("different-historical-movement-law")})
    context = path.model_copy(update={"movement": movement})
    altered = closed.outcome.model_copy(update={"exact_path_before": context})
    replacement = closed.model_copy(update={"outcome": altered})
    candidate = forged_state(learner, intervals=(replacement, *state.payload.intervals[1:]))
    with pytest.raises(ValueError, match="complete fixed movement law"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == state.to_json()


@pytest.fixture(
    params=["false_success", "false_infeasible", "unavailable_infeasible", "outside_movement", "postsolve_numeric"]
)
def invalid_solver_disposition(request):
    if request.param == "outside_movement":
        config = configuration(movement=(0.1, 0.1, 0.1))
        learner = ControlledHedge(config, owner=ref("outside-release-movement-owner"))
        result = learner.observe(panel(config, (0.0, 1.0, 0.5)))
        assert result.disposition == "released"
        assert result.emitted_origin is not None
        before = learner.export_state()
        weights = tuple(
            item.model_copy(update={"value": value})
            for item, value in zip(result.weights, (0.7, 0.15, 0.15), strict=True)
        )
        emitted = result.emitted_origin.model_copy(update={"weights": weights})
        outcome = result.model_copy(update={"weights": weights, "emitted_origin": emitted})
        payload = before.payload.model_copy(update={"origin": emitted, "last_outcome": outcome})
        candidate = before.model_copy(update={"payload": payload})
    elif request.param == "false_infeasible":
        calendar = ref("false-infeasibility-calendar")
        config = configuration(cadence=calendar)
        learner = ControlledHedge(config, owner=ref("false-infeasibility-owner"))
        learner.observe(panel(config, (0.0, 1.0, 1.0)), eligible_at=START, apply_at=START)
        before = learner.export_state()
        point = interval(0)
        result = learner.release(point, calendar=calendar, apply_at=point.release_at)
        assert result.disposition == "released"
        outcome = result.model_copy(
            update={
                "disposition": "infeasible",
                "release_count": 0,
                "consumed_panels": 0,
                "weights": before.payload.origin.weights,
                "emitted_origin": None,
                "failure": LearningFailure(code="infeasible", message="Declared empty exact set"),
            }
        )
        closed = ClosedInterval(interval=point, outcome=outcome)
        payload = before.payload.model_copy(
            update={
                "intervals": (closed,),
                "last_apply_at": point.release_at,
                "last_outcome": outcome,
                "unresolved_failure": outcome,
            }
        )
        candidate = before.model_copy(update={"payload": payload})
    else:
        config = configuration(prior=(1 / 3,) * 3, movement=(0.0,) * 3)
        learner = ControlledHedge(config, owner=ref("false-success-owner"))
        result = learner.observe(panel(config, (0.0, 0.5, 1.0)))
        assert result.disposition == "infeasible"
        before = learner.export_state()
        if request.param == "false_success":
            emitted = before.payload.origin.model_copy(update={"reference": ref("fabricated-success-origin")})
            outcome = result.model_copy(
                update={
                    "disposition": "released",
                    "release_count": 1,
                    "consumed_panels": 1,
                    "emitted_origin": emitted,
                    "failure": None,
                }
            )
            payload = before.payload.model_copy(
                update={
                    "origin": emitted,
                    "release_count": 1,
                    "consumed_panels": 1,
                    "pending": (),
                    "last_outcome": outcome,
                    "unresolved_failure": None,
                }
            )
        elif request.param == "postsolve_numeric":
            outcome = result.model_copy(
                update={
                    "disposition": "numerical_failure",
                    "failure": LearningFailure(code="numeric_range", message="Claimed post-solver failure"),
                }
            )
            payload = before.payload.model_copy(update={"last_outcome": outcome, "unresolved_failure": outcome})
        else:
            outcome = result.model_copy(
                update={
                    "proposal": None,
                    "proposal_status": "unavailable",
                    "solver_profile": None,
                    "solver_inputs": None,
                }
            )
            payload = before.payload.model_copy(update={"last_outcome": outcome, "unresolved_failure": outcome})
        candidate = before.model_copy(update={"payload": payload})
    return learner, learner.export_state(), candidate


def test_actual_restore_refuses_receipt_status_contradicting_original_geometry(invalid_solver_disposition):
    learner, before, candidate = invalid_solver_disposition
    with pytest.raises(ValueError, match="geometry disposition|observed proposal and solver|effective box"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == before.to_json()


def test_actual_bounded_reader_refuses_receipt_status_contradicting_original_geometry(invalid_solver_disposition):
    learner, before, candidate = invalid_solver_disposition
    raw = candidate.model_dump_json()
    expected = learner.binding
    with pytest.raises(ValueError, match="geometry disposition|observed proposal and solver|effective box"):
        read_algorithm_state(raw, payload_type=ControlledPayload, expected=expected)
    assert learner.export_state().to_json() == before.to_json()


def test_numerical_proposal_failure_remains_valid_before_solving_an_empty_exact_set():
    config = configuration(prior=(1 / 3,) * 3, movement=(0.0,) * 3, eta=1e308)
    learner = ControlledHedge(config, owner=ref("empty-set-before-proposal-failure"))
    result = learner.observe(panel(config, (0.0, 1.0, 1.0)))
    assert result.disposition == "numerical_failure"
    assert result.proposal is None
    assert result.solver_inputs is None
    assert result.failure is not None
    assert result.failure.code == "positive_underflow"
    assert learner.update_count == 1
    assert learner.release_count == 0
    assert clone(learner).export_state().to_json() == learner.export_state().to_json()
