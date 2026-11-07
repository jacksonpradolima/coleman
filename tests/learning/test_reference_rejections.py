"""Real bounded panel/state admission refuses before changing a learner."""

import json
from typing import Any, cast

import pytest

from coleman.core.feedback import FeedbackRow, FullPanel
from coleman.learning.contracts import (
    MAXIMUM_BYTES,
    LearningFailure,
    LearningLimits,
    LearningNumericalError,
    LearningUpdate,
    ReferenceConfig,
    ReferencePayload,
    read_loss_panel,
)
from coleman.learning.reference import AnytimeHedge, Hedge
from coleman.state.contracts import AlgorithmState
from tests.learning.test_reference import configuration, panel, ref


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"eta": -1},
        {"eta": float("inf")},
        {"eta": True},
        {"alpha": 0.1},
        {"anytime_multiplier": 2},
        {"profile": "future-hedge"},
        {"specialists": True},
        {"catalogue_mode": "dynamic"},
        {"untyped_metadata": {"loss": 0}},
    ],
)
def test_reference_configuration_rejects_unknown_future_and_incompatible_laws(changes):
    original = configuration()
    candidate = original.model_dump() | changes
    with pytest.raises(ValueError):
        ReferenceConfig.model_validate(candidate)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_fixed_share_rejects_invalid_or_nonfinite_fraction(alpha):
    original = configuration("fixed-share-excluding-origin")
    candidate = original.model_dump() | {"alpha": alpha}
    with pytest.raises(ValueError):
        ReferenceConfig.model_validate(candidate)


@pytest.mark.parametrize("field", ["maximum_members", "maximum_updates", "maximum_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_selected_capacities_are_positive_strict_and_finitely_bounded(field, value):
    candidate = {field: value}
    with pytest.raises(ValueError):
        LearningLimits.model_validate(candidate)


@pytest.mark.parametrize(
    "changes",
    [{"role": "feature"}, {"minimum": None}, {"maximum": 2.0}, {"semantics": {"unit": "USD", "meaning": "money"}}],
)
def test_reference_configuration_rejects_wrong_loss_role_units_or_range(changes):
    original = configuration()
    loss = original.loss_spec.model_dump() | changes
    candidate = original.model_dump() | {"loss_spec": loss}
    with pytest.raises(ValueError, match="loss specification"):
        ReferenceConfig.model_validate(candidate)


def test_panel_validation_work_is_bounded_even_when_typed_rows_share_one_large_specification():
    original = configuration()
    specification = original.loss_spec.model_copy(
        update={"semantics": original.loss_spec.semantics.model_copy(update={"meaning": "x" * (6 * 1024**2)})}
    )
    original_panel = panel(original, (0.0, 0.5, 1.0))
    rows = tuple(
        row.model_copy(update={"values": (row.values[0].model_copy(update={"spec": specification}),)})
        for row in original_panel.rows
    )
    offered = FullPanel.model_construct(rows=rows)
    members = tuple(item.member for item in original.prior.weights)
    with pytest.raises(ValueError, match="scalar-schema validation byte capacity"):
        read_loss_panel(offered, members=members, loss_spec=specification, transform=original.transform)


def test_positive_anytime_schedule_underflow_has_an_atomic_explicit_failure():
    config = configuration("hedge-anytime-cumulative", prior=(0.5, 0.5), anytime_multiplier=5e-324)
    learner = AnytimeHedge(config, owner=ref("tiny-rate"))
    offered = panel(config, (0.0, 0.0))
    assert learner.observe(offered).status == "accepted"
    before = learner.export_state().to_json()
    result = learner.observe(offered)
    assert result.status == "numerical_failure"
    assert result.failure is not None
    assert result.failure.code == "positive_underflow"
    assert learner.export_state().to_json() == before


def test_unrepresentable_initial_schedule_fails_before_constructing_an_owner_state():
    config = configuration("hedge-anytime-cumulative", anytime_multiplier=1.7976931348623157e308)
    owner = ref("overflow-rate")
    with pytest.raises(LearningNumericalError, match="next anytime rate"):
        AnytimeHedge(config, owner=owner)


def test_zero_reference_prior_and_wrong_constructor_profile_are_explicit_refusals():
    controlled_style = configuration().model_dump()
    controlled_style["prior"]["weights"][0]["value"] = 0
    controlled_style["prior"]["weights"][1]["value"] = 0.5
    controlled_style["prior"]["weights"][2]["value"] = 0.5
    with pytest.raises(ValueError, match="strictly positive"):
        ReferenceConfig.model_validate(controlled_style)
    config = configuration("hedge-anytime-cumulative")
    owner = ref("wrong-method")
    with pytest.raises(ValueError, match="profile differ"):
        Hedge(config, owner=owner)


@pytest.mark.parametrize(
    "change", ["missing", "member-version", "transform", "specification", "signed-zero", "null", "range", "extra-value"]
)
def test_wrong_missing_or_incompatible_full_loss_panel_preserves_complete_state(change):
    config = configuration()
    learner = Hedge(config, owner=ref("unchanged"))
    offered = panel(config, (0.0, 0.5, 1.0))
    first = offered.rows[0]
    value = first.values[0]
    if change == "missing":
        offered = FullPanel(rows=offered.rows[:-1])
    elif change == "member-version":
        changed = FeedbackRow(member=ref(first.member.id, "2"), values=first.values)
        offered = FullPanel(rows=(changed, *offered.rows[1:]))
    elif change == "extra-value":
        other = value.model_copy(update={"spec": value.spec.model_copy(update={"dimension": ref("other-loss")})})
        changed = FeedbackRow.model_construct(member=first.member, values=(value, other))
        offered = FullPanel.model_construct(rows=(changed, *offered.rows[1:]))
    else:
        updates = {
            "transform": {"transform": ref("other-transform")},
            "specification": {"spec": config.loss_spec.model_copy(update={"dimension": ref("other-loss")})},
            "signed-zero": {"spec": config.loss_spec.model_copy(update={"minimum": -0.0})},
            "null": {"value": None, "status": "unavailable"},
            "range": {"value": 1.5},
        }[change]
        changed_value = value.model_copy(update=updates)
        changed = FeedbackRow.model_construct(member=first.member, values=(changed_value,))
        offered = FullPanel.model_construct(rows=(changed, *offered.rows[1:]))
    before = learner.export_state().to_json()
    with pytest.raises(ValueError):
        learner.observe(offered)
    assert learner.export_state().to_json() == before
    assert learner.update_count == 0


@pytest.mark.parametrize("shape", ["rows-list", "oversized-rows", "custom-rows", "custom-values", "oversized-values"])
def test_unchecked_panel_cardinality_refuses_before_enumerating_custom_containers(shape):
    config = configuration()
    original = panel(config, (0.0, 0.5, 1.0))
    calls = []

    class Unsupported(tuple):
        def __iter__(self):
            calls.append("iteration")
            return super().__iter__()

        def __len__(self):
            calls.append("length")
            return super().__len__()

    if shape == "rows-list":
        rows = list(original.rows)
    elif shape == "oversized-rows":
        rows = original.rows[:1] * 4097
    elif shape == "custom-rows":
        rows = Unsupported(original.rows)
    else:
        values = Unsupported(original.rows[0].values) if shape == "custom-values" else original.rows[0].values * 65
        row = FeedbackRow.model_construct(member=original.rows[0].member, values=values)
        rows = (row, *original.rows[1:])
    offered = FullPanel.model_construct(rows=rows)
    learner = Hedge(config, owner=ref("bounded"))
    before = learner.export_state().to_json()
    with pytest.raises(ValueError):
        learner.observe(offered)
    assert not calls
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize(
    "change",
    ["owner", "codec", "implementation", "fingerprint", "configuration-zero", "weights", "compensation", "future"],
)
def test_restore_refuses_incompatible_complete_state_before_assignment(change):
    config = configuration(prior=(0.5, 0.5), eta=0.0)
    learner = Hedge(config, owner=ref("retained-owner"))
    state = learner.export_state()
    if change in {"owner", "codec", "implementation", "fingerprint"}:
        update = {
            "owner": {"owner": ref("foreign-owner")},
            "codec": {"state_codec": ref("foreign-codec")},
            "implementation": {
                "implementation": state.binding.implementation.model_copy(update={"reference": ref("foreign-method")})
            },
            "fingerprint": {"configuration_fingerprint": "sha256:" + "f" * 64},
        }[change]
        candidate = state.model_copy(update={"binding": state.binding.model_copy(update=update)})
    elif change == "configuration-zero":
        changed = config.model_copy(update={"eta": -0.0})
        candidate = state.model_copy(update={"payload": state.payload.model_copy(update={"configuration": changed})})
    elif change == "weights":
        weights = tuple(
            item.model_copy(update={"value": value})
            for item, value in zip(state.payload.weights, (0.25, 0.75), strict=True)
        )
        candidate = state.model_copy(update={"payload": state.payload.model_copy(update={"weights": weights})})
    elif change == "compensation":
        candidate = state.model_copy(update={"payload": state.payload.model_copy(update={"compensations": (1.0, 0.0)})})
    else:
        candidate = state.model_copy(update={"schema_version": 2})
    before = learner.export_state().to_json()
    with pytest.raises(ValueError):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == before


def test_anytime_restore_recomputes_weights_from_complete_cumulative_statistics():
    config = configuration("hedge-anytime-cumulative", prior=(0.5, 0.5))
    learner = AnytimeHedge(config, owner=ref("anytime"))
    assert learner.observe(panel(config, (0.0, 1.0))).status == "accepted"
    state = learner.export_state()
    changed = state.payload.model_copy(update={"cumulative_losses": (0.5, 1.0)})
    candidate = state.model_copy(update={"payload": changed})
    before = state.to_json()
    with pytest.raises(ValueError, match="coefficients disagree"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == before


def test_panel_capacity_refuses_without_eviction_or_a_second_state_assignment():
    config = configuration(limits=LearningLimits(maximum_updates=1))
    learner = Hedge(config, owner=ref("limited"))
    offered = panel(config, (0.0, 0.0, 0.0))
    assert learner.observe(offered).status == "accepted"
    before = learner.export_state().to_json()
    with pytest.raises(ValueError, match="capacity"):
        learner.observe(offered)
    assert learner.export_state().to_json() == before


def test_selected_complete_state_bytes_are_checked_before_update_assignment():
    original = configuration()
    owner = ref("byte-limited")
    baseline = Hedge(original, owner=owner)
    capacity = len(baseline.export_state().to_json().encode("utf-8"))
    config = configuration(limits=LearningLimits(maximum_bytes=capacity))
    learner = Hedge(config, owner=owner)
    before = learner.export_state().to_json()
    assert len(before.encode("utf-8")) <= capacity
    offered = panel(config, (0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="complete state exceeds"):
        learner.observe(offered)
    assert learner.export_state().to_json() == before


def test_known_wire_keeps_unknown_duplicate_and_future_fields_strict():
    config = configuration()
    original = config.to_json()
    assert ReferenceConfig.from_json(original).to_json() == original
    for raw in (original[:-1] + ',"unknown":null}', '{"schema_version":1,' + original[1:]):
        with pytest.raises(ValueError):
            ReferenceConfig.from_json(raw)
    state = Hedge(config, owner=ref("wire")).export_state()
    raw = json.loads(state.to_json())
    raw["payload"]["schema_version"] = 2
    serialized = json.dumps(raw)
    with pytest.raises(ValueError):
        AlgorithmState[ReferencePayload].from_json(serialized)


def test_reference_payload_and_prior_cardinality_are_bounded_before_deep_validation():
    config = configuration()
    candidate = config.model_copy(
        update={"prior": config.prior.model_copy(update={"weights": config.prior.weights * 1366})}
    )
    with pytest.raises(ValueError, match="member capacity"):
        ReferenceConfig.model_validate(candidate)
    learner = Hedge(config, owner=ref("bounded-state"))
    state = learner.export_state()
    candidate_state = state.model_copy(
        update={"payload": state.payload.model_copy(update={"cumulative_losses": (0.0,) * 4097})}
    )
    before = state.to_json()
    with pytest.raises(ValueError, match="bounded vector"):
        learner.restore_state(candidate_state)
    assert learner.export_state().to_json() == before


@pytest.mark.parametrize("reader", [ReferenceConfig.from_json, ReferenceConfig.model_validate_json])
@pytest.mark.parametrize("container", [str, bytes, bytearray])
def test_each_new_reference_reader_rejects_deep_input_before_json_recursion(reader, container):
    text = "[" * 1200 + "0" + "]" * 1200
    payload = text if container is str else container(text.encode("utf-8"))
    unsupported = reader == ReferenceConfig.from_json and container is bytearray
    expected = TypeError if unsupported else ValueError
    message = "built-in str or bytes" if unsupported else "depth.*64"
    with pytest.raises(expected, match=message):
        reader(payload)


@pytest.mark.parametrize("reader", [ReferenceConfig.from_json, ReferenceConfig.model_validate_json])
@pytest.mark.parametrize("container", [str, bytes, bytearray])
def test_each_new_reference_reader_bounds_complete_input_before_json_decoding(reader, container):
    original = configuration().to_json()
    text = original + " " * (MAXIMUM_BYTES + 1)
    payload = text if container is str else container(text.encode("utf-8"))
    unsupported = reader == ReferenceConfig.from_json and container is bytearray
    expected = TypeError if unsupported else ValueError
    message = "built-in str or bytes" if unsupported else "maximum_bytes|byte ceiling"
    with pytest.raises(expected, match=message):
        reader(payload)


@pytest.mark.parametrize("container", [str, bytes, bytearray])
def test_alternate_new_reference_readers_preserve_valid_fields_and_core_policies(container):
    config = configuration()
    learner = Hedge(config, owner=ref("bounded-readers"))
    result = learner.observe(panel(config, (0.0, 0.5, 1.0)))
    for model in (config, learner.export_state().payload, result):
        encoded = model.to_json()
        payload = encoded if container is str else container(encoded.encode("utf-8"))
        restored = type(model).model_validate_json(payload)
        assert restored.to_json() == encoded
    for model in (LearningLimits(), LearningFailure(code="normalization", message="typed failure")):
        encoded = model.model_dump_json()
        payload = encoded if container is str else container(encoded.encode("utf-8"))
        assert type(model).model_validate_json(payload).model_dump_json() == encoded
    original = config.to_json()
    with pytest.raises(ValueError):
        ReferenceConfig.model_validate_json(original, strict=False)
    with pytest.raises(ValueError):
        ReferenceConfig.model_validate_json(original, extra="allow")
    with pytest.raises(ValueError):
        ReferenceConfig.model_validate_json(original, extra="ignore")


@pytest.mark.parametrize("contract", [ReferencePayload, LearningUpdate, LearningLimits, LearningFailure])
def test_all_new_reference_nested_and_versioned_json_readers_apply_the_depth_guard(contract):
    payload = "[" * 1200 + "0" + "]" * 1200
    with pytest.raises(ValueError, match="depth.*64"):
        contract.model_validate_json(payload)


@pytest.mark.parametrize("model", [ReferenceConfig, ReferencePayload])
def test_new_learning_readers_refuse_nonobject_roots(model):
    with pytest.raises(ValueError, match="dictionary|instance"):
        model.model_validate([])


def test_new_learning_json_reader_refuses_callback_bearing_or_nonbuiltin_content():
    unsupported = cast(Any, memoryview(b"{}"))
    with pytest.raises(TypeError, match="built-in str, bytes or bytearray"):
        ReferenceConfig.model_validate_json(unsupported)


def test_anytime_configuration_rejects_constant_rate_even_when_finite():
    raw = configuration("hedge-anytime-cumulative").model_dump() | {"eta": 0.5}
    with pytest.raises(ValueError, match="schedule, not constant eta"):
        ReferenceConfig.model_validate(raw)


def test_selected_bytes_reject_the_complete_configuration_before_owner_construction():
    raw = configuration().model_dump() | {"limits": LearningLimits(maximum_bytes=1)}
    with pytest.raises(ValueError, match="complete content.*byte capacity"):
        ReferenceConfig.model_validate(raw)


@pytest.mark.parametrize(
    "status,failure",
    [("accepted", LearningFailure(code="normalization", message="failed")), ("numerical_failure", None)],
)
def test_learning_result_status_cannot_hide_or_fabricate_failure(status, failure):
    owner = Hedge(configuration(), owner=ref("result-consistency"))
    raw = {"schema_version": 1, "status": status, "update_count": 0, "weights": owner.weights, "failure": failure}
    with pytest.raises(ValueError, match="status and failure must agree"):
        LearningUpdate.model_validate(raw)
    assert owner.update_count == 0


@pytest.mark.parametrize("change", ["member", "statistics", "zero", "updates", "total", "corrected_total"])
def test_reference_restore_refuses_impossible_support_or_sufficient_statistics(change):
    config = configuration(limits=LearningLimits(maximum_updates=1))
    owner = Hedge(config, owner=ref("sufficient-state-owner"))
    owner.observe(panel(config, (0.0, 0.5, 1.0)))
    state = owner.export_state()
    changes = {}
    if change == "member":
        first = state.payload.weights[0]
        changes = {
            "weights": (first.model_copy(update={"member": ref(first.member.id, "2")}), *state.payload.weights[1:])
        }
    elif change == "statistics":
        changes = {"cumulative_losses": (0.0, 0.5)}
    elif change == "zero":
        changes = {
            "weights": tuple(
                item.model_copy(update={"value": value})
                for item, value in zip(state.payload.weights, (0.0, 0.5, 0.5), strict=True)
            )
        }
    elif change == "updates":
        changes = {"update_count": 2}
    elif change == "total":
        changes = {"cumulative_losses": (2.0, 0.5, 1.0)}
    else:
        changes = {"compensations": (-1e-17, 0.0, 0.0)}
    candidate = state.model_copy(update={"payload": state.payload.model_copy(update=changes)})
    with pytest.raises(ValueError, match="exact fixed member|statistics|zero support|capacity|cumulative losses"):
        owner.restore_state(candidate)
    assert owner.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("members", [[], (), (ref("expert-2"), ref("expert-1"), ref("expert-0"))])
def test_public_loss_reader_requires_a_finite_canonical_member_inventory(members):
    config = configuration()
    offered = panel(config, (0.0, 0.5, 1.0))
    with pytest.raises(ValueError, match="immutable tuple|canonical"):
        read_loss_panel(offered, members=members, loss_spec=config.loss_spec, transform=config.transform)


def test_loss_reader_refuses_an_untyped_nested_outcome_before_deep_validation():
    config = configuration()
    offered = panel(config, (0.0, 0.5, 1.0))
    first = offered.rows[0].model_copy(update={"values": ({"value": 0.0},)})
    offered = offered.model_copy(update={"rows": (first, *offered.rows[1:])})
    members = tuple(item.member for item in config.prior.weights)
    with pytest.raises(ValueError, match="concrete typed loss outcome"):
        read_loss_panel(offered, members=members, loss_spec=config.loss_spec, transform=config.transform)


def test_restore_rejects_a_bare_payload_instead_of_its_concrete_complete_state():
    learner = Hedge(configuration(), owner=ref("complete-required"))
    state = learner.export_state()
    offered = cast(Any, state.payload)
    with pytest.raises(TypeError, match="exact concrete AlgorithmState"):
        learner.restore_state(offered)
    assert learner.export_state().to_json() == state.to_json()


def test_anytime_restore_refuses_admitted_statistics_that_cannot_reproduce_finite_positive_weights():
    config = configuration("hedge-anytime-cumulative", anytime_multiplier=1e308, prior=(0.5, 0.5))
    learner = AnytimeHedge(config, owner=ref("unreproducible-anytime"))
    state = learner.export_state()
    payload = state.payload.model_copy(update={"update_count": 1, "cumulative_losses": (0.0, 1.0)})
    candidate = state.model_copy(update={"payload": payload})
    with pytest.raises(ValueError, match="cannot reproduce a finite emitted vector"):
        learner.restore_state(candidate)
    assert learner.export_state().to_json() == state.to_json()
