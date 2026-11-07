"""Exercise real direct snapshot readers, original producers and callback order."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from coleman.core.actions import ChoiceSpace, MixtureAction, MixtureSpace, Weight
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision, StateOwner
from coleman.core.feedback import FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.learning.contracts import LearningUpdate, ReferenceConfig
from coleman.learning.direct import DirectMixture, LearningApplicationError
from coleman.learning.reference import Hedge
from tests.learning.test_direct_conformance import OBJECTIVE, TASK, _cases, _config


def _ref(name):
    return Reference(id=name, version="v1")


class _CountingHedge(Hedge):
    """Count entry to actual shipped restoration without replacing its logic."""

    def __init__(self, config, *, owner):
        super().__init__(config, owner=owner)
        self.restore_calls = 0

    def restore_state(self, state, /):
        self.restore_calls += 1
        super().restore_state(state)


def _bridge(*, maximum_bytes=16777216, horizon=None):
    kernel = _CountingHedge(_config(), owner=_ref("direct-state-owner"))
    adapter = DirectMixture(kernel, task=TASK, objective=OBJECTIVE, maximum_bytes=maximum_bytes, horizon=horizon)
    return adapter, kernel


def _execution(adapter):
    return adapter.export_state().pending[0]


def _batch(executed, config, losses=(1.0, 0.0), *, horizon=None, timing=None):
    panel = FullPanel(
        rows=tuple(
            FeedbackRow(
                member=weight.member,
                values=(
                    OutcomeValue(
                        spec=config.loss_spec,
                        value=value,
                        status="observed",
                        origin="synthetic",
                        transform=config.transform.reference,
                    ),
                ),
            )
            for weight, value in zip(config.prior.weights, losses, strict=True)
        )
    )
    return FeedbackBatch(
        schema_version=1,
        decision_fingerprint=executed.fingerprint,
        feedback_id="root-feedback",
        revision_id="root",
        supersedes=None,
        horizon=horizon,
        feedback=panel,
        timing=timing,
    )


def _rebuild_pending(
    original,
    *,
    members=None,
    state_owner: StateOwner = "learner",
    decision_id=None,
    task=None,
    objective=None,
    implementation=None,
):
    request = original.proposal.request
    if members is not None:
        assert isinstance(request.space, MixtureSpace)
        space = MixtureSpace(feasible_set=request.space.feasible_set, members=members, profile=request.space.profile)
        request = DecisionRequest.model_validate({**request.model_dump(), "space": space})
    if decision_id is not None:
        request = DecisionRequest.model_validate({**request.model_dump(), "decision_id": decision_id})
    if task is not None or objective is not None:
        request = DecisionRequest.model_validate(
            request.model_dump()
            | {
                "task": request.task if task is None else task,
                "objective": request.objective if objective is None else objective,
            }
        )
    assert isinstance(request.space, MixtureSpace)
    assert isinstance(original.action, MixtureAction)
    action = MixtureAction(
        schema_version=1,
        space=request.space,
        weights=tuple(
            Weight(member=member, value=weight.value)
            for member, weight in zip(request.space.members, original.action.weights, strict=True)
        ),
    )
    sampling = SamplingRecord.model_validate(
        {**original.sampling.model_dump(), "action_fingerprint": action.fingerprint}
    )
    proposal = Decision(
        schema_version=1,
        request=request,
        implementation=original.proposal.implementation if implementation is None else implementation,
        action=action,
        sampling=sampling,
        state_owner=state_owner,
    )
    return ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=sampling,
    )


@pytest.mark.parametrize("change", ["member_id", "member_version", "state_owner"])
def test_incompatible_pending_restore_rejects_before_actual_kernel_callback(change):
    adapter, kernel = _bridge()
    adapter.decide(_cases(kernel.config)[0].request)
    before = adapter.export_state()
    members = kernel.members
    if change == "member_id":
        members = (_ref("expert-foreign"), members[1])
    elif change == "member_version":
        members = (Reference(id=members[0].id, version="future-v2"), members[1])
    pending = _rebuild_pending(
        before.pending[0], members=members, state_owner="host" if change == "state_owner" else "learner"
    )
    candidate = before.model_copy(update={"pending": (pending,)})
    with pytest.raises(ValueError, match="catalogue|state ownership|state_owner"):
        adapter.restore_state(candidate)
    assert kernel.restore_calls == 0
    assert adapter.export_state().to_json() == before.to_json()


def test_oversized_pending_restore_rejects_before_actual_kernel_callback():
    adapter, kernel = _bridge(maximum_bytes=16000)
    adapter.decide(_cases(kernel.config)[0].request)
    before = adapter.export_state()
    pending = _rebuild_pending(before.pending[0], decision_id="large-origin-" + "x" * 20000)
    candidate = before.model_copy(update={"pending": (pending,)})
    with pytest.raises(ValueError, match="byte capacity"):
        adapter.restore_state(candidate)
    assert kernel.restore_calls == 0
    assert adapter.export_state().to_json() == before.to_json()


@pytest.mark.parametrize("kind", ["str", "bytes", "bytearray"])
def test_direct_json_reader_bounds_depth_before_recursive_decode(kind):
    adapter, _ = _bridge()
    raw = "[" * 1200 + "0" + "]" * 1200
    payload = raw if kind == "str" else raw.encode() if kind == "bytes" else bytearray(raw.encode())
    reader = adapter.state_type
    with pytest.raises(ValueError, match="depth.*64"):
        reader.model_validate_json(payload)


def test_direct_json_reader_bounds_bytes_before_decode():
    adapter, _ = _bridge()
    payload = b'{"not-a-state":"' + b"x" * 16777216 + b'"}'
    reader = adapter.state_type
    with pytest.raises(ValueError, match="maximum_bytes"):
        reader.model_validate_json(payload)


def test_known_direct_json_bytearray_round_trip_preserves_original_state():
    adapter, kernel = _bridge()
    adapter.decide(_cases(kernel.config)[0].request)
    before = adapter.export_state()
    encoded = before.to_json()
    restored = adapter.state_type.model_validate_json(bytearray(encoded.encode()))
    assert restored.to_json() == encoded
    assert restored == before


def test_restored_pending_historical_weights_are_not_replaced_by_current_coefficients():
    adapter, kernel = _bridge()
    first, second = _cases(kernel.config)[:2]
    adapter.decide(first.request)
    adapter.decide(second.request)
    second_execution = adapter.export_state().pending[1]
    adapter.update(_batch(second_execution, kernel.config, (1.0, 0.0)))
    snapshot = adapter.export_state()
    original = snapshot.pending[0]
    assert isinstance(original.action, MixtureAction)
    assert original.action.weights != kernel.weights
    restored, restored_kernel = _bridge()
    restored.restore_state(type(snapshot).from_json(snapshot.to_json()))
    assert restored_kernel.restore_calls == 1
    assert restored.export_state().pending[0].to_json() == original.to_json()
    assert restored.export_state().to_json() == snapshot.to_json()
    restored.update(_batch(original, restored_kernel.config, (0.0, 1.0)))
    assert restored_kernel.update_count == 2
    assert not restored.export_state().pending


def _timed_pending():
    horizon = _ref("constructed-target")
    adapter, kernel = _bridge(horizon=horizon)
    start = datetime(2026, 11, 7, tzinfo=UTC)
    timing = DecisionTiming(
        decision_at=start,
        information_cutoff=start,
        target_start=start + timedelta(seconds=1),
        target_end=start + timedelta(seconds=2),
        eligible_input_fingerprint="sha256:" + "b" * 64,
        mode="future",
    )
    request = DecisionRequest.model_validate({**_cases(kernel.config)[0].request.model_dump(), "timing": timing})
    adapter.decide(request)
    observations = FeedbackTiming(
        event_at=start + timedelta(seconds=2),
        available_at=start + timedelta(seconds=3),
        mature_at=start + timedelta(seconds=4),
        received_at=start + timedelta(seconds=5),
    )
    batch = _batch(_execution(adapter), kernel.config, horizon=horizon, timing=observations)
    return adapter, kernel, batch, start


def test_actual_timed_feedback_uses_supplied_application_clock():
    adapter, kernel, batch, start = _timed_pending()
    result = adapter.update(batch, eligible_at=start + timedelta(seconds=5), apply_at=start + timedelta(seconds=6))
    assert isinstance(result, LearningUpdate)
    assert result.status == "accepted"
    assert kernel.update_count == 1
    assert not adapter.export_state().pending


def test_early_timed_feedback_rejects_without_kernel_update():
    adapter, kernel, batch, start = _timed_pending()
    before = adapter.export_state().to_json()
    too_early = start + timedelta(seconds=3)
    with pytest.raises(ValueError, match="before known"):
        adapter.update(batch, apply_at=too_early)
    assert kernel.update_count == 0
    assert adapter.export_state().to_json() == before


def test_generic_counter_mismatch_after_trusted_restore_invalidates_without_claiming_rollback():
    adapter, kernel = _bridge()
    before = adapter.export_state()
    other = Hedge(kernel.config, owner=kernel.binding.owner)
    other.observe(FullPanel(rows=_cases(kernel.config)[0].outcomes))
    candidate = before.model_copy(update={"method": other.export_state()})
    with pytest.raises(ValueError, match="update clocks disagree"):
        adapter.restore_state(candidate)
    assert kernel.restore_calls == 1
    assert kernel.update_count == 1
    with pytest.raises(LearningApplicationError, match="invalidated"):
        adapter.export_state()
    adapter.restore_state(before)
    assert kernel.restore_calls == 2
    assert kernel.update_count == 0
    assert adapter.export_state().to_json() == before.to_json()


@pytest.mark.parametrize("alteration", ["future_version", "missing_version", "unknown_field", "duplicate_key"])
def test_direct_json_route_preserves_strict_version_and_field_policy(alteration):
    adapter, _ = _bridge()
    document = json.loads(adapter.export_state().to_json())
    if alteration == "future_version":
        document["schema_version"] = 2
    elif alteration == "missing_version":
        document.pop("schema_version")
    elif alteration == "unknown_field":
        document["future_extension"] = 1
    payload = json.dumps(document)
    if alteration == "duplicate_key":
        payload = payload[:-1] + ',"schema_version":1}'
    reader = adapter.state_type
    with pytest.raises(ValueError, match="schema_version|extra|Duplicate JSON key"):
        reader.model_validate_json(payload)


def test_direct_json_reader_does_not_call_foreign_bytearray_conversion():
    class ForeignBytearray(bytearray):
        def __bytes__(self):
            raise AssertionError("A foreign conversion callback must not execute")

    adapter, _ = _bridge()
    reader, payload = adapter.state_type, ForeignBytearray(b"{}")
    with pytest.raises(TypeError, match="built-in"):
        reader.model_validate_json(payload)


@pytest.mark.parametrize("field", ["action", "sampling"])
def test_core_unchanged_execution_equality_is_revalidated_before_kernel_restore(field):
    adapter, kernel = _bridge()
    adapter.decide(_cases(kernel.config)[0].request)
    before = adapter.export_state()
    original = before.pending[0]
    if field == "action":
        assert isinstance(original.action, MixtureAction)
        changed = MixtureAction(
            schema_version=1,
            space=original.action.space,
            weights=(Weight(member=kernel.members[0], value=0.75), Weight(member=kernel.members[1], value=0.25)),
        )
    else:
        changed = original.sampling.model_copy(update={"mechanism_ref": _ref("another-mechanism")})
    pending = original.model_copy(update={field: changed})
    candidate = before.model_copy(update={"pending": (pending,)})
    with pytest.raises(ValueError, match="Preserved sampling"):
        adapter.restore_state(candidate)
    assert kernel.restore_calls == 0
    assert adapter.export_state().to_json() == before.to_json()


def test_pending_request_retry_and_completed_feedback_duplicate_are_distinct():
    adapter, kernel = _bridge()
    request = _cases(kernel.config)[0].request
    proposal = adapter.decide(request)
    before = adapter.export_state().to_json()
    assert adapter.decide(request).to_json() == proposal.to_json()
    assert adapter.export_state().to_json() == before
    batch = _batch(_execution(adapter), kernel.config)
    adapter.update(batch)
    accepted = adapter.export_state().to_json()
    with pytest.raises(ValueError, match="original pending producer"):
        adapter.update(batch)
    assert kernel.update_count == 1
    assert adapter.export_state().to_json() == accepted


@pytest.mark.parametrize(
    "alteration",
    [
        "binding",
        "decisions",
        "hard_clock",
        "pending_limit",
        "duplicate_pending",
        "task",
        "objective",
        "implementation",
        "mapping",
    ],
)
def test_direct_state_rejects_well_formed_nested_envelope_contradictions_before_restore(alteration):
    adapter, kernel = _bridge()
    adapter.decide(_cases(kernel.config)[0].request)
    before = adapter.export_state()
    if alteration == "binding":
        candidate = before.model_copy(
            update={"binding": before.binding.model_copy(update={"configuration_fingerprint": "sha256:" + "f" * 64})}
        )
    elif alteration == "decisions":
        candidate = before.model_copy(update={"decision_count": 0})
    elif alteration == "hard_clock":
        candidate = before.model_copy(update={"initial_updates": 1048576, "update_count": 1, "decision_count": 2})
    elif alteration == "pending_limit":
        restricted_kernel = _CountingHedge(_config(), owner=_ref("direct-state-owner"))
        # Selected options are part of the full binding; obtain them from an
        # actual independently initialized owner, rather than forge that hash.
        restricted = DirectMixture(restricted_kernel, task=TASK, objective=OBJECTIVE, maximum_pending=1)
        restricted.decide(_cases(restricted_kernel.config)[0].request)
        before = restricted.export_state()
        adapter, kernel = restricted, restricted_kernel
        other = _rebuild_pending(before.pending[0], decision_id="another-build")
        candidate = before.model_copy(update={"pending": (before.pending[0], other), "decision_count": 2})
    elif alteration == "duplicate_pending":
        candidate = before.model_copy(update={"pending": (before.pending[0], before.pending[0]), "decision_count": 2})
    else:
        pending = _rebuild_pending(
            before.pending[0],
            task=_ref("foreign-task") if alteration == "task" else None,
            objective=_ref("foreign-objective") if alteration == "objective" else None,
            implementation=_ref("foreign-method") if alteration == "implementation" else None,
        )
        if alteration == "mapping":
            pending = ExecutedDecision.model_validate(pending.model_dump() | {"mapping": "overridden"})
        candidate = before.model_copy(update={"pending": (pending,)})
    with pytest.raises(ValueError, match="binding|completed|clock|capacity|duplicate|foreign|implementation|unchanged"):
        adapter.restore_state(candidate)
    assert kernel.restore_calls == 0
    assert adapter.export_state().to_json() == before.to_json()


@pytest.mark.parametrize("change", ["task", "objective", "member", "space_kind"])
def test_wrong_direct_opportunity_rejects_before_any_mathematical_update(change):
    adapter, kernel = _bridge()
    original = _cases(kernel.config)[0].request
    if change in {"task", "objective"}:
        request = DecisionRequest.model_validate(original.model_dump() | {change: _ref("foreign")})
    elif change == "member":
        space = MixtureSpace(
            feasible_set=original.space.feasible_set,
            members=(Reference(id=kernel.members[0].id, version="future"), kernel.members[1]),
            profile="binary64-simplex-v1",
        )
        request = DecisionRequest.model_validate(original.model_dump() | {"space": space})
    else:
        space = ChoiceSpace(feasible_set=original.space.feasible_set, members=kernel.members)
        request = DecisionRequest.model_validate(original.model_dump() | {"space": space})
    before = adapter.export_state().to_json()
    with pytest.raises(ValueError, match="task/objective|catalogue"):
        adapter.decide(request)
    assert kernel.update_count == 0
    assert adapter.export_state().to_json() == before


def test_pending_conflict_and_capacity_reject_before_another_decision():
    kernel = _CountingHedge(_config(), owner=_ref("direct-state-owner"))
    adapter = DirectMixture(kernel, task=TASK, objective=OBJECTIVE, maximum_pending=1)
    first, second = _cases(kernel.config)[:2]
    adapter.decide(first.request)
    before = adapter.export_state().to_json()
    conflicting = DecisionRequest.model_validate(first.request.model_dump() | {"round": 1})
    with pytest.raises(ValueError, match="Conflicting direct request"):
        adapter.decide(conflicting)
    with pytest.raises(ValueError, match="producer capacity"):
        adapter.decide(second.request)
    assert adapter.export_state().to_json() == before and kernel.update_count == 0


@pytest.mark.parametrize("change", ["horizon", "transform"])
def test_wrong_feedback_interpretation_rejects_without_consuming_original_producer(change):
    adapter, kernel = _bridge()
    adapter.decide(_cases(kernel.config)[0].request)
    batch = _batch(_execution(adapter), kernel.config)
    assert isinstance(batch.feedback, FullPanel)
    if change == "horizon":
        candidate = FeedbackBatch.model_validate(batch.model_dump() | {"horizon": _ref("another-target")})
    else:
        rows = tuple(
            FeedbackRow(
                member=row.member,
                values=tuple(value.model_copy(update={"transform": _ref("another-transform")}) for value in row.values),
            )
            for row in batch.feedback.rows
        )
        candidate = FeedbackBatch.model_validate(batch.model_dump() | {"feedback": FullPanel(rows=rows)})
    before = adapter.export_state().to_json()
    with pytest.raises(ValueError, match="horizon|transform"):
        adapter.update(candidate)
    assert kernel.update_count == 0 and adapter.export_state().to_json() == before


def test_foreign_direct_binding_rejects_before_trusted_mathematical_restore():
    adapter, kernel = _bridge()
    other = DirectMixture(
        Hedge(kernel.config, owner=kernel.binding.owner), task=_ref("other-task"), objective=OBJECTIVE
    )
    candidate = other.export_state()
    before = adapter.export_state().to_json()
    with pytest.raises(ValueError, match="owner/options binding"):
        adapter.restore_state(candidate)
    assert kernel.restore_calls == 0 and adapter.export_state().to_json() == before


def test_real_underflow_update_invalidates_bridge_without_acknowledging_or_rewriting_panel():
    configuration = ReferenceConfig.model_validate(_config().model_dump() | {"eta": 1000.0})
    kernel = Hedge(configuration, owner=_ref("underflow-direct-owner"))
    adapter = DirectMixture(kernel, task=TASK, objective=OBJECTIVE)
    adapter.decide(_cases(kernel.config)[0].request)
    before = adapter.export_state()
    batch = _batch(_execution(adapter), kernel.config, (0.0, 1.0))
    with pytest.raises(LearningApplicationError, match="did not accept"):
        adapter.update(batch)
    assert kernel.update_count == 0 and kernel.export_state().to_json() == before.method.to_json()
    with pytest.raises(LearningApplicationError, match="invalidated"):
        adapter.export_state()
    adapter.restore_state(before)
    assert adapter.export_state().to_json() == before.to_json()


def test_external_real_kernel_update_invalidates_the_exclusive_direct_owner():
    adapter, kernel = _bridge()
    original = adapter.export_state()
    kernel.observe(FullPanel(rows=_cases(kernel.config)[0].outcomes))
    with pytest.raises(LearningApplicationError, match="outside this bridge"):
        adapter.export_state()
    adapter.restore_state(original)
    assert adapter.export_state().to_json() == original.to_json()
