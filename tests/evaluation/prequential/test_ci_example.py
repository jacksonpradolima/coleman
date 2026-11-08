"""Real one-cell recovery and declaration invariants for the public CI example.

The complete nine-mode/four-world flow is qualified separately through the
noneditable installed package; these tests do not duplicate that expensive run.
"""

import importlib.util
import json
import math
import shutil
import struct
import sys
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.state.components import StagedCollection


@pytest.fixture(scope="module")
def ci_example():
    name = "authored_prequential_ci_source_test"
    path = Path(__file__).parents[3] / "examples/prequential/ci.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def actual_cell(ci_example, tmp_path_factory):
    example = ci_example
    root = tmp_path_factory.mktemp("actual-prequential-ci-cell")
    frozen, _ = example.frozen_declarations(root)
    mode, world = "online", "world-0"
    recipe = example.MODEL.Recipe(
        mode,
        example.MODEL.configuration(mode),
        example.MODEL.ControlSchedule(schema_version=1, calendar=None),
        frozen.design.candidate(mode).candidate.configuration_fingerprint,
        frozen.design.snapshot.data_fingerprint,
        world,
    )
    owner = example.Session(root / "original", recipe, frozen.environment, create=True)
    originals = []
    for ordinal in range(3):
        at = datetime.now(UTC)
        request = owner.command("emit", ordinal, at)
        emitted = owner.emit(ordinal, at, request=request)
        before = owner.store.read_head()
        assert owner.emit(ordinal, at, request=request).to_json() == emitted.to_json()
        assert owner.store.read_head() == before
        original = owner.prepare_observation(
            ordinal, example.TABLE[0][ordinal], example.project_frozen(frozen.design, frozen.selection).fingerprint
        )
        request = owner.observation_command(original)
        owner.apply_original(original, request=request)
        before = owner.store.read_head()
        assert owner.apply_original(original, request=request).to_json() == original.step.to_json()
        assert owner.store.read_head() == before
        originals.append(original)
    return root, frozen, recipe, owner, tuple(originals)


def test_declared_complete_family_units_and_matched_controls(ci_example):
    example = ci_example
    design, selection, plan = example.declarations()
    assert len(design.candidates) == 9
    assert len(design.protected_members) == len({row.unit_id for row in design.protected_members}) == 4
    assert design.raw_cells == 36
    assert design.seeds == (0,)
    assert len(design.hypotheses) == 8
    assert len(plan.conformance_plan.opportunities) == 12
    assert len(plan.cell_result_opportunities) == 4
    assert len(design.metrics) == 5
    assert design.scope.prehistory == "known_exposed"
    assert design.evidence_basis == "constructed_mechanism"
    assert selection.rule == plan.binding
    plan.validate_for(design, selection)
    assert design.limits.reserve(design.raw_cells, design.aggregate_inputs) > 0
    controls = {mode: example.MODEL.configuration(mode) for mode in example.MODEL.MODES}
    assert controls["fixed"].eta == 0.0
    assert controls["cadence"].controls.cadence.panels == controls["penalized"].controls.cadence.panels == 2
    assert controls["penalized"].controls.output.strength == controls["online"].controls.output.strength == 0.05
    assert controls["accounting_off"].controls.output.strength == 0.05
    assert design.candidate("penalized").strategy.accounting != design.candidate("accounting_off").strategy.accounting


def test_frozen_header_reuses_original_selection_before_any_source_read(ci_example, tmp_path):
    first, resumed = ci_example.frozen_declarations(tmp_path)
    assert resumed is False
    assert not (tmp_path / "audit").exists()
    again, resumed = ci_example.frozen_declarations(tmp_path)
    assert resumed is True
    assert first.to_json() == again.to_json()
    assert first.selection.actual_selected_at == again.selection.actual_selected_at
    assert not (tmp_path / "audit").exists()


@pytest.mark.parametrize("changed", ["source", "configuration", "code_binding"])
def test_header_only_drift_refuses_before_access_or_stores(ci_example, tmp_path, monkeypatch, changed):
    example = ci_example
    frozen, _ = example.frozen_declarations(tmp_path)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    if changed == "source":
        table = (((3.0, 6.0), *example.TABLE[0][1:]), *example.TABLE[1:])
        monkeypatch.setattr(example.PROFILE, "TABLE", table)
    elif changed == "configuration":
        current = example.MODEL.configuration

        def wrong(mode):
            config = current(mode)
            return type(config).model_validate(config.model_copy(update={"eta": 0.25})) if mode == "fixed" else config

        monkeypatch.setattr(example.MODEL, "configuration", wrong)
    else:
        current = example.compatibility()
        original = current.owner_implementations[0]
        wrong = type(original).model_validate(
            original.model_copy(update={"definition_fingerprint": "sha256:" + "0" * 64})
        )
        altered = type(current).model_validate(
            current.model_copy(update={"owner_implementations": (wrong, *current.owner_implementations[1:])})
        )
        monkeypatch.setattr(example.PROFILE, "compatibility", lambda: altered)
    with pytest.raises(
        ValueError, match="Current candidate laws or source interpretation|Current runtime or copied example code"
    ):
        example.run(tmp_path)
    assert not (tmp_path / "audit").exists()
    assert not (tmp_path / "sessions").exists()
    assert before == {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert frozen.selection.actual_selected_at is not None


def test_resume_output_budget_refuses_before_new_invocation_or_owner_creation(ci_example, tmp_path):
    example = ci_example
    frozen, _ = example.frozen_declarations(tmp_path)
    for ordinal in range(3):
        invocation = example.Invocation(
            schema_version=1, attempt_id="completed-attempt-" + str(ordinal), started_at=datetime.now(UTC)
        )
        example._write(tmp_path, "work/invocation-" + invocation.attempt_id + ".json", invocation)
        example._write(
            tmp_path, "work/completed-" + invocation.attempt_id + ".json", example._attempt_ledger(invocation, ())
        )
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="output references exceed the frozen global budget"):
        example._produce(tmp_path, frozen, [], [], mode="imported")
    assert not (tmp_path / "sessions").exists()
    assert before == {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }


def test_real_original_tape_round_trip_and_complete_two_owner_reproduction(ci_example, actual_cell):
    root, frozen, recipe, owner, originals = actual_cell
    before = owner.snapshots()
    assert (before[0].state.decision_count, before[0].state.update_count, before[1].count) == (3, 3, 3)
    assert before[1].expert_history == ci_example.TABLE[0]
    assert all(len(row.step.components) == 5 for row in originals)
    restored = tuple(ci_example.ObservationInput.from_json(row.to_json()) for row in originals)
    assert tuple(row.to_json() for row in restored) == tuple(row.to_json() for row in originals)
    reopened = ci_example.Session(root / "original", recipe, frozen.environment)
    assert tuple(row.to_json() for row in reopened.snapshots()) == tuple(row.to_json() for row in before)
    replica = ci_example.Session(root / "reproduced", recipe, frozen.environment, create=True)
    after = replica.reproduce(restored)
    assert tuple(row.to_json() for row in after) == tuple(row.to_json() for row in before)
    terminal = ci_example.RawCell(
        candidate_id=recipe.mode,
        member_id=recipe.world,
        seed=0,
        state="completed",
        evidence=ci_example._primary(originals[-1].step).original,
    )
    assert terminal.record.decision_id == recipe.world


def test_online_forecasts_use_only_prior_applied_full_panels(ci_example, actual_cell):
    originals = actual_cell[-1]
    predictions = [
        tuple(row.values[0].value for row in original.step.receipt.executed.proposal.request.member_inputs)
        for original in originals
    ]
    assert predictions == [(4.0, 4.0), (2.0, 6.0), (2.0, 6.0)]
    # The terminal reversal (6,2) is received only AFTER the third decision.
    assert originals[-1].expert_seconds == (6.0, 2.0)
    assert all(row.metric.quantity.spec.semantics.unit in {"s", "1"} for row in originals[-1].step.components)
    assert all(
        row.metric.role != "physical_cost" and row.physical_group_id is None for row in originals[-1].step.components
    )
    trace = ci_example.ConformanceTrace(
        schema_version=1, recipe=ci_example.MODEL.ADAPTER, steps=tuple(row.step for row in originals)
    )
    diagnostics = ci_example._trajectory_diagnostics((trace,))
    terminal = diagnostics["terminal_by_candidate"]["online"][0]
    assert terminal["world"] == "world-0"
    assert terminal["round"] == 2
    assert len(terminal["measurements"]) == 5
    trajectory = diagnostics["complete_trajectories"]["online"][0]
    assert len(trajectory["decisions"]) == 3
    assert "not adoptable or executed" in diagnostics["stage_diagnostic_scope"]


def test_component_producer_digest_is_reused_without_changing_original_bytes(ci_example, actual_cell, monkeypatch):
    owner, original = actual_cell[-2], actual_cell[-1][0]
    primary = ci_example._primary(original.step)
    receipt = original.step.receipt
    expected = {row.metric.name: row.to_json() for row in original.step.components}
    descriptor = type(receipt).fingerprint
    reads = []

    def counted(value):
        if value is receipt:
            reads.append(value)
        return descriptor.__get__(value, type(value))

    monkeypatch.setattr(type(receipt), "fingerprint", property(counted))
    session = sys.modules[type(owner).__module__]
    actual = session._observation_components(primary.original.record, receipt, primary.timing, primary.applied_at)
    assert len(reads) == 1
    assert {row.metric.name: row.to_json() for row in actual} == expected
    reads.clear()
    monkeypatch.setattr(ci_example.MODEL, "metrics", lambda: ())
    assert session._observation_components(primary.original.record, receipt, primary.timing, primary.applied_at) == ()
    assert reads == []


@pytest.mark.parametrize("selected", [True, False], ids=["three-retained-updates", "no-eligible-update"])
def test_original_feedback_digest_is_reused_after_actual_replay_selection(
    ci_example, actual_cell, tmp_path, monkeypatch, selected
):
    _, frozen, recipe, owner, originals = actual_cell
    original = originals[-1] if selected else originals[0]
    if not selected:
        owner = ci_example.Session(tmp_path / "pending-cell", recipe, frozen.environment, create=True)
        owner.emit(0, ci_example._timing(original.coefficient.opportunity.request).decision_at)
        primary = ci_example._primary(original.step)
        timing = type(original.feedback.batch.timing).model_validate(
            original.feedback.batch.timing.model_copy(update={"received_at": primary.applied_at + timedelta(seconds=1)})
        )
        batch = type(original.feedback.batch).model_validate(
            original.feedback.batch.model_copy(update={"timing": timing})
        )
        feedback = type(original.feedback).model_validate(original.feedback.model_copy(update={"batch": batch}))
        # The public ingress already rejects this mismatch. Exercise the real
        # internal selection guard too, without replacing replay or a state owner.
        original = original.model_copy(update={"feedback": feedback})
    replay_state, domain_state = owner.snapshots()
    replay, domain = ci_example.MODEL.ReplayOwner(recipe), ci_example.MODEL.DomainOwner(recipe)
    replay.restore_state(replay_state)
    domain.restore_state(domain_state)
    before = domain.export_state().to_json()
    before_head = owner.store.read_head()
    descriptor = type(original.feedback).fingerprint
    reads = []

    def counted(value):
        if value is original.feedback:
            reads.append(value)
        return descriptor.__get__(value, type(value))

    monkeypatch.setattr(type(original.feedback), "fingerprint", property(counted))
    stage = StagedCollection(((owner.replay_binding, replay), (owner.domain_binding, domain)))
    primary = ci_example._primary(original.step)
    try:
        if selected:
            owner._apply_in_stage(stage, original, primary)
            assert len(replay.session.state.updates) == 3
            assert len(reads) == 1
            assert replay.export_state().to_json() == replay_state.to_json()
        else:
            with pytest.raises(ValueError, match="actual selected replay application"):
                owner._apply_in_stage(stage, original, primary)
            assert replay.session.state.updates == ()
            assert reads == []
        assert domain.export_state().to_json() == before
        assert owner.store.read_head() == before_head
    finally:
        stage._close()


def test_raw_identity_scopes_distinct_evaluations_and_preserves_original_reproduction(ci_example, actual_cell):
    owner, original = actual_cell[-2], actual_cell[-1][0]
    primary = ci_example._primary(original.step)
    first = owner.prepare_observation(
        0,
        original.expert_seconds,
        primary.original.record.provenance.evaluation_fingerprint,
        available_at=primary.timing.available_at,
        apply_at=primary.applied_at,
    )
    second = owner.prepare_observation(
        0,
        original.expert_seconds,
        "sha256:" + "7" * 64,
        available_at=primary.timing.available_at,
        apply_at=primary.applied_at,
    )
    assert ci_example._primary(first.step).original.record_key == ci_example._primary(original.step).original.record_key
    assert ci_example._primary(first.step).original.model_dump_json() == primary.original.model_dump_json()
    assert ci_example._primary(second.step).original.record_key != ci_example._primary(first.step).original.record_key


def _changed_unchosen(example, original, recipe):
    changed = (original.expert_seconds[0], original.expert_seconds[1] - 1.0)
    feedback = example.MODEL.feedback(
        original.coefficient, recipe.config, changed, original.feedback.batch.timing.available_at
    )
    return example.ObservationInput(
        schema_version=1,
        step=original.step,
        coefficient=original.coefficient,
        feedback=feedback,
        expert_seconds=changed,
    )


@pytest.mark.parametrize("reopen", [False, True])
def test_full_feedback_conflict_rejects_live_and_rebuilt_retry_cache(ci_example, actual_cell, reopen):
    root, frozen, recipe, owner, originals = actual_cell
    owner = ci_example.Session(root / "original", recipe, frozen.environment) if reopen else owner
    changed = _changed_unchosen(ci_example, originals[0], recipe)
    assert changed.step.to_json() == originals[0].step.to_json()
    before = owner.store.read_head()
    with pytest.raises(ValueError, match="complete retained original payload"):
        owner.apply_original(changed)
    assert owner.store.read_head() == before


def test_feedback_only_clock_drift_cannot_update_domain_ahead_of_replay(ci_example, actual_cell):
    owner, original = actual_cell[-2], actual_cell[-1][0]
    primary = ci_example._primary(original.step)
    timing = type(original.feedback.batch.timing).model_validate(
        original.feedback.batch.timing.model_copy(update={"received_at": primary.applied_at + timedelta(seconds=1)})
    )
    batch = type(original.feedback.batch).model_validate(original.feedback.batch.model_copy(update={"timing": timing}))
    feedback = type(original.feedback).model_validate(original.feedback.model_copy(update={"batch": batch}))
    unchecked = original.model_copy(update={"feedback": feedback})
    before = owner.store.read_head()
    with pytest.raises(ValueError, match="same original source processing clocks"):
        owner.apply_original(unchecked)
    assert owner.store.read_head() == before


def _shift_application(step):
    return type(step).model_validate(
        step.model_copy(
            update={
                "components": tuple(
                    type(component).model_validate(
                        component.model_copy(update={"applied_at": component.applied_at + timedelta(microseconds=1)})
                    )
                    for component in step.components
                )
            }
        )
    )


def test_reopened_domain_clock_must_match_actual_selected_replay_application(ci_example, actual_cell, tmp_path):
    example = ci_example
    root, frozen, recipe, _, _ = actual_cell
    shutil.copytree(root / "original", tmp_path / "copied-store")
    owner = example.Session(tmp_path / "copied-store", recipe, frozen.environment)
    _, state = owner.snapshots()
    changed = type(state).model_validate(
        state.model_copy(update={"observations": (_shift_application(state.observations[0]), *state.observations[1:])})
    )

    def inject_individually_valid_state(stage):
        stage.owner(owner.domain_binding).restore_state(changed)

    # Existing storage validates each complete owner; this example must also
    # validate the actual application join between its independent owners.
    owner.store.commit_transition(
        owner.command("emit", 0, datetime.now(UTC)),
        owner.bindings,
        inject_individually_valid_state,
        compatibility=frozen.environment,
    )
    before = owner.store.read_head()
    with pytest.raises(ValueError, match="actual selected replay application"):
        example.Session(tmp_path / "copied-store", recipe, frozen.environment)
    assert owner.store.read_head() == before


@pytest.mark.parametrize("fault", ["application_clock", "raw_expert_panel"])
def test_cross_owner_payload_fault_refuses_before_joint_ack(ci_example, actual_cell, tmp_path, monkeypatch, fault):
    example = ci_example
    _, frozen, recipe, _, originals = actual_cell
    original = originals[0]
    owner = example.Session(tmp_path / "actual-staged-store", recipe, frozen.environment, create=True)
    owner.emit(0, example._timing(original.coefficient.opportunity.request).decision_at)
    real_observe = example.MODEL.DomainOwner.observe

    def observe_with_retention_fault(domain, step, seconds):
        state = real_observe(domain, step, seconds)
        if fault == "application_clock":
            changed = state.model_copy(update={"observations": (_shift_application(state.observations[0]),)})
        else:
            panel = (seconds[0], seconds[1] - 1.0)
            changed = state.model_copy(update={"expert_history": (panel,), "sums": panel})
        domain.restore_state(type(state).model_validate(changed))
        return domain.export_state()

    monkeypatch.setattr(example.MODEL.DomainOwner, "observe", observe_with_retention_fault)
    before = owner.store.read_head()
    with pytest.raises(ValueError, match="exact original applied evidence|actual original full panel"):
        owner.apply_original(original)
    assert owner.store.read_head() == before


def test_missing_components_and_future_input_versions_refuse(ci_example, actual_cell):
    original = actual_cell[-1][0]
    empty = type(original.step).model_validate(original.step.model_copy(update={"components": ()}))
    incomplete = original.model_copy(update={"step": empty})
    with pytest.raises(ValueError, match="every explicit domain component"):
        ci_example.ObservationInput.model_validate(incomplete)
    data = json.loads(original.to_json())
    data["schema_version"] = 2
    future = json.dumps(data)
    with pytest.raises(ValueError):
        ci_example.ObservationInput.from_json(future)


def test_exact_zero_recognition_matches_independent_finite_rational_oracle(ci_example):
    smallest = math.ulp(0.0)
    values = [
        0.0,
        -0.0,
        smallest,
        -smallest,
        4 * smallest,
        5 * smallest,
        8 * smallest,
        sys.float_info.min,
        math.nextafter(sys.float_info.min, 0.0),
        0.1,
        1.0,
        8.0,
        sys.float_info.max,
    ]
    for value in values:
        assert ci_example.PROFILE._is_zero(value) is (Fraction(value).numerator == 0)
        scaled = value / 8.0
        assert ci_example.PROFILE._is_zero(scaled) is (Fraction(scaled).numerator == 0)


@pytest.mark.parametrize(
    "duration,rejected",
    [
        pytest.param(0.0, False, id="positive-zero"),
        pytest.param(-0.0, False, id="negative-zero"),
        pytest.param(math.ulp(0.0), True, id="smallest-subnormal-underflows"),
        pytest.param(4 * math.ulp(0.0), True, id="half-ulp-tie-underflows"),
        pytest.param(5 * math.ulp(0.0), False, id="surviving-subnormal"),
        pytest.param(8 * math.ulp(0.0), False, id="scaled-subnormal-exact"),
    ],
)
def test_complete_original_normalization_preserves_zero_and_rejects_nonzero_erasure(
    ci_example, actual_cell, duration, rejected
):
    owner, original = actual_cell[-2], actual_cell[-1][0]
    primary = ci_example._primary(original.step)
    panel = (original.expert_seconds[0], duration)
    evaluation = primary.original.record.provenance.evaluation_fingerprint
    visible, applied = primary.timing.available_at, primary.applied_at
    before = owner.store.read_head()
    if rejected:
        with pytest.raises(ValueError, match="cannot silently erase a nonzero expert duration"):
            owner.prepare_observation(0, panel, evaluation, available_at=visible, apply_at=applied)
    else:
        accepted = owner.prepare_observation(0, panel, evaluation, available_at=visible, apply_at=applied)
        assert struct.pack("!d", accepted.expert_seconds[1]) == struct.pack("!d", duration)
        assert struct.pack("!d", accepted.feedback.batch.feedback.rows[1].values[0].value) == struct.pack(
            "!d", duration / 8.0
        )
    assert owner.store.read_head() == before


def test_independent_resource_bound_oracle_rejects_equal_action_and_outcome_drift(ci_example, actual_cell):
    original = actual_cell[-1][0]
    plan = ci_example.ReferenceConformancePlan(
        schema_version=1,
        reference=ci_example.MODEL.ref("one-real-opportunity"),
        lineage=ci_example.MODEL.lineage(),
        adapter=ci_example.MODEL.ADAPTER,
        reference_recipe=ci_example.REFERENCE_RECIPE,
        baseline_candidate_id=original.step.receipt.key.cell.candidate_id,
        base_configuration_fingerprint=original.step.receipt.base_configuration_fingerprint,
        data_fingerprint=original.step.receipt.data_fingerprint,
        opportunities=(original.step.receipt.key,),
        metrics=ci_example.MODEL.metrics(),
    )
    expected = ci_example.reference_from_originals((original,))
    receipt = original.step.receipt
    constraint = receipt.feasibility.request.constraints[0]
    altered_constraint = type(constraint).model_validate(constraint.model_copy(update={"upper": 3.0}))
    altered_request = type(receipt.feasibility.request).model_validate(
        receipt.feasibility.request.model_copy(update={"constraints": (altered_constraint,)})
    )
    wrong = ci_example.MODEL.check_finite_feasibility(altered_request)
    assert wrong.feasible_members == receipt.feasibility.feasible_members
    altered_receipt = type(receipt).model_validate(receipt.model_copy(update={"feasibility": wrong}))
    components = tuple(
        type(row).model_validate(row.model_copy(update={"execution_fingerprint": altered_receipt.fingerprint}))
        for row in original.step.components
    )
    altered = type(original.step)(schema_version=1, receipt=altered_receipt, components=components)
    actual = ci_example.ConformanceTrace(schema_version=1, recipe=ci_example.MODEL.ADAPTER, steps=(altered,))
    report = ci_example.check_reference_conformance(plan, expected, actual)
    assert report.status == "fail"
    assert altered.receipt.executed.to_json() == original.step.receipt.executed.to_json()
    assert ci_example._primary(altered).value.value == ci_example._primary(original.step).value.value
    assert any(row.name == "feasibility" and row.status == "fail" for row in report.checks)


def test_original_clock_oracle_does_not_copy_actual_output_clock(ci_example, actual_cell):
    original = actual_cell[-1][0]
    plan = ci_example.ReferenceConformancePlan(
        schema_version=1,
        reference=ci_example.MODEL.ref("one-clock-opportunity"),
        lineage=ci_example.MODEL.lineage(),
        adapter=ci_example.MODEL.ADAPTER,
        reference_recipe=ci_example.REFERENCE_RECIPE,
        baseline_candidate_id=original.step.receipt.key.cell.candidate_id,
        base_configuration_fingerprint=original.step.receipt.base_configuration_fingerprint,
        data_fingerprint=original.step.receipt.data_fingerprint,
        opportunities=(original.step.receipt.key,),
        metrics=ci_example.MODEL.metrics(),
    )
    expected = ci_example.reference_from_originals((original,))
    primary = ci_example._primary(original.step)
    changed = type(primary).model_validate(
        primary.model_copy(update={"applied_at": primary.applied_at + timedelta(microseconds=1)})
    )
    components = tuple(changed if row.metric.name == primary.metric.name else row for row in original.step.components)
    step = type(original.step)(schema_version=1, receipt=original.step.receipt, components=components)
    actual = ci_example.ConformanceTrace(schema_version=1, recipe=ci_example.MODEL.ADAPTER, steps=(step,))
    report = ci_example.check_reference_conformance(plan, expected, actual)
    assert report.status == "fail"
    assert any(row.reason == "clock_drift" for row in report.checks)
