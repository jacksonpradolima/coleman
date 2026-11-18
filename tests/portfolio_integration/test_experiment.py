"""Real public authored online/controlled/constrained/funded composition."""

import json
from datetime import timedelta
from fractions import Fraction

import pytest

from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.experiment import Experiment, ExperimentState
from examples.synthetic_portfolio.reporting import publish


@pytest.fixture(scope="module")
def full_experiment():
    experiment = Experiment()
    for index in range(16):
        experiment.step(index)
    return experiment


def test_real_complete_causal_forecast_control_and_funded_flow(full_experiment):
    experiment = full_experiment
    assert experiment.ridge.export_state().payload.sample_count == 15
    assert len(experiment.periods) == 16
    for index, period in enumerate(experiment.periods):
        assert period.pending.prediction.sample_count == index
        assert period.pending.prediction.input.values[0] == 1.0
        for selection in period.pending.selections:
            assert selection.scale.snapshot.training_cutoff == model.START + timedelta(days=index)
            assert all(sample.observation != selection.outcome for sample in selection.scale.snapshot.samples)
        assert len(period.candidates) == len(model.MODES)
        for candidate in period.candidates:
            assert candidate.settlement.after.sequence == index + 1
            assert candidate.settlement.status == "executed"
            assert (
                candidate.settlement.net_value.fraction
                + candidate.settlement.proportional_paid.fraction
                + candidate.settlement.recurring_paid.fraction
                == candidate.settlement.gross_value.fraction
            )
    # Missing warm-up scales do not fabricate complete zero-loss panels.
    assert all(application.output.status == "unavailable" for application in experiment.observations[0].applications)
    assert experiment.references["hedge"].update_count == 13
    full = experiment.controlled["anchored-bounded-interval-l1"].export_state().payload
    assert full.config.controls.target is not None
    assert full.config.controls.movement is not None
    assert full.config.controls.output.strength == 0.05
    assert full.exact_path is not None
    assert full.release_count < full.update_count


def test_constant_coefficients_still_have_drift_recipe_and_executed_costs(full_experiment):
    period = full_experiment.periods[7]
    equal = next(row for row in period.candidates if row.name == "equal")
    hold = next(row for row in period.candidates if row.name == "buy-and-hold")
    plain = next(row for row in period.candidates if row.name == "plain")
    assert equal.coefficient_movement is None
    held_coefficients = next(row for row in full_experiment.periods[1].candidates if row.name == "plain")
    assert held_coefficients.coefficient_movement == 0
    assert held_coefficients.settlement.turnover.numerator > 0
    assert equal.settlement.turnover.numerator > 0
    assert equal.settlement.proportional_paid.numerator > 0
    assert hold.settlement.turnover.numerator == hold.settlement.proportional_paid.numerator == 0
    assert hold.settlement.recurring_paid.numerator > 0
    assert plain.settlement.before.reference != equal.settlement.before.reference
    # Virtual recipe modeled charges are retained and never added a second time
    # to any candidate's exact conservation identity.
    assert all(recipe.before.reference != equal.settlement.before.reference for recipe in period.recipes)


def test_penalty_objective_projection_is_independent_exact_arithmetic_not_a_fee(full_experiment):
    for period in full_experiment.periods:
        for candidate in period.candidates:
            control, objective = candidate.control, candidate.control_objective
            if control is None:
                assert objective is None
                continue
            if control.disposition != "released":
                assert objective.status == "not_applicable"
                assert objective.total is None
                continue
            proximity = (
                sum(
                    (
                        (Fraction(value.value) - Fraction(proposal.value)) ** 2
                        for value, proposal in zip(control.weights, control.proposal, strict=True)
                    ),
                    Fraction(),
                )
                / 2
            )
            movement = sum(
                (
                    abs(Fraction(value.value) - Fraction(old.value))
                    for value, old in zip(control.weights, control.origin.weights, strict=True)
                ),
                Fraction(),
            )
            assert objective.proximity.fraction == proximity
            assert objective.coefficient_l1.fraction == movement
            if control.output.kind == "l1_movement":
                expected = Fraction(control.output.strength) * movement
                assert objective.weighted_penalty.fraction == expected
                assert objective.total.fraction == proximity + expected
            else:
                assert objective.weighted_penalty is None
                assert objective.total.fraction == proximity
            # Paid charges remain exactly the independent funded-book fee law.
            assert (
                candidate.settlement.proportional_paid.fraction
                == candidate.settlement.turnover.fraction * candidate.settlement.fees.proportional.fraction
            )


def test_duplicate_period_and_complete_checkpoint_resume_are_identical(full_experiment):
    prefix = Experiment()
    for index in range(8):
        prefix.step(index)
    snapshot = ExperimentState.from_json(prefix.export_state().to_json())
    resumed = Experiment()
    resumed.restore_state(snapshot)
    before = resumed.export_state().fingerprint
    assert resumed.step(7).to_json() == prefix.step(7).to_json()
    assert resumed.export_state().fingerprint == before
    for index in range(8, 16):
        assert resumed.step(index).to_json() == full_experiment.periods[index].to_json()
    assert resumed.export_state().to_json() == full_experiment.export_state().to_json()


def test_future_price_changes_cannot_change_prefix_outputs():
    rows = model.PRICE_ROWS[:4] + tuple((value[0] + 20, value[1] + 30) for value in model.PRICE_ROWS[4:])
    original, changed = Experiment(), Experiment(rows=rows)
    for index in range(4):
        assert original.step(index).to_json() == changed.step(index).to_json()


def test_unknown_versions_and_incompatible_selected_law_refuse_atomically():
    experiment = Experiment()
    experiment.step(0)
    snapshot = experiment.export_state()
    future = json.loads(snapshot.to_json())
    future["schema_version"] = 2
    with pytest.raises(ValueError):
        ExperimentState.from_json(json.dumps(future))
    other = Experiment(strength=0.1)
    initial = other.export_state().fingerprint
    with pytest.raises(ValueError, match="another tape or selected law"):
        other.restore_state(snapshot)
    assert other.export_state().fingerprint == initial


@pytest.mark.parametrize("strength", [0.0, 0.05, 100.0])
def test_reader_preserves_each_supported_declared_strength(strength):
    original = Experiment(strength=strength).export_state()
    assert ExperimentState.from_json(original.to_json()).to_json() == original.to_json()


@pytest.mark.parametrize(
    "change", ["strength", "excess_strength", "controlled", "reference", "ridge", "configuration", "owner"]
)
def test_reader_rejects_relabelled_laws_and_logical_owners_before_atomic_restore(change):
    from coleman.learning.reference import Hedge
    from coleman.prediction import AppendOnlyRidge

    experiment = Experiment()
    original = experiment.export_state()
    wire = original.to_json()
    changed = json.loads(wire)
    if change in {"strength", "excess_strength"}:
        changed["strength"] = 0.1 if change == "strength" else 100.1
    elif change == "controlled":
        other = Experiment(strength=0.1).export_state()
        selected = next(index for index, item in enumerate(other.controlled) if item.name == "penalized")
        changed["controlled"][selected]["state"] = other.controlled[selected].state.model_dump(mode="json")
    elif change == "reference":
        configuration = model.reference_config("hedge-constant")
        altered = type(configuration).model_validate(configuration.model_dump() | {"eta": 0.25})
        state = Hedge(altered, owner=model.ref("hedge-owner")).export_state()
        selected = next(index for index, item in enumerate(original.references) if item.name == "hedge")
        changed["references"][selected]["state"] = state.model_dump(mode="json")
    elif change == "ridge":
        configuration = model.ridge_config()
        altered = type(configuration).model_validate(configuration.model_dump() | {"regularization": 2.0})
        state = AppendOnlyRidge(altered, owner=model.ref("ridge-owner")).export_state()
        changed["ridge"] = state.model_dump(mode="json")
    elif change == "configuration":
        changed["controlled"][0]["state"]["binding"]["configuration_fingerprint"] = "sha256:" + "0" * 64
    else:
        changed["controlled"][0]["state"]["binding"]["owner"]["id"] = "another-logical-owner"
    for read in (
        lambda: ExperimentState.model_validate(changed),
        lambda: ExperimentState.from_json(json.dumps(changed)),
        lambda: experiment.restore_state(ExperimentState.model_construct(**changed)),
    ):
        with pytest.raises(ValueError):
            read()
        assert experiment.export_state().to_json() == wire


def test_original_checkpoint_continues_identically_in_independent_thread_contexts():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    direct = Experiment()
    for index in range(4):
        direct.step(index)
    saved = ExperimentState.from_json(direct.export_state().to_json())
    assert saved.references[0].state.payload.update_count == 1
    direct.step(4)
    expected = direct.export_state().to_json()
    ready = Barrier(2)

    def continue_original(_):
        ready.wait(timeout=5)
        owned = Experiment()
        virgin = owned.export_state().to_json()
        for change in ("schema_version", "owner"):
            invalid = json.loads(saved.to_json())
            if change == "schema_version":
                invalid["schema_version"] = 2
            else:
                invalid["references"][0]["state"]["binding"]["owner"]["id"] = "another-logical-owner"
            with pytest.raises(ValueError):
                owned.restore_state(ExperimentState.model_construct(**invalid))
            assert owned.export_state().to_json() == virgin
        owned.restore_state(saved)
        owned.step(4)
        return owned.export_state().to_json()

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert tuple(workers.map(continue_original, (0, 1))) == (expected, expected)


def test_task_outer_snapshot_fully_validates_complete_private_capture():
    from coleman.core.primitives import fingerprint
    from coleman.objectives.bindings import DefinitionBinding
    from coleman.state.contracts import StateBinding
    from examples.synthetic_portfolio._state import PortfolioOwner
    from examples.synthetic_portfolio.flow_contracts import ExperimentStateV2
    from examples.synthetic_portfolio.task_contracts import PortfolioCheckpoint, PortfolioCheckpointV2

    original = Experiment().export_normalized_state()
    initial = PortfolioCheckpointV2(
        schema_version=2,
        binding=StateBinding(
            owner=model.ref("outer-validation-owner"),
            implementation=original.implementation,
            state_codec=model.ref("complete-task-checkpoint"),
            configuration_fingerprint=fingerprint({"strength": original.strength}),
        ),
        plan_fingerprint=fingerprint({"pure": "outer-validation-plan"}),
        coordinate_fingerprint=fingerprint({"pure": "outer-validation-coordinate"}),
        source=DefinitionBinding(reference=model.ref("tape"), definition_fingerprint=original.tape_fingerprint),
        rows=model.PRICE_ROWS,
        experiment=original,
    )
    owner = PortfolioOwner(initial)
    owner.experiment.step(0)
    expected = PortfolioCheckpointV2(
        **initial.model_dump(exclude={"experiment"}), experiment=owner.experiment.export_normalized_state()
    )
    actual = owner.export_state()
    assert actual.to_json() == expected.to_json()
    # Keep the legacy capture/reader regression as an explicit read envelope,
    # alongside the current task owner. Neither envelope below claims an ACK.
    legacy = PortfolioCheckpoint(
        **initial.model_dump(exclude={"experiment", "schema_version", "binding"}),
        schema_version=1,
        binding=initial.binding.model_copy(update={"state_codec": model.ref("complete-legacy-task-checkpoint")}),
        experiment=owner.experiment.export_state(),
    )
    profiles = (
        (PortfolioCheckpoint, ExperimentState, legacy, owner.experiment._capture_state(), 2),
        (PortfolioCheckpointV2, ExperimentStateV2, actual, owner.experiment._capture_normalized_state(), 3),
    )
    for checkpoint_type, state_type, envelope, captured, future_version in profiles:
        assert state_type.model_config["revalidate_instances"] == "always"
        assert checkpoint_type.model_config["revalidate_instances"] == "always"
        header = envelope.model_dump(exclude={"experiment"})
        audited = checkpoint_type(**(header | {"experiment": captured}))
        assert audited.experiment is not captured
        assert audited.to_json() == envelope.to_json()
        for field in ("version", "nested_version", "accounting"):
            invalid = captured.model_dump(mode="python")
            if field == "version":
                invalid["schema_version"] = future_version
            elif field == "nested_version":
                invalid["controlled"][0]["state"]["payload"]["schema_version"] = 2
            else:
                invalid["periods"][0]["candidates"][0]["settlement"]["net_value"]["numerator"] += 1
            untrusted = state_type.model_construct(**invalid)
            with pytest.raises(ValueError):
                checkpoint_type(**(header | {"experiment": untrusted}))
            with pytest.raises(ValueError):
                checkpoint_type.model_validate(envelope.model_copy(update={"experiment": untrusted}))
            assert owner.export_state().to_json() == expected.to_json()


def test_checkpoint_rejects_substituted_pending_original_features():
    from coleman.prediction import RidgeInput

    experiment = Experiment()
    experiment.step(0)
    state = experiment.export_state()
    changed = state.model_dump()
    assert state.pending is not None
    changed["pending"]["prediction"]["input"] = RidgeInput(
        schema_version=1, config=state.pending.prediction.input.config, values=(1.0, 1.0)
    )
    with pytest.raises(ValueError, match="retained original producer"):
        ExperimentState.model_validate(changed)


def test_checkpoint_rejects_substituted_duplicate_observation():
    experiment = Experiment()
    experiment.step(0)
    experiment.step(1)
    changed = experiment.export_state().model_dump()
    changed["observations"][0]["ridge_label"]["value"] = 0.5
    with pytest.raises(ValueError, match="retained original period"):
        ExperimentState.model_validate(changed)


def test_complete_artifact_rejects_missing_candidate_period_rows(full_experiment, tmp_path):
    state = full_experiment.export_state()
    changed = state.model_dump()
    changed["periods"][-1]["candidates"] = changed["periods"][-1]["candidates"][:-1]
    unchecked = ExperimentState.model_construct(**changed)
    with pytest.raises(ValueError):
        publish(unchecked, tmp_path)
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize("movement", [-1.0, 0.25, None])
def test_checkpoint_rejects_false_coefficient_movement_atomically(movement):
    experiment = Experiment()
    experiment.step(0)
    state = experiment.export_state()
    changed = state.model_dump()
    index = model.MODES.index("plain")
    changed["periods"][0]["candidates"][index]["coefficient_movement"] = movement
    with pytest.raises(ValueError):
        ExperimentState.model_validate(changed)
    # The public owner revalidates even an unchecked top-level model. Failed
    # continuation cannot advance its original state or partial math frontier.
    with pytest.raises(ValueError):
        experiment.restore_state(ExperimentState.model_construct(**changed))
    assert experiment.export_state().to_json() == state.to_json()


@pytest.mark.parametrize("field,value", [("raw", ()), ("applications", ()), ("index", 2)])
def test_checkpoint_rejects_incomplete_or_redated_original_labels_atomically(field, value):
    experiment = Experiment()
    experiment.step(0)
    experiment.step(1)
    state = experiment.export_state()
    changed = state.model_dump()
    for observation in (changed["observations"][0], changed["periods"][1]["observation"]):
        observation[field] = value
    with pytest.raises(ValueError):
        ExperimentState.model_validate(changed)
    with pytest.raises(ValueError):
        experiment.restore_state(ExperimentState.model_construct(**changed))
    assert experiment.export_state().to_json() == state.to_json()


def test_checkpoint_rejects_an_unjoined_final_funded_book():
    experiment = Experiment()
    experiment.step(0)
    state = experiment.export_state()
    changed = state.model_dump()
    changed["books"][0]["state"]["cash"] = {"numerator": 123, "denominator": 1}
    with pytest.raises(ValueError, match="actual final settlement"):
        ExperimentState.model_validate(changed)


def test_checkpoint_rejects_self_consistent_but_false_funded_raw_return():
    from coleman.core.numeric import Scalar
    from coleman.objectives.volatility import apply_signed_volatility

    experiment = Experiment()
    experiment.step(0)
    experiment.step(1)
    state = experiment.export_state()
    changed = state.model_dump()
    application = state.observations[0].applications[0]
    altered_raw = Scalar(spec=model.RAW.spec, status="observed", value=0.5)
    altered_application = apply_signed_volatility(application.selection, altered_raw)
    for copy in (changed["observations"][0], changed["periods"][1]["observation"]):
        copy["raw"] = (altered_raw, *copy["raw"][1:])
        copy["applications"] = (altered_application, *copy["applications"][1:])
    with pytest.raises(ValueError, match="actual original funded path"):
        ExperimentState.model_validate(changed)


def test_checkpoint_rejects_a_substituted_original_selected_scale():
    from coleman.core.primitives import fingerprint

    experiment = Experiment()
    experiment.step(0)
    experiment.step(1)
    changed = experiment.export_state().model_dump()
    for copy in (changed["observations"][0], changed["periods"][1]["observation"]):
        copy["applications"][0]["selection"]["decision_fingerprint"] = fingerprint({"substituted": True})
    with pytest.raises(ValueError, match="retained original producing selection"):
        ExperimentState.model_validate(changed)


def test_period_rejects_a_stale_control_objective_from_another_actual_period(full_experiment):
    from examples.synthetic_portfolio.diagnostics import _control_objective
    from examples.synthetic_portfolio.flow_contracts import Period

    before = full_experiment.periods[1]
    actual = next(row.control for row in full_experiment.periods[3].candidates if row.name == "plain")
    assert actual is not None and actual.disposition == "released"
    changed = before.model_dump()
    index = model.MODES.index("plain")
    changed["candidates"][index]["control"] = actual
    changed["candidates"][index]["control_objective"] = _control_objective(actual)
    with pytest.raises(ValueError, match="stale last outcome"):
        Period.model_validate(changed)


def test_json_parquet_reproducibility_and_no_partial_completion(full_experiment, tmp_path):
    import pyarrow.parquet as parquet

    partial = Experiment()
    partial.step(0)
    with pytest.raises(ValueError, match="every executed"):
        publish(partial.export_state(), tmp_path / "partial")
    assert not (tmp_path / "partial" / "complete.json").exists()
    first, second = tmp_path / "first", tmp_path / "second"
    publish(full_experiment.export_state(), first)
    publish(full_experiment.export_state(), second)
    assert (first / "trajectory.json").read_bytes() == (second / "trajectory.json").read_bytes()
    assert (first / "trajectory.parquet").read_bytes() == (second / "trajectory.parquet").read_bytes()
    table = parquet.read_table(first / "trajectory.parquet")
    assert table.num_rows == 16 * len(model.MODES)
    assert "executed_full_l1_notional_exact" in table.column_names
    manifest = json.loads((first / "complete.json").read_text())
    assert manifest["status"] == "complete"


def test_interrupted_real_publication_has_no_false_completed_artifact(full_experiment, tmp_path):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("Cancellation between actual immutable data publications")

    with pytest.raises(InterruptedError):
        publish(full_experiment.export_state(), tmp_path, guard=cancel)
    assert (tmp_path / "trajectory.json").exists()
    assert not (tmp_path / "trajectory.parquet").exists()
    assert not (tmp_path / "complete.json").exists()
    assert publish(full_experiment.export_state(), tmp_path).exists()
