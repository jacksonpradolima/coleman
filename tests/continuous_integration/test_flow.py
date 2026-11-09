"""Independent chronological arithmetic and actual continuous/finite composition."""

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def allocation_example():
    name = "authored_continuous_integration_allocation"
    path = Path(__file__).parents[2] / "examples/continuous/allocation.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_chronological_predict_then_update_retains_negative_result_and_exact_zero_label(continuous_example):
    result = continuous_example.run_prediction()
    rows = result["rows"]
    assert [row["eligible_samples"] for row in rows] == [0, 1, 2, 3]
    assert [row["prediction"] for row in rows] == pytest.approx([0, 1, Fraction(6, 5), Fraction(-7, 12)])
    assert [row["mean"] for row in rows] == pytest.approx([0, 2, 1, Fraction(1, 3)])
    assert [row["last"] for row in rows] == [0, 2, 0, -1]
    assert [row["target"] for row in rows] == [2, 0, -1, 1]
    assert result["samples"] == 4
    errors = result["mean_squared_errors"]
    assert errors["prediction"] == pytest.approx(float(Fraction(44449, 14400)))
    assert errors["fixed"] == 1.5
    assert errors["prediction"] > errors["fixed"]
    assert result["predictive_uncertainty"] == "not_qualified"
    assert result["inference"] == "insufficient_evidence"
    assert continuous_example.run_prediction() == result


def test_real_coupled_active_box_differs_from_feasible_coordinate_clipping(allocation_example):
    example = allocation_example
    solved = example.solve_continuous(example.problem(0.0, coupled_fixture=True))
    assert solved.status == "optimal"
    assert solved.proposal == pytest.approx((1.0, 1.0))
    assert solved.executed == pytest.approx((0.5, 1.25))
    assert solved.certificate.value == pytest.approx(-2.8125)
    assert solved.certificate.gradient == pytest.approx((-0.75, 0.0))
    clipped_value = 0.5**2 + 0.5 * 1.0 + 1.0**2 - 3 * (0.5 + 1.0)
    assert clipped_value == -2.75
    assert solved.certificate.value < clipped_value


def test_full_composed_routes_use_same_prediction_inputs_and_separate_own_path_charges(allocation_example):
    example = allocation_example
    result = example.run()
    rows = result["trace"]
    assert len(rows) == 20
    assert {row["mode"] for row in rows} == {"fixed", "periodic", "finite", "forecast-continuous", "projected-online"}
    assert result["online_observations"] == 4
    assert result["capital_reference_credit_each"] == 1
    assert result["tuning_trials_each"] == 0
    for row in rows:
        point = row["executed"]
        assert sum(point) == pytest.approx(1, abs=1e-12)
        assert all(0 <= value <= 1 for value in point)
        assert row["forecast"] == result["forecast"]["rows"][row["step"]]["prediction"]
        assert row["own_book_charge_credit"] == pytest.approx(0.001 * row["coefficient_l1_movement"])
        assert row["net_credit"] == pytest.approx(row["gross_credit"] - row["own_book_charge_credit"])
        assert row["objective_penalty_paid_as_fee"] == 0
    fixed = [row for row in rows if row["mode"] == "fixed"]
    assert all(row["coefficient_l1_movement"] == 0 for row in fixed)
    periodic = [row for row in rows if row["mode"] == "periodic"]
    assert periodic[1]["executed"] == periodic[0]["executed"]
    assert periodic[3]["executed"] == periodic[2]["executed"]
    assert result["best_fixed_same_function_comparator"] == pytest.approx([0.75, 0.25])
    assert result["comparator_specific_net_wealth_regret"] == "not_qualified"
    assert result["inference"] == "insufficient_evidence"
    assert example.run() == result


@pytest.mark.parametrize("forecast", [-1.0, 0.0, 0.4, 0.5, 1.0, 1.2])
def test_existing_finite_solver_and_continuous_solution_match_declared_discretization_envelope(
    allocation_example, forecast
):
    example = allocation_example
    finite = example.finite_baseline(forecast)
    continuous = example.solve_continuous(example.problem(forecast))
    assert continuous.status == "optimal"
    # f restricted to sum(x)=1 has curvature2. Grid spacing1/4 gives
    # squared distance at most(1/8)^2, plus at most1e-12 score rounding.
    gap = example.objective(finite, forecast) - continuous.certificate.value
    assert -1e-12 <= gap <= 1 / 64 + 1e-12


def test_restore_refuses_original_feature_substitution_even_with_a_real_matching_prediction(continuous_example):
    example = continuous_example
    owner = example.ForecastOwner()
    example.emit(owner, 0)
    current = owner.host.export_state()
    pending = current.pending[0]
    changed_input = type(pending.input).model_validate(pending.input.model_copy(update={"values": (4.0, 0.0)}))
    changed_pending = type(pending).model_validate(
        pending.model_copy(
            update={
                "input": changed_input,
                "prediction": owner.host.ridge.predict(changed_input),
            }
        )
    )
    forged = current.model_copy(update={"pending": (changed_pending,)})
    fresh = example.MODEL.ForecastHost()
    before = fresh.export_state().to_json()
    with pytest.raises(ValueError, match="original request differs"):
        fresh.restore_state(forged)
    assert fresh.export_state().to_json() == before


def test_restore_refuses_a_prediction_recomputed_after_its_own_label_and_keeps_old_state(continuous_example):
    example = continuous_example
    owner = example.ForecastOwner()
    first = example.emit(owner, 0)
    owner.replay.admit_feedback((example.feedback(first, 2.0),))
    example.emit(owner, 1)
    current = owner.host.export_state()
    observed = current.observations[0]
    leaked = owner.host.ridge.predict(observed.pending.input)
    assert leaked.sample_count == 1
    assert observed.pending.prediction.sample_count == 0
    changed_pending = type(observed.pending).model_validate(
        observed.pending.model_copy(
            update={
                "prediction": leaked,
                "mean_control": 2.0,
                "last_control": 2.0,
            }
        )
    )
    changed_observation = type(observed).model_validate(observed.model_copy(update={"pending": changed_pending}))
    forged = current.model_copy(update={"observations": (changed_observation,)})
    fresh = example.MODEL.ForecastHost()
    before = fresh.export_state().to_json()
    with pytest.raises(ValueError, match="own or a future accepted label"):
        fresh.restore_state(forged)
    assert fresh.export_state().to_json() == before


def test_synthetic_label_provenance_binds_the_actual_public_generator_source(continuous_example):
    example = continuous_example
    owner = example.ForecastOwner()
    record = example.emit(owner, 0)
    offered = example.feedback(record, 2.0)
    assert offered.origins[0].producer == example.implementation()
    assert offered.origins[0].producer != example.MODEL.implementation()
