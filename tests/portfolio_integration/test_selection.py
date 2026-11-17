"""Actual development-only tuning, durable protected access and fixed fallback."""

import json

import pytest

from coleman.evaluation.protected.audit import EvaluationAudit
from coleman.evaluation.protected.contracts import FrozenEvaluation
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.selection import DevelopmentSelection, ProtectedResult, develop, run_protected


@pytest.fixture(scope="module")
def selected():
    return develop()


def test_actual_development_grid_retains_losers_and_stops_before_protected_rows(selected):
    assert tuple(trial.strength for trial in selected.trials) == (0.025, 0.05, 0.1)
    expected = min(selected.trials, key=lambda trial: (-trial.net_value.fraction, trial.strength))
    assert selected.selected_strength == expected.strength
    assert all(trial.state.next_index == 6 for trial in selected.trials)
    assert all(trial.state.observed_prices[-1].at == model.prices(5).at for trial in selected.trials)
    assert DevelopmentSelection.from_json(selected.to_json()).to_json() == selected.to_json()


def test_actual_protected_run_retains_descriptive_limits_and_real_access(selected, tmp_path):
    result = run_protected(selected, tmp_path)
    assert result.state.next_index == 16
    assert result.state.strength == selected.selected_strength
    assert result.status == "descriptive_only" and result.fallback == "equal"
    assert result.frozen.protocol.dependence == "serial"
    assert not result.frozen.protocol.normal_differences_assumed
    assert tuple(row.candidate_id for row in result.frozen.candidates) == tuple(sorted(model.MODES))
    audit = EvaluationAudit(tmp_path / "access-audit")
    assert audit.history(result.access.holdout_id) == (result.access,)
    assert audit.load_freeze(result.frozen.fingerprint) == result.frozen
    contrasts = {row.hypothesis_id for row in result.frozen.comparison_family}
    assert {"penalty-vs-cadence", "penalty-vs-movement"} <= contrasts
    assert (tmp_path / "development-selection.json").exists()
    assert (tmp_path / "protected-freeze.json").exists()
    changed = json.loads(result.to_json())
    changed["access"]["data_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="actual frozen evaluation"):
        ProtectedResult.model_validate(changed)
    # These are structurally self-consistent alternative frozen/access records,
    # not a claim that altered receipts were reserved in the original audit.
    # Their candidate configurations still must describe the retained state.
    for field, value in (
        ("strength", 0.2),
        ("mode", "plain"),
        ("cash_reserve", "0"),
        ("recipe_target_adapter", "different-adapter-v1"),
        ("funded_profile", "different-funding-v1"),
        ("warm_state", "sha256:" + "0" * 64),
    ):
        changed = result.model_dump(mode="json")
        configuration = json.loads(changed["frozen"]["candidates"][0]["configuration_json"])
        configuration[field] = value
        changed["frozen"]["candidates"][0]["configuration_json"] = json.dumps(configuration)
        altered = FrozenEvaluation.model_validate(changed["frozen"])
        changed["access"]["evaluation_fingerprint"] = altered.fingerprint
        with pytest.raises(ValueError):
            ProtectedResult.model_validate(changed)
