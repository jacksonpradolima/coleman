"""Pure method support, full-family projection and import boundaries."""

import json
import subprocess
import sys

import pytest
from design_support import changed, make_design, make_selection

from coleman.evaluation.design.contracts import ExperimentDesign
from coleman.evaluation.design.projection import (
    UnsupportedEvaluationError,
    aggregate_meaning,
    project_frozen,
    qualify_inference,
)
from coleman.evaluation.protected.bounded_inference import BoundedMeanProtocol
from coleman.evaluation.protected.contracts import EvaluationProtocol


def edited_design(edit):
    data = make_design().model_dump(mode="json")
    edit(data)
    return ExperimentDesign.model_validate(data)


def test_bounded_projection_keeps_base_identity_and_binds_complete_frozen_context():
    design = make_design()
    selection = make_selection(design)
    original = design.to_json()
    qualification = qualify_inference(design, selection)
    frozen = project_frozen(design, selection)
    assert qualification.status == "supported"
    assert isinstance(qualification.protocol, BoundedMeanProtocol)
    assert isinstance(frozen.protocol, BoundedMeanProtocol)
    assert frozen.protocol == qualification.protocol
    assert frozen.protocol.meaning == aggregate_meaning(design)
    assert frozen.protocol.outcome_lower == 0.0
    assert frozen.protocol.outcome_upper == 1.0
    assert frozen.protocol.holdout_id == design.scope.holdout_id
    assert frozen.comparison_family == design.hypotheses
    for candidate in frozen.candidates:
        config = json.loads(candidate.configuration_json)
        assert config == {
            "schema_version": 1,
            "profile": "domain-neutral-protected-candidate-v1",
            "base": design.candidate(candidate.candidate_id).candidate.model_dump(mode="json"),
            "design_fingerprint": design.fingerprint,
            "selection_fingerprint": selection.fingerprint,
        }
        assert (
            candidate.configuration_fingerprint
            != design.candidate(candidate.candidate_id).candidate.configuration_fingerprint
        )
    assert design.to_json() == original
    assert project_frozen(design, selection).fingerprint == frozen.fingerprint


def test_unbounded_finite_t_request_uses_only_its_own_declared_normal_difference_assumptions():
    def edit(data):
        data["metrics"][0]["quantity"]["spec"].update(minimum=None, maximum=None)
        data["inference"].update(
            method="paired_t_holm_bonferroni_ci_v1", normal_differences_assumed=True, bounds_basis=None
        )

    design = edited_design(edit)
    qualification = qualify_inference(design, make_selection(design))
    assert qualification.status == "supported"
    assert isinstance(qualification.protocol, EvaluationProtocol)
    assert not isinstance(qualification.protocol, BoundedMeanProtocol)
    assert qualification.protocol.normal_differences_assumed is True


@pytest.mark.parametrize("minimum,maximum", [(None, -1.0), (-3.0, None), (-3.0, -1.0), (None, None)])
def test_t_projection_capacity_uses_a_placeholder_inside_the_declared_target(minimum, maximum):
    from coleman.evaluation.design.aggregates import require_projection_capacity

    def edit(data):
        data["metrics"][0]["quantity"]["spec"].update(minimum=minimum, maximum=maximum)
        data["inference"].update(
            method="paired_t_holm_bonferroni_ci_v1", normal_differences_assumed=True, bounds_basis=None
        )

    design = edited_design(edit)
    selection = make_selection(design)
    assert qualify_inference(design, selection).status == "supported"
    require_projection_capacity(design, selection)


def test_negative_only_t_target_retains_actual_signed_raw_and_aggregate_values():
    from design_support import make_raw_panel

    from coleman.evaluation.design.aggregates import require_projection_capacity

    original = make_design()
    metric = changed(
        original.metrics[0],
        quantity=changed(
            original.metrics[0].quantity, spec=changed(original.metrics[0].quantity.spec, minimum=None, maximum=-1.0)
        ),
    )
    inference = changed(
        original.inference, method="paired_t_holm_bonferroni_ci_v1", normal_differences_assumed=True, bounds_basis=None
    )
    design = changed(original, metrics=(metric,), inference=inference)
    selection = make_selection(design)
    require_projection_capacity(design, selection)
    values = {
        (candidate.candidate.candidate_id, member.member_id, seed): -1.5
        for candidate in design.candidates
        for member in design.protected_members
        for seed in design.seeds
    }
    panel = make_raw_panel(design, selection, values=values)
    assert all(row.record.measurement(design.primary_metric).value == -1.5 for row in panel.rows)
    assert all(row.record.measurement(design.primary_metric).value == -1.5 for row in panel.paired_observations())


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("dependence", "serial", "dependent_or_unknown_final_units"),
        ("dependence", "clustered", "dependent_or_unknown_final_units"),
        ("dependence", "unknown", "dependent_or_unknown_final_units"),
        ("assignment", "adaptive_partial", "adaptive_partial_inferential_outcomes"),
        ("analysis", "optional_stopping", "unmatched_optional_stopping"),
        ("normal_differences_assumed", True, "bounded_profile_does_not_declare_normality"),
        ("bounds_basis", None, "unknown_or_unjustified_target_bounds"),
    ],
)
def test_valid_but_unsupported_assumptions_have_typed_reasons_without_fake_freezes(field, value, reason):
    original = make_design()
    design = changed(original, inference=changed(original.inference, **{field: value}))
    selection = make_selection(design)
    qualification = qualify_inference(design, selection)
    assert qualification.status == "unsupported"
    assert qualification.protocol is None
    assert reason in qualification.reasons
    with pytest.raises(UnsupportedEvaluationError) as raised:
        project_frozen(design, selection)
    assert raised.value.qualification == qualification


@pytest.mark.parametrize(
    "minimum,maximum,reason",
    [(None, None, "unknown_or_unjustified_target_bounds"), (0.0, 0.0, "degenerate_bounded_target_range")],
)
def test_bounded_method_cannot_borrow_bounds_from_another_quantity(minimum, maximum, reason):
    def edit(data):
        data["metrics"][0]["quantity"]["spec"].update(minimum=minimum, maximum=maximum)

    design = edited_design(edit)
    assert reason in qualify_inference(design, make_selection(design)).reasons


def test_t_method_without_normal_difference_declaration_is_explicitly_unsupported():
    design = make_design()
    design = changed(design, inference=changed(design.inference, method="paired_t_holm_bonferroni_ci_v1"))
    assert qualify_inference(design, make_selection(design)).reasons == ("normal_differences_not_assumed",)


def test_complete_causal_within_unit_learning_is_supported_but_cross_unit_adaptation_is_not():
    design = make_design()
    assert design.candidate("candidate").strategy.update_scope == "within_unit"
    assert qualify_inference(design, make_selection(design)).status == "supported"
    candidate = design.candidate("candidate")
    candidate = changed(candidate, strategy=changed(candidate.strategy, update_scope="across_units"))
    design = changed(design, candidates=(design.candidate("baseline"), candidate))
    assert qualify_inference(design, make_selection(design)).reasons == ("strategy_updates_across_final_units",)


def test_a_predeclared_surrogate_comparison_is_not_relabelled_as_a_domain_outcome():
    design = make_design()
    primary = changed(design.metric("quality"), role="surrogate")
    design = changed(design, metrics=(primary,))
    frozen = project_frozen(design, make_selection(design))
    assert frozen.protocol.metric_name == "quality"
    assert frozen.protocol.meaning.startswith("authored-ci-quality-v1:")


def test_directionless_primary_and_compared_retrospective_oracle_are_unsupported():
    design = make_design()
    primary = changed(design.metric("quality"), direction="none")
    directionless = changed(design, metrics=(primary,))
    assert (
        "primary_metric_has_no_comparison_direction"
        in qualify_inference(directionless, make_selection(directionless)).reasons
    )
    candidate = changed(design.candidate("candidate"), role="retrospective_oracle")
    oracle = changed(design, candidates=(design.candidate("baseline"), candidate))
    assert qualify_inference(oracle, make_selection(oracle)).reasons == ("retrospective_oracle_comparison",)


def test_unused_diagnostic_oracle_can_remain_in_the_complete_raw_family():
    from coleman.evaluation.protected.contracts import FrozenCandidate

    design = make_design()
    oracle = changed(
        design.candidate("candidate"),
        candidate=FrozenCandidate.from_configuration("diagnostic-oracle", {"method": "constructed-retrospective"}),
        role="retrospective_oracle",
    )
    design = changed(design, candidates=(*design.candidates, oracle))
    frozen = project_frozen(design, make_selection(design))
    assert len(frozen.candidates) == 3
    assert len(frozen.comparison_family) == 1


def test_horizon_and_development_choice_change_projection_but_never_reset_source_scope():
    design = make_design()
    selection = make_selection(design)
    before = project_frozen(design, selection)
    primary = design.metric("quality")
    horizon = changed(primary.quantity.horizon, version="2")
    revised = changed(design, metrics=(changed(primary, quantity=changed(primary.quantity, horizon=horizon)),))
    after = project_frozen(revised, make_selection(revised))
    assert before.fingerprint != after.fingerprint
    assert before.protocol.holdout_id == after.protocol.holdout_id
    selected = changed(selection, actual_selected_at=selection.actual_selected_at.replace(second=1))
    assert project_frozen(design, selected).fingerprint != before.fingerprint


def test_wrong_selection_or_shrunk_final_family_and_freeze_capacity_are_rejected():
    from coleman.evaluation.protected.contracts import FrozenCandidate

    design = make_design()
    selection = make_selection(design)
    with pytest.raises(ValueError, match="design"):
        project_frozen(design, changed(selection, design_fingerprint="sha256:" + "a" * 64))
    extra = changed(
        design.candidate("candidate"),
        candidate=FrozenCandidate.from_configuration("third-candidate", {"method": "constructed-third"}),
    )
    complete_design = changed(design, candidates=(*design.candidates, extra))
    shrunk_selection = make_selection(complete_design, selected_candidate_ids=("baseline", "candidate"))
    with pytest.raises(ValueError, match="complete final candidate family"):
        project_frozen(complete_design, shrunk_selection)
    tiny = changed(design, limits=changed(design.limits, max_freeze_bytes=1))
    with pytest.raises(ValueError, match="freeze capacity"):
        project_frozen(tiny, make_selection(tiny))
    with pytest.raises(ValueError, match="unsupported qualification"):
        UnsupportedEvaluationError(qualify_inference(design, selection))


def test_projection_and_panel_cold_import_without_io_analytics_or_native_transports(tmp_path):
    script = r"""
import importlib.abc
import sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ("coleman.corpus", "coleman.artifacts", "coleman.cli", "coleman.mcp",
                 "coleman.application", "coleman.runner", "coleman.evaluation.protected.audit",
                 "coleman.evaluation.protected.comparison", "sqlite3", "numpy", "scipy", "pyarrow", "polars")
        if any(fullname == root or fullname.startswith(root + ".") for root in roots):
            raise AssertionError("forbidden pure-domain import: " + fullname)
sys.meta_path.insert(0, Reject())
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.projection import qualify_inference, project_frozen
assert callable(qualify_inference) and callable(project_frozen)
assert RawPanel.model_fields["mode"].is_required()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script], cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
