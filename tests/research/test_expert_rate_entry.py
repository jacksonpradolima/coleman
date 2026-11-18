from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.core.primitives import fingerprint
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from examples.research.expert_rate_assess import development_projection, read_family, select_development
from examples.research.expert_rate_entry import _process, publish, run_study
from examples.research.expert_rate_execute import (
    BaselineState,
    HostBaseline,
    ResourceLimitError,
    loss_panel,
    metric,
    new_owner,
    restore_checkpoint,
    run_case,
)
from examples.research.expert_rate_math import comparator_losses, periodic_trace, reference_trace, score
from examples.research.expert_rate_process_records import Attempt, DevelopmentChoice, Gate, OracleResult, StudyReport
from examples.research.expert_rate_records import (
    METRICS,
    CaseRun,
    DevelopmentFamily,
    DevelopmentScore,
    FamilyRun,
    StudyInput,
    admit_protocol,
    load_protocol,
)
from tests.installed_support import build_install_offline_wheel

SOURCE = Path(__file__).parents[2] / "examples/research/expert-rate-entry-protocol-v2.json"


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(SOURCE)


@pytest.fixture(scope="module")
def actual(protocol):
    return run_case(protocol, protocol.cases[0].case_id, "fixed-uniform", 0, lambda: None)


def test_original_protocol_roundtrip_and_full_source_admission(protocol):
    assert len(protocol.cases) == 48
    assert len(protocol.settings) == 17
    assert sum(len(c.panels) for c in protocol.cases) == 1512
    assert admit_protocol(StudyInput.from_json(protocol.to_json())).to_json() == protocol.to_json()


def test_explicit_pre_outcome_v2_preserves_v1_panels_settings_budgets_and_old_readability(protocol):
    legacy_source = SOURCE.with_name("expert-rate-entry-protocol.json")
    assert (
        hashlib.sha256(legacy_source.read_bytes()).hexdigest()
        == "b826488634a5fba7302b2575ed1bc4166982cde4785984eabc80b3fb567b1639"
    )
    legacy = load_protocol(legacy_source)
    assert legacy.cases == protocol.cases
    assert legacy.settings == protocol.settings
    first, second = json.loads(legacy_source.read_text()), json.loads(SOURCE.read_text())
    for field in ("cases", "families", "budgets", "gate", "source", "recovery_plan"):
        assert first[field] == second[field]
    assert second["previous_protocol"]["content_digest"] == first["content_digest"]
    with pytest.raises(ValueError):
        admit_protocol(legacy)


def _development_families(protocol):
    """Author exact development-only arithmetic inputs, not a physical study."""
    order = ("fixed-uniform", "constant", "anytime", "fixed-share", "periodic")
    return tuple(
        DevelopmentFamily(
            schema_version=1,
            protocol_content=protocol.protocol_content,
            family=family,
            rows=tuple(
                DevelopmentScore(
                    case_id=c.case_id, setting=s, horizon=len(c.panels), exact_loss=str(Fraction(len(c.panels), 2))
                )
                for s in protocol.settings
                if s.family == family
                for c in protocol.cases
                if c.partition == "development"
            ),
        )
        for family in order
    )


def test_development_selection_uses_all_original_families_and_original_ties(protocol):
    projections = _development_families(protocol)
    chosen, settings = select_development(projections, protocol)
    assert chosen == "fixed-uniform"
    assert settings == (("fixed-uniform", 0), ("constant", 0), ("anytime", 0), ("fixed-share", 0), ("periodic", 0))
    with pytest.raises(ValueError):
        select_development(projections[:-1], protocol)
    with pytest.raises(ValueError):
        DevelopmentFamily.from_json(projections[0].to_json()[:-1] + ',"evaluation_status":"failed"}')


@pytest.mark.parametrize("bad", (True, False, 1.0, 0.0))
@pytest.mark.parametrize("kind", ("choice", "gate"))
def test_selection_reader_refuses_coerced_original_setting_slots(protocol, bad, kind):
    slots = (("fixed-uniform", 0), ("constant", bad), ("anytime", 0), ("fixed-share", 0), ("periodic", 0))
    data: dict[str, object] = {"schema_version": 1, "selected_family": "fixed-uniform", "selected_slots": slots}
    if kind == "choice":
        data.update(
            protocol_content=protocol.protocol_content,
            development_artifacts=tuple(
                {"path": f"family{i}.json", "sha256": "0" * 64, "size_bytes": 1} for i in range(5)
            ),
        )
        reader = DevelopmentChoice
    else:
        data.update(
            shape_excess=tuple(
                (shape, 0.0)
                for shape in ("stationary-gap", "stationary-small-gap", "neutral-prefix", "contrast-pulses")
            ),
            decision="retain_baselines",
        )
        reader = Gate
    with pytest.raises(ValueError):
        reader.from_json(json.dumps(data))


def test_development_grid_rejects_signed_zero_changed_treatment(protocol):
    original = _development_families(protocol)[1]
    original.require_grid(protocol)
    first = original.rows[0]
    changed_setting = first.setting.model_copy(update={"eta": -0.0})
    assert changed_setting == first.setting
    assert fingerprint(changed_setting.model_dump(mode="json")) != fingerprint(first.setting.model_dump(mode="json"))
    changed = original.model_copy(
        update={"rows": (first.model_copy(update={"setting": changed_setting}), *original.rows[1:])}
    )
    checked = DevelopmentFamily.from_json(changed.to_json())
    with pytest.raises(ValueError, match="grid"):
        checked.require_grid(protocol)


def _structural_projection_family(protocol, actual, family="fixed-uniform"):
    """Build only a projection/identity fixture, never measured learner evidence."""
    rows = []
    for setting in (item for item in protocol.settings if item.family == family):
        for source in protocol.cases:
            steps = tuple(
                actual.trace[0].model_copy(update={"d": i, "accepted_u": i + 1}) for i in range(len(source.panels))
            )
            checkpoint = actual.checkpoints[0]
            trace = tuple(step.model_copy(update={"state_digest": checkpoint.digest}) for step in steps)
            checkpoints = tuple(checkpoint.model_copy(update={"position": i}) for i in (0, len(trace) // 2, len(trace)))
            rows.append(
                actual.model_copy(
                    update={
                        "case_id": source.case_id,
                        "setting": setting,
                        "configuration_fingerprint": fingerprint(setting.model_dump(mode="json")),
                        "horizon": len(trace),
                        "attempted": len(trace),
                        "accepted_u": len(trace),
                        "trace": trace,
                        "checkpoints": checkpoints,
                    }
                )
            )
    return FamilyRun(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        family=family,
        repetition=0,
        status="completed",
        reason=None,
        runtime=InstalledRuntime.capture(),
        driver_runtime=SelectedInstalledRuntime.capture(extras=("cli",)),
        source_files=(),
        protocol_input_bytes=1,
        typed_input_bytes=1,
        logical_checkpoint_bytes=0,
        logical_case_bytes=0,
        attempted=sum(r.attempted for r in rows),
        resources=(
            metric("process_wall", 0.0, unit="s"),
            metric("process_cpu", 0.0, unit="s"),
            metric("peak_rss", 0.0, unit="MiB"),
        ),
        cases=tuple(rows),
    )


def test_evaluation_values_and_failure_cannot_enter_development_projection(protocol, actual):
    result = _structural_projection_family(protocol, actual)
    rows = result.cases
    before = development_projection(result, protocol)
    changed = []
    for row, source in zip(rows, protocol.cases, strict=True):
        if source.partition == "evaluation-exposed":
            row = row.model_copy(
                update={
                    "status": "resource_limited",
                    "reason": "authored-evaluation-failure",
                    "metrics": tuple(metric(name, None) for name in METRICS),
                }
            )
        changed.append(row)
    after = development_projection(
        result.model_copy(update={"status": "resource_limited", "reason": "evaluation-only", "cases": tuple(changed)}),
        protocol,
    )
    assert before is not None
    assert after is not None
    assert before.to_json() == after.to_json()
    altered = tuple(
        row.model_copy(update={"metrics": tuple(metric(name, -1.0) for name in METRICS)})
        if source.partition == "evaluation-exposed"
        else row
        for row, source in zip(rows, protocol.cases, strict=True)
    )
    assert development_projection(result.model_copy(update={"cases": altered}), protocol) == before
    bad_dev = list(rows)
    bad_dev[0] = bad_dev[0].model_copy(
        update={
            "status": "resource_limited",
            "reason": "development-failure",
            "metrics": tuple(metric(name, None) for name in METRICS),
        }
    )
    assert (
        development_projection(
            result.model_copy(
                update={"status": "resource_limited", "reason": "development-failure", "cases": tuple(bad_dev)}
            ),
            protocol,
        )
        is None
    )


def test_family_grid_rejects_revalidated_signed_zero_with_recomputed_treatment_fingerprint(protocol, actual):
    original = _structural_projection_family(protocol, actual, "constant")
    original.require_grid(protocol)
    first = original.cases[0]
    altered = first.setting.model_copy(update={"eta": -0.0})
    changed = first.model_copy(
        update={"setting": altered, "configuration_fingerprint": fingerprint(altered.model_dump(mode="json"))}
    )
    candidate = original.model_copy(update={"cases": (changed, *original.cases[1:])})
    checked = FamilyRun.from_json(candidate.to_json())
    assert checked.cases[0].configuration_fingerprint != first.configuration_fingerprint
    with pytest.raises(ValueError, match="grid"):
        checked.require_grid(protocol)


@pytest.mark.parametrize("field", ("panels", "setting", "source", "case_order", "version", "typed_decimal"))
def test_same_hash_label_cannot_admit_substituted_original_data(protocol, field):
    updates = {}
    if field in ("panels", "typed_decimal"):
        row = protocol.cases[0]
        first = (0.25, 0.25, 0.25) if field == "panels" else (Decimal(0), 0.125, 0.25)
        updates["cases"] = (row.model_copy(update={"panels": (first, *row.panels[1:])}), *protocol.cases[1:])
    elif field == "setting":
        updates["settings"] = (protocol.settings[0].model_copy(update={"eta": 0.1}), *protocol.settings[1:])
    elif field == "source":
        updates["source_files"] = (("coleman/learning/reference.py", "0" * 64),)
    elif field == "case_order":
        updates["cases"] = tuple(reversed(protocol.cases))
    else:
        updates["schema_version"] = 2
    changed = protocol.model_copy(update=updates)
    with pytest.raises((ValueError, TypeError)):
        new_owner(changed, protocol.cases[0].case_id, "constant", 0)


@pytest.mark.parametrize("version", (True, 1.0, 0, 2, 99))
def test_study_reader_rejects_unknown_or_coerced_version(protocol, version):
    fields = json.loads(protocol.to_json())
    fields["schema_version"] = version
    with pytest.raises(ValueError):
        StudyInput.from_json(json.dumps(fields))


def test_unchecked_original_protocol_header_is_refused_without_conversion_hooks(protocol):
    calls = []

    class Bomb:
        def __str__(self):
            calls.append("str")
            raise AssertionError("Unchecked source header must never be stringified")

    changed = protocol.model_copy(update={"schema_version": Bomb()})
    with pytest.raises((ValueError, TypeError)):
        admit_protocol(changed)
    assert calls == []


@pytest.mark.parametrize("kind", ("fixed", "periodic"))
def test_independently_authored_control_state_goldens_restore_and_reject_future(kind):
    source = Path(__file__).with_name("fixtures") / "expert-rate-v1" / (kind + "-state.json")
    content = source.read_text()
    state = BaselineState.from_json(content)
    assert state.to_json() == content
    assert state.update_count == 8
    owner = HostBaseline(state.setting, state.binding)
    owner.restore_state(state)
    assert owner.export_state().to_json() == content
    owner.observe(loss_panel((0.5, 0.0, 1.0)))
    assert owner.update_count == 9
    assert tuple(w.value for w in owner.weights) == state.coefficients
    changed = json.loads(content)
    changed["schema_version"] = 2
    with pytest.raises(ValueError):
        BaselineState.from_json(json.dumps(changed))


def test_hand_fixture_independent_three_by_three_share_arithmetic():
    rows = ((0.0, 0.5, 1.0),)
    eta = math.log(4)
    hedge = reference_trace("hedge-constant", rows, eta=eta)
    shared = reference_trace("fixed-share-excluding-origin", rows, eta=eta, alpha=0.25)
    for actual_value, expected in zip(hedge.after[0], (Fraction(4, 7), Fraction(2, 7), Fraction(1, 7)), strict=True):
        assert float(actual_value) == pytest.approx(float(expected), abs=2e-15)
    for actual_value, expected in zip(
        shared.after[0], (Fraction(27, 56), Fraction(17, 56), Fraction(12, 56)), strict=True
    ):
        assert float(actual_value) == pytest.approx(float(expected), abs=2e-15)
    assert score(((1 / 3, 1 / 3, 1 / 3),), rows) == Fraction(1 / 3) * Fraction(3, 2)


@pytest.mark.parametrize("bad", (True, 1, float("nan"), float("inf"), -0.25, 5.0))
def test_oracle_rejects_unsupported_rate_types_and_numerics(bad):
    with pytest.raises(ValueError):
        reference_trace("hedge-constant", ((0.0, 0.5, 1.0),), eta=bad)


@pytest.mark.parametrize("family", ("constant", "anytime", "fixed-share", "periodic", "fixed-uniform"))
def test_real_methods_checkpoint_continuation_and_independent_coefficients(protocol, family):
    case = protocol.cases[0]
    result = run_case(protocol, case.case_id, family, 0, lambda: None)
    assert result.accepted_u == len(case.panels)
    assert result.attempted == len(case.panels)
    assert tuple(c.position for c in result.checkpoints) == (0, len(case.panels) // 2, len(case.panels))
    assert CaseRun.from_json(result.to_json()).to_json() == result.to_json()
    owner = new_owner(protocol, case.case_id, family, 0)
    original_state = owner.export_state().to_json()
    future = json.loads(result.checkpoints[1].content)
    future["schema_version"] = 2
    future_bytes = json.dumps(future, sort_keys=True, separators=(",", ":"))
    bad_checkpoint = result.checkpoints[1].model_copy(
        update={"content": future_bytes, "digest": "sha256:" + hashlib.sha256(future_bytes.encode()).hexdigest()}
    )
    with pytest.raises(ValueError):
        restore_checkpoint(owner, bad_checkpoint)
    assert owner.export_state().to_json() == original_state
    restore_checkpoint(owner, result.checkpoints[1])
    for row in case.panels[len(case.panels) // 2 :]:
        owner.observe(loss_panel(row))
    assert owner.export_state().to_json() == result.checkpoints[-1].content
    setting = result.setting
    if family in ("constant", "anytime", "fixed-share"):
        oracle = reference_trace(
            {
                "constant": "hedge-constant",
                "anytime": "hedge-anytime-cumulative",
                "fixed-share": "fixed-share-excluding-origin",
            }[family],
            case.panels,
            eta=setting.eta,
            multiplier=setting.multiplier,
            alpha=setting.alpha,
        )
    else:
        oracle = periodic_trace(case.panels, setting.window or 64)
    for step, expected in zip(result.trace, oracle.after, strict=True):
        target = (1 / 3, 1 / 3, 1 / 3) if family == "fixed-uniform" else tuple(map(float, expected))
        assert step.after == pytest.approx(target, abs=2e-14, rel=2e-13)
    expected_total = score(tuple(step.before for step in result.trace), case.panels)
    expected_fixed, expected_switched = comparator_losses(case.panels)
    values = {measurement.name: measurement.value for measurement in result.metrics}
    assert values["mixture_loss"] == float(expected_total)
    assert values["fixed_comparator_excess"] == float(expected_total - expected_fixed)
    assert values["one_switch_comparator_excess"] == float(expected_total - expected_switched)


def test_periodic_causality_and_binary64_tie_hand_example():
    prefix = ((0.0, 1.0, 0.5),) * 8
    left = periodic_trace((*prefix, (1.0, 0.0, 0.5)), 8)
    right = periodic_trace((*prefix, (0.0, 1.0, 0.5)), 8)
    assert left.before == right.before
    assert tuple(map(float, left.before[8])) == (1.0, 0.0, 0.0)
    with localcontext() as context:
        context.prec = 80
        tied = periodic_trace(((0.5, 0.5, 0.5),) * 8, 8)
        assert tied.after[-1] == (Decimal(1) / 3,) * 3


def test_independent_switch_oracle_is_retrospective_and_signed_metrics_are_valid():
    assert comparator_losses(((0.0, 1.0, 1.0), (1.0, 0.0, 1.0))) == (Fraction(1), Fraction(0))
    assert metric("fixed_comparator_excess", -0.25).value == -0.25
    assert metric("coefficient_L1", 0.0).status == "observed"
    assert metric("coefficient_L1", None).value is None
    assert metric("coefficient_L1", None).status == "unavailable"
    with pytest.raises(ValueError):
        metric("coefficient_L1", float("nan"))


@pytest.mark.parametrize("change", ("accepted", "attempted", "horizon", "checkpoint", "metric", "trace"))
def test_case_reader_refuses_contradictory_complete_or_limited_work(actual, change):
    fields = json.loads(actual.to_json())
    if change == "accepted":
        fields["accepted_u"] = 64
    elif change == "attempted":
        fields["attempted"] = 0
    elif change == "horizon":
        fields["horizon"] = 1
    elif change == "checkpoint":
        fields["checkpoints"] = fields["checkpoints"][:1]
    elif change == "metric":
        fields.update(status="resource_limited", reason="declared-limit")
    else:
        fields["trace"] = fields["trace"][:-1]
    with pytest.raises(ValueError):
        CaseRun.from_json(json.dumps(fields))


def test_known_limit_is_incomplete_and_keeps_initial_state_not_fabricated_work(protocol):
    def refuse():
        raise ResourceLimitError("qualified-toy-observation-limit")

    result = run_case(protocol, protocol.cases[0].case_id, "fixed-uniform", 0, refuse)
    assert result.status == "resource_limited"
    assert result.attempted == 0
    assert result.accepted_u == 0
    assert tuple(c.position for c in result.checkpoints) == (0,)
    assert all(m.value is None for m in result.metrics)
    assert all(m.status == "unavailable" for m in result.metrics)


def test_actual_parent_reader_requires_grid_not_just_a_completed_label(protocol, actual, tmp_path):
    result = FamilyRun(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        family="fixed-uniform",
        repetition=0,
        status="completed",
        reason=None,
        runtime=InstalledRuntime.capture(),
        driver_runtime=SelectedInstalledRuntime.capture(extras=("cli",)),
        source_files=(),
        protocol_input_bytes=1,
        typed_input_bytes=1,
        logical_checkpoint_bytes=0,
        logical_case_bytes=0,
        attempted=actual.attempted * 48,
        resources=(
            metric("process_wall", 0.0, unit="s"),
            metric("process_cpu", 0.0, unit="s"),
            metric("peak_rss", 0.0, unit="MiB"),
        ),
        cases=(actual,) * 48,
    )
    content = result.to_json()
    reference = publish(tmp_path, "family.json", content)
    assert reference.sha256 == hashlib.sha256(content.encode()).hexdigest()
    with pytest.raises(ValueError, match="grid"):
        read_family(tmp_path, reference, protocol)


def test_immutable_publication_duplicate_concurrent_and_changed_content(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = tuple(workers.map(lambda _: publish(tmp_path, "same.json", '{"schema_version":1}'), range(2)))
    assert results[0] == results[1]
    assert (tmp_path / "same.json").read_text() == '{"schema_version":1}'
    with pytest.raises(ValueError):
        publish(tmp_path, "same.json", '{"schema_version":2}')


def test_duplicate_study_namespace_refuses_before_any_trial(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    with pytest.raises(FileExistsError):
        run_study(root)
    assert not tuple(root.iterdir())


def test_actual_process_deadline_retains_no_falsely_complete_publication(tmp_path):
    program = "import time,pathlib,sys;time.sleep(0.25);pathlib.Path(sys.argv[1]).write_text('false')"
    # A controller-mechanics fixture, never one of the15 formal study attempts.
    ok, wall, cpu = _process(
        [sys.executable, "-c", program, str(tmp_path / "complete.json")], tmp_path, "interrupted", 0.05
    )
    assert ok is False
    assert wall >= 0.05
    assert cpu >= 0.0
    assert not (tmp_path / "complete.json").exists()


def test_late_observed_success_is_still_a_declared_process_limit(tmp_path, monkeypatch):
    import examples.research.expert_rate_entry as driver

    # Real child/exit; only the observer clock models a delayed parent wakeup.
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(ticks))
    ok, wall, cpu = driver._process([sys.executable, "-c", "pass"], tmp_path, "late-observer", 1.0)
    assert ok is False
    assert wall == 2.0
    assert cpu >= 0.0


def test_top_level_limited_report_retains_unknown_reserved_work_and_original_wall_caps(protocol):
    order = ("constant", "anytime", "fixed-share", "periodic", "fixed-uniform")
    attempts = tuple(
        Attempt(
            schema_version=1,
            family=family,
            repetition=repetition,
            status="not_started",
            reason="authored-limit-fixture",
            reserved_calls=0,
            attempted_calls=None,
            wall_seconds=None,
            cpu_seconds=None,
            artifact=None,
        )
        for repetition in range(3)
        for family in order[repetition:] + order[:repetition]
    )
    first = attempts[0].model_copy(
        update={
            "status": "resource_limited",
            "reserved_calls": 6048,
            "wall_seconds": 180.01,
            "reason": "original-process-deadline",
        }
    )
    report = StudyReport(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        status="resource_limited",
        reason="authored-controller-contract-fixture",
        runtime=InstalledRuntime.capture(),
        driver_runtime=SelectedInstalledRuntime.capture(extras=("cli",)),
        source_files=(),
        attempts=(first, *attempts[1:]),
        reserved_calls=6048,
        known_attempted_calls=0,
        unknown_attempts=1,
        pass_wall_seconds=(180.01, 0.0, 0.0),
        baseline_wall_seconds=180.01,
        oracle=None,
        oracle_wall_seconds=None,
        development_choice=None,
        gate=None,
    )
    assert StudyReport.from_json(report.to_json()).to_json() == report.to_json()
    assert report.attempts[0].attempted_calls is None
    assert report.reserved_calls == 6048
    assert report.known_attempted_calls == 0
    assert report.unknown_attempts == 1
    fields = json.loads(report.to_json())
    for updates in (
        {"status": "completed", "reason": None},
        {"schema_version": 2},
        {"baseline_wall_seconds": -1.0},
        {"oracle_wall_seconds": -1.0},
        {"oracle_wall_seconds": float("nan")},
        {"oracle_wall_seconds": float("inf")},
        {"oracle_wall_seconds": True},
    ):
        with pytest.raises(ValueError):
            StudyReport.from_json(json.dumps({**fields, **updates}))
    zero = StudyReport.from_json(json.dumps({**fields, "oracle_wall_seconds": 0.0}))
    assert zero.oracle_wall_seconds == 0.0
    assert report.oracle_wall_seconds is None
    # Structural reader fixture only; these are not measured scientific trials.
    artifact = {"path": "authored.json", "sha256": "0" * 64, "size_bytes": 1}
    complete = json.loads(report.to_json())
    for attempt in complete["attempts"]:
        count = 1512 * (1 if attempt["family"] == "fixed-uniform" else 4)
        attempt.update(
            status="completed",
            reason=None,
            reserved_calls=count,
            attempted_calls=count,
            wall_seconds=1.0,
            cpu_seconds=0.0,
            artifact=artifact,
        )
    complete.update(
        status="completed",
        reason=None,
        reserved_calls=77112,
        known_attempted_calls=77112,
        unknown_attempts=0,
        oracle=artifact,
        oracle_wall_seconds=1.0,
        development_choice=artifact,
        gate={
            "schema_version": 1,
            "selected_family": "fixed-uniform",
            "selected_slots": (
                ("fixed-uniform", 0),
                ("constant", 0),
                ("anytime", 0),
                ("fixed-share", 0),
                ("periodic", 0),
            ),
            "shape_excess": tuple(
                (shape, 0.0)
                for shape in ("stationary-gap", "stationary-small-gap", "neutral-prefix", "contrast-pulses")
            ),
            "decision": "retain_baselines",
        },
    )
    valid = StudyReport.from_json(json.dumps(complete))
    assert valid.status == "completed"
    for field in ("family", "oracle"):
        changed = json.loads(valid.to_json())
        if field == "family":
            changed["attempts"][0]["wall_seconds"] = 181.0
        else:
            changed["oracle_wall_seconds"] = 181.0
        with pytest.raises(ValueError):
            StudyReport.from_json(json.dumps(changed))


def test_scientific_zero_count_and_success_boolean_keep_strict_kinds(protocol):
    slots = (("fixed-uniform", 0), ("constant", 0), ("anytime", 0), ("fixed-share", 0), ("periodic", 0))
    gate = Gate(
        schema_version=1,
        selected_family="fixed-uniform",
        selected_slots=slots,
        shape_excess=tuple(
            (shape, 0.0) for shape in ("stationary-gap", "stationary-small-gap", "neutral-prefix", "contrast-pulses")
        ),
        decision="retain_baselines",
    )
    oracle = OracleResult(
        schema_version=1,
        protocol_content=protocol.protocol_content,
        checked_steps=77112,
        maximum_coefficient_error=0.0,
        maximum_score_error=0.0,
        repetitions_equal=True,
        gate=gate,
    )
    fields = json.loads(oracle.to_json())
    for value in (1, 1.0, False):
        with pytest.raises(ValueError):
            OracleResult.from_json(json.dumps({**fields, "repetitions_equal": value}))
    invalid = json.loads(gate.to_json())
    invalid["independent_empirical_units"] = False
    with pytest.raises(ValueError):
        Gate.from_json(json.dumps(invalid))


def test_optional_click_command_help_bad_flags_and_math_cold_import(tmp_path):
    command = [sys.executable, "-m", "examples.research.expert_rate_entry"]
    help_result = subprocess.run([*command, "--help"], capture_output=True, text=True, timeout=30, check=False)
    assert help_result.returncode == 0
    assert "--output" in help_result.stdout
    malformed = subprocess.run(
        [*command, "--output", str(tmp_path), "--family", "unknown"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert malformed.returncode == 2
    source = (
        "import sys; import examples.research.expert_rate_math; "
        "assert 'coleman' not in sys.modules; assert 'click' not in sys.modules"
    )
    cold = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=30, check=False)
    assert cold.returncode == 0, cold.stderr


def test_real_shipped_clock_translation_relabeling_and_neutral_prefix(protocol):
    original = ((0.0, 0.25, 0.5), (0.5, 0.25, 0.0), (0.25, 0.0, 0.5))
    translated = tuple(tuple(x + 0.125 for x in row) for row in original)
    rotation = (2, 0, 1)
    for family in ("constant", "anytime", "fixed-share"):
        owners = [new_owner(protocol, protocol.cases[0].case_id, family, 1) for _ in range(3)]
        for row, shifted in zip(original, translated, strict=True):
            for owner, values in zip(owners, (row, shifted, tuple(row[i] for i in rotation)), strict=True):
                owner.observe(loss_panel(values))
            expected = tuple(w.value for w in owners[0].weights)
            assert tuple(w.value for w in owners[1].weights) == pytest.approx(expected, abs=2e-14)
            assert tuple(w.value for w in owners[2].weights) == pytest.approx(
                tuple(expected[i] for i in rotation), abs=2e-14
            )
    constant = [new_owner(protocol, protocol.cases[0].case_id, "constant", 2) for _ in range(2)]
    anytime = [new_owner(protocol, protocol.cases[0].case_id, "anytime", 2) for _ in range(2)]
    for _ in range(8):
        constant[1].observe(loss_panel((0.5, 0.5, 0.5)))
        anytime[1].observe(loss_panel((0.5, 0.5, 0.5)))
    for pair in (constant, anytime):
        for owner in pair:
            owner.observe(loss_panel(original[0]))
    assert tuple(w.value for w in constant[0].weights) == pytest.approx(tuple(w.value for w in constant[1].weights))
    assert tuple(w.value for w in anytime[0].weights) != pytest.approx(tuple(w.value for w in anytime[1].weights))


def test_noneditable_installed_outside_real_five_method_checkpoint_flows(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
    virtual = tmp_path / "installed"
    wheel = build_install_offline_wheel(uv, tmp_path / "dist", virtual, environment, extras=("cli",))
    assert wheel.is_file()
    outside = tmp_path / "outside"
    destination = outside / "examples" / "research"
    destination.mkdir(parents=True)
    assert not outside.is_relative_to(SOURCE.parents[2])
    for path in SOURCE.parent.glob("expert_rate_*.py"):
        shutil.copyfile(path, destination / path.name)
    for path in SOURCE.parent.glob("expert-rate-entry-protocol*.json"):
        shutil.copyfile(path, destination / path.name)
    program = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import coleman
assert 'site-packages' in coleman.__file__
from examples.research.expert_rate_records import load_protocol,CaseRun
from examples.research.expert_rate_execute import run_case,new_owner,restore_checkpoint,loss_panel
source=Path(sys.argv[1])/'examples/research/expert-rate-entry-protocol-v2.json'
protocol=load_protocol(source)
cases=[]
for family in ('constant','anytime','fixed-share','periodic','fixed-uniform'):
    result=run_case(protocol,protocol.cases[0].case_id,family,0,lambda:None)
    restored=CaseRun.from_json(result.to_json())
    owner=new_owner(protocol,result.case_id,family,0)
    restore_checkpoint(owner,restored.checkpoints[1])
    for row in protocol.cases[0].panels[12:]:
        owner.observe(loss_panel(row))
    assert owner.export_state().to_json()==restored.checkpoints[-1].content
    cases.append((family,result.accepted_u,result.attempted))
print(json.dumps(cases))
"""
    result = subprocess.run(
        [str(virtual / "bin/python"), "-I", "-B", "-c", program, str(outside)],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == [
        [family, 24, 24] for family in ("constant", "anytime", "fixed-share", "periodic", "fixed-uniform")
    ]
    help_result = subprocess.run(
        [str(virtual / "bin/python"), "-B", "-m", "examples.research.expert_rate_entry", "--help"],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--family" in help_result.stdout
