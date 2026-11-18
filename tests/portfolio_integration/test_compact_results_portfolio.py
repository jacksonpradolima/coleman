"""Compact publications retain real full evidence and independent accounting."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CommitIntent, CommitRequest
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.experiment import Experiment
from examples.synthetic_portfolio.readback import _read_coordinate_artifact
from examples.synthetic_portfolio.recipe import (
    COORDINATE_RESULT,
    FINAL_RESULT,
    LEGACY_COORDINATE_RESULT,
    LEGACY_FINAL_RESULT,
    recipe,
)
from examples.synthetic_portfolio.recipe import (
    STATE_V2 as STATE,
)
from examples.synthetic_portfolio.task_contracts import PortfolioCheckpoint, PortfolioCoordinateResult, PortfolioReport
from examples.synthetic_portfolio.task_results import (
    PortfolioCoordinateResultV2,
    PortfolioReportItemV2,
    PortfolioReportV2,
    PortfolioTerminalOutcome,
    terminal_outcomes,
)
from tests.portfolio_integration.test_task_state_portfolio import ObservedStore, make_layout


@pytest.fixture(scope="module")
def complete_evidence(tmp_path_factory):
    """Accept a real sixteen-observation transition, separate from batch tests."""
    layout, store = make_layout(tmp_path_factory.mktemp("compact-real-portfolio"))
    head = store.read_head()
    request = CommitRequest(
        schema_version=1,
        expected=head,
        intent=CommitIntent(
            schema_version=1,
            generation=head.generation + 1,
            kind="transition",
            content_fingerprint="sha256:" + sha256(b"actual-sixteen-period-unit-fixture").hexdigest(),
        ),
    )
    selected = layout.binding

    def observe(stage):
        owner = stage.owner(selected)
        for index in range(16):
            owner.experiment.step(index)

    accepted = store.commit_transition(
        request, (selected,), observe, compatibility=layout.context.preparation.compatibility
    )
    assert accepted.status == "new"
    result = layout.complete_result(store)
    checkpoint = layout.resolve_result(store, result)
    return layout, store, result, checkpoint


def report_item(result):
    """Use independent standard JSON/SHA arithmetic for the declared file identity."""
    content = json.dumps(
        result.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return PortfolioReportItemV2(
        schema_version=2,
        coordinate=result,
        artifact_profile="portfolio-coordinate-result-v2/core-json-v1",
        artifact_fingerprint="sha256:" + sha256(content).hexdigest(),
        artifact_byte_count=len(content),
    ), content


def test_compact_summaries_retain_independently_summed_credit_charges_and_every_actual_book(complete_evidence):
    layout, store, result, checkpoint = complete_evidence
    assert recipe().reference.version == "3"
    assert result.experiment_fingerprint == checkpoint.experiment.fingerprint
    assert tuple(row.to_json() for row in result.outcomes) == tuple(
        row.to_json() for row in terminal_outcomes(checkpoint.experiment)
    )
    assert tuple(row.name for row in result.outcomes) == model.MODES
    for index, summary in enumerate(result.outcomes):
        receipts = tuple(period.candidates[index].settlement for period in checkpoint.experiment.periods)
        assert summary.currency == "synthetic-credit"
        assert summary.terminal_net_value.fraction == receipts[-1].net_value.fraction
        for output, field in (
            (summary.proportional_paid, "proportional_paid"),
            (summary.recurring_paid, "recurring_paid"),
            (summary.executed_full_l1_notional, "turnover"),
        ):
            expected = sum(
                (
                    Fraction(getattr(receipt, field).numerator, getattr(receipt, field).denominator)
                    for receipt in receipts
                ),
                Fraction(),
            )
            assert output.fraction == expected
    prepared = store.load((layout.binding,), compatibility=layout.context.preparation.compatibility)
    assert result.snapshot.component == prepared.manifest.components[0]
    assert result.snapshot.manifest_fingerprint == prepared.manifest.fingerprint
    raw = (store.root / "blobs" / (result.snapshot.component.fingerprint[7:] + ".json")).read_bytes()
    assert len(raw) == result.snapshot.component.byte_count
    assert "sha256:" + sha256(raw).hexdigest() == result.snapshot.component.fingerprint
    assert STATE.decode(raw.decode()).to_json() == checkpoint.to_json()


@pytest.mark.parametrize("profile", ["legacy", "normalized"])
@pytest.mark.parametrize("change", ["partial", "future", "missing_period", "nonfinite_strength"])
def test_public_terminal_projection_keeps_complete_admission_for_every_state_profile(profile, change):
    experiment = Experiment()
    state = experiment.export_state() if profile == "legacy" else experiment.export_normalized_state()
    if change == "future":
        state = state.model_copy(update={"schema_version": 99})
    elif change == "missing_period":
        state = state.model_copy(update={"next_index": 16})
    elif change == "nonfinite_strength":
        state = state.model_copy(update={"strength": float("nan")})
    before = experiment.export_state().to_json()
    with pytest.raises((TypeError, ValueError)):
        terminal_outcomes(state)
    assert experiment.export_state().to_json() == before


def test_complete_compact_result_roundtrip_and_exact_artifact_reference(complete_evidence):
    _, _, result, _ = complete_evidence
    item, content = report_item(result)
    assert COORDINATE_RESULT.encode(result).encode() == content
    assert COORDINATE_RESULT.decode(content.decode()).to_json() == result.to_json()
    report = PortfolioReportV2(schema_version=2, interpretation="constructed_mechanism_only", results=(item,))
    encoded = FINAL_RESULT.encode(report)
    assert FINAL_RESULT.decode(encoded).to_json() == encoded
    # The actual interactive TaskCompletion wrapper is qualified separately.
    assert len(encoded.encode()) < 1024**2


def test_complete_result_and_full_resolver_each_use_one_actual_selected_load_without_crosscall_reuse(
    complete_evidence,
):
    layout, store, expected, checkpoint = complete_evidence
    observed = ObservedStore(store)
    result = layout.complete_result(cast(LocalStateStore, observed))
    assert observed.loads == 1
    assert result.to_json() == expected.to_json()
    assert layout.resolve_result(cast(LocalStateStore, observed), result).to_json() == checkpoint.to_json()
    assert observed.loads == 2


def test_real_head_advance_during_retained_lookup_cannot_publish_an_older_terminal_receipt(complete_evidence, tmp_path):
    _, _, _, checkpoint = complete_evidence
    layout, store = make_layout(tmp_path / "independent-actual-store")
    selected = layout.binding

    def request(current, label):
        return CommitRequest(
            schema_version=1,
            expected=current,
            intent=CommitIntent(
                schema_version=1,
                generation=current.generation + 1,
                kind="transition",
                content_fingerprint="sha256:" + sha256(label.encode()).hexdigest(),
            ),
        )

    # Restore the actual complete observation history through the strict public
    # state boundary. This supplies no fresh scientific observation or worker.
    store.commit_transition(
        request(store.read_head(), "restore-original-complete-observation-history"),
        (selected,),
        lambda stage: stage.owner(selected).restore_state(checkpoint),
        compatibility=layout.context.preparation.compatibility,
    )
    original = store.read_head()

    def advance_head():
        def independent_commit():
            independent = LocalStateStore.open(store.root)
            return independent.commit_transition(
                request(independent.read_head(), "explicit-no-new-observation-transition"),
                (selected,),
                lambda _stage: None,
                compatibility=layout.context.preparation.compatibility,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(independent_commit).result().status == "new"

    observed = ObservedStore(store, before_receipt=advance_head)
    with pytest.raises(ValueError, match="exact terminal current acknowledged full state"):
        layout.complete_result(cast(LocalStateStore, observed))
    assert observed.loads == 2
    assert store.read_head() != original
    assert not (layout.context.artifact_directory / "coordinate-result.json").exists()


@pytest.mark.parametrize("change", ["state_hash", "state_size", "source", "outcome", "experiment_hash"])
def test_typed_compact_claim_does_not_replace_complete_real_artifact_verification(complete_evidence, change):
    layout, store, result, _ = complete_evidence
    data = result.model_dump(mode="json")
    if change == "state_hash":
        data["snapshot"]["component"]["fingerprint"] = "sha256:" + "0" * 64
    elif change == "state_size":
        data["snapshot"]["component"]["byte_count"] += 1
    elif change == "source":
        data["source"]["definition_fingerprint"] = "sha256:" + "0" * 64
    elif change == "outcome":
        amount = data["outcomes"][0]["terminal_net_value"]
        # gcd(n + d, d) == gcd(n, d): change the exact value by one credit
        # while keeping the deliberately false claim structurally canonical.
        amount["numerator"] += amount["denominator"]
    else:
        data["experiment_fingerprint"] = "sha256:" + "0" * 64
    candidate = PortfolioCoordinateResultV2.model_validate(data)
    original = store.read_head()
    with pytest.raises(ValueError, match="actual terminal state artifacts"):
        layout.resolve_result(store, candidate)
    assert store.read_head() == original


@pytest.mark.parametrize("change", ["future", "manifest", "missing_candidate", "negative_fee", "foreign_owner"])
def test_compact_reader_refuses_future_or_internally_contradictory_publications(complete_evidence, change):
    _, _, result, _ = complete_evidence
    data = result.model_dump(mode="json")
    if change == "future":
        data["schema_version"] = 3
    elif change == "manifest":
        data["snapshot"]["manifest_fingerprint"] = "sha256:" + "0" * 64
    elif change == "missing_candidate":
        data["outcomes"].pop()
    elif change == "negative_fee":
        data["outcomes"][0]["proportional_paid"]["numerator"] = -1
    else:
        data["snapshot"]["component"]["binding"]["owner"]["id"] = "another-original-owner"
    with pytest.raises(ValueError):
        PortfolioCoordinateResultV2.from_json(json.dumps(data))


@pytest.mark.parametrize("change", ["bytes", "missing", "oversize"])
def test_owned_fixed_artifact_reader_rejects_modified_missing_or_oversized_bytes(complete_evidence, tmp_path, change):
    layout, _, result, _ = complete_evidence
    item, content = report_item(result)
    context = replace(layout.context, artifact_directory=tmp_path)
    path = tmp_path / "coordinate-result.json"
    if change == "bytes":
        path.write_bytes(content + b" ")
    elif change == "oversize":
        path.write_bytes(b" " * (1024**2 + 1))
    with pytest.raises((ValueError, OSError)):
        _read_coordinate_artifact(context, item)


def test_real_fixed_artifact_reader_and_public_state_resolver_preserve_every_checkpoint_field(
    complete_evidence, tmp_path
):
    layout, store, result, checkpoint = complete_evidence
    item, content = report_item(result)
    context = replace(layout.context, artifact_directory=tmp_path)
    ArtifactWriter(tmp_path).write_immutable(tmp_path / "coordinate-result.json", content.decode())
    decoded = _read_coordinate_artifact(context, item)
    assert layout.resolve_result(store, decoded).to_json() == checkpoint.to_json()


def test_v1_readers_remain_explicit_and_neither_generation_silently_reinterprets_the_other(complete_evidence):
    _, _, result, checkpoint = complete_evidence
    # An explicit expanded read view is never advertised as the actual schema2
    # ACK blob or permission to resume an older image/codec.
    expanded = PortfolioCheckpoint(
        **checkpoint.model_dump(exclude={"experiment", "schema_version"}),
        schema_version=1,
        experiment=checkpoint.experiment.to_legacy(),
    )
    legacy = PortfolioCoordinateResult(schema_version=1, checkpoint=expanded, proof=result.proof)
    old = PortfolioReport(schema_version=1, interpretation="constructed_mechanism_only", results=(legacy,))
    wire = LEGACY_FINAL_RESULT.encode(old)
    assert LEGACY_FINAL_RESULT.decode(wire).to_json() == wire
    assert LEGACY_COORDINATE_RESULT.decode(legacy.to_json()).checkpoint.to_json() == expanded.to_json()
    with pytest.raises(ValueError):
        COORDINATE_RESULT.decode(legacy.to_json())
    with pytest.raises(ValueError):
        LEGACY_COORDINATE_RESULT.decode(result.to_json())
    with pytest.raises(ValueError):
        FINAL_RESULT.decode(wire)


def test_corrupt_actual_referenced_state_never_resolves_and_does_not_rewrite_the_ack(complete_evidence):
    layout, store, result, checkpoint = complete_evidence
    path = store.root / "blobs" / (result.snapshot.component.fingerprint[7:] + ".json")
    content = path.read_bytes()
    original = store.read_head()
    try:
        path.write_bytes(content + b" ")
        with pytest.raises(ValueError):
            layout.resolve_result(store, result)
    finally:
        path.write_bytes(content)
    assert store.read_head() == original
    assert layout.resolve_result(store, result).to_json() == checkpoint.to_json()


def test_independent_terminal_outcome_golden_preserves_zero_and_a_cost_dominated_loss():
    path = Path(__file__).parent / "fixtures/v1/terminal-outcome.json"
    wire = path.read_text().strip()
    value = PortfolioTerminalOutcome.from_json(wire)
    assert value.to_json() == wire
    assert value.terminal_net_value.fraction == 98
    assert value.proportional_paid.fraction == 2
    assert value.recurring_paid.fraction == 0
    assert value.executed_full_l1_notional.fraction == 200
    assert value.terminal_net_value.fraction - 100 == -2


@pytest.mark.parametrize("quantity", [None, float("nan"), float("inf"), -float("inf")])
def test_complete_summary_does_not_turn_null_or_nonfinite_values_into_observed_zero(quantity):
    wire = json.loads((Path(__file__).parent / "fixtures/v1/terminal-outcome.json").read_text())
    wire["terminal_net_value"] = quantity
    with pytest.raises((TypeError, ValueError)):
        PortfolioTerminalOutcome.model_validate(wire)
