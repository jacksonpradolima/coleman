"""Authored contract fixtures; these declarations are not physical job evidence."""

from datetime import UTC, datetime
from pathlib import Path

from coleman.evaluation._repetition import repetition_execution_id
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.contracts import ExperimentDesign
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.projection import project_frozen
from coleman.evaluation.design.selection import AttemptLedger, CellIdentity, PhysicalAttempt, SelectionManifest
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance

FIXTURES = Path(__file__).parent / "fixtures" / "design-v1"
AVAILABLE = datetime(2026, 11, 6, 12, 2, tzinfo=UTC)
ASSEMBLED = datetime(2026, 11, 6, 12, 3, tzinfo=UTC)


def changed(value, **updates):
    """Revalidate an explicit edited test value, including unchecked-instance guards."""
    return type(value).model_validate(value.model_copy(update=updates))


def make_design(**updates):
    """Start from independent hand-authored standard-JSON input, not generated expectations."""
    design = ExperimentDesign.from_json((FIXTURES / "design.json").read_bytes())
    return changed(design, **updates) if updates else design


def make_selection(design=None, **updates):
    """Bind the exact supplied design without adding protected work to development."""
    design = make_design() if design is None else design
    selection = SelectionManifest.from_json((FIXTURES / "selection.json").read_bytes())
    return changed(
        selection,
        **{
            "design_fingerprint": design.fingerprint,
            "selected_candidate_ids": tuple(item.candidate.candidate_id for item in design.candidates),
            **updates,
        },
    )


def make_raw_panel(
    design=None,
    selection=None,
    *,
    values=None,
    statuses=None,
    states=None,
    reasons=None,
    record_times=None,
    extra_measurements=None,
    mode="produced",
    ledger=None,
    assembled_at=ASSEMBLED,
):
    """Create a complete authored panel with one declared multi-cell group per candidate.

    Unknown physical times/costs remain unknown. This helper does not execute an
    experiment, measure CPU, create extra independent worlds or prove a source
    was uninspected. Actual installed/native flows provide their own producer and
    real grouped ledger instead of presenting these fixtures as physical evidence.
    """
    from coleman.evaluation.design.panel import RawPanel

    design = make_design() if design is None else design
    selection = make_selection(design) if selection is None else selection
    frozen = project_frozen(design, selection)
    primary = design.metric(design.primary_metric)
    values, statuses, states, reasons, record_times = (
        value or {} for value in (values, statuses, states, reasons, record_times)
    )
    extra_measurements = extra_measurements or {}
    cells = []
    for candidate in design.candidates:
        name = candidate.candidate.candidate_id
        for member in design.protected_members:
            for seed in design.seeds:
                key = name, member.member_id, seed
                value = values.get(key, 0.5 if name == "baseline" else 0.75)
                record = DomainRecord(
                    schema_version=1,
                    namespace="constructed.protocol.raw",
                    run_id="authored-complete-grid",
                    execution_id=repetition_execution_id(name, seed),
                    decision_id=member.member_id,
                    data_fingerprint=design.snapshot.data_fingerprint,
                    available_at=record_times.get(key, AVAILABLE),
                    provenance=ResultProvenance(
                        producer="authored-complete-grid-fixture",
                        producer_version="1",
                        spec_fingerprint=candidate.candidate.configuration_fingerprint,
                        evaluation_fingerprint=frozen.fingerprint if mode == "produced" else "sha256:" + "e" * 64,
                    ),
                    measurements=(
                        Measurement(
                            name=primary.name,
                            value=value,
                            unit=primary.quantity.spec.semantics.unit,
                            meaning=primary.quantity.spec.semantics.meaning,
                            direction=primary.direction,
                            status=statuses.get(key, "observed" if value is not None else "unavailable"),
                        ),
                        *extra_measurements.get(key, ()),
                    ),
                )
                reason = {"reason": reasons[key]} if key in reasons else {}
                cells.append(
                    RawCell(
                        candidate_id=name,
                        member_id=member.member_id,
                        seed=seed,
                        state=states.get(key, "completed"),
                        evidence=OriginalRecordEvidence.from_record(record),
                        **reason,
                    )
                )
    if ledger is None:
        attempts = []
        if mode == "produced":
            for candidate in design.candidates:
                name = candidate.candidate.candidate_id
                outputs = tuple(
                    CellIdentity(candidate_id=name, member_id=cell.member_id, seed=cell.seed)
                    for cell in cells
                    if cell.candidate_id == name and cell.state != "skipped"
                )
                if outputs:
                    attempts.append(
                        PhysicalAttempt(
                            attempt_id="authored-group-" + name,
                            group_id="authored-group-" + name,
                            trial_id=None,
                            candidate_id=name,
                            state="completed",
                            started_at=None,
                            completed_at=None,
                            outputs=outputs,
                        )
                    )
        ledger = AttemptLedger(schema_version=1, stage="protected", attempts=tuple(attempts))
    return RawPanel(
        schema_version=1,
        design=design,
        selection=selection,
        rows=tuple(cells),
        ledger=ledger,
        mode=mode,
        assembled_at=assembled_at,
    )


def make_finance_panel(*, primary="net_value"):
    """Authored four-world protocol fixture; no market or paid-cost evidence.

    Every complete world starts with 100 USD. Baseline gross changes are
    (1, 0, -1, 0) USD; the candidate adds 1/4 USD. A common authored charge of
    1/8 USD per trade is applied to one baseline trade versus four candidate
    trades. Net candidate-minus-baseline is therefore -1/8 USD in each world,
    while the gross-based normalized loss (2-gross)/4 improves by 1/16.
    Worlds are declared independent for this synthetic construction; four are
    insufficient for the predeclared minimum of twenty. No seed or calendar
    segment is presented as extra independent empirical evidence.
    """
    import hashlib

    design = make_design()
    data = design.model_dump(mode="json")
    data["reference"] = {"id": "authored-finance-protocol", "version": "1"}
    data["scope"]["reference"] = {"id": "authored-independent-finance-worlds", "version": "1"}
    data["snapshot"]["reference"] = {"id": "authored-four-world-table", "version": "1"}
    data["snapshot"]["data_fingerprint"] = (
        "sha256:" + hashlib.sha256(b"authored-gross-1,0,-1,0;delta=1/4;charge=1/8").hexdigest()
    )
    from coleman.evaluation.protected.contracts import FrozenCandidate

    for candidate in data["candidates"]:
        name = candidate["candidate"]["candidate_id"]
        candidate["candidate"] = FrozenCandidate.from_configuration(
            name,
            {
                "method": "authored-whole-world-" + name,
                "initial_capital": {"value": 100.0, "unit": "USD", "meaning": "authored-starting-cash-v1"},
                "fixed_trade_count": 4 if name == "candidate" else 1,
            },
        ).model_dump(mode="json")
        candidate["strategy"]["accounting"] = {
            "reference": {"id": "authored-one-eighth-per-own-trade", "version": "1"},
            "definition_fingerprint": "sha256:"
            + hashlib.sha256(b"modelled_charge_USD=own_trade_count/8;net=gross-charge").hexdigest(),
        }
    template = data["members"][0]
    data["members"] = []
    for index in range(4):
        member = {
            **template,
            "member_id": f"world-{index}",
            "unit_id": f"independent-world-{index}",
            "task": {"id": "constructed-financial-accounting", "version": "1"},
            "path": {"id": f"authored-market-world-{index}", "version": "1"},
            "metadata": {
                **template["metadata"],
                "observation_id": f"source-world-{index}",
                "group_id": f"source-world-{index}",
            },
        }
        data["members"].append(member)
    definitions = (
        (
            "net_value",
            "USD",
            "authored-net-final-value-change-v1",
            "monetary_amount",
            "objective",
            "executed_objective",
            -2.5,
            2.0,
            "maximize",
        ),
        (
            "gross_value",
            "USD",
            "authored-gross-final-value-change-v1",
            "monetary_amount",
            "objective",
            "raw_outcome",
            -2.0,
            2.0,
            "maximize",
        ),
        (
            "modelled_charges",
            "USD",
            "authored-one-eighth-per-trade-charge-v1",
            "monetary_amount",
            "resource",
            "diagnostic",
            0.0,
            0.5,
            "minimize",
        ),
        (
            "normalized_loss",
            "1",
            "authored-two-minus-gross-over-four-v1",
            "normalized_loss",
            "loss",
            "surrogate",
            0.0,
            1.0,
            "minimize",
        ),
    )
    data["metrics"] = []
    for name, unit, meaning, interpretation, scalar_role, metric_role, lower, upper, direction in definitions:
        data["metrics"].append(
            {
                "name": name,
                "direction": direction,
                "role": metric_role,
                "estimand": {
                    "reference": {"id": name + "-definition", "version": "1"},
                    "definition_fingerprint": "sha256:" + hashlib.sha256(meaning.encode()).hexdigest(),
                },
                "quantity": {
                    "spec": {
                        "dimension": {"id": meaning, "version": "1"},
                        "semantics": {"unit": unit, "meaning": meaning},
                        "role": scalar_role,
                        "minimum": lower,
                        "maximum": upper,
                    },
                    "interpretation": interpretation,
                    "horizon": {"id": "one-complete-authored-market-world", "version": "1"},
                    "risk_definition": None,
                },
            }
        )
    data["primary_metric"] = primary
    chosen = next(metric for metric in data["metrics"] if metric["name"] == primary)
    data["inference"]["practical_effect"] = {"value": 0.03125, "semantics": chosen["quantity"]["spec"]["semantics"]}
    design = ExperimentDesign.model_validate(data)
    values, extras = {}, {}
    for name in ("baseline", "candidate"):
        for index, baseline_gross in enumerate((1.0, 0.0, -1.0, 0.0)):
            gross = baseline_gross + (0.25 if name == "candidate" else 0.0)
            charges = 0.5 if name == "candidate" else 0.125
            metrics = {
                "net_value": gross - charges,
                "gross_value": gross,
                "modelled_charges": charges,
                "normalized_loss": (2.0 - gross) / 4.0,
            }
            key = name, f"world-{index}", 7
            values[key] = metrics[primary]
            extras[key] = tuple(
                Measurement(
                    name=metric.name,
                    value=metrics[metric.name],
                    unit=metric.quantity.spec.semantics.unit,
                    meaning=metric.quantity.spec.semantics.meaning,
                    direction=metric.direction,
                    status="observed",
                )
                for metric in design.metrics
                if metric.name != primary
            )
    return make_raw_panel(design, values=values, extra_measurements=extras)
