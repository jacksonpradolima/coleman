"""Authored four-world accounting counterexample, using the public protocol API.

This imports constructed observations; it performs no market data acquisition,
prediction, investment, trade execution or physical-cost measurement. The target
bounds belong only to the declared toy worlds, not financial outcomes generally.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ci import declarations, imported_panel

from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.units import UnitSemantics
from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.cells import RawCell
from coleman.evaluation.design.contracts import (
    BudgetPlan,
    CandidatePlan,
    ExperimentDesign,
    InferenceRequest,
    MetricPlan,
    StrategyBindings,
    Threshold,
)
from coleman.evaluation.design.evidence import OriginalRecordEvidence
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.design.selection import SelectionManifest
from coleman.evaluation.design.source_contracts import SourceSnapshot
from coleman.evaluation.protected.contracts import FrozenCandidate
from coleman.objectives.bindings import DefinitionBinding
from coleman.results.domain import DomainRecord, Measurement


def _binding(name: str, definition: str) -> DefinitionBinding:
    """Name the exact stated toy rule without claiming authenticated execution."""
    return DefinitionBinding(
        reference=Reference(id=name, version="1"),
        definition_fingerprint=fingerprint({"schema_version": 1, "definition": definition}),
    )


def quantities() -> tuple[MetricPlan, ...]:
    """Declare net target, gross outcome, modeled own charges and learner surrogate.

    Returns
    -------
    tuple of MetricPlan
        Separate typed quantities. Constructed gross values lie in [-2,2] USD;
        own modeled charges lie in [0,1/2] USD, giving net bounds [-5/2,2].
        No physical runtime measurement or general market bound is inferred.
    """
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
    return tuple(
        MetricPlan.model_validate(
            {
                "name": name,
                "direction": direction,
                "role": metric_role,
                "estimand": _binding(name, meaning),
                "quantity": {
                    "spec": {
                        "dimension": Reference(id=meaning, version="1"),
                        "semantics": UnitSemantics(unit=unit, meaning=meaning),
                        "role": scalar_role,
                        "minimum": lower,
                        "maximum": upper,
                    },
                    "interpretation": interpretation,
                    "horizon": Reference(id="one-complete-authored-market-world", version="1"),
                },
            }
        )
        for name, unit, meaning, interpretation, scalar_role, metric_role, lower, upper, direction in definitions
    )


def observations() -> tuple[tuple[str, str, int, float, float, float, float], ...]:
    """Return eight fixed cells: candidate, world, seed, net, gross, charge and loss.

    Returns
    -------
    tuple
        Four visibly authored worlds and one repetition coordinate. Baseline
        executes one modeled trade and recipe-a four, each at 1/8 USD. These
        modeled book charges are separate from physical-attempt cost evidence.
    """
    rows = []
    for candidate in ("baseline", "recipe-a"):
        for index, baseline_gross in enumerate((1.0, 0.0, -1.0, 0.0)):
            gross = baseline_gross + (0.25 if candidate == "recipe-a" else 0.0)
            charge = 0.5 if candidate == "recipe-a" else 0.125
            rows.append((candidate, f"world-{index}", 0, gross - charge, gross, charge, (2.0 - gross) / 4.0))
    return tuple(rows)


def finance_declarations() -> tuple[ExperimentDesign, SelectionManifest]:
    """Freeze the exact public constructed arithmetic before audited import.

    Returns
    -------
    tuple of ExperimentDesign and SelectionManifest
        Net-value primary comparison, unchanged complete family, capital 100 USD,
        unknown physical compute allowance and no declared physical attempts.
        No ex-ante risk match or empirical financial admission is asserted.
    """
    rows = observations()
    metrics = quantities()
    net_rows = tuple(row[:4] for row in rows)
    design, selection = declarations(rows=net_rows, metric=metrics[0], profile="finance")
    candidates = []
    for plan in design.candidates:
        candidate = plan.candidate.candidate_id
        count = 4 if candidate == "recipe-a" else 1
        accounting = canonical_json(
            {
                "initial_capital_USD": 100.0,
                "modelled_own_trades": count,
                "charge_USD_per_own_trade": 0.125,
                "net": "gross minus own charge",
            }
        )
        strategy = StrategyBindings.model_validate(
            plan.strategy.model_copy(
                update={
                    "accounting": _binding(candidate + ".modelled-accounting", accounting),
                    "transform_law": _binding(
                        "gross-surrogate", "normalized_loss=(2-gross_value)/4; retain raw net separately"
                    ),
                }
            )
        )
        candidates.append(
            CandidatePlan(
                candidate=FrozenCandidate.from_configuration(
                    candidate,
                    {
                        "method": "authored-whole-world-" + candidate,
                        "initial_capital_USD": 100.0,
                        "modelled_own_trade_count": count,
                    },
                ),
                role=plan.role,
                strategy=strategy,
            )
        )
    capital = Scalar(
        spec=ScalarSpec(
            dimension=Reference(id="starting-capital", version="1"),
            semantics=UnitSemantics(unit="USD", meaning="authored-starting-capital-v1"),
            role="resource",
            minimum=0.0,
        ),
        value=100.0,
        status="observed",
    )
    compute = Scalar(
        spec=ScalarSpec(
            dimension=Reference(id="physical-compute-allowance", version="1"),
            semantics=UnitSemantics(unit="s", meaning="unknown-physical-compute-allowance-v1"),
            role="resource",
            minimum=0.0,
        ),
        value=None,
        status="unknown",
    )
    budget = BudgetPlan.model_validate(design.budget.model_copy(update={"allowances": (capital, compute)}))
    inference = InferenceRequest.model_validate(
        design.inference.model_copy(
            update={"practical_effect": Threshold(value=0.03125, semantics=metrics[0].quantity.spec.semantics)}
        )
    )
    snapshot = SourceSnapshot.model_validate(
        design.snapshot.model_copy(
            update={
                "data_fingerprint": fingerprint({"rows": [list(row) for row in rows], "initial_capital_USD": 100.0}),
                "interpretation": _binding(
                    "four-world-interpretation",
                    "candidate, world, seed, netUSD, grossUSD, own modeled chargeUSD, normalized gross loss",
                ),
            }
        )
    )
    design = ExperimentDesign.model_validate(
        design.model_copy(
            update={
                "candidates": tuple(candidates),
                "metrics": metrics,
                "inference": inference,
                "budget": budget,
                "snapshot": snapshot,
            }
        )
    )
    selection = SelectionManifest.model_validate(
        selection.model_copy(update={"design_fingerprint": design.fingerprint, "actual_selected_at": datetime.now(UTC)})
    )
    return design, selection


def finance_panel(design: ExperimentDesign, selection: SelectionManifest) -> RawPanel:
    """Import every original net/gross/charge/surrogate cell with no physical ledger.

    Parameters
    ----------
    design, selection : ExperimentDesign, SelectionManifest
        Exact pre-load declarations supplied to the public audit adapter.

    Returns
    -------
    RawPanel
        Complete authored imported evidence, including signed negative values,
        separate raw units and an empty physical-attempt ledger.
    """
    values = observations()
    panel = imported_panel(design, selection, rows=tuple(row[:4] for row in values))
    original_values = {(row[0], row[1], row[2]): row for row in values}
    cells = []
    for cell in panel.rows:
        row = original_values[cell.candidate_id, cell.member_id, cell.seed]
        measurements = []
        for metric, value in zip(quantities(), row[3:], strict=True):
            measurements.append(
                Measurement(
                    name=metric.name,
                    value=value,
                    unit=metric.quantity.spec.semantics.unit,
                    meaning=metric.quantity.spec.semantics.meaning,
                    direction=metric.direction,
                    status="observed",
                )
            )
        record = DomainRecord.model_validate(cell.record.model_copy(update={"measurements": tuple(measurements)}))
        cells.append(
            RawCell(
                candidate_id=cell.candidate_id,
                member_id=cell.member_id,
                seed=cell.seed,
                state=cell.state,
                evidence=OriginalRecordEvidence.from_record(record),
            )
        )
    return RawPanel.model_validate(panel.model_copy(update={"rows": tuple(cells)}))


def main() -> None:
    """Publish real protocol mechanics, with no profitability or trade claim."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("finance-protocol-output")
    design, selection = finance_declarations()
    audit = ProtocolAudit(root, limits=design.limits)
    report = audit.evaluate(
        design, selection, lambda: finance_panel(design, selection), actor="constructed-finance-author", mode="imported"
    )
    if not isinstance(report, ProtocolReport):
        raise RuntimeError("The constructed bounded net-value request unexpectedly became unsupported")
    view = audit.read_report(report.locator, actor="constructed-finance-reader")
    result = report.family.results[0]
    gross_gain, surrogate_improvement = descriptive_auxiliaries(report)
    print(
        json.dumps(
            {
                "initial_capital_USD": 100.0,
                "raw_cells": len(report.panel.rows),
                "paired_units": len(result.independent_units),
                "gross_gain_USD": gross_gain,
                "surrogate_loss_improvement": surrogate_improvement,
                "surrogate_unit": "1",
                "raw_mean_difference": result.raw_mean_difference,
                "primary_metric": design.primary_metric,
                "primary_unit": design.metric(design.primary_metric).quantity.spec.semantics.unit,
                "conclusion": result.conclusion,
                "evidence_basis": design.evidence_basis,
                "physical_attempts": len(report.panel.ledger.attempts),
                "physical_compute_status": compute_status(design),
                "new_confirmation": view.new_confirmation,
            },
            sort_keys=True,
        )
    )


def descriptive_auxiliaries(report: ProtocolReport) -> tuple[float, float]:
    """Summarize the held four-world gross and surrogate observations.

    Parameters
    ----------
    report : ProtocolReport
        Actual validated constructed report; one seed per world is declared.

    Returns
    -------
    tuple of float
        Gross candidate-minus-baseline USD and minimized surrogate improvement.
        These are descriptive means of the exact dyadic toy inputs, not new
        uncertainty estimates, hypotheses or a replacement primary target.
    """
    pairs: dict[str, dict[str, tuple[float, float]]] = {}
    for cell in report.panel.rows:
        gross = cell.record.measurement("gross_value").value
        loss = cell.record.measurement("normalized_loss").value
        if gross is None or loss is None:
            raise ValueError("The authored descriptive quantities must be observed")
        pairs.setdefault(cell.member_id, {})[cell.candidate_id] = gross, loss
    gross_gain = sum(pair["recipe-a"][0] - pair["baseline"][0] for pair in pairs.values()) / len(pairs)
    loss_gain = sum(pair["baseline"][1] - pair["recipe-a"][1] for pair in pairs.values()) / len(pairs)
    return gross_gain, loss_gain


def compute_status(design: ExperimentDesign) -> str:
    """Return the declared unknown compute allowance, without measuring runtime."""
    return next(item.status for item in design.budget.allowances if item.spec.semantics.unit == "s")


if __name__ == "__main__":
    main()
