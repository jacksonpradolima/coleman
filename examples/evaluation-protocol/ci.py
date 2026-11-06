"""Import eight visibly constructed CI observations through the public protocol API.

This example performs no CI jobs, training, physical-cost measurement or empirical
admission. Its input table is already exposed and its two declared units are a
mechanics illustration. Real applications supply their own reviewed source and
actual records, execution groups, assumptions and source-access declarations.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.splits import TemporalObservation, TemporalPlan
from coleman.core.units import UnitSemantics
from coleman.evaluation.design.audit import ProtocolAudit
from coleman.evaluation.design.cells import RawCell, repetition_execution_id
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
from coleman.evaluation.design.limits import ProtocolLimits
from coleman.evaluation.design.panel import RawPanel
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.design.selection import AttemptLedger, SelectionManifest
from coleman.evaluation.design.source_contracts import SourceMember, SourceScope, SourceSnapshot
from coleman.evaluation.protected.contracts import ComparisonHypothesis, FrozenCandidate
from coleman.objectives.bindings import DefinitionBinding, QuantityMeaning
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance

TABLE = (
    ("baseline", "build-a", 0, 0.5),
    ("baseline", "build-a", 1, 0.5),
    ("baseline", "build-b", 0, 0.4),
    ("baseline", "build-b", 1, 0.6),
    ("recipe-a", "build-a", 0, 0.7),
    ("recipe-a", "build-a", 1, 0.9),
    ("recipe-a", "build-b", 0, 0.6),
    ("recipe-a", "build-b", 1, 0.8),
)
SEMANTICS = UnitSemantics(unit="1", meaning="constructed-fixed-suite-detection-fraction-v1")


def _reference(name: str) -> Reference:
    """Return an explicit versioned name without registration or source lookup.

    Parameters
    ----------
    name : str
        Logical identity in this public constructed example.

    Returns
    -------
    Reference
        Exact name with interpretation version 1. Performs no I/O.
    """
    return Reference(id=name, version="1")


def _definition(name: str, description: str) -> DefinitionBinding:
    """Identify a supplied definition, without authenticating its claims.

    Parameters
    ----------
    name, description : str
        Logical identity and complete stated behavior of this fixed import.

    Returns
    -------
    DefinitionBinding
        Name/version plus the canonical versioned description digest. No
        executable implementation, environment or source is inspected.
    """
    return DefinitionBinding(
        reference=_reference(name),
        definition_fingerprint=fingerprint({"schema_version": 1, "description": description}),
    )


def _candidate(name: str, role: Literal["candidate", "fixed_baseline"]) -> CandidatePlan:
    """Freeze a literal-observation recipe without executing a learner.

    Parameters
    ----------
    name : str
        Candidate identity used by the eight-row input table.
    role : {'candidate', 'fixed_baseline'}
        Explicit comparison role, independent of whether its outcome is better.

    Returns
    -------
    CandidatePlan
        Base configuration and fixed no-update/no-refit laws. Performs no I/O.
    """
    laws = {
        "implementation": "Import this candidate's exact authored table entries; execute no policy.",
        "initialization": "No learned state or owned random generator is needed for literal imports.",
        "update_law": "No model or coefficient updates.",
        "refit_law": "No refitting.",
        "transform_law": "Keep the declared raw quantity unchanged; no clipping or imputation.",
        "eligibility": "Retain the complete fixed original source-test membership.",
        "accounting": "No physical job or cost measurement is supplied by this example.",
        "stopping": "Import exactly the declared member/seed grid, including unfavorable cells.",
    }
    return CandidatePlan(
        candidate=FrozenCandidate.from_configuration(name, {"recipe": name, "source": "authored-table-v1"}),
        role=role,
        strategy=StrategyBindings(
            **{field: _definition(name + "." + field, text) for field, text in laws.items()},
            initial_state_fingerprint=None,
            update_scope="none",
        ),
    )


def declarations(
    *,
    rows: tuple[tuple[str, str, int, float], ...] = TABLE,
    metric: MetricPlan | None = None,
    profile: str = "ci",
) -> tuple[ExperimentDesign, SelectionManifest]:
    """Build the complete fixed comparison before the audited result loader.

    Parameters
    ----------
    rows : tuple, optional
        Complete visibly authored candidate/member/seed/raw-value table. The
        sibling finance example supplies four worlds and one seed.
    metric : MetricPlan or None, optional
        Typed raw primary quantity. None selects the constructed CI fraction.
    profile : str, optional
        Distinct logical source namespace; it is never rotated to hide reuse.

    Returns
    -------
    tuple of ExperimentDesign and SelectionManifest
        Bound source/quantity/recipe declarations and an empty development
        selection ledger. Actual selection time is captured here; historical
        source clocks stay in 2019. This is not proof of unseen data.
    """
    source_plan = TemporalPlan(
        train_end=datetime(2018, 1, 1, tzinfo=UTC),
        validation_end=datetime(2018, 12, 31, tzinfo=UTC),
        test_end=datetime(2019, 12, 31, tzinfo=UTC),
    )
    comparison_plan = TemporalPlan.model_validate(
        source_plan.model_copy(update={"test_end": datetime.now(UTC) + timedelta(hours=1)})
    )
    if metric is None:
        metric = MetricPlan(
            name="ci_detection_fraction",
            quantity=QuantityMeaning(
                spec=ScalarSpec(
                    dimension=_reference("fixed-suite-detection-fraction"),
                    semantics=SEMANTICS,
                    role="objective",
                    minimum=0.0,
                    maximum=1.0,
                ),
                interpretation="raw_outcome",
                horizon=_reference("one-constructed-ci-build"),
            ),
            direction="maximize",
            role="executed_objective",
            estimand=_definition("unit-mean", "Equal seeds within members, equal members within each unit."),
        )
    scope = SourceScope(
        schema_version=1,
        reference=_reference("documentation.constructed-" + profile + "-table"),
        upstream=_definition("constructed-table", "The eight literal rows printed in the public guide."),
        partition_definition=_definition("source-clocks", canonical_json(source_plan.model_dump(mode="json"))),
        prehistory="known_exposed",
    )
    members = tuple(
        SourceMember(
            member_id=member,
            task=_reference("constructed-" + profile + "-task"),
            path=_reference("constructed-" + profile + "-path-" + member),
            unit_id="unit-" + member,
            metadata=TemporalObservation(
                observation_id="source-" + member,
                group_id="source-leakage-" + member,
                event_at=datetime(2019, 1, 2, tzinfo=UTC) + timedelta(days=index),
                available_at=datetime(2019, 1, 2, 1, tzinfo=UTC) + timedelta(days=index),
            ),
        )
        for index, member in enumerate(sorted({row[1] for row in rows}))
    )
    design = ExperimentDesign(
        schema_version=1,
        reference=_reference("constructed-" + profile + "-comparison"),
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        scope=scope,
        snapshot=SourceSnapshot(
            reference=_reference("authored-input-table"),
            data_fingerprint=fingerprint(
                {"rows": [list(row) for row in rows], "metric": metric.model_dump(mode="json")}
            ),
            interpretation=_definition(
                "table-interpretation", "candidate, member, seed, declared raw binary64 quantity"
            ),
            captured_at=None,
        ),
        source_plan=source_plan,
        comparison_plan=comparison_plan,
        members=members,
        candidates=(_candidate("baseline", "fixed_baseline"), _candidate("recipe-a", "candidate")),
        seeds=tuple(sorted({row[2] for row in rows})),
        metrics=(metric,),
        primary_metric=metric.name,
        hypotheses=(
            ComparisonHypothesis(
                hypothesis_id="recipe-versus-baseline", candidate_id="recipe-a", baseline_id="baseline"
            ),
        ),
        inference=InferenceRequest(
            method="paired_bounded_mean_hoeffding_holm_v1",
            dependence="independent",
            assignment="complete_paired_units",
            analysis="fixed_final",
            practical_effect=Threshold(value=0.02, semantics=metric.quantity.spec.semantics),
            minimum_units=20,
            independent_unit="declared-constructed-path",
            independence_basis=_definition(
                "unit-assumption", "Declared distinct toy paths; no empirical independence claim."
            ),
            bounds_basis=_definition(
                "target-bounds",
                canonical_json({"minimum": metric.quantity.spec.minimum, "maximum": metric.quantity.spec.maximum}),
            ),
        ),
        budget=BudgetPlan(
            maximum_trials=0,
            maximum_physical_attempts=0,
            maximum_output_references=0,
            stopping=_definition("fixed-import", "No tuning or physical execution; import every declared cell."),
        ),
        limits=ProtocolLimits(schema_version=1),
    )
    selection = SelectionManifest(
        schema_version=1,
        design_fingerprint=design.fingerprint,
        ledger=AttemptLedger(schema_version=1, stage="development"),
        selected_candidate_ids=("baseline", "recipe-a"),
        rule=_definition("no-search", "Prespecified literal baseline and recipe; no configuration search."),
        actual_selected_at=datetime.now(UTC),
    )
    return design, selection


def imported_panel(
    design: ExperimentDesign,
    selection: SelectionManifest,
    *,
    rows: tuple[tuple[str, str, int, float], ...] = TABLE,
) -> RawPanel:
    """Return the complete authored original observations without inventing work.

    Parameters
    ----------
    design, selection : ExperimentDesign, SelectionManifest
        Exact pre-load declarations supplied to ProtocolAudit.evaluate.
    rows : tuple, optional
        Complete visibly authored original values, in the primary raw units.

    Returns
    -------
    RawPanel
        Eight original cells, with original historical availability and base
        specification identity. Empty physical attempts/costs mean absent
        measurements, never free or successfully executed jobs.
    """
    primary = design.metric(design.primary_metric)
    semantics = primary.quantity.spec.semantics
    members = {member.member_id: member for member in design.members}
    cells = []
    for candidate, member, seed, value in rows:
        record = DomainRecord(
            schema_version=1,
            namespace="constructed.documentation.raw",
            run_id="authored-table-v1",
            execution_id=repetition_execution_id(candidate, seed),
            decision_id=member,
            data_fingerprint=design.snapshot.data_fingerprint,
            available_at=members[member].metadata.available_at + timedelta(hours=1),
            provenance=ResultProvenance(
                producer="public-authored-table",
                producer_version="1",
                spec_fingerprint=design.candidate(candidate).candidate.configuration_fingerprint,
                evaluation_fingerprint=fingerprint(
                    {
                        "schema_version": 1,
                        "metric": semantics.meaning,
                        "purpose": "constructed table interpretation; no empirical experiment",
                        "data_fingerprint": design.snapshot.data_fingerprint,
                    }
                ),
            ),
            measurements=(
                Measurement(
                    name=design.primary_metric,
                    value=value,
                    unit=semantics.unit,
                    meaning=semantics.meaning,
                    direction=primary.direction,
                    status="observed",
                ),
            ),
        )
        cells.append(
            RawCell(
                candidate_id=candidate,
                member_id=member,
                seed=seed,
                state="completed",
                evidence=OriginalRecordEvidence.from_record(record),
            )
        )
    return RawPanel(
        schema_version=1,
        design=design,
        selection=selection,
        rows=tuple(cells),
        ledger=AttemptLedger(schema_version=1, stage="protected"),
        mode="imported",
        assembled_at=datetime.now(UTC),
    )


def main() -> None:
    """Publish the actual protocol result and record one new cached disclosure.

    Notes
    -----
    The first command argument selects a trusted local output root. Writes
    immutable metadata, access receipts and complete results, then prints JSON.
    Validation, capacity and local filesystem errors propagate. Repeating the
    program is a new audited import, not a fresh source or physical experiment.
    """
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("protocol-example-output")
    design, selection = declarations()
    audit = ProtocolAudit(root, limits=design.limits)
    report = audit.evaluate(
        design,
        selection,
        lambda: imported_panel(design, selection),
        actor="documentation-author",
        mode="imported",
    )
    if not isinstance(report, ProtocolReport):
        raise RuntimeError("The authored bounded request unexpectedly became unsupported")
    audit.save_report(report)  # Idempotent publication of the already-held object.
    view = audit.read_report(report.locator, actor="documentation-reader")
    result = report.family.results[0]
    print(
        json.dumps(
            {
                "raw_cells": len(report.panel.rows),
                "evidence_basis": report.panel.design.evidence_basis,
                "paired_units": len(result.independent_units),
                "raw_mean_difference": result.raw_mean_difference,
                "conclusion": result.conclusion,
                "original_confirmatory_eligible": report.confirmatory_eligible,
                "completed": audit.completed(report.locator),
                "same_original_report": view.original.fingerprint == report.fingerprint,
                "new_confirmation": view.new_confirmation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
