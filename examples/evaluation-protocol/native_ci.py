"""Run real local Random/UCB replays through the generic protected protocol.

Run ``python native_ci.py ./native-output``. Two copied, visibly authored CI
histories, two seeds and two policies produce sixteen raw observations, but only
ONE fixture-family inferential unit. The 2019 source clocks are fictional; source
creation, selection, execution, availability and publication retain actual UTC.
This runs replay computation, not subject test commands or cloud CI jobs.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl

from coleman.api import run
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.splits import TemporalObservation, TemporalPlan
from coleman.core.units import UnitSemantics
from coleman.corpus import tiny_ci_reference
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
from coleman.evaluation.design.projection import project_frozen
from coleman.evaluation.design.reports import ProtocolReport
from coleman.evaluation.design.selection import AttemptLedger, CellIdentity, PhysicalAttempt, SelectionManifest
from coleman.evaluation.design.source_contracts import SourceMember, SourceScope, SourceSnapshot
from coleman.evaluation.protected.contracts import ComparisonHypothesis, FrozenCandidate
from coleman.objectives.bindings import DefinitionBinding, QuantityMeaning
from coleman.pilot import SECONDS, PilotBuild, PilotTest
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance
from coleman.selection import Quantity
from coleman.spec.models import RunSpec

SEEDS = (11, 42)
POLICIES = ("Random", "UCB")
SEMANTICS = UnitSemantics(unit="1", meaning="verdict-napfd-full-universe-v1")


def definition(name: str, description: str) -> DefinitionBinding:
    """Identify a fixed public definition without claiming authentication.

    Parameters
    ----------
    name, description : str
        Logical identity and complete stated behavior.

    Returns
    -------
    DefinitionBinding
        Versioned identity with a deterministic description digest. No I/O.
    """
    return DefinitionBinding(
        reference=Reference(id=name, version="1"), definition_fingerprint=fingerprint({"description": description})
    )


def inputs(root: Path) -> tuple[dict[str, list[dict]], tuple[SourceMember, ...]]:
    """Write two copied authored histories with explicitly fictional old dates.

    Parameters
    ----------
    root : pathlib.Path
        New owned output directory. Writes input Parquet below ``inputs``.

    Returns
    -------
    tuple of dict and tuple of SourceMember
        Native table values and complete source membership. Integer durations
        are modeled seconds, not measured execution times. The four builds all
        belong to one constructed family; seeds and project names add no units.
    """
    data: dict[str, list[dict]] = {}
    members = []
    for project in ("constructed_ci_a", "constructed_ci_b"):
        rows = []
        for build in (1, 2):
            event = datetime(2019, 1, 4 + build, tzinfo=UTC)
            for name, duration, verdict in (("unit", 1, 0), ("integration", 2, 1), ("system", 4, build % 2)):
                if build == 2:
                    name = {"unit": "renamed-unit", "system": "contract"}.get(name, name)
                rows.append(
                    {
                        "BuildId": build,
                        "Name": name,
                        "Duration": float(duration),
                        "CalcPrio": 0,
                        "LastRun": event.date().isoformat(),
                        "Verdict": verdict,
                    }
                )
            member = f"{project}/build-{build}"
            members.append(
                SourceMember(
                    member_id=member,
                    task=Reference(id=project, version="1"),
                    path=Reference(id=project + ".history", version="1"),
                    unit_id="constructed-fixture-family",
                    metadata=TemporalObservation(
                        observation_id=member,
                        group_id=member,
                        event_at=event,
                        available_at=event + timedelta(hours=1),
                    ),
                )
            )
        data[project] = rows
        directory = root / "inputs" / project
        directory.mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(directory / "features-engineered.parquet")
    return data, tuple(members)


def specifications(root: Path, data_fingerprint: str) -> dict[tuple[str, int], RunSpec]:
    """Resolve source-bound native runs before any protected result is loaded.

    Parameters
    ----------
    root : pathlib.Path
        Owned source and native artifact root.
    data_fingerprint : str
        Exact authored table identity, included in native checkpoint settings.

    Returns
    -------
    dict
        Four RunSpec values. Calls no runner and creates no artifacts. Each run
        uses a local pool of one and the same declared 50-percent duration budget.
    """
    return {
        (policy, seed): RunSpec.model_validate(
            {
                "execution": {"independent_executions": 1, "parallel_pool_size": 1, "verbose": False, "seed": seed},
                "experiment": {
                    "datasets_dir": str(root / "inputs"),
                    "datasets": ["constructed_ci_a", "constructed_ci_b"],
                    "policies": [policy],
                    "rewards": ["RNFail"],
                    "budget": {"mode": "ratio", "values": [0.5]},
                },
                "results": {"out_dir": str(root / "native" / policy / str(seed)), "top_k_prioritization": 3},
                "algorithm": {"ucb": {"rnfail": {"c": 1.0}}},
                "extensions": {"corpus_source": {"data_fingerprint": data_fingerprint}},
            }
        )
        for policy in POLICIES
        for seed in SEEDS
    }


def declarations(
    data: dict[str, list[dict]],
    members: tuple[SourceMember, ...],
    specs: dict[tuple[str, int], RunSpec],
) -> tuple[ExperimentDesign, SelectionManifest]:
    """Freeze the native laws, original membership and actual analysis deadline.

    Parameters
    ----------
    data, members, specs : dict, tuple of SourceMember, dict
        Exact newly created public fixture, membership and resolved native runs.

    Returns
    -------
    tuple of ExperimentDesign and SelectionManifest
        Fixed-law comparison and actual selection time. Source creation time is
        current UTC. The known authored source is explicitly already exposed;
        the supplied declarations do not establish empirical independence.
    """
    source = TemporalPlan(
        train_end=datetime(2019, 1, 2, 23, 59, 59, tzinfo=UTC),
        validation_end=datetime(2019, 1, 4, 23, 59, 59, tzinfo=UTC),
        test_end=datetime(2019, 1, 7, 23, 59, 59, tzinfo=UTC),
    )
    candidates = []
    for policy in POLICIES:
        laws = {
            "implementation": "Native "
            + policy
            + " prioritization; RNFail reward, duration budget ratio 0.5, UCB c=1.",
            "initialization": "Fresh native state for each seed and source history; seeds 11 and 42.",
            "update_law": "Native policy update after each build's observed verdicts; retain its fixed causal law.",
            "refit_law": "No external refit or policy selection from protected outcomes.",
            "transform_law": "Retain native NAPFD unchanged, verified against the independent rational reference.",
            "eligibility": "All four original protected build members, in each history's chronological native order.",
            "accounting": "One API invocation per policy/seed; actual self-process CPU and wall seconds per invocation.",
            "stopping": "Exactly two seeds and four builds per policy, including unfavorable outcomes; no early stopping.",
        }
        candidates.append(
            CandidatePlan(
                candidate=FrozenCandidate.from_configuration(
                    policy, {"resolved_runs": [specs[policy, seed].model_dump(mode="json") for seed in SEEDS]}
                ),
                role="fixed_baseline" if policy == "Random" else "candidate",
                strategy=StrategyBindings(
                    **{key: definition(policy + "." + key, text) for key, text in laws.items()},
                    initial_state_fingerprint=None,
                    update_scope="within_unit",
                ),
            )
        )
    metric = MetricPlan(
        name="verdict_napfd",
        quantity=QuantityMeaning(
            spec=ScalarSpec(
                dimension=Reference(id="ci.verdict-napfd", version="1"),
                semantics=SEMANTICS,
                role="objective",
                minimum=0.0,
                maximum=1.0,
            ),
            interpretation="raw_outcome",
            horizon=Reference(id="one-ci-build-full-test-universe", version="1"),
        ),
        direction="maximize",
        role="executed_objective",
        estimand=definition("equal-family-means", "Equal seeds within builds, equal builds within the fixture family."),
    )
    design = ExperimentDesign(
        schema_version=1,
        reference=Reference(id="public.native-ci-protocol", version="1"),
        evidence_basis="constructed_mechanism",
        source_admission_fingerprint=None,
        scope=SourceScope(
            schema_version=1,
            reference=Reference(id="public.constructed-native-ci-source", version="1"),
            upstream=definition(
                "public-ci-fixture",
                "Owned copy of the authored mechanism in examples/corpus-evaluation/ci.py; old dates are fiction.",
            ),
            partition_definition=definition("source-calendar", canonical_json(source.model_dump(mode="json"))),
            prehistory="known_exposed",
        ),
        snapshot=SourceSnapshot(
            reference=Reference(id="authored-native-ci-input", version="1"),
            data_fingerprint=fingerprint(data),
            interpretation=definition(
                "native-table", "BuildId, Name, Duration in modeled seconds, CalcPrio, LastRun and binary Verdict."
            ),
            captured_at=datetime.now(UTC),
        ),
        source_plan=source,
        comparison_plan=TemporalPlan.model_validate(
            source.model_copy(update={"test_end": datetime.now(UTC) + timedelta(hours=1)})
        ),
        members=members,
        candidates=tuple(candidates),
        seeds=SEEDS,
        metrics=(metric,),
        primary_metric=metric.name,
        hypotheses=(ComparisonHypothesis(hypothesis_id="UCB-Random", candidate_id="UCB", baseline_id="Random"),),
        inference=InferenceRequest(
            method="paired_bounded_mean_hoeffding_holm_v1",
            dependence="independent",
            assignment="complete_paired_units",
            analysis="fixed_final",
            practical_effect=Threshold(value=0.05, semantics=SEMANTICS),
            minimum_units=20,
            independent_unit="declared-authored-fixture-family",
            independence_basis=definition(
                "one-family", "All copied histories remain one authored family; no empirical independence claim."
            ),
            bounds_basis=definition("NAPFD-bounds", "Complete binary verdict NAPFD lies in [0,1]."),
        ),
        budget=BudgetPlan(
            maximum_trials=0,
            maximum_physical_attempts=4,
            maximum_output_references=16,
            stopping=definition(
                "fixed-replays", "Run exactly Random/UCB by seeds 11/42, without tuning or optional stopping."
            ),
        ),
        limits=ProtocolLimits(schema_version=1),
    )
    selection = SelectionManifest(
        schema_version=1,
        design_fingerprint=design.fingerprint,
        ledger=AttemptLedger(schema_version=1, stage="development"),
        selected_candidate_ids=POLICIES,
        rule=definition("no-tuning", "Prespecified public native Random and UCB c=1; no protected-data choice."),
        actual_selected_at=datetime.now(UTC),
    )
    return design, selection


def check_metrics(source: list[dict], member: SourceMember, native_row: dict) -> None:
    """Verify native NAPFD and APFDc against the existing rational CI reference.

    Parameters
    ----------
    source : list of dict
        Complete authored tests for this build, with modeled integer seconds.
    member : SourceMember
        Original historical source clocks.
    native_row : dict
        One actual native Parquet row. ``cost`` means dimensionless APFDc here,
        never paid money or physical CPU time.

    Raises
    ------
    ValueError
        Actual native NAPFD/APFDc disagrees with the reference beyond 1e-12,
        or the complete constructed failing suite unexpectedly lacks a metric.
    """
    tests = tuple(
        PilotTest(
            test_id=row["Name"],
            duration=Quantity(value=Decimal(str(row["Duration"])), semantics=SECONDS),
            outcome="failed" if row["Verdict"] else "passed",
        )
        for row in source
    )
    build = PilotBuild(
        build_id=member.member_id,
        decided_at=member.metadata.event_at,
        outcomes_available_at=member.metadata.available_at,
        tests=tests,
        baseline_order=tuple(row["Name"] for row in source),
    )
    budget = Quantity(value=sum((test.duration.value for test in tests), start=Decimal(0)) / 2, semantics=SECONDS)
    reference = tiny_ci_reference(build, tuple(json.loads(native_row["prioritization_order_top_k"])), budget)
    for observed, expected in (
        (native_row["fitness"], reference.napfd.value),
        (native_row["cost"], reference.apfdc.value),
    ):
        if expected is None or not math.isclose(observed, float(expected), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Actual native metric differs from the independent rational reference")


def produce(
    root: Path,
    data: dict[str, list[dict]],
    specs: dict[tuple[str, int], RunSpec],
    design: ExperimentDesign,
    selection: SelectionManifest,
) -> RawPanel:
    """Execute four actual local replays only after both audit reservations.

    Parameters
    ----------
    root, data, specs : pathlib.Path, dict, dict
        Owned output root, complete authored source and resolved runs.
    design, selection : ExperimentDesign, SelectionManifest
        Exact declarations already persisted by ProtocolAudit.

    Returns
    -------
    RawPanel
        Sixteen original DomainRecords and four actual API-invocation groups.
        Each group retains measured wall/self-process CPU seconds once. Writes
        native artifacts/checkpoints, original producer JSON and process receipts.

    Raises
    ------
    ValueError, OSError
        Native output, independent metric, original identity or local I/O fails.
        Audit reservations survive failure; this function publishes no report.
    """
    frozen = project_frozen(design, selection)
    members = {member.member_id: member for member in design.protected_members}
    cells, attempts, executions, originals = [], [], [], []
    producer_root = root / "producer-records"
    producer_root.mkdir()
    for (policy, seed), spec in specs.items():
        started_at, wall, cpu = datetime.now(UTC), time.perf_counter(), time.process_time()
        native = run(spec)
        wall_seconds, cpu_seconds = time.perf_counter() - wall, time.process_time() - cpu
        completed_at = datetime.now(UTC)
        if native.artifacts_dir is None:
            raise ValueError("Actual native run did not publish its required artifacts")
        rows = list(
            pl.read_parquet(str(Path(native.artifacts_dir) / "results/**/*.parquet"), hive_partitioning=True).iter_rows(
                named=True
            )
        )
        if len(rows) != len(members):
            raise ValueError("Actual native run omitted or added a required build")
        outputs = []
        for raw in rows:
            member_id = f"{raw['scenario']}/build-{raw['step']}"
            member = members[member_id]
            check_metrics([row for row in data[raw["scenario"]] if row["BuildId"] == raw["step"]], member, raw)
            record = DomainRecord(
                schema_version=1,
                namespace="ci.protocol.native.raw",
                run_id=frozen.fingerprint,
                execution_id=repetition_execution_id(policy, seed),
                decision_id=member_id,
                data_fingerprint=design.snapshot.data_fingerprint,
                available_at=datetime.now(UTC),
                provenance=ResultProvenance(
                    producer="public-native-ci-projection",
                    producer_version="1",
                    spec_fingerprint=design.candidate(policy).candidate.configuration_fingerprint,
                    evaluation_fingerprint=frozen.fingerprint,
                    source_revision=fingerprint(raw),
                ),
                measurements=(
                    Measurement(
                        name=design.primary_metric,
                        unit=SEMANTICS.unit,
                        meaning=SEMANTICS.meaning,
                        direction="maximize",
                        status="observed",
                        value=raw["fitness"],
                    ),
                ),
            )
            target = producer_root / (record.identity_digest.removeprefix("sha256:") + ".json")
            target.write_text(record.to_json() + "\n")
            originals.append({"file": str(target), "key": record.logical_key, "digest": record.digest})
            cells.append(
                RawCell(
                    candidate_id=policy,
                    member_id=member_id,
                    seed=seed,
                    state="completed",
                    evidence=OriginalRecordEvidence.from_record(record),
                )
            )
            outputs.append(CellIdentity(candidate_id=policy, member_id=member_id, seed=seed))
        attempt_id = f"native-{policy}-{seed}"
        cost = DomainRecord(
            schema_version=1,
            namespace="ci.protocol.native.process",
            run_id=frozen.fingerprint,
            execution_id=attempt_id,
            decision_id="one-local-api-invocation",
            data_fingerprint=design.snapshot.data_fingerprint,
            available_at=datetime.now(UTC),
            provenance=ResultProvenance(
                producer="python-process-clock",
                producer_version="1",
                spec_fingerprint=design.candidate(policy).candidate.configuration_fingerprint,
                evaluation_fingerprint=frozen.fingerprint,
            ),
            measurements=(
                Measurement(
                    name="local_wall_seconds",
                    value=wall_seconds,
                    unit="s",
                    meaning="perf-counter-api-invocation-v1",
                    direction="minimize",
                    status="observed",
                ),
                Measurement(
                    name="self_cpu_seconds",
                    value=cpu_seconds,
                    unit="s",
                    meaning="process-time-api-invocation-excludes-child-cpu-v1",
                    direction="minimize",
                    status="observed",
                ),
            ),
        )
        attempts.append(
            PhysicalAttempt(
                attempt_id=attempt_id,
                group_id=attempt_id,
                trial_id=None,
                candidate_id=policy,
                state="completed",
                started_at=started_at,
                completed_at=completed_at,
                outputs=tuple(outputs),
                costs=(OriginalRecordEvidence.from_record(cost),),
            )
        )
        executions.append(
            {
                "candidate": policy,
                "seed": seed,
                "driver_pid": os.getpid(),
                "native_run_id": native.run_id,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "wall_seconds": wall_seconds,
                "self_cpu_seconds": cpu_seconds,
                "native_artifacts": native.artifacts_dir,
            }
        )
    (root / "actual-executions.json").write_text(json.dumps(executions, indent=2) + "\n")
    (root / "producer-manifest.json").write_text(json.dumps(originals, indent=2) + "\n")
    return RawPanel(
        schema_version=1,
        design=design,
        selection=selection,
        rows=tuple(cells),
        ledger=AttemptLedger(schema_version=1, stage="protected", attempts=tuple(attempts)),
        mode="produced",
        assembled_at=datetime.now(UTC),
    )


def main() -> None:
    """Run the real example, verify original wires and disclose its cached result.

    Notes
    -----
    The optional first argument is a trusted local output root; each invocation
    creates a distinct owned session. Exceptions propagate. Native result records
    retain their first producer wire; the new enclosing serializer must preserve
    it. A cached disclosure executes no replay and never creates confirmation.
    """
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "native-protocol-output").resolve()
    root = output / datetime.now(UTC).strftime("session-%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True)
    data, members = inputs(root)
    specs = specifications(root, fingerprint(data))
    design, selection = declarations(data, members, specs)
    audit = ProtocolAudit(root / "protocol", limits=design.limits)
    report = audit.evaluate(
        design,
        selection,
        lambda: produce(root, data, specs, design, selection),
        actor="public-native-example",
        mode="produced",
    )
    if not isinstance(report, ProtocolReport):
        raise RuntimeError("The prespecified native bounded request unexpectedly became unsupported")
    published = audit.save_report(report)
    before = published.read_bytes()
    view = audit.read_report(report.locator, actor="public-native-cached-reader")
    recovered = view.original
    originals = {tuple(item["key"]): item for item in json.loads((root / "producer-manifest.json").read_text())}
    preserved = all(
        row.record.digest == originals[row.record.logical_key]["digest"]
        and (row.record.to_json() + "\n").encode() == Path(originals[row.record.logical_key]["file"]).read_bytes()
        for row in recovered.panel.rows
    )
    if not preserved or published.read_bytes() != before:
        raise RuntimeError("Original producer wire or completed report changed during publication/disclosure")
    result = report.family.results[0]
    print(
        json.dumps(
            {
                "raw_cells": len(report.panel.rows),
                "paired_units": len(result.independent_units),
                "native_run_invocations": len(report.panel.ledger.attempts),
                "local_driver_processes": 1,
                "external_ci_jobs_claimed": False,
                "subject_test_commands_claimed": False,
                "source_dates_are_authored_fiction": True,
                "evidence_basis": design.evidence_basis,
                "conclusion": result.conclusion,
                "raw_mean_difference": result.raw_mean_difference,
                "original_confirmatory_eligible": report.confirmatory_eligible,
                "original_producer_wire_preserved": preserved,
                "new_confirmation": view.new_confirmation,
                "completed": audit.completed(report.locator),
                "report_file": str(published),
                "session": str(root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
