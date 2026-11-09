"""Compose real Hedge, a CI prefix mapping and separate finite coverage.

All observations are authored synthetic data. No physical tests are executed,
no elapsed time is measured, and no adaptive-performance claim is made.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from coleman.core.actions import MixtureAction, MixtureSpace, RankingAction, RankingSpace, SlateAction
from coleman.core.capabilities import Capability, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import AbsentFeedback, ComponentFeedback, FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.primitives import canonical_json, fingerprint
from coleman.core.quantities import Quantity
from coleman.core.sampling import SamplingRecord
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferencePayload
from coleman.learning.direct import DirectMixture, DirectMixtureState
from coleman.learning.reference import Hedge
from coleman.objectives.bindings import DefinitionBinding
from coleman.problems.tcpci import TcpciMapping, TcpciProblem, map_ranking, ranking_request
from coleman.selection import (
    CoverageCandidate,
    CoverageConstraints,
    CoverageOutcome,
    CoverageProblem,
    CoverageResult,
    SearchLimits,
    solve_coverage,
)

if TYPE_CHECKING:
    from examples.tcpci import model
else:
    _SPEC = importlib.util.spec_from_file_location("coleman_tcpci_example_inputs", Path(__file__).with_name("model.py"))
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError("Copy model.py beside this public example")
    model = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(model)

RANKING_RECIPE = DefinitionBinding(
    reference=model.ref("weighted-borda-binary64-fraction-id-ties"),
    definition_fingerprint=fingerprint(
        {
            "expert_orders": [[name, list(order)] for name, order in model.EXPERT_ORDERS],
            "score": "sum coefficient*(2-rank)",
            "tie": "ascending exact test ID",
        }
    ),
)


@dataclass(frozen=True)
class RoundEvidence:
    """Retain original coefficient feedback separately from mapped test outcomes.

    Attributes
    ----------
    coefficient_producer : ExecutedDecision
        Actual DirectMixture original offline-identity producer.
    mapping : TcpciMapping
        Original ranking, ordered slate and exact modeled-seconds accounting.
    selected_feedback : FeedbackBatch
        Synthetic verdicts only for selected tests; no implicit unselected zeros.
    full_source : FullPanel
        Explicitly authored complete source, revealed after the decision.
    expert_feedback : FeedbackBatch
        Different member universe: full expert losses bound to the original
        coefficient producer, never to the mapped test decision.
    after_state : DirectMixtureState
        Complete mathematical and producer state after the accepted update.
    """

    coefficient_producer: ExecutedDecision
    mapping: TcpciMapping
    selected_feedback: FeedbackBatch
    full_source: FullPanel
    expert_feedback: FeedbackBatch
    after_state: DirectMixtureState[ReferencePayload]


@dataclass(frozen=True)
class ExampleResult:
    """Keep the three decisions and independent unordered coverage proposal.

    Attributes
    ----------
    rounds : tuple of RoundEvidence
        Every original input, action, source, feedback and complete successor.
    coverage : CoverageResult
        Separately qualified coverage set; it is not a ranking-prefix mapping.
    """

    rounds: tuple[RoundEvidence, ...]
    coverage: CoverageResult


def _ranking_space(problem: TcpciProblem) -> RankingSpace:
    """Narrow the actual revalidated request without trusting an action union."""
    space = ranking_request(problem).space
    if not isinstance(space, RankingSpace):
        raise ValueError("The CI problem must retain a complete ranking space")
    return space


def _slate(mapping: TcpciMapping) -> SlateAction:
    """Narrow the actual mapped action before reading selected members."""
    action = mapping.executed.action
    if not isinstance(action, SlateAction):
        raise ValueError("The CI mapping must retain an ordered slate")
    return action


def rank(problem: TcpciProblem, coefficients: Decision) -> Decision:
    """Apply the declared weighted Borda rule without fitting or calibration.

    Parameters
    ----------
    problem : TcpciProblem
        Complete decision-time ranking opportunity, with no outcome labels.
    coefficients : Decision
        Actual two-expert MixtureAction from the original Hedge bridge.

    Returns
    -------
    Decision
        Complete ranking. Stored binary64 coefficients are treated as exact
        fractions for score/tie comparisons; scores are not probabilities.

    Raises
    ------
    ValueError
        Coefficients do not cover this example's exact expert catalogue.
    """
    if not isinstance(coefficients.action, MixtureAction) or coefficients.action.space.members != model.EXPERTS:
        raise ValueError("The ranking recipe requires both declared expert versions")
    weights = {weight.member.id: Fraction.from_float(weight.value) for weight in coefficients.action.weights}
    scores = {member: Fraction(0) for member in model.MEMBERS}
    for expert, order in model.EXPERT_ORDERS:
        for position, name in enumerate(order):
            scores[model.ref(name)] += weights[expert] * (len(order) - position - 1)
    action = RankingAction(
        schema_version=1,
        space=_ranking_space(problem),
        order=tuple(sorted(model.MEMBERS, key=lambda member: (-scores[member], member.id))),
    )
    return Decision(
        schema_version=1,
        request=ranking_request(problem),
        implementation=RANKING_RECIPE.reference,
        action=action,
        sampling=SamplingRecord(
            mechanism="deterministic",
            mechanism_ref=RANKING_RECIPE.reference,
            status="exact",
            action_fingerprint=action.fingerprint,
            probability=1.0,
        ),
        state_owner="stateless",
    )


def source_panel(verdicts: tuple[int, ...]) -> FullPanel:
    """Validate a complete authored binary panel after the producing decision.

    Parameters
    ----------
    verdicts : tuple of int
        Exactly three binary test-failure labels in the public input-table order.

    Returns
    -------
    FullPanel
        Synthetic test-member verdicts; availability is not asserted by a clock.

    Raises
    ------
    ValueError
        Wrong size or a nonbinary integer value, including a Boolean.
    """
    if len(verdicts) != len(model.MEMBERS) or any(
        not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1) for value in verdicts
    ):
        raise ValueError("The authored source requires three binary integer verdicts")
    return FullPanel(
        rows=tuple(
            FeedbackRow(
                member=member,
                values=(
                    OutcomeValue(
                        spec=model.VERDICT, value=value, status="observed", origin="synthetic", transform=None
                    ),
                ),
            )
            for member, value in zip(model.MEMBERS, verdicts, strict=True)
        )
    )


def selected_feedback(mapping: TcpciMapping, source: FullPanel) -> FeedbackBatch:
    """Return only permitted selected-test observations under the actual slate.

    Parameters
    ----------
    mapping : TcpciMapping
        Rechecked mapped decision whose complete fingerprint binds the batch.
    source : FullPanel
        Authored complete test source. Unselected rows stay outside this batch.

    Returns
    -------
    FeedbackBatch
        Complete selected-component coverage, or explicit absence for an empty
        slate. Unit1 verdicts remain synthetic; no physical costs are fabricated.

    Raises
    ------
    ValueError
        Source identities/versions or feedback do not match this opportunity.
    """
    if tuple(row.member for row in source.rows) != model.MEMBERS:
        raise ValueError("The source must cover the three exact test versions")
    action = mapping.executed.action
    if not isinstance(action, SlateAction):
        raise ValueError("Selected-test feedback requires the mapped ordered slate")
    selected = tuple(row for row in source.rows if row.member in action.selected)
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=mapping.executed.fingerprint,
        feedback_id="test-outcomes-" + mapping.problem.semantic.request.decision_id,
        revision_id="root",
        supersedes=None,
        horizon=mapping.problem.mapping.quantity.horizon,
        feedback=ComponentFeedback(rows=selected, completeness="complete")
        if selected
        else AbsentFeedback(reason="empty-slate"),
    )
    capability = Capability(
        schema_version=1,
        implementation=RANKING_RECIPE.reference,
        action_kinds=("ranking", "slate"),
        feedback_kinds=("component", "absent"),
        required_values=(model.VERDICT,),
        origins=("synthetic",),
        timing_modes=("untimed",),
        supports_delayed_feedback=False,
        requires_known_receipt=False,
        requires_observed=True,
        state_owner="stateless",
        requires_sampling=False,
        sampling_requirement="not_required",
    )
    if batch.horizon != mapping.problem.mapping.quantity.horizon:
        raise ValueError("Selected feedback must retain the exact CI measurement horizon")
    validate_feedback(mapping.executed, batch, capability)
    return batch


def expert_feedback(producer: ExecutedDecision, problem: TcpciProblem, source: FullPanel) -> FeedbackBatch:
    """Compute complete expert losses from a separately available full source.

    Parameters
    ----------
    producer : ExecutedDecision
        Actual original two-expert coefficient producer, before its update.
    problem : TcpciProblem
        Decision-time test catalogue and modeled-second prefix definition.
    source : FullPanel
        Complete three-test synthetic binary verdicts, supplied after decision.
        Budget-censored observations cannot substitute for this separate source.

    Returns
    -------
    FeedbackBatch
        Full expert panel: missed failures divided by the fixed three-test
        catalogue. A build without failures has genuine zero loss, not absence.

    Raises
    ------
    ValueError
        Wrong source/member/quantity, missing value or nonbinary verdict.

    Notes
    -----
    This pure counterfactual recipe does not run an expert or execute tests.
    The actual DirectMixture separately validates its original producer join.
    """
    if tuple(row.member for row in source.rows) != model.MEMBERS:
        raise ValueError("Expert loss requires every original test version")
    failures = {}
    for row in source.rows:
        if len(row.values) != 1:
            raise ValueError("Expert loss requires exactly the declared binary verdict")
        value = row.values[0]
        if (
            value.value is None
            or value.spec != model.VERDICT
            or value.origin != "synthetic"
            or not value.value.is_integer()
        ):
            raise ValueError("Expert loss needs every original synthetic failure verdict")
        failures[row.member.id] = value.value
    rows = []
    for expert, order in model.EXPERT_ORDERS:
        action = RankingAction(
            schema_version=1, space=_ranking_space(problem), order=tuple(model.ref(name) for name in order)
        )
        proposal = Decision(
            schema_version=1,
            request=ranking_request(problem),
            implementation=RANKING_RECIPE.reference,
            action=action,
            sampling=SamplingRecord(
                mechanism="deterministic",
                mechanism_ref=RANKING_RECIPE.reference,
                status="exact",
                action_fingerprint=action.fingerprint,
                probability=1.0,
            ),
            state_owner="stateless",
        )
        selected = map_ranking(problem, proposal).executed.action
        if not isinstance(selected, SlateAction):
            raise ValueError("The expert plan must yield an ordered slate")
        missed = sum(value for name, value in failures.items() if model.ref(name) not in selected.selected)
        rows.append(
            FeedbackRow(
                member=model.ref(expert),
                values=(
                    OutcomeValue(
                        spec=model.LOSS,
                        value=missed / len(model.MEMBERS),
                        status="observed",
                        origin="synthetic",
                        transform=model.IDENTITY.reference,
                    ),
                ),
            )
        )
    return FeedbackBatch(
        schema_version=1,
        decision_fingerprint=producer.fingerprint,
        feedback_id="expert-losses-" + problem.semantic.request.decision_id,
        revision_id="root",
        supersedes=None,
        horizon=model.HORIZON,
        feedback=FullPanel(rows=tuple(rows)),
    )


def coverage() -> CoverageResult:
    """Solve the separate bounded three-candidate coverage problem exactly.

    Returns
    -------
    CoverageResult
        Unordered A+C, exact modeled cost3s and covered weight1. A prefix stop
        may choose another set; this result is never relabeled as a slate.
    """
    weight_unit = UnitSemantics(unit="1", meaning="declared-capability-coverage-weight-v1")
    problem = CoverageProblem(
        schema_version=1,
        data_fingerprint=fingerprint({"scope": "authored-coverage", "table": [list(row) for row in model.TABLE]}),
        objective_semantics=weight_unit,
        candidates=tuple(
            CoverageCandidate(
                candidate_id=name, cost=Quantity(value=Decimal(seconds), semantics=model.DURATION), covers=covers
            )
            for (name, _, seconds), covers in zip(model.TABLE, (("x", "y"), ("y", "z"), ("z",)), strict=True)
        ),
        outcomes=tuple(
            CoverageOutcome(outcome_id=name, weight=Quantity(value=Decimal(value), semantics=weight_unit))
            for name, value in (("x", ".5"), ("y", ".3"), ("z", ".2"))
        ),
        constraints=CoverageConstraints(budget=Quantity(value=Decimal("3"), semantics=model.DURATION)),
    )
    return solve_coverage(
        problem,
        method="branch_bound_v1",
        limits=SearchLimits(
            deadline_seconds=None,
            max_work=128,
            max_candidates=3,
            max_outcomes=3,
            max_memberships=5,
            max_certificate_leaves=8,
        ),
    )


def run_example() -> ExampleResult:
    """Run three actual Hedge updates with separate domain and expert feedback.

    Returns
    -------
    ExampleResult
        Complete original producers, mapped slates, permitted verdicts, full
        synthetic source and expert panels, plus independent coverage evidence.

    Raises
    ------
    ValueError
        A declaration, original producer, source or solver contract is invalid.
    LearningApplicationError
        The actual learner fails to accept a complete expert panel.

    Notes
    -----
    Mutates fresh actual Hedge/DirectMixture owners. No native runner, source
    access, checkpoint store or physical test is invoked. The bounded coverage
    solver may read its monotonic guard clock; no elapsed measurement is reported.
    """
    learner = Hedge(model.configuration(), owner=model.ref("ci-example-hedge-owner"))
    bridge = DirectMixture(
        learner, task=model.ref("ci-expert-coefficients"), objective=model.OBJECTIVE.reference, horizon=model.HORIZON
    )
    rounds = []
    for ordinal, verdicts in enumerate(model.PANELS):
        request = DecisionRequest(
            schema_version=1,
            task=model.ref("ci-expert-coefficients"),
            objective=model.OBJECTIVE.reference,
            decision_id=f"coefficient-{ordinal}",
            round=ordinal,
            space=MixtureSpace(
                members=model.EXPERTS, feasible_set=model.ref("two-frozen-experts"), profile="binary64-simplex-v1"
            ),
        )
        proposal = bridge.decide(request)
        producer = next(
            item for item in bridge.export_state().pending if item.proposal.fingerprint == proposal.fingerprint
        )
        current = model.problem(ordinal)
        mapped = map_ranking(current, rank(current, proposal))
        source = source_panel(verdicts)
        observed = selected_feedback(mapped, source)
        expert = expert_feedback(producer, current, source)
        bridge.update(expert)
        rounds.append(RoundEvidence(producer, mapped, observed, source, expert, bridge.export_state()))
    return ExampleResult(tuple(rounds), coverage())


def _summary_round(row: RoundEvidence) -> dict[str, object]:
    """Project one checked typed example row only at its JSON display edge."""
    ranking = row.mapping.executed.proposal.action
    panel = row.expert_feedback.feedback
    if not isinstance(ranking, RankingAction) or not isinstance(panel, FullPanel):
        raise ValueError("Example evidence requires its original ranking and full expert panel")
    return {
        "ranking": [member.id for member in ranking.order],
        "selected": [member.id for member in _slate(row.mapping).selected],
        "modeled_cost_seconds": str(row.mapping.estimated_cost.value),
        "remaining_modeled_seconds": str(row.mapping.remaining_budget.value),
        "sampling": row.mapping.executed.sampling.status,
        "selected_feedback": row.selected_feedback.feedback.model_dump(mode="json"),
        "expert_losses": [value.values[0].value for value in panel.rows],
        "updates": row.after_state.update_count,
        "mapping_fingerprint": row.mapping.fingerprint,
        "executed_fingerprint": row.mapping.executed.fingerprint,
    }


def summary(result: ExampleResult) -> dict[str, object]:
    """Create a transport-only human summary from the retained typed evidence.

    Parameters
    ----------
    result : ExampleResult
        Completed local constructed run; no empirical or physical-cost claim.

    Returns
    -------
    dict of str to object
        JSON-compatible input/output table, three decisions, unknown physical
        elapsed time and the separate unordered coverage result.
    """
    return {
        "scope": "constructed-ci-mapping-mechanics",
        "input_table": [list(row) for row in model.TABLE],
        "modeled_budget_seconds": "3",
        "physical_elapsed_seconds": None,
        "rounds": [_summary_round(row) for row in result.rounds],
        "coverage": {
            "selected_unordered": list(result.coverage.selected_ids)
            if result.coverage.selected_ids is not None
            else None,
            "status": result.coverage.status,
            "modeled_cost_seconds": str(result.coverage.cost.value) if result.coverage.cost else None,
        },
    }


def full_report(result: ExampleResult) -> dict[str, object]:
    """Serialize complete original evidence only at the JSON output boundary.

    Parameters
    ----------
    result : ExampleResult
        All actual typed inputs/producers/feedback and complete successor state.

    Returns
    -------
    dict of str to object
        Deterministic raw evidence alongside the human summary. No artifact is
        published, no external effect is executed and no ACK is claimed.
    """
    body = summary(result)
    body["evidence"] = [
        {
            "mapping_fingerprint": row.mapping.fingerprint,
            "executed_fingerprint": row.mapping.executed.fingerprint,
            **{name: getattr(row, name).model_dump(mode="json") for name in row.__dataclass_fields__},
        }
        for row in result.rounds
    ]
    body["coverage_result"] = result.coverage.model_dump(mode="json")
    body["ranking_recipe"] = RANKING_RECIPE.model_dump(mode="json")
    return body


def main() -> None:
    """Print deterministic JSON for a fresh constructed run; optional full proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Include all original typed records and complete states")
    arguments = parser.parse_args()
    result = run_example()
    print(canonical_json(full_report(result) if arguments.full else summary(result)))


if __name__ == "__main__":
    main()
