"""Exercise chosen-suite contracts using a public synthetic CI observation."""

from coleman.core.actions import ChoiceAction, ChoiceSpace
from coleman.core.capabilities import (
    Capability,
    ProblemCapabilities,
    validate_composition,
    validate_decision,
    validate_feedback,
)
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import ChosenFeedback, FeedbackBatch, FeedbackRow, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.core.units import UnitSemantics


def main() -> None:
    """Validate a fixed host proposal, actual choice and chosen-only observations."""
    unit_suite = Reference(id="suite-unit", version="v1")
    integration_suite = Reference(id="suite-integration", version="v1")
    implementation = Reference(id="host-fixed-suite", version="v1")
    space = ChoiceSpace(feasible_set=Reference(id="ci-suites", version="v1"), members=(unit_suite, integration_suite))
    credit = ScalarSpec(
        dimension=Reference(id="detected-fault-credit", version="v1"),
        semantics=UnitSemantics(unit="1", meaning="binary-detected-fault-v1"),
        role="objective",
        minimum=0,
        maximum=1,
    )
    elapsed = ScalarSpec(
        dimension=Reference(id="elapsed", version="v1"),
        semantics=UnitSemantics(unit="s", meaning="suite-elapsed-wall-time-v1"),
        role="resource",
        minimum=0,
    )
    capability = Capability(
        schema_version=1,
        implementation=implementation,
        action_kinds=("choice",),
        feedback_kinds=("chosen",),
        required_values=(credit, elapsed),
        origins=("synthetic",),
        timing_modes=("untimed",),
        supports_delayed_feedback=False,
        requires_known_receipt=False,
        requires_observed=True,
        state_owner="stateless",
        requires_sampling=False,
        sampling_requirement="not_required",
    )
    problem = ProblemCapabilities(
        schema_version=1,
        space=space,
        feedback_kind="chosen",
        values=(credit, elapsed),
        origins=("synthetic",),
        timing_mode="untimed",
        delayed_feedback=False,
        state_owners=("stateless",),
        sampling_statuses=("not_required",),
        receipt_status="unknown",
    )
    assert validate_composition(problem, capability).compatible
    request = DecisionRequest(
        schema_version=1,
        task=Reference(id="ci-suite-selection", version="v1"),
        decision_id="build-42",
        round=0,
        space=space,
        objective=Reference(id="detect-faults-under-ci-budget", version="v1"),
    )
    action = ChoiceAction(schema_version=1, space=space, chosen=unit_suite)
    sampling = SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=implementation,
        status="not_required",
        action_fingerprint=action.fingerprint,
        probability=None,
    )
    proposal = Decision(
        schema_version=1,
        request=request,
        implementation=implementation,
        action=action,
        sampling=sampling,
        state_owner="stateless",
    )
    validate_decision(proposal, capability)
    executed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=sampling,
    )
    feedback = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=executed.fingerprint,
        feedback_id="build-42-result",
        revision_id="r1",
        supersedes=None,
        horizon=None,
        feedback=ChosenFeedback(
            row=FeedbackRow(
                member=unit_suite,
                values=(
                    OutcomeValue(spec=credit, value=1, status="observed", origin="synthetic", transform=None),
                    OutcomeValue(spec=elapsed, value=8.5, status="observed", origin="synthetic", transform=None),
                ),
            )
        ),
    )
    validate_feedback(executed, feedback, capability)
    assert FeedbackBatch.from_json(feedback.to_json()).to_json() == feedback.to_json()
    print("chosen=suite-unit; credit=1; elapsed=8.5 s; unchosen=unobserved")
    print(f"logical_identity={proposal.logical_identity}")
    print(f"feedback_digest={feedback.fingerprint}")


if __name__ == "__main__":
    main()
