"""Run a real chosen-feedback CI recipe experiment without TCPCI infrastructure."""

import json

from coleman.core.actions import ChoiceAction, ChoiceSpace
from coleman.core.decisions import DecisionRequest
from coleman.core.feedback import ChosenFeedback, FeedbackBatch, FeedbackRow, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.learning.discrete import DiscreteBandit, DiscreteConfig, DiscreteState


def reference(name: str) -> Reference:
    """Return a public synthetic CI identity with explicit generation.

    Parameters
    ----------
    name : str
        Caller-owned logical identity.

    Returns
    -------
    Reference
        Version-one semantic reference, without registry lookup.
    """
    return Reference(id=name, version="v1")


def configuration(profile: str = "exp3-dyadic53-v1") -> DiscreteConfig:
    """Declare two fixed CI recipes and a dimensionless fault-credit reward.

    Parameters
    ----------
    profile : str, optional
        Shipped Exp3 dyadic53 or UCB1 binary64 profile.

    Returns
    -------
    DiscreteConfig
        Complete experiment contract; no data loading or fitting occurs.
    """
    return DiscreteConfig.model_validate(
        {
            "schema_version": 1,
            "profile": profile,
            "task": reference("ci-recipe-choice"),
            "objective": reference("maximize-observed-fault-credit"),
            "transform": reference("identity-credit"),
            "space": ChoiceSpace(
                feasible_set=reference("fixed-ci-recipes"),
                members=(reference("recipe-a"), reference("recipe-b")),
            ),
            "reward": ScalarSpec(
                dimension=reference("fault-credit"),
                semantics=UnitSemantics(unit="1", meaning="observed-fault-credit-v1"),
                role="reward",
                minimum=0.0,
                maximum=1.0,
            ),
            "gamma": 0.5 if profile == "exp3-dyadic53-v1" else None,
        }
    )


def request(config: DiscreteConfig, round_number: int) -> DecisionRequest:
    """Build one caller-owned untimed opportunity in the fixed CI catalogue.

    Parameters
    ----------
    config : DiscreteConfig
        Fixed CI recipe and reward contract.
    round_number : int
        Zero-based completed-observation count, distinct from a timestamp.

    Returns
    -------
    DecisionRequest
        Complete typed next opportunity; no learning occurs.
    """
    return DecisionRequest(
        schema_version=1,
        task=config.task,
        decision_id=f"build-{round_number}",
        round=round_number,
        space=config.space,
        objective=config.objective,
    )


def observation(learner: DiscreteBandit, reward: float) -> FeedbackBatch:
    """Build only the actual chosen recipe's explicitly synthetic observation.

    Parameters
    ----------
    learner : DiscreteBandit
        Live learner with one unchanged actual chosen recipe pending.
    reward : float
        Observed dimensionless credit in [0, 1].

    Returns
    -------
    FeedbackBatch
        Original-producer-bound root observation, without applying an update.

    Raises
    ------
    ValueError
        No actual choice is pending or the reward violates its contract.
    """
    execution = learner.pending_execution
    if execution is None or not isinstance(execution.action, ChoiceAction):
        raise ValueError("An actual chosen recipe must be pending")
    config = learner.export_state().config
    return FeedbackBatch(
        schema_version=1,
        decision_fingerprint=execution.fingerprint,
        feedback_id=f"reward-{execution.proposal.request.round}",
        revision_id="root",
        supersedes=None,
        horizon=None,
        feedback=ChosenFeedback(
            row=FeedbackRow(
                member=execution.action.chosen,
                values=(
                    OutcomeValue(
                        spec=config.reward,
                        value=reward,
                        status="observed",
                        origin="synthetic",
                        transform=config.transform,
                    ),
                ),
            )
        ),
    )


def run() -> list[dict[str, object]]:
    """Compare continuous and restored actual CI recipe decisions and updates.

    Returns
    -------
    list of dict
        Human-readable receipts containing method, choices, actual propensities,
        counts and byte-identical continuation evidence. Constructed mechanics
        demonstrate no superiority, stationary inference or physical execution.
    """
    receipts: list[dict[str, object]] = []
    for profile in ("exp3-dyadic53-v1", "ucb1-binary64-v1"):
        config = configuration(profile)
        learner = DiscreteBandit(config, seed=19)
        choices: list[str] = []
        probabilities: list[float | None] = []
        for index in range(8):
            decision = learner.decide(request(config, index))
            action = decision.action
            if not isinstance(action, ChoiceAction):
                raise ValueError("Recipe experiment requires a choice")
            choices.append(action.chosen.id)
            probabilities.append(decision.sampling.probability)
            # The observation provider resolves only this chosen recipe. No
            # alternative reward table crosses the learner boundary.
            batch = observation(learner, 1.0 if action.chosen.id == "recipe-b" else 0.0)
            restored = DiscreteBandit(config, seed=999)
            restored.restore_state(DiscreteState.from_json(learner.export_state().to_json()))
            if restored.decide(request(config, index)).to_json() != decision.to_json():
                raise ValueError("Pending producer did not survive checkpoint continuation")
            learner.update(batch)
            restored.update(batch)
            if learner.export_state().to_json() != restored.export_state().to_json():
                raise ValueError("Actual chosen update differs after checkpoint continuation")
            learner = restored
        receipts.append(
            {
                "profile": profile,
                "choices": choices,
                "selection_masses": probabilities,
                "observed_counts": learner.export_state().counts,
                "restored_every_draw": True,
                "scope": "constructed-ci-chosen-feedback-mechanics",
            }
        )
    return receipts


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
