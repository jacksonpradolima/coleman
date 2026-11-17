"""Independent arithmetic, sampling and real chosen-only continuation oracles."""

import concurrent.futures
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from random import Random
from typing import cast

import pytest

from coleman.core.actions import ChoiceAction
from coleman.core.capabilities import ProblemCapabilities, validate_feedback
from coleman.core.decisions import Decision
from coleman.core.feedback import ChosenFeedback, FeedbackRow, FullPanel, OutcomeValue
from coleman.learning.discrete import DiscreteBandit, DiscreteConfig, DiscreteState
from coleman.learning.discrete.contracts import GRID_SIZE
from coleman.learning.discrete.sampling import exp3_tickets, ticket_index
from coleman.testing.decision_contracts import ConformanceCase, check_decision_contracts
from examples.composition.mab_only import HostMAB, capability, cases
from examples.discrete.ci import configuration, observation, reference, request, run


def _choice(decision: Decision) -> ChoiceAction:
    """Narrow the real selected action without bypassing a runtime assertion."""
    assert isinstance(decision.action, ChoiceAction)
    return decision.action


@pytest.mark.parametrize("profile", ("exp3-dyadic53-v1", "ucb1-binary64-v1"))
def test_real_shipped_and_independent_host_shared_conformance(profile):
    config = configuration(profile)

    def factory():
        return DiscreteBandit(config)

    problem = ProblemCapabilities(
        schema_version=1,
        space=config.space,
        feedback_kind="chosen",
        values=(config.reward,),
        origins=("synthetic",),
        timing_mode="untimed",
        delayed_feedback=False,
        sampling_statuses=("exact",),
        receipt_status="unknown",
        state_owners=("learner",),
    )
    fixture = tuple(
        ConformanceCase(
            problem=problem,
            request=request(config, index),
            outcomes=tuple(
                FeedbackRow(
                    member=member,
                    values=(
                        OutcomeValue(
                            spec=config.reward,
                            value=value,
                            status="observed",
                            origin="synthetic",
                            transform=config.transform,
                        ),
                    ),
                )
                for member, value in zip(config.space.members, (0.0, 1.0), strict=True)
            ),
        )
        for index in range(8)
    )
    report = check_decision_contracts(factory, factory().capability, fixture)
    assert len(report.decisions) == 8
    assert all(isinstance(batch.feedback, ChosenFeedback) for batch in report.feedback)
    assert len(check_decision_contracts(HostMAB, capability(), cases()).feedback) == 8


def test_exp3_independent_paper_arithmetic_and_true_finite_mass():
    config = configuration()
    learner = DiscreteBandit(config, seed=19)
    draw = Random(19).getrandbits(53)
    first = learner.decide(request(config, 0))
    chosen = 0 if draw < 2**52 else 1
    assert _choice(first).chosen == config.space.members[chosen]
    assert first.sampling.probability == 0.5
    learner.update(observation(learner, 0.5))
    state = learner.export_state()
    # Figure1: xhat=.5/.5=1, eta=gamma/K=.25, log ratio=.25.
    assert state.log_weights[chosen] == 0.0
    assert state.log_weights[1 - chosen] == -0.25
    assert state.counts[chosen] == 1
    assert state.counts[1 - chosen] == 0
    second = learner.decide(request(config, 1))
    pending = learner.export_state().pending
    weights = [Fraction.from_float(math.exp(value)) for value in state.log_weights]
    ideal = [Fraction(1, 2) * weight / sum(weights) + Fraction(1, 4) for weight in weights]
    assert pending is not None
    for count, target in zip(pending.tickets, ideal, strict=True):
        assert abs(Fraction(count, GRID_SIZE) - target) < Fraction(1, GRID_SIZE)
    selected = config.space.members.index(_choice(second).chosen)
    probability = second.sampling.probability
    assert probability is not None
    assert Fraction.from_float(probability) == Fraction(pending.tickets[selected], GRID_SIZE)
    old_logs = state.log_weights
    learner.update(observation(learner, 1.0))
    expected = list(old_logs)
    expected[selected] += 0.25 / probability
    center = max(expected)
    assert learner.export_state().log_weights == tuple(value - center for value in expected)


def test_ucb_counts_only_observations_and_canonical_ties():
    config = configuration("ucb1-binary64-v1")
    learner = DiscreteBandit(config)
    for index, expected in enumerate(("recipe-a", "recipe-b", "recipe-a")):
        decision = learner.decide(request(config, index))
        assert _choice(decision).chosen.id == expected
        assert decision.sampling.mechanism == "deterministic"
        assert decision.sampling.probability == 1
        assert sum(learner.export_state().counts) == index
        learner.update(observation(learner, 0.0))
    state = learner.export_state()
    assert state.counts == (2, 1)
    # Equal means: sqrt(2 log(4)/1) > sqrt(2 log(4)/2).
    assert _choice(learner.decide(request(config, 3))).chosen.id == "recipe-b"


@pytest.mark.parametrize("profile", ("exp3-dyadic53-v1", "ucb1-binary64-v1"))
def test_every_draw_and_update_boundary_roundtrips_and_duplicate_delivery(profile):
    config = configuration(profile)
    learner = DiscreteBandit(config)
    for index in range(12):
        before = learner.export_state()
        proposal = learner.decide(request(config, index))
        after = learner.export_state()
        assert learner.decide(request(config, index)).to_json() == proposal.to_json()
        assert learner.export_state() == after
        restored = DiscreteBandit(config, seed=123)
        restored.restore_state(DiscreteState.from_json(before.to_json()))
        assert restored.decide(request(config, index)).to_json() == proposal.to_json()
        restored.restore_state(DiscreteState.from_json(after.to_json()))
        batch = observation(learner, (index % 3) / 2)
        learner.update(batch)
        restored.update(batch)
        assert restored.export_state().to_json() == learner.export_state().to_json()
        completed = learner.export_state()
        with pytest.raises(ValueError, match="No original"):
            learner.update(batch)
        assert learner.export_state() == completed
        assert DiscreteState.from_json(completed.to_json()).fingerprint == completed.fingerprint


@pytest.mark.parametrize(
    "change", ({"delay_rounds": 1}, {"supersedes": "older"}, {"horizon": reference("unsupported")})
)
def test_unsupported_feedback_leaves_original_state_intact(change):
    learner = DiscreteBandit(configuration())
    learner.decide(request(configuration(), 0))
    batch = observation(learner, 0.0)
    before = learner.export_state()
    unsupported = batch.model_copy(update=change)
    with pytest.raises(ValueError):
        learner.update(unsupported)
    assert learner.export_state() == before


@pytest.mark.parametrize(
    "status,value",
    (
        ("unknown", None),
        ("unavailable", None),
        ("not_applicable", None),
        ("observed", -0.1),
        ("observed", float("nan")),
        ("observed", float("inf")),
    ),
)
def test_missing_negative_nonfinite_reward_cannot_count_as_observed(status, value):
    learner = DiscreteBandit(configuration())
    learner.decide(request(configuration(), 0))
    batch = observation(learner, 0.0)
    before = learner.export_state()
    assert isinstance(batch.feedback, ChosenFeedback)
    row = batch.feedback.row
    invalid = batch.model_copy(
        update={
            "feedback": ChosenFeedback.model_construct(
                row=row.model_copy(
                    update={"values": (row.values[0].model_copy(update={"status": status, "value": value}),)}
                )
            )
        }
    )
    with pytest.raises(ValueError):
        learner.update(invalid)
    assert learner.export_state() == before


def test_hidden_panel_wrong_chosen_transform_and_units_rejected():
    config = configuration()
    learner = DiscreteBandit(config)
    learner.decide(request(config, 0))
    batch = observation(learner, 0.0)
    before = learner.export_state()
    assert isinstance(batch.feedback, ChosenFeedback)
    row = batch.feedback.row
    value = row.values[0]
    alternatives = (
        FullPanel(rows=(row,)),
        ChosenFeedback(
            row=row.model_copy(
                update={"member": next(member for member in config.space.members if member != row.member)}
            )
        ),
        ChosenFeedback(
            row=row.model_copy(
                update={"values": (value.model_copy(update={"transform": reference("other-transform")}),)}
            )
        ),
        ChosenFeedback(
            row=row.model_copy(
                update={
                    "values": (
                        value.model_copy(
                            update={
                                "spec": value.spec.model_copy(
                                    update={"semantics": value.spec.semantics.model_copy(update={"unit": "s"})}
                                )
                            }
                        ),
                    )
                }
            )
        ),
    )
    for feedback in alternatives:
        incompatible = batch.model_copy(update={"feedback": feedback})
        with pytest.raises(ValueError):
            learner.update(incompatible)
        assert learner.export_state() == before


@pytest.mark.parametrize("profile", ("exp3-dyadic53-v1", "ucb1-binary64-v1"))
def test_compatible_narrower_reward_bounds_and_unrelated_diagnostics_preserve_update(profile):
    config = configuration(profile)
    learner = DiscreteBandit(config)
    expected = DiscreteBandit(config)
    learner.decide(request(config, 0))
    expected.decide(request(config, 0))
    batch = observation(learner, 0.5)
    assert isinstance(batch.feedback, ChosenFeedback)
    original = batch.feedback.row.values[0]
    narrowed = original.model_copy(update={"spec": config.reward.model_copy(update={"minimum": 0.25, "maximum": 0.75})})
    diagnostic = original.model_copy(
        update={
            "spec": config.reward.model_copy(
                update={"dimension": reference("a-diagnostic"), "role": "diagnostic", "minimum": None, "maximum": None}
            ),
            "value": -123.0,
        }
    )
    batch = batch.model_copy(
        update={
            "feedback": ChosenFeedback(row=batch.feedback.row.model_copy(update={"values": (diagnostic, narrowed)}))
        }
    )
    execution = learner.pending_execution
    assert execution is not None
    # The existing core accepts schema containment and unrelated diagnostics.
    validate_feedback(execution, batch, learner.capability)
    learner.update(batch)
    expected.update(observation(expected, 0.5))
    assert learner.export_state().to_json() == expected.export_state().to_json()


def test_unknown_versions_configuration_restore_and_pending_tamper_reject():
    learner = DiscreteBandit(configuration())
    learner.decide(request(configuration(), 0))
    state = learner.export_state()
    assert state.pending is not None
    data = json.loads(state.to_json())
    for version in (0, 2, True, "1"):
        unsupported = json.dumps({**data, "schema_version": version})
        with pytest.raises(ValueError):
            DiscreteState.from_json(unsupported)
    for altered in (
        state.model_copy(update={"runtime": (2, 7)}),
        state.model_copy(update={"config": state.config.model_copy(update={"gamma": 0.25})}),
        state.model_copy(update={"pending": state.pending.model_copy(update={"draw": None})}),
        state.model_copy(
            update={"pending": state.pending.model_copy(update={"tickets": (GRID_SIZE // 4, 3 * GRID_SIZE // 4)})}
        ),
    ):
        with pytest.raises(ValueError):
            learner.restore_state(altered)
        assert learner.export_state() == state


@pytest.mark.parametrize("gamma", (0.0, -1.0, 2.0, float("nan"), float("inf")))
def test_unsupported_exploration_gamma(gamma):
    unsupported = configuration().model_copy(update={"gamma": gamma})
    with pytest.raises(ValueError):
        DiscreteConfig.model_validate(unsupported)


@pytest.mark.parametrize(
    "minimum,maximum",
    (
        (math.nextafter(0.0, 1.0), 1.0),
        (math.nextafter(0.0, -1.0), 1.0),
        (0.0, math.nextafter(1.0, 0.0)),
        (0.0, math.nextafter(1.0, 2.0)),
        (None, 1.0),
        (0.0, None),
    ),
)
def test_reward_schema_requires_exact_endpoints_without_tolerance(minimum, maximum):
    config = configuration()
    reward = config.reward.model_copy(update={"minimum": minimum, "maximum": maximum})
    unsupported = config.model_copy(update={"reward": reward})
    with pytest.raises(ValueError, match="bounds"):
        DiscreteConfig.model_validate(unsupported)


def test_ticket_endpoints_frequencies_and_importance_identity():
    tickets = (GRID_SIZE // 4, GRID_SIZE // 4, GRID_SIZE // 2)
    assert [
        ticket_index(tickets, draw) for draw in (0, GRID_SIZE // 4 - 1, GRID_SIZE // 4, GRID_SIZE // 2, GRID_SIZE - 1)
    ] == [0, 0, 1, 2, 2]
    rng = Random(463)
    counts = [0, 0, 0]
    size = 40_000
    for _ in range(size):
        counts[ticket_index(tickets, rng.getrandbits(53))] += 1
    # Hoeffding union bound <= 6 exp(-2*n*.02**2) < 8e-14 under
    # independent uniform draws. This deterministic PRNG test is mechanics only.
    assert all(abs(count / size - target) < 0.02 for count, target in zip(counts, (0.25, 0.25, 0.5), strict=True))
    rewards = (Fraction(1, 2), Fraction(1, 4), Fraction(3, 4))
    for index, reward in enumerate(rewards):
        probability = Fraction(tickets[index], GRID_SIZE)
        assert probability * (reward / probability) == reward


def test_actual_shipped_sampler_frequencies_and_independent_uniform_bit_replay():
    config = configuration()
    learner = DiscreteBandit(config, seed=463)
    oracle = Random(463)
    counts = [0, 0]
    size = 1_200
    for index in range(size):
        decision = learner.decide(request(config, index))
        chosen = config.space.members.index(_choice(decision).chosen)
        expected = 0 if oracle.getrandbits(53) < 2**52 else 1
        assert chosen == expected
        assert decision.sampling.probability == 0.5
        counts[chosen] += 1
        learner.update(observation(learner, 0.0))
    # Under independent uniform bits, the two-arm Hoeffding union bound is
    # 4*exp(-2*1200*.09**2) < 1.5e-8. This fixed PRNG mechanics test makes no
    # claim that its successive rewards or random seeds are independent trials.
    assert all(abs(count / size - 0.5) < 0.09 for count in counts)
    assert learner.export_state().counts == tuple(counts)


def test_zero_reward_metamorphism_and_concurrent_independent_streams():
    def exercise(seed):
        learner = DiscreteBandit(configuration(), seed=seed)
        choices = []
        for index in range(5):
            choices.append(learner.decide(request(configuration(), index)).to_json())
            learner.update(observation(learner, 0.0))
        assert learner.export_state().log_weights == (0.0, 0.0)
        return choices, learner.export_state().to_json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        actual = list(executor.map(exercise, (19, 20, 19)))
    assert actual[0] == actual[2] == exercise(19)
    assert actual[0] != actual[1]


def test_real_end_to_end_ci_example():
    receipts = run()
    assert len(receipts) == 2
    assert all(sum(cast(tuple[int, ...], receipt["observed_counts"])) == 8 for receipt in receipts)


def test_independent_versioned_golden_finite_mass_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "dyadic53-v1.json").read_text())
    assert fixture["schema_version"] == 1
    tickets = exp3_tickets(tuple(fixture["log_weights"]), fixture["gamma"])
    assert tickets == tuple(fixture["tickets"])
    for draw, expected in fixture["draws"]:
        assert ticket_index(tickets, draw) == expected


def test_independently_authored_ucb_state_golden_and_content_digest():
    data = json.loads((Path(__file__).parent / "fixtures" / "ucb1-state-v1.json").read_text())
    state = DiscreteState.from_json(json.dumps(data))
    assert state.counts == (1, 1)
    assert state.rewards == (0.25, 0.75)
    assert state.generator is None
    assert state.pending is None
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert state.to_json() == canonical
    assert state.fingerprint == "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    # Reading an old fixture does not authorize its foreign runtime to execute.
    incompatible_owner = DiscreteBandit(configuration("ucb1-binary64-v1"))
    with pytest.raises(ValueError, match="runtime"):
        incompatible_owner.restore_state(state)
    # Produce these same statistics through real observations, without rewriting
    # the fixture's incompatible runtime attestation to permit its continuation.
    config = configuration("ucb1-binary64-v1")
    learner = DiscreteBandit(config)
    for index, reward in enumerate((0.25, 0.75)):
        learner.decide(request(config, index))
        learner.update(observation(learner, reward))
    actual = learner.export_state().model_dump(mode="json")
    assert {key: value for key, value in actual.items() if key != "runtime"} == {
        key: value for key, value in data.items() if key != "runtime"
    }
    assert _choice(learner.decide(request(config, 2))).chosen.id == "recipe-b"
    learner.update(observation(learner, 0.0))
    assert _choice(learner.decide(request(config, 3))).chosen.id == "recipe-a"


@pytest.mark.parametrize("profile", ("exp3-dyadic53-v1", "ucb1-binary64-v1"))
def test_absorbed_positive_reward_refuses_atomically_instead_of_becoming_zero(profile):
    config = configuration(profile)
    learner = DiscreteBandit(config)
    state = learner.export_state().model_copy(update={"counts": (1, 1), "rewards": (1.0, 1.0)})
    learner.restore_state(state)
    learner.decide(request(config, 2))
    before = learner.export_state()
    tiny = observation(learner, math.ulp(0.0))
    with pytest.raises(ValueError, match="absorbed"):
        learner.update(tiny)
    assert learner.export_state() == before


def test_subnormal_positive_exp3_log_increment_underflow_is_atomic():
    learner = DiscreteBandit(configuration())
    learner.decide(request(configuration(), 0))
    before = learner.export_state()
    tiny = observation(learner, math.ulp(0.0))
    with pytest.raises(ValueError, match="underflows"):
        learner.update(tiny)
    assert learner.export_state() == before


@pytest.mark.parametrize("field", ("python", "implementation", "platform", "machine", "os_release", "os_version"))
def test_incompatible_interpreter_or_numerical_platform_restore_is_atomic(field):
    learner = DiscreteBandit(configuration())
    before = learner.export_state()
    altered = before.model_copy(update={"runtime": before.runtime.model_copy(update={field: "unsupported"})})
    with pytest.raises(ValueError, match="runtime"):
        learner.restore_state(altered)
    assert learner.export_state() == before


@pytest.mark.parametrize(
    "change", ({"round": 1}, {"task": reference("other-task")}, {"objective": reference("other-objective")})
)
def test_invalid_request_is_atomic(change):
    learner = DiscreteBandit(configuration())
    before = learner.export_state()
    invalid = request(configuration(), 0).model_copy(update=change)
    with pytest.raises(ValueError):
        learner.decide(invalid)
    assert learner.export_state() == before


def test_conflicting_pending_round_and_changing_availability_reject():
    config = configuration()
    learner = DiscreteBandit(config)
    learner.decide(request(config, 0))
    before = learner.export_state()
    conflicting = request(config, 0).model_copy(update={"decision_id": "conflict"})
    with pytest.raises(ValueError, match="original producer"):
        learner.decide(conflicting)
    changed_space = config.space.model_copy(update={"members": config.space.members[:1]})
    changed_catalogue = request(config, 0).model_copy(update={"space": changed_space})
    with pytest.raises(ValueError, match="catalogue"):
        learner.decide(changed_catalogue)
    assert learner.export_state() == before


def test_numerical_envelope_failure_and_horizon_are_atomic():
    config = configuration().model_copy(update={"maximum_updates": 1})
    learner = DiscreteBandit(config)
    learner.decide(request(config, 0))
    learner.update(observation(learner, 1.0))
    completed = learner.export_state()
    exhausted = request(config, 1)
    with pytest.raises(ValueError, match="horizon"):
        learner.decide(exhausted)
    assert learner.export_state() == completed
    learner = DiscreteBandit(configuration())
    state = learner.export_state().model_copy(update={"log_weights": (0.0, -700.0)})
    learner.restore_state(state)
    learner.decide(request(configuration(), 0))
    before = learner.export_state()
    pending = learner.pending_execution
    assert pending is not None
    assert isinstance(pending.action, ChoiceAction)
    assert pending.action.chosen == configuration().space.members[0]
    excessive = observation(learner, 1.0)
    with pytest.raises(ValueError, match="envelope"):
        learner.update(excessive)
    assert learner.export_state() == before


def test_same_instance_concurrent_retries_consume_one_draw_and_one_update():
    config = configuration()
    learner = DiscreteBandit(config)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        decisions = list(executor.map(lambda _: learner.decide(request(config, 0)), range(8)))
    assert len({decision.to_json() for decision in decisions}) == 1
    batch = observation(learner, 0.5)

    def deliver(_):
        try:
            learner.update(batch)
        except ValueError:
            return False
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        accepted = list(executor.map(deliver, range(8)))
    assert sum(accepted) == 1
    assert sum(learner.export_state().counts) == 1
    sequential = DiscreteBandit(config)
    sequential.decide(request(config, 0))
    sequential.update(batch)
    assert learner.export_state() == sequential.export_state()


@pytest.mark.parametrize(
    "values,gamma",
    (((), 0.5), ((0.0,), True), ((float("nan"),), 0.5), ((-701.0,), 0.5), ((1.0,), 0.5), ((0.0, 0.0), 0.0)),
)
def test_sampler_rejects_nonfinite_out_of_envelope_and_lost_support(values, gamma):
    with pytest.raises(ValueError):
        exp3_tickets(values, gamma)


@pytest.mark.parametrize("seed", (True, None, "19"))
def test_seed_is_initialization_not_coerced_probability_or_external_generator(seed):
    config = configuration()
    with pytest.raises(ValueError, match="Seed"):
        DiscreteBandit(config, seed=seed)


@pytest.mark.parametrize(
    "tickets,draw",
    (((1, 2), 0), ((-1, GRID_SIZE + 1), 1), ((GRID_SIZE,), -1), ((GRID_SIZE,), GRID_SIZE), ((GRID_SIZE,), True)),
)
def test_malformed_sampler_support_rejects(tickets, draw):
    with pytest.raises(ValueError):
        ticket_index(tickets, draw)
