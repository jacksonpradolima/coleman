"""Shared direct-protocol checks for actual shipped methods and a tiny host.

The host comparator is independently authored here. Its successful conformance
never substitutes for exercising each actual shipped learner through the real
adapter. All observations are constructed bounded CI losses, not physical CI
jobs or independent experimental units.
"""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.core.actions import MixtureAction, MixtureSpace, Weight
from coleman.core.capabilities import Capability, ProblemCapabilities, validate_decision, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.learning.contracts import ReferenceConfig
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.testing.decision_contracts import ConformanceCase, check_decision_contracts


def _ref(name):
    return Reference(id=name, version="v1")


TASK = _ref("constructed-ci-mixture")
OBJECTIVE = _ref("minimize-declared-synthetic-loss")
MEMBERS = (_ref("expert-0"), _ref("expert-1"))
SPACE = MixtureSpace(feasible_set=_ref("constructed-ci-experts"), members=MEMBERS, profile="binary64-simplex-v1")
HOST_IMPLEMENTATION = _ref("independent-test-cumulative-leader")


def _config(profile="hedge-constant", alpha=None):
    fixture = Path(__file__).parents[1] / "fixtures" / "learning" / "v1" / "hedge-config-wire.json"
    data = json.loads(fixture.read_text())
    data["profile"], data["alpha"] = profile, alpha
    data["eta"] = None if profile == "hedge-anytime-cumulative" else 0.5
    data["anytime_multiplier"] = 2.0 if profile == "hedge-anytime-cumulative" else None
    return ReferenceConfig.model_validate(data)


def _cases(config, *, space=SPACE):
    problem = ProblemCapabilities(
        schema_version=1,
        space=space,
        feedback_kind="full_panel",
        values=(config.loss_spec,),
        origins=("synthetic",),
        timing_mode="untimed",
        delayed_feedback=False,
        state_owners=("host", "learner"),
        sampling_statuses=("not_required",),
        receipt_status="unknown",
    )
    return tuple(
        ConformanceCase(
            problem=problem,
            request=DecisionRequest(
                schema_version=1, task=TASK, decision_id=f"build-{index}", round=index, space=space, objective=OBJECTIVE
            ),
            outcomes=tuple(
                FeedbackRow(
                    member=member,
                    values=(
                        OutcomeValue(
                            spec=config.loss_spec,
                            value=loss,
                            status="observed",
                            origin="synthetic",
                            transform=config.transform.reference,
                        ),
                    ),
                )
                for member, loss in zip(space.members, losses, strict=True)
            ),
        )
        for index, losses in enumerate(((1.0, 0.0), (0.0, 1.0), (0.25, 0.75)))
    )


@dataclass(frozen=True)
class _HostState:
    """Retain every field of this bounded comparator, including pending origin."""

    loss_spec: ScalarSpec
    transform: Reference
    totals: tuple[Fraction, Fraction] = (Fraction(), Fraction())
    updates: int = 0
    pending: ExecutedDecision | None = None


class _IndependentHost:
    """Choose a cumulative-loss leader as a one-hot coefficient vector.

    This test-local consumer uses only shared core validation. It does not call
    a production learner, exponential helper or DirectMixture. Its complete
    immutable host state is intentionally a plain Python dataclass.
    """

    def __init__(self, loss_spec: ScalarSpec, transform: Reference):
        self._state = _HostState(loss_spec=loss_spec, transform=transform)

    @property
    def capability(self):
        return Capability(
            schema_version=1,
            implementation=HOST_IMPLEMENTATION,
            action_kinds=("mixture",),
            feedback_kinds=("full_panel",),
            required_values=(self._state.loss_spec,),
            origins=("synthetic",),
            timing_modes=("untimed",),
            supports_delayed_feedback=False,
            requires_known_receipt=False,
            requires_observed=True,
            state_owner="host",
            requires_sampling=False,
            sampling_requirement="not_required",
        )

    def decide(self, request: DecisionRequest, /) -> Decision:
        request = DecisionRequest.model_validate(request)
        if (
            self._state.pending is not None
            or self._state.updates >= 64
            or request.round != self._state.updates
            or request.space != SPACE
            or request.task != TASK
            or request.objective != OBJECTIVE
        ):
            raise ValueError("This bounded host requires one matching complete update per declared opportunity")
        winner = min(range(2), key=lambda i: (self._state.totals[i], MEMBERS[i].id))
        action = MixtureAction(
            schema_version=1,
            space=SPACE,
            weights=tuple(Weight(member=member, value=float(i == winner)) for i, member in enumerate(MEMBERS)),
        )
        sampling = SamplingRecord(
            mechanism="deterministic",
            mechanism_ref=HOST_IMPLEMENTATION,
            status="not_required",
            action_fingerprint=action.fingerprint,
            probability=None,
        )
        proposal = Decision(
            schema_version=1,
            request=request,
            implementation=HOST_IMPLEMENTATION,
            action=action,
            sampling=sampling,
            state_owner="host",
        )
        validate_decision(proposal, self.capability)
        pending = ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=action,
            mapping="unchanged",
            sampling_disposition="preserved",
            sampling=sampling,
        )
        self._state = _HostState(
            loss_spec=self._state.loss_spec,
            transform=self._state.transform,
            totals=self._state.totals,
            updates=self._state.updates,
            pending=pending,
        )
        return proposal

    def update(self, batch: FeedbackBatch, /) -> None:
        pending = self._state.pending
        if pending is None:
            raise ValueError("The independent host has no pending original decision")
        validate_feedback(pending, batch, self.capability)
        if not isinstance(batch.feedback, FullPanel) or batch.supersedes is not None:
            raise ValueError("The independent host requires one root complete loss panel")
        values = tuple(row.values[0] for row in batch.feedback.rows)
        if any(value.transform != self._state.transform or value.value is None for value in values):
            raise ValueError("Unknown or differently transformed host observations are not zero")
        assert values[0].value is not None and values[1].value is not None
        totals = (self._state.totals[0] + Fraction(values[0].value), self._state.totals[1] + Fraction(values[1].value))
        self._state = _HostState(
            loss_spec=self._state.loss_spec,
            transform=self._state.transform,
            totals=totals,
            updates=self._state.updates + 1,
        )

    def export_state(self) -> _HostState:
        return self._state

    def restore_state(self, state: _HostState, /) -> None:
        if (
            type(state) is not _HostState
            or state.loss_spec != self._state.loss_spec
            or state.transform != self._state.transform
        ):
            raise TypeError("The independent comparator restores its complete host state")
        self._state = state


def test_independently_authored_host_runs_the_same_shared_protocol_cases():
    config = _config()

    def factory():
        return _IndependentHost(config.loss_spec, config.transform.reference)

    report = check_decision_contracts(factory, factory().capability, _cases(config))
    assert len(report.decisions) == len(report.feedback) == 3
    weights = []
    for decision in report.decisions:
        assert isinstance(decision.action, MixtureAction)
        weights.append(tuple(weight.value for weight in decision.action.weights))
        assert decision.implementation == HOST_IMPLEMENTATION
    assert weights == [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]


def _factory(method, config):
    # Import the actual production adapter explicitly. The mathematical learner
    # remains independently usable without importing this optional protocol map.
    from coleman.learning.direct import DirectMixture

    def create():
        return DirectMixture(method(config, owner=_ref("actual-method-owner")), task=TASK, objective=OBJECTIVE)

    return create


@pytest.mark.parametrize(
    ("method", "profile", "alpha"),
    [
        (Hedge, "hedge-constant", None),
        (AnytimeHedge, "hedge-anytime-cumulative", None),
        (FixedShare, "fixed-share-excluding-origin", 0.125),
    ],
)
def test_each_actual_shipped_method_passes_shared_direct_and_restored_paths(method, profile, alpha):
    config = _config(profile, alpha=alpha)
    factory = _factory(method, config)
    capability = factory().capability
    report = check_decision_contracts(factory, capability, _cases(config))
    assert len(report.decisions) == len(report.feedback) == 3
    for decision, batch in zip(report.decisions, report.feedback, strict=True):
        assert decision.implementation == capability.implementation
        assert isinstance(decision.action, MixtureAction)
        assert tuple(weight.member for weight in decision.action.weights) == MEMBERS
        assert all(weight.value > 0 for weight in decision.action.weights)
        assert decision.sampling.status == "not_required" and decision.sampling.probability is None
        assert isinstance(batch.feedback, FullPanel)
        assert tuple(row.member for row in batch.feedback.rows) == MEMBERS
    second = report.decisions[1].action
    assert isinstance(second, MixtureAction)
    assert second.weights[1].value > 0.5


@pytest.mark.parametrize("strength", [None, 0.0, 0.05], ids=["projection", "zero-l1", "positive-l1"])
def test_controlled_anchored_bounded_round_cadence_passes_shared_restored_paths(strength):
    """The shared immediate harness checks round cadence, not interval barriers."""
    from coleman.learning.controlled import ControlledHedge
    from coleman.learning.direct import DirectMixture
    from tests.learning._controlled_cases import configuration

    config = configuration(
        prior=(0.5, 0.5),
        eta=0.5,
        strength=strength,
        target=(0.125, (0.75, 0.25)),
        lower=(0.125, 0.125),
        upper=(0.875, 0.875),
        movement=(0.125, 0.125),
        cadence=2,
    )
    members = tuple(item.member for item in config.controls.initial_prior.weights)
    space = MixtureSpace(feasible_set=_ref("controlled-ci-experts"), members=members, profile="binary64-simplex-v1")
    instances = []

    def factory():
        actual = DirectMixture(
            ControlledHedge(config, owner=_ref("actual-controlled-owner")), task=TASK, objective=OBJECTIVE
        )
        instances.append(actual)
        return actual

    report = check_decision_contracts(factory, factory().capability, _cases(config, space=space))
    assert len(report.decisions) == len(report.feedback) == 3
    for decision in report.decisions:
        assert isinstance(decision.action, MixtureAction)
        assert tuple(item.member for item in decision.action.weights) == members
        assert all(0.125 <= item.value <= 0.875 for item in decision.action.weights)
    # The first factory instance only supplies capability. Baseline, continued,
    # and final reconstructed paths all receive the same three genuine panels.
    for app in (instances[1], instances[2], instances[-1]):
        state = app.export_state().method.payload
        assert state.update_count == 3 and state.release_count == 1
        assert len(state.pending) == 1


@pytest.mark.parametrize(
    ("method", "profile", "alpha"),
    [
        (Hedge, "hedge-constant", None),
        (AnytimeHedge, "hedge-anytime-cumulative", None),
        (FixedShare, "fixed-share-excluding-origin", 0.125),
    ],
)
def test_incompatible_coverage_rejects_before_the_conformance_factory_runs(method, profile, alpha):
    config = _config(profile, alpha=alpha)
    actual_factory = _factory(method, config)
    capability = actual_factory().capability
    cases = _cases(config)
    incompatible = tuple(
        ConformanceCase(
            problem=case.problem.model_copy(update={"feedback_kind": "chosen"}),
            request=case.request,
            outcomes=case.outcomes,
        )
        for case in cases
    )
    calls = []

    def forbidden():
        calls.append(True)
        return actual_factory()

    with pytest.raises(ValueError, match="Incompatible conformance case"):
        check_decision_contracts(forbidden, capability, incompatible)
    assert not calls
