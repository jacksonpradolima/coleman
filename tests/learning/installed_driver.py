"""Copied real-wheel learning consumer, independent host and typed JSON exchange.

All data is constructed CI mechanics. Coefficients, normalized losses and
counterfactual resource models do not establish physical execution or efficacy.
This file runs outside the checkout and imports no repository test helpers.
"""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from coleman.core.actions import MixtureAction, MixtureSpace, Weight
from coleman.core.capabilities import Capability, ProblemCapabilities, validate_feedback
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.sampling import SamplingRecord
from coleman.core.timing import DecisionTiming
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferenceConfig, ReferencePayload
from coleman.learning.control_contracts import ControlledConfig, ControlledPayload, MovementPeriod
from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectMixture
from coleman.learning.reference import AnytimeHedge, FixedShare, Hedge
from coleman.objectives.bindings import DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExpertBound,
    ExpertBounds,
    L1MovementPenalty,
    MixtureControls,
    MovementAllowance,
    MovementLimit,
    ProjectionOutput,
    RoundCadence,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, DeclaredTargetShare
from coleman.state.codec import read_algorithm_state
from coleman.testing.decision_contracts import ConformanceCase, check_decision_contracts


def ref(name):
    return Reference(id=name, version="v1")


TASK, OBJECTIVE = ref("constructed-ci-mixture"), ref("minimize-declared-loss")


def config(profile="hedge-constant", alpha=None):
    source = Path(__file__).with_name("fixtures") / "learning" / "v1" / "hedge-config-wire.json"
    document = json.loads(source.read_text())
    document.update(
        profile=profile,
        alpha=alpha,
        eta=None if profile == "hedge-anytime-cumulative" else 0.5,
        anytime_multiplier=2.0 if profile == "hedge-anytime-cumulative" else None,
    )
    return ReferenceConfig.model_validate(document)


def panel(configuration, losses):
    return FullPanel(
        rows=tuple(
            FeedbackRow(
                member=weight.member,
                values=(
                    OutcomeValue(
                        spec=configuration.loss_spec,
                        value=float(value),
                        status="observed",
                        origin="synthetic",
                        transform=configuration.transform.reference,
                    ),
                ),
            )
            for weight, value in zip(configuration.prior.weights, losses, strict=True)
        )
    )


def cases(configuration):
    space = MixtureSpace(
        feasible_set=ref("constructed-ci-experts"),
        members=tuple(weight.member for weight in configuration.prior.weights),
        profile="binary64-simplex-v1",
    )
    problem = ProblemCapabilities(
        schema_version=1,
        space=space,
        feedback_kind="full_panel",
        values=(configuration.loss_spec,),
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
                schema_version=1, task=TASK, objective=OBJECTIVE, decision_id=f"build-{i}", round=i, space=space
            ),
            outcomes=panel(configuration, losses).rows,
        )
        for i, losses in enumerate(((1.0, 0.0), (0.0, 1.0), (0.25, 0.75)))
    )


class TinyHost:
    """A separately authored cumulative leader, not a shipped method substitute."""

    def __init__(self):
        self.state: tuple[float, float, int, ExecutedDecision | None] = (0.0, 0.0, 0, None)

    def capability(self):
        return Capability(
            schema_version=1,
            implementation=ref("copied-independent-host"),
            action_kinds=("mixture",),
            feedback_kinds=("full_panel",),
            required_values=(config().loss_spec,),
            origins=("synthetic",),
            timing_modes=("untimed",),
            supports_delayed_feedback=False,
            requires_known_receipt=False,
            requires_observed=True,
            state_owner="host",
            requires_sampling=False,
            sampling_requirement="not_required",
        )

    def decide(self, request):
        a, b, updates, pending = self.state
        if pending is not None or request.round != updates:
            raise ValueError("The copied host requires one complete update per opportunity")
        winner = 0 if a <= b else 1
        action = MixtureAction(
            schema_version=1,
            space=request.space,
            weights=tuple(
                Weight(member=member, value=float(i == winner)) for i, member in enumerate(request.space.members)
            ),
        )
        sampling = SamplingRecord(
            mechanism="deterministic",
            mechanism_ref=ref("copied-independent-host"),
            status="not_required",
            action_fingerprint=action.fingerprint,
            probability=None,
        )
        decision = Decision(
            schema_version=1,
            request=request,
            implementation=ref("copied-independent-host"),
            action=action,
            sampling=sampling,
            state_owner="host",
        )
        executed = ExecutedDecision(
            schema_version=1,
            proposal=decision,
            action=action,
            mapping="unchanged",
            sampling_disposition="preserved",
            sampling=sampling,
        )
        self.state = (a, b, updates, executed)
        return decision

    def update(self, batch):
        a, b, updates, pending = self.state
        if pending is None:
            raise ValueError("The copied host has no pending producer")
        validate_feedback(pending, batch, self.capability())
        assert isinstance(batch.feedback, FullPanel)
        values = tuple(row.values[0].value for row in batch.feedback.rows)
        assert values[0] is not None and values[1] is not None
        self.state = (a + values[0], b + values[1], updates + 1, None)

    def export_state(self):
        return self.state

    def restore_state(self, state):
        self.state = state


def reference_methods():
    results = []
    for method, profile, alpha in (
        (Hedge, "hedge-constant", None),
        (AnytimeHedge, "hedge-anytime-cumulative", None),
        (FixedShare, "fixed-share-excluding-origin", 0.125),
    ):
        declaration = config(profile, alpha)
        owner = ref("installed-" + profile)
        learner, restored = method(declaration, owner=owner), method(declaration, owner=owner)
        learner.observe(panel(declaration, (1.0, 0.0)))
        wire = learner.export_state().to_json()
        state = read_algorithm_state(wire, payload_type=ReferencePayload, expected=learner.binding)
        assert state.to_json() == wire
        restored.restore_state(state)
        for losses in ((0.0, 1.0), (0.25, 0.75)):
            assert learner.observe(panel(declaration, losses)).status == "accepted"
            assert restored.observe(panel(declaration, losses)).status == "accepted"
        assert learner.export_state().to_json() == restored.export_state().to_json()

        def factory(method=method, declaration=declaration, owner=owner):
            return DirectMixture(method(declaration, owner=owner), task=TASK, objective=OBJECTIVE)

        report = check_decision_contracts(factory, factory().capability, cases(declaration))
        assert len(report.decisions) == len(report.feedback) == 3
        assert all(isinstance(value.action, MixtureAction) for value in report.decisions)
        assert all(value.implementation == factory().capability.implementation for value in report.decisions)
        results.append(
            {
                "profile": profile,
                "raw_updates": learner.update_count,
                "shared_conformance_cases": 3,
                "complete_state_fingerprint": learner.export_state().fingerprint,
            }
        )
    host = check_decision_contracts(TinyHost, TinyHost().capability(), cases(config()))
    assert len(host.decisions) == len(host.feedback) == 3
    assert host.decisions[0].implementation == ref("copied-independent-host")
    return {"methods": results, "independent_host_cases": 3, "scope": "constructed-protocol-mechanics"}


def metadata(name):
    return {
        "schema_version": 1,
        "reference": ref(name),
        "lineage": DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
    }


def controlled_config(
    *, strength: float | None = 0.0625, scope: Literal["per_interval", "per_release"] = "per_interval"
):
    base = config()
    initial = CoefficientOrigin(
        **metadata("initial-emission"),
        weights=base.prior.weights,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    target = DeclaredTargetShare(
        **metadata("declared-target"),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
        alpha=0.125,
        weights=tuple(
            Weight(member=weight.member, value=value)
            for weight, value in zip(base.prior.weights, (0.75, 0.25), strict=True)
        ),
    )
    bounds = ExpertBounds(
        **metadata("bounds"),
        semantics=COEFFICIENT_SEMANTICS,
        bounds=tuple(ExpertBound(member=weight.member, lower=0.125, upper=0.875) for weight in base.prior.weights),
    )
    movement = MovementAllowance(
        **metadata("movement"),
        semantics=COEFFICIENT_SEMANTICS,
        scope=scope,
        measure="accumulated_absolute",
        limits=tuple(MovementLimit(member=weight.member, maximum=0.5) for weight in base.prior.weights),
    )
    output = (
        ProjectionOutput()
        if strength is None
        else L1MovementPenalty(**metadata("l1"), origin=initial, semantics=COEFFICIENT_SEMANTICS, strength=strength)
    )
    controls = MixtureControls(
        **metadata("controls"),
        initial_prior=base.prior,
        target=target,
        bounds=bounds,
        movement=movement,
        cadence=RoundCadence(panels=2),
        output=output,
    )
    return ControlledConfig(
        **metadata("controlled-law"),
        profile="controlled-hedge-projection" if strength is None else "controlled-hedge-l1-movement",
        controls=controls,
        loss_spec=base.loss_spec,
        transform=base.transform,
        eta=0.5,
        period_clock=ref("period-clock") if scope == "per_interval" else None,
    )


def controlled_conformance():
    """Exercise actual controlled kernels through all shared restart paths."""
    results = []
    for strength in (None, 0.0, 0.05):
        declaration = controlled_config(strength=strength, scope="per_release")

        def factory(declaration=declaration):
            return DirectMixture(
                ControlledHedge(declaration, owner=ref("installed-controlled-conformance")),
                task=TASK,
                objective=OBJECTIVE,
            )

        report = check_decision_contracts(factory, factory().capability, cases(config()))
        assert len(report.decisions) == len(report.feedback) == 3
        for decision in report.decisions:
            assert isinstance(decision.action, MixtureAction)
            assert all(0.125 <= weight.value <= 0.875 for weight in decision.action.weights)
        results.append(
            {"strength": strength, "shared_conformance_cases": 3, "anchor_and_bounds": True, "cadence_panels": 2}
        )
    return {"modes": results, "scope": "constructed-protocol-mechanics"}


def interval_replay():
    """Resume actual interval replay from the held original-producer checkpoint.

    Only the copied public CI example and installed SDK are imported. No
    repository helper, mocked replay, wall-clock CI execution or new runner is
    involved. Projection and positive-L1 modes retain the same anchor/boxes.
    """
    import importlib.util

    from coleman.learning.direct import DirectOptions
    from coleman.learning.replay import ReplayMixture
    from coleman.learning.schedule import ControlledDirect, ControlSchedule
    from coleman.objectives.controls import ReleaseInterval
    from coleman.replay.codec import read_continuation
    from coleman.replay.contracts import ReplayPolicy
    from coleman.replay.engine import CausalReplay

    path = Path(__file__).with_name("ci.py")
    spec = importlib.util.spec_from_file_location("installed_public_ci_example", path)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    modes = []
    for strength in (None, 0.05):
        calendar = example.reference("installed-ci-calendar")
        declaration = example.controlled_configuration(cadence=calendar, strength=strength)
        point = ReleaseInterval(
            interval=example.reference("installed-ci-release"),
            start=example.NOW,
            end=example.NOW + timedelta(minutes=5),
            cutoff=example.NOW + timedelta(minutes=3),
            release_at=example.NOW + timedelta(minutes=4),
        )
        schedule = ControlSchedule(schema_version=1, calendar=calendar, intervals=(point,))

        def application(declaration=declaration, schedule=schedule):
            return ControlledDirect(
                ControlledHedge(declaration, owner=example.reference("installed-ci-controller")),
                schedule,
                DirectOptions(task=example.TASK, objective=example.OBJECTIVE, horizon=example.HORIZON),
            )

        direct = application()
        host = ReplayMixture(direct)
        policy = ReplayPolicy(
            schema_version=1,
            stream=example.TASK,
            lineage=example.reference("installed-ci-lineage"),
            parent=None,
            source=example.SOURCE,
            capability=host.capability,
            delivery="delayed",
            specialists=False,
        )
        replay = CausalReplay(host, policy, state_type=direct.state_type, state_codec=direct.binding.state_codec)
        replay.admit_history(example.history())
        first = replay.emit(replay.prepare(example.opportunity(direct, 0, 0)), execution="offline_identity_v1")
        delayed = example.feedback(first, declaration, (0.0, 1.0, 0.5), visible=2)
        replay.admit_feedback((delayed, delayed))
        replay.advance(eligible_through=example.NOW + timedelta(minutes=2), apply_at=example.NOW + timedelta(minutes=3))
        assert direct.learner.update_count == 1 and direct.learner.release_count == 0
        before = replay.export_continuation()
        decoded = read_continuation(before.to_json(), state_type=direct.state_type)
        fresh = application()
        resumed = CausalReplay.from_continuation(
            ReplayMixture(fresh), decoded, state_type=fresh.state_type, state_codec=fresh.binding.state_codec
        )
        assert len(fresh.learner.export_state().payload.pending) == 1
        original_next = replay.emit(
            replay.prepare(example.opportunity(direct, 1, 4, 3)), execution="offline_identity_v1"
        )
        resumed_next = resumed.emit(
            resumed.prepare(example.opportunity(fresh, 1, 4, 3)), execution="offline_identity_v1"
        )
        assert original_next.to_json() == resumed_next.to_json()
        assert direct.export_state().to_json() == fresh.export_state().to_json()
        assert replay.export_continuation().to_json() == resumed.export_continuation().to_json()
        stable = resumed.export_continuation().to_json()
        resumed.admit_feedback((delayed, delayed))
        assert resumed.export_continuation().to_json() == stable
        assert direct.learner.last_outcome is not None and direct.learner.last_outcome.disposition == "released"
        modes.append(
            {
                "strength": strength,
                "D": len(replay.state.decisions),
                "U": direct.learner.update_count,
                "R": direct.learner.release_count,
                "restored_pending_panels": 1,
                "complete_continuation_equal": True,
                "anchor_and_bounds": True,
                "cadence": "interval",
                "state_fingerprint": direct.export_state().fingerprint,
                "physical_executions": 0,
            }
        )
    return {"modes": modes, "scope": "constructed-protocol-mechanics"}


def controlled_continuation():
    declaration = controlled_config()
    owner = ref("installed-control")
    actual, restored = ControlledHedge(declaration, owner=owner), ControlledHedge(declaration, owner=owner)
    base = config()
    start = datetime(2026, 11, 7, tzinfo=UTC)
    period = MovementPeriod(
        schema_version=1,
        calendar=ref("period-clock"),
        interval=ref("budget-1"),
        start=start,
        end=start + timedelta(seconds=100),
    )
    actual.begin_period(period, apply_at=start)
    held = actual.observe(
        panel(base, (0.0, 1.0)), eligible_at=start + timedelta(seconds=1), apply_at=start + timedelta(seconds=1)
    )
    assert held.disposition == "held" and actual.update_count == 1 and actual.release_count == 0
    pending = actual.export_state()
    state = read_algorithm_state(pending.to_json(), payload_type=ControlledPayload, expected=actual.binding)
    assert len(state.payload.pending) == 1 and state.payload.exact_path is not None
    restored.restore_state(state)
    for learner in (actual, restored):
        moved = learner.observe(
            panel(base, (0.0, 1.0)), eligible_at=start + timedelta(seconds=2), apply_at=start + timedelta(seconds=2)
        )
        assert moved.disposition == "released" and learner.release_count == 1
        assert tuple(weight.value for weight in learner.weights) != (0.5, 0.5)
        moving = learner.export_state()
        usage = moving.payload.exact_path
        assert usage is not None and any(int(value.atoms_decimal) > 0 for value in usage.used)
        learner.observe(
            panel(base, (0.5, 0.5)), eligible_at=start + timedelta(seconds=3), apply_at=start + timedelta(seconds=3)
        )
        stationary = learner.observe(
            panel(base, (0.5, 0.5)), eligible_at=start + timedelta(seconds=4), apply_at=start + timedelta(seconds=4)
        )
        assert stationary.disposition == "released" and learner.release_count == 2
        assert learner.weights == moving.payload.origin.weights
        after_usage = learner.export_state().payload.exact_path
        assert after_usage is not None and after_usage.used == usage.used
        assert learner.config.binding == declaration.binding
    assert actual.export_state().to_json() == restored.export_state().to_json()
    return {
        "profile": declaration.profile,
        "U": actual.update_count,
        "R": actual.release_count,
        "restored_pending_panels": 1,
        "positive_lambda_moving_and_hold": True,
        "state_fingerprint": actual.export_state().fingerprint,
        "physical_executions": 0,
    }


HOST_SCRIPT = r"""
import hashlib,json,sys
assert not any(n=='coleman' or n.startswith('coleman.') for n in sys.modules)
incoming=json.load(sys.stdin)
assert type(incoming['schema_version']) is int and incoming['schema_version']==1
assert set(incoming)=={'schema_version','executed','loss_spec','transform','horizon','timing'}
executed=incoming['executed']
canonical=json.dumps(executed,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
members=executed['action']['space']['members']
rows=[{'member':member,'values':[{'spec':incoming['loss_spec'],'value':float(i),'status':'observed',
 'origin':'synthetic','transform':incoming['transform']}]} for i,member in enumerate(members)]
batch={'schema_version':1,'decision_fingerprint':'sha256:'+hashlib.sha256(canonical.encode()).hexdigest(),
 'feedback_id':'stdlib-host-result','revision_id':'root','supersedes':None,'horizon':incoming['horizon'],
 'feedback':{'kind':'full_panel','rows':rows},'timing':incoming['timing'],'delay_rounds':0}
assert not any(n=='coleman' or n.startswith('coleman.') for n in sys.modules)
print(json.dumps({'schema_version':1,'executed':executed,'feedback':batch,
 'host_runtime':{'implementation':sys.implementation.name,'version':sys.version},'coleman_imported':False},
 sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False))
"""


def isolated_host_exchange():
    declaration = config()
    horizon = ref("constructed-target")
    adapter = DirectMixture(
        Hedge(declaration, owner=ref("exchange-owner")), task=TASK, objective=OBJECTIVE, horizon=horizon
    )
    start = datetime(2026, 11, 7, 0, 0, 0, 123456, tzinfo=UTC)
    timing = DecisionTiming(
        decision_at=start,
        information_cutoff=start,
        target_start=start + timedelta(seconds=1),
        target_end=start + timedelta(seconds=2),
        mode="future",
        eligible_input_fingerprint="sha256:" + "b" * 64,
    )
    missing_spec = ScalarSpec(
        dimension=ref("unused-ci-context"),
        role="feature",
        minimum=0.0,
        maximum=1.0,
        semantics=UnitSemantics(unit="1", meaning="constructed-context-not-observed"),
    )
    missing = Scalar(spec=missing_spec, value=None, status="unavailable")
    request = DecisionRequest.model_validate(
        cases(declaration)[0].request.model_dump() | {"timing": timing, "inputs": (missing,)}
    )
    adapter.decide(request)
    executed = adapter.export_state().pending[0]
    clocks = {
        "event_at": (start + timedelta(seconds=2)).isoformat(),
        "available_at": (start + timedelta(seconds=3)).isoformat(),
        "mature_at": (start + timedelta(seconds=4)).isoformat(),
        "received_at": None,
    }
    incoming = {
        "schema_version": 1,
        "executed": json.loads(executed.to_json()),
        "loss_spec": declaration.loss_spec.model_dump(mode="json"),
        "transform": declaration.transform.reference.model_dump(mode="json"),
        "horizon": horizon.model_dump(mode="json"),
        "timing": clocks,
    }
    host = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", HOST_SCRIPT],
        input=json.dumps(incoming),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert host.returncode == 0, host.stdout + host.stderr
    output = json.loads(host.stdout)
    assert output["schema_version"] == 1 and output["coleman_imported"] is False
    assert output["executed"] == incoming["executed"]
    retained_feature = output["executed"]["proposal"]["request"]["inputs"][0]
    assert retained_feature["status"] == "unavailable" and retained_feature["value"] is None
    batch = FeedbackBatch.from_json(json.dumps(output["feedback"]))
    assert batch.timing is not None and batch.timing.received_at is None
    assert batch.timing.event_at.microsecond == 123456
    assert isinstance(batch.feedback, FullPanel)
    assert batch.feedback.rows[0].values[0].value == 0.0
    adapter.update(batch, apply_at=start + timedelta(seconds=5))
    assert adapter.learner.update_count == 1
    return {
        "host_runtime": output["host_runtime"],
        "host_imported_coleman": False,
        "schema_version": 1,
        "unknown_receipt_preserved": True,
        "unavailable_feature_preserved": True,
        "microsecond_precision": 123456,
        "actual_updates": 1,
    }


def legacy_imports():
    from coleman.bandit import Bandit, DynamicBandit
    from coleman.policy import EXP3Policy, UCB1Policy

    assert Bandit.__module__ == "coleman.bandit.base" and DynamicBandit.__module__ == "coleman.bandit.dynamic"
    assert EXP3Policy.__module__.startswith("coleman.policy.") and UCB1Policy.__module__.startswith("coleman.policy.")
    from coleman.learning.reference import Hedge as NewHedge

    assert NewHedge is Hedge
    return {
        "legacy": [Bandit.__name__, DynamicBandit.__name__, EXP3Policy.__name__, UCB1Policy.__name__],
        "new_method": "Hedge",
        "profile": "ci",
    }


if __name__ == "__main__":
    functions = {
        "reference": reference_methods,
        "controlled": controlled_continuation,
        "controlled-conformance": controlled_conformance,
        "interval-replay": interval_replay,
        "exchange": isolated_host_exchange,
        "legacy": legacy_imports,
    }
    selected = functions[sys.argv[1]]
    import coleman.learning.reference as implementation

    assert Path(implementation.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert not Path(implementation.__file__).resolve().is_relative_to(Path(__file__).parent)
    print(json.dumps(selected(), sort_keys=True))
