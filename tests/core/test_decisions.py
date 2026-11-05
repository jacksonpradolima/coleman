"""Independent identity, timing and proposal-to-execution contract examples."""

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pytest

from coleman.core.actions import (
    ChoiceAction,
    ChoiceSpace,
    MixtureAction,
    MixtureSpace,
    SubsetAction,
    SubsetSpace,
    VectorAction,
    VectorSpace,
    Weight,
)
from coleman.core.capabilities import (
    Capability,
    ProblemCapabilities,
    validate_composition,
    validate_decision,
    validate_feedback,
)
from coleman.core.decisions import (
    Decision,
    DecisionRequest,
    ExecutedDecision,
    MemberInputs,
    SuppliedSignal,
    deduplicate_decisions,
)
from coleman.core.feedback import ChosenFeedback, FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarRole, ScalarSpec
from coleman.core.protocols import Decider, Solver, Stateful, Updater
from coleman.core.sampling import SamplingRecord, validate_sampling
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.testing.decision_contracts import ConformanceCase, ConformanceReport, check_decision_contracts

FIXTURES = Path(__file__).parents[1] / "fixtures/core/v1"


def ref(name: str, version: str = "v1") -> Reference:
    return Reference(id=name, version=version)


def scalar_spec(
    name: str = "allocation",
    *,
    role: ScalarRole = "action",
    unit: str = "1",
    meaning: str = "resource-share",
    minimum: float | None = None,
    maximum: float | None = None,
) -> ScalarSpec:
    return ScalarSpec(
        dimension=ref(name),
        semantics=UnitSemantics(unit=unit, meaning=meaning),
        role=role,
        minimum=minimum,
        maximum=maximum,
    )


def request(**changes: object) -> DecisionRequest:
    values = dict(
        schema_version=1,
        task=ref("owned-task"),
        decision_id="opportunity-1",
        round=0,
        space=ChoiceSpace(members=(ref("a"), ref("b")), feasible_set=ref("two-alternatives")),
        objective=ref("bounded-loss"),
    )
    return DecisionRequest.model_validate(values | changes)


def sampling(action, **changes: object) -> SamplingRecord:
    values = dict(
        mechanism="sampled",
        mechanism_ref=ref("owned-uniform"),
        status="exact",
        action_fingerprint=action.fingerprint,
        probability=0.5,
    )
    return SamplingRecord.model_validate(values | changes)


def decision(**changes: object) -> Decision:
    context = request()
    assert isinstance(context.space, ChoiceSpace)
    action = ChoiceAction(schema_version=1, space=context.space, chosen=ref("a"))
    values = dict(
        schema_version=1,
        request=context,
        implementation=ref("owned-component"),
        action=action,
        sampling=sampling(action),
        state_owner="learner",
    )
    return Decision.model_validate(values | changes)


def test_logical_identity_does_not_hide_a_changed_retry_or_implementation():
    original = decision()
    changed_request = request(round=1)
    changed = decision(request=changed_request)
    assert changed.logical_identity == original.logical_identity
    assert changed.fingerprint != original.fingerprint
    assert decision(implementation=ref("other-component")).logical_identity == original.logical_identity
    assert request(task=ref("owned-task", "v2")).logical_identity != original.logical_identity
    assert request(decision_id="opportunity-2").logical_identity != original.logical_identity
    exact_retry = Decision.from_json(original.to_json())
    assert deduplicate_decisions((original, exact_retry, original)) == (original,)
    for attempts in ((original, changed), (changed, original)):
        with pytest.raises(ValueError):
            deduplicate_decisions(attempts)
    for value in ([], (original,) * 4097):
        with pytest.raises(ValueError):
            deduplicate_decisions(cast(tuple[Decision, ...], value))


def test_request_set_order_is_canonical_but_input_content_and_signed_zero_are_not_erased():
    x = Scalar(spec=scalar_spec("x", role="feature"), value=0.0, status="observed")
    y = Scalar(spec=scalar_spec("y", role="prediction"), value=-0.25, status="observed")
    contexts = {request(inputs=values).fingerprint for values in itertools.permutations((x, y))}
    assert len(contexts) == 1
    negative_zero = Scalar(spec=x.spec, value=-0.0, status="observed")
    assert request(inputs=(x,)).fingerprint != request(inputs=(negative_zero,)).fingerprint
    assert request(inputs=(x,)).logical_identity == request(inputs=(negative_zero,)).logical_identity
    for invalid in ((x, x), (Scalar(spec=scalar_spec(role="reward"), value=0.0, status="observed"),)):
        with pytest.raises(ValueError):
            request(inputs=invalid)
    for member in (ref("a", "v2"), ref("outside")):
        with pytest.raises(ValueError):
            request(member_inputs=(MemberInputs(member=member, values=(x,)),))


def test_nested_input_is_copied_and_unchecked_replacements_are_revalidated():
    original = request()
    payload = original.model_dump(mode="json")
    copied = DecisionRequest.model_validate(payload)
    payload["space"]["members"][0]["version"] = "modified-after-validation"
    assert copied == original
    for changes in ({"schema_version": 2}, {"round": True}, {"round": -1}):
        invalid = original.model_copy(update=changes)
        with pytest.raises(ValueError):
            invalid.to_json()
    for field in ("workspace", "run_id", "artifact_path", "attempt_id"):
        with pytest.raises(ValueError):
            request(**{field: "not-required-by-mathematics"})


def test_proposal_requires_the_exact_space_including_its_semantic_version():
    other = ChoiceAction(
        schema_version=1,
        space=ChoiceSpace(members=(ref("a"), ref("b")), feasible_set=ref("two-alternatives", "v2")),
        chosen=ref("a"),
    )
    with pytest.raises(ValueError):
        decision(action=other, sampling=sampling(other))


@pytest.mark.parametrize(
    "status,extras",
    [
        ("exact", {}),
        ("qualified_approximation", {"error_bound": 0.01, "qualification": ref("enumerated-qualification")}),
        ("unavailable", {"probability": None, "reason": "host-overrode-selection"}),
        ("not_required", {"probability": None}),
    ],
)
def test_sampling_evidence_states_do_not_coerce_into_one_another(status, extras):
    action = decision().action
    record = sampling(action, status=status, **extras)
    validate_sampling(action, record)
    assert record.status == status
    for invalid in ({"probability": 0.0}, {"probability": True}, {"probability": float("nan")}):
        with pytest.raises(ValueError):
            SamplingRecord.model_validate(record.model_dump(mode="python") | invalid)


def test_sampling_mass_does_not_become_a_mixture_weight_or_continuous_density():
    space = MixtureSpace(members=(ref("a"), ref("b")), feasible_set=ref("simplex"), profile="binary64-simplex-v1")
    action = MixtureAction(
        schema_version=1,
        space=space,
        weights=(Weight(member=ref("a"), value=0.25), Weight(member=ref("b"), value=0.75)),
    )
    with pytest.raises(ValueError):
        validate_sampling(action, sampling(action, probability=0.25))
    validate_sampling(action, sampling(action, status="not_required", probability=None))
    with pytest.raises(ValueError):
        sampling(decision().action, mechanism="deterministic", probability=0.5)


def test_execution_override_must_bind_actual_content_and_disclose_sampling_loss():
    proposal = decision()
    assert isinstance(proposal.request.space, ChoiceSpace)
    actual = ChoiceAction(schema_version=1, space=proposal.request.space, chosen=ref("b"))
    unavailable = sampling(actual, status="unavailable", probability=None, reason="overridden")
    executed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=actual,
        mapping="overridden",
        sampling_disposition="invalidated",
        sampling=unavailable,
    )
    assert executed.logical_identity == proposal.logical_identity
    assert executed.action != proposal.action
    assert ExecutedDecision.from_json(executed.to_json()) == executed
    for mapping, disposition, evidence in (
        ("unchanged", "invalidated", unavailable),
        ("overridden", "preserved", unavailable),
        ("overridden", "invalidated", sampling(actual)),
        ("overridden", "recomputed", proposal.sampling),
    ):
        with pytest.raises(ValueError):
            ExecutedDecision(
                schema_version=1,
                proposal=proposal,
                action=actual,
                mapping=mapping,
                sampling_disposition=disposition,
                sampling=evidence,
            )
    recomputed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=actual,
        mapping="overridden",
        sampling_disposition="recomputed",
        sampling=sampling(actual, probability=0.25),
    )
    assert recomputed.sampling.probability == 0.25


def test_preserved_sampling_requires_equal_bytes_even_when_the_action_did_not_change():
    proposal = decision()
    same = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=proposal.action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=proposal.sampling,
    )
    assert same.action == proposal.action
    with pytest.raises(ValueError):
        ExecutedDecision(
            schema_version=1,
            proposal=proposal,
            action=proposal.action,
            mapping="unchanged",
            sampling_disposition="preserved",
            sampling=sampling(proposal.action, probability=0.25),
        )


def test_signal_availability_uses_the_information_cutoff_and_never_wall_time():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timing = DecisionTiming(
        decision_at=start + timedelta(seconds=1),
        information_cutoff=start,
        target_start=start + timedelta(seconds=2),
        target_end=start + timedelta(seconds=3),
        eligible_input_fingerprint="sha256:" + "0" * 64,
        mode="future",
    )
    value = Scalar(spec=scalar_spec(role="prediction"), value=0.0, status="observed")
    for available in (None, start + timedelta(microseconds=1)):
        signal = SuppliedSignal(source=ref("external-signal"), transform=None, values=(value,), available_at=available)
        with pytest.raises(ValueError):
            request(timing=timing, signals=(signal,))
    known = SuppliedSignal(source=ref("external-signal"), transform=None, values=(value,), available_at=start)
    assert request(timing=timing, signals=(known,)).signals == (known,)
    unknown = known.model_copy(update={"available_at": None})
    assert request(signals=(unknown,)).timing is None


def test_feedback_receipt_remains_unknown_and_maturity_can_follow_receipt():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    times = dict(event_at=start, available_at=start + timedelta(seconds=1), mature_at=start + timedelta(seconds=3))
    assert FeedbackTiming(**times, received_at=None).usable_at is None
    assert FeedbackTiming(**times, received_at=start + timedelta(seconds=2)).usable_at == times["mature_at"]
    with pytest.raises(ValueError):
        FeedbackTiming(**times, received_at=start)


def test_proposal_space_equality_retains_signed_zero_in_coordinate_bounds():
    plus = scalar_spec(minimum=0.0, maximum=1.0)
    minus = scalar_spec(minimum=-0.0, maximum=1.0)
    plus_space = VectorSpace(coordinates=(plus,), signed=False, feasible_set=ref("bounded-vector"))
    minus_space = VectorSpace(coordinates=(minus,), signed=False, feasible_set=ref("bounded-vector"))
    context = request(space=plus_space)
    action = VectorAction(
        schema_version=1, space=minus_space, values=(Scalar(spec=minus, value=0.0, status="observed"),)
    )
    evidence = sampling(action, status="not_required", probability=None)
    with pytest.raises(ValueError, match="space"):
        decision(request=context, action=action, sampling=evidence)


@pytest.mark.parametrize("mapping", ["filtered", "projected", "overridden"])
def test_explicit_execution_mapping_can_change_shape_and_universe_without_relabeling_the_task(mapping):
    proposal = decision()
    actual = SubsetAction(
        schema_version=1,
        space=SubsetSpace(
            members=(ref("c"), ref("d")), feasible_set=ref("new-feasibility"), minimum_selected=1, maximum_selected=1
        ),
        selected=(ref("d"),),
    )
    evidence = sampling(actual, status="unavailable", probability=None, reason="shape-and-universe-changed")
    executed = ExecutedDecision(
        schema_version=1,
        proposal=proposal,
        action=actual,
        mapping=mapping,
        sampling_disposition="invalidated",
        sampling=evidence,
    )
    assert executed.logical_identity == proposal.logical_identity
    assert executed.proposal.request.task == proposal.request.task
    assert executed.fingerprint != proposal.fingerprint
    assert ExecutedDecision.from_json(executed.to_json()) == executed
    with pytest.raises(ValueError):
        ExecutedDecision.model_validate(executed.model_dump(mode="python") | {"sampling": proposal.sampling})


def _conformance_inputs(kind: Literal["full_panel", "chosen"]) -> tuple[Capability, tuple[ConformanceCase, ...]]:
    role = "loss" if kind == "full_panel" else "reward"
    spec = scalar_spec("bounded-outcome", role=role, meaning=f"synthetic-{role}", minimum=0.0, maximum=1.0)
    members = (ref("a"), ref("b"))
    space = (
        MixtureSpace(members=members, feasible_set=ref("two-alternatives"), profile="binary64-simplex-v1")
        if kind == "full_panel"
        else ChoiceSpace(members=members, feasible_set=ref("two-alternatives"))
    )
    problem = ProblemCapabilities(
        schema_version=1,
        space=space,
        feedback_kind=kind,
        values=(spec,),
        origins=("synthetic",),
        timing_mode="untimed",
        delayed_feedback=False,
        state_owners=("learner",),
        sampling_statuses=("not_required",) if kind == "full_panel" else ("exact",),
        receipt_status="unknown",
    )
    capability = Capability(
        schema_version=1,
        implementation=ref("host-reference-consumer"),
        action_kinds=(space.kind,),
        feedback_kinds=(kind,),
        required_values=(spec,),
        origins=("synthetic",),
        timing_modes=("untimed",),
        supports_delayed_feedback=False,
        requires_known_receipt=False,
        requires_observed=True,
        state_owner="learner",
        requires_sampling=kind == "chosen",
        sampling_requirement="exact" if kind == "chosen" else "not_required",
    )
    cases = tuple(
        ConformanceCase(
            problem=problem,
            request=request(space=space, decision_id=f"owned-opportunity-{index}", round=index),
            outcomes=tuple(
                FeedbackRow(
                    member=member,
                    values=(
                        OutcomeValue(
                            spec=spec,
                            value=value,
                            status="observed",
                            origin="synthetic",
                            transform=ref("identity-outcome"),
                        ),
                    ),
                )
                for member, value in zip(members, values, strict=True)
            ),
        )
        for index, values in enumerate(((1.0, 0.0), (0.0, 1.0), (0.25, 0.75), (0.75, 0.25)))
    )
    return capability, cases


@dataclass(frozen=True)
class _HostState:
    version: int
    totals: tuple[float, ...]
    updates: int
    rng: tuple[object, ...]

    def __eq__(self, _other):
        raise AssertionError("Host state is opaque; conformance must compare continuations")


class _HostConsumer:
    """Test-owned reference implementation, without framework state or algorithms."""

    def __init__(self, kind: Literal["full_panel", "chosen"]):
        self.kind = kind
        self.totals: list[float] = [0.0, 0.0]
        self.updates = 0
        self.rng = random.Random(23)
        self.deliveries = []

    def decide(self, context: DecisionRequest) -> Decision:
        action: MixtureAction | ChoiceAction
        if self.kind == "full_panel":
            assert isinstance(context.space, MixtureSpace)
            best = 0 if self.totals[0] <= self.totals[1] else 1
            action = MixtureAction(
                schema_version=1,
                space=context.space,
                weights=tuple(
                    Weight(member=member, value=float(index == best))
                    for index, member in enumerate(context.space.members)
                ),
            )
            evidence = sampling(action, mechanism="deterministic", status="not_required", probability=None)
        else:
            assert isinstance(context.space, ChoiceSpace)
            chosen = self.rng.randrange(2)
            action = ChoiceAction(schema_version=1, space=context.space, chosen=context.space.members[chosen])
            evidence = sampling(action)
        return decision(
            request=context, implementation=ref("host-reference-consumer"), action=action, sampling=evidence
        )

    def update(self, batch: FeedbackBatch) -> None:
        if self.kind == "full_panel":
            assert isinstance(batch.feedback, FullPanel)
            rows = batch.feedback.rows
        else:
            assert isinstance(batch.feedback, ChosenFeedback)
            rows = (batch.feedback.row,)
        for row in rows:
            assert row.values[0].value is not None
            self.totals[0 if row.member.id == "a" else 1] += row.values[0].value
        self.deliveries.append(batch)
        self.updates += 1

    def export_state(self) -> _HostState:
        return _HostState(1, tuple(self.totals), self.updates, self.rng.getstate())

    def restore_state(self, state: _HostState) -> None:
        if state.version != 1 or len(state.totals) != 2:
            raise ValueError("Unknown host snapshot version")
        self.totals = list(state.totals)
        self.updates = state.updates
        self.rng.setstate(state.rng)


@pytest.mark.parametrize("kind", ["full_panel", "chosen"])
def test_shared_conformance_checks_actual_host_updates_and_owned_rng_restore_without_inheritance(kind):
    capability, cases = _conformance_inputs(kind)
    instances = []

    def factory():
        consumer = _HostConsumer(kind)
        instances.append(consumer)
        return consumer

    # Protocols are structural: no registration, inheritance or runtime needed.
    component = _HostConsumer(kind)
    decider: Decider[DecisionRequest, Decision] = component
    updater: Updater[FeedbackBatch, None] = component
    stateful: Stateful[_HostState] = component
    assert callable(decider.decide) and callable(updater.update) and callable(stateful.export_state)
    report = check_decision_contracts(factory, capability, cases)
    assert len(report.decisions) == len(report.feedback) == 4
    assert len(instances) == 6
    assert instances[0].updates == instances[1].updates == 4
    assert instances[0].totals == instances[1].totals
    assert instances[0].deliveries == instances[1].deliveries == list(report.feedback)
    if kind == "full_panel":
        assert all(isinstance(batch.feedback, FullPanel) and len(batch.feedback.rows) == 2 for batch in report.feedback)
        assert len({output.action.fingerprint for output in report.decisions}) == 2
    else:
        for output, batch in zip(report.decisions, report.feedback, strict=True):
            assert isinstance(batch.feedback, ChosenFeedback)
            assert isinstance(output.action, ChoiceAction)
            assert batch.feedback.row.member == output.action.chosen
        assert sum(instances[0].totals) < sum(value.values[0].value for case in cases for value in case.outcomes)


def test_solver_protocol_supports_an_independent_host_owned_problem_and_result():
    class AddOne:
        def solve(self, number: int, /) -> int:
            return number + 1

    solver: Solver[int, int] = AddOne()
    assert solver.solve(-2) == -1


def test_conformance_preflights_all_cases_before_calling_host_code():
    capability, cases = _conformance_inputs("chosen")

    def forbidden():
        raise AssertionError("Incompatible declarations must not call host code")

    for invalid in ([], cases[:1], cases * 17, (cases[0], cases[0])):
        with pytest.raises(ValueError):
            check_decision_contracts(forbidden, capability, cast(tuple[ConformanceCase, ...], invalid))
    with pytest.raises(ValueError, match="Incompatible"):
        check_decision_contracts(forbidden, capability.model_copy(update={"action_kinds": ("mixture",)}), cases)
    with pytest.raises(ValueError, match="receipt"):
        check_decision_contracts(forbidden, capability.model_copy(update={"requires_known_receipt": True}), cases)


def test_conformance_rejects_reused_instances_and_incomplete_state_restoration():
    capability, cases = _conformance_inputs("chosen")
    singleton = _HostConsumer("chosen")
    with pytest.raises(ValueError, match="fresh"):
        check_decision_contracts(lambda: singleton, capability, cases)

    class IncompleteRestore(_HostConsumer):
        def restore_state(self, state):
            self.totals = list(state.totals)
            self.updates = state.updates
            # Deliberate bug: owned RNG state is omitted from restoration.

    with pytest.raises(ValueError, match="restored/continued"):
        check_decision_contracts(lambda: IncompleteRestore("chosen"), capability, cases)


def test_conformance_rejects_changed_requests_and_uncopyable_host_snapshots():
    capability, cases = _conformance_inputs("chosen")

    class ChangedRequest(_HostConsumer):
        def decide(self, context):
            return super().decide(context.model_copy(update={"decision_id": "host-replaced-identity"}))

    with pytest.raises(ValueError, match="changed the supplied request"):
        check_decision_contracts(lambda: ChangedRequest("chosen"), capability, cases)

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise TypeError("Host snapshot cannot be copied")

    class UncopyableConsumer(_HostConsumer):
        def export_state(self):
            return Uncopyable()

    with pytest.raises(ValueError, match="deepcopy"):
        check_decision_contracts(lambda: UncopyableConsumer("chosen"), capability, cases)


def test_conformance_oracle_cannot_fabricate_missing_members_units_origins_or_duplicates():
    _, cases = _conformance_inputs("chosen")
    original = cases[0]
    row = original.outcomes[0]
    value = row.values[0]
    for outcomes in (
        original.outcomes[:1],
        (row, row),
        (row.model_copy(update={"member": ref("a", "v2")}), original.outcomes[1]),
        (row.model_copy(update={"values": (value.model_copy(update={"origin": "realized"}),)}), original.outcomes[1]),
        (
            row.model_copy(
                update={"values": (value.model_copy(update={"spec": scalar_spec(role="reward", unit="s")}),)}
            ),
            original.outcomes[1],
        ),
    ):
        with pytest.raises(ValueError):
            ConformanceCase.model_validate(original.model_dump(mode="python") | {"outcomes": outcomes})


def test_conformance_case_is_canonical_and_explicitly_rejects_unsupported_delivery_modes():
    _, cases = _conformance_inputs("chosen")
    original = cases[0]
    reversed_rows = ConformanceCase.model_validate(
        original.model_dump(mode="python") | {"outcomes": original.outcomes[::-1]}
    )
    assert reversed_rows == original
    for changes in (
        {"delayed_feedback": True},
        {"timing_mode": "future"},
        {"feedback_kind": "aggregate"},
        {"receipt_status": "known"},
    ):
        with pytest.raises(ValueError):
            ConformanceCase.model_validate(
                original.model_dump(mode="python") | {"problem": original.problem.model_copy(update=changes)}
            )
    assert isinstance(original.problem.space, ChoiceSpace)
    changed_space = ChoiceSpace(members=original.problem.space.members, feasible_set=ref("different-feasibility"))
    with pytest.raises(ValueError, match="spaces differ"):
        ConformanceCase.model_validate(original.model_dump(mode="python") | {"request": request(space=changed_space)})


def test_conformance_rejects_undeclared_actual_evidence_even_when_the_consumer_does_not_require_it():
    capability, cases = _conformance_inputs("chosen")
    relaxed = capability.model_copy(update={"requires_sampling": False, "sampling_requirement": "not_required"})
    declared = tuple(
        ConformanceCase.model_validate(
            case.model_dump(mode="python")
            | {"problem": case.problem.model_copy(update={"sampling_statuses": ("unavailable",)})}
        )
        for case in cases
    )
    with pytest.raises(ValueError, match="undeclared sampling"):
        check_decision_contracts(lambda: _HostConsumer("chosen"), relaxed, declared)


def test_conformance_report_rejects_mismatched_counts_instead_of_silently_dropping_feedback():
    capability, cases = _conformance_inputs("chosen")
    report = check_decision_contracts(lambda: _HostConsumer("chosen"), capability, cases)
    with pytest.raises(ValueError, match="equal lengths"):
        ConformanceReport.model_validate(report.model_dump(mode="python") | {"feedback": report.feedback[:-1]})


@pytest.mark.parametrize(
    "filename,model",
    [
        ("decision-request.json", DecisionRequest),
        ("decision.json", Decision),
        ("executed-decision.json", ExecutedDecision),
        ("feedback-batch.json", FeedbackBatch),
        ("capability.json", Capability),
        ("problem-capabilities.json", ProblemCapabilities),
    ],
)
def test_independently_frozen_ci_contracts_have_exact_v1_bytes_and_sha256(filename, model):
    wire = (FIXTURES / filename).read_bytes().removesuffix(b"\n")
    expected = json.loads((FIXTURES / "digests.json").read_text())[filename]
    assert hashlib.sha256(wire).hexdigest() == expected
    for reader in (model.from_json, model.model_validate_json):
        value = reader(wire)
        assert value.to_json().encode() == wire
        assert value.fingerprint == "sha256:" + expected
    payload = json.loads(wire)
    for version in (None, 0, 2, True, "1"):
        changed = dict(payload)
        if version is None:
            changed.pop("schema_version")
        else:
            changed["schema_version"] = version
        with pytest.raises(ValueError):
            model.from_json(json.dumps(changed))
    with pytest.raises(ValueError):
        model.from_json(wire[:-1] + b',"unexpected":null}')
    with pytest.raises(ValueError):
        model.from_json(wire[:-1] + b',"schema_version":1}')


def test_frozen_ci_goldens_form_a_complete_validated_proposal_execution_and_feedback_cycle():
    capability = Capability.from_json((FIXTURES / "capability.json").read_bytes())
    problem = ProblemCapabilities.from_json((FIXTURES / "problem-capabilities.json").read_bytes())
    proposal = Decision.from_json((FIXTURES / "decision.json").read_bytes())
    executed = ExecutedDecision.from_json((FIXTURES / "executed-decision.json").read_bytes())
    batch = FeedbackBatch.from_json((FIXTURES / "feedback-batch.json").read_bytes())
    assert validate_composition(problem, capability).compatible
    validate_decision(proposal, capability)
    validate_feedback(executed, batch, capability)
    assert executed.proposal.to_json() == proposal.to_json()
    assert batch.decision_fingerprint == executed.fingerprint
