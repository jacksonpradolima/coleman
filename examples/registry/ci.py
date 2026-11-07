"""Constructed CI plan lifecycle using real pure registry/feedback contracts.

The host emits uniform coefficients to make eligibility visible. It implements
no adaptive learner, probability calibration, performance claim or physical CI
execution. The separate collection example runs actual owned CI consumers.
"""

from __future__ import annotations

import json
from typing import Any

from coleman.core.actions import AbstainAction, MixtureAction, MixtureSpace, Weight
from coleman.core.capabilities import Capability
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import AbsentFeedback, FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.sampling import SamplingRecord
from coleman.core.units import UnitSemantics
from coleman.registry.definitions import Catalogue, MemberDefinition
from coleman.registry.envelopes import RegisteredExecution, RegisteredFeedback, RegistryRequest, seal_request
from coleman.registry.lifecycle import LifecycleFact, Operation, PopulationPolicy, RegistryState
from coleman.registry.limits import RegistryLimits


def _ref(name: str) -> Reference:
    """Use explicit local v1 names without global registration."""
    return Reference(id=name, version="v1")


def _initial() -> RegistryState:
    """Declare three public constructed plans; introduction is still separate."""
    from coleman.objectives.bindings import DefinitionBinding  # pylint: disable=import-outside-toplevel

    leaves = tuple(
        MemberDefinition(
            schema_version=1,
            kind="expert",
            member=_ref("ci-plan-" + name),
            implementation=DefinitionBinding(
                reference=_ref("constructed-ci-plan"),
                definition_fingerprint=fingerprint({"plan": name, "origin": "constructed-ci-example"}),
            ),
            configuration_fingerprint=fingerprint({"plan": name, "version": 1}),
        )
        for name in ("a", "b", "c")
    )
    return RegistryState(
        schema_version=1,
        catalogue=Catalogue(schema_version=1, reference=_ref("constructed-ci-catalogue"), definitions=leaves),
        policy=PopulationPolicy(
            population=_ref("ci-plans"), task=_ref("ci-build-decisions"), kind="expert", clock="untimed"
        ),
        limits=RegistryLimits(),
    )


def _fact(state: RegistryState, name: str, operation: Operation, ordinal: int) -> RegistryState:
    """Append explicit prospective transitions with required epoch declarations."""
    member = _ref("ci-plan-" + name)
    restarting = operation == "reenter"
    epoch = _ref(member.id + (":epoch-1" if restarting else ":epoch-0"))
    return state.append(
        LifecycleFact(
            schema_version=1,
            event_id=f"ci-lifecycle-{len(state.facts)}",
            sequence=len(state.facts),
            member=member,
            definition_fingerprint=state.catalogue.resolve("expert", member).fingerprint,
            operation=operation,
            effective_from_round=ordinal,
            admitted_before_round=ordinal,
            known_at=None,
            reason="constructed-ci-lifecycle",
            state_epoch=epoch if operation in {"introduce", "reenter"} else None,
            predecessor_epoch=_ref(member.id + ":epoch-0") if restarting else None,
            disposition="restart" if restarting else None,
        )
    )


def _execution(state: RegistryState, ordinal: int) -> RegisteredExecution:
    """Emit an explicit uniform presentation host with no sampling claim."""
    snapshot = state.snapshot(ordinal)
    space = MixtureSpace(
        feasible_set=snapshot.feasible_set("ci-plan-coefficients"),
        members=snapshot.active,
        profile="binary64-simplex-v1",
    )
    request = RegistryRequest(
        schema_version=1,
        snapshot=snapshot,
        request=DecisionRequest(
            schema_version=1,
            task=state.policy.task,
            decision_id=f"build-{ordinal}",
            round=ordinal,
            space=space,
            objective=_ref("constructed-normalized-ci-loss"),
        ),
    )
    action = (
        MixtureAction(
            schema_version=1,
            space=space,
            weights=tuple(Weight(member=member, value=1 / len(space.members)) for member in space.members),
        )
        if space.members
        else AbstainAction(schema_version=1, space=space, reason="no-active-ci-plan")
    )
    sampling = SamplingRecord(
        mechanism="deterministic",
        mechanism_ref=_ref("constructed-uniform-host"),
        status="not_required",
        action_fingerprint=action.fingerprint,
        probability=None,
    )
    proposed = Decision(
        schema_version=1,
        request=request.request,
        implementation=_ref("constructed-uniform-host"),
        action=action,
        sampling=sampling,
        state_owner="stateless",
    )
    return RegisteredExecution(
        schema_version=1,
        request=request,
        executed=ExecutedDecision(
            schema_version=1,
            proposal=proposed,
            action=action,
            mapping="unchanged",
            sampling_disposition="preserved",
            sampling=sampling,
        ),
    )


def _loss_spec() -> ScalarSpec:
    """Keep synthetic bounded normalized loss distinct from raw CI quantities."""
    return ScalarSpec(
        dimension=_ref("ci-loss"),
        semantics=UnitSemantics(unit="1", meaning="constructed-ci-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
    )


def _capability(*, observed: bool) -> Capability:
    """Declare actual complete-panel eligibility, including explicit absence."""
    return Capability(
        schema_version=1,
        implementation=_ref("constructed-uniform-host"),
        action_kinds=("mixture",),
        feedback_kinds=("full_panel", "absent"),
        required_values=(_loss_spec(),),
        origins=("synthetic",),
        timing_modes=("untimed",),
        supports_delayed_feedback=True,
        requires_known_receipt=False,
        requires_observed=observed,
        state_owner="stateless",
        requires_sampling=False,
        sampling_requirement="not_required",
    )


def _feedback(
    producer: RegisteredExecution,
    values: tuple[float | None, ...],
    *,
    revision_id: str = "initial",
    supersedes: str | None = None,
    delay_rounds: int = 0,
) -> RegisteredFeedback:
    """Retain the original active panel and each explicit missing/observed value."""
    rows = tuple(
        FeedbackRow(
            member=member,
            values=(
                OutcomeValue(
                    spec=_loss_spec(),
                    value=value,
                    status="unavailable" if value is None else "observed",
                    origin="synthetic",
                    transform=_ref("identity-normalized-ci-loss"),
                ),
            ),
        )
        for member, value in zip(producer.request.snapshot.active, values, strict=True)
    )
    return RegisteredFeedback(
        schema_version=1,
        producer_fingerprint=producer.fingerprint,
        batch=FeedbackBatch(
            schema_version=1,
            decision_fingerprint=producer.executed.fingerprint,
            feedback_id=f"ci-outcome-{producer.request.snapshot.round}",
            revision_id=revision_id,
            supersedes=supersedes,
            delay_rounds=delay_rounds,
            horizon=None,
            feedback=FullPanel(rows=rows) if rows else AbsentFeedback(reason="no-active-ci-plan"),
        ),
    )


def _round_receipt(producer: RegisteredExecution, values: tuple[float | None, ...], ready: bool) -> dict[str, Any]:
    """Present the actual active panel without changing the fixed consumer."""
    action = producer.executed.action
    return {
        "round": producer.request.snapshot.round,
        "active": [member.id for member in producer.request.snapshot.active],
        "coefficients": [weight.value for weight in action.weights] if isinstance(action, MixtureAction) else [],
        "losses": list(values),
        "statuses": ["unavailable" if value is None else "observed" for value in values],
        "update_eligible": ready,
        "action": action.kind,
    }


def _delayed_receipt(originals: list[RegisteredExecution], consumer: Capability) -> dict[str, Any]:
    """Audit a correctly identified delayed revision against its original epoch."""
    delayed = _feedback(originals[2], (0.25, 0.75), revision_id="completed", supersedes="initial", delay_rounds=3)
    delayed.validate_for(originals[2], consumer)
    old_b = next(member for member in originals[2].request.snapshot.members if member.member == _ref("ci-plan-b"))
    new_b = next(member for member in originals[4].request.snapshot.members if member.member == _ref("ci-plan-b"))
    assert old_b.state_epoch is not None and new_b.state_epoch is not None
    return {
        "original_round": 2,
        "original_epoch": old_b.state_epoch.id,
        "replacement_epoch": new_b.state_epoch.id,
        "delay_rounds": delayed.batch.delay_rounds,
        "disposition": "audit_only",
    }


def run() -> dict[str, Any]:
    """Execute six constructed CI lifecycle rounds through the actual contracts.

    Returns
    -------
    dict
        JSON presentation of active plans, dimensionless coefficients, original
        outcome statuses, eligibility for a complete-observed update, original
        delayed-feedback epoch and deterministic final state fingerprint.

    Raises
    ------
    ValueError
        A lifecycle, core execution, full panel or original producer join fails.

    Notes
    -----
    Uses explicit untimed logical admission rounds, no ambient clock, storage,
    RNG, learner update or physical CI job. Uniform coefficients demonstrate
    eligibility only. A null active outcome stays unavailable; an observed zero
    stays observed. The restarted plan receives no old live update.
    """
    state = _initial()
    panels = ((0.25, 0.75), (0.5,), (0.25, None), (0.25, 0.5), (0.25, 0.75, 0.0), ())
    operations: tuple[tuple[tuple[str, Operation], ...], ...] = (
        (("a", "introduce"), ("b", "introduce")),
        (("b", "inactive"),),
        (("b", "resume"),),
        (("c", "introduce"), ("b", "retire")),
        (("b", "reenter"),),
        (("a", "retire"), ("b", "retire"), ("c", "retire")),
    )
    rows, originals = [], []
    audit_contract, consumer_contract = _capability(observed=False), _capability(observed=True)
    for ordinal, (events, values) in enumerate(zip(operations, panels, strict=True)):
        for name, operation in events:
            state = _fact(state, name, operation, ordinal)
        producer = _execution(state, ordinal)
        state = seal_request(state, producer.request)
        observation = _feedback(producer, values)
        observation.validate_for(producer, audit_contract)
        ready = False
        if values:
            try:
                observation.validate_for(producer, consumer_contract)
            except ValueError:
                if all(value is not None for value in values):
                    raise
            else:
                ready = True
        rows.append(_round_receipt(producer, values, ready))
        originals.append(producer)
    return {
        "origin": "constructed_ci_example",
        "loss_unit": "1",
        "rounds": rows,
        "delayed": _delayed_receipt(originals, consumer_contract),
        "state_fingerprint": state.fingerprint,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
