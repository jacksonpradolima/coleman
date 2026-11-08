"""Run shipped expert mixtures over explicitly constructed CI loss panels.

This is an offline mathematical CI example, not a benchmark, physical test run
or evidence that an adaptive policy beats a baseline. All clocks and synthetic
losses are consciously supplied. Run this file with the installed v5 wheel.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from coleman.core.actions import MixtureSpace, Weight
from coleman.core.decisions import DecisionRequest
from coleman.core.feedback import FeedbackBatch, FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.core.timing import DecisionTiming, FeedbackTiming
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferenceConfig, ReferenceProfile
from coleman.learning.control_contracts import ControlledConfig
from coleman.learning.controlled import ControlledHedge
from coleman.learning.direct import DirectOptions
from coleman.learning.replay import ReplayMixture
from coleman.learning.schedule import ControlledDirect, ControlSchedule
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage
from coleman.objectives.controls import (
    CoefficientOrigin,
    ExpertBound,
    ExpertBounds,
    IntervalCadence,
    L1MovementPenalty,
    MixtureControls,
    MovementAllowance,
    MovementLimit,
    ProjectionOutput,
    ReleaseInterval,
    RoundCadence,
)
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, DeclaredTargetShare, InitialPrior
from coleman.observations.contracts import ObservationHistory
from coleman.replay.contracts import (
    ReplayDecisionRecord,
    ReplayFeedback,
    ReplayOpportunity,
    ReplayPolicy,
    ReplayTarget,
    ValueOriginEvidence,
)
from coleman.replay.engine import CausalReplay

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def reference(name: str) -> Reference:
    """Return an explicit local version-one identity without source lookup.

    Parameters
    ----------
    name : str
        Bounded logical identifier.

    Returns
    -------
    Reference
        Immutable supplied identity/version.
    """
    return Reference(id=name, version="1")


MEMBERS = tuple(reference("ci-plan-" + name) for name in ("duration", "failure-history", "stable-order"))
SOURCE = reference("constructed-ci-loss-stream")
TASK = reference("constructed-ci-coefficient-decisions")
OBJECTIVE = reference("minimize-constructed-normalized-ci-loss")
HORIZON = reference("one-constructed-ci-build")
GENERATOR = DefinitionBinding(
    reference=reference("constructed-ci-loss-generator"),
    definition_fingerprint=fingerprint({"authored": "complete-ci-panels-v1"}),
)


def _lineage() -> DefinitionLineage:
    """Retain explicit unknown declaration facts instead of inventing evidence."""
    return DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)


def loss_spec() -> ScalarSpec:
    """Return the exact dimensionless synthetic bounded CI loss interpretation.

    Returns
    -------
    ScalarSpec
        Inclusive [0,1] loss, neither runtime seconds nor a failure probability.
    """
    return ScalarSpec(
        dimension=reference("constructed-ci-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
        semantics=UnitSemantics(unit="1", meaning="constructed-normalized-ci-loss-v1"),
    )


def reference_configuration(profile: ReferenceProfile = "hedge-constant") -> ReferenceConfig:
    """Declare one shipped static reference on three public constructed CI plans.

    Parameters
    ----------
    profile : str, optional
        Constant Hedge, current-rate cumulative anytime Hedge, or excluding-
        origin Fixed Share. Default is constant Hedge.

    Returns
    -------
    ReferenceConfig
        Complete positive prior, exact loss/transform, fixed parameters and caps.
    """
    prior = InitialPrior(
        schema_version=1,
        reference=reference("initial-ci-expert-coefficients"),
        lineage=_lineage(),
        weights=tuple(
            Weight(member=member, value=value) for member, value in zip(MEMBERS, (0.5, 0.25, 0.25), strict=True)
        ),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    return ReferenceConfig(
        schema_version=1,
        reference=reference("constructed-" + profile),
        lineage=_lineage(),
        profile=profile,
        prior=prior,
        loss_spec=loss_spec(),
        transform=DefinitionBinding(
            reference=reference("identity-bounded-ci-loss"),
            definition_fingerprint=fingerprint({"transform": "already-bounded-identity-v1"}),
        ),
        eta=None if profile == "hedge-anytime-cumulative" else 0.5,
        alpha=0.1 if profile == "fixed-share-excluding-origin" else None,
    )


def controlled_configuration(*, cadence: int | Reference = 2, strength: float | None = 0.05) -> ControlledConfig:
    """Declare the required anchored/bounded/cadenced optional-penalty combination.

    Parameters
    ----------
    cadence : int or Reference, optional
        Accepted panels per sum-loss block, or exact external release calendar.
    strength : float or None, optional
        Nonnegative full-L1 coefficient penalty, default .05; None explicitly
        selects the separate plain projection profile. Not a paid fee.

    Returns
    -------
    ControlledConfig
        Full fixed law; its dynamic origin remains future complete state.
    """
    source = reference_configuration()
    prior = source.prior
    origin = CoefficientOrigin(
        schema_version=1,
        reference=reference("initial-ci-emitted-origin"),
        lineage=_lineage(),
        weights=prior.weights,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    output = (
        ProjectionOutput()
        if strength is None
        else L1MovementPenalty(
            schema_version=1,
            reference=reference("development-declared-ci-l1"),
            lineage=_lineage(),
            strength=strength,
            origin=origin,
            semantics=COEFFICIENT_SEMANTICS,
        )
    )
    bounds = ExpertBounds(
        schema_version=1,
        reference=reference("declared-ci-boxes"),
        lineage=_lineage(),
        bounds=tuple(ExpertBound(member=member, lower=0.1, upper=0.7) for member in MEMBERS),
        semantics=COEFFICIENT_SEMANTICS,
    )
    movement = MovementAllowance(
        schema_version=1,
        reference=reference("declared-ci-release-movement"),
        lineage=_lineage(),
        limits=tuple(MovementLimit(member=member, maximum=0.2) for member in MEMBERS),
        scope="per_release",
        measure="accumulated_absolute",
        semantics=COEFFICIENT_SEMANTICS,
    )
    target = DeclaredTargetShare(
        schema_version=1,
        reference=reference("researcher-declared-ci-target"),
        lineage=_lineage(),
        weights=tuple(Weight(member=member, value=1 / 3) for member in MEMBERS),
        alpha=0.1,
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    controls = MixtureControls(
        schema_version=1,
        reference=reference("constructed-ci-controlled-law"),
        lineage=_lineage(),
        initial_prior=prior,
        target=target,
        bounds=bounds,
        movement=movement,
        cadence=RoundCadence(panels=cadence) if isinstance(cadence, int) else IntervalCadence(calendar=cadence),
        output=output,
    )
    return ControlledConfig(
        schema_version=1,
        reference=reference("constructed-controlled-ci"),
        lineage=_lineage(),
        profile="controlled-hedge-projection" if strength is None else "controlled-hedge-l1-movement",
        controls=controls,
        loss_spec=source.loss_spec,
        transform=source.transform,
        eta=0.5,
    )


def history() -> ObservationHistory:
    """Return an empty explicit input history; future outcome panels stay separate.

    Returns
    -------
    ObservationHistory
        Versioned authored CI source with no hidden expert outcomes.
    """
    return ObservationHistory(schema_version=1, source_id=SOURCE.id, source_version=SOURCE.version, revisions=())


def opportunity(application, index: int, at: float, cutoff: float | None = None) -> ReplayOpportunity:
    """Build one timed original CI opportunity from its actual eligible view.

    Parameters
    ----------
    application : ControlledDirect or DirectMixture
        Actual independently owned shipped mathematical application.
    index : int
        Zero-based decision index D, separate from accepted panels U/releases R.
    at, cutoff : float or None
        Authored minute offsets from NOW. A null cutoff equals the explicit at.

    Returns
    -------
    ReplayOpportunity
        Timed full-mixture request and declared build target/horizon.
    """
    information = NOW + timedelta(minutes=at if cutoff is None else cutoff)
    decision = NOW + timedelta(minutes=at)
    space = MixtureSpace(
        feasible_set=reference("ci-expert-coefficient-simplex"),
        members=application.learner.members,
        profile="binary64-simplex-v1",
    )
    request = DecisionRequest(
        schema_version=1,
        task=TASK,
        decision_id="constructed-build-" + str(index),
        round=index,
        space=space,
        objective=OBJECTIVE,
        timing=DecisionTiming(
            decision_at=decision,
            information_cutoff=information,
            target_start=decision,
            target_end=decision,
            mode="future",
            eligible_input_fingerprint=history().as_of(information).fingerprint,
        ),
    )
    return ReplayOpportunity(
        schema_version=1,
        request=request,
        context=None,
        target=ReplayTarget(
            reference=reference("constructed-build-target-" + str(index)),
            horizon=HORIZON,
            groups=(reference("constructed-ci-family"),),
        ),
    )


def feedback(
    record: ReplayDecisionRecord,
    configuration: ControlledConfig | ReferenceConfig,
    losses: tuple[float, ...],
    *,
    visible: float,
    delay: int = 0,
) -> ReplayFeedback:
    """Supply driver-owned complete synthetic outcomes with exact provenance.

    Parameters
    ----------
    record : ReplayDecisionRecord
        Original actual offline-identity execution, preserving its full producer.
    configuration : ControlledConfig or ReferenceConfig
        Exact loss and transform declaration used by the producing method.
    losses : tuple of float
        Complete authored [0,1] vector in the fixed expert order.
    visible : float
        Authored availability, maturity and known receipt minute offset.
    delay : int, optional
        Exact number of intervening emitted decisions at application, default0.

    Returns
    -------
    ReplayFeedback
        Complete original-producer full panel; provenance labels are synthetic,
        never manufactured realized observations or independent seeds.
    """
    timing = record.opportunity.request.timing
    if timing is None:
        raise ValueError("This authored example requires an explicitly timed producer")
    space = record.opportunity.request.space
    if not isinstance(space, MixtureSpace):
        raise ValueError("Constructed mixture feedback requires the original complete expert space")
    members = space.members
    rows = tuple(
        FeedbackRow(
            member=member,
            values=(
                OutcomeValue(
                    spec=configuration.loss_spec,
                    value=value,
                    status="observed",
                    origin="synthetic",
                    transform=configuration.transform.reference,
                ),
            ),
        )
        for member, value in zip(members, losses, strict=True)
    )
    instant = NOW + timedelta(minutes=visible)
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=record.executed.fingerprint,
        feedback_id="constructed-ci-outcome",
        revision_id="initial",
        supersedes=None,
        horizon=HORIZON,
        feedback=FullPanel(rows=rows),
        delay_rounds=delay,
        timing=FeedbackTiming(event_at=timing.target_end, available_at=instant, mature_at=instant, received_at=instant),
    )
    origins = tuple(
        ValueOriginEvidence(
            member=row.member,
            dimension=value.spec.dimension,
            value_fingerprint=fingerprint(value.model_dump(mode="json")),
            origin="synthetic",
            producer=GENERATOR,
        )
        for row in rows
        for value in row.values
    )
    return ReplayFeedback(
        schema_version=1, lineage=record.lineage, batch=batch, target=record.opportunity.target, origins=origins
    )


def run(*, strength: float | None = 0.05) -> dict[str, object]:
    """Run the real shared causal replay and penalty application, without mocks.

    Parameters
    ----------
    strength : float or None, optional
        Full-L1 coefficient strength, default .05. None selects the separately
        required unpenalized projection with the same target/bounds/cadence.

    Returns
    -------
    dict
        Human-readable original/final coefficients, separate D/U/R clocks,
        actual receipt/application instants and deterministic state digest.

    Notes
    -----
    The synthetic late panel is accepted at minute3 and released once inside
    the minute4 decision callback after the closed drain. Lambda is a declared
    development parameter, not an adaptive win, executed-book fee or risk cap.
    """
    calendar = reference("authored-ci-interval-clock")
    config = controlled_configuration(cadence=calendar, strength=strength)
    interval = ReleaseInterval(
        interval=reference("authored-ci-release-0"),
        start=NOW,
        end=NOW + timedelta(minutes=5),
        cutoff=NOW + timedelta(minutes=3),
        release_at=NOW + timedelta(minutes=4),
    )
    application = ControlledDirect(
        ControlledHedge(config, owner=reference("authored-ci-controller")),
        ControlSchedule(schema_version=1, calendar=calendar, intervals=(interval,)),
        DirectOptions(task=TASK, objective=OBJECTIVE, horizon=HORIZON),
    )
    host = ReplayMixture(application)
    policy = ReplayPolicy(
        schema_version=1,
        stream=TASK,
        lineage=reference("authored-mixture-ci-live"),
        parent=None,
        source=SOURCE,
        capability=host.capability,
        delivery="delayed",
        specialists=False,
    )
    replay = CausalReplay(host, policy, state_type=application.state_type, state_codec=application.binding.state_codec)
    replay.admit_history(history())
    first = replay.emit(replay.prepare(opportunity(application, 0, 0)), execution="offline_identity_v1")
    replay.admit_feedback((feedback(first, config, (0.0, 1.0, 0.5), visible=2),))
    replay.advance(eligible_through=NOW + timedelta(minutes=2), apply_at=NOW + timedelta(minutes=3))
    held = application.learner.last_outcome
    assert held is not None and held.disposition == "held"
    replay.emit(replay.prepare(opportunity(application, 1, 4, cutoff=3)), execution="offline_identity_v1")
    result = application.learner.last_outcome
    assert result is not None
    output = config.controls.output
    receipt_timing = replay.state.updates[0].application.feedback.batch.timing
    assert receipt_timing is not None and receipt_timing.received_at is not None
    return {
        "evidence": "constructed-synthetic-ci-full-information",
        "execution": "offline_identity_v1",
        "initial_coefficients": [item.value for item in config.controls.initial_prior.weights],
        "coefficients": [item.value for item in application.learner.weights],
        "decisions_D": len(replay.state.decisions),
        "accepted_panels_U": application.learner.update_count,
        "successful_releases_R": application.learner.release_count,
        "disposition": result.disposition,
        "output_profile": config.profile,
        "lambda_coefficient_units": output.strength if isinstance(output, L1MovementPenalty) else None,
        "feedback_received_at": receipt_timing.received_at.isoformat(),
        "actual_applied_at": replay.state.updates[0].application.apply_at.isoformat(),
        "release_at": interval.release_at.isoformat(),
        "state_fingerprint": application.export_state().fingerprint,
        "paid_cost": "not_measured",
        "executed_book_turnover": "not_measured",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
