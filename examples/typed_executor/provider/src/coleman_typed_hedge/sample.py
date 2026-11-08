"""Small authored CI loss table and actual task request, without runtime capture."""

from math import log

from coleman.core.actions import Weight
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.numeric import ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.learning.contracts import ReferenceConfig
from coleman.learning.reference import Hedge
from coleman.objectives.bindings import DefinitionLineage
from coleman.objectives.preferences import COEFFICIENT_SEMANTICS, InitialPrior
from coleman.tasks.contracts import TaskCoordinate, TaskFrontier, TaskLimits
from coleman.tasks.protocols import encode_value
from coleman.tasks.requests import TaskPlanningRequest

from .contracts import HedgeConfiguration, HedgeInputs, HedgeTrajectory
from .recipe import CONFIGURATION, COORDINATE, INPUTS, binding, frontier_definition, recipe, reference

LOSS_TABLE = ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.25, 0.75, 0.5))


def configuration() -> HedgeConfiguration:
    """Bind an actual constant Hedge rule to explicit bounded synthetic losses.

    Returns
    -------
    HedgeConfiguration
        Prior(1/2,1/4,1/4), eta=log2, full-information unit-one losses[0,1].
        Two real panels can be accepted in one storage publication.
    """
    lineage = DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None)
    prior = InitialPrior(
        schema_version=1,
        reference=reference("ci-positive-prior"),
        lineage=lineage,
        weights=tuple(
            Weight(member=reference("expert-" + name), value=value)
            for name, value in zip(("a", "b", "c"), (0.5, 0.25, 0.25), strict=True)
        ),
        profile="binary64-simplex-v1",
        semantics=COEFFICIENT_SEMANTICS,
    )
    loss = ScalarSpec(
        dimension=reference("ci-normalized-loss"),
        role="loss",
        minimum=0.0,
        maximum=1.0,
        semantics=UnitSemantics(unit="1", meaning="constructed-full-information-loss-v1"),
    )
    return HedgeConfiguration(
        contract="coleman-typed-hedge-reference-v1",
        schema_version=1,
        reference=ReferenceConfig(
            schema_version=1,
            reference=reference("ci-constant-hedge"),
            lineage=lineage,
            profile="hedge-constant",
            prior=prior,
            loss_spec=loss,
            transform=binding("already-bounded-identity", {"operation": "already-bounded-identity-v1"}),
            eta=log(2.0),
        ),
        panels_per_commit=2,
    )


def inputs(config: HedgeConfiguration | None = None) -> HedgeInputs:
    """Create exact public source data with explicit ephemeral virgin snapshots.

    Parameters
    ----------
    config : HedgeConfiguration or None, optional
        Actual selected constant rule. None selects the public authored example.

    Returns
    -------
    HedgeInputs
        One independent trajectory with three explicit original full-loss panels.

    Notes
    -----
    Constructs a direct learner only to export its actual immutable U0 snapshot.
    No owner is retained, no store is opened and no runtime/provider is captured.
    """
    selected = configuration() if config is None else HedgeConfiguration.model_validate(config)
    owner = reference("ci-trajectory")
    panels = tuple(
        FullPanel(
            rows=tuple(
                FeedbackRow(
                    member=weight.member,
                    values=(
                        OutcomeValue(
                            spec=selected.reference.loss_spec,
                            value=value,
                            status="observed",
                            origin="synthetic",
                            transform=selected.reference.transform.reference,
                        ),
                    ),
                )
                for weight, value in zip(selected.reference.prior.weights, losses, strict=True)
            )
        )
        for losses in LOSS_TABLE
    )
    trajectory = HedgeTrajectory(
        contract="coleman-typed-hedge-reference-v1",
        schema_version=1,
        reference=owner,
        initial_state=Hedge(selected.reference, owner=owner).export_state(),
        panels=panels,
    )
    return HedgeInputs(
        contract="coleman-typed-hedge-reference-v1",
        schema_version=1,
        reference=reference("ci-loss-table"),
        lineage=DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
        trajectories=(trajectory,),
    )


def planning_request() -> TaskPlanningRequest:
    """Encode the complete public example without approving or loading code.

    Returns
    -------
    TaskPlanningRequest
        Actual typed request for an independently approved external recipe, with
        no package installation, path, URL or executable request parameter.
    """
    config = configuration()
    source = inputs(config)
    coordinates = tuple(
        TaskCoordinate(
            reference=item.reference,
            body=encode_value(COORDINATE, item),
            seed=None,
            target=TaskFrontier(profile=frontier_definition(), coordinate=item.reference, position=len(item.panels)),
        )
        for item in source.trajectories
    )
    return TaskPlanningRequest(
        contract="coleman-tasks-v1",
        schema_version=1,
        artifact_type="planning_request",
        executor=recipe().reference,
        configuration=encode_value(CONFIGURATION, config),
        inputs=encode_value(INPUTS, source),
        source=source.binding,
        objective=binding("ci-loss-objective", {"unit": "1", "goal": "lower-constructed-normalized-loss"}),
        protocol=None,
        stage=reference("constructed-descriptive"),
        coordinates=coordinates,
        limits=TaskLimits(
            max_coordinates=1,
            max_work_units=6,
            max_attempts=2,
            max_invocations=3,
            attempt_seconds=60.0,
            max_output_bytes=4 * 1024**2,
        ),
    )
