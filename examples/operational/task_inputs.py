"""Construct public authored CI tapes for the existing typed-Hedge provider.

The original small/larger workloads are declarations, never measurements of
physical work. Provider imports occur only during an explicit operator run.
"""

from __future__ import annotations

import importlib

from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference
from coleman.learning.reference import Hedge
from coleman.objectives.bindings import DefinitionLineage
from coleman.tasks.contracts import TaskCoordinate, TaskFrontier
from coleman.tasks.protocols import encode_value
from coleman.tasks.requests import TaskPlanningRequest


def _tape(config, panels):
    """Declare original alternating synthetic CI labels, never measured work."""
    return tuple(
        FullPanel(
            rows=tuple(
                FeedbackRow(
                    member=weight.member,
                    values=(
                        OutcomeValue(
                            spec=config.reference.loss_spec,
                            value=float((period + index) % 2),
                            status="observed",
                            origin="synthetic",
                            transform=config.reference.transform.reference,
                        ),
                    ),
                )
                for index, weight in enumerate(config.reference.prior.weights)
            )
        )
        for period in range(panels)
    )


def _source(size, example):
    """Construct only the existing provider's bounded concrete source values."""
    contracts = importlib.import_module("coleman_typed_hedge.contracts")
    experts, trajectories, panels = (2, 1, 1) if size == "small" else (8, 4, 8)
    config = contracts.HedgeConfiguration(
        schema_version=1,
        contract="coleman-typed-hedge-reference-v1",
        reference=example.configuration(experts),
        panels_per_commit=2,
    )
    tape = _tape(config, panels)
    owners = tuple(Reference(id=f"operational-trajectory-{index}", version="1") for index in range(trajectories))
    source = contracts.HedgeInputs(
        schema_version=1,
        contract="coleman-typed-hedge-reference-v1",
        reference=Reference(id="operational-authored-ci-source", version="1"),
        lineage=DefinitionLineage(source=None, source_fingerprint=None, declared_at=None, supersedes=None),
        trajectories=tuple(
            contracts.HedgeTrajectory(
                schema_version=1,
                contract="coleman-typed-hedge-reference-v1",
                reference=owner,
                initial_state=Hedge(config.reference, owner=owner).export_state(),
                panels=tape,
            )
            for owner in owners
        ),
    )
    return config, source


def _direct(source, config):
    """Calculate full direct states from the exact same original source tape."""
    states = []
    for item in source.trajectories:
        learner = Hedge(config.reference, owner=item.reference)
        for panel in item.panels:
            if learner.observe(panel).status != "accepted":
                raise ValueError("Direct learner refused original constructed feedback")
        states.append(learner.export_state().to_json())
    return tuple(states)


def planning_request(size, example, limits):
    """Encode one original public CI workload using existing provider codecs.

    Parameters
    ----------
    size : str
        Declared small/larger workload, preserving its trajectories and panels.
    example : module
        Explicitly copied operational CI module providing real Hedge settings.
    limits : TaskLimits
        Unchanged selected reservations from the finite experiment profile.

    Returns
    -------
    tuple
        Concrete TaskPlanningRequest and independently executed direct-state
        JSON strings in original trajectory order. No task is admitted here.

    Raises
    ------
    ImportError, ValueError
        Required installed provider is absent, its codecs reject the original
        typed inputs, or the real direct learner refuses authored feedback.
    """
    recipe = importlib.import_module("coleman_typed_hedge.recipe")
    config, source = _source(size, example)
    coordinates = tuple(
        TaskCoordinate(
            reference=item.reference,
            body=encode_value(recipe.COORDINATE, item),
            seed=None,
            target=TaskFrontier(
                profile=recipe.frontier_definition(), coordinate=item.reference, position=len(item.panels)
            ),
        )
        for item in source.trajectories
    )
    request = TaskPlanningRequest(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="planning_request",
        executor=recipe.recipe().reference,
        configuration=encode_value(recipe.CONFIGURATION, config),
        inputs=encode_value(recipe.INPUTS, source),
        source=source.binding,
        objective=recipe.binding("operational-constructed-loss", {"unit": "1", "goal": "descriptive-only"}),
        protocol=None,
        stage=Reference(id="operational-mechanism", version="1"),
        coordinates=coordinates,
        limits=limits,
    )
    return request, _direct(source, config)
