"""Construct the same frozen bounded typed task without running mathematics."""

from coleman.tasks.contracts import TaskCoordinate, TaskFrontier, TaskLimits
from coleman.tasks.protocols import encode_value
from coleman.tasks.requests import TaskPlanningRequest

from . import model
from .experiment import Experiment
from .recipe import CONFIGURATION, COORDINATE, INPUTS, binding, frontier_definition, recipe
from .task_contracts import PortfolioConfiguration, PortfolioInputs, PortfolioTrajectory


def planning_request(*, strength: float = 0.05) -> TaskPlanningRequest:
    """Declare one fixed authored source, work calendar and development strength.

    Parameters
    ----------
    strength : float, optional
        Previously development-selected nonnegative L1 coefficient strength.

    Returns
    -------
    TaskPlanningRequest
        Concrete complete input under the operator-approved portfolio recipe.
        Construction creates virgin pure owners only, with no observations,
        source acquisition, state store or job admission.
    """
    config = PortfolioConfiguration(schema_version=1, strength=strength)
    trajectory = PortfolioTrajectory(
        schema_version=1,
        reference=model.ref("task-trajectory"),
        initial_state=Experiment(strength=strength).export_state(),
        rows=model.PRICE_ROWS,
    )
    source = PortfolioInputs(
        schema_version=1, reference=model.ref("authored-task-source"), lineage=model.LINEAGE, trajectories=(trajectory,)
    )
    return TaskPlanningRequest(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="planning_request",
        executor=recipe().reference,
        configuration=encode_value(CONFIGURATION, config),
        inputs=encode_value(INPUTS, source),
        source=source.binding,
        objective=binding("descriptive-funded-path", {"unit": "synthetic-credit", "meaning": "own-book-net-value"}),
        protocol=None,
        stage=model.ref("authored-descriptive"),
        coordinates=(
            TaskCoordinate(
                reference=trajectory.reference,
                seed=None,
                body=encode_value(COORDINATE, trajectory),
                target=TaskFrontier(profile=frontier_definition(), coordinate=trajectory.reference, position=16),
            ),
        ),
        limits=TaskLimits(
            max_coordinates=1,
            max_work_units=32,
            max_attempts=2,
            max_invocations=3,
            max_output_bytes=16 * 1024**2,
            max_memory_bytes=512 * 1024**2,
            attempt_seconds=120.0,
        ),
    )
