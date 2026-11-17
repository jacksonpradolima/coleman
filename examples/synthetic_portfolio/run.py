"""Run or declare the authored public synthetic portfolio example."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from .experiment import Experiment
from .recipe import recipe
from .reporting import publish
from .sample import planning_request
from .selection import develop, run_protected


def main() -> None:
    """Run the direct example or print concrete optional host declarations.

    Raises
    ------
    SystemExit
        Click handles help or invalid flags before experiment execution. A
        missing optional CLI extra exits with installation guidance.

    Notes
    -----
    Operator-selected paths are trusted CLI arguments. ``--recipe`` and
    ``--task-request`` emit declarations without executing observations or
    approving installed code. Default execution writes immutable local modeled
    artifacts only; ``--protected`` also records actual audit access. Click is
    imported only on invocation; mathematical/library imports need no CLI extra.
    Combined declaration flags preserve recipe, then request precedence.
    """
    try:
        click = import_module("click")
    except ImportError as error:
        raise SystemExit("Install the reviewed COLEMAN wheel with its [cli] extra") from error

    @click.option(
        "--output", type=click.Path(path_type=Path), default=Path("synthetic-portfolio-artifacts"), show_default=True
    )
    @click.option("--recipe", "emit_recipe", is_flag=True)
    @click.option("--task-request", is_flag=True)
    @click.option("--protected", is_flag=True)
    def _command(output: Path, emit_recipe: bool, task_request: bool, protected: bool) -> None:
        """Preserve declarations and the same complete sixteen-period execution."""
        if emit_recipe:
            print(recipe().to_json())
            return
        if task_request:
            print(planning_request().to_json())
            return
        if protected:
            state = run_protected(develop(), output / "protected").state
        else:
            experiment = Experiment()
            for index in range(16):
                experiment.step(index)
            state = experiment.export_state()
        print(publish(state, output))

    command = click.command(help=__doc__)(_command)
    command.main()


if __name__ == "__main__":
    main()
