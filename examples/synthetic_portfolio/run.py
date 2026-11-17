"""Run or declare the authored public synthetic portfolio example."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import Experiment
from .recipe import recipe
from .reporting import publish
from .sample import planning_request
from .selection import develop, run_protected


def main() -> None:
    """Run the direct example or print concrete optional host declarations.

    Notes
    -----
    Operator-selected paths are trusted CLI arguments. ``--recipe`` and
    ``--task-request`` emit declarations without executing observations or
    approving installed code. Default execution writes immutable local modeled
    artifacts only; ``--protected`` also records actual audit access.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic-portfolio-artifacts"))
    parser.add_argument("--recipe", action="store_true")
    parser.add_argument("--task-request", action="store_true")
    parser.add_argument("--protected", action="store_true")
    args = parser.parse_args()
    if args.recipe:
        print(recipe().to_json())
        return
    if args.task_request:
        print(planning_request().to_json())
        return
    if args.protected:
        state = run_protected(develop(), args.output / "protected").state
    else:
        experiment = Experiment()
        for index in range(16):
            experiment.step(index)
        state = experiment.export_state()
    print(publish(state, args.output))


if __name__ == "__main__":
    main()
