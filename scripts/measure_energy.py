"""Estimate energy and carbon impact for a coleman experiment run.

This script uses CodeCarbon to track estimated CO2 emissions during
an actual experiment execution.  It serves as a starting point for
comparing implementations through their estimated energy cost.

Usage
-----
.. code-block:: bash

    uv run python scripts/measure_energy.py
"""

from pathlib import Path

from codecarbon import EmissionsTracker

from coleman.api import load_spec, run

_REPO_ROOT = Path(__file__).resolve().parent.parent


def workload() -> None:
    """Run a coleman experiment as the measured workload."""
    spec = load_spec(_REPO_ROOT / "run.yaml")
    result = run(spec)
    print(f"Completed run_id={result.run_id}")


if __name__ == "__main__":
    tracker = EmissionsTracker()
    emissions: float = 0.0
    tracker.start()
    try:
        workload()
    finally:
        emissions = tracker.stop()
    print(f"Estimated emissions (kg CO2eq): {emissions}")
