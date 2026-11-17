"""Exercise installed shipped and independent host consumers outside the checkout."""

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

from coleman.core.capabilities import ProblemCapabilities
from coleman.core.feedback import ChosenFeedback, FeedbackRow, OutcomeValue
from coleman.learning.discrete import DiscreteBandit
from coleman.testing.decision_contracts import ConformanceCase, check_decision_contracts


def _load(name, path):
    """Load an explicit copied public example despite isolated script startup."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Installed driver requires the copied public example")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    """Run actual installed algorithms and independent host through one harness."""
    local = Path(__file__).parent
    ci = _load("installed_discrete_ci", local / "ci.py")
    host = _load("installed_independent_host_mab", local / "host_mab.py")
    receipts = []
    for profile in ("exp3-dyadic53-v1", "ucb1-binary64-v1"):
        config = ci.configuration(profile)
        factory = partial(DiscreteBandit, config)
        problem = ProblemCapabilities(
            schema_version=1,
            space=config.space,
            feedback_kind="chosen",
            values=(config.reward,),
            origins=("synthetic",),
            timing_mode="untimed",
            delayed_feedback=False,
            sampling_statuses=("exact",),
            receipt_status="unknown",
            state_owners=("learner",),
        )
        cases = tuple(
            ConformanceCase(
                problem=problem,
                request=ci.request(config, index),
                outcomes=tuple(
                    FeedbackRow(
                        member=member,
                        values=(
                            OutcomeValue(
                                spec=config.reward,
                                value=value,
                                status="observed",
                                origin="synthetic",
                                transform=config.transform,
                            ),
                        ),
                    )
                    for member, value in zip(config.space.members, (0.0, 1.0), strict=True)
                ),
            )
            for index in range(8)
        )
        report = check_decision_contracts(factory, factory().capability, cases)
        assert all(isinstance(batch.feedback, ChosenFeedback) for batch in report.feedback)
        receipts.append({"profile": profile, "shared_conformance_cases": len(report.feedback)})
    independent = check_decision_contracts(host.HostMAB, host.capability(), host.cases())
    print(json.dumps({"shipped": receipts, "independent_host_cases": len(independent.feedback)}, sort_keys=True))


if __name__ == "__main__":
    main()
