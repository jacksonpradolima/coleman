"""Actual Hedge/replay and a complete independently authored finite CI owner."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

from coleman.core.base import CoreContract
from coleman.evaluation.prequential.contracts import DomainExecutionReceipt
from coleman.evaluation.prequential.execution import DomainStepPlan
from coleman.learning.direct import DirectMixture
from coleman.learning.reference import Hedge
from coleman.learning.replay import ReplayMixture
from coleman.replay.contracts import ReplayPolicy
from coleman.replay.engine import CausalReplay
from tests.evaluation.prequential.conformance_support import DATA, binding, changed, key, receipt


def learning_example():
    """Load the actual public CI declarations, never a replacement learner."""
    name = "prequential_execution_actual_learning_ci"
    if name not in sys.modules:
        source = Path(__file__).parents[3] / "examples/learning/ci.py"
        spec = importlib.util.spec_from_file_location(name, source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def session():
    """Construct the shipped method and actual existing causal engine."""
    ci = learning_example()
    config = ci.reference_configuration()
    direct = DirectMixture(
        Hedge(config, owner=ci.reference("prequential-driver-actual-hedge")),
        task=ci.TASK,
        objective=ci.OBJECTIVE,
        horizon=ci.HORIZON,
    )
    host = ReplayMixture(direct)
    replay = CausalReplay(
        host,
        ReplayPolicy(
            schema_version=1,
            stream=ci.TASK,
            lineage=ci.reference("prequential-driver-live"),
            parent=None,
            source=ci.SOURCE,
            capability=host.capability,
            delivery="delayed",
            specialists=False,
        ),
        state_type=direct.state_type,
        state_codec=direct.binding.state_codec,
    )
    replay.admit_history(ci.history())
    return ci, direct, replay


def step_plan(index=0):
    return DomainStepPlan(
        schema_version=1,
        key=key(decision_id="build-" + str(index)),
        adapter=binding("ci-adapter"),
        base_configuration_fingerprint="sha256:" + "b" * 64,
        data_fingerprint=DATA,
    )


class DriverDomainState(CoreContract):
    """Retain all actions and callback accounting, including deliberate failures."""

    receipts: tuple[DomainExecutionReceipt, ...] = ()
    calls: int = 0
    padding: str = ""


class FiniteDomain:
    """Execute a real 2/6-second finite model under a three-second bound."""

    def __init__(self, fault=None, state=None):
        self.state = DriverDomainState(schema_version=1) if state is None else state
        self.fault = fault

    def export_state(self):
        return self.state

    def restore_state(self, state):
        self.state = DriverDomainState.model_validate(state)

    def execution(self, opportunity):
        return next((row for row in self.state.receipts if row.key == opportunity), None)

    def execute(self, context):
        original = receipt(context.plan.key)
        coefficient = context.coefficient
        timing = coefficient.opportunity.request.timing
        assert timing is not None
        request = changed(
            original.executed.proposal.request,
            round=coefficient.opportunity.request.round,
            timing=timing,
        )
        if self.fault in {"clock", "target", "input"}:
            updates = {
                "clock": {"information_cutoff": timing.information_cutoff - timedelta(microseconds=1)},
                "target": {"target_end": timing.target_end + timedelta(microseconds=1)},
                "input": {"eligible_input_fingerprint": "sha256:" + "9" * 64},
            }[self.fault]
            request = changed(request, timing=changed(timing, **updates))
        if self.fault == "round":
            request = changed(request, round=request.round + 1)
        executed = changed(original.executed, proposal=changed(original.executed.proposal, request=request))
        values = {
            "adapter": context.plan.adapter,
            "base_configuration_fingerprint": context.plan.base_configuration_fingerprint,
            "data_fingerprint": context.plan.data_fingerprint,
            "coefficient_record_fingerprint": coefficient.fingerprint,
            "coefficient_execution_fingerprint": coefficient.executed.fingerprint,
            "prior_domain_state_fingerprint": context.prior_domain_state_fingerprint,
            "executed": executed,
            "executed_at": timing.decision_at,
        }
        faults = {
            "source": "data_fingerprint",
            "base": "base_configuration_fingerprint",
            "producer": "coefficient_record_fingerprint",
            "execution": "coefficient_execution_fingerprint",
            "prior": "prior_domain_state_fingerprint",
        }
        if self.fault in faults:
            values[faults[self.fault]] = "sha256:" + "9" * 64
        if self.fault == "adapter":
            values["adapter"] = binding("wrong-adapter")
        result = changed(original, **values)
        self.state = changed(
            self.state,
            calls=self.state.calls + 1,
            receipts=self.state.receipts if self.fault == "unretained" else (*self.state.receipts, result),
        )
        return result
