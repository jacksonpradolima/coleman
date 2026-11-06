import pytest

from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.native.decisions import (
    OBSERVATION_SEMANTICS,
    FrozenNativeDecision,
    NativeDecisionContext,
    NativeLossUndefined,
    NativeSamplingUnavailable,
    native_binding,
)


@pytest.fixture
def context():
    return NativeDecisionContext(
        schema_version=1,
        run=Reference(id="ci.example", version="1"),
        execution_id="execution-1",
        decision_id="build-1",
        purpose="evaluation_ranking",
        expected_round=0,
        binding=native_binding("EXP3", "RNFail"),
        configuration_fingerprint=fingerprint({"gamma": 0.2, "reward": "RNFail", "seed": 17}),
        input_fingerprint=fingerprint({"build": 1, "tests": ["A", "B", "C"]}),
        producer_fingerprint=fingerprint({"fixture_producer": "authored-contract-v1"}),
        policy_state_group=None,
    )


@pytest.fixture
def record(context):
    return FrozenNativeDecision(
        schema_version=1,
        context=context,
        candidates=("A", "B", "C"),
        ranking=("B", "A", "C"),
        observation_round=0,
        observation_semantics=OBSERVATION_SEMANTICS,
        window_size=None,
        phase="initial_shuffle",
        sampling=NativeSamplingUnavailable(status="not_recorded", value=None, reason="Native mass was not retained"),
        loss=NativeLossUndefined(status="not_defined", value=None, reason="No native generic loss is defined"),
    )
