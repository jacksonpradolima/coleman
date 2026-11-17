"""Run the actual copied public flow with the noneditable installed package."""

import importlib.util
import json
import sys
from pathlib import Path


def load(name, filename):
    specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if specification is None or specification.loader is None:
        raise RuntimeError("The explicitly copied public example is required")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def run():
    import coleman.core.base
    import coleman.optimization.continuous
    import coleman.prediction.ridge

    prefix = Path(sys.prefix).resolve()
    modules = (coleman.core.base, coleman.optimization.continuous, coleman.prediction.ridge)
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(prefix)
    flow = load("installed_actual_continuous_flow", "flow.py")
    allocation = load("installed_actual_continuous_allocation", "allocation.py")
    original = flow.ForecastOwner()
    first = flow.emit(original, 0)
    original.replay.admit_feedback((flow.feedback(first, 2.0),))
    second = flow.emit(original, 1)
    zero = flow.feedback(second, 0.0)
    original.replay.admit_feedback((zero, zero))
    pending = original.export_state()
    assert pending.host_state.ridge.payload.sample_count == 1
    assert len(pending.host_state.pending) == 1
    assert pending.host_state.pending[0].input.values == (1.0, 1.0)
    serialized = pending.to_json()
    from coleman.replay.retained_codec import read_retained_continuation

    decoded = read_retained_continuation(serialized, state_type=flow.MODEL.ForecastHostState)
    resumed = flow.ForecastOwner()
    resumed.restore_state(decoded)
    old_record = flow.emit(original, 2)
    new_record = flow.emit(resumed, 2)
    assert old_record.to_json() == new_record.to_json()
    assert original.export_state().to_json() == resumed.export_state().to_json()
    assert resumed.host.ridge.sample_count == 2
    assert resumed.host.export_state().pending[0].prediction.value.value == 1.2000000000000002
    worked = allocation.solve_continuous(allocation.problem(0.0, coupled_fixture=True))
    assert worked.executed == (0.5, 1.25)
    assert worked.certificate.value == -2.8125
    composed = allocation.run()
    assert len(composed["trace"]) == 20
    return {
        "installed_modules": [str(Path(module.__file__).resolve()) for module in modules],
        "outside": str(Path.cwd()),
        "real_updates_before_resume": 1,
        "real_updates_after_resume": resumed.host.ridge.sample_count,
        "complete_continuation_equal": True,
        "coupled_optimum": list(worked.executed),
        "composed": composed,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
