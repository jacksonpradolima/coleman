"""Exercise the real public CI composition from a noneditable base wheel.

This file is copied beside the six-file public example outside the checkout.
Fault hooks run the actual component before terminating at a named boundary;
they never replace a learner, replay, domain receipt or acknowledgement.
"""

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import timedelta
from fractions import Fraction
from pathlib import Path


def _example():
    path = Path(__file__).with_name("ci.py")
    spec = importlib.util.spec_from_file_location("installed_prequential_ci", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def _read(path):
    return json.loads(path.read_text())


def _wait(path, *, seconds=600):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError("The explicit process boundary was not released: " + str(path))
        time.sleep(0.01)


def _recipe(example, frozen, mode, world):
    return example.MODEL.Recipe(
        mode,
        example.MODEL.configuration(mode),
        example.MODEL.ControlSchedule(schema_version=1, calendar=None),
        frozen.design.candidate(mode).candidate.configuration_fingerprint,
        frozen.design.snapshot.data_fingerprint,
        world,
    )


def _session(example, frozen, root, mode="penalized", world="world-0", *, create=False):
    return example.Session(root, _recipe(example, frozen, mode, world), frozen.environment, create=create)


def _body(example, root):
    return example.CachedExample.from_json((root / "completed-evidence.json").read_text())


def _originals(body, *, mode="penalized", world="world-0"):
    return tuple(
        observation
        for tick in sorted(body.tape.ticks, key=lambda tick: tick.ordinal)
        if tick.world == world
        for observation in tick.observations
        if observation.step.receipt.key.cell.candidate_id == mode
    )


def _complete_bytes(owner):
    return tuple(row.to_json() for row in owner.snapshots())


def _heads(example, root):
    return {
        mode + "/" + world: example.LocalStateStore.open(root / "sessions" / mode / world)
        .read_head()
        .model_dump(mode="json")
        for mode in example.MODEL.MODES
        for world in example.WORLDS
    }


def _source_history(example, root, frozen):
    return example.ProtocolAudit(root / "audit", limits=frozen.design.limits).sources.history(frozen.design.scope)


def _assert_installed():
    installed = Path(sys.prefix).resolve()
    paths = []
    for name, module in sys.modules.items():
        if name == "coleman" or name.startswith("coleman."):
            path = getattr(module, "__file__", None)
            if path:
                resolved = Path(path).resolve()
                assert resolved.is_relative_to(installed), (name, resolved, installed)
                assert not resolved.is_relative_to(Path.cwd())
                paths.append(str(resolved))
    assert paths
    blocked = ("numpy", "pandas", "polars", "pyarrow", "scipy", "mcp", "coleman.cli", "coleman.transports")
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in blocked)
    return len(paths)


def _scalar_oracle(example, tape):
    """Calculate the declared five scalar components independently with fractions."""
    previous, past, observed = {}, {}, {}
    checked = 0
    for tick in sorted(tape.ticks, key=lambda tick: (tick.world, tick.ordinal)):
        for original in tick.observations:
            step = original.step
            mode, world = step.receipt.key.cell.candidate_id, tick.world
            key = mode, world
            weights = tuple(row.value for row in original.coefficient.executed.action.weights)
            before = previous.get(key, (0.5, 0.5))
            movement = sum(abs(Fraction(new) - Fraction(old)) for new, old in zip(weights, before, strict=True))
            rate = 0 if mode == "accounting_off" else 4
            action = step.receipt.executed.action.chosen
            proposal = step.receipt.executed.proposal.action.chosen
            gross = Fraction(original.expert_seconds[example.MODEL.PLANS.index(action)])
            counterfactual = Fraction(original.expert_seconds[example.MODEL.PLANS.index(proposal)])
            expected = {
                "terminal_net_seconds": gross + rate * movement,
                "gross_seconds": gross,
                "reconfiguration_seconds": rate * movement,
                "coefficient_l1": movement,
                "proposal_without_intervention_seconds": counterfactual,
            }
            for component in step.components:
                assert component.value.status == "observed"
                assert component.value.value is not None
                assert math.isclose(component.value.value, float(expected[component.metric.name]), abs_tol=1e-13)
                unit = "1" if component.metric.name == "coefficient_l1" else "s"
                assert component.metric.quantity.spec.semantics.unit == unit
                assert component.metric.role != "physical_cost" and component.physical_group_id is None
                assert component.original.record.measurement(component.measurement_name).value == component.value.value
            assert len(step.components) == len(expected) == 5
            predictions = tuple(row.values[0].value for row in step.receipt.executed.proposal.request.member_inputs)
            history = past.get(key, [])
            forecast = (
                tuple(sum(Fraction(panel[index]) for panel in history) / len(history) for index in range(2))
                if mode == "online" and history
                else (Fraction(4), Fraction(4))
            )
            assert all(
                math.isclose(value, float(want), abs_tol=1e-13)
                for value, want in zip(predictions, forecast, strict=True)
            )
            if mode == "fixed":
                assert weights == (0.5, 0.5)
            observed[mode, world, tick.ordinal] = weights, action, predictions, float(rate * movement)
            previous[key] = weights
            past.setdefault(key, []).append(original.expert_seconds)
            checked += 1
    for world in example.WORLDS:
        for ordinal in range(3):
            penalized = observed["penalized", world, ordinal]
            uncharged = observed["accounting_off", world, ordinal]
            assert penalized[:3] == uncharged[:3]
            assert uncharged[3] == 0.0
            # Exogenous complete panels are identical: changing the domain's
            # predictor cannot retrospectively alter the coefficient learner.
            assert observed["online", world, ordinal][0] == penalized[0]
    assert checked == 108
    return checked


def full(example, root, *, crash=False):
    if crash:
        original_write = example._write

        def interrupted_write(directory, relative, value):
            original_write(directory, relative, value)
            if relative == "completed-evidence.json":
                assert (root / "prepared-result-locator.json").is_file()
                assert not (root / "completed-locator.json").exists()
                frozen, _ = example.frozen_declarations(root)
                history = _source_history(example, root, frozen)
                _save(
                    root.parent / "publication-boundary.json",
                    {
                        "report": value.report_fingerprint,
                        "heads": _heads(example, root),
                        "source_receipts": [receipt.fingerprint for receipt in history.receipts],
                    },
                )
                os._exit(73)

        example._write = interrupted_write
    result = example.run(root)
    frozen, resumed = example.frozen_declarations(root)
    assert resumed
    assert result.conformance.status == "pass"
    assert result.disposition.status == "descriptive_only"
    assert not result.disposition.eligible_candidates
    assert result.disposition.baseline.candidate_id == "fixed"
    assert len(result.report.panel.rows) == 36 and len(result.report.family.results) == 8
    assert not result.report.confirmatory_eligible
    assert {row.conclusion for row in result.report.family.results} == {"insufficient_evidence"}
    assert _scalar_oracle(example, result.tape) == 108
    assert example.SourceTape.from_json(result.tape.to_json()).to_json() == result.tape.to_json()
    snapshots = {}
    for mode in example.MODEL.MODES:
        for world in example.WORLDS:
            owner = _session(example, frozen, root / "sessions" / mode / world, mode, world)
            states = owner.snapshots()
            assert (states[0].state.decision_count, states[0].state.update_count, states[1].count) == (3, 3, 3)
            assert len(states[1].receipts) == len(states[1].observations) == 3
            snapshots[mode + "/" + world] = [hashlib.sha256(row.to_json().encode()).hexdigest() for row in states]
    before_heads = _heads(example, root)
    publication = _read(root.parent / "publication-boundary.json")
    assert publication["report"] == result.report.fingerprint
    assert publication["heads"] == before_heads
    before_access = _source_history(example, root, frozen)
    assert [receipt.fingerprint for receipt in before_access.receipts[:-1]] == publication["source_receipts"]
    assert before_access.receipts[-1].request.purpose == "result_disclosure"
    assert before_access.receipts[-1].request.report_fingerprint == result.report.fingerprint
    original_body = (root / "completed-evidence.json").read_bytes()
    cached = example.run(root)
    assert cached.report.to_json() == result.report.to_json()
    assert cached.tape.to_json() == result.tape.to_json()
    assert cached.disposition.to_json() == result.disposition.to_json()
    assert (root / "completed-evidence.json").read_bytes() == original_body
    assert _heads(example, root) == before_heads
    after_access = _source_history(example, root, frozen)
    assert after_access.receipts[:-1] == before_access.receipts
    assert after_access.receipts[-1].request.purpose == "result_disclosure"
    assert after_access.receipts[-1].request.report_fingerprint == result.report.fingerprint
    drifts = {}
    for drift in ("action", "cost", "time"):
        failed = example.assess_existing(frozen, result.report, result.traces, result.tape, drift=drift)
        assert failed.conformance.status == "fail"
        assert failed.disposition.status == "failed_conformance"
        assert failed.disposition.baseline == result.disposition.baseline
        assert failed.report.to_json() == result.report.to_json()
        drifts[drift] = failed.conformance.status
    proof = {
        "summary": result.summary,
        "original_report": result.report.fingerprint,
        "snapshots": snapshots,
        "scalar_oracle_decisions": 108,
        "reference_drift_diagnostics": drifts,
        "cached_all36_heads_unchanged": True,
        "publication_recovered_original": True,
        "publication_and_cached_reads_preserve_prefix_and_reserve_disclosures": True,
        "installed_modules": _assert_installed(),
    }
    _save(root.parent / "full-proof.json", proof)
    return proof


def prepare(example, root, case):
    frozen, _ = example.frozen_declarations(root)
    originals = _originals(_body(example, root))
    assert len(originals) == 3
    owner = _session(example, frozen, case / "store", create=True)
    at = example._timing(originals[0].step.receipt.executed.proposal.request).decision_at
    request = owner.command("emit", 0, at)
    (case / "emit-request.json").write_text(request.to_json())
    _save(case / "initial-head.json", owner.store.read_head().model_dump(mode="json"))
    return {"generation": owner.store.read_head().generation}


def _callback_log(case, role, event):
    with (case / (role + "-calls.log")).open("a") as stream:
        stream.write(event + "\n")


def emit(example, root, case, fault, *, role="single", divergent=False):
    frozen, _ = example.frozen_declarations(root)
    originals = _originals(_body(example, root))
    owner = _session(example, frozen, case / "store")
    request = example.CommitRequest.from_json((case / "emit-request.json").read_text())
    at = example._timing(originals[0].step.receipt.executed.proposal.request).decision_at
    if divergent:
        at += timedelta(microseconds=1)
        changed = owner.command("emit", 0, at)
        request = example.CommitRequest(schema_version=1, expected=request.expected, intent=changed.intent)
    if role == "loser":
        (case / "loser-ready").write_text("actual session opened before writer contention")
        _wait(case / "start-loser")
        (case / "loser-attempting").write_text("enter the actual store with the retained expected head")
    original_execute = example.MODEL.DomainOwner.execute

    def observed_execute(domain, context):
        _callback_log(case, role, "entered-domain")
        if fault == "coefficient":
            (case / "fault-barrier").write_text("actual coefficient emitted; domain not executed")
            os._exit(81)
        if fault == "hold":
            (case / "writer-held").write_text("actual store callback holds the acknowledgement lock")
            _wait(case / "release-writer")
        value = original_execute(domain, context)
        _callback_log(case, role, "completed-domain")
        if fault == "domain":
            (case / "fault-barrier").write_text("actual domain computed; joint state not acknowledged")
            os._exit(82)
        return value

    example.MODEL.DomainOwner.execute = observed_execute
    if fault == "ack":
        original_commit = example.LocalStateStore.commit_transition

        def acknowledged_then_exit(store, *args, **kwargs):
            receipt = original_commit(store, *args, **kwargs)
            assert receipt.status == "new"
            (case / "fault-barrier").write_text("actual joint ACK committed; response not returned")
            os._exit(83)

        example.LocalStateStore.commit_transition = acknowledged_then_exit
    try:
        result = owner.emit(0, at, request=request)
    except ValueError as error:
        if not divergent:
            raise
        assert "stale" in str(error) or "conflicts" in str(error)
        assert not (case / (role + "-calls.log")).exists()
        return {"rejected": str(error), "callbacks": 0}
    assert not divergent, "A different intent with the same stale predecessor cannot be acknowledged"
    assert (
        result.to_json()
        == example.PrequentialStep(
            schema_version=1, coefficient=originals[0].coefficient, domain=originals[0].step.receipt
        ).to_json()
    )
    head = owner.store.read_head()
    before = _complete_bytes(owner)
    assert owner.emit(0, at, request=request).to_json() == result.to_json()
    assert owner.store.read_head() == head and _complete_bytes(owner) == before
    return {"head": head.model_dump(mode="json"), "receipt": result.fingerprint}


def inspect(example, root, case):
    frozen, _ = example.frozen_declarations(root)
    owner = _session(example, frozen, case / "store")
    replay, domain = owner.snapshots()
    return {
        "head": owner.store.read_head().model_dump(mode="json"),
        "D": replay.state.decision_count,
        "U": replay.state.update_count,
        "domain_receipts": len(domain.receipts),
        "domain_observations": domain.count,
    }


def finish(example, root, case):
    frozen, _ = example.frozen_declarations(root)
    originals = _originals(_body(example, root))
    owner = _session(example, frozen, case / "store")
    states = owner.reproduce(originals)
    baseline = _session(example, frozen, root / "sessions" / "penalized" / "world-0")
    assert tuple(row.to_json() for row in states) == _complete_bytes(baseline)
    return {"complete_two_owner_bytes_equal": True, "D": states[0].state.decision_count, "U": states[1].count}


def main():
    command, root = sys.argv[1], Path(sys.argv[2])
    example = _example()
    if command in {"full", "full-fault"}:
        output = full(example, root, crash=command == "full-fault")
    else:
        case = Path(sys.argv[3])
        if command == "prepare":
            output = prepare(example, root, case)
        elif command == "inspect":
            output = inspect(example, root, case)
        elif command == "finish":
            output = finish(example, root, case)
        else:
            assert command == "emit"
            output = emit(
                example,
                root,
                case,
                sys.argv[4],
                role=sys.argv[5] if len(sys.argv) > 5 else "single",
                divergent=len(sys.argv) > 6 and sys.argv[6] == "divergent",
            )
    _assert_installed()
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
