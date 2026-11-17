"""Copied fixed launcher; principal components come only from installed wheels."""

from __future__ import annotations

import asyncio
import importlib.abc
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def boundaries():
    """Run real funded arithmetic while rejecting any outer transport import."""
    forbidden = ("coleman.application", "coleman.transports", "coleman.runner", "mcp", "numpy", "pyarrow")

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
                raise AssertionError("Portfolio domain crossed import boundary: " + fullname)
            return None

    guard = Guard()
    sys.meta_path.insert(0, guard)
    from datetime import UTC, datetime

    from coleman.core.identity import Reference
    from coleman.core.primitives import fingerprint
    from coleman.problems.portfolio import (
        AssetWeight,
        Book,
        BookSession,
        Exact,
        Fees,
        Position,
        PricePanel,
        PriceQuote,
        Target,
    )

    asset = Reference(id="authored-asset", version="1")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    book = Book(
        schema_version=1, reference=asset, cash=Exact.of(100), positions=(Position(asset=asset, units=Exact.of(0)),)
    )
    prices = PricePanel(
        schema_version=1,
        source=asset,
        source_fingerprint=fingerprint({"authored": True}),
        at=now,
        quotes=(PriceQuote(asset=asset, value=Exact.of(10), status="observed", event_at=now, available_at=now),),
    )
    target = Target(schema_version=1, reference=asset, weights=(AssetWeight(asset=asset, value=Exact.of(1)),))
    fees = Fees(
        schema_version=1,
        reference=asset,
        period=asset,
        declared_at=now,
        proportional=Exact.of("1/100"),
        recurring=Exact.of(0),
        status="observed",
    )
    result = BookSession(book).deliver(prices, target, fees, request=asset)
    assert result.net_value.fraction.numerator == 10000
    assert result.net_value.fraction.denominator == 101
    assert not any(
        any(name == blocked or name.startswith(blocked + ".") for blocked in forbidden) for name in sys.modules
    )
    sys.meta_path.remove(guard)
    return {"actual_net_value": str(result.net_value.fraction), "transport_imports": False}


def direct():
    """Resume actual installed composition with its full original pending state."""
    from coleman_synthetic_portfolio.experiment import Experiment, ExperimentState

    experiment = Experiment()
    for index in range(8):
        experiment.step(index)
    wire = experiment.export_state().to_json()
    restored = Experiment()
    restored.restore_state(ExperimentState.from_json(wire))
    assert restored.step(7).to_json() == experiment.step(7).to_json()
    for index in range(8, 16):
        assert experiment.step(index).to_json() == restored.step(index).to_json()
    assert experiment.export_state().to_json() == restored.export_state().to_json()
    return {
        "state": experiment.export_state().fingerprint,
        "periods": 16,
        "predictor_labels": 15,
        "complete_continuation": True,
    }


def approval():
    """Approve the actual installed public image explicitly in this trusted test."""
    from coleman_synthetic_portfolio.recipe import recipe

    from coleman.application.executor_capture import capture_executor_identity
    from coleman.core.identity import Reference
    from coleman.tasks.identity import ApprovedExecutor, HostBootstrapProfile

    selected = recipe()
    image = capture_executor_identity(selected)
    profile = HostBootstrapProfile(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="host_bootstrap",
        reference=Reference(id="public-synthetic-qualification", version="1"),
        executors=(ApprovedExecutor(recipe=selected, implementation=image.binding),),
    )
    return profile, image


def _state_attempt(open_store, commit):
    """Attempt once; only numerical SQLite contention or lock timeout is busy."""
    try:
        return commit(open_store())
    except TimeoutError:
        return "busy"
    except sqlite3.OperationalError as error:
        code = getattr(error, "sqlite_errorcode", None)
        if type(code) is not int or code < 0 or code & 0xFF not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            raise
        return "busy"


def durable(root):
    """Interrupt real staged math, then recover from an actual store ACK."""
    from coleman_synthetic_portfolio._state import Layout
    from coleman_synthetic_portfolio.recipe import CONFIGURATION, COORDINATE
    from coleman_synthetic_portfolio.sample import planning_request

    from coleman.application.tasks import TaskService
    from coleman.state.storage_contracts import ExecutionCompatibility
    from coleman.tasks.protocols import CoordinateContext, PreparationContext, decode_value

    profile, image = approval()
    service = TaskService(root / "workspace", bootstrap_profile=profile)
    preview = service.plan(planning_request())
    plan = service.saved_plan(preview.plan_id)
    coordinate = plan.coordinates[0]
    source = decode_value(COORDINATE, coordinate.body)
    context = CoordinateContext(
        PreparationContext(
            plan.fingerprint,
            plan.source,
            image.binding,
            ExecutionCompatibility(schema_version=1, runtime=image.runtime, owner_implementations=(image.binding,)),
            root,
        ),
        coordinate.fingerprint,
        coordinate.reference,
        coordinate.target,
        root / "owned-state",
    )
    layout = Layout(decode_value(CONFIGURATION, plan.configuration), source, context)
    store = layout.open(create=True)
    layout.initialize(store)
    halfway = layout.advance(store, lambda: None)
    assert halfway.frontier.position == 8
    original_head = store.read_head()
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("Actual cancellation after one unacknowledged period")

    try:
        layout.advance(store, interrupt)
    except InterruptedError:
        pass
    else:
        raise AssertionError("The actual staged operation must interrupt")
    assert store.read_head() == original_head
    assert not (context.artifact_directory / "coordinate-result.json").exists()
    reopened, resumed_checkpoint = layout.open_with_checkpoint()
    assert reopened is not None and resumed_checkpoint is not None
    assert resumed_checkpoint.experiment.next_index == 8
    # Competing independent store clients propose the same one-period command.
    # An explicit five-second busy outcome is valid; retry must be retained and
    # must execute no second mathematical body. No lock/resource fence changes.
    from concurrent.futures import ThreadPoolExecutor

    from coleman.core.primitives import fingerprint
    from coleman.state.local import LocalStateStore
    from coleman.state.storage_contracts import CommitIntent, CommitRequest

    binding = layout.binding
    head = reopened.read_head()
    request = CommitRequest(
        schema_version=1,
        expected=head,
        intent=CommitIntent(
            schema_version=1,
            generation=head.generation + 1,
            kind="transition",
            content_fingerprint=fingerprint({"from": 8, "through": 9, "source": source.fingerprint}),
        ),
    )
    bodies = []

    def transition(stage):
        bodies.append("actual-owned-body")
        stage.owner(binding).experiment.step(8)

    def concurrent(_):
        return _state_attempt(
            lambda: LocalStateStore.open(reopened.root),
            lambda independent: (
                independent.commit_transition(
                    request, (binding,), transition, compatibility=context.preparation.compatibility
                ).status
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = tuple(executor.map(concurrent, (0, 1)))
    assert statuses.count("new") == 1 and len(bodies) == 1

    def forbidden(_):
        raise AssertionError("An acknowledged duplicate cannot repeat mathematical work")

    retained_command = reopened.commit_transition(
        request, (binding,), forbidden, compatibility=context.preparation.compatibility
    )
    assert retained_command.status == "retained"
    assert layout.checkpoint(reopened).experiment.next_index == 9
    completed = layout.advance(reopened, lambda: None)
    retained, final_checkpoint = layout.reconcile_with_checkpoint(reopened)
    assert completed.frontier.position == retained.frontier.position == 16
    final = final_checkpoint.experiment
    from coleman_synthetic_portfolio.experiment import Experiment

    expected = Experiment()
    for index in range(16):
        expected.step(index)
    assert final.to_json() == expected.export_state().to_json()
    return {
        "state": final.fingerprint,
        "acknowledged_before_interruption": 8,
        "unacknowledged_period_reapplied": True,
        "retained_frontier": 16,
        "concurrent_writer_statuses": statuses,
        "concurrent_mathematical_bodies": len(bodies),
    }


def replayed():
    """Use the actual installed causal bridge and source-free consumer."""
    from datetime import timedelta

    from coleman_synthetic_portfolio import model
    from coleman_synthetic_portfolio.replay import (
        PortfolioReplayHost,
        PortfolioReplayState,
        feedback,
        opportunity,
        start,
    )

    from coleman.replay.codec import read_continuation

    host, replay = start()
    assert host.experiment.rows is None
    record = replay.emit(replay.prepare(opportunity(replay, 0)), execution="offline_identity_v1")
    for index in range(1, 16):
        replay.admit_feedback((feedback(record, model.PRICE_ROWS[index]),))
        at = model.START + timedelta(days=index)
        replay.advance(eligible_through=at, apply_at=at)
        if index == 8:
            saved = read_continuation(replay.export_continuation().to_json(), state_type=PortfolioReplayState)
            host = PortfolioReplayHost(source_fingerprint=host.experiment.source_fingerprint)
            replay = host.resume(saved)
        record = replay.emit(replay.prepare(opportunity(replay, index)), execution="offline_identity_v1")
    return {
        "state": host.experiment.export_state().fingerprint,
        "periods": 16,
        "actual_causal_engine": True,
        "future_tape_in_consumer": False,
    }


def _execute_request(plan_id, key):
    """Declare the same owned task with a stable actual execution request key."""
    from coleman.tasks.requests import TaskExecuteRequest

    return TaskExecuteRequest(
        schema_version=1, contract="coleman-tasks-v1", artifact_type="execute_request", request_id=key, plan_id=plan_id
    )


def _wait(service, identity):
    """Poll only within the declared120-second observation allowance."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            view = service.status(identity)
        except sqlite3.OperationalError as error:
            code = getattr(error, "sqlite_errorcode", 0)
            if not isinstance(code, int) or code & 0xFF not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise
            time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
            continue
        if not view.worker_active and view.job.state not in {"queued", "running"}:
            assert view.job.state == "completed", view.to_json()
            return view
        time.sleep(0.025)
    raise TimeoutError("Synthetic task did not finish within its original120-second fence")


def task(root):
    """Execute the same concrete portfolio task through the real shared facade."""
    from coleman_synthetic_portfolio.readback import read_full_task_result
    from coleman_synthetic_portfolio.recipe import FINAL_RESULT
    from coleman_synthetic_portfolio.sample import planning_request

    from coleman.application.tasks import TaskService
    from coleman.tasks.protocols import decode_value

    profile, _ = approval()
    service = TaskService(root, bootstrap_profile=profile)
    preview = service.plan(planning_request())
    request = _execute_request(preview.plan_id, "synthetic-same-original-request")
    started = service.execute(request)
    assert service.execute(request).job.job_id == started.job.job_id
    completed = _wait(service, started.job.job_id)
    result = decode_value(FINAL_RESULT, service.report(completed.job.job_id).result)
    state = read_full_task_result(service, completed.job.job_id)[0].experiment
    assert result.results[0].coordinate.experiment_fingerprint == state.fingerprint
    return {
        "state": state.fingerprint,
        "job": completed.job.job_id,
        "periods": state.next_index,
        "report": result.fingerprint,
    }


def cli(root):
    """Use actual installed console JSON planning/execution for the same source."""
    from coleman_synthetic_portfolio.readback import read_full_task_result
    from coleman_synthetic_portfolio.recipe import FINAL_RESULT
    from coleman_synthetic_portfolio.sample import planning_request

    from coleman.application.tasks import TaskService
    from coleman.artifacts import ArtifactWriter
    from coleman.tasks.protocols import decode_value
    from coleman.tasks.requests import TaskJobCommand
    from coleman.tasks.results import TaskCompletion

    profile, _ = approval()
    approval_path = root.parent / "cli-approval.json"
    ArtifactWriter(root.parent).write_immutable(approval_path, profile.to_json())
    prefix = [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "coleman.application.task_cli",
        "--workspace",
        str(root),
        "--bootstrap",
        str(approval_path),
    ]

    wires = {}

    def invoke(operation, body):
        completed = subprocess.run(
            [*prefix, operation, "--request", "-"], input=body, capture_output=True, text=True, check=False, timeout=60
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads(completed.stdout)
        assert result.get("artifact_type") != "failure", result
        wires[operation] = completed.stdout
        return result

    preview = invoke("plan", planning_request().to_json())
    start = invoke("execute", _execute_request(preview["plan_id"], "synthetic-cli-original").to_json())
    service = TaskService(root, bootstrap_profile=profile)
    completed = _wait(service, start["job"]["job_id"])
    command = TaskJobCommand(
        schema_version=1, contract="coleman-tasks-v1", artifact_type="job_command", job_id=completed.job.job_id
    ).to_json()
    before = invoke("status", command)
    completion = service.report(completed.job.job_id)
    reported = invoke("report", command)
    assert wires["report"] == completion.to_json() + "\n"
    assert TaskCompletion.model_validate(reported).fingerprint == before["job"]["result_digest"]
    after = invoke("status", command)
    assert before["job"] == after["job"] and after["job"]["state"] == "completed"
    assert not before["worker_active"] and not after["worker_active"]
    result = decode_value(FINAL_RESULT, completion.result)
    state = read_full_task_result(service, completed.job.job_id)[0].experiment
    assert result.results[0].coordinate.experiment_fingerprint == state.fingerprint
    return {
        "state": state.fingerprint,
        "periods": 16,
        "transport": "actual-cli",
        "report_transfer": "complete-compact-v2",
        "inline_completion_bytes": len(completion.to_json().encode()),
        "completed_digest": completion.fingerprint,
        "full_artifact_resolved": True,
    }


async def mcp(root):
    """Run the same approved task through the actual optional stdio transport."""
    from coleman_synthetic_portfolio.readback import read_full_task_result
    from coleman_synthetic_portfolio.recipe import FINAL_RESULT
    from coleman_synthetic_portfolio.sample import planning_request
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    from coleman.application.tasks import TaskService
    from coleman.artifacts import ArtifactWriter
    from coleman.tasks.protocols import decode_value
    from coleman.tasks.requests import TaskJobCommand
    from coleman.tasks.results import TaskCompletion

    profile, _ = approval()
    profile_path = root.parent / "approval.json"
    ArtifactWriter(root.parent).write_immutable(profile_path, profile.to_json())
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-I",
            "-B",
            "-m",
            "coleman.transports.task_mcp",
            "--workspace",
            str(root),
            "--bootstrap",
            str(profile_path),
        ],
        cwd=str(Path.cwd()),
    )

    def payload(reply):
        assert not reply.is_error
        value = json.loads(reply.content[0].text)
        assert value.get("artifact_type") != "failure", value
        return value

    async with Client(parameters) as client:
        tools = {item.name for item in (await client.list_tools()).tools}
        assert "task_plan" in tools and "task_report" in tools
        preview = payload(await client.call_tool("task_plan", {"request_json": planning_request().to_json()}))
        start = payload(
            await client.call_tool(
                "task_execute",
                {"request_json": _execute_request(preview["plan_id"], "synthetic-mcp-original").to_json()},
            )
        )
    service = TaskService(root, bootstrap_profile=profile)
    completed = _wait(service, start["job"]["job_id"])
    command = TaskJobCommand(
        schema_version=1, contract="coleman-tasks-v1", artifact_type="job_command", job_id=completed.job.job_id
    ).to_json()
    completion = service.report(completed.job.job_id)
    async with Client(parameters) as client:
        before = payload(await client.call_tool("task_status", {"command_json": command}))
        reply = await client.call_tool("task_report", {"command_json": command})
        reported = payload(reply)
        assert reply.content[0].text == completion.to_json()
        assert TaskCompletion.model_validate(reported).fingerprint == before["job"]["result_digest"]
        after = payload(await client.call_tool("task_status", {"command_json": command}))
        assert before["job"] == after["job"] and after["job"]["state"] == "completed"
        assert not before["worker_active"] and not after["worker_active"]
    result = decode_value(FINAL_RESULT, completion.result)
    state = read_full_task_result(service, completed.job.job_id)[0].experiment
    assert result.results[0].coordinate.experiment_fingerprint == state.fingerprint
    return {
        "state": state.fingerprint,
        "periods": 16,
        "transport": "actual-stdio",
        "report_transfer": "complete-compact-v2",
        "inline_completion_bytes": len(completion.to_json().encode()),
        "completed_digest": completion.fingerprint,
        "full_artifact_resolved": True,
    }


def main():
    """Dispatch a fixed local qualification mode, outside repository imports."""
    mode = sys.argv[1]
    root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()
    if mode == "boundaries":
        result = boundaries()
    elif mode == "direct":
        result = direct()
    elif mode == "durable":
        result = durable(root)
    elif mode == "replay":
        result = replayed()
    elif mode == "task":
        result = task(root)
    elif mode == "cli":
        result = cli(root)
    elif mode == "mcp":
        result = asyncio.run(mcp(root))
    else:
        raise ValueError("Unknown fixed qualification mode")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
