"""Copied installed-process helpers; no provider is imported before approval."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import coleman
from coleman.application.executor_capture import capture_executor_identity, load_approved_executor
from coleman.application.tasks import TaskService
from coleman.artifacts import ArtifactWriter
from coleman.core.identity import Reference
from coleman.learning.reference import Hedge
from coleman.state.local import LocalStateStore
from coleman.state.storage_contracts import CollectionManifest
from coleman.tasks.identity import ApprovedExecutor, ExecutorRecipe, HostBootstrapProfile
from coleman.tasks.requests import TaskExecuteRequest

JOB_OBSERVATION_SECONDS = 60
PROCESS_OBSERVATION_SECONDS = 180


def save(path: Path, value: object) -> None:
    """Publish one complete test observation without replacing prior evidence."""
    ArtifactWriter(path.parent).write_immutable(path, json.dumps(value, sort_keys=True, allow_nan=False))


def replace_callback(owner: object, name: str, callback: object) -> None:
    """Install a trusted host-only wrapper while leaving installed bytes intact."""
    setattr(owner, name, callback)


def run_checked(arguments: list[str], *, body: str | None = None):
    """Run one real consumer process under the finite180-second observation fence."""
    result = subprocess.run(
        arguments,
        cwd=Path.cwd(),
        input=body,
        capture_output=True,
        text=True,
        timeout=PROCESS_OBSERVATION_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def profile(path: Path) -> HostBootstrapProfile:
    """Read only a deliberately supplied current operator approval."""
    with path.open("rb") as stream:
        return HostBootstrapProfile.from_json(stream.read(16 * 1024 + 1))


def provider_loaded() -> bool:
    """Observe whether external provider/support code has actually been loaded."""
    return any(name.startswith(("coleman_typed_hedge", "coleman_typed_hedge_support")) for name in sys.modules)


def require_base() -> None:
    """Refuse an accidental optional CI/MCP/scientific installation in base cases."""
    for name in ("numpy", "polars", "pyarrow", "mcp"):
        assert importlib.util.find_spec(name) is None, name


def approve(recipe_path: Path, target: Path) -> dict[str, Any]:
    """Capture actual installed files without importing or approving code from SQL."""
    assert not provider_loaded()
    with recipe_path.open("rb") as stream:
        recipe = ExecutorRecipe.from_json(stream.read(16 * 1024 + 1))
    image = capture_executor_identity(recipe)
    assert not provider_loaded()
    current = HostBootstrapProfile(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="host_bootstrap",
        reference=Reference(id="installed-task-qualification", version="1"),
        executors=(ApprovedExecutor(recipe=recipe, implementation=image.binding),),
    )
    ArtifactWriter(target.parent).write_immutable(target, current.to_json())
    ArtifactWriter(target.parent).write_immutable(target.with_suffix(".image.json"), image.to_json())
    return {
        "profile": current.fingerprint,
        "image": image.fingerprint,
        "distributions": [item.name for item in image.distributions],
        "files": len(image.files),
        "provider_loaded": provider_loaded(),
    }


def sample(current: HostBootstrapProfile):
    """Validate actual image/origins before importing the supplied public sample."""
    assert len(current.executors) == 1
    approval = current.executors[0]
    image = capture_executor_identity(approval.recipe)
    load_approved_executor(approval, expected=image)
    return importlib.import_module("coleman_typed_hedge.sample")


def command(plan_id: str, key: str) -> TaskExecuteRequest:
    """Use actual public immutable-plan execution requests with stable caller keys."""
    return TaskExecuteRequest(
        schema_version=1,
        contract="coleman-tasks-v1",
        artifact_type="execute_request",
        request_id=key,
        plan_id=plan_id,
    )


def prepare(root: Path, approval_path: Path, key: str) -> dict[str, Any]:
    """Admit real work while a trusted test controls only the startup callback."""
    current = profile(approval_path)
    supplied = sample(current)
    service = TaskService(root, bootstrap_profile=current)
    preview = service.plan(supplied.planning_request())
    replace_callback(service, "_start", lambda: None)
    request = command(preview.plan_id, key)
    view = service.execute(request)
    assert view.job.state == "queued"
    assert view.job.attempts == 0
    assert view.job.invocations == 0
    return {"job_id": view.job.job_id, "plan_id": preview.plan_id, "request": request.to_json()}


def wait(service: TaskService, identity: str):
    """Observe one real job under the unchanged60-second finite task allowance."""
    deadline = time.monotonic() + JOB_OBSERVATION_SECONDS
    while time.monotonic() < deadline:
        try:
            view = service.status(identity)
        except sqlite3.OperationalError as error:
            code = getattr(error, "sqlite_errorcode", 0)
            if not isinstance(code, int) or code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
        else:
            if view.job.state not in {"queued", "running"} and not view.worker_active:
                assert view.job.state == "completed", view.to_json()
                return view
        time.sleep(0.025)
    raise TimeoutError("Installed task exceeded its declared60-second observation window")


def _selected_checkpoint(directory: Path, head):
    """Read only the canonical bytes selected by one actual acknowledged head."""
    content = (directory / "manifests" / f"{head.manifest_fingerprint[7:]}.json").read_bytes()
    assert "sha256:" + sha256(content).hexdigest() == head.manifest_fingerprint
    manifest = CollectionManifest.from_json(content)
    assert manifest.intent.generation == head.generation
    assert len(manifest.components) == 1
    selected = manifest.components[0]
    blob = (directory / "blobs" / f"{selected.fingerprint[7:]}.json").read_bytes()
    assert len(blob) == selected.byte_count
    assert "sha256:" + sha256(blob).hexdigest() == selected.fingerprint
    checkpoint = json.loads(blob)
    return manifest, {
        "generation": head.generation,
        "manifest": head.manifest_fingerprint,
        "component": selected.fingerprint,
        "component_bytes": selected.byte_count,
        "U": checkpoint["learner"]["payload"]["update_count"],
        "learner": json.dumps(checkpoint["learner"], sort_keys=True, separators=(",", ":")),
        "checkpoint": blob.decode(),
    }


def _selected_chain(directory: Path, head):
    """Follow the explicit bounded ancestry of a selected head, never a glob."""
    chain = []
    for _ in range(10):
        manifest, checkpoint = _selected_checkpoint(directory, head)
        chain.append(checkpoint)
        if manifest.predecessor is None:
            break
        head = manifest.predecessor
    else:
        raise AssertionError("Selected fixture ancestry exceeds its at-most8-panel bound")
    chain.reverse()
    return chain


def state_snapshot(root: Path, identity: str) -> dict[str, Any]:
    """Follow the ACTUAL current ACK and its selected canonical manifest chain.

    Notes
    -----
    File existence never selects an ACK. The real LocalStateStore reads its
    acknowledged head first; only that head and explicit predecessors select
    manifests/components. This inspection restores no owner or learner.
    """
    service = TaskService(root)
    job = service.status(identity).job
    plan = service.saved_plan(job.plan_id)
    assert len(plan.coordinates) == 1
    directory = root / "jobs" / identity / "evidence" / plan.coordinates[0].fingerprint[7:] / "state"
    head = LocalStateStore.open(directory).read_head()
    assert head is not None
    return {
        "head": json.loads(head.to_json()),
        "chain": _selected_chain(directory, head),
        "job": json.loads(job.to_json()),
    }


def _direct_states(supplied):
    """Run every original source panel through fresh actual direct Hedge owners."""
    source = supplied.inputs(supplied.configuration())
    direct = []
    for trajectory in source.trajectories:
        learner = Hedge(trajectory.initial_state.payload.configuration, owner=trajectory.reference)
        for panel in trajectory.panels:
            assert learner.observe(panel).status == "accepted"
        direct.append(learner.export_state().to_json())
    return direct


def _independent_weights(weights) -> None:
    """Check the frozen sample's independent exponential-weights formula."""
    totals = (1.25, 1.75, 1.5)
    masses = tuple(prior * math.exp(-math.log(2) * loss) for prior, loss in zip((0.5, 0.25, 0.25), totals, strict=True))
    expected = tuple(value / math.fsum(masses) for value in masses)
    for actual, wanted in zip(weights, expected, strict=True):
        assert math.isclose(actual.value, wanted, rel_tol=0.0, abs_tol=1e-13)


def inspect_complete(root: Path, approval_path: Path, identity: str) -> dict[str, Any]:
    """Compare COMPLETE actual durable/direct Hedge state, plus an independent formula."""
    current = profile(approval_path)
    supplied = sample(current)
    service = TaskService(root, bootstrap_profile=current)
    view = wait(service, identity)
    outer = service.report(identity)
    report = cast(Any, service.decode_result(identity))
    direct = _direct_states(supplied)
    assert tuple(item.checkpoint.learner.to_json() for item in report.results) == tuple(direct)
    assert report.interpretation == "constructed_mechanism_only"
    snapshot = state_snapshot(root, identity)
    assert [item["U"] for item in snapshot["chain"]] == [0, 2, 3]
    assert [item["generation"] for item in snapshot["chain"]] == [0, 1, 2]
    assert snapshot["chain"][-1]["learner"] == direct[0]
    assert report.results[0].proof.observed_head.to_json() == json.dumps(
        snapshot["head"], sort_keys=True, separators=(",", ":")
    )
    _independent_weights(report.results[0].checkpoint.learner.payload.weights)
    result = {
        "job_id": identity,
        "plan": outer.plan_fingerprint,
        "result_digest": view.job.result_digest,
        "state_equal": True,
        "updates": 3,
        "generation": 2,
        "attempts": view.job.attempts,
        "invocations": view.job.invocations,
        "interpretation": report.interpretation,
        "snapshot": snapshot,
        "direct_state": direct[0],
        "outer": outer.to_json(),
    }
    save(root.parent / f"{root.name}-{view.job.job_id}-complete-proof.json", result)
    return result


def execute(root: Path, approval_path: Path, key: str) -> dict[str, Any]:
    """Run the exact copied PUBLIC example, then inspect full actual evidence."""
    require_base()
    assert key == "installed-base"
    process = run_checked(
        [sys.executable, "-I", "-B", str(Path(__file__).with_name("ci.py")), "run", str(root), str(approval_path)]
    )
    example = json.loads(process.stdout)
    assert example["complete_direct_state_equal"] is True
    assert example["updates"] == [3]
    assert example["generations"] == [2]
    result = inspect_complete(root, approval_path, example["job_id"])
    result["public_example"] = example
    assert result["attempts"] == 1
    assert result["invocations"] == 1
    require_base()
    return result


def inspect_single(root: Path, approval_path: Path) -> dict[str, Any]:
    """Inspect the actual one-job workspace created by all four literal guide blocks."""
    service = TaskService(root)
    with closing(service.store.connect()) as connection:
        jobs = service.store.records(connection)
    assert len(jobs) == 1
    return inspect_complete(root, approval_path, jobs[0].job_id)


def provider_free(root: Path, approval_path: Path, identity: str) -> dict[str, Any]:
    """Refuse missing/wrong queue approval without loading or importing a provider."""
    assert not provider_loaded()
    service = TaskService(root)
    original = service.status(identity).job
    assert original.state == "queued"
    try:
        service.reconnect()
    except ValueError as error:
        assert "explicit current" in str(error)
    else:
        raise AssertionError("Stored SQL approval must not schedule a task")
    current = profile(approval_path)
    wrong = HostBootstrapProfile.model_validate(
        current.model_copy(update={"reference": Reference(id="wrong-current-operator", version="1")})
    )
    try:
        TaskService(root, bootstrap_profile=wrong)
    except ValueError as error:
        assert "cannot be silently replaced" in str(error)
    else:
        raise AssertionError("A different supplied approval must not schedule the retained queue")
    after = service.status(identity).job
    assert after.to_json() == original.to_json()
    assert not provider_loaded()
    return {
        "provider_loaded": False,
        "job": json.loads(after.to_json()),
        "missing_refused": True,
        "wrong_refused": True,
    }


def _archive_bytes(wheel: Path, prefix: str, distribution, virtual: Path):
    """Compare every package-resource member with its noneditable installed byte."""
    files = {}
    with ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.startswith(prefix) and not member.endswith("/"):
                path = Path(str(distribution.locate_file(member))).resolve(strict=True)
                assert path.is_relative_to(virtual.resolve())
                raw = path.read_bytes()
                assert raw == archive.read(member)
                files[member] = sha256(raw).hexdigest()
    assert files
    return files


def _installed_distribution(wheel: Path, name: str, prefix: str, virtual: Path):
    """Bind actual distribution metadata, source bytes and its built wheel."""
    distribution = metadata.distribution(name)
    direct_text = distribution.read_text("direct_url.json")
    assert direct_text is not None
    direct = json.loads(direct_text)
    assert direct.get("dir_info", {}).get("editable", False) is False
    return {
        "name": name,
        "version": distribution.version,
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
        "files": _archive_bytes(wheel, prefix, distribution, virtual),
        "direct_url": direct,
    }


def installation(wheels: tuple[Path, ...], virtual: Path) -> dict[str, Any]:
    """Verify actual noneditable package bytes against all three built archives."""
    require_base()
    assert Path(coleman.__file__).resolve().is_relative_to(virtual.resolve())
    packages = (
        ("coleman", "coleman/"),
        ("coleman-typed-hedge-reference", "coleman_typed_hedge/"),
        ("coleman-typed-hedge-support", "coleman_typed_hedge_support/"),
    )
    observed = [
        _installed_distribution(wheel, name, prefix, virtual)
        for wheel, (name, prefix) in zip(wheels, packages, strict=True)
    ]
    assert not Path.cwd().is_relative_to(Path(coleman.__file__).resolve().parent)
    return {
        "noneditable": True,
        "base_only": True,
        "distributions": observed,
        "cwd": str(Path.cwd()),
        "python": sys.executable,
    }
