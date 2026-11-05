"""Thin host setup and durable workflow CLI; native execution stays in shared services."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from coleman.application.baseline_storage import read_baseline_artifact
from coleman.application.workflow_registry import owned_fixture, snapshot_history
from coleman.application.workflows import WorkflowService
from coleman.results.domain import canonical_json
from coleman.workflows.contracts import DatasetManifest, PlanningRequest
from coleman.workflows.lifecycle import ExecutePlan


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add trusted setup and shared service operations to the existing CLI.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Existing ``coleman workflow`` subparser. Paths belong to trusted host setup,
        never per-request MCP arguments. JSON commands use the same strict contracts.
    """
    parser.add_argument("--workspace", type=Path, required=True, help="Trusted persistent local workspace")
    parser.add_argument(
        "operation",
        choices=(
            "init-fixture",
            "register",
            "register-karaf",
            "discover",
            "plan",
            "execute",
            "status",
            "cancel",
            "resume",
            "inspect",
            "evaluate",
            "report",
            "export",
            "import",
            "cleanup",
        ),
    )
    parser.add_argument("--request", help="Versioned planning/execution JSON")
    parser.add_argument("--job", help="Host-issued durable job ID")
    parser.add_argument("--input", type=Path, help="Trusted host Parquet/portable-evidence file, never a tool argument")
    parser.add_argument(
        "--manifest", type=Path, help="Reviewed dataset manifest declaring source/projection/clocks/units"
    )
    parser.add_argument("--offset", type=int, default=0, help="Inspection page offset")
    parser.add_argument("--limit", type=int, default=4, help="Inspection page size, one to eight")
    parser.add_argument("--cache", type=Path, help="Trusted pre-populated original-source cache")
    parser.add_argument("--source-output", type=Path, help="Trusted bounded original-source ETL output")


def run_command(args: argparse.Namespace) -> None:
    """Invoke shared typed workflows and print one canonical JSON response.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments produced by configure_parser. Registration checks the reviewed
        manifest against actual exact bytes/dimensions rather than trusting a name.

    Raises
    ------
    ValueError, OSError
        Missing required setup fields, malformed/oversized JSON, invalid registration,
        incompatible versions/caps or unavailable owned evidence. Work is never guessed.
    """
    service = WorkflowService(args.workspace)
    operation = args.operation
    if operation == "init-fixture":
        with TemporaryDirectory(prefix="coleman-workflow-registration-") as temporary:
            staging = Path(temporary)
            value = service.register(owned_fixture(staging), snapshot_root=staging)
    elif operation in ("register", "register-karaf"):
        value = _register(args, service)
    elif operation == "discover":
        value = service.discover()
    elif operation in ("plan", "execute"):
        value = _request_command(args, service)
    elif operation == "cleanup":
        print(canonical_json({"schema_version": 1, "expired_jobs": list(service.cleanup())}))
        return
    elif operation == "import":
        if args.input is None:
            raise ValueError("import requires --input portable evidence")
        with args.input.open("rb") as stream:
            payload = stream.read(2097153)
        if len(payload) > 2097152:
            raise ValueError("Portable evidence exceeds 2 MiB")
        value = service.import_evidence(payload.decode("utf-8"))
    else:
        value = _job_command(args, service)
    print(value.to_json())


def _request_command(args, service):
    """Validate bounded JSON before invoking the matching typed planning/admission operation."""
    if args.request is None or len(args.request.encode()) > 65536:
        raise ValueError("A bounded versioned --request is required")
    if args.operation == "plan":
        return service.plan(PlanningRequest.from_json(args.request))
    return service.execute(ExecutePlan.from_json(args.request))


def _job_command(args, service):
    """Dispatch only a parsed known job operation, preserving caller-owned service checks."""
    if args.job is None:
        raise ValueError("A host-issued --job is required")
    if args.operation == "inspect":
        return service.inspect(args.job, offset=args.offset, limit=args.limit)
    functions = {
        "status": service.status,
        "cancel": service.cancel,
        "resume": service.resume,
        "evaluate": service.evaluate,
        "report": service.report,
        "export": service.export,
    }
    return functions[args.operation](args.job)


def _register(args, service):
    """Prepare trusted source data and validate reviewed manifests before registry admission."""
    if args.operation == "register-karaf":
        from coleman.application.workflow_presets import register_karaf
        from coleman.corpus.materialization import SourceCache

        if args.cache is None or args.source_output is None:
            raise ValueError("register-karaf requires --cache and --source-output; acquisition stays separate")
        value = register_karaf(service, SourceCache(args.cache), args.source_output)
    else:
        if args.input is None or args.manifest is None:
            raise ValueError("register requires --input and --manifest from trusted host setup")
        reviewed = read_baseline_artifact(args.manifest, DatasetManifest)
        with TemporaryDirectory(prefix="coleman-workflow-registration-") as temporary:
            staging = Path(temporary)
            actual = snapshot_history(
                staging,
                args.input,
                dataset_id=reviewed.dataset_id,
                source_fingerprint=reviewed.source_fingerprint,
                projection_fingerprint=reviewed.projection_fingerprint,
                evidence=reviewed.evidence,
                assumptions=reviewed.assumptions,
                metadata=reviewed.metadata,
                temporal_plan=reviewed.temporal_plan,
            )
            if reviewed != actual:
                raise ValueError("Reviewed dataset manifest differs from actual snapshot bytes/dimensions")
            value = service.register(actual, snapshot_root=staging)
    return value
