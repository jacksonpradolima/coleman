"""Offline noneditable CI-capability wheel, transport isolation, actual TCPCI and optional MCP stdio client."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_installed_ci_and_optional_tools_outside_checkout_offline(tmp_path):
    # The explicit optional test job pre-caches the pinned SDK; execution below
    # remains offline and never uses a model, credentials or a provider endpoint.
    pytest.importorskip("mcp")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {**os.environ, "UV_OFFLINE": "1", "UV_NO_PROGRESS": "1", "UV_LINK_MODE": "copy"}

    def command(arguments, cwd=tmp_path):
        completed = subprocess.run(arguments, cwd=cwd, env=environment, capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stderr
        return completed

    distributions = tmp_path / "dist"
    command([uv, "build", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(distributions)], ROOT)
    wheel = next(distributions.glob("*.whl"))
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable])
    python = virtual / "bin/python"

    def dependency_setup(*extras):
        # Frozen sync uses audited artifact URLs from uv.lock. Offline pip
        # resolution instead requires index metadata not populated by CI's
        # frozen install, even when every required wheel is already cached.
        arguments = [
            uv,
            "sync",
            "--frozen",
            "--offline",
            "--no-default-groups",
            "--no-dev",
            "--no-install-project",
            "--inexact",
            "--python",
            str(python),
            *extras,
        ]
        setup = subprocess.run(
            arguments,
            cwd=ROOT,
            env={**environment, "UV_PROJECT_ENVIRONMENT": str(virtual)},
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert setup.returncode == 0, setup.stderr

    dependency_setup("--extra", "ci")
    command([uv, "pip", "install", "--no-deps", "--python", str(python), str(wheel) + "[ci]"])
    assert not tmp_path.is_relative_to(ROOT)
    base = tmp_path / "base.py"
    base.write_text("""
import base64
import importlib.util
import json
import polars as pl
from pathlib import Path
from coleman.application import ApplicationService, Caller, Limits, ServerPolicy, Submit, default_registry
import coleman
assert Path(coleman.__file__).resolve().is_relative_to(Path(__file__).parent / "installed")
assert importlib.util.find_spec("mcp") is None
from decimal import Decimal
from coleman.selection.contracts import Quantity, UnitSemantics
from coleman.allocation.quantities import ExactPayoff, SignedPayoff
units = UnitSemantics(unit="s", meaning="owned-ci-work-v1")
for kind in (Quantity, SignedPayoff, ExactPayoff):
    try:
        kind.model_validate({"value":"1e-999999999","semantics":units})
    except ValueError:
        pass
    else:
        raise AssertionError("Compact exponent bypassed installed decimal contract")
assert SignedPayoff(value=Decimal("-1.25"),semantics=units).value == Decimal("-1.25")
assert ExactPayoff(value=Decimal("1e-48"),semantics=units).model_dump(mode="json")["value"].endswith("1")
caller = Caller(subject="installed")
with ApplicationService(default_registry(), root=Path("jobs"),
                        policy=ServerPolicy(caps=Limits(deadline_seconds=120))) as service:
    for spec_id in ("weighted-overlap-3", "native-ci-smoke-v1"):
        request=Submit(schema_version=1,request_id=spec_id,spec_id=spec_id,limits=Limits(deadline_seconds=120))
        initial = service.submit(request,caller=caller)
        final = service.wait(initial.job_id,caller=caller)
        assert final.state == "completed", final
        bundle = service.result(final.handle,caller=caller)
        if bundle.kind == "coverage":
            assert bundle.result.selected_ids == ("A","C")
            assert bundle.result.objective.value == 1
            Path("expected.json").write_text(bundle.to_json())
        else:
            assert bundle.result.work_units == 2
            assert any(file.name.endswith(".parquet") for file in bundle.result.files)
            artifact=next(file for file in bundle.result.files if file.name.endswith(".parquet"))
            scientific=["step","detected","missed","tests_ran","fitness","cost","prioritization_order_hash"]
            rows=pl.read_parquet(base64.b64decode(artifact.content_base64)).select(scientific).to_dicts()
            Path("expected-native.json").write_text(json.dumps({
                "configuration_fingerprint":bundle.configuration_fingerprint,
                "run_id":bundle.result.run_id,"rows":rows}))
print(json.dumps({"base":"verified","mcp":False}))
""")
    assert json.loads(command([str(python), str(base)]).stdout) == {"base": "verified", "mcp": False}
    request = '{"schema_version":1,"request_id":"cli","spec_id":"weighted-overlap-3"}'
    cli = json.loads(command([str(virtual / "bin/coleman"), "application", "--request", request]).stdout)
    assert cli["ok"]
    assert cli["data"] == json.loads((tmp_path / "expected.json").read_text())
    dependency_setup("--extra", "ci", "--extra", "mcp")
    host = tmp_path / "host.py"
    host.write_text("""
from pathlib import Path
from tempfile import TemporaryDirectory
from coleman.application import ApplicationService,Caller,Limits,ServerPolicy,default_registry
from coleman.transports.mcp import create_server
with TemporaryDirectory() as directory:
    with ApplicationService(default_registry(),root=Path(directory),
                            policy=ServerPolicy(caps=Limits(deadline_seconds=120))) as service:
        create_server(service,caller=Caller(subject="installed-stdio")).run(transport="stdio")
""")
    optional = tmp_path / "tools.py"
    optional.write_text("""
import asyncio,base64,json,sys
import polars as pl
import coleman
from pathlib import Path
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from coleman.application import Limits,ResultBundle,Submit

async def main():
    assert Path(coleman.__file__).resolve().is_relative_to(Path.cwd() / "installed")
    async with Client(StdioServerParameters(command=sys.executable,args=["host.py"])) as client:
        assert len((await client.list_tools()).tools) == 7
        async def call(name,arguments):
            reply=await client.call_tool(name,arguments)
            assert not reply.is_error
            return json.loads(reply.content[0].text)
        assert (await call("discover",{}))["transport_contract"] == "coleman-application-v1"
        for spec_id in ("weighted-overlap-3","native-ci-smoke-v1"):
            request=Submit(schema_version=1,request_id=spec_id,spec_id=spec_id,limits=Limits(deadline_seconds=120))
            view=(await call("submit",{"request_json":request.to_json()}))["data"]
            command=json.dumps({"schema_version":1,"job_id":view["job_id"]})
            while view["state"] in ("queued","running"):
                await asyncio.sleep(.02)
                view=(await call("status",{"command_json":command}))["data"]
            assert view["state"] == "completed",view
            handle=json.dumps(view["handle"])
            bundle=(await call("result",{"handle_json":handle}))["data"]
            if spec_id == "weighted-overlap-3":
                assert bundle == json.loads(Path("expected.json").read_text())
            else:
                assert bundle["result"]["work_units"] == 2
                artifact=next(file for file in bundle["result"]["files"] if file["name"].endswith(".parquet"))
                scientific=["step","detected","missed","tests_ran","fitness","cost","prioritization_order_hash"]
                rows=pl.read_parquet(base64.b64decode(artifact["content_base64"])).select(scientific).to_dicts()
                assert {"configuration_fingerprint":bundle["configuration_fingerprint"],
                        "run_id":bundle["result"]["run_id"],"rows":rows} == json.loads(
                            Path("expected-native.json").read_text())
            exported=(await call("export_result",{"handle_json":handle}))["data"]
            imported=(await call("import_result",{"bundle_json":json.dumps(exported)}))["data"]
            assert imported["handle"]["digest"] == view["handle"]["digest"]
            assert (await call("cancel",{"command_json":command}))["data"]["state"] == "completed"
        cancellation=Submit(schema_version=1,request_id="cancel",spec_id="native-ci-smoke-v1")
        view=(await call("submit",{"request_json":cancellation.to_json()}))["data"]
        cancelled=await call("cancel",{"command_json":json.dumps({"schema_version":1,"job_id":view["job_id"]})})
        assert cancelled["data"]["state"] == "cancelled"
        assert (await call("submit",{"request_json":'{"schema_version":2}'}))["error"] == "invalid_input"
    print(json.dumps({"tools":"verified","model":False,"network":False}))
asyncio.run(asyncio.wait_for(main(),timeout=150))
""")
    assert json.loads(command([str(python), str(optional)]).stdout) == {
        "tools": "verified",
        "model": False,
        "network": False,
    }
