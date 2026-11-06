"""Run actual copied host components from a noneditable minimal installed wheel."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_installed_base_composition_and_foreign_host_outside_checkout(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1")

    def command(arguments, *, cwd=tmp_path, extra=None):
        result = subprocess.run(
            arguments, cwd=cwd, env={**environment, **(extra or {})}, text=True, capture_output=True, timeout=180
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return result.stdout

    distributions = tmp_path / "dist"
    command(
        [uv, "build", "--wheel", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(distributions)],
        cwd=ROOT,
    )
    wheel = next(distributions.glob("*.whl"))
    virtual = tmp_path / "minimal"
    command([uv, "venv", str(virtual), "--python", sys.executable])
    python = virtual / "bin/python"
    # Root's separate packaging profile tests qualify normal dependency resolution.
    # This closure is the exact frozen base with no CI/analytics/SDK adapter extras.
    command(
        [
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
        ],
        cwd=ROOT,
        extra={"UV_PROJECT_ENVIRONMENT": str(virtual)},
    )
    command([uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", str(python), str(wheel)])
    outside = tmp_path / "outside"
    outside.mkdir()
    for name in ("learning_only", "mab_only", "solver_only", "foreign_consumer"):
        shutil.copyfile(ROOT / f"examples/composition/{name}.py", outside / f"{name}.py")
    shutil.copytree(ROOT / "tests/fixtures/interchange/v1", outside / "golden")
    cold = """
import importlib, importlib.abc, importlib.metadata, json, runpy, sys
from pathlib import Path
blocked=('numpy','scipy','polars','pyarrow','duckdb','mcp','openai','anthropic','opentelemetry',
         'boto3','clickhouse_connect','coleman.cli','coleman.runner','coleman.application',
         'coleman.workflows','coleman.transports','coleman.artifacts','coleman.telemetry')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in blocked):
            raise AssertionError('Minimal direct component imported optional boundary: '+fullname)
sys.meta_path.insert(0,Boundary())
from coleman.interchange.codec import decode_request, decode_response, wire_fingerprint
from coleman.interchange.contracts import InterchangeResponse, CoverageReply, accept_response
from coleman.core.primitives import fingerprint
from coleman.selection import solve_coverage
import coleman.interchange.contracts as installed
assert Path(installed.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(installed.__file__).resolve().is_relative_to(Path.cwd())
assert not {'numpy','scipy','polars','pyarrow','duckdb','mcp'} & {
    distribution.metadata['Name'].lower() for distribution in importlib.metadata.distributions()}
for name in ('learning_only','mab_only','solver_only'):
    runpy.run_path(name+'.py',run_name='__main__')
# Copied user-owned state is restored after an actual chosen-only update and JSON round trip.
mab=runpy.run_path('mab_only.py')
host=mab['HostMAB']()
case=mab['cases']()[0]
proposal=host.decide(case.request)
from coleman.core.feedback import FeedbackBatch, ChosenFeedback
executed=host.export_state().pending
row=next(row for row in case.outcomes if row.member==proposal.action.chosen)
batch=FeedbackBatch(schema_version=1,decision_fingerprint=executed.fingerprint,
    feedback_id='installed-chosen-0',revision_id='root',supersedes=None,horizon=None,
    feedback=ChosenFeedback(row=row))
host.update(batch)
restored=mab['HostMAB'](seed=987)
restored.restore_state(mab['MABState'].from_json(host.export_state().to_json()))
next_request=mab['cases']()[1].request
assert host.decide(next_request).to_json()==restored.decide(next_request).to_json()
assert host.export_state().to_json()==restored.export_state().to_json()
# Execute the real installed solver; this is not a fixture-only echoed response.
request=decode_request(Path('golden/coverage-request.json').read_bytes())
call=request.input
result=solve_coverage(call.problem,method=call.method,limits=call.limits)
assert result.selected_ids==('A','C')
reply=CoverageReply(result=result)
response=InterchangeResponse(schema_version=1,request=request,
    request_fingerprint=request.fingerprint,input_fingerprint=request.input_fingerprint,
    reply_fingerprint=fingerprint(reply.model_dump(mode='json')),implementation=request.implementation,
    produced_at='2026-01-01T10:01:00Z',state=None,reply=reply)
assert accept_response(request,response,received_at='2026-01-01T10:02:00Z')==response
Path('actual-request.json').write_text(request.to_json())
Path('actual-response.json').write_text(response.to_json())
Path('pins.json').write_text(json.dumps({'request_wire':wire_fingerprint(request.to_json()),
    'response_wire':wire_fingerprint(response.to_json()),'request_fingerprint':request.fingerprint,
    'input_fingerprint':request.input_fingerprint}))
assert not any(any(name==item or name.startswith(item+'.') for item in blocked) for name in sys.modules)
print(json.dumps({'installed_base':True,'restored_owned_rng':True,'real_finite_result':True}))
"""
    lines = command([str(python), "-I", "-c", cold], cwd=outside).splitlines()
    learning, mab, solver, receipt = [json.loads(line) for line in lines]
    assert learning["choices"] == ["recipe-a", "recipe-b"]
    assert learning["builtin_methods"] == "pending-368"
    assert mab["choices"] == ["recipe-a"] * 3 + ["recipe-b"] * 5
    assert mab["builtin_methods"] == "pending-370"
    assert solver["selected"] == ["A", "C"]
    assert receipt == {"installed_base": True, "restored_owned_rng": True, "real_finite_result": True}
    pins = json.loads((outside / "pins.json").read_text())
    foreign = """
import importlib.abc, runpy, sys
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname=='coleman' or fullname.startswith('coleman.'):
            raise AssertionError('Foreign host imported SDK')
sys.meta_path.insert(0,NoSDK())
sys.argv=['foreign_consumer.py']+sys.argv[1:]
runpy.run_path('foreign_consumer.py',run_name='__main__')
"""
    actual = json.loads(
        command(
            [
                str(python),
                "-I",
                "-c",
                foreign,
                "--request",
                "actual-request.json",
                "--response",
                "actual-response.json",
                "--request-wire",
                pins["request_wire"],
                "--response-wire",
                pins["response_wire"],
                "--request-fingerprint",
                pins["request_fingerprint"],
                "--input-fingerprint",
                pins["input_fingerprint"],
                "--received-at",
                "2026-01-01T10:02:00Z",
            ],
            cwd=outside,
        )
    )
    assert actual["selected"] == ["A", "C"]
    assert actual["status"] == "optimal"
    assert actual["cost"]["semantics"]["unit"] == "s"
    assert actual["sdk_imported"] is False
