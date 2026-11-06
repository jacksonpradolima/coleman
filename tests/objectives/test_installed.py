"""Qualify the real objective layer and examples from a noneditable wheel."""

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from . import ALLOWED, DEPENDENCIES, FORBIDDEN

ROOT = Path(__file__).parents[2]


def _digest(payload):
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(wire.encode()).hexdigest()


def test_versioned_golden_bytes_and_independently_worked_public_meanings():
    directory = ROOT / "tests/fixtures/objectives"
    manifest = json.loads((directory / "manifest-v1.json").read_text())
    digests = json.loads((directory / "digests-v1.json").read_text())
    payloads = {}
    assert manifest["schema_version"] == 1 and len(manifest["records"]) == 17
    for row in manifest["records"]:
        wire = (directory / row["file"]).read_bytes().removesuffix(b"\n")
        assert hashlib.sha256(wire).hexdigest() == digests[row["file"]]
        payload = json.loads(wire)
        assert _digest(payload) == "sha256:" + digests[row["file"]]
        model = getattr(importlib.import_module(row["module"]), row["class"])
        assert model.from_json(wire).to_json().encode() == wire
        payloads[row["file"]] = payload

    converted = payloads["conversion-receipt-v1.json"]
    assert converted["raw"]["value"] == -500
    assert converted["definition"]["input"]["spec"]["semantics"]["unit"] == "ms"
    assert converted["definition"]["output"]["spec"]["semantics"]["unit"] == "s"
    assert converted["converted"]["value"] == float(Fraction(-500, 1000)) == -0.5
    affine = payloads["affine-application-v1.json"]
    assert affine["definition"]["direction"] == "maximize" and affine["raw"]["value"] == -2.5
    assert affine["definition"]["bound_basis"] == "modeling_assumption"
    assert affine["definition"]["lower"] == -5 and affine["definition"]["upper"] == 5
    assert affine["output"]["value"] == float((Fraction(5) - Fraction(-5, 2)) / 10) == 0.75
    assert affine["status"] == "applied" and affine["clipping"] == "none"
    identity = payloads["identity-application-v1.json"]
    for value in (identity["raw"]["value"], identity["output"]["value"], identity["preclip_loss"]):
        assert value == 0 and math.copysign(1, value) == -1

    volatility = payloads["volatility-application-v1.json"]
    selection = volatility["selection"]
    frozen = selection["scale"]
    assert [sample["value"]["value"] for sample in frozen["snapshot"]["samples"]] == [-1, 0, 1]
    assert frozen["definition"]["window"] == 3 and frozen["scale"]["value"] == 1
    assert sum(Fraction(value) ** 2 for value in (-1, 0, 1)) / 2 == 1
    assert selection["definition"]["c"] == 2 and volatility["raw"]["value"] == -1
    assert volatility["z"] == -1 and volatility["output"]["value"] == float((1 - Fraction(-1, 2)) / 2) == 0.75
    assert volatility["reason"] == "computed" and volatility["clipping"] == "none"
    cutoff = datetime.fromisoformat(frozen["snapshot"]["training_cutoff"])
    frozen_at = datetime.fromisoformat(frozen["frozen_at"])
    information = datetime.fromisoformat(selection["timing"]["information_cutoff"])
    assert cutoff <= frozen_at <= information
    assert all(
        datetime.fromisoformat(sample["timing"]["received_at"]) <= cutoff for sample in frozen["snapshot"]["samples"]
    )
    assert selection["outcome"]["id"] not in {row["observation"]["id"] for row in frozen["snapshot"]["samples"]}
    stats = payloads["clipping-statistics-v1.json"]
    assert stats["observed"] == stats["computed"] == 3
    assert stats["unknown"] == stats["lower"] == stats["upper"] == 1
    assert stats["noncomputable"] == stats["unavailable"] == stats["not_applicable"] == 0
    assert stats["lower_rate"] == stats["upper_rate"] == float(Fraction(1, 3))

    preference = payloads["preference-evaluation-v1.json"]
    assert [value["value"] for value in preference["inputs"]] == [0.2, 0.6]
    weights = payloads["objective-weights-v1.json"]
    assert preference["specification"] == weights
    assert [term["weight"] for term in weights["terms"]] == [0.75, 0.25]
    assert (
        preference["output"]["value"] == float(Fraction(3, 4) * Fraction(1, 5) + Fraction(1, 4) * Fraction(3, 5)) == 0.3
    )
    assert preference["output"]["spec"]["role"] == "objective"
    anchor = payloads["static-quadratic-anchor-v1.json"]
    assert [coordinate["value"] for coordinate in anchor["weights"]] == [0.75, 0.25]
    assert anchor["strength"] == 0.4
    assert float(Fraction(2, 5) * (Fraction(1, 4) ** 2 + Fraction(-1, 4) ** 2)) == 0.05
    assert payloads["human-judgment-v1.json"]["relation"] == "left"
    assert {
        point["quantity"]["spec"]["dimension"]["id"] for point in payloads["reference-points-v1.json"]["points"]
    } == {"duration", "net"}
    controls = payloads["mixture-controls-l1-v1.json"]
    assert controls["output"]["kind"] == "l1_movement" and controls["output"]["strength"] == 1.5
    assert controls["output"]["objective_convention"] == "half_squared_plus_full_l1_v1"
    assert controls["cadence"]["kind"] == "interval_clock" and controls["movement"]["scope"] == "per_interval"
    assert controls["target"]["alpha"] == 0.1
    intersection = payloads["mixture-intersection-budget-v1.json"]
    assert intersection["status"] == "infeasible" and intersection["reason"] == "period_budget"
    assert intersection["exact_feasible"] is False
    assert all(bound["lower"] == bound["upper"] == 0.5 for bound in intersection["effective"])
    assert all(limit["maximum"] == 0 for limit in intersection["movement_definition"]["limits"])
    assert all(used["maximum"] == 2**-60 for used in intersection["usage"]["used"])

    semantic = payloads["semantic-request-v1.json"]
    context = semantic["context"]
    assert context["controls"] == controls and context["preferences"] == weights
    assert context["anchor"] == anchor and semantic["request"]["objective"] == context["objective"]["reference"]
    change = payloads["semantic-change-v1.json"]
    assert change["previous"]["definition_fingerprint"] == _digest(context)
    assert change["replacement"]["definition_fingerprint"] == _digest({**context, "preferences": None})
    assert change["effective_round"] == 2 and change["disposition"] == "restart" and change["migration"] is None
    finite = payloads["finite-request-v1.json"]
    result = payloads["finite-result-v1.json"]
    assert result["request"] == finite
    assert finite["constraints"][0]["upper"] == 3
    assert {row["member"]["id"]: row["values"][0]["value"] for row in finite["alternatives"]} == {
        "unit": 2,
        "integration": 4,
    }
    assert result["request_fingerprint"] == _digest(finite) and result["status"] == "feasible"
    assert {row["member"]["id"]: row["status"] for row in result["alternatives"]} == {
        "unit": "feasible",
        "integration": "infeasible",
    }
    fallback = payloads["fallback-v1.json"]
    assert fallback["kind"] == "baseline" and fallback["member"]["id"] == "unit"
    assert fallback["evidence"] == _digest(result)


def test_installed_objectives_goldens_boundaries_and_complete_examples(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    environment.update(UV_OFFLINE="1", UV_LINK_MODE="copy", UV_NO_PROGRESS="1")

    def command(arguments, *, cwd=tmp_path, extra=None):
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            env={**environment, **(extra or {})},
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout

    distributions = tmp_path / "dist"
    command(
        [uv, "build", "--wheel", "--no-build-isolation", "--python", sys.executable, "--out-dir", str(distributions)],
        cwd=ROOT,
    )
    wheels = tuple(distributions.glob("*.whl"))
    assert len(wheels) == 1
    virtual = tmp_path / "installed"
    command([uv, "venv", str(virtual), "--python", sys.executable])
    python = virtual / "bin/python"
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
    command([uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", str(python), str(wheels[0])])
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copytree(ROOT / "tests/fixtures/objectives", outside / "golden")
    shutil.copytree(ROOT / "tests/fixtures/core/v1", outside / "retained")
    modules = [
        "coleman.objectives." + source.stem
        for source in sorted((ROOT / "coleman/objectives").glob("*.py"))
        if source.stem != "__init__"
    ]
    boundary = {
        "forbidden": FORBIDDEN,
        "allowed": sorted(ALLOWED),
        "dependencies": {module: sorted(names) for module, names in DEPENDENCIES.items()},
        "modules": modules,
    }
    script = """
import ast, hashlib, importlib, importlib.abc, importlib.util, json, sys
from pathlib import Path
boundary=json.loads(sys.argv[1])
forbidden=boundary['forbidden']
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in forbidden):
            raise AssertionError('Forbidden installed objective dependency: '+fullname)
guard=Boundary()
sys.meta_path.insert(0,guard)
package=importlib.import_module('coleman.objectives')
directory=Path(package.__file__).resolve().parent
assert directory.is_relative_to(Path(sys.prefix).resolve()),directory
assert not directory.is_relative_to(Path.cwd().resolve()),directory
files=sorted(directory.glob('*.py'))
actual=['coleman.objectives.'+source.stem for source in files if source.stem!='__init__']
assert actual==boundary['modules'],(actual,boundary['modules'])
graph={source.stem:set() for source in files}
for source in files:
    assert source.stem in boundary['dependencies']
    if source.stem!='__init__':
        importlib.import_module('coleman.objectives.'+source.stem)
    for node in ast.walk(ast.parse(source.read_text(),filename=str(source))):
        if isinstance(node,ast.Import):
            names=[alias.name for alias in node.names]
        elif isinstance(node,ast.ImportFrom):
            assert not node.level,(source,node.lineno)
            names=[node.module or '']
        else:
            continue
        for name in names:
            assert name.startswith(('coleman.core.','coleman.objectives.')) or name.split('.')[0] in boundary['allowed']
            assert not any(name==item or name.startswith(item+'.') for item in forbidden),(source,node.lineno,name)
            if name.startswith('coleman.objectives.'):
                dependency=name.split('.')[2]
                assert dependency in boundary['dependencies'][source.stem],(source,node.lineno,dependency)
                graph[source.stem].add(dependency)
while graph:
    leaves={module for module,deps in graph.items() if not deps.intersection(graph)}
    assert leaves,graph
    graph={module:deps for module,deps in graph.items() if module not in leaves}
manifest=json.loads(Path('golden/manifest-v1.json').read_text())
golden_digests=json.loads(Path('golden/digests-v1.json').read_text())
assert manifest['schema_version']==1 and manifest['records']
for row in manifest['records']:
    assert row['module'].startswith('coleman.objectives.')
    model=getattr(importlib.import_module(row['module']),row['class'])
    wire=Path('golden',row['file']).read_bytes().removesuffix(b'\\n')
    assert hashlib.sha256(wire).hexdigest()==golden_digests[row['file']],row['file']
    value=model.from_json(wire)
    assert value.to_json().encode()==wire,row['file']
    assert value.fingerprint=='sha256:'+hashlib.sha256(wire).hexdigest(),row['file']
    payload=json.loads(wire)
    for replacement in (None,0,2,True):
        altered=dict(payload)
        if replacement is None:
            del altered['schema_version']
        else:
            altered['schema_version']=replacement
        try:
            model.from_json(json.dumps(altered))
        except ValueError:
            pass
        else:
            raise AssertionError(('Accepted unsupported schema version',row['file'],replacement))
from coleman.core.actions import ChoiceAction,RankingAction,MixtureAction
from coleman.core.capabilities import Capability,ProblemCapabilities
from coleman.core.decisions import DecisionRequest,Decision,ExecutedDecision
from coleman.core.feedback import FeedbackBatch
retained=[('choice-action.json',ChoiceAction),('ranking-action.json',RankingAction),
          ('mixture-action.json',MixtureAction),('decision-request.json',DecisionRequest),
          ('decision.json',Decision),('executed-decision.json',ExecutedDecision),
          ('feedback-batch.json',FeedbackBatch),('capability.json',Capability),
          ('problem-capabilities.json',ProblemCapabilities)]
digests=json.loads(Path('retained/digests.json').read_text())
for filename,model in retained:
    wire=Path('retained',filename).read_bytes().removesuffix(b'\\n')
    value=model.from_json(wire)
    assert value.to_json().encode()==wire,filename
    assert value.fingerprint=='sha256:'+digests[filename],filename
assert not any(any(name==block or name.startswith(block+'.') for block in forbidden) for name in sys.modules)
sys.meta_path.remove(guard)
assert importlib.util.find_spec('mcp') is None
print(json.dumps({'modules':len(actual),'objective_goldens':len(manifest['records']),
                  'retained_goldens':len(retained),'installed_ast':True,'forbidden_loaded':False}))
"""
    report = json.loads(command([str(python), "-I", "-c", script, json.dumps(boundary)], cwd=outside))
    fixture_manifest = json.loads((outside / "golden/manifest-v1.json").read_text())
    assert report == {
        "modules": len(modules),
        "objective_goldens": len(fixture_manifest["records"]),
        "retained_goldens": 9,
        "installed_ast": True,
        "forbidden_loaded": False,
    }
    for example in ("ci.py", "allocation.py"):
        shutil.copyfile(ROOT / "examples/objectives" / example, outside / example)
    ci = json.loads(command([str(python), "-I", "ci.py"], cwd=outside))
    assert ci == {
        "origin": "synthetic",
        "objectives": [{"suite": "unit", "objective": 0.275}, {"suite": "integration", "objective": 0.25}],
        "duration_seconds": [2.0, 4.0],
        "feasible": ["unit"],
        "fallback": "baseline",
        "selected": "unit",
    }
    allocation = json.loads(command([str(python), "-I", "allocation.py"], cwd=outside))
    scale_fingerprint = allocation.pop("scale_fingerprint")
    assert scale_fingerprint.startswith("sha256:") and len(scale_fingerprint) == 71
    assert allocation == {
        "origin": "synthetic",
        "gross_credit": "-2",
        "paid_cost_credit": "0.5",
        "net_credit": "-2.5",
        "affine_loss": 0.75,
        "historical_scale_credit": 1.0,
        "volatility_loss": 1.0,
        "volatility_clipping": "upper",
        "net_losses": [1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0],
        "clipping_sides": ["upper", "none", "none", "none", "none", "none", "lower"],
        "counts": {"computed": 7, "unknown": 1, "lower": 1, "upper": 1},
        "clipping_denominator": 7,
    }
    repeated = json.loads(command([str(python), "-I", "allocation.py"], cwd=outside))
    assert repeated == {**allocation, "scale_fingerprint": scale_fingerprint}
