"""Qualify noneditable installed contracts, retained artifacts and actual native CI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_noneditable_contracts_and_real_tcpci_outside_checkout(tmp_path):
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
            "--extra",
            "ci",
            "--inexact",
            "--python",
            str(python),
        ],
        cwd=ROOT,
        extra={"UV_PROJECT_ENVIRONMENT": str(virtual)},
    )
    command(
        [uv, "pip", "install", "--offline", "--no-index", "--no-deps", "--python", str(python), str(wheel) + "[ci]"]
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copytree(ROOT / "tests/fixtures/core/v1", outside / "golden")
    shutil.copytree(ROOT / "tests/fixtures/domain-results", outside / "legacy")
    script = """
import ast, hashlib, importlib, importlib.abc, importlib.util, json, sys
from pathlib import Path
forbidden=('coleman.results','coleman.selection','coleman.allocation','coleman.application','coleman.workflows',
           'coleman.runner','coleman.api','coleman.agent','coleman.policy','coleman.reward','coleman.spec',
           'coleman.cli','coleman.transports','coleman.telemetry','coleman.corpus','coleman.evaluation',
           'coleman.observations','coleman.environment','coleman.bandit','coleman.scenarios',
           'mcp','duckdb','polars','pyarrow','numpy','scipy','opentelemetry','clickhouse_connect',
           'openai','anthropic','boto3')
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in forbidden):
            raise AssertionError('Core imported optional/domain dependency: '+fullname)
guard=Boundary()
sys.meta_path.insert(0,guard)
from coleman.core.actions import ChoiceAction, RankingAction, MixtureAction, MixtureSpace, Weight
from coleman.core.base import VersionedContract
from coleman.core.capabilities import Capability, ProblemCapabilities
from coleman.core.decisions import DecisionRequest, Decision, ExecutedDecision
from coleman.core.feedback import FeedbackBatch, FeedbackRow, OutcomeValue, FullPanel, ChosenFeedback
from coleman.core.sampling import SamplingRecord
from coleman.core.units import UnitSemantics
from coleman.core.primitives import fingerprint
from coleman.core import actions
from coleman.testing.decision_contracts import ConformanceCase, check_decision_contracts
assert Path(actions.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not Path(actions.__file__).resolve().is_relative_to(Path.cwd())
allowed={'__future__','collections','dataclasses','datetime','decimal','enum','fractions','functools',
         'hashlib','itertools','json','math','operator','re','types','typing','typing_extensions','pydantic'}
core_files=sorted(Path(actions.__file__).parent.glob('*.py'))
assert core_files
for source in core_files:
    if source.stem!='__init__':
        importlib.import_module('coleman.core.'+source.stem)
    for node in ast.walk(ast.parse(source.read_text(),filename=str(source))):
        if isinstance(node,ast.Import):
            imports=[alias.name for alias in node.names]
        elif isinstance(node,ast.ImportFrom):
            assert not node.level, (source,node.lineno)
            imports=[node.module or '']
        else:
            continue
        for name in imports:
            assert name.startswith('coleman.core.') or name.split('.')[0] in allowed,(source,node.lineno,name)
            assert not any(name==block or name.startswith(block+'.') for block in forbidden)
digests=json.loads(Path('golden/digests.json').read_text())
models=[('choice-action.json',ChoiceAction),('ranking-action.json',RankingAction),
        ('mixture-action.json',MixtureAction),('decision-request.json',DecisionRequest),
        ('decision.json',Decision),('executed-decision.json',ExecutedDecision),
        ('feedback-batch.json',FeedbackBatch),('capability.json',Capability),
        ('problem-capabilities.json',ProblemCapabilities)]
for filename,model in models:
    wire=Path('golden',filename).read_bytes().removesuffix(b'\\n')
    value=model.from_json(wire)
    assert value.to_json().encode()==wire
    assert value.fingerprint=='sha256:'+digests[filename]
    assert hashlib.sha256(wire).hexdigest()==digests[filename]
base_request=DecisionRequest.from_json(Path('golden/decision-request.json').read_bytes())
base_capability=Capability.from_json(Path('golden/capability.json').read_bytes())
base_problem=ProblemCapabilities.from_json(Path('golden/problem-capabilities.json').read_bytes())
spec=base_problem.values[0]
# This host-owned consumer uses public structural entry points and plain tuple state.
# It is authored here, outside the wheel, and is not a shipped learning algorithm.
class HostConsumer:
    def __init__(self,kind):
        self.kind=kind
        self.round=0
    def decide(self,request):
        selected=self.round % 2
        if self.kind=='chosen':
            action=ChoiceAction(schema_version=1,space=request.space,chosen=request.space.members[selected])
            status,probability='exact',1.0
        else:
            action=MixtureAction(schema_version=1,space=request.space,
                weights=tuple(Weight(member=member,value=float(index==selected))
                              for index,member in enumerate(request.space.members)))
            status,probability='not_required',None
        sampling=SamplingRecord(mechanism='deterministic',mechanism_ref=base_capability.implementation,
            status=status,action_fingerprint=action.fingerprint,probability=probability)
        return Decision(schema_version=1,request=request,implementation=base_capability.implementation,
            action=action,sampling=sampling,state_owner='host')
    def update(self,batch):
        assert isinstance(batch.feedback,ChosenFeedback if self.kind=='chosen' else FullPanel)
        self.round+=1
    def export_state(self):
        return (1,self.round)
    def restore_state(self,state):
        assert state[0]==1
        self.round=state[1]
for kind in ('chosen','full_panel'):
    space=base_request.space if kind=='chosen' else MixtureSpace(
        members=base_request.space.members,feasible_set=base_request.space.feasible_set,profile='binary64-simplex-v1')
    capability=Capability.model_validate(base_capability.model_dump(mode='python')|{
        'action_kinds':(space.kind,),'feedback_kinds':(kind,), 'state_owner':'host',
        'requires_sampling':kind=='chosen','sampling_requirement':'exact' if kind=='chosen' else 'not_required'})
    problem=ProblemCapabilities.model_validate(base_problem.model_dump(mode='python')|{
        'space':space,'feedback_kind':kind,'state_owners':('host',),
        'sampling_statuses':('exact',) if kind=='chosen' else ('not_required',)})
    cases=tuple(ConformanceCase(problem=problem,
        request=DecisionRequest.model_validate(base_request.model_dump(mode='python')|{
            'space':space,'decision_id':'installed-'+str(index),'round':index}),
        outcomes=tuple(FeedbackRow(member=member,values=(OutcomeValue(spec=spec,value=float(index % 2),
            status='observed',origin='synthetic',transform=None),)) for member in space.members)) for index in range(3))
    report=check_decision_contracts(lambda:HostConsumer(kind),capability,cases)
    assert len(report.decisions)==len(report.feedback)==3
    assert len({decision.action.fingerprint for decision in report.decisions})==2
assert not any(any(name==block or name.startswith(block+'.') for block in forbidden) for name in sys.modules)
sys.meta_path.remove(guard)
assert importlib.util.find_spec('mcp') is None
# Retained public symbols refer to the same leaves; no old wire is reinterpreted.
from coleman.results.versioned import VersionedContract as LegacyContract
from coleman.results.domain import fingerprint as legacy_fingerprint
from coleman.selection.contracts import UnitSemantics as LegacyUnits
assert LegacyContract is VersionedContract
assert LegacyUnits is UnitSemantics
assert legacy_fingerprint is fingerprint
from coleman.results.duckdb_catalog import DuckDBCatalog
artifact=Path('legacy/native-v3.2.parquet')
before=artifact.read_bytes()
legacy=json.loads(Path('legacy/native-v3.2.json').read_text())
catalog=DuckDBCatalog('legacy')
try:
    actual=catalog.query('SELECT fitness, cost, rewards, ttf_duration FROM results').row(0,named=True)
    assert actual=={key:legacy[key] for key in actual}
finally:
    catalog.close()
assert artifact.read_bytes()==before
print(json.dumps({'goldens':9,'forbidden_loaded':False,'legacy_aliases':True,'installed_ast':True,
                  'legacy_artifact':True,'host_conformance_modes':2}))
"""
    assert json.loads(command([str(python), "-I", "-c", script], cwd=outside).strip().splitlines()[-1]) == {
        "goldens": 9,
        "forbidden_loaded": False,
        "legacy_aliases": True,
        "installed_ast": True,
        "legacy_artifact": True,
        "host_conformance_modes": 2,
    }
    shutil.copyfile(ROOT / "examples/core_contracts_ci.py", outside / "core_contracts_ci.py")
    core_example = command([str(python), "-I", "core_contracts_ci.py"], cwd=outside).splitlines()
    assert core_example[0] == "chosen=suite-unit; credit=1; elapsed=8.5 s; unchosen=unobserved"
    assert len(core_example) == 3
    assert core_example[1].startswith("logical_identity=sha256:")
    assert core_example[2].startswith("feedback_digest=sha256:")
    shutil.copyfile(ROOT / "examples/scenario-payoffs/ci.py", outside / "ci.py")
    native = json.loads(command([str(python), "ci.py", "native-output"], cwd=outside).strip().splitlines()[-1])
    assert native == {"selected_action_id": "stable", "status": "optimal", "native_budget_seconds": 3.0}
    assert list((outside / "native-output/native").rglob("*.parquet"))
