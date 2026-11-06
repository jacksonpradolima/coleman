import os
import subprocess
import sys
from pathlib import Path


def test_pure_native_contracts_and_reader_do_not_import_backend_or_transport(tmp_path, record):
    repository = Path(__file__).resolve().parents[2]
    wire = tmp_path / "retained.json"
    wire.write_text(record.to_json(), encoding="utf-8")
    child = r"""
import importlib.abc, importlib.resources, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
forbidden=('numpy','polars','pyarrow','duckdb','scipy','coleman.policy','coleman.agent',
           'coleman.runner','coleman.cli','coleman.transports','mcp')
class RejectBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname==name or fullname.startswith(name+'.') for name in forbidden):
            raise RuntimeError('forbidden import: '+fullname)
sys.meta_path.insert(0,RejectBackend())
def no_resource_io(*args,**kwargs):
    raise RuntimeError('pure native import/reader tried resource I/O')
importlib.resources.files=no_resource_io
import coleman.core
import coleman.native
from coleman.native.decisions import FrozenNativeDecision
from coleman.native.publication import read_native_decision
record=read_native_decision(Path(sys.argv[2]).read_bytes())
assert record.schema_version==1 and len(record.ranking)==3
assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules for prefix in forbidden)
assert not Path.cwd().is_relative_to(Path(sys.argv[1]))
print(record.logical_identity)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", child, str(repository), str(wire)],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key not in {"PYTHONHOME", "PYTHONPATH"}},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == record.logical_identity
