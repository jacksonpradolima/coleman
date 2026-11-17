"""Copied isolated import probe; no source checkout or optional provider loading."""

import importlib
import importlib.abc
import importlib.metadata
import json
import sys
import threading
from pathlib import Path

FORBIDDEN = (
    "click",
    "opentelemetry",
    "mcp",
    "duckdb",
    "numpy",
    "polars",
    "pyarrow",
    "scipy",
    "clickhouse_connect",
    "coleman.cli",
    "coleman.runner",
    "coleman.transports",
)


class Boundary(importlib.abc.MetaPathFinder):
    """Refuse optional/provider/transport imports even in an all-extra environment."""

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in FORBIDDEN):
            raise AssertionError("Cold import crossed a forbidden boundary: " + fullname)


def forbid_effect(event, _arguments):
    if event in {"socket.connect", "socket.bind", "subprocess.Popen", "os.system"}:
        raise AssertionError("Cold import attempted an external effect: " + event)


def main():
    sys.addaudithook(forbid_effect)
    before = {thread.ident for thread in threading.enumerate()}
    sys.meta_path.insert(0, Boundary())
    modules = (
        "coleman.core.actions",
        "coleman.learning.reference",
        "coleman.learning.controlled",
        "coleman.telemetry.lifecycle",
        "coleman.telemetry.resources",
        "coleman.telemetry.views",
        "coleman.telemetry.export",
    )
    for name in modules:
        module = importlib.import_module(name)
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert not Path(module.__file__).resolve().is_relative_to(Path.cwd())
    assert {thread.ident for thread in threading.enumerate()} == before
    forbidden = [name for name in sys.modules if any(name == p or name.startswith(p + ".") for p in FORBIDDEN)]
    assert not forbidden
    print(
        json.dumps(
            {
                "new_threads": [],
                "optional_imports": forbidden,
                "installed": sorted(
                    d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
