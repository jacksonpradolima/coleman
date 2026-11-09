"""Fixed copied test launcher; all mathematical work uses installed packages."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _sibling(name: str, filename: str):
    """Load only this explicit immutable copied harness inventory under-I."""
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError("Installed qualification helper is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


SHARED = _sibling("task_installed_support", "_installed_support.py")
FAULTS = _sibling("task_installed_faults", "_installed_faults.py")
INTERFACES = _sibling("task_installed_interfaces", "_installed_interfaces.py")


def main() -> None:
    """Dispatch known test modes; these are not public transport operations."""
    mode, *arguments = sys.argv[1:]
    paired = {
        "inspect-guide": SHARED.inspect_single,
        "recover": FAULTS.recover,
        "cancel-resume": FAULTS.cancel_resume,
        "same-version": FAULTS.same_version_image_change,
        "generic-interfaces": INTERFACES.generic_interfaces,
        "native-interfaces": INTERFACES.native_and_catalogues,
    }
    if mode in paired:
        result = paired[mode](Path(arguments[0]), Path(arguments[1]))
    elif mode == "installation":
        virtual, *wheels = map(Path, arguments)
        result = SHARED.installation(tuple(wheels), virtual)
    elif mode == "approve":
        result = SHARED.approve(Path(arguments[0]), Path(arguments[1]))
    elif mode == "prepare":
        result = SHARED.prepare(Path(arguments[0]), Path(arguments[1]), arguments[2])
    elif mode == "execute":
        result = SHARED.execute(Path(arguments[0]), Path(arguments[1]), arguments[2])
    elif mode == "inspect":
        result = SHARED.inspect_complete(Path(arguments[0]), Path(arguments[1]), arguments[2])
    elif mode == "provider-free":
        result = SHARED.provider_free(Path(arguments[0]), Path(arguments[1]), arguments[2])
    elif mode == "detached-parent":
        FAULTS.detached_parent(Path(arguments[0]), Path(arguments[1]), arguments[2], Path(arguments[3]))
        raise AssertionError("Detached submitting parent must actually exit")
    elif mode == "fault-worker":
        FAULTS.fault_worker(
            Path(arguments[0]),
            Path(arguments[1]),
            arguments[2],
            int(arguments[3]),
            FAULTS.Fault(arguments[4], Path(arguments[5])),
        )
        raise AssertionError("Actual fault child must pause until killed/cancelled")
    elif mode == "refuse-image":
        result = FAULTS.refuse_changed_image(Path(arguments[0]), Path(arguments[1]), arguments[2])
    else:
        raise ValueError("Unknown fixed installed qualification mode")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
