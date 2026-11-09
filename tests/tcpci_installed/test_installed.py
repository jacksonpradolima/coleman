"""Actual base-wheel TCPCI mapping/Hedge/coverage outside the checkout.

The fixture consumes ROOT's current prebuilt wheel when supplied. ROOT owns
actual build/install/global execution while SDK source writers are active.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from tests.installed_support import build_install_stdlib_wheel, seed_locked_dependencies

ROOT = Path(__file__).parents[2]


def _command(arguments, *, environment, cwd):
    """Run one bounded actual consumer command, retaining failure output."""
    result = subprocess.run(
        arguments, cwd=cwd, env=environment, text=True, capture_output=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(name="installed_mapping", scope="module")
def _installed_mapping(tmp_path_factory):
    """Install one real final wheel, then run the shared outside-checkout proof."""
    owned = tmp_path_factory.mktemp("installed-tcpci-mapping")
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
    virtual = owned / "base"
    supplied = environment.get("COLEMAN_TEST_LEARNING_WHEEL")
    if supplied:
        wheel = Path(supplied).resolve(strict=True)
        assert wheel.is_file()
        _command(
            [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(virtual)],
            environment=environment,
            cwd=owned,
        )
        seed_locked_dependencies(uv, virtual, environment)
        _command(
            [
                uv,
                "pip",
                "install",
                "--offline",
                "--no-index",
                "--no-deps",
                "--python",
                str(virtual / "bin/python"),
                str(wheel),
            ],
            environment=environment,
            cwd=owned,
        )
        _command([uv, "pip", "check", "--python", str(virtual / "bin/python")], environment=environment, cwd=owned)
    else:
        wheel = build_install_stdlib_wheel(uv, owned / "dist", virtual, environment)
    outside = owned / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    copied = {}
    for source in (
        ROOT / "examples/tcpci/ci.py",
        ROOT / "examples/tcpci/model.py",
        Path(__file__).with_name("driver.py"),
    ):
        target = outside / source.name
        shutil.copyfile(source, target)
        copied[source.name] = sha256(target.read_bytes()).hexdigest()
    output = _command(
        [str(virtual / "bin/python"), "-I", "-B", str(outside / "driver.py"), str(wheel)],
        environment=environment,
        cwd=outside,
    )
    proof = json.loads(output)
    assert proof["wheel_sha256"] == sha256(wheel.read_bytes()).hexdigest()
    (owned / "installation.json").write_text(
        json.dumps(
            {
                "wheel": str(wheel),
                "wheel_sha256": proof["wheel_sha256"],
                "copied": copied,
                "supplied_prebuilt": bool(supplied),
            },
            sort_keys=True,
        )
    )
    (owned / "proof.json").write_text(json.dumps(proof, sort_keys=True))
    return proof


def test_noneditable_base_mapping_uses_real_hedge_and_coverage_without_native_or_io(installed_mapping):
    """The actual independent wheel flow loads no native/transport/storage stack."""
    assert installed_mapping["installed_noneditable"] is True
    assert installed_mapping["boundary_clear"] is True
    assert installed_mapping["SDK_resources_verified"] > 0
    assert installed_mapping["summary"]["physical_elapsed_seconds"] is None
    assert installed_mapping["summary"]["coverage"]["selected_unordered"] == ["A", "C"]


def test_installed_complete_original_feedback_and_versions_survive_roundtrip(installed_mapping):
    """Every raw producer, mapped test batch and full successor remains inspectable."""
    rows = installed_mapping["full_report"]["evidence"]
    assert len(rows) == 3
    for ordinal, row in enumerate(rows):
        assert row["executed_fingerprint"] == row["selected_feedback"]["decision_fingerprint"]
        assert row["mapping_fingerprint"] == installed_mapping["summary"]["rounds"][ordinal]["mapping_fingerprint"]
        assert row["mapping"]["schema_version"] == 1
        assert row["selected_feedback"]["feedback"]["kind"] == "component"
        assert row["expert_feedback"]["feedback"]["kind"] == "full_panel"
        assert row["after_state"]["update_count"] == ordinal + 1
        assert row["mapping"]["executed"]["sampling"]["probability"] is None
