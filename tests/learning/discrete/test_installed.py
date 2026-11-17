"""Noneditable base-wheel MAB-only execution and strict cold import boundaries."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

from tests.installed_support import build_install_offline_wheel
from tests.learning.test_imports import BLOCKED, COLD_SCRIPT

ROOT = Path(__file__).parents[3]


def test_discrete_static_imports_do_not_cross_transport_runner_or_storage_boundaries():
    for source in (ROOT / "coleman/learning/discrete").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            imported = ()
            if isinstance(node, ast.Import):
                imported = tuple(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                imported = (node.module or "",)
            for name in imported:
                assert not any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED)


def test_noneditable_installed_mab_only_ci_and_actual_cold_dependency_boundary(tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(UV_OFFLINE="1", UV_NO_INDEX="1", UV_NO_PROGRESS="1", UV_LINK_MODE="copy")
    virtual = tmp_path / "base"
    wheel = build_install_offline_wheel(uv, tmp_path / "dist", virtual, environment)
    assert wheel.is_file()
    outside = tmp_path / "outside"
    outside.mkdir()
    assert not outside.is_relative_to(ROOT)
    shutil.copyfile(ROOT / "examples/discrete/ci.py", outside / "ci.py")
    shutil.copyfile(ROOT / "examples/composition/mab_only.py", outside / "host_mab.py")
    shutil.copyfile(Path(__file__).with_name("installed_driver.py"), outside / "driver.py")
    python = str(virtual / "bin/python")
    result = subprocess.run(
        [python, "-I", "-B", str(outside / "ci.py")],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert {row["profile"] for row in receipt} == {"exp3-dyadic53-v1", "ucb1-binary64-v1"}
    assert all(row["restored_every_draw"] and sum(row["observed_counts"]) == 8 for row in receipt)
    conformance = subprocess.run(
        [python, "-I", "-B", str(outside / "driver.py")],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert conformance.returncode == 0, conformance.stdout + conformance.stderr
    direct = json.loads(conformance.stdout)
    assert direct["independent_host_cases"] == 8
    assert {row["profile"] for row in direct["shipped"]} == {"exp3-dyadic53-v1", "ucb1-binary64-v1"}
    assert all(row["shared_conformance_cases"] == 8 for row in direct["shipped"])
    site = subprocess.run(
        [python, "-I", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    modules = [
        "coleman.learning.discrete." + path.stem
        for path in sorted((ROOT / "coleman/learning/discrete").glob("*.py"))
        if path.stem != "__init__"
    ]
    cold = subprocess.run(
        [
            python,
            "-I",
            "-S",
            "-B",
            "-c",
            COLD_SCRIPT,
            json.dumps([site]),
            json.dumps(modules),
            site,
            json.dumps(BLOCKED),
        ],
        cwd=outside,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert cold.returncode == 0, cold.stdout + cold.stderr
    report = json.loads(cold.stdout)
    assert report["forbidden"] == []
    assert all("site-packages" in path for path in report["modules"].values())
