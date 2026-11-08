"""Actual directory metadata, stream hashes and import-order capture regressions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

from coleman.application.executor_capture import CaptureLimits
from coleman.application.executor_capture_types import require_capture_profile
from coleman.tasks.identity import ExecutorRecipe
from tests.copied_coverage import finish_copied_coverage, prepare_copied_coverage

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks"
SOURCE = FIXTURES / "executor-capture-v1" / "provider.py.txt"


def _recipe(module="fixture_executor"):
    value = json.loads((FIXTURES / "v1" / "recipe.json").read_text())
    value["reference"] = {"id": "authored-capture-fixture", "version": "1"}
    value["effects"] = "none"
    value["codecs"] = [item for item in value["codecs"] if item["role"] not in {"state", "progress_proof"}]
    value["roots"] = [{"name": "coleman", "extras": []}, {"name": "fixture-provider", "extras": []}]
    value["entry_point"] = {
        "distribution": "fixture-provider",
        "group": "coleman.executors",
        "name": "fixture",
        "module": module,
        "attribute": "factory",
    }
    return ExecutorRecipe.model_validate(value)


def _record(site, metadata, files):
    path = metadata / "RECORD"
    names = [
        *files,
        *[p.relative_to(site).as_posix() for p in metadata.iterdir() if p.is_file() and p.name != "RECORD"],
        path.relative_to(site).as_posix(),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows((name, "", "") for name in names)


def _distribution(site, name, version, files, *, requires=(), entry=None):
    metadata = site / (name.replace("-", "_") + f"-{version}.dist-info")
    metadata.mkdir()
    body = ["Metadata-Version: 2.3", f"Name: {name}", f"Version: {version}"]
    body.extend("Requires-Dist: " + requirement for requirement in requires)
    (metadata / "METADATA").write_text("\n".join(body) + "\n")
    (metadata / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: authored-unit-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    if entry is not None:
        (metadata / "entry_points.txt").write_text("[coleman.executors]\nfixture = " + entry + "\n")
    _record(site, metadata, files)
    return metadata


def _installation(tmp_path, *, package=False):
    site = tmp_path / "site"
    site.mkdir()
    sdk_root = Path(__file__).resolve().parents[2] / "coleman"
    shutil.copytree(
        sdk_root, site / "coleman", symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    )
    sdk_files = [p.relative_to(site).as_posix() for p in (site / "coleman").rglob("*") if p.is_file()]
    sentinels = tmp_path / "sentinels"
    sentinels.mkdir()
    module = "fixture_package.executor" if package else "fixture_executor"
    recipe = _recipe(module)
    source = site / (module.replace(".", "/") + ".py")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(SOURCE.read_text().replace("SENTINEL_DIRECTORY", repr(str(sentinels))))
    source.with_suffix(".declarations.json").write_text(recipe.to_json())
    files = [source.relative_to(site).as_posix(), source.with_suffix(".declarations.json").relative_to(site).as_posix()]
    if package:
        parent = source.parent / "__init__.py"
        parent.write_text(
            "from pathlib import Path\nPath("
            + repr(str(sentinels / "parent-imported"))
            + ").write_text('actual parent')\n"
        )
        files.append(parent.relative_to(site).as_posix())
    dependency = site / "fixture_dependency.py"
    dependency.write_text("VALUE = 1\n")
    coleman = _distribution(site, "coleman", "4.20.0", sdk_files)
    provider = _distribution(
        site,
        "fixture-provider",
        "1.0",
        files,
        requires=("coleman==4.20.0", "fixture-dependency==1.0"),
        entry=module + ":factory",
    )
    dependency_meta = _distribution(site, "fixture-dependency", "1.0", ["fixture_dependency.py"])
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(recipe.to_json())
    return {
        "site": site,
        "sentinels": sentinels,
        "recipe": recipe_path,
        "source": source,
        "provider_metadata": provider,
        "dependency_metadata": dependency_meta,
        "coleman_metadata": coleman,
    }


_CHILD = r"""
import importlib
import importlib.abc
import json
import sys
from pathlib import Path
configuration = json.loads(Path(sys.argv[1]).read_text())
sys.path[:0] = ([*SOURCE_PATHS, configuration["site"]] if configuration.get("sdk_from_source")
                else [configuration["site"], *SOURCE_PATHS])
from coleman.application.executor_capture import (
    CaptureLimits, capture_executor_identity, require_executor_identity, load_approved_executor)
from coleman.tasks.identity import ApprovedExecutor, ExecutorRecipe
import coleman.spec.runtime_identity as runtime_module
original_producer = runtime_module.installed_producer_identity
producer_calls = []
def observed_producer():
    producer_calls.append("actual original producer invoked")
    return original_producer()
runtime_module.installed_producer_identity = observed_producer
recipe = ExecutorRecipe.from_json(Path(configuration["recipe"]).read_bytes())
site = Path(configuration["site"])
# SDK dependencies are already imported. Metadata discovery now sees only the
# authored directory catalogues plus standard-library paths, not unrelated envs.
stdlib = [p for p in sys.path if p and p not in SOURCE_PATHS and "site-packages" not in p]
sys.path[:] = configuration.get("extra_paths", []) + [str(site)] + stdlib
importlib.invalidate_caches()
action = configuration["action"]
capacities = CaptureLimits(**configuration.get("limits", {}))
if action == "custom_meta":
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            Path(configuration["sentinels"], "custom-finder-called").write_text(fullname)
    sys.meta_path.insert(0, Finder())
if action == "custom_hook":
    def hook(path):
        Path(configuration["sentinels"], "custom-hook-called").write_text(path)
        raise ImportError(path)
    sys.path_hooks.append(hook)
if action == "custom_cache":
    sys.path_importer_cache["authored-unsupported-cache"] = object()
try:
    identity = capture_executor_identity(recipe, limits=capacities)
    initial = identity.fingerprint
    load_actions = {"load", "load_without_expected", "changed_provider",
                    "changed_dependency", "changed_cache", "wrong_approval"}
    if action in load_actions:
        if action == "changed_provider":
            p = Path(configuration["source"]);p.write_text(p.read_text() + "\n# actual same-version source change\n")
        if action == "changed_dependency":
            (site / "fixture_dependency.py").write_text("VALUE = 2\n")
        if action == "changed_cache":
            p = Path(configuration["cache"]);p.write_bytes(p.read_bytes() + b"actual-cache-change")
        binding = identity.binding
        if action == "wrong_approval":
            binding = type(binding)(reference=binding.reference, definition_fingerprint="sha256:" + "f" * 64)
        approval = ApprovedExecutor(recipe=recipe, implementation=binding)
        executor = load_approved_executor(
            approval, expected=None if action == "load_without_expected" else identity, limits=capacities)
        result = {"outcome":"loaded", "definition":executor.definition.model_dump(mode="json"),
                  "codec_output":executor.configuration_codec.encode((1.0, -0.0)),
                  "identity":identity.model_dump(mode="json")}
    elif action == "require_changed":
        p=Path(configuration["source"]);p.write_text(p.read_text() + "\n# changed after capture\n")
        require_executor_identity(identity, recipe=recipe, limits=capacities)
        result={"outcome":"unexpected_accepted"}
    else:
        result={"outcome":"captured", "identity":identity.model_dump(mode="json"), "fingerprint":initial}
        if action == "runtime_parity":
            from coleman.spec.selected_runtime import SelectedInstalledRuntime
            actual = SelectedInstalledRuntime.capture(extras=identity.runtime.selected_extras)
            result["existing_runtime_wire_identical"] = actual.to_json() == identity.runtime.to_json()
except (ValueError, TypeError, OSError, ImportError) as error:
    result={"outcome":"rejected", "error_type":type(error).__name__, "message":str(error)}
result["old_producer_calls"] = len(producer_calls)
print(json.dumps(result,sort_keys=True))
"""


def _run(installation, action="capture", *, limits=None, extra_paths=(), python_flags=("-B",), **options):
    payload: dict[str, object] = {key: str(value) for key, value in installation.items()}
    payload.update(action=action, limits=limits or {}, extra_paths=[str(path) for path in extra_paths], **options)
    request = installation["site"].parent / "request.json"
    request.write_text(json.dumps(payload))
    root = str(Path(__file__).resolve().parents[2])
    source_paths = [root, sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    child = _CHILD.replace("SOURCE_PATHS", repr(source_paths))
    measurement = prepare_copied_coverage(
        Path(root) / "coleman", installation["site"] / "coleman", raw_parent=installation["site"].parent
    )
    if measurement is not None:
        child = measurement.instrument(child, before="from coleman.application.executor_capture import (")
    result = subprocess.run(
        [sys.executable, "-I", "-S", *python_flags, "-c", child, str(request)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    if measurement is not None:
        finish_copied_coverage(measurement)
    return json.loads(result.stdout)


def _no_effects(installation):
    assert not (installation["sentinels"] / "provider-imported").exists()
    assert not (installation["sentinels"] / "factory-called").exists()
    assert not (installation["sentinels"] / "prepare-called").exists()


def test_capture_actual_stream_bytes_without_provider_import(tmp_path):
    installation = _installation(tmp_path)
    result = _run(installation)
    assert result["outcome"] == "captured", result
    identity = result["identity"]
    pins = {(row["distribution"], row["path"]): row for row in identity["files"]}
    source = installation["source"].read_bytes()
    pin = pins[("fixture-provider", "fixture_executor.py")]
    assert pin["sha256"] == "sha256:" + hashlib.sha256(source).hexdigest()
    assert pin["size_bytes"] == len(source)
    assert [item["name"] for item in identity["distributions"]] == ["coleman", "fixture-dependency", "fixture-provider"]
    assert identity["reference"]["version"] == "directory-wheel-closure-v1"
    assert result["old_producer_calls"] == 0
    _no_effects(installation)


@pytest.mark.parametrize("action", ("load", "load_without_expected"))
def test_actual_frozen_bootstrap_and_structural_codecs_before_prepare(tmp_path, action):
    installation = _installation(tmp_path)
    result = _run(installation, action)
    assert result["outcome"] == "loaded", result
    assert result["codec_output"] == "[1.0,-0.0]"
    assert (installation["sentinels"] / "provider-imported").exists()
    assert (installation["sentinels"] / "factory-called").exists()
    assert not (installation["sentinels"] / "prepare-called").exists()


@pytest.mark.parametrize("action", ("changed_provider", "changed_dependency", "wrong_approval", "require_changed"))
def test_changed_same_version_image_or_approval_refuses_before_provider_effects(tmp_path, action):
    installation = _installation(tmp_path)
    result = _run(installation, action)
    assert result["outcome"] == "rejected", result
    assert "identity" in result["message"] or "image" in result["message"]
    _no_effects(installation)


@pytest.mark.parametrize("action", ("custom_meta", "custom_hook", "custom_cache"))
def test_custom_import_context_refuses_without_invoking_custom_callbacks(tmp_path, action):
    installation = _installation(tmp_path)
    result = _run(installation, action)
    assert result["outcome"] == "rejected", result
    assert "profile" in result["message"]
    assert list(installation["sentinels"].iterdir()) == []


def test_existing_real_flat_module_cache_bytes_are_bound_before_load(tmp_path):
    installation = _installation(tmp_path)
    compiled = py_compile.compile(str(installation["source"]), doraise=True)
    assert compiled is not None
    cache = Path(compiled)
    result = _run(installation, "changed_cache", cache=str(cache))
    assert result["outcome"] == "rejected", result
    assert "identity" in result["message"]
    _no_effects(installation)


@pytest.mark.parametrize("package", (False, True))
def test_standard_shadow_origins_refuse_before_loading_parent_or_provider(tmp_path, package):
    installation = _installation(tmp_path, package=package)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    if package:
        directory = shadow / "fixture_package"
        directory.mkdir()
        (directory / "__init__.py").write_text("raise AssertionError('shadow parent must never execute')\n")
    else:
        (shadow / "fixture_executor.py").write_text("raise AssertionError('shadow module must never execute')\n")
    result = _run(installation, "load", extra_paths=(shadow,))
    assert result["outcome"] == "rejected", result
    assert "shadowed" in result["message"]
    _no_effects(installation)
    assert not (installation["sentinels"] / "parent-imported").exists()


def test_regular_parent_package_is_checked_and_then_loaded_normally(tmp_path):
    installation = _installation(tmp_path, package=True)
    result = _run(installation, "load")
    assert result["outcome"] == "loaded", result
    assert (installation["sentinels"] / "parent-imported").exists()
    assert not (installation["sentinels"] / "prepare-called").exists()


def test_namespace_parent_is_explicitly_unsupported_before_provider_load(tmp_path):
    installation = _installation(tmp_path, package=True)
    parent = installation["source"].parent / "__init__.py"
    parent.unlink()
    paths = [
        installation["source"].relative_to(installation["site"]).as_posix(),
        installation["source"].with_suffix(".declarations.json").relative_to(installation["site"]).as_posix(),
    ]
    _record(installation["site"], installation["provider_metadata"], paths)
    result = _run(installation, "load")
    assert result["outcome"] == "rejected", result
    assert "namespace" in result["message"]
    _no_effects(installation)


@pytest.mark.parametrize(
    "metadata_case", ("editable", "missing_record", "duplicate_pointer", "wrong_pointer", "malformed_pointer")
)
def test_real_metadata_refusals_happen_before_bootstrap(tmp_path, metadata_case):
    installation = _installation(tmp_path)
    metadata = installation["provider_metadata"]
    if metadata_case == "editable":
        (metadata / "direct_url.json").write_text('{"url":"file:///not-resolved","dir_info":{"editable":true}}')
        message = "editable"
    elif metadata_case == "missing_record":
        (metadata / "RECORD").unlink()
        message = "RECORD"
    else:
        pointer = metadata / "entry_points.txt"
        if metadata_case == "duplicate_pointer":
            pointer.write_text(pointer.read_text() + "fixture = fixture_executor:factory\n")
            message = "one exact"
        elif metadata_case == "wrong_pointer":
            pointer.write_text("[coleman.executors]\nfixture = different_module:factory\n")
            message = "pointer differs"
        else:
            pointer.write_text("[coleman.executors]\nfixture = not a module pointer\n")
            message = "pointer differs"
    result = _run(installation)
    assert result["outcome"] == "rejected", result
    assert message in result["message"]
    _no_effects(installation)


@pytest.mark.parametrize("file_case", ("resource_symlink", "metadata_symlink", "unlisted_resource"))
def test_owned_resource_profiles_refuse_unsafe_or_unlisted_real_files(tmp_path, file_case):
    installation = _installation(tmp_path, package=True)
    if file_case == "unlisted_resource":
        (installation["source"].parent / "not_declared.bin").write_bytes(b"actual unlisted bytes")
        message = "unlisted"
    else:
        target = (
            installation["source"]
            if file_case == "resource_symlink"
            else installation["provider_metadata"] / "METADATA"
        )
        outside = tmp_path / "outside-owned-root.bin"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
        message = None
    result = _run(installation)
    assert result["outcome"] == "rejected", result
    if message is not None:
        assert message in result["message"]
    else:
        assert result["error_type"] in {"OSError", "ValueError"}
    _no_effects(installation)


def test_oversized_actual_metadata_refuses_before_invalid_requirement_parse(tmp_path):
    installation = _installation(tmp_path)
    metadata = installation["coleman_metadata"] / "METADATA"
    metadata.write_text(
        metadata.read_text() + "Requires-Dist: deliberately%%%invalid\n" + "X-Large: " + "x" * 2048 + "\n"
    )
    result = _run(installation, limits={"max_metadata_bytes": 1024})
    assert result["outcome"] == "rejected", result
    assert "bounded regular file" in result["message"]
    _no_effects(installation)


def test_new_capture_limits_are_coupled_and_real_record_count_is_bounded(tmp_path):
    with pytest.raises(ValueError, match="file capacity"):
        CaptureLimits(max_files=4, max_entries=3)
    with pytest.raises(ValueError, match="total bytes"):
        CaptureLimits(max_total_bytes=100)
    installation = _installation(tmp_path)
    result = _run(installation, limits={"max_files": 2, "max_entries": 2})
    assert result["outcome"] == "rejected", result
    assert "RECORD" in result["message"]
    _no_effects(installation)


def test_wrong_concrete_codec_is_rejected_after_approved_factory_before_prepare(tmp_path):
    installation = _installation(tmp_path)
    source = installation["source"]
    source.write_text(
        source.read_text().replace(
            'TupleCodec(RECIPE.codec_for("configuration"))', 'TupleCodec(RECIPE.codec_for("inputs"))'
        )
    )
    result = _run(installation, "load")
    assert result["outcome"] == "rejected", result
    assert "concrete codec differs" in result["message"]
    assert (installation["sentinels"] / "factory-called").exists()
    assert not (installation["sentinels"] / "prepare-called").exists()


def test_exact_generated_launchers_are_bound_as_metadata_but_never_acquired_or_executed(tmp_path):
    installation = _installation(tmp_path)
    metadata = installation["provider_metadata"]
    entry = metadata / "entry_points.txt"
    entry.write_text(entry.read_text() + "\n[console_scripts]\nfixture-command = fixture_executor:factory\n")
    files = ["fixture_executor.py", "fixture_executor.declarations.json", "../../../bin/fixture-command"]
    _record(installation["site"], metadata, files)
    result = _run(installation)
    assert result["outcome"] == "captured", result
    assert not any(".." in row["path"] for row in result["identity"]["files"])
    _no_effects(installation)


def test_unrelated_package_bytes_are_not_walked_or_added_to_the_active_closure(tmp_path):
    installation = _installation(tmp_path)
    site = installation["site"]
    (site / "unrelated.py").write_text("raise AssertionError('unrelated provider must not run')\n")
    _distribution(site, "unrelated", "1.0", ["unrelated.py"])
    original = _run(installation)
    (site / "unrelated.py").write_text("arbitrary unrelated changed bytes\n")
    changed = _run(installation)
    assert original["outcome"] == "captured", original
    assert changed["outcome"] == "captured", changed
    assert changed["fingerprint"] == original["fingerprint"]
    assert "unrelated" not in {row["name"] for row in changed["identity"]["distributions"]}
    _no_effects(installation)


@pytest.mark.parametrize("flags", ((), ("-B", "-O"), ("-B", "-X", "pycache_prefix=not-the-standard-cache")))
def test_capture_profile_flags_are_explicit_without_temporary_process_mutation(tmp_path, flags):
    installation = _installation(tmp_path)
    result = _run(installation, python_flags=flags)
    assert result["outcome"] == "rejected", result
    assert "requires" in result["message"]
    _no_effects(installation)


def test_existing_default_process_must_opt_in_to_no_bytecode_profile():
    if sys.dont_write_bytecode:
        assert require_capture_profile() is None
    else:
        with pytest.raises(ValueError, match="bytecode writes disabled"):
            require_capture_profile()


@pytest.mark.parametrize("active", (False, True))
def test_active_url_requirements_are_unsupported_and_inactive_urls_are_not_resolved(tmp_path, active):
    installation = _installation(tmp_path)
    metadata = installation["provider_metadata"] / "METADATA"
    marker = "" if active else " ; python_version < '1'"
    metadata.write_text(
        metadata.read_text()
        + "Requires-Dist: absent-url-project @ https://example.invalid/no-fetch.whl"
        + marker
        + "\n"
    )
    result = _run(installation)
    assert result["outcome"] == ("rejected" if active else "captured"), result
    if active:
        assert "direct-URL" in result["message"]
    else:
        assert "absent-url-project" not in {row["name"] for row in result["identity"]["distributions"]}
    _no_effects(installation)


def test_provider_induced_coleman_extra_preserves_old_schema2_capture_wire(tmp_path):
    installation = _installation(tmp_path)
    site = installation["site"]
    provider = installation["provider_metadata"] / "METADATA"
    provider.write_text(provider.read_text().replace("coleman==4.20.0", "coleman[cap]==4.20.0"))
    coleman = installation["coleman_metadata"] / "METADATA"
    coleman.write_text(coleman.read_text() + "Provides-Extra: cap\nRequires-Dist: fixture-extra==1.0; extra == 'cap'\n")
    (site / "fixture_extra.bin").write_bytes(b"actual extra-selected resource")
    _distribution(site, "fixture-extra", "1.0", ["fixture_extra.bin"])
    result = _run(installation, "runtime_parity")
    assert result["outcome"] == "captured", result
    assert result["existing_runtime_wire_identical"] is True
    assert result["identity"]["runtime"]["selected_extras"] == ["cap"]
    assert {row["name"] for row in result["identity"]["runtime"]["distributions"]} == {"coleman", "fixture-extra"}
    _no_effects(installation)


def test_zip_distribution_metadata_is_not_mistaken_for_a_directory_wheel(tmp_path):
    installation = _installation(tmp_path)
    archive = tmp_path / "unsupported-provider.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for file in installation["provider_metadata"].iterdir():
            stream.writestr(file.relative_to(installation["site"]).as_posix(), file.read_bytes())
    result = _run(installation, extra_paths=(archive,))
    assert result["outcome"] == "rejected", result
    assert "directory" in result["message"]
    _no_effects(installation)


def test_shared_namespace_resource_claims_and_arbitrary_record_traversal_are_refused(tmp_path):
    installation = _installation(tmp_path, package=True)
    site = installation["site"]
    resource = site / "fixture_package" / "foreign_resource.bin"
    resource.write_bytes(b"second distribution in same claimed package")
    _record(
        site, installation["dependency_metadata"], ["fixture_dependency.py", "fixture_package/foreign_resource.bin"]
    )
    result = _run(installation)
    assert result["outcome"] == "rejected", result
    assert "shared or ambiguous" in result["message"]
    _no_effects(installation)


def test_record_traversal_does_not_authorize_outside_file_access(tmp_path):
    installation = _installation(tmp_path)
    _record(
        installation["site"],
        installation["provider_metadata"],
        ["fixture_executor.py", "fixture_executor.declarations.json", "../../../not-an-approved-launcher"],
    )
    result = _run(installation)
    assert result["outcome"] == "rejected", result
    assert "escapes" in result["message"]
    _no_effects(installation)


def test_preloaded_sdk_origin_must_genuinely_belong_to_the_declared_coleman_image(tmp_path):
    installation = _installation(tmp_path)
    result = _run(installation, sdk_from_source=True)
    assert result["outcome"] == "rejected", result
    assert "preloaded module file differs" in result["message"]
    _no_effects(installation)


def test_actual_default_producer_digest_equals_bounded_streamed_sdk_digest(tmp_path):
    installation = _installation(tmp_path)
    result = _run(installation, "runtime_parity")
    assert result["outcome"] == "captured", result
    assert result["existing_runtime_wire_identical"] is True
    assert result["old_producer_calls"] == 1
    _no_effects(installation)


def test_duplicate_same_version_metadata_locations_are_not_silently_selected(tmp_path):
    installation = _installation(tmp_path)
    extra = tmp_path / "duplicate"
    extra.mkdir()
    shutil.copytree(installation["provider_metadata"], extra / installation["provider_metadata"].name)
    result = _run(installation, extra_paths=(extra,))
    assert result["outcome"] == "rejected", result
    assert "ambiguous" in result["message"]
    _no_effects(installation)


def test_future_wheel_metadata_and_identity_byte_caps_are_explicit_refusals(tmp_path):
    installation = _installation(tmp_path)
    wheel = installation["provider_metadata"] / "WHEEL"
    original = wheel.read_text()
    wheel.write_text(original.replace("Wheel-Version: 1.0", "Wheel-Version: 2.0"))
    refused = _run(installation)
    assert refused["outcome"] == "rejected", refused
    assert "metadata version 1.0" in refused["message"]
    wheel.write_text(original)
    small = _run(installation, limits={"max_identity_bytes": 64})
    assert small["outcome"] == "rejected", small
    assert "maximum_bytes" in small["message"]
    _no_effects(installation)


def test_metadata_fifo_refuses_without_waiting_for_a_writer(tmp_path):
    installation = _installation(tmp_path)
    metadata = installation["provider_metadata"] / "METADATA"
    metadata.unlink()
    os.mkfifo(metadata)
    result = _run(installation)
    assert result["outcome"] == "rejected", result
    assert "bounded regular file" in result["message"]
    _no_effects(installation)


def test_missing_flat_cache_does_not_admit_a_symlinked_cache_parent(tmp_path):
    installation = _installation(tmp_path)
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    (installation["site"] / "__pycache__").symlink_to(outside, target_is_directory=True)
    result = _run(installation)
    assert result["outcome"] == "rejected", result
    _no_effects(installation)


@pytest.mark.parametrize("parent_kind", ("dangling_symlink", "regular_file"))
def test_selected_flat_cache_parent_requires_an_actual_nofollow_directory(tmp_path, parent_kind):
    installation = _installation(tmp_path)
    cache = installation["site"] / "__pycache__"
    if parent_kind == "dangling_symlink":
        cache.symlink_to(tmp_path / "missing-cache-directory", target_is_directory=True)
    else:
        cache.write_bytes(b"not a directory")
    result = _run(installation)
    assert result["outcome"] == "rejected", result
    _no_effects(installation)
