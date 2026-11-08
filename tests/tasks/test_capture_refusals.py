"""Public capture/loading refusals with real source copies and effect counters."""

from __future__ import annotations

import csv
import json
import os
import py_compile
import subprocess
import sys
import sysconfig
from pathlib import Path
from textwrap import indent

import pytest

from coleman.application.executor_capture import CaptureLimits
from tests.copied_coverage import finish_copied_coverage, prepare_copied_coverage
from tests.tasks.test_executor_capture import _CHILD, _distribution, _installation, _record, _run


def _fixture(tmp_path: Path, *, package: bool = False) -> dict[str, Path]:
    """Keep the original actual provider, adding counters before its callbacks."""
    installation = _installation(tmp_path, package=package)
    path = installation["source"]
    body = path.read_text()
    boundary = "SENTINELS.mkdir(exist_ok=True)\n"
    counter = """
def _record_call(name):
    path = SENTINELS / name
    count = int(path.read_text()) if path.exists() else 0
    path.write_text(str(count + 1))
"""
    assert body.count(boundary) == 1
    body = body.replace(boundary, boundary + counter, 1)
    replacements = {
        '(SENTINELS / "provider-imported").write_text("actual provider module executed")': (
            '_record_call("provider-imported")'
        ),
        '(SENTINELS / "factory-called").write_text(str(fixture_dependency.VALUE))': '_record_call("factory-called")',
        '(SENTINELS / "prepare-called").write_text("must not run while loading")': '_record_call("prepare-called")',
    }
    for original, replacement in replacements.items():
        assert body.count(original) == 1
        body = body.replace(original, replacement, 1)
    path.write_text(body)
    return installation


def _calls(installation: dict[str, Path]) -> tuple[int, int, int]:
    """Read actual provider/factory/prepare counts; absent markers mean zero."""
    values = tuple(
        int(path.read_text()) if path.exists() else 0
        for path in (
            installation["sentinels"] / name for name in ("provider-imported", "factory-called", "prepare-called")
        )
    )
    return values[0], values[1], values[2]


def _refused(result: dict, message: str, installation: dict[str, Path], *, effects=(0, 0, 0)) -> None:
    """Require the intended public refusal and the exact actual effect boundary."""
    assert result["outcome"] == "rejected", result
    assert message in result["message"], result
    assert _calls(installation) == effects


def _rows(installation: dict[str, Path]) -> list[list[str]]:
    """Read the actual bounded provider inventory for deliberate fixture changes."""
    with (installation["provider_metadata"] / "RECORD").open(newline="") as stream:
        return list(csv.reader(stream))


def _write_rows(installation: dict[str, Path], rows: list[list[str]]) -> None:
    """Write actual RECORD bytes; the principal still parses and validates them."""
    with (installation["provider_metadata"] / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _declared_file_count(installation: dict[str, Path]) -> int:
    """Count all actual bounded distribution rows without retaining open files."""
    count = 0
    for path in installation["site"].glob("*.dist-info/RECORD"):
        with path.open(newline="") as stream:
            count += sum(1 for _ in csv.reader(stream))
    return count


def _module(installation: dict[str, Path], module: str) -> None:
    """Freeze one consistent alternate recipe/pointer before actual capture."""
    recipe = json.loads(installation["recipe"].read_text())
    recipe["entry_point"]["module"] = module
    body = json.dumps(recipe)
    installation["recipe"].write_text(body)
    installation["source"].with_suffix(".declarations.json").write_text(body)
    (installation["provider_metadata"] / "entry_points.txt").write_text(
        f"[coleman.executors]\nfixture = {module}:factory\n"
    )


def _run_setup(installation: dict[str, Path], setup: str, *, after_capture: bool = False) -> dict:
    """Prepare an actual unusual host context around the unchanged child body.

    Notes
    -----
    Only trusted test setup is inserted. The real scanner, loader, factory and
    codec validation are unchanged, with the original isolated flags and30s
    deadline. The shared exact-byte coverage wrapper applies when active.
    """
    root = Path(__file__).parents[2]
    paths = [str(root), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    child = _CHILD.replace("SOURCE_PATHS", repr(paths))
    boundary = (
        "    initial = identity.fingerprint\n" if after_capture else "try:\n    identity = capture_executor_identity"
    )
    insertion = indent(setup, "    ") + "\n" if after_capture else setup + "\n"
    assert child.count(boundary) == 1
    child = child.replace(boundary, insertion + boundary, 1)
    payload: dict[str, object] = {key: str(value) for key, value in installation.items()}
    payload.update(action="load", limits={}, extra_paths=[])
    request = installation["site"].parent / "refusal-request.json"
    request.write_text(json.dumps(payload))
    measurement = prepare_copied_coverage(root / "coleman", installation["site"] / "coleman", raw_parent=request.parent)
    if measurement is not None:
        child = measurement.instrument(child, before="from coleman.application.executor_capture import (")
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", child, str(request)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    if measurement is not None:
        finish_copied_coverage(measurement)
    return json.loads(result.stdout)


@pytest.mark.parametrize("metadata", ("[]", '{"dir_info":[]}', '{"dir_info":{"editable":1}}'))
def test_nonobject_or_nonboolean_installation_origin_refuses_before_code(tmp_path, metadata):
    """Malformed PEP610 shapes cannot authorize a directory installation."""
    installation = _fixture(tmp_path)
    (installation["provider_metadata"] / "direct_url.json").write_text(metadata)
    expected = "must be an object" if metadata == "[]" else "editable"
    _refused(_run(installation), expected, installation)


@pytest.mark.parametrize("kind", ("egg_info", "symlink"))
def test_nonwheel_or_symlinked_metadata_directory_refuses_before_parsing(tmp_path, kind):
    """Real discovery must not turn unsupported metadata roots into approval."""
    installation = _fixture(tmp_path)
    metadata = installation["provider_metadata"]
    if kind == "egg_info":
        target = installation["site"] / "fixture_provider.egg-info"
        metadata.rename(target)
        (target / "PKG-INFO").write_bytes((target / "METADATA").read_bytes())
        expected = "dist-info"
    else:
        target = tmp_path / "outside-metadata"
        metadata.rename(target)
        metadata.symlink_to(target, target_is_directory=True)
        expected = "symlink"
    _refused(_run(installation), expected, installation)


@pytest.mark.parametrize("which", ("requirements", "entry_points", "distributions"))
def test_actual_declaration_expansion_is_bounded_before_imports(tmp_path, which):
    """Bound complete real declaration sets before following or loading code."""
    installation = _fixture(tmp_path)
    if which == "entry_points":
        path = installation["provider_metadata"] / "entry_points.txt"
        path.write_text(path.read_text() + "\n[console_scripts]\nother-command = fixture_executor:factory\n")
        limits, message = {"max_entry_points": 1}, "entry-point declaration count"
    elif which == "requirements":
        limits, message = {"max_requirements": 1}, "dependency declaration count"
    else:
        limits, message = {"max_distributions": 2}, "metadata acquisition count"
    _refused(_run(installation, limits=limits), message, installation)


@pytest.mark.parametrize("case", ("short_row", "empty_path", "long_path", "empty_inventory", "duplicate"))
def test_record_shape_and_unique_inventory_are_required_before_code(tmp_path, case):
    """A wheel inventory must retain complete three-column unique paths."""
    installation = _fixture(tmp_path)
    variants = {
        "short_row": [["fixture_executor.py", ""]],
        "empty_path": [["", "", ""]],
        "long_path": [["x" * 1025, "", ""]],
        "empty_inventory": [],
        "duplicate": _rows(installation) + [["fixture_executor.py", "", ""]],
    }
    _write_rows(installation, variants[case])
    message = "unique nonempty" if case in {"empty_inventory", "duplicate"} else "invalid or excessive"
    _refused(_run(installation), message, installation)


@pytest.mark.parametrize("missing", ("METADATA", "WHEEL", "RECORD"))
def test_record_must_declare_its_actual_complete_wheel_metadata(tmp_path, missing):
    """Existing metadata bytes do not replace their required inventory entries."""
    installation = _fixture(tmp_path)
    excluded = f"{installation['provider_metadata'].name}/{missing}"
    _write_rows(installation, [row for row in _rows(installation) if row[0] != excluded])
    _refused(_run(installation), "declare its own complete wheel metadata", installation)


@pytest.mark.parametrize("path", ("/outside.bin", "./fixture_executor.py", "nested//file", "bad\\name", "C:outside"))
def test_record_paths_require_portable_normalized_owned_names(tmp_path, path):
    """Absolute, ambiguous or foreign-platform names cannot acquire resources."""
    installation = _fixture(tmp_path)
    _write_rows(installation, _rows(installation) + [[path, "", ""]])
    _refused(_run(installation), "normalized owned relative paths", installation)


def test_executable_path_file_is_refused_without_running_its_payload(tmp_path):
    """A declared .pth file is unsupported even with a literal local payload."""
    installation = _fixture(tmp_path)
    path = installation["site"] / "fixture.pth"
    marker = installation["sentinels"] / "path-file-executed"
    path.write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('forbidden')\n")
    _write_rows(installation, _rows(installation) + [[path.name, "", ""]])
    _refused(_run(installation), "executable path files", installation)
    assert not marker.exists()


def test_missing_recorded_package_resource_cannot_be_silently_filtered(tmp_path):
    """A real missing nested file stays missing through actual RECORD parsing."""
    installation = _fixture(tmp_path, package=True)
    installation["source"].unlink()
    _refused(_run(installation), "recorded resources are missing", installation)


@pytest.mark.parametrize("package", (False, True))
def test_actual_owned_fifo_refuses_without_a_writer_or_provider_import(tmp_path, package):
    """Neither a top-level nor nested special file may block resource capture."""
    installation = _fixture(tmp_path, package=package)
    installation["source"].unlink()
    os.mkfifo(installation["source"])
    _refused(_run(installation), "special files", installation)


def test_actual_directory_entries_are_charged_beyond_declared_file_count(tmp_path):
    """A complete RECORD file allowance does not exempt scoped directories."""
    installation = _fixture(tmp_path)
    count = _declared_file_count(installation)
    _refused(_run(installation, limits={"max_files": count, "max_entries": count}), "scoped entry count", installation)


def test_actual_owned_directory_depth_has_a_finite_capture_limit(tmp_path):
    """The copied SDK's genuine nested resources consume the same depth cap."""
    installation = _fixture(tmp_path)
    _refused(_run(installation, limits={"max_depth": 1}), "directory depth", installation)


def test_complete_actual_resource_bytes_must_fit_the_shared_byte_allowance(tmp_path):
    """Real metadata and resource bytes cannot silently exceed the total cap."""
    installation = _fixture(tmp_path)
    files = [path for path in installation["site"].rglob("*") if path.is_file()]
    metadata = [path for path in files if path.parent.name.endswith(".dist-info")]
    limits = {
        "max_total_bytes": sum(path.stat().st_size for path in files) - 1,
        "max_file_bytes": max(path.stat().st_size for path in files),
        "max_metadata_bytes": max(path.stat().st_size for path in metadata),
        "max_metadata_total_bytes": sum(path.stat().st_size for path in metadata),
    }
    _refused(_run(installation, limits=limits), "bounded regular file", installation)


def test_existing_selected_cache_consumes_an_additional_file_slot(tmp_path):
    """Immutable existing bytecode contributes to capacity instead of vanishing."""
    installation = _fixture(tmp_path)
    py_compile.compile(str(installation["site"] / "fixture_dependency.py"), doraise=True)
    count = _declared_file_count(installation)
    _refused(_run(installation, limits={"max_files": count}), "selected caches exceed", installation)


def test_selected_flat_cache_must_itself_be_a_regular_owned_file(tmp_path):
    """A no-follow cache parent cannot authorize a selected symlink leaf."""
    installation = _fixture(tmp_path)
    compiled = py_compile.compile(str(installation["site"] / "fixture_dependency.py"), doraise=True)
    assert compiled is not None
    cache = Path(compiled)
    outside = tmp_path / "outside-cache.bin"
    outside.write_bytes(cache.read_bytes())
    cache.unlink()
    cache.symlink_to(outside)
    _refused(_run(installation), "selected cache must be a regular", installation)


def test_record_cannot_claim_a_nested_shared_flat_cache_directory(tmp_path):
    """A RECORD does not authorize walking deeper shared cache namespaces."""
    installation = _fixture(tmp_path)
    relative = "__pycache__/nested/foreign.pyc"
    path = installation["site"] / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"actual unqualified nested cache bytes")
    _write_rows(installation, _rows(installation) + [[relative, "", ""]])
    _refused(_run(installation), "flat cache paths require their exact standard parent", installation)


def test_metadata_only_dependency_is_retained_and_counts_against_the_same_capacity(tmp_path):
    """A real metapackage has owned metadata without invented source/cache files."""
    installation = _fixture(tmp_path)
    metadata = installation["provider_metadata"] / "METADATA"
    metadata.write_text(metadata.read_text() + "Requires-Dist: fixture-meta==1.0\n")
    _distribution(installation["site"], "fixture-meta", "1.0", [])
    accepted = _run(installation)
    assert accepted["outcome"] == "captured", accepted
    assert "fixture-meta" in {item["name"] for item in accepted["identity"]["distributions"]}
    assert _calls(installation) == (0, 0, 0)
    _refused(_run(installation, limits={"max_distributions": 3}), "metadata acquisition count", installation)


@pytest.mark.parametrize(
    "module", ("absent_selected_module", "fixture_executor.child", "fixture_dependency", ".".join(["a"] * 65))
)
def test_selected_module_must_exist_with_regular_parents_and_the_declared_owner(tmp_path, module):
    """Actual PathFinder resolution must join the complete named owned path."""
    installation = _fixture(tmp_path)
    _module(installation, module)
    message = {
        "absent_selected_module": "regular source/extension origins",
        "fixture_executor.child": "parent module is not a regular package",
        "fixture_dependency": "not owned by its approved distribution",
    }.get(module, "module nesting")
    _refused(_run(installation, "load"), message, installation)


def test_sourceless_provider_is_not_loaded_as_a_regular_source_profile(tmp_path):
    """A real pyc-only module has an explicitly unsupported standard loader."""
    installation = _fixture(tmp_path)
    target = installation["site"] / "fixture_executor.pyc"
    py_compile.compile(str(installation["source"]), cfile=str(target), doraise=True)
    installation["source"].unlink()
    _record(
        installation["site"], installation["provider_metadata"], [target.name, "fixture_executor.declarations.json"]
    )
    _refused(_run(installation, "load"), "regular source/extension origins", installation)


@pytest.mark.parametrize(
    "case,message,effects",
    [
        ("not_callable", "zero-argument bootstrap", (1, 0, 0)),
        ("missing_prepare", "definition/capabilities differ", (1, 1, 0)),
        ("wrong_definition", "definition/capabilities differ", (1, 1, 0)),
        ("missing_codec", "concrete codec differs", (1, 1, 0)),
        ("undeclared_progress", "undeclared optional progress", (1, 1, 0)),
        ("missing_validate", "all declared concrete operations", (1, 1, 0)),
        ("missing_decode", "all declared concrete operations", (1, 1, 0)),
        ("missing_encode", "all declared concrete operations", (1, 1, 0)),
        ("unloaded_itself", "complete module path", (1, 1, 0)),
    ],
)
def test_approved_actual_bootstrap_must_supply_its_complete_concrete_capabilities(tmp_path, case, message, effects):
    """Approved code may initialize, but malformed capabilities never prepare."""
    installation = _fixture(tmp_path)
    if case == "not_callable":
        addition = "\nfactory = 7\n"
    elif case == "missing_prepare":
        addition = "\nFixtureExecutor.prepare = None\n"
    elif case == "unloaded_itself":
        addition = (
            "\nimport sys\n_original_factory = factory\ndef factory():\n"
            "    result = _original_factory()\n    del sys.modules[__name__]\n    return result\n"
        )
    elif case.startswith("missing_") and case != "missing_codec":
        addition = f"\nTupleCodec.{case.removeprefix('missing_')} = None\n"
    else:
        argument = {
            "missing_codec": "input_codec=None",
            "undeclared_progress": 'progress_proof_codec=TupleCodec(RECIPE.codec_for("configuration"))',
            "wrong_definition": (
                'definition=DefinitionBinding(reference=RECIPE.reference, definition_fingerprint="sha256:" + "0" * 64)'
            ),
        }[case]
        addition = (
            "\nfrom dataclasses import replace\nfrom coleman.objectives.bindings import DefinitionBinding\n"
            f"_original_factory = factory\ndef factory():\n    return replace(_original_factory(), {argument})\n"
        )
    source = installation["source"]
    source.write_text(source.read_text() + addition)
    _refused(_run(installation, "load"), message, installation, effects=effects)


@pytest.mark.parametrize("target", ("require", "load"))
def test_retained_other_recipe_refuses_before_an_approved_provider_is_imported(tmp_path, target):
    """An earlier valid image cannot be rebound to a different current recipe."""
    installation = _fixture(tmp_path)
    setup = """
other = type(recipe).model_validate({**recipe.model_dump(), "reference": {"id": "another-recipe", "version": "1"}})
"""
    if target == "require":
        setup += "require_executor_identity(identity, recipe=other)\n"
        message = "differs from the current approved recipe"
    else:
        setup += """
retained = type(identity).model_validate({**identity.model_dump(), "recipe": other})
approval = ApprovedExecutor(recipe=recipe, implementation=identity.binding)
load_approved_executor(approval, expected=retained)
"""
        message = "does not match current host approval"
    _refused(_run_setup(installation, setup, after_capture=True), message, installation)


@pytest.mark.parametrize(
    "case", ("nonstr_path", "path_capacity", "cache_capacity", "source_only_hook", "source_only_cache")
)
def test_unsupported_actual_host_import_context_refuses_before_callbacks(tmp_path, case):
    """Actual standard finder instances with changed scope are not silently trusted."""
    installation = _fixture(tmp_path)
    setup = {
        "nonstr_path": 'sys.path.insert(0, Path(configuration["site"]))',
        "path_capacity": 'sys.path[:0] = [configuration["site"]] * 4097',
        "cache_capacity": 'sys.path_importer_cache.update({f"authored-{i}": None for i in range(8193)})',
        "source_only_hook": (
            "from importlib.machinery import FileFinder, SourceFileLoader\n"
            'sys.path_hooks[1] = FileFinder.path_hook((SourceFileLoader, [".py"]))'
        ),
        "source_only_cache": (
            "from importlib.machinery import FileFinder, SourceFileLoader\n"
            'sys.path_importer_cache[configuration["site"]] = '
            'FileFinder(configuration["site"], (SourceFileLoader, [".py"]))'
        ),
    }[case]
    _refused(_run_setup(installation, setup), "profile", installation)


@pytest.mark.parametrize("case", ("missing_spec", "wrong_file", "wrong_loader", "changed_path"))
def test_preloaded_real_provider_origin_changes_refuse_before_factory(tmp_path, case):
    """A previously imported real module cannot silently substitute its origin."""
    installation = _fixture(tmp_path)
    setup = 'preloaded = importlib.import_module("fixture_executor")\n'
    setup += {
        "missing_spec": "preloaded.__spec__ = None",
        "wrong_file": 'preloaded.__file__ = str(Path(configuration["site"]) / "fixture_dependency.py")',
        "wrong_loader": (
            "from importlib.machinery import SourcelessFileLoader\n"
            'preloaded.__spec__.loader = SourcelessFileLoader("fixture_executor", preloaded.__file__)'
        ),
        "changed_path": 'preloaded.__path__ = [configuration["site"]]',
    }[case]
    _refused(_run_setup(installation, setup), "preloaded", installation, effects=(1, 0, 0))


def test_metadata_body_limit_must_fit_its_selected_total_before_acquisition():
    """A valid finite body limit cannot exceed the enclosing metadata budget."""
    with pytest.raises(ValueError, match="metadata body capacity"):
        CaptureLimits(max_metadata_bytes=1024, max_metadata_total_bytes=512)


def test_individually_bounded_records_must_also_fit_the_complete_closure(tmp_path):
    """The aggregate file budget is not a fresh allowance per distribution."""
    installation = _fixture(tmp_path)
    counts = []
    for path in installation["site"].glob("*.dist-info/RECORD"):
        with path.open(newline="") as stream:
            counts.append(sum(1 for _ in csv.reader(stream)))
    assert sum(counts) > max(counts)
    _refused(_run(installation, limits={"max_files": max(counts)}), "aggregate declared file count", installation)


def test_real_sdk_cache_is_in_the_image_but_not_the_preserved_legacy_package_digest(tmp_path):
    """Existing SDK bytecode changes the new image, preserving the old v1 digest."""
    installation = _fixture(tmp_path)
    before = _run(installation, "runtime_parity")
    source = installation["site"] / "coleman" / "core" / "primitives.py"
    compiled = py_compile.compile(str(source), doraise=True)
    assert compiled is not None
    after = _run(installation, "runtime_parity")
    assert before["outcome"] == "captured", before
    assert after["outcome"] == "captured", after
    assert after["existing_runtime_wire_identical"] is True
    assert after["identity"]["runtime"]["package_fingerprint"] == before["identity"]["runtime"]["package_fingerprint"]
    assert after["fingerprint"] != before["fingerprint"]
    relative = Path(compiled).relative_to(installation["site"]).as_posix()
    assert relative in {pin["path"] for pin in after["identity"]["files"]}
    assert _calls(installation) == (0, 0, 0)


def test_existing_sdk_cache_also_consumes_the_shared_resource_file_budget(tmp_path):
    """Unrecorded but supported actual SDK caches cannot bypass the file cap."""
    installation = _fixture(tmp_path)
    py_compile.compile(str(installation["site"] / "coleman" / "core" / "primitives.py"), doraise=True)
    _refused(
        _run(installation, limits={"max_files": _declared_file_count(installation)}),
        "captured file count",
        installation,
    )


@pytest.mark.parametrize("selection", ("root", "transitive"))
def test_undeclared_installed_extras_refuse_before_code_even_for_real_metadata(tmp_path, selection):
    """Only actual Provides-Extra declarations can authorize either selection."""
    installation = _fixture(tmp_path)
    if selection == "root":
        recipe = json.loads(installation["recipe"].read_text())
        for declaration in recipe["roots"]:
            if declaration["name"] == "fixture-provider":
                declaration["extras"] = ["not-provided"]
        body = json.dumps(recipe)
        installation["recipe"].write_text(body)
        installation["source"].with_suffix(".declarations.json").write_text(body)
    else:
        path = installation["provider_metadata"] / "METADATA"
        path.write_text(path.read_text().replace("fixture-dependency==1.0", "fixture-dependency[not-provided]==1.0"))
    _refused(_run(installation), "undeclared installed extra", installation)


def test_noneditable_archive_origin_and_stable_public_recapture_do_not_resolve_the_url(tmp_path):
    """PEP610 archive_info{} remains descriptive metadata under exact recapture."""
    installation = _fixture(tmp_path)
    archive = tmp_path / "not-fetched.whl"
    (installation["provider_metadata"] / "direct_url.json").write_text(
        json.dumps({"url": archive.as_uri(), "archive_info": {}})
    )
    result = _run_setup(
        installation, "assert require_executor_identity(identity, recipe=recipe) is None", after_capture=True
    )
    assert result["outcome"] == "loaded", result
    assert _calls(installation) == (1, 1, 0)
    assert not archive.exists()


@pytest.mark.parametrize("case", ("negative_module_cache", "nontext_spec_origin"))
def test_preloaded_cache_and_spec_must_still_describe_a_regular_module(tmp_path, case):
    """A real negative import-cache entry or altered origin cannot reach factory."""
    installation = _fixture(tmp_path)
    if case == "negative_module_cache":
        setup = 'sys.modules["fixture_executor"] = None'
        message, effects = "unknown preloaded module objects", (0, 0, 0)
    else:
        setup = 'preloaded = importlib.import_module("fixture_executor")\npreloaded.__spec__.origin = None'
        message, effects = "preloaded loader/spec differs", (1, 0, 0)
    _refused(_run_setup(installation, setup), message, installation, effects=effects)


@pytest.mark.parametrize("case", ("callable_instance", "custom_function", "finder_subclass", "tuple_suffixes"))
def test_alternate_real_path_hooks_are_inspected_without_invoking_them(tmp_path, case):
    """Unsupported callback forms and real FileFinder profiles cannot run code."""
    installation = _fixture(tmp_path)
    marker = installation["sentinels"] / "host-hook-invoked"
    if case == "callable_instance":
        setup = f"""
class HostHook:
    def __call__(self, path):
        Path({str(marker)!r}).write_text("unexpected host callback")
        raise ImportError(path)
sys.path_hooks[1] = HostHook()
"""
    elif case == "custom_function":
        setup = f"""
def host_hook(path):
    Path({str(marker)!r}).write_text("unexpected host callback")
    raise ImportError(path)
sys.path_hooks[1] = host_hook
"""
    else:
        setup = """
from importlib.machinery import (
    FileFinder, ExtensionFileLoader, SourceFileLoader, SourcelessFileLoader,
    EXTENSION_SUFFIXES, SOURCE_SUFFIXES, BYTECODE_SUFFIXES,
)
"""
        if case == "finder_subclass":
            setup += f"""
class HostFinder(FileFinder):
    def __init__(self, *args, **kwargs):
        Path({str(marker)!r}).write_text("unexpected host finder construction")
        super().__init__(*args, **kwargs)
sys.path_hooks[1] = HostFinder.path_hook(
    (ExtensionFileLoader, EXTENSION_SUFFIXES), (SourceFileLoader, SOURCE_SUFFIXES),
    (SourcelessFileLoader, BYTECODE_SUFFIXES))
"""
        else:
            setup += """
sys.path_hooks[1] = FileFinder.path_hook(
    (ExtensionFileLoader, EXTENSION_SUFFIXES), (SourceFileLoader, tuple(SOURCE_SUFFIXES)),
    (SourcelessFileLoader, BYTECODE_SUFFIXES))
"""
    _refused(_run_setup(installation, setup), "standard CPython path hooks", installation)
    assert not marker.exists()


def test_named_metadata_variant_budget_bounds_actual_repeated_search_results(tmp_path):
    """Even identical catalogue results consume the named-variant query budget."""
    installation = _fixture(tmp_path)
    setup = """
sys.path.insert(0, configuration["site"])
capture_executor_identity(recipe, limits=CaptureLimits(max_distributions=1))
"""
    _refused(_run_setup(installation, setup, after_capture=True), "named metadata variants", installation)


def test_scoped_entry_budget_is_shared_through_the_final_distribution(tmp_path):
    """The last flat resource cannot receive a reset per-distribution allowance."""
    installation = _fixture(tmp_path)
    site = installation["site"]
    # Directory roots select their descendants; flat roots are themselves
    # entries. This raw filesystem count uses no production scanning helper.
    entries = sum(1 for _ in site.rglob("*")) - sum(path.is_dir() for path in site.iterdir())
    files = _declared_file_count(installation)
    assert entries - 1 >= files
    admitted = _run(installation, limits={"max_files": files, "max_entries": entries})
    assert admitted["outcome"] == "captured", admitted
    _refused(
        _run(installation, limits={"max_files": files, "max_entries": entries - 1}),
        "scoped entry count",
        installation,
    )
