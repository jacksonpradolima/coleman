"""Runtime generations preserve historical bytes and bind explicit selected capabilities."""

import copy
import hashlib
import itertools
import json
from email.message import Message
from functools import partial
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement
from packaging.utils import canonicalize_name
from pydantic import TypeAdapter

import coleman.spec.runtime_identity as legacy
import coleman.spec.selected_runtime as selected
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import (
    RuntimeIdentity,
    SelectedInstalledRuntime,
    read_runtime,
    require_native_runtime,
)

FIXTURES = Path(__file__).with_name("fixtures")
GOLDENS = (
    ("runtime-v1.json", InstalledRuntime, "bfd1edbb1f69d831f5312bbc9250ee148dc6f90e154bdfe1aa2f5e2ae78cd246"),
    ("runtime-v2.json", SelectedInstalledRuntime, "611b1e4ff1602383954a4ad5ff4164c035e6a66bf192347193c343608e9170e7"),
)


class MetadataDistribution(Distribution):
    """Provide synthetic installed metadata; the production traversal and codecs remain real."""

    def __init__(self, name, version="1.0", *, requires=(), extras=(), default_extras=()):
        message = Message()
        if name is not None:
            message["Name"] = name
        message["Version"] = version
        for value in requires:
            message["Requires-Dist"] = value
        for value in extras:
            message["Provides-Extra"] = value
        for value in default_extras:
            message["Default-Extra"] = value
        self.content = str(message)

    def read_text(self, filename):
        return self.content if filename in {"METADATA", "PKG-INFO"} else None

    def locate_file(self, path):
        raise AssertionError("Dependency inventory must not execute or open provider code")


def graph():
    """Include extra propagation, a revisit, a cycle and an inactive platform dependency."""
    return {
        "coleman": MetadataDistribution(
            "coleman",
            requires=(
                "pydantic>=2",
                "packaging>=26",
                'numpy[fast]>=2; extra == "ci"',
                'plugin>=1; extra == "ci"',
                'scipy>=1; extra == "analytics"',
                'mcp>=2; extra == "mcp"',
                'not-installed>=1; sys_platform == "not-a-supported-platform"',
            ),
            extras=("ci", "analytics", "mcp"),
        ),
        "pydantic": MetadataDistribution("pydantic", "2.12.5", requires=("pydantic-core>=2",)),
        "pydantic-core": MetadataDistribution("pydantic-core", "2.41.5"),
        "packaging": MetadataDistribution("packaging", "26.3"),
        "numpy": MetadataDistribution(
            "NumPy",
            "2.4.6",
            requires=('helper>=1; extra == "fast"', 'second-helper>=1; extra == "second"'),
            extras=("fast", "second"),
        ),
        "plugin": MetadataDistribution("plugin", requires=("numpy[second]>=2",)),
        "helper": MetadataDistribution("helper", requires=("numpy[fast]>=2",)),
        "second-helper": MetadataDistribution("second-helper"),
        "scipy": MetadataDistribution("scipy", "1.17.1"),
        "mcp": MetadataDistribution("mcp", "2.3.0"),
    }


def install_metadata(monkeypatch, values, *, variants=None):
    """Replace only the installed metadata boundary; no algorithm or reader is replaced."""

    def distribution(name):
        try:
            return values[canonicalize_name(name)]
        except KeyError as error:
            raise PackageNotFoundError(name) from error

    def distributions(*, name):
        canonical = canonicalize_name(name)
        return (variants or {}).get(canonical, (distribution(canonical),))

    monkeypatch.setattr(legacy, "distribution", distribution)
    monkeypatch.setattr(legacy, "distributions", distributions)
    monkeypatch.setattr(selected, "distributions", distributions)
    monkeypatch.setattr(legacy, "installed_producer_identity", lambda: "sha256:" + "a" * 64)


def payload():
    """Return independent mutable synthetic generation2 fixture content."""
    return json.loads((FIXTURES / "runtime-v2.json").read_bytes())


@pytest.mark.parametrize("name,contract,digest", GOLDENS)
def test_runtime_goldens_retain_explicit_generations_and_exact_bytes(name, contract, digest):
    raw = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    value = read_runtime(raw)
    assert type(value) is contract
    assert value.to_json().encode() + b"\n" == raw
    assert contract.from_json(raw) == value
    assert contract.model_validate_json(raw) == value
    assert TypeAdapter(RuntimeIdentity).validate_python(value) == value
    assert value.fingerprint == "sha256:" + hashlib.sha256(raw.rstrip(b"\n")).hexdigest()


def test_existing_parent_plan_retains_bytes_while_nested_runtime2_is_explicit():
    from coleman.workflows.contracts import ExperimentPlan

    raw = Path("tests/workflows/fixtures/experiment-plan-v1.json").read_bytes().strip()
    old = ExperimentPlan.from_json(raw)
    assert isinstance(old.runtime, InstalledRuntime)
    assert old.to_json().encode() == raw
    data = old.model_dump(mode="json")
    data["runtime"] = payload()
    new = ExperimentPlan.model_validate(data)
    assert new.schema_version == 1
    assert new.contract == "coleman-workflows-v1"
    assert isinstance(new.runtime, SelectedInstalledRuntime)
    assert new.fingerprint != old.fingerprint
    assert new.trials == old.trials
    assert ExperimentPlan.from_json(new.to_json()) == new


@pytest.mark.parametrize("version", [None, 0, 3, -1, True, False, 1.0, 2.0, "1", "2", float("inf")])
def test_runtime_dispatch_never_coerces_or_guesses_a_generation(version):
    data = {**payload(), "schema_version": version}
    with pytest.raises(ValueError):
        TypeAdapter(RuntimeIdentity).validate_python(data)
    if version is not None:
        with pytest.raises(ValueError):
            read_runtime(json.dumps(data))


@pytest.mark.parametrize("version", [None, 0, 3, True, 2.0, "2"])
def test_parent_plan_rejects_unknown_or_coerced_nested_versions_even_with_relaxed_option(version):
    from coleman.workflows.contracts import ExperimentPlan

    data = json.loads(Path("tests/workflows/fixtures/experiment-plan-v1.json").read_bytes())
    data["runtime"] = {**payload(), "schema_version": version}
    with pytest.raises(ValueError):
        ExperimentPlan.model_validate(data)
    with pytest.raises(ValueError):
        ExperimentPlan.model_validate(data, strict=False)


def test_missing_duplicate_and_unknown_fields_reject_in_known_readers():
    data = payload()
    del data["schema_version"]
    with pytest.raises(ValueError):
        TypeAdapter(RuntimeIdentity).validate_python(data)
    raw = (FIXTURES / "runtime-v2.json").read_text().strip()
    duplicate = raw[:-1] + ',"schema_version":2}'
    with pytest.raises(ValueError, match="Duplicate"):
        read_runtime(duplicate)
    with pytest.raises(ValueError, match="Duplicate"):
        SelectedInstalledRuntime.model_validate_json(duplicate)
    for name in ("not-a-field", "capability_override"):
        data = {**payload(), name: "ci"}
        with pytest.raises(ValueError, match="unknown"):
            TypeAdapter(RuntimeIdentity).validate_python(data, extra="ignore")
    with pytest.raises(ValueError):
        TypeAdapter(RuntimeIdentity).validate_python(object())
    with pytest.raises(ValueError, match="typed object"):
        SelectedInstalledRuntime.model_validate(object())


def test_unchecked_instances_cannot_bypass_generation_or_unknown_field_guards():
    runtime = SelectedInstalledRuntime.from_json((FIXTURES / "runtime-v2.json").read_bytes())
    for updates in (
        {"schema_version": True},
        {"schema_version": 2.0},
        {"unknown": 1},
        {"selected_extras": ("ci", "ci")},
    ):
        unchecked = runtime.model_copy(update=updates)
        with pytest.raises(ValueError):
            unchecked.to_json()
    with pytest.raises(ValueError, match="strict"):
        SelectedInstalledRuntime.model_validate(payload(), strict=False)


@pytest.mark.parametrize(
    "updates",
    [
        {"inventory_profile": "unknown"},
        {"selected_extras": ["CI"]},
        {"selected_extras": ["ci", "ci"]},
        {"selected_extras": ["ci_extra"]},
        {"selected_extras": ["bad extra"]},
        {"distributions": []},
        {"distributions": [{"name": "NumPy", "version": "2.4.6"}]},
        {"distributions": [{"name": "bad name", "version": "1.0"}]},
        {"distributions": [{"name": "coleman", "version": "1.0"}] * 2},
    ],
)
def test_selected_inventory_invariants_are_explicit(updates):
    data = {**payload(), **updates}
    with pytest.raises(ValueError):
        SelectedInstalledRuntime.model_validate(data)


def test_declaration_enumeration_permutations_preserve_canonical_identity():
    data = payload()
    reference = None
    for extras in itertools.permutations(("ci", "analytics")):
        for distributions in itertools.permutations(data["distributions"]):
            value = SelectedInstalledRuntime.model_validate(
                {**data, "selected_extras": extras, "distributions": distributions}
            )
            reference = reference or value
            assert value == reference
            assert value.to_json() == reference.to_json()
            assert value.fingerprint == reference.fingerprint


def test_captured_base_remains_base_only_and_native_closure_propagates_transitive_extras(monkeypatch):
    install_metadata(monkeypatch, graph())
    base = InstalledRuntime.capture()
    assert base.schema_version == 1
    assert {item.name for item in base.distributions} == {"coleman", "packaging", "pydantic", "pydantic-core"}
    native = selected.capture_native_runtime()
    assert native.schema_version == 2
    assert native.selected_extras == ("ci",)
    assert {item.name for item in native.distributions} == {
        "coleman",
        "packaging",
        "pydantic",
        "pydantic-core",
        "numpy",
        "plugin",
        "helper",
        "second-helper",
    }
    assert native.package_fingerprint == base.package_fingerprint
    assert native.fingerprint != base.fingerprint
    require_native_runtime(native)
    with pytest.raises(ValueError, match="pinned v4"):
        require_native_runtime(base)


def test_changed_optional_native_version_changes_runtime_without_changing_base_or_package_identity(monkeypatch):
    values = graph()
    install_metadata(monkeypatch, values)
    base, original = InstalledRuntime.capture(), selected.capture_native_runtime()
    values["numpy"] = MetadataDistribution("numpy", "2.5.0", extras=("fast", "second"))
    changed = selected.capture_native_runtime()
    assert InstalledRuntime.capture() == base
    assert changed.package_fingerprint == original.package_fingerprint
    assert changed.fingerprint != original.fingerprint
    with pytest.raises(ValueError, match="incompatible"):
        require_native_runtime(original)


def test_selected_capture_order_and_unrelated_installed_packages_do_not_change_scope(monkeypatch):
    values = graph()
    install_metadata(monkeypatch, values)
    expected = SelectedInstalledRuntime.capture(extras=("ci", "analytics"))
    root = values["coleman"]
    values["coleman"] = MetadataDistribution(
        "coleman", requires=tuple(reversed(root.requires or ())), extras=("mcp", "analytics", "ci")
    )
    values["unrelated-provider"] = MetadataDistribution("unrelated-provider", "99.0")
    assert SelectedInstalledRuntime.capture(extras=("analytics", "ci")) == expected
    assert "unrelated-provider" not in {item.name for item in expected.distributions}
    with pytest.raises(ValueError, match="pinned v4"):
        require_native_runtime(expected)


@pytest.mark.parametrize("extras", [("unknown",), ("ci", "ci"), ("CI",), (True,), ("a",) * 129])
def test_invalid_or_unknown_selected_capabilities_fail_before_package_resource_capture(monkeypatch, extras):
    install_metadata(monkeypatch, graph())

    def forbidden():
        raise AssertionError("Invalid capability cannot certify a producer")

    monkeypatch.setattr(legacy, "installed_producer_identity", forbidden)
    with pytest.raises(ValueError):
        SelectedInstalledRuntime.capture(extras=extras)


def test_base_only_selected_generation_never_qualifies_native_execution(monkeypatch):
    install_metadata(monkeypatch, graph())
    core_only = SelectedInstalledRuntime.capture(extras=())
    assert {item.name for item in core_only.distributions} == {"coleman", "packaging", "pydantic", "pydantic-core"}
    with pytest.raises(ValueError, match="pinned v4"):
        require_native_runtime(core_only)


@pytest.mark.parametrize(
    "failure", ["missing", "version", "name", "duplicate-version", "duplicate-metadata", "default-extra"]
)
def test_selected_capture_rejects_missing_incompatible_or_ambiguous_metadata(monkeypatch, failure):
    values = graph()
    variants = {}
    if failure == "missing":
        del values["numpy"]
    elif failure == "version":
        values["numpy"] = MetadataDistribution("numpy", "1.0", extras=("fast", "second"))
    elif failure == "name":
        values["numpy"] = MetadataDistribution("another-project", "2.4.6", extras=("fast", "second"))
    elif failure == "duplicate-version":
        variants["numpy"] = (values["numpy"], MetadataDistribution("numpy", "2.5.0"))
    elif failure == "duplicate-metadata":
        variants["numpy"] = (values["numpy"], MetadataDistribution("numpy", "2.4.6", extras=("fast", "second")))
    else:
        values["numpy"] = MetadataDistribution("numpy", "2.4.6", default_extras=("fast",))
    install_metadata(monkeypatch, values, variants=variants)
    with pytest.raises((ValueError, PackageNotFoundError)):
        selected.capture_native_runtime()


def test_same_version_duplicate_requirement_order_is_equivalent(monkeypatch):
    values = graph()
    numpy = values["numpy"]
    reordered = MetadataDistribution(
        "numpy", "2.4.6", requires=tuple(reversed(numpy.requires or ())), extras=("second", "fast")
    )
    install_metadata(monkeypatch, values, variants={"numpy": (numpy, reordered)})
    assert selected.capture_native_runtime().selected_extras == ("ci",)


def test_selected_unknown_transitive_extra_and_malformed_provided_metadata_reject(monkeypatch):
    values = graph()
    values["numpy"] = MetadataDistribution("numpy", "2.4.6", extras=("fast",))
    install_metadata(monkeypatch, values)
    with pytest.raises(ValueError, match="not declared"):
        selected.capture_native_runtime()
    values["numpy"] = MetadataDistribution("numpy", "2.4.6", extras=("fast", "second", "bad extra"))
    with pytest.raises(ValueError, match="extra metadata"):
        selected.capture_native_runtime()
    values["numpy"] = MetadataDistribution("numpy", "2.4.6", extras=("fast", "second", "fast"))
    with pytest.raises(ValueError, match="extra metadata"):
        selected.capture_native_runtime()


def test_transitive_extra_aliases_use_packaging_normalization(monkeypatch):
    values = graph()
    values["plugin"] = MetadataDistribution("plugin", requires=("numpy[other_extra]>=2",))
    values["numpy"] = MetadataDistribution("numpy", "2.4.6", extras=("fast", "other-extra"))
    install_metadata(monkeypatch, values)
    assert selected.capture_native_runtime().selected_extras == ("ci",)


def test_existing_legacy_empty_inventory_is_not_reinterpreted():
    data = json.loads((FIXTURES / "runtime-v1.json").read_bytes())
    data["distributions"] = []
    value = read_runtime(json.dumps(data))
    assert type(value) is InstalledRuntime
    assert value.distributions == ()


def test_unknown_v2_nested_keys_reject_even_if_parent_reader_ignores_extras():
    from coleman.workflows.contracts import ExperimentPlan

    data = json.loads(Path("tests/workflows/fixtures/experiment-plan-v1.json").read_bytes())
    data["runtime"] = {**payload(), "unrecognized": True}
    with pytest.raises(ValueError, match="unknown fields"):
        ExperimentPlan.model_validate(data, extra="ignore")


def test_parent_generation2_does_not_authorize_reusing_legacy_checkpoint_binding():
    from coleman.results.domain import fingerprint
    from coleman.workflows.contracts import ExperimentPlan

    data = json.loads(Path("tests/workflows/fixtures/experiment-plan-v1.json").read_bytes())
    old = ExperimentPlan.model_validate(data)
    changed = ExperimentPlan.model_validate({**copy.deepcopy(data), "runtime": payload()})
    trial = old.trials[0]
    assert changed.trials[0].trial_id == trial.trial_id
    assert fingerprint([old.fingerprint, trial.trial_id]) != fingerprint([changed.fingerprint, trial.trial_id])


@pytest.mark.parametrize("bound", ["distributions", "transitive-extras"])
def test_metadata_traversal_rejects_inventory_growth_beyond_qualified_bounds(monkeypatch, bound):
    if bound == "distributions":
        children = tuple(f"node-{index}" for index in range(4096))
        values = {name: MetadataDistribution(name) for name in children}
        values["coleman"] = MetadataDistribution("coleman", requires=children, extras=("ci",))
    else:
        extras = tuple(f"feature-{index}" for index in range(129))
        values = {
            "coleman": MetadataDistribution("coleman", requires=("child[" + ",".join(extras) + "]",), extras=("ci",)),
            "child": MetadataDistribution("child", extras=extras),
        }
    install_metadata(monkeypatch, values)
    with pytest.raises(ValueError, match="bounds"):
        selected.capture_native_runtime()


def test_legacy_reader_retains_its_original_canonical_name_validation():
    data = json.loads((FIXTURES / "runtime-v1.json").read_bytes())
    data["distributions"] = [{"name": "Non_Canonical", "version": "1.0"}]
    with pytest.raises(ValueError, match="unique canonical"):
        InstalledRuntime.model_validate(data)


def test_duplicate_metadata_with_missing_name_cannot_certify_a_selected_closure(monkeypatch):
    values = graph()
    unnamed = MetadataDistribution(None, "2.4.6", extras=("fast", "second"))
    install_metadata(monkeypatch, values, variants={"numpy": (values["numpy"], unnamed)})
    with pytest.raises(ValueError, match="names must be strings"):
        selected.capture_native_runtime()


@pytest.mark.parametrize(
    "updates",
    [
        {"machine": b"x86_64"},
        {"selected_extras": [b"ci"]},
        {"distributions": [{"name": "coleman", "version": b"0.0.0"}]},
        {"distributions": [{"name": "coleman", "version": "0.0.0", "unknown": True}]},
    ],
)
def test_legacy_parent_cannot_relax_new_selected_runtime_leaf_contracts(updates):
    from coleman.workflows.contracts import ExperimentPlan

    data = json.loads(Path("tests/workflows/fixtures/experiment-plan-v1.json").read_bytes())
    data["runtime"] = {**payload(), **updates}
    with pytest.raises(ValueError):
        ExperimentPlan.model_validate(data, strict=False, extra="ignore")


def metadata_lookups(monkeypatch):
    """Observe named metadata acquisition without replacing dependency validation."""
    observed = []
    original = legacy.distribution

    def tracing(name):
        observed.append(canonicalize_name(name))
        return original(name)

    monkeypatch.setattr(legacy, "distribution", tracing)
    return observed


@pytest.mark.parametrize(
    "capture", [InstalledRuntime.capture, partial(SelectedInstalledRuntime.capture, extras=("ci",))]
)
def test_root_self_requirement_can_activate_its_own_extra_without_looping(monkeypatch, capture):
    """Qualify a self-referencing root's transitive extra in both generations."""
    values = {
        "coleman": MetadataDistribution(
            "coleman",
            requires=("coleman[feature]>=1", 'child>=1; extra == "feature"'),
            extras=("ci", "feature"),
        ),
        "child": MetadataDistribution("child"),
    }
    install_metadata(monkeypatch, values)
    runtime = capture()
    assert {item.name for item in runtime.distributions} == {"coleman", "child"}


@pytest.mark.parametrize(
    "capture", [InstalledRuntime.capture, partial(SelectedInstalledRuntime.capture, extras=("ci",))]
)
@pytest.mark.parametrize("malformed", ["not%%%valid", "absent; extra ==="])
def test_malformed_metadata_is_rejected_before_following_any_requirement(monkeypatch, capture, malformed):
    """Preserve eager requirement parsing before any valid child is inspected."""
    values = {
        "coleman": MetadataDistribution("coleman", requires=("child>=1", malformed), extras=("ci",)),
        "child": MetadataDistribution("child"),
    }
    install_metadata(monkeypatch, values)
    observed = metadata_lookups(monkeypatch)
    with pytest.raises(InvalidRequirement):
        capture()
    assert observed == ["coleman"]


@pytest.mark.parametrize(
    "requirements,expected_error,expected_message,first_dependency",
    [
        (("child>=2", "absent"), ValueError, "does not satisfy", "child"),
        (("absent", "child>=2"), PackageNotFoundError, "absent", "absent"),
    ],
)
def test_active_requirements_keep_declared_first_failure_order(
    monkeypatch, requirements, expected_error, expected_message, first_dependency
):
    """Keep acquisition errors tied to the first declared active requirement."""
    values = {
        "coleman": MetadataDistribution("coleman", requires=requirements),
        "child": MetadataDistribution("child"),
    }
    install_metadata(monkeypatch, values)
    observed = metadata_lookups(monkeypatch)
    with pytest.raises(expected_error, match=expected_message):
        InstalledRuntime.capture()
    assert observed == ["coleman", first_dependency]
