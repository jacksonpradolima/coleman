"""Declaration-only selected/base joins preserve versioned runtime evidence."""

import hashlib
from pathlib import Path

import pytest

import coleman.spec.selected_runtime as runtime_module
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime

FIXTURES = Path(__file__).with_name("fixtures")


def records():
    """Read independently versioned original base/selected identity fixtures."""
    base = InstalledRuntime.from_json((FIXTURES / "runtime-v1.json").read_bytes())
    selected = SelectedInstalledRuntime.from_json((FIXTURES / "runtime-v2.json").read_bytes())
    return base, selected


def test_declared_join_preserves_original_golden_bytes_and_digests():
    base, selected = records()
    before = base.to_json(), selected.to_json()
    assert selected.require_base(base, extras=("ci",), pins=(("numpy", "2.4.6"),)) is None
    assert (base.to_json(), selected.to_json()) == before
    for record, name in ((base, "runtime-v1.json"), (selected, "runtime-v2.json")):
        expected = (FIXTURES / name).read_text().strip()
        assert record.to_json() == expected
        assert record.fingerprint == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()


def test_declared_join_performs_no_capture_resource_read_or_dependency_discovery(monkeypatch):
    base, selected = records()

    def forbidden(*_arguments, **_keywords):
        raise AssertionError("Declared runtime consistency must not inspect the installed environment")

    with monkeypatch.context() as patch:
        patch.setattr(runtime_module, "_dependency_distributions", forbidden)
        patch.setattr(runtime_module, "_capture_environment", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        selected.require_base(base)
        selected.require_base(base, extras=("ci",), pins=(("numpy", "2.4.6"),))


@pytest.mark.parametrize(
    "field,value",
    (
        ("package_fingerprint", "sha256:" + "b" * 64),
        ("python_implementation", "another-python"),
        ("python_version", "3.14.99"),
        ("operating_system", "another-os"),
        ("machine", "another-machine"),
    ),
)
def test_changed_declared_base_environment_refuses_without_rewriting_records(field, value):
    base, selected = records()
    changed = base.model_copy(update={field: value})
    before = selected.to_json()
    with pytest.raises(ValueError):
        selected.require_base(changed)
    assert selected.to_json() == before


@pytest.mark.parametrize("damage", ("missing", "changed_version", "wrong_selection", "wrong_pin"))
def test_exact_base_dependency_selection_and_required_pins_are_independent(damage):
    base, selected = records()
    if damage == "missing":
        values = base.model_dump()
        values["distributions"] = (*base.distributions, {"name": "unretained", "version": "1"})
        base = InstalledRuntime.model_validate(values)
    elif damage == "changed_version":
        base = InstalledRuntime.model_validate(
            {**base.model_dump(), "distributions": ({"name": "coleman", "version": "different"},)}
        )
    selection = ("cli",) if damage == "wrong_selection" else None
    pins = (("numpy", "different"),) if damage == "wrong_pin" else ()
    with pytest.raises(ValueError):
        selected.require_base(base, extras=selection, pins=pins)


@pytest.mark.parametrize("target,updates", (("selected", {"schema_version": 3}), ("base", {"schema_version": 2})))
def test_unchecked_future_record_cannot_bypass_the_declared_join(target, updates):
    base, selected = records()
    if target == "selected":
        selected = selected.model_copy(update=updates)
    else:
        base = base.model_copy(update=updates)
    with pytest.raises(ValueError):
        selected.require_base(base)


@pytest.mark.parametrize(
    "arguments",
    (
        {"extras": ["ci"]},
        {"extras": ("CI",)},
        {"extras": ("ci", "ci")},
        {"pins": [("numpy", "2.4.6")]},
        {"pins": (("numpy",),)},
        {"pins": ((1, "2.4.6"),)},
    ),
)
def test_invalid_extra_or_pin_requirement_shape_refuses(arguments):
    base, selected = records()
    with pytest.raises((ValueError, TypeError)):
        selected.require_base(base, **arguments)


def test_additional_selected_dependency_does_not_infer_an_extra_or_change_base():
    base, selected = records()
    extended = SelectedInstalledRuntime.model_validate(
        {
            **selected.model_dump(),
            "distributions": (*selected.distributions, {"name": "other", "version": "1"}),
        }
    )
    assert extended.require_base(base) is None
    assert extended.require_base(base, extras=("ci",), pins=(("numpy", "2.4.6"),)) is None
    assert extended.selected_extras == selected.selected_extras == ("ci",)
