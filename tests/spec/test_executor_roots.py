"""Real installed metadata fixtures for the internal multi-root dependency walker."""

from __future__ import annotations

from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement

import coleman.spec.runtime_identity as runtime
import coleman.spec.selected_runtime as selected
from coleman.spec.selected_runtime import _qualify_extras


def _install(tmp_path, monkeypatch, specifications):
    """Expose actual directory-backed distribution metadata through a finite catalogue."""
    items = {}
    lookups = []
    for name, (requirements, extras) in specifications.items():
        directory = tmp_path / f"{name}-1.0.dist-info"
        directory.mkdir()
        lines = ["Metadata-Version: 2.3", f"Name: {name}", "Version: 1.0"]
        lines.extend(f"Requires-Dist: {value}" for value in requirements)
        lines.extend(f"Provides-Extra: {value}" for value in extras)
        (directory / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
        items[name] = Distribution.at(directory)

    def acquire(name):
        lookups.append(name)
        if name not in items:
            raise PackageNotFoundError(name)
        return items[name]

    monkeypatch.setattr(runtime, "distribution", acquire)
    monkeypatch.setattr(runtime, "distributions", lambda *, name: (items[name],))
    monkeypatch.setattr(selected, "distributions", lambda *, name: (items[name],))
    return items, lookups


def _names(items):
    """Read names only after the principal operation has completed."""
    return tuple(item.metadata["Name"] for item in items)


def test_declared_roots_order_and_real_metadata_objects(tmp_path, monkeypatch):
    items, lookups = _install(
        tmp_path,
        monkeypatch,
        {
            "coleman": (("base>=1",), ()),
            "base": ((), ()),
            "external": (("helper>=1",), ()),
            "helper": ((), ()),
            "unrelated": (("not%%%a-requirement",), ()),
        },
    )
    result = runtime._dependency_distribution_roots((("coleman", frozenset()), ("external", frozenset())))
    assert _names(result) == ("coleman", "base", "external", "helper")
    assert all(item is items[name] for item, name in zip(result, _names(result), strict=True))
    assert lookups == ["coleman", "base", "base", "external", "helper", "helper"]


def test_external_root_does_not_implicitly_add_coleman(tmp_path, monkeypatch):
    items, lookups = _install(tmp_path, monkeypatch, {"external": ((), ())})
    result = runtime._dependency_distribution_roots((("external", frozenset()),))
    assert result == (items["external"],)
    assert lookups == ["external"]


def test_shared_dependency_revisits_union_extras_without_duplicate_result(tmp_path, monkeypatch):
    items, _ = _install(
        tmp_path,
        monkeypatch,
        {
            "coleman": (("shared[fast]>=1",), ()),
            "external": (("shared[slow]>=1",), ()),
            "shared": (("fast-helper; extra == 'fast'", "slow-helper; extra == 'slow'"), ("fast", "slow")),
            "fast-helper": ((), ()),
            "slow-helper": ((), ()),
        },
    )
    visits = []

    def qualify(item, extras):
        visits.append((item.metadata["Name"], frozenset(extras)))
        _qualify_extras(item, extras)

    roots = (("coleman", frozenset()), ("external", frozenset()))
    result = runtime._dependency_distribution_roots(roots, qualify=qualify)
    assert _names(result) == ("coleman", "shared", "fast-helper", "external", "slow-helper")
    assert result[1] is items["shared"]
    assert [extras for name, extras in visits if name == "shared"] == [
        frozenset({"fast"}),
        frozenset({"fast", "slow"}),
    ]
    assert roots == (("coleman", frozenset()), ("external", frozenset()))


def test_cross_root_cycle_self_extra_and_inactive_marker_are_finite(tmp_path, monkeypatch):
    _install(
        tmp_path,
        monkeypatch,
        {
            "coleman": (("external[feature]>=1",), ()),
            "external": (
                ("coleman>=1", "external[self]>=1", "helper; extra == 'self'", "absent; python_version < '1'"),
                ("feature", "self"),
            ),
            "helper": ((), ()),
        },
    )
    visits = []

    def qualify(item, extras):
        visits.append((item.metadata["Name"], frozenset(extras)))
        _qualify_extras(item, extras)

    result = runtime._dependency_distribution_roots(
        (("coleman", frozenset()), ("external", frozenset())), qualify=qualify
    )
    assert _names(result) == ("coleman", "external", "helper")
    assert [extras for name, extras in visits if name == "external"] == [
        frozenset({"feature"}),
        frozenset({"feature", "self"}),
    ]


@pytest.mark.parametrize(
    "roots,message",
    [
        ((), "one to 4096"),
        ([("external", frozenset())], "one to 4096"),
        ((("external",),), "name/extra pairs"),
        ((("external", frozenset()), ("external", frozenset({"extra"}))), "unique"),
        ((("External", frozenset()),), "names must be canonical"),
        ((("external_name", frozenset()),), "names must be canonical"),
        ((("external", set()),), "bounded frozenset"),
        ((("external", frozenset({"Extra"})),), "extras must be canonical"),
        ((("external", frozenset({1})),), "extras must be canonical"),
        ((("external", frozenset(f"extra{i}" for i in range(129))),), "bounded frozenset"),
        (tuple((f"root{i}", frozenset()) for i in range(4097)), "one to 4096"),
    ],
)
def test_invalid_new_roots_refuse_before_metadata(tmp_path, monkeypatch, roots, message):
    _, lookups = _install(tmp_path, monkeypatch, {"external": ((), ())})
    with pytest.raises(ValueError, match=message):
        runtime._dependency_distribution_roots(roots)
    assert lookups == []


def test_exact_root_extra_bound_is_admitted_and_qualifier_sees_copy(tmp_path, monkeypatch):
    extras = frozenset(f"extra{i}" for i in range(128))
    _install(tmp_path, monkeypatch, {"external": ((), tuple(extras))})
    visits = []

    def qualify(item, selected):
        _qualify_extras(item, selected)
        visits.append(frozenset(selected))

    result = runtime._dependency_distribution_roots((("external", extras),), qualify=qualify)
    assert _names(result) == ("external",)
    assert visits == [extras]
    assert len(extras) == 128


def test_active_transitive_extra_growth_keeps_existing_bound(tmp_path, monkeypatch):
    requirement = "shared[" + ",".join(f"extra{i}" for i in range(129)) + "]>=1"
    _, lookups = _install(tmp_path, monkeypatch, {"external": ((requirement,), ()), "shared": ((), ())})
    with pytest.raises(ValueError, match="supported bounds"):
        runtime._dependency_distribution_roots((("external", frozenset()),))
    assert lookups == ["external", "shared"]


@pytest.mark.parametrize(
    "requirements,error,message",
    [
        (("child>=2", "absent"), ValueError, "does not satisfy"),
        (("absent", "child>=2"), PackageNotFoundError, "absent"),
    ],
)
def test_new_roots_keep_declared_first_failure_order(tmp_path, monkeypatch, requirements, error, message):
    _, lookups = _install(tmp_path, monkeypatch, {"external": (requirements, ()), "child": ((), ())})
    with pytest.raises(error, match=message):
        runtime._dependency_distribution_roots((("external", frozenset()), ("later", frozenset())))
    assert lookups == ["external", "child" if requirements[0].startswith("child") else "absent"]


def test_new_roots_parse_all_requirements_before_following_any(tmp_path, monkeypatch):
    _, lookups = _install(tmp_path, monkeypatch, {"external": (("child>=1", "not%%%valid"), ()), "child": ((), ())})
    with pytest.raises(InvalidRequirement):
        runtime._dependency_distribution_roots((("external", frozenset()),))
    assert lookups == ["external"]


def test_legacy_version_recording_precedes_requirement_inspection(tmp_path, monkeypatch):
    items, _ = _install(tmp_path, monkeypatch, {"coleman": ((), ())})
    events = []
    original = items["coleman"]

    class TracedDistribution(Distribution):
        """Delegate to real metadata while recording the legacy observation order."""

        def read_text(self, filename):
            return original.read_text(filename)

        def locate_file(self, path):
            return original.locate_file(path)

        @property
        def version(self):
            events.append("version")
            return original.version

        @property
        def requires(self):
            events.append("requires")
            return original.requires

    items["coleman"] = TracedDistribution()

    def qualify(_item, _extras):
        events.append("qualify")

    result = runtime._dependency_distributions(set(), qualify=qualify)
    assert result == {"coleman": "1.0"}
    assert events[events.index("qualify") :] == ["qualify", "version", "requires"]


def test_returned_object_is_last_acquisition_after_extra_revisit(tmp_path, monkeypatch):
    items, _ = _install(
        tmp_path,
        monkeypatch,
        {"external": (("external[feature]>=1",), ("feature",))},
    )
    visits = []

    def acquire(_name):
        return Distribution.at(Path(items["external"].locate_file("external-1.0.dist-info")))

    monkeypatch.setattr(runtime, "distribution", acquire)

    def qualify(item, _extras):
        visits.append(item)

    result = runtime._dependency_distribution_roots((("external", frozenset()),), qualify=qualify)
    assert len(visits) == 2
    assert result[0] is visits[-1]
    assert result[0] is not visits[0]


class _GuardedDistribution(Distribution):
    """Require an actual raw metadata bound before inherited parsing properties."""

    def __init__(self, inner, path):
        self.inner = inner
        self.metadata_path = path
        self.checked = False
        self.reads = []

    def read_text(self, filename):
        assert self.checked, "Distribution metadata was parsed before preflight"
        self.reads.append(filename)
        return self.inner.read_text(filename)

    def locate_file(self, path):
        return self.inner.locate_file(path)


def _guarded(tmp_path, monkeypatch, specifications):
    items, lookups = _install(tmp_path, monkeypatch, specifications)
    for name, item in items.items():
        items[name] = _GuardedDistribution(item, tmp_path / f"{name}-1.0.dist-info" / "METADATA")
    return items, lookups


def _preflight(item, maximum_bytes=4096):
    """Check real metadata bytes before permitting any Distribution property read."""
    with item.metadata_path.open("rb") as stream:
        content = stream.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ValueError("Installed metadata exceeds selected byte bound")
    item.checked = True


def test_pre_read_bounds_actual_item_and_every_variant_before_parsing(tmp_path, monkeypatch):
    items, _ = _guarded(tmp_path, monkeypatch, {"external": ((), ())})
    variant = _GuardedDistribution(items["external"].inner, items["external"].metadata_path)
    monkeypatch.setattr(runtime, "distributions", lambda *, name: (variant, items[name]))
    observed = []

    def before_read(item):
        observed.append(item)
        _preflight(item)

    result = runtime._dependency_distribution_roots((("external", frozenset()),), before_read=before_read)
    assert observed == [items["external"], variant, items["external"]]
    assert result[0] is items["external"]
    assert items["external"].reads
    assert variant.reads


@pytest.mark.parametrize("oversized", ["item", "variant"])
def test_pre_read_oversized_metadata_refuses_before_any_unbounded_parse(tmp_path, monkeypatch, oversized):
    items, _ = _guarded(tmp_path, monkeypatch, {"external": ((), ())})
    directory = tmp_path / "variant-1.0.dist-info"
    directory.mkdir()
    path = directory / "METADATA"
    path.write_text("Metadata-Version: 2.3\nName: external\nVersion: 1.0\n", encoding="utf-8")
    variant = _GuardedDistribution(Distribution.at(directory), path)
    blocked = items["external"] if oversized == "item" else variant
    blocked.metadata_path.write_text("x" * 4097, encoding="ascii")
    monkeypatch.setattr(runtime, "distributions", lambda *, name: (variant,))
    with pytest.raises(ValueError, match="selected byte bound"):
        runtime._dependency_distribution_roots((("external", frozenset()),), before_read=_preflight)
    assert blocked.reads == []
    assert items["external"].reads == []


def test_pre_read_reaches_transitive_requirement_acquisition_before_version_check(tmp_path, monkeypatch):
    items, lookups = _guarded(tmp_path, monkeypatch, {"external": (("child>=2",), ()), "child": ((), ())})
    observed = []

    def before_read(item):
        observed.append(item)
        _preflight(item)

    with pytest.raises(ValueError, match="does not satisfy"):
        runtime._dependency_distribution_roots((("external", frozenset()),), before_read=before_read)
    assert lookups == ["external", "child"]
    assert observed == [items["external"], items["external"], items["child"], items["child"]]
    assert items["child"].reads


def test_pre_read_transitive_byte_refusal_precedes_incompatible_version(tmp_path, monkeypatch):
    items, lookups = _guarded(tmp_path, monkeypatch, {"external": (("child>=2",), ()), "child": ((), ())})
    items["child"].metadata_path.write_text("x" * 4097, encoding="ascii")
    with pytest.raises(ValueError, match="selected byte bound"):
        runtime._dependency_distribution_roots((("external", frozenset()),), before_read=_preflight)
    assert lookups == ["external", "child"]
    assert items["child"].reads == []


def test_pre_read_is_repeated_on_extra_revisit_and_runs_before_qualify(tmp_path, monkeypatch):
    items, _ = _guarded(tmp_path, monkeypatch, {"external": (("external[feature]>=1",), ("feature",))})
    observed = []

    def before_read(item):
        observed.append(("before", item))
        _preflight(item)

    def qualify(item, extras):
        assert item.checked
        observed.append(("qualify", frozenset(extras)))

    result = runtime._dependency_distribution_roots(
        (("external", frozenset()),), qualify=qualify, before_read=before_read
    )
    assert result == (items["external"],)
    assert sum(kind == "before" for kind, _ in observed) == 8
    assert [value for kind, value in observed if kind == "qualify"] == [frozenset(), frozenset({"feature"})]
    assert observed[:3] == [("before", items["external"]), ("before", items["external"]), ("qualify", frozenset())]


@pytest.mark.parametrize("legacy", [False, True])
def test_absent_hook_keeps_bare_legacy_acquisition_call_shapes(tmp_path, monkeypatch, legacy):
    _install(tmp_path, monkeypatch, {"coleman": (("child>=1",), ()), "child": ((), ())})
    acquire, include, dependency = (
        runtime._installed_distribution,
        runtime._include_requirement,
        runtime._qualified_dependency,
    )
    observed = []

    def bare_acquire(name):
        observed.append(("acquire", name))
        return acquire(name)

    def bare_include(requirement, requested, pending):
        observed.append(("include", requirement.name))
        return include(requirement, requested, pending)

    def bare_dependency(requirement):
        observed.append(("dependency", requirement.name))
        return dependency(requirement)

    monkeypatch.setattr(runtime, "_installed_distribution", bare_acquire)
    monkeypatch.setattr(runtime, "_include_requirement", bare_include)
    monkeypatch.setattr(runtime, "_qualified_dependency", bare_dependency)
    if legacy:
        result = runtime._dependency_distributions(set())
        assert result == {"coleman": "1.0", "child": "1.0"}
    else:
        result = runtime._dependency_distribution_roots((("coleman", frozenset()),), before_read=None)
        assert _names(result) == ("coleman", "child")
    assert observed == [
        ("acquire", "coleman"),
        ("include", "child"),
        ("dependency", "child"),
        ("acquire", "child"),
        ("acquire", "child"),
    ]


def test_root_preflight_rejects_before_metadata_or_pre_read_callback(tmp_path, monkeypatch):
    _, lookups = _install(tmp_path, monkeypatch, {"external": ((), ())})
    observed = []

    def before_read(item):
        observed.append(item)

    with pytest.raises(ValueError, match="canonical"):
        runtime._dependency_distribution_roots((("External", frozenset()),), before_read=before_read)
    assert observed == []
    assert lookups == []
