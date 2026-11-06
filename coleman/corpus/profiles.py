"""Bounded native descriptions with optional Parquet operations.

Models and JSON readers need no numerical backend. Explicit table operations
load Polars/PyArrow from the CI or analytics extra; profiling never admits a
source or changes its evidence status.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self, cast

from pydantic import Field, field_validator, model_validator

from coleman.artifacts import ArtifactWriter
from coleman.corpus.contracts import DataCard
from coleman.corpus.registry import CorpusRegistry
from coleman.results.domain import Measurement
from coleman.results.versioned import VersionedContract
from coleman.selection.contracts import Immutable, UnitSemantics

if TYPE_CHECKING:
    import polars as pl

Count = Annotated[int, Field(strict=True, ge=0, le=2_000_000)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_DECODED_BYTES = 256 * 1024 * 1024
MAX_ROWS = 2_000_000
RAW_DURATION = UnitSemantics(unit="source-unit-unverified", meaning="raw-Duration-field-unverified-v1")
GROUP_ROWS = UnitSemantics(unit="1", meaning="raw-rows-per-complete-execution-group-v1")


class NumericProfile(Immutable):
    """Raw numeric distribution in explicit units, retaining null/nonfinite/negative/zero distinctions.

    Quantiles use the inverse empirical CDF: sorted[ceil(p*n)-1], with the
    minimum for p=0. These are descriptive row statistics, never independent
    trials or calibrated probabilities; declared duration semantics are inherited.
    """

    semantics: UnitSemantics
    finite_count: Count
    null_count: Count
    nonfinite_count: Count
    negative_count: Count
    zero_count: Count
    minimum: Finite | None
    median: Finite | None
    p95: Finite | None
    maximum: Finite | None

    @field_validator("minimum", "median", "p95", "maximum")
    @classmethod
    def _canonical_zero(cls, value: float | None) -> float | None:
        """Treat both binary64 zero signs as one statistical value on construction and reads."""
        return 0.0 if value == 0 else value

    @model_validator(mode="after")
    def _distribution(self) -> Self:
        """Validate availability, ordered quantiles and finite category counts against their exact ranks."""
        values = (self.minimum, self.median, self.p95, self.maximum)
        if self.negative_count + self.zero_count > self.finite_count:
            raise ValueError("Finite category counts exceed finite observations")
        if self.finite_count == 0:
            if any(value is not None for value in values):
                raise ValueError("Empty distribution requires null quantiles")
        elif any(value is None for value in values) or list(values) != sorted(
            value for value in values if value is not None
        ):
            raise ValueError("Nonempty distribution requires ordered finite quantiles")
        if self.finite_count:
            _quantile_categories(self, tuple(value for value in values if value is not None))
        return self

    @property
    def rows(self) -> int:
        """Return the conserved raw row count, including null and nonfinite values."""
        return self.finite_count + self.null_count + self.nonfinite_count


def _quantile_categories(profile: NumericProfile, values: tuple[float, ...]) -> None:
    """Require each inverse-CDF rank's sign to agree with negative/zero/positive counts."""
    count = profile.finite_count
    ranks = (1, (count + 1) // 2, (95 * count + 99) // 100, count)
    previous_rank, previous_value = 0, 0.0
    for rank, value in zip(ranks, values, strict=True):
        if rank == previous_rank and value != previous_value:
            raise ValueError("The same inverse-CDF rank must have the same quantile value")
        previous_rank, previous_value = rank, value
        if rank <= profile.negative_count:
            compatible = value < 0
        elif rank <= profile.negative_count + profile.zero_count:
            compatible = value == 0
        else:
            compatible = value > 0
        if not compatible:
            raise ValueError("Quantile sign contradicts category counts at its inverse-CDF rank")


class ColumnMissingness(Immutable):
    """One source column's raw null count; NaN is non-null and counted separately for Duration."""

    name: str
    null_count: Count


class NativeSourceProfile(VersionedContract):
    """Version-one source-bound, descriptive native Parquet artifact.

    The embedded card and original admission blockers are unchanged. Duration
    values retain declared units or explicit unverified source units. Build
    IDs and variant execution groups are logical labels, not independent trials.
    Quantities can be recomputed only with the matching source bytes. Reading
    validates invariants, not the truth of an arbitrary producer's observations.
    Legacy cards remain readable separately; missing/future versions fail closed.
    """

    recipe: Literal["native-parquet-description-v1"] = "native-parquet-description-v1"
    card: DataCard
    columns: Annotated[tuple[str, ...], Field(min_length=1, max_length=64)]
    column_missingness: Annotated[tuple[ColumnMissingness, ...], Field(min_length=1, max_length=64)]
    rows: Count
    missing_identity_rows: Count
    unique_observations: Count
    duplicate_identity_rows: Count
    conflicting_identities: Count
    logical_builds: Count
    execution_groups: Count
    missing_group_rows: Count
    failed_rows: Count
    passed_rows: Count
    null_verdict_rows: Count
    invalid_verdict_rows: Count
    raw_duration: NumericProfile
    tests_per_execution_group: NumericProfile
    chronology_status: Literal["not_evaluated"] = "not_evaluated"
    generalization: Literal["insufficient_evidence"] = "insufficient_evidence"

    @model_validator(mode="after")
    def _counts(self) -> Self:
        """Conserve raw rows and reject impossible identities or invented duration semantics."""
        _native_card(self.card)
        _source_bindings(self)
        _column_bindings(self)
        expected_duration = self.card.manifest.outcomes.duration or RAW_DURATION
        if self.raw_duration.semantics != expected_duration or self.tests_per_execution_group.semantics != GROUP_ROWS:
            raise ValueError("Incompatible numeric profile units/meaning")
        if self.raw_duration.rows != self.rows:
            raise ValueError("Duration counts must conserve source rows")
        if self.failed_rows + self.passed_rows + self.null_verdict_rows + self.invalid_verdict_rows != self.rows:
            raise ValueError("Verdict counts must conserve source rows")
        if self.unique_observations + self.duplicate_identity_rows + self.missing_identity_rows != self.rows:
            raise ValueError("Identity counts must conserve source rows")
        if self.conflicting_identities > min(self.unique_observations, self.duplicate_identity_rows):
            raise ValueError("Conflicts require repeated identified observations")
        _group_bindings(self)
        return self

    @property
    def duration_median(self) -> Measurement:
        """Expose raw median in declared or explicitly unverified units; null means unavailable."""
        value = self.raw_duration.median
        semantics = self.card.manifest.outcomes.duration
        return Measurement(
            name="corpus_raw_duration_median",
            value=value,
            unit=semantics.unit if semantics is not None else "source-unit-unverified",
            meaning="raw-Duration-inverse-empirical-cdf-v1:" + (semantics.meaning if semantics else "unverified"),
            direction="none",
            status="observed" if value is not None else "unavailable",
        )


def _source_bindings(profile: NativeSourceProfile) -> None:
    """Reapply available recipe/resource bounds and frozen source-row/byte declarations on reads."""
    manifest = profile.card.manifest
    # _native_card has already required a non-null integral byte count.
    size = cast(int, manifest.source.size_bytes)
    if profile.rows > min(MAX_ROWS, manifest.resources.maximum_rows):
        raise ValueError("Profile rows exceed recipe or declared resource ceiling")
    if size > min(MAX_SOURCE_BYTES, manifest.resources.maximum_source_bytes):
        raise ValueError("Profile source bytes exceed recipe or declared resource ceiling")
    declared_rows = manifest.conversion.input_rows
    if declared_rows is not None and profile.rows != declared_rows:
        raise ValueError("Profile rows disagree with source conversion input-row declaration")
    declarations = {"corpus_source_rows": (profile.rows, "1"), "corpus_source_bytes": (size, "B")}
    for fact in profile.card.measurements:
        expected = declarations.get(fact.name)
        if fact.status == "observed" and expected is not None:
            value, unit = expected
            if fact.value != value or fact.unit != unit:
                raise ValueError("Profile disagrees with observed source row/byte declaration")


def _group_bindings(profile: NativeSourceProfile) -> None:
    """Conserve complete group rows and the positive integral counts underlying their quantiles."""
    known_rows = profile.rows - profile.missing_group_rows
    if bool(profile.logical_builds) != bool(profile.execution_groups) or bool(profile.execution_groups) != bool(
        known_rows
    ):
        raise ValueError("Complete group rows require a logical build and execution group")
    if not profile.logical_builds <= profile.execution_groups <= known_rows:
        raise ValueError("Execution groups must preserve logical build identity")
    if "Variant" not in profile.columns and profile.logical_builds != profile.execution_groups:
        raise ValueError("Without Variant, logical builds and execution groups must match")
    groups = profile.tests_per_execution_group
    if groups.null_count or groups.nonfinite_count or groups.negative_count or groups.zero_count:
        raise ValueError("Observed execution groups require positive finite raw row counts")
    if groups.rows != profile.execution_groups:
        raise ValueError("Group distribution must conserve execution groups")
    _group_sum_bounds(groups, known_rows)


def _group_sum_bounds(groups: NumericProfile, known_rows: int) -> None:
    """Require integral row-count quantiles admitting a total equal to all complete group rows.

    For consecutive fixed ranks, unobserved counts lie between the two anchors.
    Their minimum/maximum possible sums bound the declared total without expanding
    millions of groups. Exact source verification remains a separate operation.
    """
    count = groups.finite_count
    if not count:
        return
    values = tuple(value for value in (groups.minimum, groups.median, groups.p95, groups.maximum) if value is not None)
    if any(not value.is_integer() for value in values):
        raise ValueError("Execution-group row counts require integral quantiles")
    ranks = (1, (count + 1) // 2, (95 * count + 99) // 100, count)
    anchors = sorted(dict(zip(ranks, values, strict=True)).items())
    lower = upper = anchors[0][1]
    for (left_rank, left_value), (right_rank, right_value) in zip(anchors, anchors[1:], strict=False):
        gap = right_rank - left_rank
        lower += (gap - 1) * left_value + right_value
        upper += gap * right_value
    if not lower <= known_rows <= upper:
        raise ValueError("Group quantiles cannot conserve complete source rows")


def _column_bindings(profile: NativeSourceProfile) -> None:
    """Canonicalize column declarations and conserve null counts shared with numeric/verdict summaries."""
    required = set(profile.card.manifest.identity.observation_fields) | {"Duration", "Verdict"}
    if not required.issubset(profile.columns):
        raise ValueError("Missing native profile columns")
    if len(set(profile.columns)) != len(profile.columns):
        raise ValueError("Duplicate column declaration")
    columns = tuple(sorted(profile.columns))
    missingness = tuple(sorted(profile.column_missingness, key=lambda item: item.name))
    if tuple(item.name for item in missingness) != columns or any(
        item.null_count > profile.rows for item in missingness
    ):
        raise ValueError("Missingness must describe every column within source row bounds")
    counts = {item.name: item.null_count for item in missingness}
    if counts["Duration"] != profile.raw_duration.null_count or counts["Verdict"] != profile.null_verdict_rows:
        raise ValueError("Column null counts disagree with numeric/verdict summaries")
    _identity_bindings(profile, counts)
    object.__setattr__(profile, "columns", columns)
    object.__setattr__(profile, "column_missingness", missingness)


def _identity_bindings(profile: NativeSourceProfile, null_counts: dict[str, int]) -> None:
    """Bind missing key rows to column-null union bounds; duplicates require an original identity."""
    identity = profile.card.manifest.identity.observation_fields
    group = ("BuildId", "Variant") if "Variant" in profile.columns else ("BuildId",)
    _missing_union(null_counts, identity, profile.missing_identity_rows, profile.rows)
    _missing_union(null_counts, group, profile.missing_group_rows, profile.rows)
    identity_fields = set[str](identity)
    group_fields = set[str](group)
    extra_identity = sum(null_counts[field] for field in identity_fields - group_fields)
    extra_group = sum(null_counts[field] for field in group_fields - identity_fields)
    if (
        profile.missing_identity_rows > profile.missing_group_rows + extra_identity
        or profile.missing_group_rows > profile.missing_identity_rows + extra_group
    ):
        raise ValueError("Missing identity/group union subsets are inconsistent")
    if bool(profile.rows - profile.missing_identity_rows) != bool(profile.unique_observations):
        raise ValueError("Complete identity rows require at least one unique original observation")
    if group_fields.issubset(identity_fields):
        minimum_unique = profile.execution_groups - (profile.missing_identity_rows - profile.missing_group_rows)
    else:
        minimum_unique = profile.logical_builds - (profile.missing_identity_rows - null_counts["BuildId"])
    if profile.unique_observations < minimum_unique:
        raise ValueError("Unique observations cannot cover the declared complete build/group identities")


def _missing_union(null_counts: dict[str, int], fields: tuple[str, ...], missing: int, rows: int) -> None:
    """Bound a null-key union without assuming nulls in different columns are disjoint."""
    counts = tuple(null_counts[field] for field in fields)
    if not max(counts) <= missing <= min(rows, sum(counts)):
        raise ValueError("Missing identity/group rows disagree with key-column null counts")


def _native_card(card: DataCard) -> None:
    """Restrict this recipe to declared native identities and explicit binary verdicts."""
    manifest = card.manifest
    if (
        manifest.adapter.name != "native-parquet-corpus"
        or manifest.identity.observation_fields not in {("BuildId", "Name"), ("BuildId", "Name", "Variant")}
        or manifest.identity.group_fields != ("BuildId",)
    ):
        raise ValueError("This descriptive recipe requires declared native identity")
    if manifest.outcomes.unit != "failing_test" or {
        (item.source_value, item.verdict) for item in manifest.outcomes.mapping
    } != {("0", 0), ("1", 1)}:
        raise ValueError("Native profile requires explicit binary failing-test verdict mapping")
    if manifest.source.sha256 is None or manifest.source.size_bytes is None:
        raise ValueError("Source digest and byte size are required for profiling")


def _numbers(series: pl.Series, semantics: UnitSemantics) -> NumericProfile:
    """Compute finite binary64 order statistics; retain invalid values as counts without imputation."""
    import polars as pl

    values = series.cast(pl.Float64, strict=True)
    finite = values.filter(values.is_finite()).sort()
    # Binary64 signed zeros are one statistical value; ties must not retain source order.
    finite = finite.set(finite == 0, 0.0)
    count = len(finite)
    return NumericProfile(
        semantics=semantics,
        finite_count=count,
        null_count=values.null_count(),
        nonfinite_count=len(values) - values.null_count() - count,
        negative_count=int((finite < 0).sum()),
        zero_count=int((finite == 0).sum()),
        minimum=float(finite[0]) if count else None,
        median=float(finite[(count - 1) // 2]) if count else None,
        p95=float(finite[(95 * count + 99) // 100 - 1]) if count else None,
        maximum=float(finite[-1]) if count else None,
    )


def _source_frame(card: DataCard, path: Path) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Hash one bounded local snapshot before decoding; never follow a URI or import an adapter.

    Footer row/column/uncompressed-byte limits precede column decoding. These are
    conservative trusted-local artifact limits, not an OS sandbox for hostile
    native codecs. No file is modified and no third-party bytes are published.
    """
    import polars as pl
    import pyarrow.parquet as pq

    manifest = card.manifest
    maximum = min(MAX_SOURCE_BYTES, manifest.resources.maximum_source_bytes)
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("Source exceeds profile byte limit")
    if (
        len(payload) != manifest.source.size_bytes
        or "sha256:" + hashlib.sha256(payload).hexdigest() != manifest.source.sha256
    ):
        raise ValueError("Source bytes disagree with frozen digest/size")
    source = pq.ParquetFile(BytesIO(payload))
    metadata = source.metadata
    decoded = sum(metadata.row_group(index).total_byte_size for index in range(metadata.num_row_groups))
    if metadata.num_rows > min(MAX_ROWS, manifest.resources.maximum_rows) or decoded > MAX_DECODED_BYTES:
        raise ValueError("Source exceeds profile decoded row/byte limit")
    if len(source.schema_arrow) > 64:
        raise ValueError("Source exceeds profile column limit")
    names = source.schema_arrow.names
    required = set(manifest.identity.observation_fields) | {"Duration", "Verdict"}
    if not required.issubset(names) or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate native profile columns")
    selected = required | ({"Variant"} if "Variant" in names else set())
    frame = pl.DataFrame(source.read())
    _native_column_types(frame, selected)
    return frame, tuple(names)


def _native_column_types(frame: pl.DataFrame, selected: set[str]) -> None:
    """Reject unsupported identity/quantity types without coercing decimal verdict labels."""
    import polars as pl

    for name in sorted(selected):
        dtype = frame.schema[name]
        allowed = dtype.is_integer() or dtype in {pl.String, pl.Null}
        if name in {"Duration", "Verdict"}:
            allowed = dtype.is_numeric() or dtype == pl.Null
        if not allowed:
            raise ValueError(f"Unsupported profile column type: {name}")


def profile_native_source(card: DataCard, path: str | Path) -> NativeSourceProfile:
    """Describe matching local native bytes, including blocked sources, without admission.

    Parameters
    ----------
    card : DataCard
        Versioned source card with explicit identity, byte digest/size, resource
        limits and declared duration units, or explicitly unverified raw units.
    path : str or pathlib.Path
        Trusted local Parquet snapshot matching the card. URLs are not fetched.

    Returns
    -------
    NativeSourceProfile
        Descriptive counts and inverse empirical quantiles, including null,
        nonfinite, negative and zero categories. No admission is granted.

    Raises
    ------
    ValueError
        Unsupported contract, changed source or exceeded resource/schema limits.
    ImportError
        Required Polars/PyArrow backends are absent; select the reviewed wheel's
        ``ci`` or ``analytics`` extra. Model construction/reading needs neither.
    OSError
        Local source acquisition fails. Parquet decoding errors propagate.

    Notes
    -----
    Reads and hashes bounded local bytes, then decodes the admitted table shape.
    Writes no files. Source order does not affect statistics; retries do not
    become independent observations. Variant identity and existing v1 profile
    bytes/meanings are retained even when the card groups by logical build.
    """
    import polars as pl

    card = DataCard.model_validate(card)
    _native_card(card)
    frame, columns = _source_frame(card, Path(path))
    identity = list(card.manifest.identity.observation_fields)
    identified = frame.drop_nulls(identity)
    unique_count = identified.unique(identity).height
    keys = ["BuildId"] + (["Variant"] if "Variant" in frame.columns else [])
    grouped = frame.drop_nulls(keys)
    counts = grouped.group_by(keys).len()["len"]
    conflicts = identified.group_by(identity).agg(pl.struct("Duration", "Verdict").n_unique().alias("values"))
    verdict = frame["Verdict"]
    if verdict.dtype == pl.Null:
        verdict = verdict.cast(pl.Float64)
    return NativeSourceProfile(
        schema_version=1,
        card=card,
        columns=columns,
        column_missingness=tuple(ColumnMissingness(name=name, null_count=frame[name].null_count()) for name in columns),
        rows=frame.height,
        missing_identity_rows=frame.height - identified.height,
        unique_observations=unique_count,
        duplicate_identity_rows=identified.height - unique_count,
        conflicting_identities=conflicts.filter(pl.col("values") > 1).height,
        logical_builds=grouped["BuildId"].n_unique(),
        execution_groups=len(counts),
        missing_group_rows=frame.height - grouped.height,
        failed_rows=int((verdict == 1).sum()),
        passed_rows=int((verdict == 0).sum()),
        null_verdict_rows=verdict.null_count(),
        invalid_verdict_rows=int((~verdict.is_in([0, 1])).sum()),
        raw_duration=_numbers(frame["Duration"], card.manifest.outcomes.duration or RAW_DURATION),
        tests_per_execution_group=_numbers(counts, GROUP_ROWS),
    )


def publish_source_profile(report: NativeSourceProfile, writer: ArtifactWriter) -> Path:
    """Atomically publish descriptive JSON once; identical concurrent replay is idempotent.

    Uses the existing artifact writer's local hard-link durability contract. An
    interrupted temporary file is not a completed profile; retry recomputes from
    the frozen source. Conflicting content at the same path raises ValueError.
    """
    report = NativeSourceProfile.model_validate(report)
    path = writer.run_root / "corpus-profiles" / f"{report.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, report.to_json() + "\n")
    return path


class CorpusProfileInventory(VersionedContract):
    """Frozen diversity matrix tying descriptive profiles to exact registered cards.

    Rows and execution groups cannot satisfy source admission or diversity gates.
    Existing card coverage declarations remain unchanged; this inventory certifies
    no regime and makes no cross-project independence or policy ranking claim.
    """

    registry: CorpusRegistry
    profiles: Annotated[tuple[NativeSourceProfile, ...], Field(max_length=64)]
    uncovered_required_regimes: tuple[str, ...]
    evidence_status: Literal["insufficient_diversity"] = "insufficient_diversity"

    @model_validator(mode="after")
    def _membership(self) -> Self:
        """Reject conflicting evidence and weakened diversity; collapse identical replay."""
        unique: dict[str, NativeSourceProfile] = {}
        for profile in self.profiles:
            source_id = profile.card.manifest.benchmark_id
            if self.registry.card(source_id) != profile.card:
                raise ValueError("Profile does not match registered card")
            previous = unique.setdefault(source_id, profile)
            if previous != profile:
                raise ValueError("Conflicting profiles for one registered source")
        if tuple(sorted(self.uncovered_required_regimes)) != self.registry.required_regimes:
            raise ValueError("Descriptive profiles cannot establish diversity coverage")
        object.__setattr__(self, "profiles", tuple(unique[key] for key in sorted(unique)))
        object.__setattr__(self, "uncovered_required_regimes", self.registry.required_regimes)
        return self


def publish_profile_inventory(inventory: CorpusProfileInventory, writer: ArtifactWriter) -> Path:
    """Publish the complete deterministic matrix atomically; retries reuse identical bytes."""
    inventory = CorpusProfileInventory.model_validate(inventory)
    path = writer.run_root / "corpus-profile-inventories" / f"{inventory.fingerprint.removeprefix('sha256:')}.json"
    writer.write_immutable(path, inventory.to_json() + "\n")
    return path
