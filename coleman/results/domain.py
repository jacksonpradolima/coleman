"""Immutable, transport-independent version-1 measurement contracts.

Units and meanings are opaque, case-sensitive identifiers: no conversion or
statistical inference is implicit. Native scheduling rows are a separate legacy
contract and cannot be interpreted as these records without explicit provenance.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from coleman.core.primitives import (
    Fingerprint as Fingerprint,
)
from coleman.core.primitives import (
    Finite as Finite,
)
from coleman.core.primitives import (
    Identifier as Identifier,
)
from coleman.core.primitives import (
    canonical_json as canonical_json,
)
from coleman.core.primitives import (
    decode_json as decode_json,
)
from coleman.core.primitives import (
    fingerprint as fingerprint,
)

# Names belonging to MonitorCollector's legacy scheduling contract. Extensions
# use their own measurement names, never impersonating a native output column.
RESERVED_METRICS = frozenset(
    [
        "scenario",
        "experiment",
        "step",
        "execution_id",
        "worker_id",
        "parallel_mode",
        "policy",
        "reward_function",
        "budget_mode",
        "budget_value",
        "total_build_duration",
        "prioritization_time",
        "process_memory_rss_mib",
        "process_memory_peak_rss_mib",
        "process_cpu_utilization_percent",
        "process_cpu_time_seconds",
        "wall_time_seconds",
        "detected",
        "missed",
        "tests_ran",
        "tests_not_ran",
        "ttf",
        "ttf_duration",
        "time_reduction",
        "fitness",
        "cost",
        "rewards",
        "avg_precision",
        "prioritization_order",
        "prioritization_order_hash",
        "prioritization_order_top_k",
        "variant",
    ]
)


class _Contract(BaseModel):  # pylint: disable=too-few-public-methods
    """Strict frozen base; use validated constructors, never ``model_construct``."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class Measurement(_Contract):  # pylint: disable=too-few-public-methods
    """Represent one signed observation with explicit units and meaning.

    Attributes
    ----------
    name : str
        Unique measurement name within its record; native metric names are reserved.
    value : float or None
        Required raw value. Observations are finite and may be zero or negative.
        Other statuses require explicit null; omitting a measurement is distinct.
    unit : str
        Case-sensitive unit identifier; no conversion is implicit.
    meaning : str
        Versioned semantic identifier specifying what was measured.
    direction : {"minimize", "maximize", "none"}
        Declared objective direction, without performing a comparison.
    status : {"observed", "unknown", "unavailable", "not_applicable"}
        Observation state. Only ``observed`` permits a numeric value.
    minimum : float or None, optional
        Optional finite lower bound in the same unit. No implicit bound is
        imposed on arbitrary custom units.

    Raises
    ------
    pydantic.ValidationError
        If required fields, finite values, reserved names or state/bound
        invariants are violated.

    Notes
    -----
    Frozen and strict. A learning reward is a separate transformed signal;
    this model preserves the raw observation and performs no I/O.
    """

    name: Identifier
    value: Finite | None
    unit: Identifier
    meaning: Identifier
    direction: Literal["minimize", "maximize", "none"]
    status: Literal["observed", "unknown", "unavailable", "not_applicable"]
    minimum: Finite | None = None

    @model_validator(mode="after")
    def _check_observation(self) -> Self:
        """Enforce availability, reserved-name and optional lower-bound invariants."""
        if self.name in RESERVED_METRICS:
            raise ValueError(f"Reserved native metric: {self.name}")
        if (self.status == "observed") != (self.value is not None):
            raise ValueError("Observed requires a value; other statuses require explicit null")
        if self.value is not None and self.minimum is not None and self.value < self.minimum:
            raise ValueError("Measurement is below its declared minimum")
        return self


class LearningReward(_Contract):  # pylint: disable=too-few-public-methods
    """Keep a finite transformed learning signal separate from raw outcomes.

    Attributes
    ----------
    value : float
        Finite transformed value; its sign does not change raw observations.
    unit : str
        Explicit output-scale identifier.
    transform_id : str
        Versioned identifier of the transformation that produced the signal.

    Notes
    -----
    Frozen and strict; no transformation, comparison or I/O is performed.
    Invalid or missing fields raise ``pydantic.ValidationError``.
    """

    value: Finite
    unit: Identifier
    transform_id: Identifier


class ResultProvenance(_Contract):  # pylint: disable=too-few-public-methods
    """Identify a producer and the configuration/evaluation it actually used.

    Attributes
    ----------
    producer : str
        Explicit producer identifier.
    producer_version : str
        Explicit producer version, not inferred from the running environment.
    spec_fingerprint : str
        Identified SHA-256 of the resolved specification used for the observation.
    evaluation_fingerprint : str
        Identified SHA-256 of the declared metric/protocol/data interpretation.
    source_revision : str or None, optional
        Explicit source revision; absence remains absence.

    Notes
    -----
    Frozen metadata with no environment collection or I/O. Fingerprints record
    provenance, not proof of independent observations or scientific admission.
    Invalid fields raise ``pydantic.ValidationError``.
    """

    producer: Identifier
    producer_version: Identifier
    spec_fingerprint: Fingerprint
    evaluation_fingerprint: Fingerprint
    source_revision: Identifier | None = None


class DomainRecord(_Contract):
    """Bundle immutable versioned measurements for one logical decision.

    Attributes
    ----------
    schema_version : int
        Exact supported integer version, currently 1; unknown versions reject.
    namespace : str
        Extension namespace; ``coleman`` and its dotted descendants are reserved.
    run_id : str
        Stable logical run identity.
    execution_id : str
        Logical execution identity preserved on resume, not a worker/retry ID.
    decision_id : str
        Decision identity distinguishing relevant policy/reward/build/variant keys.
    data_fingerprint : str
        Identified SHA-256 of the interpreted input data.
    available_at : datetime
        Aware outcome-availability timestamp, normalized to UTC. This is not
        automatically the application's ingestion/first-receipt time.
    provenance : ResultProvenance
        Producer, resolved specification and evaluation identity.
    measurements : tuple[Measurement, ...]
        Named observations, sorted by name after validation; duplicates reject.
    learning_reward : LearningReward or None, optional
        Separate transformed signal, without replacing raw measurements.

    Raises
    ------
    pydantic.ValidationError
        If fields, version, timestamps, namespace or measurement invariants fail.

    Notes
    -----
    The stable logical key is namespace/run/execution/decision, excluding schema
    version and contents. Value changes alter the content digest, not that key.
    Legacy native rows require their existing reader; no migration invents
    missing units, availability or provenance. Construction performs no I/O.
    """

    schema_version: Annotated[int, Field(strict=True)]
    namespace: Identifier
    run_id: Identifier
    execution_id: Identifier
    decision_id: Identifier
    data_fingerprint: Fingerprint
    available_at: AwareDatetime
    provenance: ResultProvenance
    measurements: tuple[Measurement, ...]
    learning_reward: LearningReward | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, value: int) -> int:
        """Fail closed for unsupported versions, including older unversioned data."""
        if value != 1:
            raise ValueError(f"Unsupported domain schema version: {value}")
        return value

    @field_validator("available_at", mode="before")
    @classmethod
    def _explicit_timestamp(cls, value: Any) -> Any:
        """Reject epoch numbers whose scale/timezone would otherwise be inferred."""
        if isinstance(value, str):
            # ISO parsing excludes Pydantic's numeric-string epoch inference.
            # AwareDatetime subsequently rejects date-only/naive ISO datetimes.
            return datetime.fromisoformat(value)
        if not isinstance(value, datetime):
            raise ValueError("Availability must be an explicitly zoned timestamp")
        return value

    @field_validator("available_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        """Normalize explicit offsets to UTC without inferring a timezone."""
        return value.astimezone(UTC)

    @field_validator("measurements")
    @classmethod
    def _sort_measurements(cls, value: tuple[Measurement, ...]) -> tuple[Measurement, ...]:
        """Reject ambiguous bundles and canonicalize declaration order."""
        names = [item.name for item in value]
        if not names or len(names) != len(set(names)):
            raise ValueError("Measurements must be nonempty with unique names")
        return tuple(sorted(value, key=lambda item: item.name))

    @field_validator("namespace")
    @classmethod
    def _check_namespace(cls, value: str) -> str:
        """Reserve the core namespace to prevent impersonating native records."""
        if value == "coleman" or value.startswith("coleman."):
            raise ValueError("Reserved native namespace")
        return value

    @property
    def logical_key(self) -> tuple[str, str, str, str]:
        """Return a retry-stable key; content conflicts at this key are errors."""
        return self.namespace, self.run_id, self.execution_id, self.decision_id

    @property
    def identity_digest(self) -> str:
        """Return SHA-256 of the logical key, independent of measurement values."""
        return fingerprint(list(self.logical_key))

    @property
    def digest(self) -> str:
        """Return identified SHA-256 of canonical serialized record content."""
        return "sha256:" + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Serialize v1 deterministically; absent optional fields stay absent.

        Explicit optional nulls remain null. The wire format uses an array of
        named measurements; tuples prevent mutable dictionaries in core APIs.
        Raises validation errors for non-JSON/non-finite values.
        """
        validated = type(self).model_validate(self)
        return canonical_json(validated.model_dump(mode="json", exclude_unset=True))

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Read and validate a v1 record without migrating legacy native rows.

        Parameters
        ----------
        payload : str or bytes
            Complete JSON document. Its acquisition layer owns byte/depth limits.

        Returns
        -------
        DomainRecord
            Validated frozen record with normalized timestamp and measurement order.

        Raises
        ------
        ValueError
            If JSON syntax, duplicate keys, root shape, fields or schema version
            are invalid. Pydantic validation errors are included in this category.

        Notes
        -----
        Performs no I/O and invents no missing unit, timestamp or provenance.
        Unknown versions and unknown fields reject rather than receiving defaults.
        """
        raw = decode_json(payload)
        if not isinstance(raw, dict):
            raise ValueError("Domain record must be a JSON object")
        return cls.model_validate(raw)

    def measurement(self, name: str) -> Measurement:
        """Look up one declared observation without inventing missing values.

        Parameters
        ----------
        name : str
            Exact case-sensitive declared measurement name.

        Returns
        -------
        Measurement
            Matching frozen measurement, including its explicit status and units.

        Raises
        ------
        KeyError
            If the measurement was omitted from the record.
        """
        for item in self.measurements:
            if item.name == name:
                return item
        raise KeyError(name)

    def require_comparable(self, other: DomainRecord, name: str) -> None:
        """Require compatible metadata and observed values without inference.

        Parameters
        ----------
        other : DomainRecord
            Record to compare against the current record's declared evaluation.
        name : str
            Exact measurement name required in both records.

        Raises
        ------
        KeyError
            If either record omits the measurement.
        ValueError
            If metadata differs or either value is unknown, unavailable or not applicable.

        Notes
        -----
        Returns ``None`` on success and mutates neither record. Performs no unit
        conversion, ranking, probability estimation or statistical test. Use
        ``require_compatible`` to validate metadata while reporting incomplete evidence.
        """
        self.require_compatible(other, name)
        if self.measurement(name).status != "observed" or other.measurement(name).status != "observed":
            raise ValueError("Comparison requires observed values")

    def require_compatible(self, other: DomainRecord, name: str) -> None:
        """Validate measurement and evaluation metadata without comparing values.

        Parameters
        ----------
        other : DomainRecord
            Record whose schema, namespace, data and evaluation must agree.
        name : str
            Exact measurement name required in both records.

        Raises
        ------
        KeyError
            If either record omits the named measurement.
        ValueError
            If units, meanings, directions, lower bounds, data, schema, namespace
            or evaluation identities are incompatible.

        Notes
        -----
        Returns ``None`` on success and mutates neither record. Missing values are
        allowed. Producers/specifications may differ within the same declared
        evaluation; compatibility does not establish statistical comparability.
        """
        left, right = self.measurement(name), other.measurement(name)

        def metadata(item: Measurement) -> tuple[str, str, str, float | None]:
            """Extract explicit semantics, without coercing or comparing values."""
            return item.unit, item.meaning, item.direction, item.minimum

        if (
            self.schema_version != other.schema_version
            or self.namespace != other.namespace
            or self.data_fingerprint != other.data_fingerprint
            or self.provenance.evaluation_fingerprint != other.provenance.evaluation_fingerprint
            or metadata(left) != metadata(right)
        ):
            raise ValueError("Incompatible measurement comparison")
