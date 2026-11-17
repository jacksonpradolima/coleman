"""Finite, versioned operational observations without an exporter or host calls.

These observations describe admitted facts. They neither complete a job nor
change a learner, acknowledge state, infer a missing label, or certify a cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint
from coleman.core.temporal import Timestamp

MetricName = Literal[
    "decisions",
    "updates",
    "pending",
    "pending_age",
    "update_age",
    "missing_feedback",
    "concentration",
    "constraint",
    "fallback",
    "rss",
    "peak_rss",
    "cpu_time",
    "wall_time",
    "cpu_utilization",
    "task_progress",
    "attempts",
    "invocations",
    "output_usage",
    "queue_usage",
]
MetricStatus = Literal["observed", "unknown", "unavailable", "not_applicable"]
MetricUnit = Literal["1", "s", "MiB", "%"]
Route = Literal["direct", "replay", "task", "native"]
Outcome = Literal["accepted", "pending", "infeasible", "unsupported", "fallback", "failed", "none"]


@dataclass(frozen=True)
class MetricDefinition:
    """Specify one v1 metric's units, owner and operational response.

    Attributes
    ----------
    name, unit : str
        Closed metric name and unit; ``1`` is a count or dimensionless ratio.
    interpretation, owner, response : str
        Population/clock meaning, responsible operator and suggested response.
    count : bool, optional
        Whether observed values must be integral. No value may be negative.
    maximum : float or None, optional
        Optional upper bound; absent means no invented physical upper bound.
    """

    name: MetricName
    unit: MetricUnit
    interpretation: str
    owner: str
    response: str
    count: bool = False
    maximum: float | None = None


_DEFINITIONS = (
    MetricDefinition(
        "decisions", "1", "Successful logical decisions, not physical work", "host", "Inspect progress", True
    ),
    MetricDefinition(
        "updates", "1", "Accepted completed updates, not independent observations", "host", "Inspect delivery", True
    ),
    MetricDefinition(
        "pending", "1", "Emitted decisions without an accepted update", "host", "Inspect label delivery", True
    ),
    MetricDefinition(
        "pending_age",
        "s",
        "Age since oldest pending decision at explicit observation clock",
        "source owner",
        "Check source clocks and delivery",
    ),
    MetricDefinition(
        "update_age",
        "s",
        "Age since latest accepted update at explicit observation clock",
        "host",
        "Inspect cadence and delivery; do not invent an update",
    ),
    MetricDefinition(
        "missing_feedback",
        "1",
        "Pending decisions with no admitted feedback revision",
        "source owner",
        "Inspect receipt; missing does not imply failure",
        True,
    ),
    MetricDefinition(
        "concentration",
        "1",
        "Sum of squared emitted mixture coefficients; not risk or calibrated probability",
        "learner owner",
        "Inspect diversification within the declared coefficient space",
        maximum=1.000000000000004,
    ),
    MetricDefinition(
        "constraint",
        "1",
        "One declared constraint outcome in this snapshot",
        "adapter owner",
        "Inspect feasibility and solver diagnostics",
        True,
        1,
    ),
    MetricDefinition(
        "fallback",
        "1",
        "One explicitly declared fallback in this snapshot",
        "host",
        "Inspect fallback reason before adoption",
        True,
        1,
    ),
    MetricDefinition(
        "rss",
        "MiB",
        "Sampled current executable RSS, excluding unobserved children",
        "operator",
        "Compare with the existing worker memory guard",
    ),
    MetricDefinition(
        "peak_rss",
        "MiB",
        "Current executable lifetime high-water RSS",
        "operator",
        "Inspect worker reuse and peak attribution",
    ),
    MetricDefinition(
        "cpu_time",
        "s",
        "Current executable CPU seconds since tracker creation",
        "operator",
        "Compare matched workloads; do not sum cumulative samples",
    ),
    MetricDefinition(
        "wall_time",
        "s",
        "Monotonic seconds since tracker creation",
        "operator",
        "Inspect the existing attempt deadline",
    ),
    MetricDefinition(
        "cpu_utilization",
        "%",
        "Process CPU / wall interval; may exceed100 with threads",
        "operator",
        "Inspect native thread/process concurrency",
    ),
    MetricDefinition(
        "task_progress",
        "1",
        "Completed / declared task coordinates",
        "operator",
        "Inspect incomplete coordinates, not scientific quality",
        maximum=1,
    ),
    MetricDefinition(
        "attempts",
        "1",
        "Actual physical task attempts retained by the job",
        "operator",
        "Inspect retry budget and original receipts",
        True,
    ),
    MetricDefinition(
        "invocations", "1", "Actual worker starts retained by the job", "operator", "Inspect invocation cap", True
    ),
    MetricDefinition(
        "output_usage",
        "1",
        "Observed output bytes / declared byte cap",
        "operator",
        "Use existing fail-closed output guard; ratio may exceed1",
    ),
    MetricDefinition(
        "queue_usage",
        "1",
        "Observed queue depth / explicitly declared capacity",
        "operator",
        "Stop admission at the existing queue cap; ratio may exceed1",
    ),
)
_BY_NAME = {item.name: item for item in _DEFINITIONS}


def metric_catalog() -> tuple[MetricDefinition, ...]:
    """Return the immutable closed v1 catalog in canonical name order.

    Returns
    -------
    tuple of MetricDefinition
        Nineteen definitions. No plugins, runtime metadata, I/O or providers load.
    """
    return tuple(sorted(_DEFINITIONS, key=lambda item: item.name))


class MetricPoint(CoreImmutable):
    """Keep units and missingness explicit for one instantaneous observation.

    Attributes
    ----------
    name, unit : str
        Exact catalog coordinate and its required unit. No conversion occurs.
    status : {'observed', 'unknown', 'unavailable', 'not_applicable'}
        Observed requires a finite nonnegative value, including zero. All other
        statuses require null. Missing feedback is never recoded as failure.
    value : int or float or None
        Nonnegative signed-int64 or binary64 diagnostic, not paid accounting.
        Counts require integer input/tokens and remain exact beyond2**53;
        floating/decimal count tokens reject instead of being rounded.

    Raises
    ------
    ValueError
        Unsupported name/unit/status, nonfinite, negative or out-of-range value.
    """

    name: MetricName
    unit: MetricUnit
    status: MetricStatus
    value: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)] | Binary64 | None

    @model_validator(mode="after")
    def _semantics(self) -> Self:
        """Reject ambiguous missingness and incompatible catalog coordinates."""
        spec = _BY_NAME[self.name]
        if self.unit != spec.unit or (self.status == "observed") != (self.value is not None):
            raise ValueError("Metric units and observed/null status must match the v1 catalog")
        if self.value is None:
            return self
        if (
            self.value < 0
            or (spec.count and not isinstance(self.value, int))
            or (spec.maximum is not None and self.value > spec.maximum)
        ):
            raise ValueError("Metric value violates its nonnegative count/ratio envelope")
        return self


class LifecycleSnapshot(CoreContract):
    """Retain a deterministic v1 host observation with bounded metric dimensions.

    Attributes
    ----------
    route, outcome : str
        Closed route/outcome labels. They are observations, not job transitions.
    observed_at : datetime
        Caller-supplied zoned clock; no wall clock is read by construction.
    source_fingerprint : str
        Digest of the exact inspected typed source, not authentication.
    algorithm, state_codec : Reference or None
        Exact versions present in the inspected view. Null means that the view
        does not supply that identity, not proof of statelessness. These never become metric
        labels; arbitrary producer IDs would create unbounded cardinality.
    points : tuple of MetricPoint
        One to19 unique coordinates, canonicalized by name.

    Notes
    -----
    V1 rejects unknown versions and fields; no automatic migration. Reading
    this observation does not qualify resuming its source checkpoint. No RNG,
    counters, buffers, side effects or delivery guarantees are owned here.
    """

    route: Route
    outcome: Outcome
    observed_at: Timestamp
    source_fingerprint: Fingerprint
    algorithm: Reference | None
    state_codec: Reference | None
    points: Annotated[tuple[MetricPoint, ...], Field(min_length=1, max_length=19)]

    @model_validator(mode="after")
    def _coordinates(self) -> Self:
        """Keep one complete finite observation for each named coordinate."""
        if len({item.name for item in self.points}) != len(self.points):
            raise ValueError("A lifecycle snapshot cannot repeat a metric coordinate")
        object.__setattr__(self, "points", tuple(sorted(self.points, key=lambda item: item.name)))
        return self


def point(name: MetricName, value: float | int | None, *, missing: MetricStatus = "unavailable") -> MetricPoint:
    """Build a catalog point without treating an absent reading as zero.

    Parameters
    ----------
    name : str
        Closed v1 catalog coordinate, whose units are supplied explicitly.
    value : float or int or None
        Actual diagnostic, or null. No clipping or physical measurement occurs.
    missing : str, optional
        Null interpretation, default unavailable; observed with null rejects.

    Returns
    -------
    MetricPoint
        Revalidated immutable observation. Observed zero remains observed.

    Raises
    ------
    ValueError, KeyError
        Unsupported name, units/status or numeric value. No I/O occurs.
    """
    return MetricPoint(
        name=name, unit=_BY_NAME[name].unit, status=missing if value is None else "observed", value=value
    )
