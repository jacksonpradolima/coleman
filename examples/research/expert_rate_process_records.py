"""Versioned process, selection and assessment records for the finite public study."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from examples.research.expert_rate_records import Artifact, Count, Family, FamilyHeader, Record, Slot

SelectionSlots = Annotated[tuple[tuple[Family, Slot], ...], Field(min_length=5, max_length=5)]


def _selection_slots(slots: SelectionSlots) -> None:
    """Retain every original family and its actual declared configuration slot."""
    order = ("fixed-uniform", "constant", "anytime", "fixed-share", "periodic")
    if tuple(f for f, _ in slots) != order:
        raise ValueError("Selection must retain all original families in their declared tie order")
    if any(slot != 0 for f, slot in slots if f == "fixed-uniform"):
        raise ValueError("The untuned fixed control has only its original zero slot")


class Attempt(Record):
    """Retain one physical slot without releasing unknown killed-child work.

    Attributes
    ----------
    reserved_calls : int
        Entire original family allowance reserved before launching; zero only
        when not started. It is not the number of physically executed calls.
    attempted_calls : int or None
        Exact returned family count, or unavailable after uncertain termination.
        An observed zero is distinct from null and from not-started disposition.
    wall_seconds, cpu_seconds : float or None
        Complete parent-observed child lifetime and isolated child CPU delta in
        seconds. Null represents unavailable/unstarted observations, never zero.
    """

    family: Family
    repetition: Annotated[int, Field(strict=True, ge=0, le=2)]
    status: Literal["completed", "numerical_failure", "resource_limited", "not_started"]
    reason: str | None
    reserved_calls: Count
    attempted_calls: Count | None
    wall_seconds: Binary64 | None
    cpu_seconds: Binary64 | None
    artifact: Artifact | None

    @model_validator(mode="after")
    def _work(self) -> Self:
        if self.attempted_calls is not None and self.attempted_calls > self.reserved_calls:
            raise ValueError("Known attempts cannot exceed the original reserved family workload")
        if any(v is not None and v < 0 for v in (self.wall_seconds, self.cpu_seconds)):
            raise ValueError("Physical elapsed work cannot be negative")
        return self

    @model_validator(mode="after")
    def _status(self) -> Self:
        """Keep unstarted/unknown physical work separate from completed evidence."""
        if self.status == "not_started":
            if self.reserved_calls or any(
                v is not None for v in (self.attempted_calls, self.wall_seconds, self.cpu_seconds, self.artifact)
            ):
                raise ValueError("Unstarted child has no acquired physical observations or reservation")
        elif self.wall_seconds is None:
            raise ValueError("Every launched child requires actual complete process wall time")
        if self.status == "completed" and (self.artifact is None or self.attempted_calls != self.reserved_calls):
            raise ValueError("Completion requires the entire reserved work and its checked family artifact")
        if self.status == "completed" and self.wall_seconds is not None and self.wall_seconds > 180:
            raise ValueError("Completed child cannot exceed the original180-second ceiling")
        if (self.reason is None) != (self.status == "completed"):
            raise ValueError("Every noncompleted attempt requires its actual disposition")
        return self


class DevelopmentChoice(Record):
    """Freeze all five family winners before any evaluation-field inspection."""

    protocol_content: Fingerprint
    selected_family: Family
    selected_slots: SelectionSlots
    development_artifacts: Annotated[tuple[Artifact, ...], Field(min_length=5, max_length=5)]

    @model_validator(mode="after")
    def _families(self) -> Self:
        _selection_slots(self.selected_slots)
        return self


class ChildSummary(FamilyHeader):
    """Process bookkeeping has no evaluation metrics and does not certify efficacy."""

    attempted: Count
    source_files: tuple[tuple[str, str], ...]
    artifact: Artifact
    development_artifact: Artifact | None


class UnavailableChoice(Record):
    """Freeze refusal of selection before inspecting any evaluation fields."""

    protocol_content: Fingerprint
    status: Literal["unavailable"] = "unavailable"
    reason: Literal["missing-original-development-evidence"] = "missing-original-development-evidence"


class Gate(Record):
    """Finite descriptive entry decision, never a protected/unseen efficacy claim."""

    selected_family: Family
    selected_slots: SelectionSlots
    shape_excess: tuple[tuple[str, Binary64], ...]
    threshold: Annotated[Binary64, Field(ge=0.015625, le=0.015625)] = 1 / 64
    decision: Literal["retain_baselines", "formulation_review_supported"]
    independent_empirical_units: Annotated[int, Field(strict=True, ge=0, le=0)] = 0
    exposure: Literal["public_authored_evaluation_exposed"] = "public_authored_evaluation_exposed"

    @model_validator(mode="after")
    def _decision(self) -> Self:
        """Refuse a promoted decision inconsistent with all four original rate shapes."""
        _selection_slots(self.selected_slots)
        shapes = ("stationary-gap", "stationary-small-gap", "neutral-prefix", "contrast-pulses")
        if tuple(shape for shape, _ in self.shape_excess) != shapes:
            raise ValueError("Entry decision requires every distinct original primary rate shape")
        supported = sum(value > self.threshold for _, value in self.shape_excess) >= 2
        if (self.decision == "formulation_review_supported") != supported:
            raise ValueError("Entry decision differs from the declared descriptive shape threshold")
        return self


class OracleResult(Record):
    """Independent fixed-envelope arithmetic and exact three-repetition identity checks."""

    protocol_content: Fingerprint
    checked_steps: Annotated[int, Field(strict=True, ge=0, le=77112)]
    maximum_coefficient_error: Annotated[Binary64, Field(ge=0)]
    maximum_score_error: Annotated[Binary64, Field(ge=0)]
    repetitions_equal: Annotated[bool, Field(strict=True)]
    gate: Gate

    @model_validator(mode="after")
    def _equal(self) -> Self:
        """A successful arithmetic artifact cannot hide a repetition mismatch."""
        if not self.repetitions_equal:
            raise ValueError("Successful oracle result requires identical complete repeated mathematics")
        return self


class StudyReport(Record):
    """Retain the original fifteen slots and an honest complete/limited outcome.

    Attributes
    ----------
    reserved_calls, known_attempted_calls, unknown_attempts : int
        Declared spent allowance, lower bound from returned family counts, and
        number of launched children with unknown counts. They are different
        quantities; unknown calls cannot free reservation or become exact zero.
    pass_wall_seconds, baseline_wall_seconds : tuple of float, float
        Actual full parent windows in seconds, including validation and report
        overhead. Completion requires the original 900/2700-second ceilings.
    oracle_wall_seconds : float or None
        Separate actual oracle process window in seconds, unavailable if not run.
    gate : Gate or None
        Descriptive exposed-source decision only after all original evidence is
        checked. Limited evidence cannot claim an advanced method is warranted.
    """

    protocol_content: Fingerprint
    status: Literal["completed", "resource_limited", "numerical_failure"]
    reason: str | None
    runtime: InstalledRuntime
    driver_runtime: SelectedInstalledRuntime
    source_files: tuple[tuple[str, str], ...]
    attempts: Annotated[tuple[Attempt, ...], Field(min_length=15, max_length=15)]
    reserved_calls: Count
    known_attempted_calls: Count
    unknown_attempts: Annotated[int, Field(strict=True, ge=0, le=15)]
    pass_wall_seconds: tuple[Binary64, Binary64, Binary64]
    baseline_wall_seconds: Binary64
    oracle: Artifact | None
    oracle_wall_seconds: Annotated[Binary64, Field(ge=0)] | None
    development_choice: Artifact | None
    gate: Gate | None

    @model_validator(mode="after")
    def _timings(self) -> Self:
        """Retain actual parent overhead and refuse completed work past its caps."""
        if any(value < 0 for value in self.pass_wall_seconds) or self.baseline_wall_seconds < 0:
            raise ValueError("Actual pass/baseline elapsed observations cannot be negative")
        if self.status == "completed" and (max(self.pass_wall_seconds) > 900 or self.baseline_wall_seconds > 2700):
            raise ValueError("Completed study cannot exceed the original pass or baseline wall ceilings")
        if self.status == "completed" and self.oracle_wall_seconds is not None and self.oracle_wall_seconds > 180:
            raise ValueError("Completed oracle cannot exceed the original180-second ceiling")
        return self

    @model_validator(mode="after")
    def _complete(self) -> Self:
        self.driver_runtime.require_base(self.runtime, extras=("cli",), pins=(("click", "8.1.8"),))
        order = ("constant", "anytime", "fixed-share", "periodic", "fixed-uniform")
        expected = tuple((r, f) for r in range(3) for f in order[r:] + order[:r])
        if tuple((a.repetition, a.family) for a in self.attempts) != expected:
            raise ValueError("All fifteen original physical slots must retain their declared execution order")
        return self

    @model_validator(mode="after")
    def _work(self) -> Self:
        """Reserve all declared calls, retaining unknown physical outcomes."""
        for attempt in self.attempts:
            expected_calls = (
                0 if attempt.status == "not_started" else 1512 * (1 if attempt.family == "fixed-uniform" else 4)
            )
            if attempt.reserved_calls != expected_calls:
                raise ValueError("Each launched family must reserve its entire original declared workload")
        return self

    @model_validator(mode="after")
    def _work_totals(self) -> Self:
        """Separate known actual work from unavailable launched-child counts."""
        if self.reserved_calls != sum(a.reserved_calls for a in self.attempts):
            raise ValueError("Reserved calls must include unknown post-timeout work")
        if self.known_attempted_calls != sum(a.attempted_calls or 0 for a in self.attempts):
            raise ValueError("Known work cannot hide original reported attempts")
        if self.unknown_attempts != sum(a.status != "not_started" and a.attempted_calls is None for a in self.attempts):
            raise ValueError("Unknown launched work must remain explicitly unavailable")
        return self

    @model_validator(mode="after")
    def _completion(self) -> Self:
        """Require the full original work, frozen selection and separate oracle."""
        if self.status == "completed":
            if self.reason is not None or any(a.status != "completed" for a in self.attempts):
                raise ValueError("Complete study requires all original physical slots")
            if (
                self.reserved_calls != 77112
                or self.oracle is None
                or self.gate is None
                or self.oracle_wall_seconds is None
                or self.development_choice is None
            ):
                raise ValueError("Complete study requires all original work and the independent checked oracle/gate")
        elif self.reason is None or self.gate is not None:
            raise ValueError("Limited study needs its reason and cannot claim a completed scientific entry gate")
        return self
