"""Typed finite inputs/results for one pinned public research protocol.

These contracts add no learner, scheduler, source admission or empirical
inference. A checkpoint records readable complete bytes, not a durable ACK.
Runtime compatibility and the selected actual owner still govern restoration.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from pydantic import Field, model_validator

# Raw protocol headers require builtin integers, never bool/coerced versions.
# pylint: disable=unidiomatic-typecheck
from coleman.core._wire import BoundedContract
from coleman.core.base import CoreImmutable
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, validate_simplex
from coleman.core.primitives import Fingerprint, fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.results.domain import Measurement
from coleman.spec.runtime_identity import InstalledRuntime
from coleman.spec.selected_runtime import SelectedInstalledRuntime
from coleman.state.components import ComponentCodec
from coleman.state.contracts import StateBinding

LEGACY_PROTOCOL_SHA256 = "b826488634a5fba7302b2575ed1bc4166982cde4785984eabc80b3fb567b1639"
PROTOCOL_SHA256 = "7f7a0c739b8275b093ef10227a7cca832508f5cca7650d9fe7c1ed078826c0bb"
Family = Literal["constant", "anytime", "fixed-share", "periodic", "fixed-uniform"]
Partition = Literal["development", "evaluation-exposed"]
Status = Literal["completed", "numerical_failure", "resource_limited", "not_started"]
Vector = tuple[Binary64, Binary64, Binary64]
Count = Annotated[int, Field(strict=True, ge=0, le=77112)]
Slot = Annotated[int, Field(strict=True, ge=0, le=3)]


METRICS = (
    "mixture_loss",
    "mean_mixture_loss",
    "informative_mean_loss",
    "fixed_comparator_excess",
    "one_switch_comparator_excess",
    "coefficient_L1",
)
REFERENCE_PROFILES = {
    "constant": "hedge-constant",
    "anytime": "hedge-anytime-cumulative",
    "fixed-share": "fixed-share-excluding-origin",
}


class Record(BoundedContract):
    """Retain version1 artifacts within16MiB/depth64 without any migration."""

    _maximum_bytes = 16 * 1024**2


def read_artifact(path: Path) -> bytes:
    """Read one actual artifact within the original16MiB acquisition envelope.

    Parameters
    ----------
    path : pathlib.Path
        Trusted selected local artifact, not an import or executable recipe.

    Returns
    -------
    bytes
        Complete bounded original bytes, without claiming type or authenticity.

    Raises
    ------
    ValueError, OSError
        Empty/oversized or unreadable input. Typed readers still validate it.
    """
    with path.open("rb") as stream:
        content = stream.read(16 * 1024**2 + 1)
    if not content or len(content) > 16 * 1024**2:
        raise ValueError("Artifact exceeds its original nonempty16MiB envelope")
    return content


class Case(CoreImmutable):
    """One authored role/path; related paths are not independent observations.

    Attributes
    ----------
    case_id, partition, shape : str
        Original frozen identity, role and authored construction.
    informative_start : int
        Zero-based start of the scored informative suffix.
    panels : tuple
        Complete once-converted binary64 losses, in[0,1], at most64rows.
    """

    case_id: str
    partition: Partition
    shape: str
    informative_start: Annotated[int, Field(strict=True, ge=0, le=32)]
    panels: Annotated[tuple[Vector, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        if self.informative_start >= len(self.panels) or any(not 0 <= x <= 1 for row in self.panels for x in row):
            raise ValueError("Case requires complete bounded losses and a nonempty informative suffix")
        return self


class Setting(CoreImmutable):
    """One declared trial; null means the parameter is inapplicable.

    Attributes
    ----------
    family, slot : str, int
        Original family and zero-based setting order, not a new method.
    eta, multiplier, alpha : float or None
        Frozen rate, anytime multiplier and excluding-origin share respectively.
    window : int or None
        Trailing accepted panels for the simple periodic host baseline.
    """

    family: Family
    slot: Slot
    eta: Binary64 | None = None
    multiplier: Binary64 | None = None
    alpha: Binary64 | None = None
    window: Annotated[int, Field(strict=True, ge=8, le=64)] | None = None

    @model_validator(mode="after")
    def _parameters(self) -> Self:
        values = (self.eta is not None, self.multiplier is not None, self.alpha is not None, self.window is not None)
        expected = {
            "constant": (True, False, False, False),
            "anytime": (False, True, False, False),
            "fixed-share": (True, False, True, False),
            "periodic": (False, False, False, True),
            "fixed-uniform": (False, False, False, False),
        }
        if values != expected[self.family]:
            raise ValueError("Setting parameters must match its declared family")
        return self


def _treatment_key(setting: Setting, case_id: str, horizon: int):
    """Bind complete canonical treatment bytes, preserving signed binary64 zero."""
    checked = Setting.model_validate(setting)
    return fingerprint(checked.model_dump(mode="json")), case_id, horizon


def _original_grid(protocol: StudyInput, family: Family, partition: Partition | None = None):
    """Derive one complete declared source grid within an already admitted call."""
    return tuple(
        _treatment_key(s, c.case_id, len(c.panels))
        for s in protocol.settings
        if s.family == family
        for c in protocol.cases
        if partition is None or c.partition == partition
    )


class StudyInput(Record):
    """Parsed typed view of exactly the locally frozen immutable protocol.

    Attributes
    ----------
    protocol_sha256, protocol_content, source_catalogue : str
        Actual complete file digest, declared content digest and case-catalogue
        digest. These attest consistency, not authenticity or unseen outcomes.
    source_revision : str
        Frozen public baseline revision, separately from actual runtime identity.
    source_files : tuple
        Declared baseline source-path/SHA pairs; reading them performs no import.
    cases, settings : tuple
        All48 authored cases and17 declared configurations, without omission.
    """

    protocol_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    protocol_content: Fingerprint
    source_catalogue: Fingerprint
    source_revision: str
    source_files: tuple[tuple[str, str], ...]
    cases: Annotated[tuple[Case, ...], Field(min_length=48, max_length=48)]
    settings: Annotated[tuple[Setting, ...], Field(min_length=17, max_length=17)]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if self.protocol_sha256 not in (PROTOCOL_SHA256, LEGACY_PROTOCOL_SHA256):
            raise ValueError("Unknown public study protocol bytes")
        if len({c.case_id for c in self.cases}) != 48 or len({(s.family, s.slot) for s in self.settings}) != 17:
            raise ValueError("The frozen study requires every unique declared case and setting")
        return self


def load_protocol(path: Path) -> StudyInput:
    """Read only the bounded hash-pinned authored protocol, never arbitrary settings.

    Parameters
    ----------
    path : pathlib.Path
        Trusted local copy of the exact locally frozen authored protocol, at most128KiB.

    Returns
    -------
    StudyInput
        Typed complete cases/settings with one rational-to-binary64 conversion.

    Raises
    ------
    ValueError, OSError
        Missing, oversized, changed, unknown-version or inconsistent protocol.

    Notes
    -----
    The raw SHA is deliberately pinned before execution. An amendment needs a
    new reviewed protocol/version, not an outcome-dependent CLI option. Only
    this acquisition function uses the JSON mapping; the study uses typed DTOs.
    """
    with path.open("rb") as stream:
        content = stream.read(128 * 1024 + 1)
    if len(content) > 128 * 1024 or hashlib.sha256(content).hexdigest() not in (
        PROTOCOL_SHA256,
        LEGACY_PROTOCOL_SHA256,
    ):
        raise ValueError("The study requires its exact bounded locally frozen authored protocol")
    data = json.loads(content)
    expected_version = 2 if hashlib.sha256(content).hexdigest() == PROTOCOL_SHA256 else 1
    if type(data["schema_version"]) is not int or data["schema_version"] != expected_version:
        raise ValueError("Unsupported protocol schema version")
    cases = tuple(
        Case(
            case_id=row["id"],
            partition=row["partition"],
            shape=row["shape"],
            informative_start=row["informative_start"],
            panels=tuple(cast(Vector, tuple(float(Fraction(x)) for x in panel)) for panel in row["panels"]),
        )
        for row in data["cases"]
    )
    settings = tuple(
        Setting(
            family=family["id"],
            slot=slot,
            eta=None if "eta" not in values else float(Fraction(values["eta"])),
            multiplier=None if "anytime_multiplier" not in values else float(Fraction(values["anytime_multiplier"])),
            alpha=None if "alpha" not in values else float(Fraction(values["alpha"])),
            window=values.get("trailing_window"),
        )
        for family in data["families"]
        for slot, values in enumerate(family["configurations"])
    )
    return StudyInput(
        schema_version=1,
        protocol_sha256=hashlib.sha256(content).hexdigest(),
        protocol_content=data["content_digest"],
        source_catalogue=data["source"]["all_panels_digest"],
        source_revision=data["source_commit"],
        source_files=tuple(sorted(data["reviewed_source_sha256"].items())),
        cases=cases,
        settings=settings,
    )


@cache
def _declared_protocol_bytes() -> bytes:
    """Retain immutable declared source bytes, never an input validation result."""
    source = load_protocol(Path(__file__).with_name("expert-rate-entry-protocol-v2.json"))
    return source.to_json().encode("utf-8")


def admit_protocol(value: StudyInput) -> StudyInput:
    """Validate the original typed view against complete pinned source content.

    Parameters
    ----------
    value : StudyInput
        Complete original caller-owned DTO, not a hash label or witness flag.

    Returns
    -------
    StudyInput
        Fresh fully checked content for this one operation's private helpers.

    Raises
    ------
    ValueError, TypeError, OSError
        Any original field/type/version changed, source acquisition fails, or
        the selected public codec cannot reverse the complete typed input.

    Notes
    -----
    The private cache contains only immutable bytes of the publicly fixed
    protocol, derived from its verified raw SHA. No caller validation result,
    state snapshot, acknowledgement or cross-call permission is cached.
    The original input still passes the real full lossless codec on every call.
    """
    identity = "sha256:" + PROTOCOL_SHA256
    binding = StateBinding(
        owner=Reference(id="entry-protocol-input", version="1"),
        implementation=DefinitionBinding(
            reference=Reference(id="entry-pinned-input", version="1"), definition_fingerprint=identity
        ),
        state_codec=Reference(id="entry-input-view", version="1"),
        configuration_fingerprint=identity,
    )
    content = ComponentCodec(binding, StudyInput).encode(value).content
    if content != _declared_protocol_bytes():
        raise ValueError("Original study input differs from the complete locally frozen panel/configuration contract")
    return StudyInput.from_json(content)


class Step(CoreImmutable):
    """One actual before-update loss and complete accepted-state digest, unit1."""

    d: Annotated[int, Field(strict=True, ge=0, le=63)]
    accepted_u: Annotated[int, Field(strict=True, ge=1, le=64)]
    before: Vector
    after: Vector
    state_digest: Fingerprint

    @model_validator(mode="after")
    def _simplex(self) -> Self:
        validate_simplex(self.before)
        validate_simplex(self.after)
        return self


class Checkpoint(CoreImmutable):
    """Complete initial/midpoint/final state bytes; reading grants no ACK."""

    position: Annotated[int, Field(strict=True, ge=0, le=64)]
    content: Annotated[str, Field(min_length=1, max_length=16 * 1024**2)]
    digest: Fingerprint

    @model_validator(mode="after")
    def _bytes(self) -> Self:
        if self.digest != "sha256:" + hashlib.sha256(self.content.encode()).hexdigest():
            raise ValueError("Checkpoint digest must identify its exact complete UTF-8 bytes")
        return self


class CaseRun(Record):
    """Retain complete or limited actual work; unavailable terminal scores are null.

    Attributes
    ----------
    case_id, setting : str, Setting
        Exact original source and mathematical treatment identities.
    status, reason : str, str or None
        Completed, numerical refusal, known resource limit or not started.
    horizon, attempted, accepted_u : int, int, int or None
        Declared panels, actual observe calls and accepted updates separately.
    trace, checkpoints : tuple
        Actual original coefficients/state evidence; no implicit idempotency.
    metrics : tuple of Measurement
        Explicit units/statuses, including valid zero/negative comparator excess.
    """

    case_id: str
    setting: Setting
    configuration_fingerprint: Fingerprint
    status: Status
    reason: str | None
    horizon: Annotated[int, Field(strict=True, ge=1, le=64)]
    attempted: Annotated[int, Field(strict=True, ge=0, le=64)]
    accepted_u: Annotated[int, Field(strict=True, ge=0, le=64)] | None
    trace: Annotated[tuple[Step, ...], Field(max_length=64)]
    checkpoints: Annotated[tuple[Checkpoint, ...], Field(max_length=3)]
    metrics: tuple[Measurement, ...]

    @model_validator(mode="after")
    def _work(self) -> Self:
        if self.configuration_fingerprint != fingerprint(self.setting.model_dump(mode="json")):
            raise ValueError("Case treatment fingerprint must retain the complete original setting")
        if tuple(m.name for m in self.metrics) != METRICS or any(m.unit != "1" for m in self.metrics):
            raise ValueError("Case scores require every original metric in dimensionless loss/coefficient units")
        return self

    @model_validator(mode="after")
    def _clocks(self) -> Self:
        """Count every actual accepted/attempted prefix without inventing work."""
        if tuple(s.d for s in self.trace) != tuple(range(len(self.trace))):
            raise ValueError("Original decision trace must be contiguous")
        if tuple(s.accepted_u for s in self.trace) != tuple(range(1, len(self.trace) + 1)):
            raise ValueError("Accepted update clock must retain its original chronological prefix")
        if not len(self.trace) <= self.attempted <= self.horizon:
            raise ValueError("Accepted trace cannot exceed attempted work or the original horizon")
        if self.accepted_u is not None and self.accepted_u != len(self.trace):
            raise ValueError("Known accepted work must equal the complete retained chronological trace")
        return self

    @model_validator(mode="after")
    def _checkpoint_prefix(self) -> Self:
        """Join every retained complete checkpoint to actual chronological evidence."""
        positions = tuple(c.position for c in self.checkpoints)
        if self.status == "not_started":
            if self.accepted_u is not None or self.attempted or self.trace or self.checkpoints:
                raise ValueError("Unstarted work has unavailable accepted state and no acquired trace/checkpoint")
        elif self.accepted_u is None or not positions or positions[0] != 0 or positions[-1] != self.accepted_u:
            raise ValueError("Started work requires known initial and final complete state checkpoints")
        return self

    @model_validator(mode="after")
    def _checkpoint_content(self) -> Self:
        """Bind ordered retained state bytes to their original trace positions."""
        positions = tuple(c.position for c in self.checkpoints)
        if positions != tuple(sorted(set(positions))):
            raise ValueError("Checkpoints must retain unique chronological accepted positions")
        if any(p > len(self.trace) for p in positions):
            raise ValueError("A checkpoint cannot precede unrecorded accepted work")
        for checkpoint in self.checkpoints:
            if checkpoint.position and self.trace[checkpoint.position - 1].state_digest != checkpoint.digest:
                raise ValueError("Retained checkpoint differs from its actual accepted trace digest")
        return self

    @model_validator(mode="after")
    def _completion(self) -> Self:
        """Keep unavailable partial scores distinct from completed terminal metrics."""
        positions = tuple(c.position for c in self.checkpoints)
        if self.status == "completed":
            self._completed(positions)
        else:
            self._incomplete()
        return self

    def _completed(self, positions: tuple[int, ...]) -> None:
        """Require every accepted panel, checkpoint and observed terminal score."""
        if (self.reason, self.attempted, self.accepted_u) != (None, self.horizon, self.horizon):
            raise ValueError("Completion requires all original attempted and accepted panels")
        if len(self.trace) != self.horizon or any(m.status != "observed" for m in self.metrics):
            raise ValueError("Completion requires full state trace and observed scores")
        if len(self.trace) // 2 not in positions:
            raise ValueError("Completed work requires its original full midpoint continuation checkpoint")

    def _incomplete(self) -> None:
        """Refuse terminal-score claims for original partial/unstarted work."""
        if self.reason is None:
            raise ValueError("Noncompleted work requires an explicit refusal/limit reason")
        if any(m.status != "unavailable" or m.value is not None for m in self.metrics):
            raise ValueError("Incomplete work cannot advertise full-horizon observed terminal scores")


class FamilyHeader(Record):
    """Bind one original family/process coordinate and its explicit disposition.

    Notes
    -----
    Shared only by the complete family artifact and its development-blind
    process summary. Inheritance retains the same required field types/order
    and canonical JSON; this header is neither completion proof nor authority.
    """

    protocol_content: Fingerprint
    family: Family
    repetition: Annotated[int, Field(strict=True, ge=0, le=2)]
    status: Literal["completed", "numerical_failure", "resource_limited"]
    reason: str | None


class FamilyRun(FamilyHeader):
    """Retain one full family grid and its observed work, never new empirical units.

    Attributes
    ----------
    protocol_input_bytes, typed_input_bytes : int
        Original raw UTF-8 file bytes and the complete converted DTO bytes.
        These are logical input sizes, not network traffic or allocated storage.
    logical_checkpoint_bytes, logical_case_bytes : int
        Sum of retained checkpoint content and standalone case-artifact bytes.
        Consolidation preserves those bodies inside the family file; these
        logical sums are not its final on-disk allocation or output reservation.
    attempted : int
        Known actual observe calls across every original treatment, including
        numerical refusals. Accepted U is retained separately in each case.
    resources : tuple of Measurement
        Worker inner wall/process CPU in seconds and own lifetime high-water
        in MiB observed before final publication. Attempt records retain the
        complete child wall/CPU; the two measurement windows are distinct.
    """

    runtime: InstalledRuntime
    driver_runtime: SelectedInstalledRuntime
    source_files: tuple[tuple[str, str], ...]
    protocol_input_bytes: Annotated[int, Field(strict=True, ge=1, le=128 * 1024)]
    typed_input_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024**2)]
    logical_checkpoint_bytes: Annotated[int, Field(strict=True, ge=0, le=64 * 1024**2)]
    logical_case_bytes: Annotated[int, Field(strict=True, ge=0, le=64 * 1024**2)]
    attempted: Count
    resources: tuple[Measurement, ...]
    cases: Annotated[tuple[CaseRun, ...], Field(min_length=48, max_length=192)]

    @model_validator(mode="after")
    def _identity(self) -> Self:
        self.driver_runtime.require_base(self.runtime, extras=("cli",), pins=(("click", "8.1.8"),))
        if self.attempted != sum(c.attempted for c in self.cases):
            raise ValueError("Family work must include every actual case's attempted calls")
        if (self.status == "completed") != all(c.status == "completed" for c in self.cases):
            raise ValueError("Family completion cannot hide incomplete or refused cases")
        if (self.reason is None) != (self.status == "completed"):
            raise ValueError("Limited family work needs a reason; completed work cannot retain a hidden limit")
        return self

    @model_validator(mode="after")
    def _resources(self) -> Self:
        """Keep actual inner timing and own-process RSS in their declared units."""
        expected = (("process_wall", "s"), ("process_cpu", "s"), ("peak_rss", "MiB"))
        if tuple((m.name, m.unit) for m in self.resources) != expected:
            raise ValueError("Family resources require all original explicit names and compatible units")
        if any(m.value is not None and m.value < 0 for m in self.resources):
            raise ValueError("Physical elapsed time and peak RSS cannot be negative")
        return self

    def require_grid(self, protocol: StudyInput) -> None:
        """Refuse missing, duplicate or substituted original treatment/source rows.

        Parameters
        ----------
        protocol : StudyInput
            Complete exact original hash-pinned source and family specification.

        Raises
        ------
        ValueError
            The report differs from the complete original ordered grid.
        """
        original = admit_protocol(protocol)
        expected = _original_grid(original, self.family)
        if (
            self.protocol_content != original.protocol_content
            or tuple(_treatment_key(c.setting, c.case_id, c.horizon) for c in self.cases) != expected
        ):
            raise ValueError("Family report must retain the complete exact ordered configuration/case grid")


class Artifact(CoreImmutable):
    """Identify actual retained relative bytes; the parent still reads the full DTO."""

    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: Annotated[int, Field(strict=True, ge=1, le=16 * 1024**2)]


class DevelopmentScore(CoreImmutable):
    """Only one original development path's exact stored-binary64 total loss."""

    case_id: str
    setting: Setting
    horizon: Annotated[int, Field(strict=True, ge=1, le=64)]
    exact_loss: Annotated[str, Field(max_length=1024)]

    @model_validator(mode="after")
    def _score(self) -> Self:
        try:
            value = Fraction(self.exact_loss)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("Development loss requires a finite canonical rational") from error
        if str(value) != self.exact_loss or not 0 <= value <= self.horizon:
            raise ValueError("Development loss requires its original canonical exact bounded score")
        return self


class DevelopmentFamily(Record):
    """Selection ingress deliberately has no evaluation fields or family-drop flag."""

    protocol_content: Fingerprint
    family: Family
    rows: Annotated[tuple[DevelopmentScore, ...], Field(min_length=24, max_length=96)]

    def require_grid(self, protocol: StudyInput) -> None:
        """Require all original development cases/settings, including unfavorable ones.

        Parameters
        ----------
        protocol : StudyInput
            Complete original source, admitted again at this public boundary.

        Raises
        ------
        ValueError
            Any missing, duplicate, evaluation-role or substituted treatment row.
        """
        original = admit_protocol(protocol)
        expected = _original_grid(original, self.family, "development")
        if (
            self.protocol_content != original.protocol_content
            or tuple(_treatment_key(r.setting, r.case_id, r.horizon) for r in self.rows) != expected
        ):
            raise ValueError("Development-only projection requires the complete exact original development grid")
