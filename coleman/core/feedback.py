"""Typed feedback coverage, per-value origin and bounded local retry checking."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.numeric import Scalar
from coleman.core.primitives import MAX_DIMENSIONS, MAX_MEMBERS, Fingerprint, Identifier
from coleman.core.timing import FeedbackTiming

Origin = Literal["realized", "modeled_counterfactual", "synthetic"]
FeedbackKind = Literal["full_panel", "chosen", "component", "aggregate", "absent"]


class OutcomeValue(Scalar):
    """Retain one outcome's value state, origin and optional transform identity.

    Attributes
    ----------
    origin : {'realized', 'modeled_counterfactual', 'synthetic'}
        Declared origin of this value, independent of action coverage and scalar
        availability. A modeled value is never made realized by being present.
    transform : Reference or None
        Required explicit reference or null. Loss/reward roles require a named
        transform, including an explicit identity transform when appropriate.

    Raises
    ------
    ValueError
        Invalid scalar, origin or a missing loss/reward transform.

    Notes
    -----
    Inherited scalar units, inclusive bounds and null states remain unchanged.
    The reference identifies an operation; no transformation, imputation, source
    authentication, filesystem access or algorithm update is performed.
    """

    origin: Origin
    transform: Reference | None

    @model_validator(mode="after")
    def _transform(self) -> Self:
        """Require explicit interpretation of loss and reward without executing it."""
        if self.spec.role in {"loss", "reward"} and self.transform is None:
            raise ValueError("Loss and reward outcomes require a transform reference")
        return self


def _values(values: tuple[OutcomeValue, ...]) -> tuple[OutcomeValue, ...]:
    """Canonicalize outcome dimensions without merging conflicting versions."""
    ids = [value.spec.dimension.id for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Feedback dimension IDs must be unique, including their versions")
    return tuple(sorted(values, key=lambda value: value.spec.dimension.id))


class FeedbackRow(CoreImmutable):
    """Describe one member with independently interpreted outcome values.

    Attributes
    ----------
    member : Reference
        Exact action/expert identity and semantic version. Membership in an
        executed decision is checked separately by capability validation.
    values : tuple of OutcomeValue
        One through 64 uniquely named dimensions, canonicalized by dimension ID.
        Each value retains its own unit, role, missingness and origin.

    Raises
    ------
    ValueError
        Invalid fields, empty/oversized values or duplicate dimension IDs.

    Notes
    -----
    Multiple dimensions for one chosen member do not reveal other members.
    Construction performs no observation lookup, update or I/O.
    """

    member: Reference
    values: Annotated[tuple[OutcomeValue, ...], Field(min_length=1, max_length=MAX_DIMENSIONS)]

    @field_validator("values")
    @classmethod
    def _dimensions(cls, values: tuple[OutcomeValue, ...]) -> tuple[OutcomeValue, ...]:
        """Keep dimension order canonical while rejecting duplicate identities."""
        return _values(values)


def _rows(rows: tuple[FeedbackRow, ...]) -> tuple[FeedbackRow, ...]:
    """Canonicalize member rows; duplicate IDs cannot conceal version changes."""
    ids = [row.member.id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Feedback member IDs must be unique, including their versions")
    return tuple(sorted(rows, key=lambda row: row.member.id))


class FullPanel(CoreImmutable):
    """Declare complete member coverage without asserting value eligibility.

    Attributes
    ----------
    kind : {'full_panel'}
        Distinct coverage discriminator, independent of outcome dimensions.
    rows : tuple of FeedbackRow
        One through 4096 unique members, canonicalized by member ID. The exact
        expected universe/version and observed losses are checked against the
        executed decision and its consumer capability.

    Raises
    ------
    ValueError
        Invalid fields, empty/oversized panel or duplicate member identities.

    Notes
    -----
    Explicit unknown/null values may be retained structurally; they cannot meet
    a complete-observed-loss capability. No unobserved outcomes are invented.
    """

    kind: Literal["full_panel"] = "full_panel"
    rows: Annotated[tuple[FeedbackRow, ...], Field(min_length=1, max_length=MAX_MEMBERS)]

    @field_validator("rows")
    @classmethod
    def _members(cls, rows: tuple[FeedbackRow, ...]) -> tuple[FeedbackRow, ...]:
        """Preserve unique full-panel members in canonical order."""
        return _rows(rows)


class ChosenFeedback(CoreImmutable):
    """Retain outcomes only for one actual chosen alternative.

    Attributes
    ----------
    kind : {'chosen'}
        Chosen-only coverage, even when the row contains several measurements.
    row : FeedbackRow
        Member and per-value observations; decision matching is checked later.

    Notes
    -----
    No alternative's outcome, success count or sampling probability is inferred.
    Validation is pure and invalid fields raise ``ValueError``.
    """

    kind: Literal["chosen"] = "chosen"
    row: FeedbackRow


class ComponentFeedback(CoreImmutable):
    """Retain member feedback for an executed subset, slate or acquisition.

    Attributes
    ----------
    kind : {'component'}
        Component coverage, distinct from a whole-action aggregate.
    rows : tuple of FeedbackRow
        One through 4096 unique observed members, canonicalized by member ID.
    completeness : {'complete', 'partial'}
        Whether rows claim all actually selected members. Decision validation
        checks that claim; unselected members are never eligible components.

    Raises
    ------
    ValueError
        Invalid fields, empty/oversized rows or duplicate member identities.

    Notes
    -----
    Completeness refers to selected members, not numeric availability. A null
    value remains null. Use explicit absent feedback when no member was observed.
    """

    kind: Literal["component"] = "component"
    rows: Annotated[tuple[FeedbackRow, ...], Field(min_length=1, max_length=MAX_MEMBERS)]
    completeness: Literal["complete", "partial"]

    @field_validator("rows")
    @classmethod
    def _members(cls, rows: tuple[FeedbackRow, ...]) -> tuple[FeedbackRow, ...]:
        """Reject duplicates rather than silently completing component coverage."""
        return _rows(rows)


class AggregateFeedback(CoreImmutable):
    """Describe the executed action as a whole, without attributing components.

    Attributes
    ----------
    kind : {'aggregate'}
        Whole-action coverage discriminator.
    values : tuple of OutcomeValue
        One through 64 unique dimensions with per-value origin and semantics.

    Raises
    ------
    ValueError
        Invalid fields, empty/oversized values or duplicate dimension identities.

    Notes
    -----
    An aggregate slate reward cannot be fanned out into member observations.
    Construction performs no accounting allocation, learning or I/O.
    """

    kind: Literal["aggregate"] = "aggregate"
    values: Annotated[tuple[OutcomeValue, ...], Field(min_length=1, max_length=MAX_DIMENSIONS)]

    @field_validator("values")
    @classmethod
    def _dimensions(cls, values: tuple[OutcomeValue, ...]) -> tuple[OutcomeValue, ...]:
        """Preserve unique aggregate dimensions in canonical order."""
        return _values(values)


class AbsentFeedback(CoreImmutable):
    """Record explicit absence without constructing a fake observed zero.

    Attributes
    ----------
    kind : {'absent'}
        No value-bearing feedback is supplied.
    reason : str
        Caller-declared explanation; this is not an executed effect receipt.

    Notes
    -----
    An admitted absent notification does not perform an adaptive update.
    Invalid fields raise ``ValueError``; construction performs no I/O.
    """

    kind: Literal["absent"] = "absent"
    reason: Identifier


Feedback = Annotated[
    FullPanel | ChosenFeedback | ComponentFeedback | AggregateFeedback | AbsentFeedback,
    Field(discriminator="kind"),
]


class FeedbackBatch(CoreContract):
    """Bind typed feedback to one executed decision and revision identity.

    Attributes
    ----------
    decision_fingerprint : str
        Complete executed-decision content fingerprint, not only its proposal
        or logical decision key. Capability validation checks the actual binding.
    feedback_id, revision_id : str
        Stable caller-owned feedback and revision identities.
    supersedes : str or None
        Required predecessor revision or explicit null for a root. Self-reference
        rejects; graph ordering/parent presence belongs to replay, not this model.
    horizon : Reference or None
        Required target/measurement horizon identity or explicit null when the
        direct mathematical mode does not declare a horizon.
    feedback : Feedback
        Coverage discriminator and immutable payload, independent of dimensions.
    timing : FeedbackTiming or None, optional
        Optional declared event/availability/maturity/receipt metadata. Unknown
        receipt remains unknown; no ambient time is collected.
    delay_rounds : int, optional
        Nonnegative logical additional rounds before feedback, default zero.
        Ordinary waiting within one round is not an additional round.

    Raises
    ------
    ValueError
        Invalid fields, unsupported/missing schema version, negative/coerced
        delay, duplicate nested identities or a self-superseding revision.

    Notes
    -----
    This envelope performs no delivery, algorithm update, durable deduplication
    or checkpoint restore. Source lineage and timing are supplied attestations.
    """

    decision_fingerprint: Fingerprint
    feedback_id: Identifier
    revision_id: Identifier
    supersedes: Identifier | None
    horizon: Reference | None
    feedback: Feedback
    timing: FeedbackTiming | None = None
    delay_rounds: Annotated[int, Field(strict=True, ge=0)] = 0

    @model_validator(mode="after")
    def _revision(self) -> Self:
        """Reject a revision naming itself without inventing unseen predecessors."""
        if self.supersedes == self.revision_id:
            raise ValueError("A feedback revision cannot supersede itself")
        return self


def deduplicate_feedback(batches: Iterable[FeedbackBatch]) -> tuple[FeedbackBatch, ...]:
    """Check bounded local retries and return a canonical unique feedback tuple.

    Parameters
    ----------
    batches : iterable of FeedbackBatch
        At most 4096 attempted records, including duplicates. Each typed record
        is revalidated; keys are executed-decision fingerprint/feedback/revision.

    Returns
    -------
    tuple of FeedbackBatch
        Unique validated records sorted by that logical key. Identical canonical
        retries are idempotent; caller inputs and record values are unchanged.

    Raises
    ------
    ValueError
        Invalid/unchecked nested content, more than 4096 attempted records or
        conflicting canonical content at one key, including signed-zero changes.
    TypeError
        An element is not a typed FeedbackBatch or input is not iterable.

    Notes
    -----
    Consumes at most 4097 elements before refusing an oversized iterable. This
    pure local check neither persists acceptance nor promises idempotent learner
    updates, revision-graph validity, concurrency safety or exactly-once effects.
    """
    unique: dict[tuple[str, str, str], FeedbackBatch] = {}
    for index, batch in enumerate(batches):
        if index >= MAX_MEMBERS:
            raise ValueError("Feedback retry input exceeds 4096 attempted records")
        if not isinstance(batch, FeedbackBatch):
            raise TypeError("Feedback retry checking requires typed FeedbackBatch records")
        checked = FeedbackBatch.model_validate(batch)
        key = checked.decision_fingerprint, checked.feedback_id, checked.revision_id
        previous = unique.setdefault(key, checked)
        if previous.to_json() != checked.to_json():
            raise ValueError("Conflicting feedback content at one logical revision key")
    return tuple(unique[key] for key in sorted(unique))
