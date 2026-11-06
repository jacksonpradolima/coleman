"""Future-effective semantic changes without rewriting history or restoring state."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.core.primitives import Identifier
from coleman.core.temporal import Timestamp
from coleman.objectives.bindings import DefinitionBinding, DefinitionLineage


class SemanticChange(CoreContract):
    """Declare a future transition between exact semantic-context contents.

    Attributes
    ----------
    stream : Reference
        Exact host task/history identity; versions never match by display name.
    change_id : str
        Stable host-owned logical change identity.
    previous, replacement : DefinitionBinding
        Explicit old/new definition or context reference and content fingerprint.
    effective_round : int
        Nonnegative future logical position; applicability requires a frontier.
    effective_at : datetime or None
        Supplied zoned effective instant or explicit null for untimed change.
    disposition : {'restart', 'external_migration'}
        Declared state policy; neither is executed by this record.
    migration : Reference or None
        Required named migration only for external_migration, otherwise null.
    lineage : DefinitionLineage
        Source/declaration facts, without fabricated evidence or timestamps.

    Raises
    ------
    ValueError
        Invalid fields, identical old/new bindings or inconsistent migration.

    Notes
    -----
    A declaration cannot authorize itself against history. Use validate_change
    with the actual stream/current binding/frontier. Replay and committed state
    recovery retain their separate ownership; no history, budget or I/O changes.
    """

    stream: Reference
    change_id: Identifier
    previous: DefinitionBinding
    replacement: DefinitionBinding
    effective_round: Annotated[int, Field(strict=True, ge=0)]
    effective_at: Timestamp | None
    disposition: Literal["restart", "external_migration"]
    migration: Reference | None
    lineage: DefinitionLineage

    @model_validator(mode="after")
    def _transition(self) -> Self:
        """Keep old/new contents distinct and migration declarations explicit."""
        if self.previous == self.replacement:
            raise ValueError("Semantic change requires different definition content")
        if (self.disposition == "external_migration") != (self.migration is not None):
            raise ValueError("Exactly an external migration requires its named implementation")
        if (
            self.effective_at is not None
            and self.lineage.declared_at is not None
            and self.lineage.declared_at > self.effective_at
        ):
            raise ValueError("Known change declaration cannot follow its effective instant")
        return self


def validate_change(
    change: SemanticChange,
    *,
    stream: Reference,
    current: DefinitionBinding,
    frontier: int,
    frontier_at: Timestamp | None = None,
) -> SemanticChange:
    """Check future-only applicability against explicitly supplied current facts.

    Parameters
    ----------
    change : SemanticChange
        Typed immutable proposed transition.
    stream : Reference
        Actual host stream/task identity and version.
    current : DefinitionBinding
        Definition content currently used by that stream.
    frontier : int
        Last committed nonnegative logical position; bool/coerced input rejects.
    frontier_at : datetime or None, optional
        Last committed supplied instant, if timed. Timed changes require both
        instants and strict future ordering; no ambient clock is read.

    Returns
    -------
    SemanticChange
        Revalidated declaration, with caller inputs unchanged.

    Raises
    ------
    ValueError
        Wrong current binding/stream, nonfuture position, mismatched clock modes
        or an invalid declaration/frontier.

    Notes
    -----
    Does not execute or qualify migration, consume an event, reset movement usage
    or alter protected history. Durable acceptance remains a session obligation.
    """
    checked = SemanticChange.model_validate(change)
    # pylint: disable-next=unidiomatic-typecheck
    if type(frontier) is not int or frontier < 0:
        raise ValueError("Change frontier requires a nonnegative integer")
    if checked.stream != Reference.model_validate(stream) or checked.previous != DefinitionBinding.model_validate(
        current
    ):
        raise ValueError("Change does not extend the actual current stream/binding")
    if checked.effective_round <= frontier:
        raise ValueError("Semantic change cannot rewrite a committed logical position")
    if (checked.effective_at is None) != (frontier_at is None):
        raise ValueError("Timed change applicability requires both supplied instants")
    if frontier_at is not None:
        instant = TypeAdapter(Timestamp).validate_python(frontier_at)
        if checked.effective_at is None or checked.effective_at <= instant:
            raise ValueError("Semantic change must follow the committed instant")
    return checked
