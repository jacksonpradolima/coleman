"""Small real-component preparations and consciously authored outcome evidence."""

from pathlib import Path

from coleman.core.actions import SubsetAction
from coleman.core.feedback import ChosenFeedback, ComponentFeedback, FeedbackRow, FullPanel
from coleman.core.primitives import fingerprint
from coleman.replay.contracts import ReplayFeedback, ValueOriginEvidence
from coleman.replay.engine import CausalReplay

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/replay/v1"


def session(example, profile="chosen", *, policy=None, host=None):
    host = example.HostConsumer(profile) if host is None else host
    replay = CausalReplay(
        host,
        example.policy(profile) if policy is None else policy,
        state_type=example.HostConsumerState,
        state_codec=example.CODEC,
    )
    replay.admit_history(example.history())
    return replay, host


def emit(example, replay, index, *, profile="chosen", **clocks):
    opportunity = example.opportunity(index, profile=profile, **clocks)
    return replay.emit(replay.prepare(opportunity), execution="offline_identity_v1")


def chosen(record):
    return record.executed.action.chosen.id


def selected(record):
    action = record.executed.action
    assert isinstance(action, SubsetAction)
    return tuple(member.id for member in action.selected)


def with_rows(example, feedback, rows):
    """Rebuild real typed content plus matching origin receipts at its actual boundary."""
    content = feedback.batch.feedback
    if isinstance(content, FullPanel):
        replaced = FullPanel(rows=rows)
    elif isinstance(content, ChosenFeedback):
        replaced = ChosenFeedback(row=rows[0])
    else:
        assert isinstance(content, ComponentFeedback)
        replaced = ComponentFeedback(rows=rows, completeness=content.completeness)
    batch = type(feedback.batch).model_validate(feedback.batch.model_copy(update={"feedback": replaced}))
    origins = tuple(
        ValueOriginEvidence(
            member=row.member,
            dimension=value.spec.dimension,
            value_fingerprint=fingerprint(value.model_dump(mode="json")),
            origin=value.origin,
            producer=None if value.origin == "realized" else example.GENERATOR,
        )
        for row in rows
        for value in row.values
    )
    return ReplayFeedback.model_validate(feedback.model_copy(update={"batch": batch, "origins": origins}))


def rows(feedback):
    content = feedback.batch.feedback
    return (content.row,) if isinstance(content, ChosenFeedback) else content.rows


def set_clocks(feedback, **changes):
    timing = type(feedback.batch.timing).model_validate(feedback.batch.timing.model_copy(update=changes))
    batch = type(feedback.batch).model_validate(feedback.batch.model_copy(update={"timing": timing}))
    return ReplayFeedback.model_validate(feedback.model_copy(update={"batch": batch}))


def changed_value(example, feedback, value, *, status="observed"):
    modified = tuple(
        FeedbackRow(
            member=row.member,
            values=tuple(
                type(outcome).model_validate(outcome.model_copy(update={"value": value, "status": status}))
                for outcome in row.values
            ),
        )
        for row in rows(feedback)
    )
    return with_rows(example, feedback, modified)
