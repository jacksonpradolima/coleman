"""Authored complete-grid diagnostics; no physical execution or empirical claim."""

from datetime import UTC, datetime, timedelta

from coleman.evaluation.prequential.evidence import DiagnosticPanel
from tests.evaluation.design_support import changed, make_design, make_raw_panel, make_selection


def declarations(*, unsupported=False, units=2):
    """Keep original source events but admit current actual assembly clocks."""
    now = datetime.now(UTC)
    original = make_design()
    members = tuple(
        changed(
            original.members[0],
            member_id=f"world-{index}",
            unit_id=f"independent-world-{index}",
            path=changed(original.members[0].path, id=f"independent-path-{index}"),
            metadata=changed(
                original.members[0].metadata, observation_id=f"source-{index}", group_id=f"source-group-{index}"
            ),
        )
        for index in range(units)
    )
    design = changed(
        original,
        members=members,
        snapshot=changed(original.snapshot, captured_at=None),
        comparison_plan=changed(original.comparison_plan, test_end=now + timedelta(days=1)),
    )
    if unsupported:
        design = changed(design, inference=changed(design.inference, dependence="unknown"))
    return design, make_selection(design, actual_selected_at=now - timedelta(minutes=1))


def diagnostic_panel(design, selection, *, mode="produced", values=None, states=None):
    """Retain original producer records even for an unsupported inference request."""
    supported = changed(design, inference=changed(design.inference, dependence="independent"))
    producer_selection = make_selection(supported, actual_selected_at=selection.actual_selected_at)
    now = datetime.now(UTC)
    times = {
        (item.candidate.candidate_id, member.member_id, seed): now
        for item in supported.candidates
        for member in supported.protected_members
        for seed in supported.seeds
    }
    original = make_raw_panel(
        supported, producer_selection, mode=mode, values=values, states=states, record_times=times, assembled_at=now
    )
    return DiagnosticPanel(
        schema_version=1,
        design=design,
        selection=selection,
        rows=original.rows,
        ledger=original.ledger,
        mode=mode,
        assembled_at=now,
    )


def protected_panel(design, selection, *, values=None):
    """Return an actual supported RawPanel at callback time with exact producer FP."""
    now = datetime.now(UTC)
    times = {
        (item.candidate.candidate_id, member.member_id, seed): now
        for item in design.candidates
        for member in design.protected_members
        for seed in design.seeds
    }
    return make_raw_panel(design, selection, values=values, record_times=times, assembled_at=now)
