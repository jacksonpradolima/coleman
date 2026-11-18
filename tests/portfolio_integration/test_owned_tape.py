"""Owned source rows bind live quotes and complete checkpoint continuation."""

from datetime import timedelta

import pytest

from coleman.core.primitives import fingerprint
from coleman.problems.portfolio import PricePanel
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio.experiment import Experiment, ExperimentState, ExperimentStateV2


def _owners(experiment):
    """Retain actual live owner identities as well as their complete wire state."""
    groups = (experiment.books, experiment.references, experiment.controlled)
    return (
        tuple((id(group), tuple(id(owner) for owner in group.values())) for group in groups),
        id(experiment.ridge),
        experiment.export_state().to_json(),
    )


def _restore(experiment: Experiment, state: ExperimentState | ExperimentStateV2) -> None:
    """Use the actual public restore for the selected complete representation."""
    if isinstance(state, ExperimentStateV2):
        experiment.restore_normalized_state(state)
    else:
        experiment.restore_state(state)


def _changed_quote(index, change):
    """Vary actual source fields or present unchecked unsupported wire fields."""
    if change == "value":
        first, second = model.PRICE_ROWS[index]
        return model.price_panel(index, (first + 100, second))
    wire = model.prices(index).model_dump(mode="python")
    if change == "source":
        wire["source"]["id"] += "-another-source"
    elif change == "source_version":
        wire["source"]["version"] = "2"
    elif change == "fingerprint":
        wire["source_fingerprint"] = fingerprint({"another": "current-source-identity"})
    elif change == "status":
        wire["quotes"][0]["status"] = "stale"
    elif change == "event":
        wire["quotes"][0]["event_at"] -= timedelta(seconds=1)
    elif change == "unit":
        wire["quotes"][0]["unit"] = "another-credit/asset-unit"
    else:
        assert change == "version"
        wire["schema_version"] = 2
    return PricePanel.model_construct(**wire)


@pytest.fixture(scope="module")
def owned_frontiers():
    experiment = Experiment()
    initial = experiment.export_state()
    for index in range(4):
        experiment.step(index)
    decided = experiment.export_state()
    experiment.apply_feedback(4)
    return initial, decided, experiment.export_state()


@pytest.mark.parametrize("phase", ["first", "later", "feedback", "ready-feedback"])
@pytest.mark.parametrize(
    "change", ["value", "source", "source_version", "fingerprint", "status", "event", "unit", "version"]
)
def test_owned_quote_mismatch_refuses_without_changing_any_live_owner(owned_frontiers, phase, change):
    initial, decided, ready = owned_frontiers
    experiment = Experiment()
    experiment.restore_state(initial if phase == "first" else ready if phase == "ready-feedback" else decided)
    before = _owners(experiment)
    index = len(experiment.periods)
    operation = experiment.apply_feedback if "feedback" in phase else experiment.step
    with pytest.raises(ValueError):
        operation(index, prices=_changed_quote(index, change))
    assert _owners(experiment) == before


@pytest.fixture(scope="module")
def foreign_frontiers():
    declared = Experiment().source_fingerprint
    states = []
    for changed_index in (0, 3, 4):
        external = Experiment(rows=None, source_fingerprint=declared)
        for index in range(4):
            quote = _changed_quote(index, "value") if index == changed_index else model.prices(index)
            external.step(index, prices=quote)
        if changed_index == 4:
            external.apply_feedback(4, prices=_changed_quote(4, "value"))
        # Both real complete public readers accept these internally consistent
        # histories. A declared digest alone cannot authenticate external rows.
        legacy = ExperimentState.from_json(external.export_state().to_json())
        normalized = ExperimentStateV2.from_json(external.export_normalized_state().to_json())
        states.append((legacy, normalized))
    return states


@pytest.mark.parametrize("position", [0, 1, 2], ids=["first-mark", "later-mark", "feedback-ready-mark"])
@pytest.mark.parametrize("representation", [0, 1], ids=["legacy", "normalized"])
def test_owned_restore_rejects_consistent_foreign_prefix_with_same_claimed_digest(
    foreign_frontiers, owned_frontiers, position, representation
):
    state = foreign_frontiers[position][representation]
    external = Experiment(rows=None, source_fingerprint=state.tape_fingerprint)
    _restore(external, state)
    assert external.export_state().to_json() == foreign_frontiers[position][0].to_json()
    owned = Experiment()
    owned.restore_state(owned_frontiers[1])
    before = _owners(owned)
    with pytest.raises(ValueError, match="owned frozen source"):
        _restore(owned, state)
    assert _owners(owned) == before


@pytest.mark.parametrize("representation", [0, 1], ids=["legacy", "normalized"])
def test_matching_explicit_quotes_and_feedback_ready_continuation_match_owned_route(owned_frontiers, representation):
    direct = Experiment()
    explicit = Experiment()
    for index in range(4):
        assert explicit.step(index, prices=model.prices(index)) == direct.step(index)
    assert explicit.apply_feedback(4, prices=model.prices(4)) == direct.apply_feedback(4)
    state = explicit.export_state() if representation == 0 else explicit.export_normalized_state()
    restored = Experiment()
    _restore(restored, state)
    assert restored.export_state().to_json() == owned_frontiers[2].to_json()
    # Exact feedback retries retain their original producers and do no work.
    before = restored.export_state().to_json()
    assert restored.apply_feedback(4, prices=model.prices(4)) == direct.ready_observation
    assert restored.export_state().to_json() == before
    for index in (4, 5):
        assert restored.step(index, prices=model.prices(index)) == direct.step(index)
    assert restored.export_state().to_json() == direct.export_state().to_json()


@pytest.mark.parametrize("representation", [0, 1], ids=["legacy", "normalized"])
def test_changed_future_tail_preserves_prefix_but_requires_its_own_complete_source_identity(representation):
    changed_rows = model.PRICE_ROWS[:4] + tuple((first + 20, second + 30) for first, second in model.PRICE_ROWS[4:])
    original, changed = Experiment(), Experiment(rows=changed_rows)
    for index in range(4):
        assert changed.step(index, prices=model.prices(index)) == original.step(index)
    before = _owners(changed)
    original_state = original.export_state() if representation == 0 else original.export_normalized_state()
    with pytest.raises(ValueError, match="another tape or selected law"):
        _restore(changed, original_state)
    assert _owners(changed) == before
    declared = changed.source_fingerprint
    external = Experiment(rows=None, source_fingerprint=declared)
    changed_state = changed.export_state() if representation == 0 else changed.export_normalized_state()
    _restore(external, changed_state)
    assert external.step(4, prices=model.prices(4, changed_rows)) == changed.step(4)
    assert external.export_state().to_json() == changed.export_state().to_json()


def test_external_source_requires_each_new_current_quote_and_preserves_atomic_refusal():
    external = Experiment(rows=None, source_fingerprint=fingerprint({"declared": "external-frozen-source"}))
    before = _owners(external)
    with pytest.raises(ValueError, match="unavailable"):
        external.step(0)
    assert _owners(external) == before
    external.step(0, prices=model.price_panel(0, (200, 100)))
    before = _owners(external)
    with pytest.raises(ValueError, match="unavailable"):
        external.apply_feedback(1)
    assert _owners(external) == before
    external.apply_feedback(1, prices=model.price_panel(1, (205, 98)))
    external.step(1)
    assert external.ridge.export_state().payload.sample_count == 1
