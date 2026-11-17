"""The task's complete outer audit precedes unchanged real staged restoration."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.contracts import StateBinding
from examples.synthetic_portfolio import model
from examples.synthetic_portfolio._state import PortfolioOwner
from examples.synthetic_portfolio.experiment import Experiment, ExperimentState
from examples.synthetic_portfolio.task_contracts import PortfolioCheckpoint


@pytest.fixture(scope="module")
def actual_prefix():
    virgin = Experiment().export_state()
    initial = PortfolioCheckpoint(
        schema_version=1,
        binding=StateBinding(
            owner=model.ref("checked-restore-test-owner"),
            implementation=virgin.implementation,
            state_codec=model.ref("checked-restore-test-checkpoint"),
            configuration_fingerprint=fingerprint({"strength": virgin.strength}),
        ),
        plan_fingerprint=fingerprint({"pure": "checked-restore-plan"}),
        coordinate_fingerprint=fingerprint({"pure": "checked-restore-coordinate"}),
        source=DefinitionBinding(
            reference=model.ref("checked-restore-tape"), definition_fingerprint=virgin.tape_fingerprint
        ),
        rows=model.PRICE_ROWS,
        experiment=virgin,
    )
    owner = PortfolioOwner(initial)
    for index in range(4):
        owner.experiment.step(index)
    saved = PortfolioCheckpoint.from_json(owner.export_state().to_json())
    return initial, saved


def restored_continuation(initial, saved):
    """Exercise the actual outer reader, all core restores and complete book paths."""
    owner = PortfolioOwner(initial)
    owner.restore_state(saved)
    actual = owner.export_state()
    books = tuple((name, book.export_state().to_json()) for name, book in sorted(owner.experiment.books.items()))
    owner.experiment.step(4)
    return actual.to_json(), books, owner.experiment.export_state().to_json()


def test_outer_checked_restore_matches_public_restore_and_independent_thread_contexts(actual_prefix):
    initial, saved = actual_prefix
    direct = Experiment()
    direct.restore_state(ExperimentState.from_json(saved.experiment.to_json()))
    expected_books = tuple((name, book.export_state().to_json()) for name, book in sorted(direct.books.items()))
    assert direct.export_state().to_json() == saved.experiment.to_json()
    direct.step(4)
    expected = saved.to_json(), expected_books, direct.export_state().to_json()
    assert restored_continuation(initial, saved) == expected
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(restored_continuation, initial, saved) for _ in range(2))
        assert tuple(future.result() for future in futures) == (expected, expected)


@pytest.mark.parametrize(
    "change", ["outer_version", "inner_version", "nested_version", "producer", "book", "law", "tape"]
)
def test_forged_outer_child_is_rejected_before_private_restore_and_without_live_mutation(
    actual_prefix, monkeypatch, change
):
    initial, saved = actual_prefix
    owner = PortfolioOwner(initial)
    before = owner.export_state().to_json()
    calls = []
    actual_restore = owner.experiment._restore_checked

    def observed_restore(checked):
        calls.append(checked)
        actual_restore(checked)

    monkeypatch.setattr(owner.experiment, "_restore_checked", observed_restore)
    changed = saved.model_dump(mode="python")
    inner = changed["experiment"]
    if change == "outer_version":
        changed["schema_version"] = 2
    elif change == "inner_version":
        inner["schema_version"] = 2
    elif change == "nested_version":
        inner["controlled"][0]["state"]["payload"]["schema_version"] = 2
    elif change == "producer":
        inner["pending"]["prediction"]["input"]["values"] = (2.0, 0.0)
    elif change == "book":
        inner["books"][0]["state"]["cash"]["numerator"] += 1
    elif change == "law":
        inner["strength"] = 0.1
    else:
        inner["tape_fingerprint"] = "sha256:" + "0" * 64
    unchecked = PortfolioCheckpoint.model_construct(
        **(changed | {"experiment": ExperimentState.model_construct(**inner)})
    )
    with pytest.raises(ValueError):
        owner.restore_state(unchecked)
    assert calls == []
    assert owner.export_state().to_json() == before


@pytest.mark.parametrize("change", ["version", "producer", "book"])
def test_public_inner_restore_still_revalidates_unchecked_state_atomically(actual_prefix, change):
    _, saved = actual_prefix
    direct = Experiment()
    before = direct.export_state().to_json()
    changed = saved.experiment.model_dump(mode="python")
    if change == "version":
        changed["schema_version"] = 2
    elif change == "producer":
        changed["pending"]["prediction"]["input"]["values"] = (2.0, 0.0)
    else:
        changed["books"][0]["state"]["cash"]["numerator"] += 1
    with pytest.raises(ValueError):
        direct.restore_state(ExperimentState.model_construct(**changed))
    assert direct.export_state().to_json() == before


@pytest.mark.parametrize("change", ["law", "implementation"])
def test_fully_typed_but_incompatible_inner_law_refuses_before_atomic_adoption(actual_prefix, change):
    initial, saved = actual_prefix
    owner = PortfolioOwner(initial)
    before = owner.export_state().to_json()
    if change == "law":
        another = Experiment(strength=0.1)
        for index in range(4):
            another.step(index)
        incompatible = another.export_state()
    else:
        incompatible = ExperimentState.model_validate(
            saved.experiment.model_copy(
                update={
                    "implementation": DefinitionBinding(
                        reference=saved.experiment.implementation.reference,
                        definition_fingerprint="sha256:" + "0" * 64,
                    )
                }
            )
        )
    outer = PortfolioCheckpoint(**initial.model_dump(exclude={"experiment"}), experiment=incompatible)
    with pytest.raises(ValueError, match="another tape or selected law|exact selected host implementation"):
        owner.restore_state(outer)
    assert owner.export_state().to_json() == before


def test_public_restore_retains_exact_tape_compatibility(actual_prefix):
    _, saved = actual_prefix
    different = Experiment(rows=model.PRICE_ROWS[:-1] + ((106, 104),))
    before = different.export_state().to_json()
    with pytest.raises(ValueError, match="another tape or selected law"):
        different.restore_state(saved.experiment)
    assert different.export_state().to_json() == before
