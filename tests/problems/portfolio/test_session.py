"""Actual deduplication, concurrent owner and complete exact continuation."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from coleman.problems.portfolio import Book, Exact
from coleman.problems.portfolio.session import BookSession, BookSnapshot
from tests.problems.portfolio.test_accounting import book, fees, prices, ref, target


def test_same_original_delivery_is_idempotent_and_changed_content_conflicts():
    owner = BookSession(book())
    first = owner.deliver(prices(), target(), fees(), request=ref("original"))
    before = owner.export_state().to_json()
    assert owner.deliver(prices(), target(), fees(), request=ref("original")).to_json() == first.to_json()
    assert owner.export_state().to_json() == before
    with pytest.raises(ValueError, match="Conflicting"):
        owner.deliver(prices(), target(("1/4", "3/4")), fees(), request=ref("original"))
    assert owner.export_state().to_json() == before


def test_concurrent_duplicate_delivery_commits_one_receipt_and_one_charge():
    owner = BookSession(book())

    def deliver(_):
        return owner.deliver(prices(), target(), fees(), request=ref("same-request")).to_json()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(executor.map(deliver, range(12)))
    assert len(set(results)) == 1
    assert owner.book.sequence == 1 and len(owner.export_state().receipts) == 1


def test_checkpoint_continuation_retains_original_path_and_dedup_receipt():
    owner = BookSession(book())
    first = owner.deliver(prices(), target(), fees(), request=ref("first"))
    checkpoint = BookSnapshot.from_json(owner.export_state().to_json())
    restored = BookSession(book())
    restored.restore_state(checkpoint)
    assert restored.deliver(prices(), target(), fees(), request=ref("first")).to_json() == first.to_json()
    expected = owner.deliver(prices((12, 18), day=1), target(), fees(period="second"), request=ref("next"))
    actual = restored.deliver(prices((12, 18), day=1), target(), fees(period="second"), request=ref("next"))
    assert actual.to_json() == expected.to_json()
    assert restored.export_state().to_json() == owner.export_state().to_json()


def test_unchecked_complete_history_still_gets_full_public_restore_and_atomic_refusal():
    original = BookSession(book())
    original.deliver(prices(), target(), fees(), request=ref("first"))
    last = original.deliver(prices((12, 18), day=1), target(), fees(period="second"), request=ref("next"))
    snapshot = original.export_state()
    constructed = BookSnapshot.model_construct(**snapshot.model_dump(mode="python"))
    restored = BookSession(book())
    restored.restore_state(constructed)
    assert restored.export_state() is not constructed
    before = restored.export_state().to_json()
    assert before == snapshot.to_json()
    assert (
        restored.deliver(prices((12, 18), day=1), target(), fees(period="second"), request=ref("next")).to_json()
        == last.to_json()
    )
    for change in ("root_version", "nested_version", "false_chain", "false_accounting"):
        invalid = snapshot.model_dump(mode="python")
        if change == "root_version":
            invalid["schema_version"] = 2
        elif change == "nested_version":
            invalid["receipts"][0]["schema_version"] = 2
        elif change == "false_chain":
            # Both individual receipts are valid. The entire original sequence
            # must still be checked, rather than trusting only their fields.
            invalid["receipts"] = tuple(reversed(snapshot.receipts))
        else:
            invalid["receipts"][1]["net_value"]["numerator"] += 1
        with pytest.raises(ValueError):
            restored.restore_state(BookSnapshot.model_construct(**invalid))
        assert restored.export_state().to_json() == before


def test_complete_byte_budget_refuses_before_acceptance_within_the_numeric_and_count_limits():
    import json
    from datetime import UTC, datetime, timedelta
    from fractions import Fraction

    from coleman.core.primitives import fingerprint
    from coleman.problems.portfolio import AssetWeight, Fees, Position, PricePanel, PriceQuote, Target, execute

    # Every stored rational remains within4096bits: common denominators cancel
    # in units*(S/T), while both alternating target vectors sum exactly to1.
    denominator_units, denominator_value = 2**4091 - 1, 2**4093 - 1
    base, shift = denominator_value // 8, 2**4088
    first = tuple(base + int(index < 7) for index in range(8))
    second = tuple(value + shift * (1 if index % 2 == 0 else -1) for index, value in enumerate(first))
    assert sum(first) == sum(second) == denominator_value and min(second) > 0
    assets = tuple(ref("capacity-asset-" + str(index)) for index in range(8))
    initial = Book(
        schema_version=1,
        reference=ref("capacity-book"),
        cash=Exact.of(0),
        positions=tuple(
            Position(asset=asset, units=Exact.of(Fraction(value, denominator_units)))
            for asset, value in zip(assets, first, strict=True)
        ),
    )
    owner = BookSession(initial)
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def wire(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    for index in range(128):
        at = start + timedelta(days=index)
        panel = PricePanel(
            schema_version=1,
            source=ref("authored-capacity-source"),
            source_fingerprint=fingerprint({"case": "common-denominator-long-only"}),
            at=at,
            quotes=tuple(
                PriceQuote(
                    asset=asset,
                    value=Exact.of(Fraction(denominator_units, denominator_value)),
                    status="observed",
                    event_at=at,
                    available_at=at,
                )
                for asset in assets
            ),
        )
        proposed = Target(
            schema_version=1,
            reference=ref("capacity-target-" + str(index)),
            weights=tuple(
                AssetWeight(asset=asset, value=Exact.of(Fraction(value, denominator_value)))
                for asset, value in zip(assets, second if index % 2 == 0 else first, strict=True)
            ),
        )
        schedule = Fees(
            schema_version=1,
            reference=ref("capacity-fees"),
            period=ref("capacity-period-" + str(index)),
            declared_at=start,
            proportional=Exact.of(0),
            recurring=Exact.of(0),
            status="observed",
        )
        request = ref("capacity-request-" + str(index))
        before = owner.export_state()
        try:
            receipt = owner.deliver(panel, proposed, schedule, request=request)
        except ValueError as error:
            assert "byte" in str(error)
            assert 0 < len(before.receipts) < 128
            assert owner.export_state() is before
            accepted = wire(before.model_dump(mode="json"))
            assert len(accepted.encode()) <= 16 * 1024**2 - 1
            decoded = BookSnapshot.from_json(accepted)
            assert decoded.book == owner.book
            next_receipt = execute(before.book, panel, proposed, schedule, request=request)
            assert next_receipt.status == "executed" and next_receipt.net_value.fraction == 1
            oversized = before.model_dump(mode="json")
            oversized["receipts"].append(next_receipt.model_dump(mode="json"))
            rejected = wire(oversized)
            assert len(rejected.encode()) > 16 * 1024**2 - 1
            with pytest.raises(ValueError, match="byte"):
                BookSnapshot.from_json(rejected)
            with pytest.raises(ValueError, match="byte"):
                owner.deliver(panel, proposed, schedule, request=request)
            assert owner.export_state() is before
            assert wire(owner.export_state().model_dump(mode="json")) == accepted
            return
        assert receipt.status == "executed" and receipt.net_value.fraction == 1
    raise AssertionError("The complete real history must reach its byte ceiling before128receipts")


def test_forks_share_only_immutable_history_and_wrong_owner_restore_is_atomic():
    owner = BookSession(book())
    staged = owner.fork()
    staged.deliver(prices(), target(), fees(), request=ref("independent"))
    assert owner.book.sequence == 0 and staged.book.sequence == 1
    other = Book.model_validate(book().model_dump() | {"cash": Exact.of(200)})
    with pytest.raises(ValueError, match="another"):
        owner.restore_state(BookSession(other).export_state())
    assert owner.book.sequence == 0


def test_backwards_clock_is_refused_without_lost_state():
    owner = BookSession(book())
    owner.deliver(prices(day=2), target(), fees(), request=ref("first"))
    original = owner.export_state().to_json()
    with pytest.raises(ValueError, match="rewind"):
        owner.deliver(prices(day=1), target(), fees(period="later"), request=ref("backwards"))
    assert owner.export_state().to_json() == original
