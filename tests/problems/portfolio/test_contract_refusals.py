"""Public-wire financial integrity regressions with independent arithmetic.

All inputs are authored synthetic credits. The opening hand trace solves
W + W/10 + 1 = 100, hence W=90, bought units=9, and paid fee=9.
Adversarial wire records preserve valid nested types and, where applicable,
recompute the declared input identity so an identity shortcut cannot replace
the accounting checks. A digest is not source authentication.
"""

import json
from datetime import UTC, datetime, timedelta
from fractions import Fraction

import pytest

from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.problems.portfolio import (
    AssetWeight,
    Book,
    BookSession,
    BookSnapshot,
    Exact,
    Execution,
    Fees,
    Position,
    PricePanel,
    PriceQuote,
    Target,
    Valuation,
    execute,
    mark,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _ref(name, version="1"):
    return Reference(id=name, version=version)


def _book(*, cash=100, units=(0, 0), name="contract-funded-path"):
    return Book(
        schema_version=1,
        reference=_ref(name),
        cash=Exact.of(cash),
        positions=tuple(
            Position(asset=_ref(asset), units=Exact.of(amount))
            for asset, amount in zip(("asset-a", "asset-b"), units, strict=True)
        ),
    )


def _prices(*, day=0, asset_version="1"):
    at = NOW + timedelta(days=day)
    return PricePanel(
        schema_version=1,
        source=_ref("constant-authored-price-path"),
        source_fingerprint=fingerprint({"prices": [10, 20], "version": asset_version}),
        at=at,
        quotes=tuple(
            PriceQuote(
                asset=_ref(asset, asset_version),
                value=Exact.of(price),
                status="observed",
                event_at=at,
                available_at=at,
            )
            for asset, price in (("asset-a", 10), ("asset-b", 20))
        ),
    )


def _fees(*, period="opening", unavailable=False):
    return Fees(
        schema_version=1,
        reference=_ref("explicit-executed-notional-fees"),
        period=_ref(period),
        declared_at=NOW,
        proportional=None if unavailable else Exact.of("1/10"),
        recurring=None if unavailable else Exact.of(1),
        status="unavailable" if unavailable else "observed",
    )


def _target(*, asset_version="1"):
    return Target(
        schema_version=1,
        reference=_ref("fully-funded-a"),
        weights=tuple(
            AssetWeight(asset=_ref(asset, asset_version), value=Exact.of(weight))
            for asset, weight in (("asset-a", 1), ("asset-b", 0))
        ),
    )


def _wire(value):
    return json.loads(value.to_json())


def _amount(value):
    rational = Fraction(value)
    return {"numerator": rational.numerator, "denominator": rational.denominator}


def _replace(document, path, value):
    current = document
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _settled():
    return execute(_book(), _prices(), _target(), _fees(), request=_ref("original-opening"))


def _rebind_inputs(document):
    """Use only public content identities, never a settlement/oracle helper."""
    book = Book.from_json(json.dumps(document["before"]))
    prices = PricePanel.from_json(json.dumps(document["valuation"]["prices"]))
    target = None if document["target"] is None else Target.from_json(json.dumps(document["target"]))
    fees = Fees.from_json(json.dumps(document["fees"]))
    document["input_fingerprint"] = fingerprint(
        {
            "book": book.fingerprint,
            "prices": prices.fingerprint,
            "target": None if target is None else target.fingerprint,
            "fees": fees.fingerprint,
            "request": document["request"],
        }
    )


def test_independent_post_charge_hand_trace_is_valid_before_receipt_tampering():
    receipt = _settled()
    assert receipt.status == "executed"
    assert receipt.gross_value == Exact.of(100)
    assert receipt.net_value == Exact.of(90)
    assert receipt.turnover == Exact.of(90)
    assert receipt.proportional_paid == Exact.of(9)
    assert receipt.recurring_paid == Exact.of(1)
    assert receipt.after.cash == Exact.of(0)
    assert tuple(position.units.fraction for position in receipt.after.positions) == (Fraction(9), Fraction())
    assert tuple(trade.notional.fraction for trade in receipt.trades) == (Fraction(90), Fraction())
    assert receipt.after.sequence == 1
    assert receipt.after.charged_periods == (_ref("opening"),)
    assert Execution.from_json(receipt.to_json()).to_json() == receipt.to_json()


@pytest.mark.parametrize(
    "path,value",
    [
        (("quotes", 0, "value"), None),
        (("quotes", 0, "status"), "unknown"),
        (("quotes", 0, "value"), _amount(0)),
        (("quotes", 0, "event_at"), (NOW + timedelta(seconds=1)).isoformat()),
        (("quotes", 0, "available_at"), (NOW + timedelta(seconds=1)).isoformat()),
        (("quotes", 0, "unit"), "USD"),
        (("quotes", 1, "asset"), {"id": "asset-a", "version": "1"}),
    ],
    ids=[
        "observed-null",
        "unknown-with-value",
        "zero-price",
        "event-after-availability",
        "future-quote",
        "wrong-unit",
        "duplicate-asset",
    ],
)
def test_public_price_panel_reader_refuses_inconsistent_availability_units_or_clocks(path, value):
    document = _wire(_prices())
    _replace(document, path, value)
    encoded = json.dumps(document)
    with pytest.raises(ValueError):
        PricePanel.from_json(encoded)


@pytest.mark.parametrize(
    "status,proportional,recurring",
    [
        ("observed", None, _amount(1)),
        ("observed", _amount("1/10"), None),
        ("unavailable", _amount("1/10"), None),
        ("unavailable", None, _amount(1)),
        ("observed", _amount(1), _amount(1)),
        ("observed", _amount("1/10"), _amount(-1)),
    ],
    ids=[
        "missing-rate",
        "missing-recurring",
        "unavailable-rate-present",
        "unavailable-recurring-present",
        "rate-one",
        "negative-recurring",
    ],
)
def test_fee_reader_never_relabels_partial_or_unavailable_charges_as_free(status, proportional, recurring):
    document = _wire(_fees())
    document.update(status=status, proportional=proportional, recurring=recurring)
    encoded = json.dumps(document)
    with pytest.raises(ValueError):
        Fees.from_json(encoded)


@pytest.mark.parametrize(
    "kind,path,value",
    [
        ("book", ("cash",), _amount(-1)),
        ("book", ("positions", 0, "units"), _amount(-1)),
        ("book", ("positions", 1, "asset"), {"id": "asset-a", "version": "1"}),
        ("book", ("sequence",), 1),
        ("target", ("weights", 0, "value"), _amount(-1)),
        ("target", ("weights", 0, "value"), _amount(2)),
        ("target", ("weights", 1, "value"), _amount("1/2")),
        ("target", ("weights", 1, "asset"), {"id": "asset-a", "version": "1"}),
    ],
    ids=[
        "negative-cash",
        "short-position",
        "duplicate-position",
        "unsettled-frontier",
        "short-target",
        "leveraged-coordinate",
        "negative-implied-cash",
        "duplicate-target",
    ],
)
def test_book_and_target_wire_readers_refuse_unfunded_or_ambiguous_inputs(kind, path, value):
    document = _wire(_book() if kind == "book" else _target())
    _replace(document, path, value)
    encoded = json.dumps(document)
    reader = Book.from_json if kind == "book" else Target.from_json
    with pytest.raises(ValueError):
        reader(encoded)


def test_book_reader_refuses_double_charging_one_period_under_a_new_version():
    document = _wire(_book())
    document["sequence"] = 2
    document["charged_periods"] = [
        {"id": "same-period", "version": "1"},
        {"id": "same-period", "version": "2"},
    ]
    encoded = json.dumps(document)
    with pytest.raises(ValueError, match="charged twice"):
        Book.from_json(encoded)


@pytest.mark.parametrize("wrong_catalogue", ["prices", "target"])
def test_same_asset_labels_with_other_versions_refuse_delivery_atomically(wrong_catalogue):
    owner = BookSession(_book())
    prices = _prices(asset_version="2" if wrong_catalogue == "prices" else "1")
    target = _target(asset_version="2" if wrong_catalogue == "target" else "1")
    schedule = _fees()
    request = _ref("catalogue-request")
    before = owner.export_state().to_json()
    with pytest.raises(ValueError, match="catalogue"):
        owner.deliver(prices, target, schedule, request=request)
    assert owner.export_state().to_json() == before


@pytest.mark.parametrize("reason", ["missing_price", "missing_fee", "ineligible_fee", "stale_price"])
def test_refusal_retains_unknown_charges_and_deduplicates_the_original_negative_receipt(reason):
    prices_document, fees_document = _wire(_prices()), _wire(_fees())
    if reason == "missing_price":
        prices_document["quotes"][0].update(status="unavailable", value=None)
    elif reason == "missing_fee":
        fees_document.update(status="unavailable", proportional=None, recurring=None)
    elif reason == "ineligible_fee":
        fees_document["declared_at"] = (NOW + timedelta(seconds=1)).isoformat()
    else:
        old = (NOW - timedelta(seconds=1)).isoformat()
        prices_document["quotes"][0].update(event_at=old, available_at=old)
    prices = PricePanel.from_json(json.dumps(prices_document))
    schedule = Fees.from_json(json.dumps(fees_document))
    initial, target, request = _book(), _target(), _ref("negative-original")
    owner = BookSession(initial)
    receipt = owner.deliver(prices, target, schedule, request=request)
    assert receipt.status == "refused"
    assert receipt.reason == reason
    assert receipt.after.to_json() == initial.to_json()
    assert receipt.net_value is None
    assert receipt.turnover is None
    assert receipt.proportional_paid is None
    assert receipt.recurring_paid is None
    retained = owner.export_state().to_json()
    assert owner.deliver(prices, target, schedule, request=request).to_json() == receipt.to_json()
    assert owner.export_state().to_json() == retained


def test_missing_quote_is_not_missing_wealth_for_zero_holdings_but_is_for_positive_holdings():
    document = _wire(_prices())
    document["quotes"][0].update(status="unknown", value=None)
    prices = PricePanel.from_json(json.dumps(document))
    positive = mark(_book(cash=0, units=(1, 0)), prices)
    assert positive.status == "unknown"
    assert positive.value is None
    assert positive.asset_values == (None, Exact.of(0))
    assert positive.drifted_weights == (None, None)
    empty = mark(_book(cash=0), prices)
    assert empty.status == "complete"
    assert empty.value == Exact.of(0)
    assert empty.asset_values == (Exact.of(0), Exact.of(0))
    assert empty.drifted_weights == (None, None)
    assert Valuation.from_json(empty.to_json()).to_json() == empty.to_json()


@pytest.mark.parametrize(
    "corruption", ["shape", "negative-asset", "negative-implied-cash", "missing-quote-positive-mark"]
)
def test_valuation_reader_refuses_false_financial_amounts_not_just_wrong_status(corruption):
    receipt = _settled()
    document = _wire(mark(receipt.after, _prices()))
    if corruption == "shape":
        document["drifted_weights"].pop()
    elif corruption == "negative-asset":
        document["asset_values"][0] = _amount(-1)
    elif corruption == "negative-implied-cash":
        document["value"] = _amount(89)
    else:
        document["prices"]["quotes"][0].update(status="unknown", value=None)
    encoded = json.dumps(document)
    with pytest.raises(ValueError):
        Valuation.from_json(encoded)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing-fee",
        "future-fee",
        "stale-price",
        "target-version",
        "trade-catalogue",
        "after-owner",
        "after-frontier",
        "invented-asset-mark",
        "invented-gross",
        "trade-notional",
        "invented-cash",
        "negative-net",
        "hold-with-trade",
        "wrong-target",
    ],
)
def test_execution_reader_rejects_semantic_forgery_even_with_rebound_input_identity(corruption):
    document = _wire(_settled())
    if corruption == "missing-fee":
        document["fees"].update(status="unavailable", proportional=None, recurring=None)
    elif corruption == "future-fee":
        document["fees"]["declared_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif corruption == "stale-price":
        old = (NOW - timedelta(seconds=1)).isoformat()
        document["valuation"]["prices"]["quotes"][0].update(status="stale", event_at=old, available_at=old)
    elif corruption == "target-version":
        document["target"]["weights"][0]["asset"]["version"] = "2"
    elif corruption == "trade-catalogue":
        document["trades"].reverse()
    elif corruption == "after-owner":
        document["after"]["reference"]["id"] = "another-funded-path"
    elif corruption == "after-frontier":
        document["after"]["sequence"] = 2
        document["after"]["charged_periods"].append({"id": "unearned-period", "version": "1"})
    elif corruption == "invented-asset-mark":
        document["valuation"]["asset_values"][0] = _amount(1)
        document["valuation"]["drifted_weights"][0] = _amount("1/100")
    elif corruption == "invented-gross":
        document["valuation"]["value"] = _amount(99)
        document["gross_value"] = _amount(99)
    elif corruption == "trade-notional":
        document["trades"][0]["notional"] = _amount(89)
    elif corruption == "invented-cash":
        # Local asset+cash agrees with the forged net, but creates one credit:
        # 91 + 9 + 1 != the actual original100.
        document["after"]["cash"] = _amount(1)
        document["net_value"] = _amount(91)
    elif corruption == "negative-net":
        document["net_value"] = _amount(-1)
    elif corruption == "hold-with-trade":
        document["target"] = None
    else:
        document["target"]["weights"][0]["value"] = _amount("1/2")
    _rebind_inputs(document)
    encoded = json.dumps(document)
    with pytest.raises(ValueError):
        Execution.from_json(encoded)


@pytest.mark.parametrize("join", ["input-digest", "valuation-book"])
def test_receipt_identity_joins_remain_required_beside_exact_financial_arithmetic(join):
    document = _wire(_settled())
    wrong_digest = "sha256:" + "f" * 64
    if join == "input-digest":
        document["input_fingerprint"] = wrong_digest
    else:
        document["valuation"]["book_fingerprint"] = wrong_digest
    encoded = json.dumps(document)
    with pytest.raises(ValueError):
        Execution.from_json(encoded)


@pytest.mark.parametrize("corruption", ["observed-zero-net", "changed-book", "wrong-reason", "false-executed"])
def test_refused_receipt_cannot_be_promoted_to_an_observed_zero_or_success(corruption):
    receipt = execute(_book(), _prices(), _target(), _fees(unavailable=True), request=_ref("missing-fees"))
    document = _wire(receipt)
    if corruption == "observed-zero-net":
        document["net_value"] = _amount(0)
    elif corruption == "changed-book":
        document["after"]["cash"] = _amount(99)
    elif corruption == "wrong-reason":
        document["reason"] = "insufficient_funding"
    else:
        document["status"] = "executed"
        document["reason"] = "settled"
    encoded = json.dumps(document)
    with pytest.raises(ValueError):
        Execution.from_json(encoded)


@pytest.mark.parametrize("hold", [False, True], ids=["target-with-feasible-purchase", "hold-with-sufficient-cash"])
def test_actually_feasible_funding_cannot_be_relabelled_as_an_insufficient_funding_receipt(hold):
    target = None if hold else _target()
    receipt = execute(_book(), _prices(), target, _fees(), request=_ref("feasible-original"))
    assert receipt.status == "executed"
    document = _wire(receipt)
    document.update(status="refused", reason="insufficient_funding", trades=[])
    document["after"] = document["before"]
    for field in ("net_value", "turnover", "proportional_paid", "recurring_paid"):
        document[field] = None
    encoded = json.dumps(document)
    with pytest.raises(ValueError, match="funding is feasible"):
        Execution.from_json(encoded)


@pytest.mark.parametrize("corruption", ["duplicate-request", "capital-reset", "rewound-refusals"])
def test_snapshot_reader_replays_original_delivery_joins_not_only_individual_receipts(corruption):
    initial = _book()
    first = execute(initial, _prices(), _target(), _fees(), request=_ref("first"))
    if corruption == "duplicate-request":
        receipts = [first, first]
    elif corruption == "capital-reset":
        # The second receipt is individually valid, but starts again from100
        # cash instead of the original successor with9units and zero cash.
        second = execute(initial, _prices(day=1), _target(), _fees(period="next"), request=_ref("second"))
        receipts = [first, second]
    else:
        later = execute(initial, _prices(day=1), None, _fees(period="later", unavailable=True), request=_ref("later"))
        earlier = execute(initial, _prices(), None, _fees(unavailable=True), request=_ref("earlier"))
        receipts = [later, earlier]
    document = {"schema_version": 1, "initial": _wire(initial), "receipts": [_wire(row) for row in receipts]}
    encoded = json.dumps(document)
    with pytest.raises(ValueError):
        BookSnapshot.from_json(encoded)


@pytest.mark.parametrize("other", ["identity", "capital"])
def test_restore_requires_the_complete_original_book_and_refuses_atomically(other):
    owner = BookSession(_book())
    owner.deliver(_prices(), _target(), _fees(), request=_ref("original"))
    foreign = _book(name="foreign-path") if other == "identity" else _book(cash=99)
    foreign_snapshot = BookSession(foreign).export_state()
    before = owner.export_state().to_json()
    with pytest.raises(ValueError, match="another original owner"):
        owner.restore_state(foreign_snapshot)
    assert owner.export_state().to_json() == before


def test_backwards_new_refused_delivery_cannot_rewind_the_retained_session_clock():
    owner = BookSession(_book())
    later_prices, later_fees, request = _prices(day=1), _fees(period="later", unavailable=True), _ref("later")
    original = owner.deliver(later_prices, None, later_fees, request=request)
    assert original.status == "refused"
    before = owner.export_state().to_json()
    earlier_prices, earlier_fees, earlier_request = _prices(), _fees(unavailable=True), _ref("earlier")
    with pytest.raises(ValueError, match="clocks cannot rewind"):
        owner.deliver(earlier_prices, None, earlier_fees, request=earlier_request)
    assert owner.export_state().to_json() == before
    assert owner.deliver(later_prices, None, later_fees, request=request).to_json() == original.to_json()
