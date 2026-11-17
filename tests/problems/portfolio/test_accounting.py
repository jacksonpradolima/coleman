"""Independent rational funded-book arithmetic and contract boundaries."""

import json
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest

from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.problems.portfolio import (
    AssetWeight,
    Book,
    Exact,
    Execution,
    Fees,
    Position,
    PricePanel,
    PriceQuote,
    Target,
    execute,
    mark,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
ASSETS = (Reference(id="synthetic-a", version="1"), Reference(id="synthetic-b", version="1"))


def ref(name):
    return Reference(id=name, version="1")


def book():
    return Book(
        schema_version=1,
        reference=ref("own-path"),
        cash=Exact.of(100),
        positions=tuple(Position(asset=asset, units=Exact.of(0)) for asset in ASSETS),
    )


def prices(values=(10, 20), *, day=0, statuses=("observed", "observed")):
    return PricePanel(
        schema_version=1,
        source=ref("authored-prices"),
        source_fingerprint=fingerprint({"authored": list(values)}),
        at=NOW + timedelta(days=day),
        quotes=tuple(
            PriceQuote(
                asset=asset,
                value=None if value is None else Exact.of(value),
                status=status,
                event_at=NOW if status == "stale" else NOW + timedelta(days=day),
                available_at=NOW if status == "stale" else NOW + timedelta(days=day),
            )
            for asset, value, status in zip(ASSETS, values, statuses, strict=True)
        ),
    )


def fees(*, rate="1/100", recurring=0, period="initial"):
    return Fees(
        schema_version=1,
        reference=ref("explicit-fees"),
        period=ref(period),
        declared_at=NOW,
        proportional=Exact.of(rate),
        recurring=Exact.of(recurring),
        status="observed",
    )


def target(values=("1/2", "1/2")):
    return Target(
        schema_version=1,
        reference=ref("recipe-target"),
        weights=tuple(
            AssetWeight(asset=asset, value=Exact.of(value)) for asset, value in zip(ASSETS, values, strict=True)
        ),
    )


def test_independent_post_charge_target_arithmetic_and_drift():
    first = execute(book(), prices(), target(), fees(), request=ref("initial"))
    assert first.status == "executed"
    assert first.net_value.fraction == Fraction(10000, 101)
    assert first.proportional_paid.fraction == Fraction(100, 101)
    assert tuple(position.units.fraction for position in first.after.positions) == (
        Fraction(500, 101),
        Fraction(250, 101),
    )
    second_prices = prices((12, 18), day=1)
    valuation = mark(first.after, second_prices)
    assert valuation.value.fraction == Fraction(10500, 101)
    assert tuple(weight.fraction for weight in valuation.drifted_weights) == (Fraction(4, 7), Fraction(3, 7))
    hold = execute(first.after, second_prices, None, fees(period="next"), request=ref("hold"))
    assert hold.turnover.fraction == hold.proportional_paid.fraction == 0
    assert hold.net_value.fraction == Fraction(10500, 101)
    rebalance = execute(first.after, second_prices, target(), fees(period="next"), request=ref("rebalance"))
    assert rebalance.turnover.fraction == Fraction(1500, 101)
    assert rebalance.proportional_paid.fraction == Fraction(15, 101)
    assert rebalance.net_value.fraction == Fraction(10485, 101)
    assert tuple(position.units.fraction for position in rebalance.after.positions) == (
        Fraction(3495, 808),
        Fraction(1165, 404),
    )
    assert rebalance.after.cash.fraction == 0
    assert Execution.from_json(rebalance.to_json()).to_json() == rebalance.to_json()


def test_authored_version_one_golden_has_independent_funded_arithmetic():
    fixture = Path(__file__).parent / "fixtures" / "v1" / "funded-book.json"
    expected = Book.from_json(fixture.read_bytes())
    actual = execute(book(), prices(), target(), fees(), request=ref("initial"))
    assert actual.after.to_json() == expected.to_json()
    assert expected.to_json() + "\n" == fixture.read_text()


def test_opposing_recipe_target_changes_net_before_executed_book_charges():
    # Independent arithmetic: two equally weighted recipes switch (1,0)->(0,1)
    # and (0,1)->(1,0). Each recipe trades its full holdings in both directions;
    # their combined target stays (1/2,1/2). The actual deployed book has no
    # trade. Modeling each virtual recipe's charges is a separate calculation.
    funded = execute(book(), prices(), target(), fees(rate=0), request=ref("initial-net"))
    deployed = execute(funded.after, prices(day=1), target(), fees(period="net"), request=ref("netted"))
    assert deployed.turnover.fraction == deployed.proportional_paid.fraction == 0
    assert deployed.net_value.fraction == 100
    virtual_turnover = Fraction()
    virtual_fees = Fraction()
    for index, (opening, closing) in enumerate((((1, 0), (0, 1)), ((0, 1), (1, 0)))):
        initial = execute(book(), prices(), target(opening), fees(rate=0), request=ref("virtual-open-" + str(index)))
        changed = execute(
            initial.after, prices(day=1), target(closing), fees(period="virtual"), request=ref("switch-" + str(index))
        )
        # Sell100 and buy9900/101, pay200/101. These are virtual costs, not
        # additional executed charges on the independently funded mixture.
        assert changed.turnover.fraction == Fraction(20000, 101)
        assert changed.proportional_paid.fraction == Fraction(200, 101)
        virtual_turnover += changed.turnover.fraction / 2
        virtual_fees += changed.proportional_paid.fraction / 2
    assert virtual_turnover == Fraction(20000, 101)
    assert virtual_fees == Fraction(200, 101)


def test_legacy_domain_record_reader_keeps_existing_units_and_meaning():
    # The new namespace does not add a new interpretation to a retained public
    # TCPCI result. Exercise its actual existing reader beside the new codec.
    from coleman.results.domain import DomainRecord

    root = Path(__file__).parents[2]
    payload = (root / "fixtures" / "domain-results" / "record-v1.json").read_text()
    original = DomainRecord.from_json(payload)
    assert DomainRecord.from_json(original.to_json()).to_json() == original.to_json()


def test_inconsistent_book_frontier_rejects():
    with pytest.raises(ValueError, match="frontier"):
        Book.model_validate(book().model_dump() | {"sequence": 1})


def test_recurring_fee_is_separate_and_does_not_invent_a_trade():
    result = execute(book(), prices(), None, fees(recurring=2), request=ref("hold"))
    assert result.net_value.fraction == 98
    assert result.recurring_paid.fraction == 2
    assert result.turnover.fraction == result.proportional_paid.fraction == 0
    with pytest.raises(ValueError, match="consumed"):
        execute(result.after, prices(), None, fees(recurring=2), request=ref("repeat"))


@pytest.mark.parametrize(
    "values,statuses,reason",
    [((None, 20), ("unavailable", "observed"), "missing_price"), ((10, 20), ("stale", "observed"), "stale_price")],
)
def test_missing_or_carried_price_refuses_without_free_execution(values, statuses, reason):
    result = execute(book(), prices(values, statuses=statuses), target(), fees(), request=ref("missing"))
    assert result.status == "refused" and result.reason == reason
    assert result.after.to_json() == result.before.to_json()
    assert result.turnover is result.proportional_paid is result.net_value is None


def test_unknown_held_asset_price_keeps_total_unknown_and_stale_mark_keeps_age():
    first = execute(book(), prices(), target(), fees(), request=ref("first"))
    unknown = mark(first.after, prices((None, 20), statuses=("unknown", "observed"), day=5))
    assert unknown.status == "unknown" and unknown.value is None
    assert unknown.asset_values[0] is None and all(value is None for value in unknown.drifted_weights)
    carried = mark(first.after, prices(day=5, statuses=("stale", "observed")))
    assert carried.status == "stale" and carried.prices.quotes[0].event_at == NOW
    assert carried.value is not None


def test_missing_fee_is_not_zero_and_unfunded_charge_is_infeasible():
    unavailable = Fees(
        schema_version=1,
        reference=ref("missing-fee"),
        period=ref("period"),
        declared_at=NOW,
        proportional=None,
        recurring=None,
        status="unavailable",
    )
    assert execute(book(), prices(), target(), unavailable, request=ref("missing")).reason == "missing_fee"
    refused = execute(book(), prices(), target(), fees(recurring=101), request=ref("unfunded"))
    assert refused.reason == "insufficient_funding" and refused.net_value is None


def test_observed_old_quote_is_only_a_stale_mark_and_future_fee_cannot_trade():
    current = prices(day=2)
    changed = current.model_dump()
    changed["quotes"][0]["event_at"] = NOW
    changed["quotes"][0]["available_at"] = NOW
    old = PricePanel.model_validate(changed)
    first = execute(book(), prices(), target(), fees(), request=ref("first"))
    valuation = mark(first.after, old)
    assert valuation.status == "stale"
    assert execute(first.after, old, target(), fees(period="later"), request=ref("aged")).reason == "stale_price"
    future = Fees.model_validate(fees().model_dump() | {"declared_at": NOW + timedelta(days=1)})
    refused = execute(book(), prices(), target(), future, request=ref("future-fee"))
    assert refused.reason == "ineligible_fee" and refused.proportional_paid is None


@pytest.mark.parametrize(
    "payload",
    [
        {"numerator": 0, "denominator": 2},
        {"numerator": 1, "denominator": 0},
        {"numerator": 1.0},
        {"numerator": float("nan")},
        {"numerator": True},
        {"numerator": 1, "denominator": 1 << 4097},
    ],
)
def test_noncanonical_nonfinite_or_excessive_exact_values_reject(payload):
    with pytest.raises(ValueError):
        Exact.model_validate(payload)


def test_negative_signed_trade_valid_but_negative_cash_weights_or_future_versions_reject():
    assert Exact.of("-3/2").fraction == Fraction(-3, 2)
    with pytest.raises(ValueError):
        AssetWeight(asset=ASSETS[0], value=Exact.of(-1))
    with pytest.raises(ValueError):
        Book.from_json(book().to_json().replace('"schema_version":1', '"schema_version":2'))
    with pytest.raises(ValueError):
        PriceQuote(asset=ASSETS[0], value=Exact.of(10), status="observed", event_at=NOW, available_at=NOW, unit="USD")
    with pytest.raises(TypeError):
        Exact.of(0.1)


def test_deterministic_identity_and_round_trip():
    initial = book()
    assert Book.from_json(initial.to_json()).fingerprint == initial.fingerprint
    payload = json.loads(initial.to_json())
    reordered = json.dumps(dict(reversed(tuple(payload.items()))))
    assert Book.from_json(reordered).to_json() == initial.to_json()
    assert initial.cash.fraction == 100


@pytest.mark.parametrize(
    "field,number", [("net_value", 1000000), ("proportional_paid", -1), ("recurring_paid", 12), ("turnover", 0)]
)
def test_direct_execution_reader_rejects_false_wealth_or_paid_charges(field, number):
    result = execute(book(), prices(), target(), fees(), request=ref("actual"))
    payload = json.loads(result.to_json())
    payload[field] = {"numerator": number, "denominator": 1}
    with pytest.raises(ValueError):
        Execution.from_json(json.dumps(payload))


def test_direct_valuation_reader_rejects_missingness_and_false_drift_claims():
    from coleman.problems.portfolio import Valuation

    result = execute(book(), prices(), target(), fees(), request=ref("actual"))
    payload = json.loads(mark(result.after, prices((12, 18), day=1)).to_json())
    payload["status"] = "unknown"
    with pytest.raises(ValueError):
        Valuation.from_json(json.dumps(payload))
    payload["status"] = "complete"
    payload["drifted_weights"][0] = {"numerator": 1, "denominator": 2}
    with pytest.raises(ValueError):
        Valuation.from_json(json.dumps(payload))


def test_scaling_capital_preserves_weights_and_scales_paid_costs():
    # Exact metamorphic check across varying prices, targets and proportional fees.
    for index in range(1, 17):
        quote = prices((index + 1, 20 - index))
        chosen = target((str(Fraction(index, 20)), str(Fraction(20 - index, 20))))
        small = execute(book(), quote, chosen, fees(rate=str(Fraction(index, 100))), request=ref("small"))
        large_book = Book.model_validate(book().model_dump() | {"cash": Exact.of(700)})
        large = execute(large_book, quote, chosen, fees(rate=str(Fraction(index, 100))), request=ref("large"))
        assert large.net_value.fraction == 7 * small.net_value.fraction
        assert large.proportional_paid.fraction == 7 * small.proportional_paid.fraction
        for larger, smaller in zip(large.after.positions, small.after.positions, strict=True):
            assert larger.units.fraction == 7 * smaller.units.fraction
