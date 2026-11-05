"""Legacy aliases, artifact bytes and numerical parsing remain unchanged."""

import json
import math
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from coleman.core import base, primitives, temporal, units
from coleman.evaluation.protected.temporal import TemporalObservation, Timestamp
from coleman.results import domain, versioned
from coleman.selection import contracts

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_all_legacy_public_leaf_imports_are_identical_reexports():
    for name in ("Identifier", "Fingerprint", "Finite", "canonical_json", "decode_json", "fingerprint"):
        assert getattr(domain, name) is getattr(primitives, name)
    assert versioned.VersionedContract is base.VersionedContract
    assert contracts.Immutable is base.Immutable
    assert contracts.UnitSemantics is units.UnitSemantics
    assert Timestamp is temporal.Timestamp


def test_legacy_finite_still_accepts_documented_projection_including_underflow():
    adapter = TypeAdapter(domain.Finite)
    assert adapter.validate_python(2**53 + 1) == float(2**53)
    assert adapter.validate_python(Decimal("1e-400")) == 0.0
    assert math.copysign(1.0, adapter.validate_python(Decimal("-1e-400"))) == -1.0

    class LegacyNumber(versioned.VersionedContract):
        value: domain.Finite

    assert LegacyNumber.from_json('{"schema_version":1,"value":1e-400}').value == 0.0
    # Inherited legacy Pydantic reader still has its historical duplicate-key behavior.
    assert LegacyNumber.model_validate_json('{"schema_version":1,"value":0,"value":1}').value == 1.0


def test_legacy_domain_record_exact_golden_and_signed_zero_digest():
    path = FIXTURES / "domain-results" / "record-v1.json"
    record = domain.DomainRecord.from_json(path.read_bytes())
    assert record.to_json() + "\n" == path.read_text()
    digests = json.loads((path.parent / "digests-v1.json").read_bytes())
    assert record.digest == digests["content"]
    assert record.identity_digest == digests["identity"]
    assert domain.canonical_json({"value": 0.0}) == '{"value":0.0}'
    assert domain.canonical_json({"value": -0.0}) == '{"value":-0.0}'
    assert domain.fingerprint({"value": 0.0}) != domain.fingerprint({"value": -0.0})


def test_legacy_decimal_zero_and_timestamp_semantics_do_not_change():
    semantics = units.UnitSemantics(unit="s", meaning="duration-v1")
    zero = contracts.Quantity(value=Decimal("0"), semantics=semantics)
    signed = contracts.Quantity(value=Decimal("-0"), semantics=semantics)
    assert zero.model_dump_json() == signed.model_dump_json()
    assert zero.model_dump(mode="json")["value"] == "0"
    metadata = TemporalObservation.model_validate(
        {
            "observation_id": "a",
            "group_id": "g",
            "event_at": "2026-01-01T01:00:00+01:00",
            "available_at": "2026-01-01T00:00:00Z",
        }
    )
    assert metadata.event_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert metadata.target_end is None
    for value in (0, "0", datetime(2026, 1, 1)):
        with pytest.raises(ValueError):
            TypeAdapter(Timestamp).validate_python(value)
