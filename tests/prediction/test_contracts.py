"""Strict ridge schemas, numerical refusals and independently authored goldens."""

import json
from decimal import Decimal

import pytest

from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.units import UnitSemantics
from coleman.prediction import RidgeConfig, RidgeInput, RidgeLimits, RidgePayload, RidgeSample
from coleman.state.contracts import AlgorithmState
from tests.prediction._support import FIXTURES, config, features, owner, ref, sample


@pytest.mark.parametrize("kind", ["config", "initial-state"])
def test_hand_authored_version_one_fixtures_roundtrip_without_mapper_generated_expectations(kind):
    """Raw declaration and lambdaI/zero sufficient statistics have authored bindings."""
    raw = (FIXTURES / f"{kind}.json").read_text().rstrip("\n")
    contract = RidgeConfig if kind == "config" else AlgorithmState[RidgePayload]
    for content in (raw, raw.encode()):
        assert contract.from_json(content).to_json() == raw
    actual = config() if kind == "config" else owner().export_state()
    assert actual.to_json() == raw


@pytest.mark.parametrize("change", ["missing", "future", "bool", "extra"])
def test_unknown_coerced_or_missing_schema_versions_never_migrate(change):
    """Closed version-one roots do not coerce booleans or erase extra fields."""
    raw = json.loads((FIXTURES / "config.json").read_text())
    if change == "missing":
        del raw["schema_version"]
    elif change == "extra":
        raw["forgotten_data"] = 0
    else:
        raw["schema_version"] = True if change == "bool" else 2
    with pytest.raises(ValueError):
        RidgeConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), Decimal("1e-9999")])
def test_regularization_is_fixed_strictly_positive_finite_and_nonunderflowing(rate):
    """Invalid lambda cannot create an unregularized or silently rounded model."""
    with pytest.raises(ValueError):
        config(regularization=rate)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1", Decimal("1e-9999")])
def test_nonfinite_coerced_or_erased_feature_values_refuse_before_the_owner(value):
    """Typed ingress preserves finite nonzero meaning before any model call."""
    model = owner()
    before = model.export_state().to_json()
    with pytest.raises(ValueError):
        features(model, (value, 0.0))
    assert model.export_state().to_json() == before


@pytest.mark.parametrize(
    "capacity,value",
    [(False, float("nan")), (False, float("inf")), (False, -float("inf")), (False, True), (True, 1e101)],
)
def test_nonfinite_or_out_of_capacity_labels_never_change_the_actual_owner(capacity, value):
    """Unknown numerical labels do not become zero targets or partial updates."""
    model = owner()
    before = model.export_state().to_json()
    if capacity:
        incoming = sample(model, (1.0, 0.0), value)
        with pytest.raises(ValueError, match="magnitude"):
            model.append(incoming)
    else:
        with pytest.raises(ValueError):
            sample(model, (1.0, 0.0), value)
    assert model.export_state().to_json() == before


@pytest.mark.parametrize("value", [-2.0, 2.0])
def test_declared_feature_bounds_are_checked_before_computing_a_forecast(value):
    """Equal feature dimensions do not erase the selected schema's bounds."""
    raw = config().model_dump(mode="json")
    raw["features"][0]["minimum"] = -1.0
    raw["features"][0]["maximum"] = 1.0
    model = owner(RidgeConfig.model_validate(raw))
    before = model.export_state().to_json()
    incoming = features(model, (value, 0.0))
    with pytest.raises(ValueError, match="schema"):
        model.predict(incoming)
    assert model.export_state().to_json() == before


@pytest.mark.parametrize("values", [(), (0.0,), (0.0, 1.0, 2.0), (1e101, 0.0)])
def test_incompatible_dimensions_and_profile_input_magnitudes_never_change_state(values):
    """The selected owner refuses foreign shape or magnitude without fitting."""
    model = owner()
    before = model.export_state().to_json()
    if not values:
        with pytest.raises(ValueError):
            features(model, values)
    else:
        incoming = features(model, values)
        with pytest.raises(ValueError):
            model.predict(incoming)
    assert model.export_state().to_json() == before


@pytest.mark.parametrize("status", ["unknown", "unavailable", "not_applicable"])
def test_missing_target_never_turns_into_zero_training_data(status):
    """Every missingness status stays distinct from an observed zero label."""
    model = owner()
    missing = Scalar(spec=model.config.target, value=None, status=status)
    original = features(model, (1.0, 0.0))
    with pytest.raises(ValueError, match="observed target"):
        RidgeSample(schema_version=1, input=original, target=missing)
    assert model.sample_count == 0


def test_wrong_target_units_and_changed_preprocessing_binding_refuse_atomically():
    """Equal shapes do not permit a different target or preprocessing definition."""
    model = owner()
    before = model.export_state().to_json()
    wrong = ScalarSpec(
        dimension=model.config.target.dimension,
        role="objective",
        semantics=UnitSemantics(unit="s", meaning="actual-elapsed-seconds-v1"),
    )
    incoming = RidgeSample(
        schema_version=1,
        input=features(model, (1.0, 0.0)),
        target=Scalar(spec=wrong, value=0.0, status="observed"),
    )
    with pytest.raises(ValueError, match="target schema"):
        model.append(incoming)
    other = owner(config(regularization=2.0))
    stale = features(other, (1.0, 0.0))
    with pytest.raises(ValueError, match="configuration schema"):
        model.predict(stale)
    assert model.export_state().to_json() == before


@pytest.mark.parametrize("change", ["role", "unit", "duplicate", "count"])
def test_features_are_an_explicit_fixed_dimensionless_schema(change):
    """Roles, units, versions and finite dimensions are declared before fitting."""
    raw = config().model_dump(mode="json")
    if change == "role":
        raw["features"][0]["role"] = "prediction"
    elif change == "unit":
        raw["features"][0]["semantics"]["unit"] = "s"
    elif change == "duplicate":
        raw["features"][1] = raw["features"][0]
    else:
        raw["features"] *= 9
    with pytest.raises(ValueError):
        RidgeConfig.model_validate(raw)


def test_declared_label_bounds_do_not_clip_or_bound_the_computed_prediction():
    """The forecast is a separately typed value, not an observed/clipped label."""
    raw = config().model_dump(mode="json")
    raw["target"]["minimum"] = -1.0
    raw["target"]["maximum"] = 1.0
    model = owner(RidgeConfig.model_validate(raw))
    assert model.append(sample(model, (1.0, 0.0), 1.0)).status == "accepted"
    forecast = model.predict(features(model, (10.0, 0.0)))
    assert forecast.value.value is not None
    assert forecast.value.value > 1.0
    assert forecast.value.spec.role == "prediction"
    assert forecast.value.spec.maximum is None
    assert forecast.value.spec.semantics == model.config.target.semantics


@pytest.mark.parametrize("contract", [RidgeConfig, RidgePayload, RidgeInput, RidgeLimits])
def test_all_new_json_roots_cap_mutable_bytes_before_copy(contract):
    """The hard16MiB bytearray ceiling precedes conversion or JSON decoding."""
    incoming = bytearray(16 * 1024**2 + 1)
    with pytest.raises(ValueError, match="before copying"):
        contract.model_validate_json(incoming)


def test_callback_bearing_json_duplicate_keys_depth_and_relaxed_readers_refuse():
    """New records retain shared strict ingress and execute no conversion hook."""
    calls = []

    class Hostile(bytearray):
        """Make a conversion callback observable if the reader were to call it."""

        def __bytes__(self):
            """Record a forbidden conversion; the test requires zero invocations."""
            calls.append("copied")
            return b"{}"

    with pytest.raises(TypeError, match="built-in"):
        RidgeConfig.model_validate_json(Hostile(b"{}"))
    with pytest.raises(ValueError, match="Duplicate"):
        RidgeConfig.from_json('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="depth"):
        RidgeConfig.from_json("[" * 65 + "0" + "]" * 65)
    with pytest.raises(ValueError, match="strict"):
        RidgeConfig.model_validate_json((FIXTURES / "config.json").read_bytes(), strict=False)
    assert not calls


def test_signed_zero_input_and_target_stay_explicit_without_becoming_missing():
    """Known signed zeros retain their actual source bits before append."""
    model = owner()
    original = sample(model, (-0.0, 0.0), -0.0)
    assert original.input.values[0].hex() == "-0x0.0p+0"
    assert original.target.value is not None
    assert original.target.value.hex() == "-0x0.0p+0"
    assert model.append(original).status == "accepted"
    assert model.sample_count == 1


def test_schema_versions_and_owner_ids_are_not_inferred_from_names():
    """Only the selected full configuration joins features to an actual owner."""
    model = owner()
    altered = RidgeInput(schema_version=1, config=config(regularization=2.0).binding, values=(1.0, 0.0))
    with pytest.raises(ValueError, match="configuration schema"):
        model.predict(altered)
    assert model.binding.owner == ref("ci-ridge-owner")
