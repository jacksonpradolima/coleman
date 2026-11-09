"""Versioned independent ridge inputs, outcomes and complete sufficient state."""

from __future__ import annotations

from struct import pack
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from coleman.core._wire import BoundedContract
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, Scalar, ScalarSpec
from coleman.core.primitives import Fingerprint, canonical_json
from coleman.objectives.bindings import DefinitionBinding, SemanticDefinition
from coleman.prediction._numerics import NumericalError, solve

MAXIMUM_BYTES = 16 * 1024**2
MAXIMUM_FEATURES = 16
MAXIMUM_SAMPLES = 1_048_576
Count = Annotated[int, Field(strict=True, ge=0, le=MAXIMUM_SAMPLES)]
Values = Annotated[tuple[Binary64, ...], Field(min_length=1, max_length=MAXIMUM_FEATURES)]
Matrix = Annotated[tuple[Values, ...], Field(min_length=1, max_length=MAXIMUM_FEATURES)]


class RidgeContract(BoundedContract):
    """Bound every new ridge JSON root to16MiB and lexical depth64."""

    _maximum_bytes = MAXIMUM_BYTES

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> Self:
        """Reject callback-bearing or oversized mutable JSON before copying.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete built-in UTF-8 data; no path, coercion or migration.
        **kwargs : object
            Shared guarded reader options; relaxed strictness/extra fields reject.

        Returns
        -------
        Self
            Concrete revalidated known-version content.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Wrong kind, byte/depth bound, unknown version or invalid content.
        """
        if type(json_data) not in {str, bytes, bytearray}:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Ridge JSON requires built-in str, bytes or bytearray")
        if type(json_data) is bytearray and len(json_data) > MAXIMUM_BYTES:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Ridge bytearray exceeds its byte ceiling before copying")
        return super().model_validate_json(json_data, **kwargs)


class RidgeLimits(RidgeContract):
    """Select finite append work and already-preprocessed input magnitude bounds.

    Attributes
    ----------
    maximum_samples : int, optional
        At most1,048,576 accepted single-sample appends, default65,536.
    maximum_input_magnitude : float, optional
        Inclusive absolute feature/label ceiling, at most1e100, default1e100.
        Units are each supplied field's units, not a new normalization operation.
    """

    maximum_samples: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_SAMPLES)] = 65_536
    maximum_input_magnitude: Annotated[Binary64, Field(gt=0, le=1e100)] = 1e100


class RidgeConfig(SemanticDefinition, RidgeContract):
    """Freeze one sum-squared-error ridge law and its ordered input schemas.

    Attributes
    ----------
    features : tuple of ScalarSpec
        One to16 unique ordered feature dimensions, role feature and unit1.
        Order is meaningful and never sorted. Values are supplied preprocessed.
    target : ScalarSpec
        Observed label role objective with its explicit units/meaning/bounds.
    horizon : Reference
        Declared label measurement horizon; this component reads no clock.
    preprocessing : DefinitionBinding
        Exact caller-owned dimensionless preprocessing definition. No fitting,
        centering or automatic scaling is performed here.
    regularization : float
        Positive fixed dimensionless lambda, at most1e100. Zero center, no
        forgetting, no implicit intercept; every declared coefficient is penalized.
    limits : RidgeLimits, optional
        Declared append/magnitude bounds. No old observations are evicted.
    profile, coefficient_convention : str, optional
        Named fixed binary64 Cholesky and all-coefficient penalty conventions.

    Notes
    -----
    A=lambda*I+sum(x*xT), b=sum(x*y), beta=solve(A,b). A constant feature may
    represent an intercept, but is penalized identically. The feature dimension
    count and all configuration bytes stay fixed across continuation. The1MiB
    configuration ceiling reserves room for the complete16MiB state envelope.
    """

    features: Annotated[tuple[ScalarSpec, ...], Field(min_length=1, max_length=MAXIMUM_FEATURES)]
    target: ScalarSpec
    horizon: Reference
    preprocessing: DefinitionBinding
    regularization: Annotated[Binary64, Field(gt=0, le=1e100)]
    limits: RidgeLimits = Field(default_factory=lambda: RidgeLimits(schema_version=1))
    profile: Literal["append-only-ridge-cholesky-binary64-v1"] = "append-only-ridge-cholesky-binary64-v1"
    coefficient_convention: Literal["no-implicit-intercept-all-penalized-zero-center-v1"] = (
        "no-implicit-intercept-all-penalized-zero-center-v1"
    )

    @model_validator(mode="after")
    def _schema(self) -> Self:
        """Require exact ordered dimensions and keep complete configuration bounded."""
        ordered_references(tuple(item.dimension for item in self.features))
        if any(item.role != "feature" or item.semantics.unit != "1" for item in self.features):
            raise ValueError("Ridge features require declared dimensionless feature schemas")
        if self.target.role != "objective":
            raise ValueError("Ridge target requires its declared objective schema")
        if len(canonical_json(self.model_dump(mode="json")).encode()) > 1024**2:
            raise ValueError("Ridge configuration exceeds its1MiB capacity")
        return self

    @property
    def prediction_spec(self) -> ScalarSpec:
        """Return an unbounded prediction-role quantity in the target units.

        Returns
        -------
        ScalarSpec
            Same dimension/version and units/meaning; no clipping to label bounds.
        """
        return ScalarSpec(dimension=self.target.dimension, semantics=self.target.semantics, role="prediction")


class RidgeInput(RidgeContract):
    """Retain the complete fixed schema binding and ordered preprocessed values.

    Attributes
    ----------
    config : DefinitionBinding
        Exact full RidgeConfig definition, including preprocessing and horizon.
    values : tuple of float
        One to16 finite binary64 values. The selected owner checks dimensions,
        bounds and schema identity before prediction or append.
    """

    config: DefinitionBinding
    values: Values


class RidgeSample(RidgeContract):
    """Supply one actual available label, with no implicit delivery idempotency.

    Attributes
    ----------
    input : RidgeInput
        Original supplied preprocessed feature values and configuration binding.
    target : Scalar
        Exact target schema, observed finite zero/negative/positive value.
        Unknown or unavailable labels cannot become training targets.

    Notes
    -----
    Eligibility, revisions, pending ownership and event deduplication belong to
    the host. This record captures no future labels, timestamps or source access.
    """

    input: RidgeInput
    target: Scalar

    @model_validator(mode="after")
    def _observed(self) -> Self:
        """Require a real supplied label rather than filling an absent outcome."""
        if self.target.status != "observed" or self.target.value is None:
            raise ValueError("Ridge append requires an observed target; null is not zero")
        return self


class RidgeFailure(RidgeContract):
    """Explain one refused finite computation without an accepted append.

    Attributes
    ----------
    code : str
        Closed range, underflow, nonpositive-pivot, residual or capacity reason.
    message : str
        Bounded human diagnostic. A failure grants no automatic retry or reset.
    """

    code: Literal["numeric_range", "underflow", "non_positive_pivot", "residual", "capacity"]
    message: Annotated[str, Field(strict=True, min_length=1, max_length=4096)]


class RidgePrediction(RidgeContract):
    """Retain the actual pre-update model output, not an observed future target.

    Attributes
    ----------
    input : RidgeInput
        Complete original features and configuration binding used for prediction.
    state_fingerprint : str
        Content identity of the full mathematical state at this call.
    sample_count : int
        Accepted labels so far; predictions do not increment it.
    value : Scalar
        Prediction-role value in target units, or explicit unavailable/null on
        numerical refusal. Observed status means the computed forecast exists,
        not that the future label is known, calibrated or certain.
    failure : RidgeFailure or None
        Explicit numerical refusal. No uncertainty/calibration is inferred.
    """

    input: RidgeInput
    state_fingerprint: Fingerprint
    sample_count: Count
    value: Scalar
    failure: RidgeFailure | None

    @model_validator(mode="after")
    def _result(self) -> Self:
        """Keep a failed forecast unavailable instead of inventing a zero."""
        expected = "observed" if self.failure is None else "unavailable"
        if self.value.status != expected or self.value.spec.role != "prediction":
            raise ValueError("Ridge prediction status/role differs from its computation disposition")
        return self


class RidgeUpdate(RidgeContract):
    """Report an accepted append or an unchanged-state computation refusal.

    Attributes
    ----------
    status : {'accepted', 'failed'}
        Actual mathematical append disposition, separate from delivery or ACK.
    sample_count, update_count : int
        Equal accepted-label counters after the call, never storage generations.
    before_fingerprint, after_fingerprint : str
        Full original/successor state identities; identical on refusal.
    failure : RidgeFailure or None
        Exactly a failed computation has a declared reason.
    """

    status: Literal["accepted", "failed"]
    sample_count: Count
    update_count: Count
    before_fingerprint: Fingerprint
    after_fingerprint: Fingerprint
    failure: RidgeFailure | None

    @model_validator(mode="after")
    def _result(self) -> Self:
        """Require equal append counters and no state change on a failure."""
        if self.sample_count != self.update_count:
            raise ValueError("Ridge single-sample and update counters must agree")
        if (self.status == "accepted") != (self.failure is None):
            raise ValueError("Exactly accepted ridge updates have no failure")
        if self.status == "accepted" and (not self.sample_count or self.before_fingerprint == self.after_fingerprint):
            raise ValueError("An accepted ridge append must increment its complete state")
        if self.status == "failed" and self.before_fingerprint != self.after_fingerprint:
            raise ValueError("A refused ridge append must retain its original state")
        return self


class RidgePayload(RidgeContract):
    """Export the entire append-only normal-equation state, without private caches.

    Attributes
    ----------
    config : RidgeConfig
        Complete original mathematical/schema/preprocessing/horizon definition.
    gram, cross : tuple
        A=lambda*I+sum(x*xT), b=sum(x*y), in declared feature order. No inverse.
    coefficients : tuple of float
        Recomputed checked Cholesky solution, retaining the binary64 convention.
    sample_count, update_count : int
        Equal accepted-label counters, unrelated to calls or storage generations.

    Notes
    -----
    This is sufficient state, not authenticated history. It cannot prove that
    supplied labels were eligible. No pending labels, RNG or optimizer is owned.
    Reads rederive coefficients; unknown versions or altered claims reject.
    """

    config: RidgeConfig
    gram: Matrix
    cross: Values
    coefficients: Values
    sample_count: Count
    update_count: Count

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Check fixed dimensions, exact symmetry, counters and sufficient solve."""
        size = len(self.config.features)
        if len(self.gram) != size or any(len(row) != size for row in self.gram):
            raise ValueError("Ridge Gram dimensions differ from its fixed features")
        if len(self.cross) != size or len(self.coefficients) != size:
            raise ValueError("Ridge coefficient/cross dimensions differ from its fixed features")
        if self.sample_count != self.update_count or self.sample_count > self.config.limits.maximum_samples:
            raise ValueError("Ridge accepted sample/update counters are incompatible")
        if any(pack("!d", self.gram[i][j]) != pack("!d", self.gram[j][i]) for i in range(size) for j in range(size)):
            raise ValueError("Ridge Gram must retain exact symmetric entries")
        if any(self.gram[i][i] < self.config.regularization for i in range(size)):
            raise ValueError("Ridge Gram diagonal cannot be below its regularization")
        _statistic_capacity(self)
        try:
            expected = solve(self.gram, self.cross)
        except NumericalError as error:
            raise ValueError("Ridge state cannot reproduce its finite Cholesky solution") from error
        if tuple(pack("!d", value) for value in expected) != tuple(pack("!d", value) for value in self.coefficients):
            raise ValueError("Ridge coefficients differ from its complete sufficient statistics")
        if self.sample_count == 0:
            _initial(self)
        return self


def _initial(state: RidgePayload) -> None:
    """Refuse a nonzero statistic falsely labeled as an untrained ridge owner."""
    regularization = state.config.regularization
    for i, row in enumerate(state.gram):
        if any(pack("!d", value) != pack("!d", regularization if i == j else 0.0) for j, value in enumerate(row)):
            raise ValueError("Untrained ridge state must contain exactly lambda*I")
    if any(state.cross) or any(state.coefficients):
        raise ValueError("Untrained ridge state must have zero cross-products and coefficients")


def _statistic_capacity(state: RidgePayload) -> None:
    """Refuse statistics beyond bounded samples with conservative rounding headroom."""
    allowance = state.sample_count * state.config.limits.maximum_input_magnitude**2
    # At most2**20 rounded single additions: factor2 safely exceeds their finite
    # forward-error bound. This capacity check does not authenticate source history.
    gram_limit = 2 * (state.config.regularization + allowance)
    if any(abs(value) > gram_limit for row in state.gram for value in row):
        raise ValueError("Ridge Gram exceeds its declared sample/magnitude capacity")
    if any(abs(value) > 2 * allowance for value in state.cross):
        raise ValueError("Ridge cross-products exceed their declared sample/magnitude capacity")
