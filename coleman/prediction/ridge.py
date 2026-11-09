"""Independently usable append-only ridge with atomic numerical refusals."""

from coleman.core.identity import Reference
from coleman.core.numeric import Scalar, ScalarSpec
from coleman.core.primitives import fingerprint
from coleman.objectives.bindings import DefinitionBinding
from coleman.prediction._numerics import NumericalError, advance, product, total
from coleman.prediction.contracts import (
    RidgeConfig,
    RidgeFailure,
    RidgeInput,
    RidgePayload,
    RidgePrediction,
    RidgeSample,
    RidgeUpdate,
)
from coleman.state.contracts import AlgorithmState, StateBinding

CODEC = Reference(id="append-only-ridge-state", version="1")
IMPLEMENTATION = DefinitionBinding(
    reference=Reference(id="append-only-ridge-cholesky", version="1"),
    definition_fingerprint=fingerprint(
        {
            "law": "A=lambda*I+sum(xxT);b=sum(xy);beta=CholeskySolve(A,b)",
            "intercept": "none implicit; all declared coefficients penalized; center zero",
            "append": "one label, no forgetting; direct call is non-idempotent",
            "arithmetic": "binary64 products/fsum updates; SPD Cholesky; no inverse or jitter",
            "refusals": "nonfinite,nonzero-product/quotient-underflow,nonpositive-pivot,capacity,residual",
            "residual": "per row abs(A*beta-b)<=1e-12+1e-10*max(1,abs(b),sum(abs(Aij*betaj)))",
            "order": "original ordered feature schema and original accepted append order",
        }
    ),
)


def _failure(error: NumericalError) -> RidgeFailure:
    """Construct one closed public numerical failure without untyped state."""
    return RidgeFailure.model_validate({"schema_version": 1, "code": error.code, "message": str(error)})


def _bound(value: float, spec: ScalarSpec, maximum: float) -> None:
    """Validate an admitted finite scalar against its exact input schema."""
    if abs(value) > maximum:
        raise ValueError("Ridge input exceeds its declared absolute magnitude capacity")
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError("Ridge input is below its declared schema minimum")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError("Ridge input exceeds its declared schema maximum")


class AppendOnlyRidge:
    """Own complete normal-equation state independently from scheduling/storage.

    Parameters
    ----------
    config : RidgeConfig
        Fixed positive lambda, ordered dimensionless inputs, target units,
        preprocessing/horizon versions and finite capacities.
    owner : Reference
        Stable caller-owned component identity/version; no registration occurs.

    Notes
    -----
    Labels are supplied to append, which is deliberately non-idempotent.
    Predictions do not mutate state. The host owns causal eligibility, pending
    labels, revisions/deduplication and external effects. No optimizer, RNG,
    clock, filesystem, numerical backend or transport is accessed. This owner
    has no concurrent same-instance update guarantee; stage separate owners.
    """

    def __init__(self, config: RidgeConfig, *, owner: Reference) -> None:
        """Initialize lambda*I and zero sufficient statistics without source access.

        Parameters
        ----------
        config : RidgeConfig
            Complete revalidated immutable mathematical convention.
        owner : Reference
            Stable logical owner; included in every exported state binding.

        Raises
        ------
        ValueError, TypeError
            Invalid schema/configuration, capacity or numerical initial state.
        """
        checked = RidgeConfig.model_validate(config)
        binding = StateBinding(
            owner=Reference.model_validate(owner),
            implementation=IMPLEMENTATION,
            state_codec=CODEC,
            configuration_fingerprint=checked.fingerprint,
        )
        size = len(checked.features)
        self._state = AlgorithmState[RidgePayload](
            schema_version=1,
            binding=binding,
            payload=RidgePayload(
                schema_version=1,
                config=checked,
                gram=tuple(tuple(checked.regularization if i == j else 0.0 for j in range(size)) for i in range(size)),
                cross=(0.0,) * size,
                coefficients=(0.0,) * size,
                sample_count=0,
                update_count=0,
            ),
        )

    @property
    def config(self) -> RidgeConfig:
        """Return the complete immutable mathematical configuration.

        Returns
        -------
        RidgeConfig
            Actual fixed schema, preprocessing and all numerical conventions.
        """
        return self._state.payload.config

    @property
    def binding(self) -> StateBinding:
        """Return this owner's exact implementation/configuration/codec identity.

        Returns
        -------
        StateBinding
            Supplied logical owner and declared complete interpretation.
        """
        return self._state.binding

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Return actual fitted coefficients in the original feature order.

        Returns
        -------
        tuple of float
            Target units per dimensionless feature; no uncertainty/calibration.
        """
        return self._state.payload.coefficients

    @property
    def sample_count(self) -> int:
        """Return accepted labels, excluding predictions and failed appends.

        Returns
        -------
        int
            Mathematical sample count, not a delivery/storage generation.
        """
        return self._state.payload.sample_count

    def _input(self, features: RidgeInput) -> RidgeInput:
        """Check complete schema identity and magnitude before any calculation."""
        checked = RidgeInput.model_validate(features)
        if checked.config != self.config.binding or len(checked.values) != len(self.config.features):
            raise ValueError("Ridge input differs from its exact fixed feature/configuration schema")
        for value, spec in zip(checked.values, self.config.features, strict=True):
            _bound(value, spec, self.config.limits.maximum_input_magnitude)
        return checked

    def predict(self, features: RidgeInput, /) -> RidgePrediction:
        """Compute from the current accepted prefix without changing any state.

        Parameters
        ----------
        features : RidgeInput
            Complete preprocessed feature vector, eligible by host declaration.

        Returns
        -------
        RidgePrediction
            Current-state-bound forecast in target units. Numerical refusal is
            explicit unavailable/null, never an invented zero or uncertainty.

        Raises
        ------
        ValueError, TypeError
            Invalid input/dimensions, incompatible schema or declared input bound.

        Notes
        -----
        A forecast is a linear score, not a calibrated probability. No label,
        clock, optimizer or source is read, and no sample/update counter changes.
        """
        checked = self._input(features)
        failure = None
        try:
            value = total(product(a, b) for a, b in zip(self.coefficients, checked.values, strict=True))
        except NumericalError as error:
            failure = _failure(error)
            value = None
        return RidgePrediction(
            schema_version=1,
            input=checked,
            state_fingerprint=self._state.fingerprint,
            sample_count=self.sample_count,
            value=Scalar(
                spec=self.config.prediction_spec,
                value=value,
                status="observed" if failure is None else "unavailable",
            ),
            failure=failure,
        )

    def append(self, sample: RidgeSample, /) -> RidgeUpdate:
        """Append one observed label atomically; duplicate calls append twice.

        Parameters
        ----------
        sample : RidgeSample
            Original fixed-schema features and exact observed target quantity.
            The caller separately establishes maturity/availability/receipt.

        Returns
        -------
        RidgeUpdate
            Accepted successor or a typed numerical/capacity failure. Failure
            preserves every original statistic, coefficient, counter and byte.

        Raises
        ------
        ValueError, TypeError
            Invalid/incompatible input, nonfinite or missing target, wrong units
            or declared scalar bounds. Rejection leaves state unchanged.

        Notes
        -----
        Computes a fresh candidate before one final assignment. Cholesky uses no
        inverse, pseudoinverse or jitter. A valid mathematical SPD problem may
        refuse in this finite normal-equation profile. The checked residual is
        not a condition-number or coefficient-error guarantee. No old label is
        replaced, no forgetting occurs and no delivery idempotency is supplied.
        """
        original = RidgeSample.model_validate(sample)
        features = self._input(original.input)
        target = original.target.value
        if original.target.spec != self.config.target or target is None:
            raise ValueError("Ridge label differs from its exact target schema")
        _bound(target, self.config.target, self.config.limits.maximum_input_magnitude)
        before = self._state.fingerprint
        failure = None
        candidate = self._state
        try:
            candidate = self._candidate(features, target)
        except NumericalError as error:
            failure = _failure(error)
        result = RidgeUpdate(
            schema_version=1,
            status="accepted" if failure is None else "failed",
            sample_count=candidate.payload.sample_count,
            update_count=candidate.payload.update_count,
            before_fingerprint=before,
            after_fingerprint=candidate.fingerprint,
            failure=failure,
        )
        self._state = candidate
        return result

    def _candidate(self, features: RidgeInput, target: float) -> AlgorithmState[RidgePayload]:
        """Prepare a complete successor without assigning owner state."""
        if self.sample_count >= self.config.limits.maximum_samples:
            raise NumericalError("capacity", "Ridge append exceeds its declared sample capacity")
        previous = self._state.payload
        gram, cross, coefficients = advance(previous.gram, previous.cross, features.values, target)
        return AlgorithmState[RidgePayload](
            schema_version=1,
            binding=self.binding,
            payload=RidgePayload(
                schema_version=1,
                config=self.config,
                gram=gram,
                cross=cross,
                coefficients=coefficients,
                sample_count=previous.sample_count + 1,
                update_count=previous.update_count + 1,
            ),
        )

    def export_state(self) -> AlgorithmState[RidgePayload]:
        """Export complete immutable sufficient state with its concrete payload.

        Returns
        -------
        AlgorithmState of RidgePayload
            Revalidated A, b, coefficients, counters and full configuration.

        Raises
        ------
        ValueError, TypeError
            State content fails its fixed numerical/versioned invariants.

        Notes
        -----
        No I/O, ACK, authentication or pending-label ownership is supplied.
        Existing ComponentCodec may persist this concrete state independently.
        """
        return AlgorithmState[RidgePayload].model_validate(self._state)

    def restore_state(self, state: AlgorithmState[RidgePayload], /) -> None:
        """Restore a complete compatible state, rejecting before assignment.

        Parameters
        ----------
        state : AlgorithmState of RidgePayload
            Caller-selected concrete schema1 snapshot with exact complete owner,
            implementation, configuration, payload and codec bindings.

        Raises
        ------
        ValueError, TypeError
            Unknown schema, incompatible binding/configuration, altered or
            unrepresentable sufficient-statistic/Cholesky claims.

        Notes
        -----
        Reads rederive coefficients. A rejection preserves the old owner. This
        operation supplies no migration, history authentication, rollback floor,
        RNG restoration, pending label selection or source access.
        """
        checked = AlgorithmState[RidgePayload].model_validate(state)
        if checked.binding != self.binding or checked.payload.config.to_json() != self.config.to_json():
            raise ValueError("Ridge restore differs from its exact complete binding/configuration")
        self._state = checked
