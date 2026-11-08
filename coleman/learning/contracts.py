"""Finite immutable configuration, panel admission and direct learning results."""

from __future__ import annotations

from math import fsum, isfinite, ulp
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from coleman.core.actions import Weight
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.feedback import FeedbackRow, FullPanel, OutcomeValue
from coleman.core.identity import Reference, ordered_references
from coleman.core.numeric import Binary64, ScalarSpec, validate_simplex
from coleman.core.primitives import Identifier, canonical_json
from coleman.core.reading import bounded_json_bytes
from coleman.objectives.bindings import DefinitionBinding, SemanticDefinition
from coleman.objectives.preferences import InitialPrior, has_full_support

MAXIMUM_BYTES = 16 * 1024**2
ReferenceProfile = Literal["hedge-constant", "hedge-anytime-cumulative", "fixed-share-excluding-origin"]
FailureCode = Literal["numeric_range", "positive_underflow", "normalization", "infeasible", "capacity"]
Weights = Annotated[tuple[Weight, ...], Field(min_length=1, max_length=4096)]
Values = Annotated[tuple[Binary64, ...], Field(min_length=1, max_length=4096)]
UpdateCount = Annotated[int, Field(strict=True, ge=0, le=1048576)]


def _json_input(json_data: str | bytes | bytearray) -> bytes:
    """Admit a new learning JSON route before copying or recursive decoding."""
    if type(json_data) is bytearray:  # pylint: disable=unidiomatic-typecheck
        if len(json_data) > MAXIMUM_BYTES:
            raise ValueError("Learning JSON exceeds its 16 MiB byte ceiling")
        json_data = bytes(json_data)
    if not isinstance(json_data, str | bytes):
        raise TypeError("Learning JSON requires built-in str, bytes or bytearray")
    return bounded_json_bytes(json_data, MAXIMUM_BYTES)


class _BoundedImmutable(CoreImmutable):
    """Keep new nested learning JSON readers bounded without adding v1 fields."""

    @classmethod
    def model_validate_json(  # pylint: disable=too-many-arguments
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Bound nested DTO JSON before core duplicate-safe token decoding.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete UTF-8 content, at most16MiB and lexical depth64. A
            bytearray copies only after its length is admitted.
        strict : {True} or None, optional
            Existing core strict mode. False cannot relax field validation.
        extra : {'forbid'} or None, optional
            Existing unknown-field policy; allow/ignore reject.
        context : object or None, optional
            Explicit transient Pydantic validation context, never serialized.
        by_alias, by_name : bool or None, optional
            Existing core field-name selection, unchanged by this byte guard.

        Returns
        -------
        Self
            Revalidated immutable nested value with the original field layout.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid size/depth/UTF-8/JSON/content or relaxed-field policy.
        """
        return super().model_validate_json(
            _json_input(json_data), strict=strict, extra=extra, context=context, by_alias=by_alias, by_name=by_name
        )


class LearningLimits(_BoundedImmutable):
    """Bound one direct learner without allocating or importing a runtime.

    Attributes
    ----------
    maximum_members : int, optional
        Positive fixed coordinate capacity, default and hard limit4096.
    maximum_updates : int, optional
        Accepted complete-panel capacity, default and hard limit1,048,576.
    maximum_bytes : int, optional
        Complete canonical configuration/state ceiling, default/hard16MiB.
        Independent count maxima need not fit simultaneously.
    """

    maximum_members: Annotated[int, Field(strict=True, ge=1, le=4096)] = 4096
    maximum_updates: Annotated[int, Field(strict=True, ge=1, le=1048576)] = 1048576
    maximum_bytes: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_BYTES)] = MAXIMUM_BYTES


class _BoundedContract(CoreContract):
    """Keep v1 core bytes with a finite depth/size envelope."""

    def to_json(self) -> str:
        """Write complete canonical v1 content within16MiB and depth64.

        Returns
        -------
        str
            Core canonical UTF-8 JSON text; no fields or signed zeros are lost.

        Raises
        ------
        ValueError, TypeError
            Invalid typed content or the hard byte/depth bound is exceeded.
        """
        result = super().to_json()
        bounded_json_bytes(result, MAXIMUM_BYTES)
        return result

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Read bounded duplicate-safe core JSON with strict v1 fields.

        Parameters
        ----------
        payload : str or bytes
            Complete encoded value, at most16MiB and lexical depth64.

        Returns
        -------
        Self
            Concrete revalidated configuration/result/state.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Unknown fields/version, invalid JSON/content or exceeded bounds.
        """
        return super().from_json(bounded_json_bytes(payload, MAXIMUM_BYTES))

    @classmethod
    def model_validate_json(  # pylint: disable=too-many-arguments
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Bound alternate v1 JSON ingress before core recursive decoding.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            Complete UTF-8 content, at most16MiB and depth64. Bytearray length
            is checked before copying; callback-bearing subclasses reject.
        strict : {True} or None, optional
            Existing core strict mode; False cannot weaken declared fields.
        extra : {'forbid'} or None, optional
            Unknown fields stay forbidden; allow/ignore reject.
        context : object or None, optional
            Explicit transient validation context, not captured as state.
        by_alias, by_name : bool or None, optional
            Existing core field-name selection after bounded token decoding.

        Returns
        -------
        Self
            Concrete complete known-version immutable value.

        Raises
        ------
        ValueError, TypeError, UnicodeError
            Invalid size/depth/UTF-8/JSON/version/content or relaxed policy.
        """
        return super().model_validate_json(
            _json_input(json_data), strict=strict, extra=extra, context=context, by_alias=by_alias, by_name=by_name
        )


def _size(value: CoreContract, maximum: int) -> None:
    """Check already-validated immutable content without recursive validation."""
    if len(canonical_json(value.model_dump(mode="json")).encode("utf-8")) > maximum:
        raise ValueError("Learning complete content exceeds its selected byte capacity")


def _raw_prior(value: Any) -> Any:
    """Inspect only a finite flat prior envelope before recursive admission."""
    if isinstance(value, InitialPrior):
        return value.weights
    return value.get("weights", ()) if isinstance(value, dict) else ()


class ReferenceConfig(SemanticDefinition, _BoundedContract):
    """Bind one static full-information reference method and numerical law.

    Attributes
    ----------
    profile : str
        ``hedge-constant``, ``hedge-anytime-cumulative`` or the distinct
        ``fixed-share-excluding-origin`` recurrence. Dynamic/specialist
        catalogue controls are unsupported and unknown fields reject.
    prior : InitialPrior
        Complete positive stored-binary64 simplex. For exact mathematical
        comparisons its ideal prior is a/C, where C is the exact real sum of
        the stored entries a. Emitted normalization is separately rounded.
    loss_spec : ScalarSpec
        One dimensionless loss dimension, exact role loss and bounds[0,1].
    transform : DefinitionBinding
        Pinned definition; each outcome must name its exact reference. A panel
        carries no definition preimage and does not authenticate that binding.
    eta : float or None, optional
        Constant nonnegative finite rate for constant Hedge/Fixed Share.
        Eta0 is an admitted identity-loss endpoint, outside divided-by-eta
        positive-rate comparator statements. Required null for anytime.
    alpha : float or None, optional
        Fixed Share's fraction[0,1]; required null for both Hedge profiles.
    anytime_multiplier : float or None, optional
        Positive finite multiplier c in eta_t=c*sqrt(log(n)/t), default2 for
        anytime, null otherwise. The n1 profile is absorbing weight one.
    limits : LearningLimits, optional
        Selected coupled count/byte bounds. No history is silently evicted.
    numerical_profile, prior_normalization : str, optional
        Exact versioned binary64 computation/normalization conventions.

    Notes
    -----
    Reference, lineage and every parameter are bound by inherited semantic
    identity. Rounded emitted weights obey binary64-simplex-v1 (exact stored
    rational residual at most2**-49), not exact real probability arithmetic.
    Numerical refusal leaves mathematical state unchanged. This configuration
    supplies no delivery idempotency, asynchronous theorem or efficacy claim.
    """

    profile: ReferenceProfile
    prior: InitialPrior
    loss_spec: ScalarSpec
    transform: DefinitionBinding
    eta: Annotated[Binary64, Field(ge=0)] | None = None
    alpha: Annotated[Binary64, Field(ge=0, le=1)] | None = None
    anytime_multiplier: Annotated[Binary64, Field(gt=0)] | None = None
    limits: LearningLimits = Field(default_factory=LearningLimits)
    numerical_profile: Literal["centered-binary64-v1"] = "centered-binary64-v1"
    prior_normalization: Literal["ideal-raw-mass-emitted-fsum-v1"] = "ideal-raw-mass-emitted-fsum-v1"

    @model_validator(mode="before")
    @classmethod
    def _work(cls, value: Any) -> Any:
        """Cap flat unchecked prior cardinality before nested validation."""
        if isinstance(value, ReferenceConfig):
            prior, limits = value.prior, value.limits
        elif isinstance(value, dict):
            prior, limits = value.get("prior"), value.get("limits", LearningLimits())
        else:
            return value
        selected = LearningLimits.model_validate(limits)
        weights = _raw_prior(prior)
        if type(weights) not in {tuple, list} or not 1 <= len(weights) <= selected.maximum_members:
            raise ValueError("Reference prior exceeds its finite selected member capacity")
        return value

    @model_validator(mode="after")
    def _configuration(self) -> Self:
        """Require positive support, exact bounded loss and one chosen law."""
        validate_loss_spec(self.loss_spec)
        if not has_full_support(self.prior):
            raise ValueError("Reference methods require a strictly positive prior")
        if self.profile == "hedge-anytime-cumulative":
            if self.eta is not None or self.alpha is not None:
                raise ValueError("Anytime cumulative Hedge requires its schedule, not constant eta/share")
            if self.anytime_multiplier is None:
                object.__setattr__(self, "anytime_multiplier", 2.0)
        elif self.eta is None or self.anytime_multiplier is not None:
            raise ValueError("Constant reference methods require eta and no anytime schedule")
        elif (self.alpha is not None) != (self.profile == "fixed-share-excluding-origin"):
            raise ValueError("Only excluding-origin Fixed Share requires alpha")
        _size(self, self.limits.maximum_bytes)
        return self


class LearningFailure(_BoundedImmutable):
    """Retain a typed mathematical refusal without a fabricated update.

    Attributes
    ----------
    code : str
        Numeric range, positive underflow, normalization, infeasibility or
        selected capacity. A consumer declares its failure/acceptance policy.
    message : str
        Finite nonempty diagnostic; it is not observed loss or paid cost.
    """

    code: FailureCode
    message: Identifier


class LearningNumericalError(ArithmeticError):
    """Signal an unrepresentable pure numerical result before state assignment.

    Attributes
    ----------
    failure : LearningFailure
        Supported typed cause, available to an optional caller-owned adapter.
    """

    def __init__(self, code: FailureCode, message: str):
        """Retain a supported typed cause for a caller-selected failure route.

        Parameters
        ----------
        code : str
            Supported ``LearningFailure`` diagnostic discriminator.
        message : str
            Explanation of the finite numerical qualification that failed.
        """
        self.failure = LearningFailure(code=code, message=message)
        super().__init__(message)


class LearningUpdate(_BoundedContract):
    """Report one pure panel operation separately from durable delivery.

    Attributes
    ----------
    status : {'accepted', 'numerical_failure'}
        An accepted complete panel advances U once. A numerical failure retains
        the original U/weights/state; invalid input raises before this result.
    update_count : int
        Successful accepted-panel U after this operation, never decision count.
    weights : tuple of Weight
        Actual emitted binary64 coefficients, including a failure's old vector.
    failure : LearningFailure or None
        Required for failure, null for accepted input.
    """

    status: Literal["accepted", "numerical_failure"]
    update_count: UpdateCount
    weights: Weights
    failure: LearningFailure | None

    @model_validator(mode="after")
    def _result(self) -> Self:
        """Keep result status, vector and typed numerical cause coherent."""
        ordered_references(tuple(item.member for item in self.weights))
        validate_simplex(tuple(item.value for item in self.weights))
        if (self.status == "accepted") != (self.failure is None):
            raise ValueError("Learning result status and failure must agree")
        return self


class ReferencePayload(_BoundedContract):
    """Capture complete immutable reference state without caches or RNG.

    Attributes
    ----------
    configuration : ReferenceConfig
        Full fixed law/prior/quantity/transform/lineage, not just display IDs.
    weights : tuple of Weight
        Actual emitted vector used by the next constant-rate/share operation.
    update_count : int
        U accepted complete panels. No decision or delivery counter is inferred.
    cumulative_losses, compensations : tuple of float
        Per-member Neumaier sum and correction, both needed for continuation.
        Anytime recomputes with eta_(U+1) on their compensated cumulative value;
        it never stores a product of historical learning rates.
    """

    configuration: ReferenceConfig
    weights: Weights
    update_count: UpdateCount
    cumulative_losses: Values
    compensations: Values

    @model_validator(mode="before")
    @classmethod
    def _work(cls, value: Any) -> Any:
        """Bound unchecked vector cardinalities before recursive validation."""
        if isinstance(value, ReferencePayload):
            vectors = (value.weights, value.cumulative_losses, value.compensations)
        elif isinstance(value, dict):
            vectors = tuple(value.get(name, ()) for name in ("weights", "cumulative_losses", "compensations"))
        else:
            return value
        if any(type(vector) not in {tuple, list} or not 1 <= len(vector) <= 4096 for vector in vectors):
            raise ValueError("Reference state requires finite bounded vector inventories")
        return value

    @model_validator(mode="after")
    def _state(self) -> Self:
        """Check exact static support, bounded statistics and selected bytes."""
        expected = tuple(item.member for item in self.configuration.prior.weights)
        if tuple(item.member for item in self.weights) != expected:
            raise ValueError("Reference state requires its exact fixed member versions")
        if len(self.cumulative_losses) != len(expected) or len(self.compensations) != len(expected):
            raise ValueError("Reference sufficient statistics require every exact member")
        validate_simplex(tuple(item.value for item in self.weights))
        if any(item.value <= 0 for item in self.weights):
            raise ValueError("Reference state cannot create structural zero support")
        if self.update_count > self.configuration.limits.maximum_updates:
            raise ValueError("Reference accepted-panel capacity exhausted")
        correction_bound = 2 * self.update_count * ulp(float(max(1, self.update_count)))
        for total, correction in zip(self.cumulative_losses, self.compensations, strict=True):
            if not 0 <= total <= self.update_count:
                raise ValueError("Reference cumulative losses exceed accepted bounded panels")
            if abs(correction) > correction_bound:
                raise ValueError("Reference compensation exceeds its declared finite accumulation envelope")
            if not 0 <= fsum((total, correction)) <= self.update_count:
                raise ValueError("Reference cumulative losses exceed accepted bounded panels")
        _size(self, self.configuration.limits.maximum_bytes)
        return self


def validate_loss_spec(loss_spec: ScalarSpec) -> None:
    """Require one explicit dimensionless normalized-loss specification.

    Parameters
    ----------
    loss_spec : ScalarSpec
        Exact role loss, unit1, inclusive finite minimum0 and maximum1.

    Raises
    ------
    ValueError
        Wrong role, units or bounds; no conversion or clipping is performed.
    """
    checked = ScalarSpec.model_validate(loss_spec)
    if checked.role != "loss" or checked.minimum != 0 or checked.maximum != 1 or checked.semantics.unit != "1":
        raise ValueError("Learning requires a dimensionless loss specification with exact bounds[0,1]")


def _panel_rows(panel: FullPanel, count: int) -> None:
    """Reject unsupported/oversized typed envelopes before deep validation."""
    # Exact containers keep unchecked subclasses from invoking iteration hooks.
    # pylint: disable=unidiomatic-typecheck
    if type(panel) is not FullPanel or type(panel.rows) is not tuple or len(panel.rows) != count:
        raise ValueError("Learning requires one bounded full panel over its exact members")
    for row in panel.rows:
        if type(row) is not FeedbackRow or type(row.values) is not tuple or len(row.values) != 1:
            raise ValueError("Learning requires exactly one bounded loss value per member")
        if type(row.values[0]) is not OutcomeValue:
            raise ValueError("Learning requires a concrete typed loss outcome")
    # pylint: enable=unidiomatic-typecheck


def _panel_loss(row: FeedbackRow, expected_spec: str, transform: Reference) -> float:
    """Read one already-revalidated exact observed normalized loss."""
    value = row.values[0]
    if value.spec.model_dump_json() != expected_spec or value.transform != transform:
        raise ValueError("Loss panel differs from the exact loss specification or transform")
    if value.status != "observed" or value.value is None:
        raise ValueError("Every active reference member requires an observed loss; null is not zero")
    if not isfinite(value.value) or not 0 <= value.value <= 1:
        raise ValueError("Learning loss must be finite and bounded[0,1]")
    return value.value


def read_loss_panel(
    panel: FullPanel,
    *,
    members: tuple[Reference, ...],
    loss_spec: ScalarSpec,
    transform: DefinitionBinding,
) -> tuple[float, ...]:
    """Admit a complete observed bounded vector without inventing missing loss.

    Parameters
    ----------
    panel : FullPanel
        Concrete typed panel, exactly one value for each fixed member. Raw row
        and value cardinalities are capped before recursive model validation.
        Repeated pinned scalar/transform schemas plus member encodings must
        fit the16MiB validation-work envelope before deep panel validation.
    members : tuple of Reference
        One through4096 unique fixed members in canonical logical-ID order.
    loss_spec : ScalarSpec
        Exact expected quantity/role/units/bounds, including signed-zero bytes.
    transform : DefinitionBinding
        Pinned operation; each value must name its exact versioned reference.

    Returns
    -------
    tuple of float
        Complete observed losses in member order, each in[0,1].

    Raises
    ------
    ValueError, TypeError
        Wrong/unbounded container, incomplete/versioned coverage, extra
        dimensions, unavailable/null values or mismatched interpretation.

    Notes
    -----
    Provenance remains the supplied core origin. This reads no future panels,
    authenticates no source and performs no update, delivery deduplication,
    transform, state assignment or I/O.
    """
    if type(members) is not tuple or not 1 <= len(members) <= 4096:  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Learning members require a finite immutable tuple")
    _panel_rows(panel, len(members))
    expected_members = tuple(Reference.model_validate(member) for member in members)
    ordered_references(expected_members)
    if tuple(sorted(expected_members, key=lambda member: member.id)) != expected_members:
        raise ValueError("Learning member order must be canonical")
    validate_loss_spec(loss_spec)
    operation = DefinitionBinding.model_validate(transform)
    expected_spec = loss_spec.model_dump_json()
    schema_bytes = len(expected_spec.encode("utf-8")) + len(operation.reference.model_dump_json().encode("utf-8"))
    member_bytes = sum(len(member.model_dump_json().encode("utf-8")) for member in expected_members)
    if schema_bytes * len(members) + member_bytes > MAXIMUM_BYTES:
        raise ValueError("Loss panel exceeds its finite scalar-schema validation byte capacity")
    checked = FullPanel.model_validate(panel)
    if tuple(row.member for row in checked.rows) != expected_members:
        raise ValueError("Loss panel differs from the exact fixed member versions")
    return tuple(_panel_loss(row, expected_spec, operation.reference) for row in checked.rows)
