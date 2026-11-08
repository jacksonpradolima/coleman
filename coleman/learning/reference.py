"""Static full-information Hedge and excluding-origin Fixed Share references."""

from __future__ import annotations

from math import fsum, isfinite, log, sqrt
from typing import ClassVar

from coleman.core.actions import Weight
from coleman.core.feedback import FullPanel
from coleman.core.identity import Reference
from coleman.core.numeric import ScalarSpec
from coleman.core.primitives import canonical_json, fingerprint
from coleman.learning._math import accumulate_losses, excluding_origin_share, exp_update, normalize_positive_prior
from coleman.learning.contracts import (
    LearningNumericalError,
    LearningUpdate,
    ReferenceConfig,
    ReferencePayload,
    ReferenceProfile,
    read_loss_panel,
)
from coleman.objectives.bindings import DefinitionBinding
from coleman.state.contracts import AlgorithmState, StateBinding


def _implementation(profile: ReferenceProfile) -> DefinitionBinding:
    """Bind the selected mathematical/numerical method, without source lookup."""
    return DefinitionBinding(
        reference=Reference(id="coleman-" + profile, version="1"),
        definition_fingerprint=fingerprint(
            {
                "profile": profile,
                "version": 1,
                "loss": "complete-observed-dimensionless-[0,1]",
                "numerical": "centered-binary64-v1",
                "constant_state": "actual-emitted-coefficients",
                "anytime_state": "current-rate-times-compensated-cumulative-loss",
                "share": "excluding-origin-n-minus-one-singleton-absorbing",
                "clock": "accepted-complete-panels-U-next-rate-U-plus-one",
            }
        ),
    )


def _rate(configuration: ReferenceConfig, update_count: int) -> float:
    """Select the next prediction's rate without consulting decision/receipt time."""
    if configuration.profile != "hedge-anytime-cumulative":
        if configuration.eta is None:
            raise ValueError("A constant reference requires its configured eta")
        return configuration.eta
    if len(configuration.prior.weights) == 1:
        return 0.0
    multiplier = configuration.anytime_multiplier
    if multiplier is None:
        raise ValueError("Anytime Hedge requires its configured schedule multiplier")
    result = multiplier * sqrt(log(len(configuration.prior.weights)) / (update_count + 1))
    if not isfinite(result):
        raise LearningNumericalError("numeric_range", "The next anytime rate is outside finite binary64")
    if result == 0:
        raise LearningNumericalError("positive_underflow", "The positive next anytime rate underflows binary64")
    return result


def _state(binding: StateBinding, payload: ReferencePayload) -> AlgorithmState[ReferencePayload]:
    """Validate complete state and selected bytes before any owner assignment."""
    result = AlgorithmState[ReferencePayload](schema_version=1, binding=binding, payload=payload)
    if len(result.to_json().encode("utf-8")) > payload.configuration.limits.maximum_bytes:
        raise ValueError("Reference complete state exceeds its selected byte capacity")
    return result


class _ReferenceLearner:
    """Own exactly one closed immutable complete state, replaced atomically."""

    _profile: ClassVar[ReferenceProfile]

    def __init__(self, config: ReferenceConfig, *, owner: Reference):
        """Initialize a positive static reference without an optional runtime.

        Parameters
        ----------
        config : ReferenceConfig
            Complete static method/prior/loss/transform/rate/lineage/bounds.
        owner : Reference
            Explicit logical state owner and version; no registration occurs.

        Raises
        ------
        ValueError, TypeError
            Wrong profile, invalid configuration or selected capacity.
        LearningNumericalError
            Initialization or the initial positive rate is unrepresentable.
        """
        checked = ReferenceConfig.model_validate(config)
        if checked.profile != self._profile:
            raise ValueError("Reference constructor and configured method profile differ")
        selected = Reference.model_validate(owner)
        _rate(checked, 0)
        weights = normalize_positive_prior(checked.prior.weights)
        binding = StateBinding(
            owner=selected,
            implementation=_implementation(checked.profile),
            state_codec=Reference(id="coleman-reference-learning-state", version="1"),
            configuration_fingerprint=checked.fingerprint,
        )
        payload = ReferencePayload(
            schema_version=1,
            configuration=checked,
            weights=weights,
            update_count=0,
            cumulative_losses=(0.0,) * len(weights),
            compensations=(0.0,) * len(weights),
        )
        self._state = _state(binding, payload)

    @property
    def config(self) -> ReferenceConfig:
        """Return the complete immutable fixed law.

        Returns
        -------
        ReferenceConfig
            Exact configuration, including positive raw prior and lineage.
        """
        return self._state.payload.configuration

    @property
    def binding(self) -> StateBinding:
        """Return exact owner, method, codec and configuration interpretation.

        Returns
        -------
        StateBinding
            Immutable binding; it authenticates no source or execution history.
        """
        return self._state.binding

    @property
    def members(self) -> tuple[Reference, ...]:
        """Return the exact static expert universe, in canonical member order.

        Returns
        -------
        tuple of Reference
            Complete member versions. Sleeping/dynamic subsets are unsupported.
        """
        return tuple(item.member for item in self.config.prior.weights)

    @property
    def loss_spec(self) -> ScalarSpec:
        """Return the declared dimensionless[0,1] loss schema.

        Returns
        -------
        ScalarSpec
            Exact dimension, units, role and inclusive numerical bounds.
        """
        return self.config.loss_spec

    @property
    def transform(self) -> DefinitionBinding:
        """Return the pinned loss-transform interpretation.

        Returns
        -------
        DefinitionBinding
            Complete bound definition; panels name its versioned reference.
        """
        return self.config.transform

    @property
    def weights(self) -> tuple[Weight, ...]:
        """Inspect actual emitted coefficients without deciding or updating.

        Returns
        -------
        tuple of Weight
            Finite positive binary64-simplex-v1 coefficients. These are neither
            sampled actions, propensity estimates nor domain holdings.
        """
        return self._state.payload.weights

    @property
    def update_count(self) -> int:
        """Return successful accepted complete-panel U, independent of D.

        Returns
        -------
        int
            U counts direct operations; raw repeated panels are new operations.
        """
        return self._state.payload.update_count

    def observe(self, panel: FullPanel) -> LearningUpdate:
        """Apply one exact observed loss vector under the selected U-clock law.

        Parameters
        ----------
        panel : FullPanel
            Complete fixed-member panel, exactly one observed matching loss
            per row. Null/partial values are not zero observations.

        Returns
        -------
        LearningUpdate
            Accepted new U/weights or a typed numerical failure retaining the
            entire original state and U. No delivery receipt is inferred.

        Raises
        ------
        ValueError, TypeError
            Invalid input/static universe/interpretation or selected capacity.

        Notes
        -----
        Constant Hedge and Fixed Share start from the actual emitted vector.
        Anytime recomputes from its emitted initial prior with eta_(U+1) on
        all compensated accepted losses. Candidate construction and byte checks
        precede the single state assignment. A replay adapter must explicitly
        handle numerical_failure rather than pretending it applied an update.
        """
        configuration = self.config
        losses = read_loss_panel(
            panel, members=self.members, loss_spec=configuration.loss_spec, transform=configuration.transform
        )
        if self.update_count >= configuration.limits.maximum_updates:
            raise ValueError("Reference accepted-panel capacity exhausted")
        previous = self._state.payload
        following = previous.update_count + 1
        try:
            totals, corrections = accumulate_losses(previous.cumulative_losses, previous.compensations, losses)
            if configuration.profile == "hedge-anytime-cumulative":
                effective = tuple(fsum(pair) for pair in zip(totals, corrections, strict=True))
                initial = normalize_positive_prior(configuration.prior.weights)
                weights = exp_update(initial, effective, _rate(configuration, following))
            else:
                weights = exp_update(previous.weights, losses, _rate(configuration, following))
                if configuration.profile == "fixed-share-excluding-origin":
                    if configuration.alpha is None:
                        raise ValueError("Excluding-origin Fixed Share requires its alpha")
                    weights = excluding_origin_share(weights, configuration.alpha)
            payload = ReferencePayload(
                schema_version=1,
                configuration=configuration,
                weights=weights,
                update_count=following,
                cumulative_losses=totals,
                compensations=corrections,
            )
            candidate = _state(self.binding, payload)
        except LearningNumericalError as error:
            return LearningUpdate(
                schema_version=1,
                status="numerical_failure",
                update_count=previous.update_count,
                weights=previous.weights,
                failure=error.failure,
            )
        result = LearningUpdate(
            schema_version=1, status="accepted", update_count=following, weights=weights, failure=None
        )
        self._state = candidate
        return result

    def update(self, panel: FullPanel, /) -> LearningUpdate:
        """Apply the same pure direct operation through the Updater protocol.

        Parameters
        ----------
        panel : FullPanel
            Same complete bounded input admitted by ``observe``.

        Returns
        -------
        LearningUpdate
            Identical acceptance/failure semantics, without automatic dedup.

        Raises
        ------
        ValueError, TypeError
            Invalid panel, unsupported static coverage or selected capacity.
        """
        return self.observe(panel)

    def export_state(self) -> AlgorithmState[ReferencePayload]:
        """Export a complete independently revalidated immutable continuation.

        Returns
        -------
        AlgorithmState[ReferencePayload]
            Exact fixed configuration, emitted coefficients, U, cumulative
            losses and every compensation. No RNG/cache/workspace is omitted.

        Raises
        ------
        ValueError
            Typed state or its selected byte bound no longer validates.
        """
        checked = AlgorithmState[ReferencePayload].model_validate(self._state)
        return _state(checked.binding, checked.payload)

    def restore_state(self, state: AlgorithmState[ReferencePayload], /) -> None:
        """Restore one exact compatible complete state before any assignment.

        Parameters
        ----------
        state : AlgorithmState[ReferencePayload]
            Concrete known-version snapshot with this exact owner, algorithm,
            codec and complete configuration, including original signed zeros.

        Raises
        ------
        ValueError, TypeError
            Future/unknown/wrong type, binding/configuration mismatch, invalid
            statistics or inconsistent recomputable anytime/initial weights.

        Notes
        -----
        This is no migration, source authentication or delivery rewind policy.
        A rejected restore preserves the original complete owner state.
        """
        if type(state) is not AlgorithmState[ReferencePayload]:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("Reference restore requires its exact concrete AlgorithmState payload")
        checked = AlgorithmState[ReferencePayload].model_validate(state)
        if checked.binding.model_dump_json() != self.binding.model_dump_json():
            raise ValueError("Reference restore differs from its exact owner/method/configuration/codec binding")
        if checked.payload.configuration.to_json() != self.config.to_json():
            raise ValueError("Reference restore changes its complete fixed configuration")
        payload = checked.payload
        if payload.update_count == 0 or self.config.profile == "hedge-anytime-cumulative":
            expected = normalize_positive_prior(self.config.prior.weights)
            if payload.update_count:
                losses = tuple(
                    fsum(pair) for pair in zip(payload.cumulative_losses, payload.compensations, strict=True)
                )
                try:
                    expected = exp_update(expected, losses, _rate(self.config, payload.update_count))
                except LearningNumericalError as error:
                    raise ValueError("Restored anytime state cannot reproduce a finite emitted vector") from error
            if canonical_json([item.model_dump(mode="json") for item in expected]) != canonical_json(
                [item.model_dump(mode="json") for item in payload.weights]
            ):
                raise ValueError("Restored initial/anytime coefficients disagree with complete sufficient statistics")
        candidate = _state(checked.binding, payload)
        self._state = candidate


class Hedge(_ReferenceLearner):
    """Own constant-rate exponential weights over a fixed positive-prior panel.

    Parameters
    ----------
    config : ReferenceConfig
        Exact hedge-constant law, positive prior and bounded loss declaration.
    owner : Reference
        Explicit keyword-only logical state owner and version.

    Notes
    -----
    Uses emitted p_i*exp(-eta*loss_i), stable normalized binary64 arithmetic.
    Ideal positive-rate synchronous best-fixed comparisons require their stated
    hypotheses; this class promises no asynchronous or domain-return result.
    Calls on one owner must be sequential; this class is not thread-synchronized.
    Replacing an immutable payload does not prevent concurrent lost updates.
    """

    _profile = "hedge-constant"


class AnytimeHedge(_ReferenceLearner):
    """Own current-rate cumulative-loss Hedge with an explicit accepted-U clock.

    Parameters
    ----------
    config : ReferenceConfig
        Exact hedge-anytime-cumulative law and positive schedule multiplier.
    owner : Reference
        Explicit keyword-only logical state owner and version.

    Notes
    -----
    After U panels, next coefficients use eta_(U+1)=c*sqrt(log(n)/(U+1))
    against complete cumulative prior loss; c defaults2. No historical-rate
    multiplication, doubling reset or hidden decision-clock substitution runs.
    Singletons emit one. General-prior comparisons are the separately stated
    ideal normalized-prior derivation, not an unconditional finite theorem.
    Calls on one owner must be sequential; this class is not thread-synchronized.
    Replacing an immutable payload does not prevent concurrent lost updates.
    """

    _profile = "hedge-anytime-cumulative"


class FixedShare(_ReferenceLearner):
    """Own constant-rate excluding-origin Fixed Share, including both endpoints.

    Parameters
    ----------
    config : ReferenceConfig
        Exact fixed-share-excluding-origin law, eta and alpha in[0,1].
    owner : Reference
        Explicit keyword-only logical state owner and version.

    Notes
    -----
    After each accepted complete panel, normalized exponential q becomes
    (1-alpha)q_i+alpha*sum(j!=i,q_j)/(n-1). This convention differs from sharing
    back to every origin or toward a declared target. n1 is absorbing one.
    Order dependence and positive-prior path-comparator hypotheses remain.
    Calls on one owner must be sequential; this class is not thread-synchronized.
    Replacing an immutable payload does not prevent concurrent lost updates.
    """

    _profile = "fixed-share-excluding-origin"
