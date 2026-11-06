"""Frozen native ranking evidence, separate from generic learner contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint, Identifier, fingerprint
from coleman.core.units import UnitSemantics
from coleman.native.semantics import (
    NativeAgentRoute,
    NativePolicyEntry,
    NativeRewardDefinition,
    get_native_policy,
    get_native_reward,
    native_catalog,
)

NativeNames = tuple[Annotated[str, Field(strict=True)], ...]
ObservationCount = Annotated[int, Field(strict=True, ge=0)]
WindowCount = Annotated[int, Field(strict=True, gt=0)]
OBSERVATION_SEMANTICS = UnitSemantics(unit="step", meaning="coleman.native.accepted_observation_count.v1")


class NativeSemanticBinding(CoreImmutable):
    """Retain the actual native definitions rather than inferring them by name.

    Attributes
    ----------
    catalog_fingerprint : str
        Content digest of the retained, versioned source-audit catalog.
    policy : NativePolicyEntry
        Complete native ranking definition, including its source evidence.
    route : NativeAgentRoute
        Exact factory or manual agent route belonging to this policy.
    reward : NativeRewardDefinition
        Native credit definition and units; not a generic bounded loss.

    Notes
    -----
    Construction checks consistency without importing a backend or consulting
    a registry. A declaration and its digest do not authenticate execution.
    The capture boundary additionally checks the installed catalog and classes.
    """

    catalog_fingerprint: Fingerprint
    policy: NativePolicyEntry
    route: NativeAgentRoute
    reward: NativeRewardDefinition

    @model_validator(mode="after")
    def _route(self) -> Self:
        """Require the complete selected route, not only an ambiguous route ID."""
        if self.route not in self.policy.routes:
            raise ValueError("Native route must be an exact member of the retained policy definition")
        return self


def native_binding(policy: str, reward: str, *, route: str = "factory") -> NativeSemanticBinding:
    """Select audited native definitions without loading algorithms.

    Parameters
    ----------
    policy, reward : str
        Exact case-sensitive catalog names, for example ``EXP3`` and ``RNFail``.
    route : {'factory', 'manual-window'}, optional
        Explicit construction route. Unsupported factory routes remain cataloged
        but are refused by ``choose_and_freeze_native`` before choosing.

    Returns
    -------
    NativeSemanticBinding
        Immutable self-contained definitions and their catalog content digest.

    Raises
    ------
    ValueError
        Unknown policy, reward or route. No approximate name matching occurs.
    OSError
        The installed local catalog resource could not be read.

    Notes
    -----
    Lazily reads the packaged data-only catalog. Does not instantiate an agent,
    execute a reward, import CI dependencies or change legacy dispatch.
    """
    entry = get_native_policy(policy)
    selected = next((item for item in entry.routes if item.semantic_ref.id.endswith("." + route)), None)
    if selected is None:
        raise ValueError(f"Native policy {policy!r} has no audited route {route!r}")
    return NativeSemanticBinding(
        catalog_fingerprint=native_catalog().fingerprint,
        policy=entry,
        route=selected,
        reward=get_native_reward(reward),
    )


class NativeDecisionContext(CoreContract):
    """Declare stable host identities and retained inputs before native choice.

    Attributes
    ----------
    schema_version : int
        Required integer 1; future or coerced versions reject.
    run : Reference
        Stable host run identity and semantic version.
    execution_id, decision_id : str
        Logical execution and decision coordinates, stable across publication
        retries. Forecast calls require their own decision ID.
    purpose : {'evaluation_ranking', 'forecast'}
        Host-declared use; neither value proves evaluation or external execution.
    expected_round : int
        Expected pre-observe ``agent.t`` in accepted-observation steps, not time.
    binding : NativeSemanticBinding
        Exact native policy, agent-route and reward definitions.
    configuration_fingerprint, input_fingerprint, producer_fingerprint : str
        Digests of separately retained configuration, current input and producer
        inventory. The host retains those artifacts; this is not attestation.
    policy_state_group : Reference or None
        Declared stable identity of shared policy state, or explicitly unknown.
        An object address, worker PID or seed is not such an identity.

    Notes
    -----
    Requires no clocks, transport, algorithm or storage. Candidate names are
    captured verbatim separately and do not become generic Reference IDs.
    """

    run: Reference
    execution_id: Identifier
    decision_id: Identifier
    purpose: Literal["evaluation_ranking", "forecast"]
    expected_round: ObservationCount
    binding: NativeSemanticBinding
    configuration_fingerprint: Fingerprint
    input_fingerprint: Fingerprint
    producer_fingerprint: Fingerprint
    policy_state_group: Reference | None

    @property
    def logical_identity(self) -> str:
        """Return the host logical key independently of content or delivery attempts.

        Returns
        -------
        str
            Identified SHA-256 over the family, run and logical coordinates.

        Notes
        -----
        A changed purpose, input or ranking at the same key conflicts when
        published. No I/O, global registration or uniqueness proof occurs.
        """
        return fingerprint(
            {
                "family": "coleman.native.decision.v1",
                "run": self.run.model_dump(mode="json"),
                "execution_id": self.execution_id,
                "decision_id": self.decision_id,
            }
        )


class NativeSamplingUnavailable(CoreImmutable):
    """State that the actual native ranking's sampling mass was not retained.

    Attributes
    ----------
    status : {'not_recorded'}
        No native ranking propensity is available through this capture profile.
    value : None
        Explicit absence, distinct from zero probability.
    reason : str
        Actionable explanation; internal scores are not substituted for mass.
    """

    status: Literal["not_recorded"]
    value: None
    reason: Identifier


class NativeLossUndefined(CoreImmutable):
    """Keep undefined native loss distinct from zero credit or unavailable data.

    Attributes
    ----------
    status : {'not_defined'}
        This legacy ranking/reward route declares no generic bounded loss.
    value : None
        Explicit absence; no implicit ``1 - reward`` transform is performed.
    reason : str
        Explanation of the missing loss definition, not a paid cost or outcome.
    """

    status: Literal["not_defined"]
    value: None
    reason: Identifier


class FrozenNativeDecision(CoreContract):
    """Retain one native ranking immediately after choice and before evaluation.

    Attributes
    ----------
    schema_version : int
        Required integer 1, independent of native behavior versions.
    context : NativeDecisionContext
        Stable logical key, exact semantics and retained-artifact digests.
    candidates : tuple of str
        Current action-table order, preserving whitespace, Unicode and empty
        native strings. Null and duplicate names reject; no 4096-member cap.
    ranking : tuple of str
        Complete emitted permutation of exactly the candidates, highest first.
    observation_round : int
        Actual pre-observe counter in accepted-observation steps.
    observation_semantics : UnitSemantics
        Exact ``step``/``coleman.native.accepted_observation_count.v1`` meaning.
    window_size : int or None
        Actual positive history window in accepted-observation steps, or absent
        for the selected non-window route.
    phase : {'initial_shuffle', 'policy_ranking'}
        Actual agent path. Contextual agents delegate even at counter zero.
    sampling : NativeSamplingUnavailable
        Explicitly unrecorded action mass; scores are never probabilities here.
    loss : NativeLossUndefined
        Explicit missing loss definition, independent of native credit units.

    Notes
    -----
    This is pre-evaluation evidence, not a completed observation, checkpoint,
    causal certificate, propensity estimator or atomic execution transaction.
    Old native artifacts remain readable by their old readers and are never
    backfilled with this stronger claim. Canonical JSON retains ranking order.
    """

    context: NativeDecisionContext
    candidates: NativeNames
    ranking: NativeNames
    observation_round: ObservationCount
    observation_semantics: UnitSemantics
    window_size: WindowCount | None
    phase: Literal["initial_shuffle", "policy_ranking"]
    sampling: NativeSamplingUnavailable
    loss: NativeLossUndefined

    @model_validator(mode="after")
    def _native_ranking(self) -> Self:
        """Check exact members, units, counter, window and actual selection path."""
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("Native candidate names must be unique; no deduplication is performed")
        if len(self.ranking) != len(self.candidates) or set(self.ranking) != set(self.candidates):
            raise ValueError("Native ranking must be an exact full permutation of current candidates")
        if self.observation_round != self.context.expected_round:
            raise ValueError("Native counter differs from the expected pre-observe round")
        if self.observation_semantics != OBSERVATION_SEMANTICS:
            raise ValueError("Native counter requires its exact accepted-observation step semantics")
        route = self.context.binding.route
        if (self.window_size is not None) != (route.window_mode == "required"):
            raise ValueError("Native window presence must match the exact selected agent route")
        expected_phase = (
            "initial_shuffle" if self.observation_round == 0 and not route.context_required else "policy_ranking"
        )
        if self.phase != expected_phase:
            raise ValueError("Native phase must describe the actual agent choose path")
        return self

    @property
    def logical_identity(self) -> str:
        """Return the host key; full content has the separate inherited fingerprint.

        Returns
        -------
        str
            Stable logical decision digest, independent of the emitted ranking.
        """
        return self.context.logical_identity
