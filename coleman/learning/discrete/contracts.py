"""Versioned fixed-catalogue, immediate chosen-reward bandit contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from coleman.core.actions import ChoiceSpace
from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.decisions import ExecutedDecision
from coleman.core.feedback import Origin
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64, ScalarSpec
from coleman.core.primitives import Identifier
from coleman.state.random_state import PythonRandomState

GRID_SIZE = 2**53
MAX_ARMS = 256
Count = Annotated[int, Field(strict=True, ge=0, le=2**53)]
Ticket = Annotated[int, Field(strict=True, ge=0, le=GRID_SIZE)]
Profile = Literal["exp3-dyadic53-v1", "ucb1-binary64-v1"]


class NumericalRuntime(CoreImmutable):
    """Bind continuation to an explicitly matching interpreter and host platform.

    Attributes
    ----------
    python, implementation : str
        Exact Python version/build description and interpreter implementation.
    platform, machine : str
        Operating-system platform and processor architecture.
    os_release, os_version : str
        Exact host kernel release and version. Matching fields are necessary,
        not a proof of identical arbitrarily replaced math libraries. The
        qualified continuation envelope is the same compatible host environment.
    """

    python: Identifier
    implementation: Identifier
    platform: Identifier
    machine: Identifier
    os_release: Identifier
    os_version: Identifier


class DiscreteConfig(CoreContract):
    """Bind one synchronous learner to explicit reward units and fixed arms.

    Attributes
    ----------
    profile : str
        Separately versioned numerical method; neither names a ranking policy.
    task, objective, transform : Reference
        Exact task, maximized objective and externally applied reward transform.
    space : ChoiceSpace
        One through 256 fixed, canonically ordered alternatives.
    reward : ScalarSpec
        Dimensionless reward with role reward and exact inclusive bounds [0, 1].
        Signed source values require an explicit external transform.
    origins : tuple of str, optional
        Permitted observations; modeled counterfactual observations are refused.
    gamma : float or None, optional
        Exp3 exploration coefficient in [K/2**53, 1]; eta is gamma/K. Must be
        explicitly null for UCB1. No tuning or transformation is performed.
    maximum_updates : int, optional
        Finite numerical horizon, at most 2**40 completed chosen observations.

    Notes
    -----
    Fixed availability, untimed immediate root feedback and one outstanding
    decision only. No contexts, revisions, costs, inventory or delays. Unknown
    versions reject; legacy ranking policy state is not a migration source.
    """

    profile: Profile
    task: Reference
    objective: Reference
    transform: Reference
    space: ChoiceSpace
    reward: ScalarSpec
    origins: tuple[Origin, ...] = ("realized", "synthetic")
    gamma: Binary64 | None = None
    maximum_updates: Annotated[int, Field(strict=True, ge=1, le=2**40)] = 1_000_000

    @model_validator(mode="after")
    def _supported(self) -> Self:
        """Reject unsupported scientific and numerical profiles without coercion."""
        size = len(self.space.members)
        if not 1 <= size <= MAX_ARMS:
            raise ValueError("Discrete profiles require 1 through 256 fixed alternatives")
        self._reward_contract()
        if not self.origins or len(set(self.origins)) != len(self.origins):
            raise ValueError("Origins must be nonempty and unique")
        if "modeled_counterfactual" in self.origins:
            raise ValueError("Chosen learners do not admit modeled alternative outcomes")
        object.__setattr__(self, "origins", tuple(sorted(self.origins)))
        self._parameters(size)
        return self

    def _reward_contract(self) -> None:
        """Require the exact mathematical reward domain without transforming it."""
        # These are exact schema endpoints, rather than rounded measurements.
        if (
            self.reward.role != "reward"
            or self.reward.semantics.unit != "1"
            or self.reward.minimum != 0
            or self.reward.maximum != 1
        ):
            raise ValueError("Reward must explicitly declare dimensionless reward units and bounds [0, 1]")

    def _parameters(self, size: int) -> None:
        """Validate profile-specific parameters and finite exploration support."""
        if self.profile == "exp3-dyadic53-v1":
            if self.gamma is None or not size / GRID_SIZE <= self.gamma <= 1.0:
                raise ValueError("Exp3 gamma must guarantee at least one exploration ticket per alternative")
        elif self.gamma is not None:
            raise ValueError("UCB1 does not have an exploration gamma parameter")

    @property
    def implementation(self) -> Reference:
        """Return the exact independent method identity without registry lookup.

        Returns
        -------
        Reference
            Public fixed-catalogue profile and its numerical generation.
        """
        return Reference(id=self.profile, version="v1")


class PendingChoice(CoreImmutable):
    """Retain original producer, complete finite support and actual integer draw.

    Attributes
    ----------
    executed : ExecutedDecision
        Unchanged host execution of the original proposal; never a later policy.
    tickets : tuple of int
        Canonical arm masses summing to 2**53. UCB1 has one nonzero coordinate.
    draw : int or None
        Owned uniform 53-bit draw for Exp3; null for deterministic UCB1.
    """

    executed: ExecutedDecision
    tickets: Annotated[tuple[Ticket, ...], Field(min_length=1, max_length=MAX_ARMS)]
    draw: Annotated[int, Field(strict=True, ge=0, lt=GRID_SIZE)] | None

    @model_validator(mode="after")
    def _mass(self) -> Self:
        """Retain exact total finite-grid probability without renormalization."""
        if sum(self.tickets) != GRID_SIZE:
            raise ValueError("Ticket counts must sum exactly to 2**53")
        return self


class DiscreteState(CoreContract):
    """Complete immutable owned state at action and feedback boundaries.

    Attributes
    ----------
    config : DiscreteConfig
        Exact configuration, units, transforms and profile compatibility binding.
    runtime : NumericalRuntime
        Exact interpreter build, platform, OS release/version and architecture.
        Cross-runtime/platform continuation rejects before state replacement.
    counts, rewards, log_weights : tuple
        Actual per-arm observations, bounded reward sums and centered Exp3 log
        weights; UCB1 log weights stay zero. Rewards are dimensionless.
    generator : PythonRandomState or None
        All owned MT19937 words, index and Gaussian cache, not merely a seed.
    pending : PendingChoice or None
        Complete previously emitted producer and sampler state until observation.

    Notes
    -----
    Reading a snapshot does not resume it. Restore additionally checks its exact
    configuration, runtime and original proposal. No storage or durable delivery
    acknowledgement is implied. Schema1 is the only readable generation.
    """

    config: DiscreteConfig
    runtime: NumericalRuntime
    counts: Annotated[tuple[Count, ...], Field(min_length=1, max_length=MAX_ARMS)]
    rewards: Annotated[tuple[Binary64, ...], Field(min_length=1, max_length=MAX_ARMS)]
    log_weights: Annotated[tuple[Binary64, ...], Field(min_length=1, max_length=MAX_ARMS)]
    generator: PythonRandomState | None
    pending: PendingChoice | None = None

    @model_validator(mode="after")
    def _statistics(self) -> Self:
        """Validate finite sufficient statistics, support and pending ownership."""
        size = len(self.config.space.members)
        if any(len(values) != size for values in (self.counts, self.rewards, self.log_weights)):
            raise ValueError("State statistics must match the fixed catalogue")
        if sum(self.counts) > self.config.maximum_updates:
            raise ValueError("State exceeds its declared numerical horizon")
        if any(not 0 <= reward <= count for reward, count in zip(self.rewards, self.counts, strict=True)):
            raise ValueError("Reward totals must lie between zero and actual chosen counts")
        self._numerical_profile()
        return self

    def _numerical_profile(self) -> None:
        """Check method-owned RNG, centered logs and original pending support."""
        if max(self.log_weights) != 0 or min(self.log_weights) < -700:
            raise ValueError("Centered log weights exceed the supported [-700, 0] numerical envelope")
        exp3 = self.config.profile == "exp3-dyadic53-v1"
        if exp3 != (self.generator is not None) or (not exp3 and any(self.log_weights)):
            raise ValueError("RNG ownership and log weights must match the method profile")
        if self.pending is not None and len(self.pending.tickets) != len(self.counts):
            raise ValueError("Pending support differs from the fixed catalogue")
