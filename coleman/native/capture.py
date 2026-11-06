"""Explicit CI edge for one real native choice and its pre-evaluation snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from coleman.capabilities import require_capability
from coleman.native.decisions import (
    OBSERVATION_SEMANTICS,
    FrozenNativeDecision,
    NativeDecisionContext,
    NativeLossUndefined,
    NativeSamplingUnavailable,
    native_binding,
)

if TYPE_CHECKING:
    from coleman.agent import RewardAgent


def _names(values: object, *, label: str) -> tuple[str, ...]:
    """Copy exact native string members without stripping, sorting or coercion."""
    # Exact native strings retain their identity without subclass conversions.
    # pylint: disable-next=unidiomatic-typecheck
    if not isinstance(values, list | tuple) or any(type(value) is not str for value in values):
        raise ValueError(f"{label} must contain exact native string names, never null or coerced IDs")
    names = cast(tuple[str, ...], tuple(values))
    if len(names) != len(set(names)):
        raise ValueError(f"{label} must contain unique names; native duplicates cannot be silently discarded")
    return names


def _check_binding(context: NativeDecisionContext) -> None:
    """Refuse changed or unsupported definitions before importing native backends."""
    binding = context.binding
    route = binding.route.semantic_ref.id.rsplit(".", 1)[-1]
    current = native_binding(binding.policy.name, binding.reward.name, route=route)
    if binding != current:
        raise ValueError("Native capture requires the exact installed catalog definitions and fingerprint")
    if binding.route.status == "unsupported":
        raise ValueError(
            f"Unsupported native route {binding.route.semantic_ref.id}: "
            + "; ".join(binding.route.restrictions)
            + ". Consult the retained migration guidance or use the unmodified legacy API explicitly."
        )


def _counter(value: object) -> int:
    """Reject coerced or negative observation counts before a stateful selection."""
    # Boolean and foreign integer representations are not accepted step counts.
    # pylint: disable-next=unidiomatic-typecheck
    if type(value) is not int or value < 0:
        raise ValueError("Native agent.t must be a nonnegative integer in accepted-observation steps")
    return value


def _window(agent: RewardAgent, context: NativeDecisionContext) -> int | None:
    """Read the exact route's actual window; this is not a wall-clock duration."""
    value = getattr(agent, "window_size", None)
    if context.binding.route.window_mode == "required":
        # A boolean window must not be accepted as integer one.
        # pylint: disable-next=unidiomatic-typecheck
        if type(value) is not int or value <= 0:
            raise ValueError("Native window requires a positive integer count of accepted-observation steps")
        return value
    if value is not None:
        raise ValueError("Non-window native route cannot silently retain a window_size")
    return None


def _check_classes(agent: RewardAgent, context: NativeDecisionContext) -> None:
    """Check closed built-in classes without importing a caller-specified module."""
    # The explicit CI guard precedes these optional backend imports.
    from coleman import policy as policies  # pylint: disable=import-outside-toplevel
    from coleman import reward as rewards  # pylint: disable=import-outside-toplevel
    from coleman.agent import (  # pylint: disable=import-outside-toplevel
        ContextualAgent,
        RewardAgent,
        RewardSlidingWindowAgent,
        SlidingWindowContextualAgent,
    )

    known_agents = (RewardAgent, ContextualAgent, RewardSlidingWindowAgent, SlidingWindowContextualAgent)
    binding = context.binding
    actual_class = f"{type(agent).__module__}.{type(agent).__qualname__}"
    if type(agent) not in known_agents or actual_class != binding.route.agent_class:
        raise ValueError("Native capture requires the exact audited built-in agent class; subclasses need an adapter")
    policy_type = getattr(policies, binding.policy.policy_class.rsplit(".", 1)[-1], None)
    reward_type = getattr(rewards, binding.reward.reward_class.rsplit(".", 1)[-1], None)
    # An inherited override is a new implementation requiring its own adapter.
    # pylint: disable-next=unidiomatic-typecheck
    if type(agent.policy) is not policy_type or type(agent.reward_function) is not reward_type:
        raise ValueError("Actual native policy/reward classes differ from the selected audited definitions")


def choose_and_freeze_native(agent: RewardAgent, *, context: NativeDecisionContext) -> FrozenNativeDecision:
    """Choose exactly once and immediately freeze native pre-evaluation evidence.

    Parameters
    ----------
    agent : RewardAgent
        Exact audited built-in agent with the current bandit/actions/context
        already prepared. The caller exclusively owns this agent, policy and
        any shared process-global RNG during the complete call.
    context : NativeDecisionContext
        Revalidated stable logical coordinates, exact semantic binding, expected
        pre-observe counter and retained input/configuration/producer digests.

    Returns
    -------
    FrozenNativeDecision
        One complete copied ranking and its pre-observe metadata. This does not
        evaluate tests, apply credit, advance the observation counter or save state.

    Raises
    ------
    ValueError, TypeError
        Invalid context, unknown/unsupported route, different built-in class,
        counter/window mismatch, duplicate/null names, or incomplete/retired
        emitted members. Metadata preconditions reject before choosing; result
        validation occurs after the real stateful choice and cannot roll it back.
    ImportError
        The explicit ``ci`` installation capability is unavailable.
    Exception
        Native choice errors propagate. They can leave partially changed native
        caches/RNG and produce no completed snapshot.

    Notes
    -----
    This stateful call is neither thread safe nor idempotent. ``choose`` can
    mutate policy caches and owned/global RNG, including forecasts and calls at
    the same observation count. Keep the returned record and retry its immutable
    publication; never choose again as a storage retry. There is no user callback
    or await between choice and copying. The helper does not alter Environment,
    legacy results, checkpoint formats or generic core contracts. Digests and
    exact class checks are consistency declarations, not execution attestation.
    """
    context = NativeDecisionContext.model_validate(context)
    _check_binding(context)
    require_capability("ci")
    _check_classes(agent, context)
    count = _counter(agent.t)
    if count != context.expected_round:
        raise ValueError("Native capture must precede observe at the explicitly expected round")
    window = _window(agent, context)
    candidates = _names(agent.actions["Name"].to_list(), label="Native candidates")
    ranking = _names(agent.choose(), label="Native emitted ranking")
    if _counter(agent.t) != count or _window(agent, context) != window:
        raise ValueError("Native choose changed the observation counter or history-window definition")
    if _names(agent.actions["Name"].to_list(), label="Native post-choice candidates") != candidates:
        raise ValueError("Native choose changed candidate membership/order; no completed snapshot is emitted")
    if _names(agent.last_prioritization, label="Native last prioritization") != ranking:
        raise ValueError("Native returned ranking differs from its retained last_prioritization")
    return FrozenNativeDecision(
        schema_version=1,
        context=context,
        candidates=candidates,
        ranking=ranking,
        observation_round=count,
        observation_semantics=OBSERVATION_SEMANTICS,
        window_size=window,
        phase="initial_shuffle" if count == 0 and not context.binding.route.context_required else "policy_ranking",
        sampling=NativeSamplingUnavailable(
            status="not_recorded",
            value=None,
            reason="Native choose does not retain the joint emitted-ranking sampling mass; scores are not propensities",
        ),
        loss=NativeLossUndefined(
            status="not_defined", value=None, reason="Native credit has no declared generic bounded-loss transform"
        ),
    )
