"""Source-pinned native ranking definitions without importing their implementations.

Catalog references identify preserved behavior, not original-paper guarantees or
execution attestations. Loading is explicit and lazy; importing this module does
not inspect installed packages, initialize a native RNG or read the data resource.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.identity import Reference
from coleman.core.numeric import Binary64
from coleman.core.primitives import Fingerprint, Identifier
from coleman.core.units import UnitSemantics

BASELINE_REVISION: Final = "1edbee874129aa85e26e592d9649826e85e1ac6b"
NATIVE_SEMANTIC_VERSION: Final = "legacy-4.20.0-v1"
NATIVE_POLICY_NAMES: Final = (
    "BayesianUCB",
    "BootstrappedThompson",
    "ChangeDetectionUCB",
    "CombinatorialThompson",
    "CombinatorialUCB",
    "ContextualEpsilonGreedy",
    "DecayEpsilonGreedy",
    "DiscountedUCB",
    "DuelingUCB",
    "EpsilonDecreasing",
    "EpsilonGreedy",
    "EXP3",
    "EXP3IX",
    "FRRMAB",
    "Greedy",
    "KLUCB",
    "LinTS",
    "LinUCB",
    "MOSSUCB",
    "OptimisticGreedy",
    "PairwiseThompsonRanking",
    "PHE",
    "PortfolioUCB",
    "Pursuit",
    "Random",
    "SlidingWindowUCB",
    "SlMAB",
    "Softmax",
    "SWContextualEpsilonGreedy",
    "SWLinTS",
    "SWLinUCB",
    "ThompsonSampling",
    "UCB",
    "UCB1",
    "UCB2",
    "UCBTuned",
    "UCBV",
)
NATIVE_REWARD_NAMES: Final = ("APFDc", "DiscountedFailure", "ReciprocalRank", "RNFail", "TimeRank", "TopKRNFail")


class NativeSourcePin(CoreImmutable):
    """Locate the exact public source supporting a semantic declaration.

    Attributes
    ----------
    revision : str
        Public baseline Git commit, not an execution or authenticity claim.
    path : str
        Package-relative Python source path, never dynamically imported.
    fingerprint : str
        SHA-256 of the entire original file at that revision.
    first_line, last_line : int
        Inclusive positive line range within the pinned file.
    """

    revision: Literal["1edbee874129aa85e26e592d9649826e85e1ac6b"]
    path: Annotated[Identifier, Field(pattern=r"^coleman/[A-Za-z0-9_/]+\.py$")]
    fingerprint: Fingerprint
    first_line: Annotated[int, Field(strict=True, ge=1)]
    last_line: Annotated[int, Field(strict=True, ge=1)]

    @model_validator(mode="after")
    def _range(self) -> Self:
        """Reject reversed source ranges without reading or executing a file."""
        if self.first_line > self.last_line:
            raise ValueError("Native source line range is reversed")
        return self


class RequiredDefault(CoreImmutable):
    """Declare a required constructor argument without inventing its value.

    Attributes
    ----------
    kind : {'required'}
        Distinguishes absent defaults from zero, false and an explicit value.
    value : None
        Required explicit null because no constructor default exists.
    """

    kind: Literal["required"]
    value: None


class NumberDefault(CoreImmutable):
    """Retain a finite binary64 constructor default.

    Attributes
    ----------
    kind : {'number'}
        Floating constructor-default representation.
    value : float
        Finite guarded binary64; signed zero is retained.
    """

    kind: Literal["number"]
    value: Binary64


class IntegerDefault(CoreImmutable):
    """Retain an integer default without retyping it as floating point.

    Attributes
    ----------
    kind : {'integer'}
        Integer representation, distinct from boolean and binary64.
    value : int
        Strict source default; its units belong to its parameter definition.
    """

    kind: Literal["integer"]
    value: Annotated[int, Field(strict=True)]


class BooleanDefault(CoreImmutable):
    """Retain an explicit boolean constructor default.

    Attributes
    ----------
    kind : {'boolean'}
        Boolean representation, never an integer-count default.
    value : bool
        Strict source default, including false.
    """

    kind: Literal["boolean"]
    value: Annotated[bool, Field(strict=True)]


ParameterDefault = Annotated[
    RequiredDefault | NumberDefault | IntegerDefault | BooleanDefault, Field(discriminator="kind")
]


class NativeParameterDefinition(CoreImmutable):
    """Describe a legacy constructor argument without running its validator.

    Attributes
    ----------
    name : str
        Exact constructor parameter name.
    kind : {'number', 'integer', 'boolean', 'policy_sequence'}
        Semantic argument shape; numeric parameters may have integer defaults.
    semantics : UnitSemantics
        Explicit units and distinct parameter meaning; undeclared native feature
        scaling remains undeclared rather than acquiring physical units.
    default : RequiredDefault or NumberDefault or IntegerDefault or BooleanDefault
        Required/default status and literal numeric representation.
    definition : str
        Human description of the actual parameter/clock used by the code.
    constraints : str
        Actual legacy validation and limits, not a newly enforced domain.
    """

    name: Identifier
    kind: Literal["number", "integer", "boolean", "policy_sequence"]
    semantics: UnitSemantics
    default: ParameterDefault
    definition: Identifier
    constraints: Identifier

    @model_validator(mode="after")
    def _default_type(self) -> Self:
        """Keep required, numeric, count and boolean defaults distinct."""
        permitted = {
            "number": (RequiredDefault, NumberDefault, IntegerDefault),
            "integer": (RequiredDefault, IntegerDefault),
            "boolean": (RequiredDefault, BooleanDefault),
            "policy_sequence": (RequiredDefault,),
        }
        if not isinstance(self.default, permitted[self.kind]):
            raise ValueError("Native parameter default has the wrong representation")
        return self


class NativeRNGProfile(CoreImmutable):
    """Declare real selection randomness and tie limits independently of mass.

    Attributes
    ----------
    warmup : {'agent_generator', 'policy_from_first_call'}
        Native agent-local initial shuffle or contextual direct policy route.
    policy : {'none', 'shared_generator', 'selected_child'}
        Policy RNG ownership after warmup; no independent stream is implied.
    operations : tuple[str, ...]
        Actual draw/shuffle operations or an explicit no-draw description.
    ties : str
        Actual ordering rule and qualified determinism limitations.
    """

    warmup: Literal["agent_generator", "policy_from_first_call"]
    policy: Literal["none", "shared_generator", "selected_child"]
    operations: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=32)]
    ties: Identifier


class NativeLifecycleProfile(CoreImmutable):
    """Keep native resets, name changes, sharing and durability claims explicit.

    Attributes
    ----------
    reset, discovery, retirement : str
        Actual state lifetime; reset is not fresh construction or RNG rewind.
    sharing : str
        Agent/policy/window/candidate coupling that must not imply independence.
    checkpoint : {'not_qualified', 'unsupported', 'constructor_save_load_observed'}
        Narrow ordinary native pickle evidence, not workflow/RNG certification.
    checkpoint_definition : str
        Exact evidence and limitations of the recorded checkpoint disposition.
    """

    reset: Identifier
    discovery: Identifier
    retirement: Identifier
    sharing: Identifier
    checkpoint: Literal["not_qualified", "unsupported", "constructor_save_load_observed"]
    checkpoint_definition: Identifier


class NativeAgentRoute(CoreImmutable):
    """Name one exact agent/window/feedback route, including preserved refusals.

    Attributes
    ----------
    semantic_ref : Reference
        Explicit versioned route identity; factory and manual routes differ.
    agent_class : str
        Exact module plus qualified class name, never an import instruction.
    origin : {'factory', 'manual'}
        Actual runner dispatch versus a separately constructed window agent.
    window_mode : {'none', 'required'}
        Whether agent.window_size is required, in native observe/build steps.
    context_required : bool
        Required explicit contextual setup declaration.
    status : {'supported', 'restricted', 'unsupported'}
        Qualified native route disposition, never an original-theorem status.
    feedback_ref : Reference
        Exact per-route feedback semantic identity.
    feedback_definition : str
        What the agent and policy actually consume; credit is not raw outcomes.
    restrictions : tuple[str, ...]
        Required limits and actionable factory/manual guidance.
    source_evidence : tuple[NativeSourcePin, ...]
        Nonempty pinned public source evidence; no filesystem checks occur here.
    """

    semantic_ref: Reference
    agent_class: Identifier
    origin: Literal["factory", "manual"]
    window_mode: Literal["none", "required"]
    context_required: Annotated[bool, Field(strict=True)]
    status: Literal["supported", "restricted", "unsupported"]
    feedback_ref: Reference
    feedback_definition: Identifier
    restrictions: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=64)]
    source_evidence: Annotated[tuple[NativeSourcePin, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _references(self) -> Self:
        """Reject unknown behavior versions and mismatched feedback identities."""
        if self.semantic_ref.version != NATIVE_SEMANTIC_VERSION:
            raise ValueError("Unknown native route semantic version")
        expected = Reference(id=self.semantic_ref.id + ".feedback", version=NATIVE_SEMANTIC_VERSION)
        if self.feedback_ref != expected:
            raise ValueError("Native feedback reference differs from its route")
        if not self.agent_class.startswith("coleman.agent."):
            raise ValueError("Native route needs its exact built-in agent class path")
        return self


def _unique_names(values: tuple[NativeParameterDefinition, ...]) -> tuple[NativeParameterDefinition, ...]:
    """Reject conflicting parameters before canonicalizing declaration order."""
    if len({value.name for value in values}) != len(values):
        raise ValueError("Duplicate native constructor parameter")
    return tuple(sorted(values, key=lambda value: value.name))


class NativePolicyEntry(CoreContract):
    """Describe one preserved native ranking recipe without implementing it.

    Attributes
    ----------
    schema_version : int
        Required strict version 1; unknown versions reject.
    name, policy_class : str
        Exact public runner alias and actual module/qualified class name.
    semantic_ref : Reference
        Native behavior identity, separate from schema and package versions.
    constructor : str
        Literal inherited/declared signature; definitions do not instantiate it.
    parameters : tuple[NativeParameterDefinition, ...]
        Unique named defaults, units and actual validation, sorted by name.
    routes : tuple[NativeAgentRoute, ...]
        Distinct versioned factory/manual agent routes, sorted by identity.
    action_definition, counter_definition : str
        Actual ranking/feedback-clock formulas, including known deviations.
    rng : NativeRNGProfile
        Warmup, shared RNG and actual tie behavior; no propensity is inferred.
    lifecycle : NativeLifecycleProfile
        Reset, discovery, retirement, sharing and checkpoint qualifications.
    theory_status : {'not_claimed'}
        No original-paper theorem is certified by the native name or smoke run.
    theory_statement : str
        Actual assumption mismatch and limits of the qualified observations.
    migration : {'preserved_qualified', 'factory_refused', 'preserved_fallback', 'restricted'}
        Explicit legacy disposition; a correction requires another identity.
    guidance : str
        Actionable supported native/manual route and unsupported generic use.
    source_evidence : tuple[NativeSourcePin, ...]
        Original public revision and full-file hashes supporting the definition.
    """

    name: Identifier
    semantic_ref: Reference
    policy_class: Identifier
    constructor: Identifier
    parameters: Annotated[tuple[NativeParameterDefinition, ...], Field(max_length=16)]
    routes: Annotated[tuple[NativeAgentRoute, ...], Field(min_length=1, max_length=4)]
    action_definition: Identifier
    counter_definition: Identifier
    rng: NativeRNGProfile
    lifecycle: NativeLifecycleProfile
    theory_status: Literal["not_claimed"]
    theory_statement: Identifier
    migration: Literal["preserved_qualified", "factory_refused", "preserved_fallback", "restricted"]
    guidance: Identifier
    source_evidence: Annotated[tuple[NativeSourcePin, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Validate this fixed native family without importing a class registry."""
        expected = Reference(id="coleman.native.ranking." + self.name, version=NATIVE_SEMANTIC_VERSION)
        if self.name not in NATIVE_POLICY_NAMES or self.semantic_ref != expected:
            raise ValueError("Unknown native policy identity or semantic version")
        if not self.policy_class.startswith("coleman.policy.") or not self.policy_class.endswith(
            "." + self.name + "Policy"
        ):
            raise ValueError("Native policy class differs from its runner alias")
        route_ids = [route.semantic_ref.id for route in self.routes]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("Duplicate native agent route")
        if any(not name.startswith(expected.id + ".") for name in route_ids):
            raise ValueError("Native route belongs to another policy")
        object.__setattr__(self, "parameters", _unique_names(self.parameters))
        object.__setattr__(self, "routes", tuple(sorted(self.routes, key=lambda route: route.semantic_ref.id)))
        return self


class NativeRewardDefinition(CoreContract):
    """Identify a native rank-credit formula separately from losses and outcomes.

    Attributes
    ----------
    schema_version : int
        Required strict version 1, independent of the native behavior reference.
    name, reported_name, reward_class : str
        Exact runner alias, get_name result and actual module/qualified class.
    semantic_ref : Reference
        Versioned native reward identity; no implicit loss conversion.
    semantics : UnitSemantics
        Exact dimensionless unit ``1`` and versioned reward-specific meaning;
        another unit or meaning requires a different qualified contract.
    formula : str
        Actual calculation and edge cases, rather than its theoretical nickname.
    input_fields : tuple[str, ...]
        Exact metric/ranking/constructor fields consumed by this formula.
    coverage : str
        Budget/unscheduled-credit meaning; zero credit is not an observed pass.
    parameters : tuple[NativeParameterDefinition, ...]
        Constructor defaults, including top-k and time-budget behavior.
    source_evidence : tuple[NativeSourcePin, ...]
        Pinned public source evidence; no import or execution is performed.
    """

    name: Identifier
    semantic_ref: Reference
    reported_name: Identifier
    reward_class: Identifier
    semantics: UnitSemantics
    formula: Identifier
    input_fields: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=32)]
    coverage: Identifier
    parameters: Annotated[tuple[NativeParameterDefinition, ...], Field(max_length=16)]
    source_evidence: Annotated[tuple[NativeSourcePin, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def _identity(self) -> Self:
        """Keep built-in reward aliases, semantics and defaults unambiguous."""
        expected = Reference(id="coleman.native.reward." + self.name, version=NATIVE_SEMANTIC_VERSION)
        if self.name not in NATIVE_REWARD_NAMES or self.semantic_ref != expected:
            raise ValueError("Unknown native reward identity or semantic version")
        if not self.reward_class.startswith("coleman.reward.") or not self.reward_class.endswith(
            "." + self.name + "Reward"
        ):
            raise ValueError("Native reward class differs from its runner alias")
        expected_semantics = UnitSemantics(unit="1", meaning=expected.id + "." + expected.version)
        if self.semantics != expected_semantics:
            raise ValueError("Native reward requires its exact dimensionless credit semantics")
        if len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError("Duplicate native reward input field")
        object.__setattr__(self, "parameters", _unique_names(self.parameters))
        return self


class NativeCatalog(CoreContract):
    """Retain the complete authored 37-policy/six-reward native inventory.

    Attributes
    ----------
    schema_version : int
        Required strict version 1. Unknown fields/versions reject.
    baseline_revision : str
        Exact public source revision supporting preserved behavior.
    policies : tuple[NativePolicyEntry, ...]
        Exactly the 37 unique public native runner aliases, sorted by name.
    rewards : tuple[NativeRewardDefinition, ...]
        Exactly six unique built-in reward aliases, sorted by name.

    Notes
    -----
    No code is dynamically loaded or certified by these definitions. Source
    pins identify public bytes; callers separately qualify actual execution.
    """

    baseline_revision: Literal["1edbee874129aa85e26e592d9649826e85e1ac6b"]
    policies: Annotated[tuple[NativePolicyEntry, ...], Field(min_length=37, max_length=37)]
    rewards: Annotated[tuple[NativeRewardDefinition, ...], Field(min_length=6, max_length=6)]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Reject omitted/duplicate aliases before deterministic inventory sorting."""
        if sorted(row.name for row in self.policies) != sorted(NATIVE_POLICY_NAMES):
            raise ValueError("Native catalog requires exactly the 37 unique policy aliases")
        if sorted(row.name for row in self.rewards) != sorted(NATIVE_REWARD_NAMES):
            raise ValueError("Native catalog requires exactly the six unique reward aliases")
        object.__setattr__(self, "policies", tuple(sorted(self.policies, key=lambda row: row.name)))
        object.__setattr__(self, "rewards", tuple(sorted(self.rewards, key=lambda row: row.name)))
        return self


@cache
def native_catalog() -> NativeCatalog:
    """Read and validate the immutable bundled native catalog on first use.

    Returns
    -------
    NativeCatalog
        Shared frozen inventory, independent of optional native providers.

    Raises
    ------
    ValueError, TypeError
        Bundled JSON, versions, references or typed invariants are invalid.
    OSError
        The installed resource cannot be read.

    Notes
    -----
    Reads only the local resource, then caches immutable data. It never imports
    policies, installs dependencies, initializes RNGs or verifies execution.
    Concurrent first reads may repeat harmless validation, without file writes.
    """
    payload = files("coleman.native").joinpath("catalog-v1.json").read_bytes()
    if len(payload) > 1024 * 1024:
        raise ValueError("Bundled native catalog exceeds its one-MiB resource limit")
    return NativeCatalog.from_json(payload)


def list_native_policies() -> tuple[NativePolicyEntry, ...]:
    """Return all source-pinned native policies in exact alias order.

    Returns
    -------
    tuple of NativePolicyEntry
        Frozen 37-entry inventory; listing does not run a native algorithm.

    Notes
    -----
    The first call reads the local catalog and propagates its validation/I/O
    errors. No optional backend or transport is imported.
    """
    return native_catalog().policies


def get_native_policy(name: str, *, use: str = "native_ranking") -> NativePolicyEntry:
    """Look up one native recipe and refuse unsupported generic interpretations.

    Parameters
    ----------
    name : str
        Exact case-sensitive public runner alias.
    use : str, optional
        Only ``native_ranking`` is supported. Sampled choice, subset, mixture,
        pairwise outcomes or calibrated inference need their own implementations.

    Returns
    -------
    NativePolicyEntry
        Frozen definition, including any restricted/unsupported agent routes.

    Raises
    ------
    ValueError
        Unknown alias or unsupported use. Route lookup never repairs a factory.
    OSError
        The bundled resource cannot be read; validation errors also propagate.

    Notes
    -----
    Does not instantiate classes, change defaults, initialize RNGs or convert a
    native score into an action propensity. The first valid lookup reads data.
    """
    if not isinstance(use, str) or use != "native_ranking":
        raise ValueError(
            "Native policies support native_ranking only; use an explicitly qualified generic learner for "
            "sampled choice, subsets, mixtures or observed pairwise feedback"
        )
    if not isinstance(name, str):
        raise ValueError("Native policy lookup requires an exact string alias")
    for entry in list_native_policies():
        if entry.name == name:
            return entry
    raise ValueError(f"Unknown native policy {name!r}; use list_native_policies() for exact runner names")


def list_native_rewards() -> tuple[NativeRewardDefinition, ...]:
    """Return all six source-pinned built-in rank-credit definitions.

    Returns
    -------
    tuple of NativeRewardDefinition
        Frozen definitions in exact alias order, without evaluating rewards.

    Notes
    -----
    The first call reads the local catalog and propagates its validation/I/O
    errors. Units identify transformed credit, never an automatic bounded loss.
    """
    return native_catalog().rewards


def get_native_reward(name: str) -> NativeRewardDefinition:
    """Find an exact reward alias without inferring it from a display string.

    Parameters
    ----------
    name : str
        Case-sensitive runner alias, such as ``RNFail`` or ``TimeRank``.

    Returns
    -------
    NativeRewardDefinition
        Source-pinned formula, units, inputs, defaults and coverage limits.

    Raises
    ------
    ValueError
        Unknown alias, including an unregistered custom reward.
    OSError
        The bundled resource cannot be read; validation errors also propagate.

    Notes
    -----
    Reads the catalog lazily and never imports/evaluates the reward class.
    ``reported_name`` separately retains native get_name spelling differences.
    """
    if not isinstance(name, str):
        raise ValueError("Native reward lookup requires an exact string alias")
    for entry in list_native_rewards():
        if entry.name == name:
            return entry
    raise ValueError(
        f"Unknown native reward {name!r}; use list_native_rewards() or retain its explicit custom legacy route"
    )
