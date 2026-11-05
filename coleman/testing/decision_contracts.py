"""Bounded structural checks for direct-use, host-authored decision consumers."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Annotated, Protocol, Self

from pydantic import Field, field_validator, model_validator

from coleman.core.actions import ChoiceAction, space_members
from coleman.core.base import CoreImmutable
from coleman.core.capabilities import (
    Capability,
    ProblemCapabilities,
    validate_composition,
    validate_decision,
    validate_feedback,
)
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import ChosenFeedback, FeedbackBatch, FeedbackRow, FullPanel
from coleman.core.primitives import MAX_MEMBERS, canonical_json
from coleman.core.protocols import Decider, Stateful, Updater

MAX_CONFORMANCE_CASES = 64


def _check_oracle(problem: ProblemCapabilities, outcomes: tuple[FeedbackRow, ...]) -> None:
    """Reject oracle rows contradicting the declared members, units or origins."""
    if tuple(row.member for row in outcomes) != space_members(problem.space):
        raise ValueError("Conformance oracle must cover every exact member and version")
    schemas = canonical_json([value.model_dump(mode="json") for value in problem.values])
    for row in outcomes:
        if canonical_json([value.spec.model_dump(mode="json") for value in row.values]) != schemas:
            raise ValueError("Conformance oracle outcome schemas differ from the problem")
        if any(value.origin not in problem.origins for value in row.values):
            raise ValueError("Conformance oracle contains an undeclared outcome origin")


class ConformanceConsumer[StateT](
    Decider[DecisionRequest, Decision], Updater[FeedbackBatch, object], Stateful[StateT], Protocol
):
    """Combine structural entry points without prescribing a state representation.

    Notes
    -----
    Host components need not inherit this protocol. Their state must support
    ``copy.deepcopy`` and contain every stateful dependency, including an owned
    random generator when applicable. No framework snapshot type is required.
    """


class ConformanceCase(CoreImmutable):
    """Retain an independently supplied, untimed full-information oracle.

    Attributes
    ----------
    problem : ProblemCapabilities
        Actual untimed action, units, bounds, origin and feedback declarations.
        Only ``full_panel`` and ``chosen`` coverage are supported by this helper.
        Receipt is explicitly unknown because no receipt clock is fabricated.
    request : DecisionRequest
        Exact request passed to the host. Its space must match ``problem`` byte
        for byte under COLEMAN JSON v1; signed zero is retained.
    outcomes : tuple of FeedbackRow
        Complete oracle, at most 4096 members, with exactly the declared outcome
        schemas and origins. Chosen-only consumers receive only the selected row.

    Raises
    ------
    ValueError
        Unsupported timing/coverage, mismatched spaces, missing/duplicate member
        identities, different outcome schemas or undeclared origins.

    Notes
    -----
    Oracle values are supplied by the caller, never inferred by the harness.
    They retain explicit units and may be synthetic; the declaration proves no
    causal, statistical, source-authentication or algorithmic claim.
    """

    problem: ProblemCapabilities
    request: DecisionRequest
    outcomes: Annotated[tuple[FeedbackRow, ...], Field(min_length=1, max_length=MAX_MEMBERS)]

    @field_validator("outcomes")
    @classmethod
    def _ordered(cls, rows: tuple[FeedbackRow, ...]) -> tuple[FeedbackRow, ...]:
        """Keep independent oracle rows canonical without hiding duplicates."""
        if len({row.member.id for row in rows}) != len(rows):
            raise ValueError("Conformance oracle contains duplicate member identities")
        return tuple(sorted(rows, key=lambda row: row.member.id))

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Reject a case whose oracle contradicts its explicit declarations."""
        if self.problem.timing_mode != "untimed" or self.request.timing is not None:
            raise ValueError("Conformance cases support untimed direct use only")
        if self.problem.delayed_feedback or self.problem.feedback_kind not in {"full_panel", "chosen"}:
            raise ValueError("Conformance cases require immediate full-panel or chosen feedback")
        if self.problem.receipt_status != "unknown":
            raise ValueError("Untimed conformance cases cannot declare a known receipt")
        if canonical_json(self.problem.space.model_dump(mode="json")) != canonical_json(
            self.request.space.model_dump(mode="json")
        ):
            raise ValueError("Conformance request and declared problem spaces differ")
        _check_oracle(self.problem, self.outcomes)
        return self


class ConformanceReport(CoreImmutable):
    """Return typed decisions and exactly the feedback delivered to consumers.

    Attributes
    ----------
    decisions : tuple of Decision
        Baseline decisions, one per supplied case, in caller execution order.
    feedback : tuple of FeedbackBatch
        Actual validated batches in that order. Chosen feedback exposes no other
        oracle member, regardless of the number of dimensions on its chosen row.

    Notes
    -----
    This is a bounded in-memory structural report, not an artifact, score,
    theorem, runtime checkpoint or scientific efficacy comparison.
    """

    decisions: Annotated[tuple[Decision, ...], Field(min_length=2, max_length=MAX_CONFORMANCE_CASES)]
    feedback: Annotated[tuple[FeedbackBatch, ...], Field(min_length=2, max_length=MAX_CONFORMANCE_CASES)]

    @model_validator(mode="after")
    def _paired(self) -> Self:
        """Require exactly one delivered feedback batch per reported decision."""
        if len(self.decisions) != len(self.feedback):
            raise ValueError("Conformance decisions and feedback must have equal lengths")
        return self


def _checked_cases(capability: Capability, cases: tuple[ConformanceCase, ...]) -> tuple[ConformanceCase, ...]:
    """Check every bounded case before invoking any host component."""
    if not isinstance(cases, tuple) or not 2 <= len(cases) <= MAX_CONFORMANCE_CASES:
        raise ValueError("Conformance requires a tuple of two through 64 cases")
    checked = tuple(ConformanceCase.model_validate(case) for case in cases)
    if capability.requires_known_receipt:
        raise ValueError("Untimed conformance cannot invent a known feedback receipt")
    if len({case.request.logical_identity for case in checked}) != len(checked):
        raise ValueError("Conformance case logical decision identities must be distinct")
    for case in checked:
        compatibility = validate_composition(case.problem, capability)
        if not compatibility.compatible:
            raise ValueError(f"Incompatible conformance case: {compatibility.reasons}")
    return checked


def _matched_decision[StateT](
    components: tuple[ConformanceConsumer[StateT], ...], case: ConformanceCase, capability: Capability
) -> Decision:
    """Require identical valid decisions from continuous and restored paths."""
    outputs = tuple(Decision.model_validate(component.decide(case.request)) for component in components)
    for output in outputs:
        if output.request.to_json() != case.request.to_json():
            raise ValueError("Conformance consumer changed the supplied request")
        if output.sampling.status not in case.problem.sampling_statuses:
            raise ValueError("Conformance decision uses undeclared sampling evidence")
        validate_decision(output, capability)
    if len({output.to_json() for output in outputs}) != 1:
        raise ValueError("Conformance restored/continued decisions differ")
    return outputs[0]


def _observed_batch(case: ConformanceCase, decision: Decision, index: int, capability: Capability) -> FeedbackBatch:
    """Bind an unchanged actual decision and expose only admitted oracle rows."""
    executed = ExecutedDecision(
        schema_version=1,
        proposal=decision,
        action=decision.action,
        mapping="unchanged",
        sampling_disposition="preserved",
        sampling=decision.sampling,
    )
    if case.problem.feedback_kind == "full_panel":
        observed = FullPanel(rows=case.outcomes)
    else:
        if not isinstance(decision.action, ChoiceAction):
            raise ValueError("Chosen conformance requires an actual choice action")
        observed = ChosenFeedback(row=next(row for row in case.outcomes if row.member == decision.action.chosen))
    batch = FeedbackBatch(
        schema_version=1,
        decision_fingerprint=executed.fingerprint,
        feedback_id=f"conformance-{index}",
        revision_id="root",
        supersedes=None,
        horizon=None,
        feedback=observed,
    )
    validate_feedback(executed, batch, capability)
    return batch


def _restore[StateT](target: ConformanceConsumer[StateT], source: ConformanceConsumer[StateT]) -> None:
    """Copy a host-owned snapshot and preserve a clear failure boundary."""
    state = source.export_state()
    try:
        snapshot = deepcopy(state)
    except Exception as error:
        raise ValueError("Conformance state snapshot must support deepcopy") from error
    target.restore_state(snapshot)


def check_decision_contracts[StateT](
    factory: Callable[[], ConformanceConsumer[StateT]],
    capability: Capability,
    cases: tuple[ConformanceCase, ...],
) -> ConformanceReport:
    """Check direct calls, declared coverage and state-restored continuations.

    Parameters
    ----------
    factory : callable
        Creates independent, fresh host components implementing ``Decider``,
        ``Updater`` and ``Stateful`` structurally. Called ``len(cases) + 2``
        times. Return values of ``update`` are deliberately unconstrained.
    capability : Capability
        Exact implementation profile, including explicit units and bounds.
    cases : tuple of ConformanceCase
        Two through 64 independent caller-owned oracle cases, in execution
        order, with distinct logical decision identities. This limited helper
        supports immediate untimed full-panel and chosen feedback only.

    Returns
    -------
    ConformanceReport
        Validated baseline decisions and delivered feedback. All three paths
        produced byte-identical decisions at each step: continuous execution,
        a restored path continued through updates, and a fresh checkpoint restore.

    Raises
    ------
    ValueError
        Invalid/incompatible cases, duplicate logical identities, reused factory
        objects, uncopyable snapshots, request changes, divergent restored
        decisions, or invalid decisions/feedback. User component errors propagate.
    AttributeError
        A host component omits one of the required structural entry points.

    Notes
    -----
    Calls host code and mutates only the created components through their public
    entry points. State snapshots are explicitly deep-copied; immutable host
    snapshots are recommended. Opaque state objects are never compared by
    equality. The complete oracle remains inside this helper: a chosen consumer
    sees only its actual chosen row. No scheduler, registry, RunSpec, CLI,
    transport, filesystem, clock, storage or installed algorithm is consulted.
    A successful report verifies structural consistency on these cases only;
    it establishes no performance, convergence, unbiasedness or efficacy claim.
    """
    capability = Capability.model_validate(capability)
    checked = _checked_cases(capability, cases)

    instances: list[ConformanceConsumer[StateT]] = []

    def fresh() -> ConformanceConsumer[StateT]:
        """Retain factory results so identity reuse cannot conceal shared state."""
        component = factory()
        if any(component is previous for previous in instances):
            raise ValueError("Conformance factory must create independent fresh objects")
        instances.append(component)
        return component

    baseline, continued = fresh(), fresh()
    _restore(continued, baseline)
    decisions: list[Decision] = []
    feedback: list[FeedbackBatch] = []
    for index, case in enumerate(checked):
        reconstructed = fresh()
        _restore(reconstructed, baseline)
        decisions.append(_matched_decision((baseline, continued, reconstructed), case, capability))
        batch = _observed_batch(case, decisions[-1], index, capability)
        for component in (baseline, continued, reconstructed):
            component.update(batch)
        feedback.append(batch)
    return ConformanceReport(decisions=tuple(decisions), feedback=tuple(feedback))
