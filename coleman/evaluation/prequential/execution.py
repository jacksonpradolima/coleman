"""Compose unchanged coefficient replay with a separately owned offline action.

This optional boundary executes no external effect and owns no scheduler, store,
predictor or inference engine. Use existing local-state staging to acknowledge
the complete replay and domain owners together.
"""

from __future__ import annotations

from typing import Protocol, Self

from pydantic import model_validator

from coleman.core.actions import MixtureAction
from coleman.core.base import CoreContract
from coleman.core.primitives import Fingerprint
from coleman.core.protocols import Stateful
from coleman.evaluation.design._wire import guard_graph
from coleman.evaluation.prequential.contracts import DomainExecutionReceipt, ExecutionKey, PrequentialContract
from coleman.objectives.bindings import DefinitionBinding
from coleman.replay.contracts import ReplayDecisionRecord, ReplayOpportunity
from coleman.replay.engine import CausalReplay


class DomainStepPlan(PrequentialContract):
    """Declare one logical domain opportunity before a coefficient call.

    Attributes
    ----------
    key : ExecutionKey
        Exact candidate/member/seed and distinct domain task/opportunity.
    adapter : DefinitionBinding
        Frozen offline mapping definition, not executable serialized code.
    base_configuration_fingerprint, data_fingerprint : str
        BASE treatment and interpreted source identities. The outer comparison
        binds these declarations to its actual selected configurations and input.

    Notes
    -----
    Strict v1 declaration only; construction performs no lookup or state change.
    Identity does not authenticate an arbitrary callback or its supplied source.
    """

    key: ExecutionKey
    adapter: DefinitionBinding
    base_configuration_fingerprint: Fingerprint
    data_fingerprint: Fingerprint


class DomainExecutionContext(PrequentialContract):
    """Supply only an earlier coefficient producer and the actual prior state.

    Attributes
    ----------
    plan : DomainStepPlan
        Exact predeclared domain law, source and logical cell.
    coefficient : ReplayDecisionRecord
        Complete immutable unchanged offline mixture execution and eligible view.
        Its original expert-loss producer remains unchanged.
    prior_domain_state_fingerprint : str
        Complete state exported immediately before a new domain action. It is
        never the digest of an after-state containing its own receipt.

    Raises
    ------
    ValueError
        Untimed or non-mixture coefficient input. This initial composition does
        not admit partial-feedback sampling or replacement of that same action.
    """

    plan: DomainStepPlan
    coefficient: ReplayDecisionRecord
    prior_domain_state_fingerprint: Fingerprint

    @model_validator(mode="after")
    def _producer(self) -> Self:
        """Keep the admitted profile explicit without changing replay v1."""
        if not isinstance(self.coefficient.executed.action, MixtureAction):
            raise ValueError("Prequential domain composition requires an unchanged mixture producer")
        if self.coefficient.opportunity.request.timing is None:
            raise ValueError("Prequential domain composition requires explicit coefficient clocks")
        return self


class DomainExecutor[S: CoreContract](Stateful[S], Protocol):
    """Describe a complete independent domain, predictor and accounting owner.

    Notes
    -----
    Implementations retain their actual pending observations, processed keys,
    forecast state and accounting in S. Exports are pure complete snapshots;
    factories and state completeness remain host obligations. Calls are serial.
    No runtime registration, external execution or storage is implied.
    """

    def execution(self, key: ExecutionKey, /) -> DomainExecutionReceipt | None:
        """Inspect a previously retained logical action without changing state.

        Parameters
        ----------
        key : ExecutionKey
            Exact versioned logical opportunity.

        Returns
        -------
        DomainExecutionReceipt or None
            Original complete receipt or explicit absence. Conflicting versions
            must not be silently selected or treated as a fresh opportunity.
        """
        ...

    def execute(self, context: DomainExecutionContext, /) -> DomainExecutionReceipt:
        """Create and retain one distinct offline finite domain action.

        Parameters
        ----------
        context : DomainExecutionContext
            Actual earlier producer, frozen bindings and complete prior digest.

        Returns
        -------
        DomainExecutionReceipt
            Original proposal, feasible executed choice and explicit intervention.

        Notes
        -----
        May change this owner's complete state. It must retain the receipt for
        execution(key), use only eligible inputs and perform bounded local work.
        Later observations and predictor updates have their own explicit clocks;
        no future outcome belongs in this decision-time receipt.
        """
        ...


class PrequentialStep(PrequentialContract):
    """Retain both distinct actions without attributing domain feedback to weights.

    Attributes
    ----------
    coefficient : ReplayDecisionRecord
        Actual original unchanged mixture action and full-information producer.
    domain : DomainExecutionReceipt
        Separate offline domain proposal, feasibility intervention and execution.

    Notes
    -----
    A strict v1 exchange, not a store acknowledgement or physical execution proof.
    Outputs, charges and original expert panels remain separately owned records.
    """

    coefficient: ReplayDecisionRecord
    domain: DomainExecutionReceipt

    @model_validator(mode="after")
    def _join(self) -> Self:
        """Bind the complete earlier producer and its declared logical calendar."""
        _require_producer(self.coefficient, self.domain)
        return self


def _require_producer(coefficient: ReplayDecisionRecord, receipt: DomainExecutionReceipt) -> None:
    """Check actual producer content and exact common cutoff/target clocks."""
    if not isinstance(coefficient.executed.action, MixtureAction):
        raise ValueError("Domain composition requires an actual mixture execution")
    if (
        receipt.coefficient_record_fingerprint != coefficient.fingerprint
        or receipt.coefficient_execution_fingerprint != coefficient.executed.fingerprint
    ):
        raise ValueError("Domain receipt differs from its complete earlier coefficient producer")
    source = coefficient.opportunity.request.timing
    target = receipt.executed.proposal.request.timing
    if source is None or target is None:
        raise ValueError("Both coefficient and domain decisions require explicit clocks")
    if (
        target != source
        or receipt.executed.proposal.request.round != coefficient.opportunity.request.round
        or receipt.executed_at < coefficient.accepted_at
    ):
        raise ValueError("Offline domain composition requires the same declared decision and target calendar")


def _require_plan(plan: DomainStepPlan, receipt: DomainExecutionReceipt) -> None:
    """Reject changed source, BASE configuration, opportunity or adapter law."""
    if (
        receipt.key != plan.key
        or receipt.adapter != plan.adapter
        or receipt.base_configuration_fingerprint != plan.base_configuration_fingerprint
        or receipt.data_fingerprint != plan.data_fingerprint
    ):
        raise ValueError("Domain receipt differs from its complete predeclared step plan")


def _domain_fingerprint[D: CoreContract](domain: DomainExecutor[D]) -> str:
    """Bound and revalidate the complete prior snapshot before coefficient mutation."""
    snapshot = domain.export_state()
    if not isinstance(snapshot, CoreContract):
        raise TypeError("Domain owner must export a complete concrete CoreContract snapshot")
    state_type = type(snapshot)
    if state_type is CoreContract or state_type.__pydantic_generic_metadata__.get("parameters"):
        raise TypeError("Domain owner requires a concrete state type, not the base or an unbound generic")
    guard_graph(snapshot, maximum_bytes=16 * 1024**2, maximum_items=65536)
    return state_type.model_validate(snapshot).fingerprint


def execute_prequential_step[S: CoreContract, D: CoreContract](
    replay: CausalReplay[S],
    domain: DomainExecutor[D],
    opportunity: ReplayOpportunity,
    *,
    plan: DomainStepPlan,
) -> PrequentialStep:
    """Emit weights, then execute a distinct offline finite domain action.

    Parameters
    ----------
    replay : CausalReplay
        Actual independently owned causal engine, including RetainedReplay.
        Admit history/feedback and use advance for explicit eligible application.
    domain : DomainExecutor
        Separate complete owner, including any predictor/accounting state.
        Apply its eligible observations before deriving the opportunity inputs.
    opportunity : ReplayOpportunity
        Complete timed coefficient request built from the eligible source view.
        Existing replay drains permitted expert feedback before its decision.
    plan : DomainStepPlan
        Exact frozen domain mapping and logical opportunity bindings.

    Returns
    -------
    PrequentialStep
        Complete actual coefficient record and separately retained domain receipt.
        Exact retained retries invoke no new domain execution or learner decision.

    Raises
    ------
    ValueError, TypeError
        Wrong profile, clock/source/law/producer join, incomplete retention,
        stale/conflicting retry or a domain snapshot exceeding 16 MiB/65536 items.
    Exception
        Real host/codec failures propagate under their existing contracts.

    Notes
    -----
    Mutates the supplied owners in order. A later domain failure may follow an
    already emitted coefficient; this direct call is not an atomic transaction.
    For durable comparisons invoke it on fresh owners inside the existing
    LocalStateStore.commit_transition callback and publish both complete owners.
    Failed staging retains the old acknowledged head. External effects and
    arbitrary callback CPU/RAM are outside this bounded trusted offline profile.
    No net domain outcome is fed back as a complete expert-loss panel.
    """
    selected = DomainStepPlan.model_validate(plan)
    checked = ReplayOpportunity.model_validate(opportunity)
    retained = domain.execution(selected.key)
    if retained is not None:
        retained = DomainExecutionReceipt.model_validate(retained)
        _require_plan(selected, retained)
        # A receipt cannot turn a new/unrelated coefficient request into a retry.
        producers = tuple(
            row
            for row in replay.state.decisions
            if row.opportunity.request.logical_identity == checked.request.logical_identity
        )
        if len(producers) != 1 or producers[0].fingerprint != retained.coefficient_record_fingerprint:
            raise ValueError("Retained domain action requires its retained matching coefficient opportunity")
    prior = retained.prior_domain_state_fingerprint if retained is not None else _domain_fingerprint(domain)
    coefficient = replay.emit(replay.prepare(checked), execution="offline_identity_v1")
    if retained is not None:
        return PrequentialStep(schema_version=1, coefficient=coefficient, domain=retained)
    context = DomainExecutionContext(
        schema_version=1,
        plan=selected,
        coefficient=coefficient,
        prior_domain_state_fingerprint=prior,
    )
    receipt = DomainExecutionReceipt.model_validate(domain.execute(context))
    _require_plan(selected, receipt)
    _require_producer(coefficient, receipt)
    if receipt.prior_domain_state_fingerprint != context.prior_domain_state_fingerprint:
        raise ValueError("Domain receipt must bind the actual complete pre-execution state")
    actual = domain.execution(selected.key)
    if actual is None or DomainExecutionReceipt.model_validate(actual).to_json() != receipt.to_json():
        raise ValueError("Domain owner must retain the exact returned receipt in its complete state")
    return PrequentialStep(schema_version=1, coefficient=coefficient, domain=receipt)
