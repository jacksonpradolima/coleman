"""Closed typed request/reply codecs for an optional isolated local host."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from coleman.core.base import CoreContract, CoreImmutable
from coleman.core.decisions import Decision, DecisionRequest, ExecutedDecision
from coleman.core.feedback import FeedbackBatch
from coleman.core.identity import Reference
from coleman.core.primitives import Fingerprint, Identifier, canonical_json, fingerprint
from coleman.core.temporal import Timestamp
from coleman.selection.contracts import CoverageProblem
from coleman.selection.resources import Method, SearchLimits
from coleman.selection.results import CoverageResult

ExchangeId = Annotated[Identifier, Field(max_length=128)]


def _same(left: CoreImmutable | CoreContract, right: CoreImmutable | CoreContract) -> bool:
    """Compare canonical content, retaining signed zero and exact decimal strings."""
    return canonical_json(left.model_dump(mode="json")) == canonical_json(right.model_dump(mode="json"))


class StateBinding(CoreImmutable):
    """Name a non-executable host state codec and its supplied content digest.

    Attributes
    ----------
    codec : Reference
        Exact host-defined state representation/version; never imported by name.
    digest : str
        Supplied SHA-256 of the host's declared state representation.

    Notes
    -----
    No state bytes, pickle, restoration or authenticity claim is supplied. A
    digest binds a declaration; the host separately owns state validation and
    recovery. Unknown codec names acquire no executable support by being stored.
    """

    codec: Reference
    digest: Fingerprint


class DecisionCall(CoreImmutable):
    """Exchange a typed direct decision request without running a learner.

    Attributes
    ----------
    kind : {'decision'}
        Operation/codec discriminator.
    request : DecisionRequest
        Complete action space, mathematical units and any declared source clocks.
        Capability negotiation and execution remain explicit host operations.
    """

    kind: Literal["decision"] = "decision"
    request: DecisionRequest


class FeedbackCall(CoreImmutable):
    """Exchange feedback bound to an actual executed decision, without updating.

    Attributes
    ----------
    kind : {'feedback'}
        Operation/codec discriminator.
    executed : ExecutedDecision
        Exact actual action and proposal/selection evidence.
    batch : FeedbackBatch
        Explicit coverage, values, origin, horizon, clocks and revision identity.

    Raises
    ------
    ValueError
        The batch names another executed decision. Consumer-specific capability,
        maturity/application and lineage checks still belong to the actual host.
    """

    kind: Literal["feedback"] = "feedback"
    executed: ExecutedDecision
    batch: FeedbackBatch

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Reject a feedback batch attached to another actual execution."""
        if self.batch.decision_fingerprint != self.executed.fingerprint:
            raise ValueError("Interchange feedback must bind the exact executed decision")
        return self


class CoverageCall(CoreImmutable):
    """Exchange an existing finite coverage problem and explicit search limits.

    Attributes
    ----------
    kind : {'coverage'}
        Existing finite-solver operation, not a generic optimizer invocation.
    problem : CoverageProblem
        Complete versioned costs, weighted coverage, units and feasible scope.
    method : {'greedy_density_v1', 'branch_bound_v1'}
        Exact existing solver method; no dynamic callable lookup occurs.
    limits : SearchLimits
        Existing bounded work/memory/deadline envelope in its original units.
    """

    kind: Literal["coverage"] = "coverage"
    problem: CoverageProblem
    method: Method
    limits: SearchLimits


InterchangeInput = Annotated[DecisionCall | FeedbackCall | CoverageCall, Field(discriminator="kind")]


class InterchangeRequest(CoreContract):
    """Retain an optional local call envelope independently of direct core inputs.

    Attributes
    ----------
    schema_version : int
        Required integer 1; unknown versions/fields reject without migration.
    task : Reference
        Versioned host task; decision/feedback inputs must name this exact task.
    exchange_id, attempt_id : str
        Stable logical operation and explicit physical attempt, at most 128 characters.
        Input/implementation/time changes under one attempt are conflicts.
    implementation : Reference
        Expected producing implementation/version. Coverage uses method ID and
        existing solver version ``coverage-v1``.
    state : StateBinding or None
        Required explicit input-state attestation, or null for a stateless call.
        State-codec support is supplied by the host, never by this envelope.
    created_at, expires_at : datetime or None
        Required explicit UTC-normalized creation/deadline pair, or both null for
        untimed exchange. Equal instants are permitted; no ambient clock is read.
    input : DecisionCall or FeedbackCall or CoverageCall
        Closed typed payload retaining existing core/finite-solver semantics.

    Raises
    ------
    ValueError
        Partial/reversed clocks, task mismatch, unsupported solver identity/state
        or malformed nested input. No algorithm, I/O or action is invoked.
    """

    task: Reference
    exchange_id: ExchangeId
    attempt_id: ExchangeId
    implementation: Reference
    state: StateBinding | None
    created_at: Timestamp | None
    expires_at: Timestamp | None
    input: InterchangeInput

    @model_validator(mode="after")
    def _binding(self) -> Self:
        """Keep host task, explicit clock mode and existing solver identity coherent."""
        if (self.created_at is None) != (self.expires_at is None):
            raise ValueError("Interchange clocks must both be known or both null")
        if self.created_at is not None and self.expires_at is not None and self.created_at > self.expires_at:
            raise ValueError("Interchange request expires before creation")
        if isinstance(self.input, CoverageCall):
            expected = Reference(id=self.input.method, version="coverage-v1")
            if self.implementation != expected or self.state is not None:
                raise ValueError("Coverage exchange requires its exact stateless solver identity")
        else:
            request = (
                self.input.request if isinstance(self.input, DecisionCall) else self.input.executed.proposal.request
            )
            if request.task != self.task:
                raise ValueError("Interchange task differs from the mathematical request")
        return self

    @property
    def logical_identity(self) -> str:
        """Return logical task/operation identity excluding input and attempt bytes.

        Returns
        -------
        str
            Canonical task/version, exchange ID and operation SHA-256. Different
            content for one logical attempt must conflict at publication.
        """
        return fingerprint(
            {"task": self.task.model_dump(mode="json"), "exchange_id": self.exchange_id, "operation": self.input.kind}
        )

    @property
    def attempt_identity(self) -> str:
        """Return a deterministic logical-attempt key without encoding caller paths.

        Returns
        -------
        str
            SHA-256 over logical identity and attempt ID, excluding payload bytes.
            No filesystem, randomness or global registration is involved.
        """
        return fingerprint({"logical_identity": self.logical_identity, "attempt_id": self.attempt_id})

    @property
    def input_fingerprint(self) -> str:
        """Return typed canonical input identity, distinct from exact wire bytes.

        Returns
        -------
        str
            SHA-256 including the complete input operation and nested contracts.
        """
        return fingerprint(self.input.model_dump(mode="json"))


class DecisionReply(CoreImmutable):
    """Retain a typed proposal from the expected implementation.

    Attributes
    ----------
    kind : {'decision'}
        Reply operation; never executes the proposed action.
    decision : Decision
        Existing immutable proposal and action-bound sampling declaration.
    """

    kind: Literal["decision"] = "decision"
    decision: Decision


class FeedbackReply(CoreImmutable):
    """Retain a host-declared update disposition, without proving its execution.

    Attributes
    ----------
    kind : {'feedback'}
        Reply operation.
    disposition : {'applied', 'ignored'}
        Host assertion; ignored means no update and requires unchanged state.
        Output state binding is recorded by the outer response.
        This is neither exactly-once acceptance nor a recovery/effect receipt.
    """

    kind: Literal["feedback"] = "feedback"
    disposition: Literal["applied", "ignored"]


class CoverageReply(CoreImmutable):
    """Retain the existing self-validating finite result, including valid limits.

    Attributes
    ----------
    kind : {'coverage'}
        Reply operation.
    result : CoverageResult
        Existing result with independently checked source, feasibility and proof.
        Limited/infeasible results are valid and never become fake optima.
    """

    kind: Literal["coverage"] = "coverage"
    result: CoverageResult


class UnavailableReply(CoreImmutable):
    """Keep unsupported/failed work explicit without inventing a numeric result.

    Attributes
    ----------
    kind : {'unavailable'}
        Absence of a completed mathematical output.
    reason : {'unsupported', 'failed'}
        Declared producer disposition, not an automatic exception translation.
    detail : str
        Bounded stable explanation; no traceback, credentials or executable code.
    """

    kind: Literal["unavailable"] = "unavailable"
    reason: Literal["unsupported", "failed"]
    detail: Annotated[Identifier, Field(max_length=512)]


InterchangeReply = Annotated[
    DecisionReply | FeedbackReply | CoverageReply | UnavailableReply, Field(discriminator="kind")
]


class InterchangeResponse(CoreContract):
    """Retain a complete typed response and recomputable producing-call bindings.

    Attributes
    ----------
    request : InterchangeRequest
        Complete producing call; accepting it for another caller still requires
        ``accept_response`` against that caller's actual retained request.
    request_fingerprint, input_fingerprint, reply_fingerprint : str
        Revalidated canonical request/input/reply digests; none authenticates a host.
    implementation : Reference
        Actual declared producer, equal to the requested implementation/version.
    produced_at : datetime or None
        Explicit production instant within timed request bounds, otherwise null.
    state : StateBinding or None
        Output-state attestation using the input codec, or null when no input
        state was declared. Unsupported work must retain the input binding.
    reply : DecisionReply or FeedbackReply or CoverageReply or UnavailableReply
        Complete typed output, or explicit absent/failed/unsupported status.

    Raises
    ------
    ValueError
        Digest/operation/implementation/input/state/clock mismatch. Finite result
        problem, method and limits must match the actual producing call exactly.

    Notes
    -----
    Reads verify declared arithmetic/bindings, not authenticity or host effects.
    Construction performs no learning, solver invocation, I/O or clock read.
    """

    request: InterchangeRequest
    request_fingerprint: Fingerprint
    input_fingerprint: Fingerprint
    reply_fingerprint: Fingerprint
    implementation: Reference
    produced_at: Timestamp | None
    state: StateBinding | None
    reply: InterchangeReply

    @model_validator(mode="after")
    def _bindings(self) -> Self:
        """Validate independent binding groups before a caller can accept output."""
        if (
            self.request_fingerprint != self.request.fingerprint
            or self.input_fingerprint != self.request.input_fingerprint
        ):
            raise ValueError("Interchange response request/input digest mismatch")
        if self.reply_fingerprint != fingerprint(self.reply.model_dump(mode="json")):
            raise ValueError("Interchange response reply digest mismatch")
        if self.implementation != self.request.implementation:
            raise ValueError("Interchange response implementation/version mismatch")
        self._clocks()
        self._state()
        if isinstance(self.reply, UnavailableReply):
            if self.state != self.request.state:
                raise ValueError("Unavailable work cannot claim a changed state")
            return self
        if self.reply.kind != self.request.input.kind:
            raise ValueError("Interchange response operation differs from the request")
        self._output()
        return self

    def _clocks(self) -> None:
        """Check explicit production time without inferring current delivery time."""
        start, end = self.request.created_at, self.request.expires_at
        if start is None:
            if self.produced_at is not None:
                raise ValueError("Untimed exchange cannot invent a production clock")
        elif self.produced_at is None or end is None or not start <= self.produced_at <= end:
            raise ValueError("Interchange production time is outside request bounds")

    def _state(self) -> None:
        """Preserve exact state codec/version while treating hashes as attestations."""
        before = self.request.state
        if before is None:
            if self.state is not None:
                raise ValueError("Stateless exchange cannot acquire an undeclared state codec")
        elif self.state is None or self.state.codec != before.codec:
            raise ValueError("Interchange output state codec/version mismatch")

    def _output(self) -> None:
        """Bind typed outputs to actual mathematical input, implementation and limits."""
        call = self.request.input
        if isinstance(call, DecisionCall) and isinstance(self.reply, DecisionReply):
            _decision_output(call, self.reply, self.request.implementation)
        elif isinstance(call, FeedbackCall) and isinstance(self.reply, FeedbackReply):
            if self.reply.disposition == "ignored" and self.state != self.request.state:
                raise ValueError("Ignored feedback cannot claim a changed state")
        elif isinstance(call, CoverageCall) and isinstance(self.reply, CoverageReply):
            _coverage_output(call, self.reply, self.request.implementation)


def _decision_output(call: DecisionCall, reply: DecisionReply, implementation: Reference) -> None:
    """Check complete mathematical request and actual producing implementation."""
    if not _same(reply.decision.request, call.request):
        raise ValueError("Interchange decision changes the mathematical request")
    if reply.decision.implementation != implementation:
        raise ValueError("Interchange decision implementation/version mismatch")


def _coverage_output(call: CoverageCall, reply: CoverageReply, implementation: Reference) -> None:
    """Check actual self-validating finite result against every supplied search input."""
    result = reply.result
    if (
        result.problem.to_json() != call.problem.to_json()
        or result.method != call.method
        or canonical_json(result.limits.model_dump(mode="json")) != canonical_json(call.limits.model_dump(mode="json"))
        or result.solver_version != implementation.version
    ):
        raise ValueError("Interchange coverage result differs from problem/method/limits")


def accept_response(
    request: InterchangeRequest, response: InterchangeResponse, *, received_at: Timestamp | None
) -> InterchangeResponse:
    """Validate a reply against the retained current request and explicit receipt.

    Parameters
    ----------
    request : InterchangeRequest
        Actual caller-retained task/input/attempt/implementation/state contract.
    response : InterchangeResponse
        Complete typed response, revalidated including unchecked nested instances.
    received_at : datetime or None
        Supplied local receipt within the timed deadline and no earlier than
        production. Required known for timed exchange; null for untimed exchange.

    Returns
    -------
    InterchangeResponse
        Revalidated exact response. Inputs remain unchanged; nothing executes.

    Raises
    ------
    ValueError
        Missing/stale output, wrong request/attempt/input/implementation/state
        binding, unknown receipt or incompatible explicit clock mode.
    TypeError
        Caller passes an untyped request or response.

    Notes
    -----
    This checks consistency rather than authentication, learner execution or
    causal identification. The caller owns receipt facts and any state recovery.
    """
    if not isinstance(request, InterchangeRequest) or not isinstance(response, InterchangeResponse):
        raise TypeError("Interchange acceptance requires typed request and response")
    request = InterchangeRequest.model_validate(request)
    response = InterchangeResponse.model_validate(response)
    if response.request.to_json() != request.to_json():
        raise ValueError("Interchange response belongs to another request/input/attempt")
    if request.created_at is None:
        if received_at is not None:
            raise ValueError("Untimed interchange requires an explicit null receipt")
    else:
        if received_at is None:
            raise ValueError("Timed interchange requires a known receipt")
        receipt = TypeAdapter(Timestamp).validate_python(received_at)
        if (
            response.produced_at is None
            or request.expires_at is None
            or not response.produced_at <= receipt <= request.expires_at
        ):
            raise ValueError("Interchange response is stale or received before production")
    return response
