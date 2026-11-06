import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from coleman.core.feedback import ChosenFeedback
from coleman.core.identity import Reference
from coleman.core.primitives import fingerprint
from coleman.interchange.codec import MAXIMUM_WIRE_BYTES, decode_request, decode_response, wire_fingerprint
from coleman.interchange.contracts import (
    CoverageCall,
    CoverageReply,
    DecisionReply,
    FeedbackCall,
    FeedbackReply,
    InterchangeRequest,
    InterchangeResponse,
    StateBinding,
    UnavailableReply,
    accept_response,
)
from coleman.selection import SearchLimits, solve_coverage

GOLDEN = Path(__file__).parents[1] / "fixtures/interchange/v1"


def _chosen(request: InterchangeRequest) -> ChosenFeedback:
    assert isinstance(request.input, FeedbackCall)
    assert isinstance(request.input.batch.feedback, ChosenFeedback)
    return request.input.batch.feedback


def test_all_versioned_goldens_are_canonical_and_byte_pinned():
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert len(manifest["fixtures"]) == 11
    for name, specification in manifest["fixtures"].items():
        wire = (GOLDEN / name).read_bytes()
        model = InterchangeRequest if specification["model"] == "InterchangeRequest" else InterchangeResponse
        actual = model.from_json(wire)
        assert actual.to_json().encode() == wire.removesuffix(b"\n"), name
        assert actual.fingerprint == specification["canonical_fingerprint"], name
        assert wire_fingerprint(wire) == specification["wire_fingerprint"], name
        assert wire_fingerprint(wire) == "sha256:" + hashlib.sha256(wire).hexdigest(), name
    response = decode_response((GOLDEN / "coverage-response.json").read_bytes())
    assert isinstance(response.reply, CoverageReply)
    assert response.reply.result.cost is not None
    assert response.reply.result.objective is not None
    assert response.reply.result.selected_ids == ("A", "C")
    assert str(response.reply.result.cost.value) == "3"
    assert str(response.reply.result.objective.value) == "1"


@pytest.mark.parametrize("filename", ["coverage-request.json", "coverage-response.json"])
@pytest.mark.parametrize("version", [None, True, 0, 2, "1", 1.0])
def test_unknown_missing_and_coerced_outer_versions_reject(filename, version):
    payload = json.loads((GOLDEN / filename).read_text())
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    decoder = decode_request if filename.endswith("request.json") else decode_response
    with pytest.raises(ValueError):
        decoder(json.dumps(payload))


@pytest.mark.parametrize("filename", ["coverage-request.json", "learning_only-request.json", "feedback-request.json"])
def test_future_nested_versions_unknown_operations_and_fields_reject(filename):
    payload = json.loads((GOLDEN / filename).read_text())
    nested = {"coverage": "problem", "decision": "request", "feedback": "batch"}[payload["input"]["kind"]]
    payload["input"][nested]["schema_version"] = 2
    with pytest.raises(ValueError):
        decode_request(json.dumps(payload))
    payload = json.loads((GOLDEN / filename).read_text())
    payload["input"]["kind"] = "execute_python"
    with pytest.raises(ValueError):
        decode_request(json.dumps(payload))
    payload = json.loads((GOLDEN / filename).read_text())
    payload["unexpected"] = "import os"
    with pytest.raises(ValueError):
        decode_request(json.dumps(payload))


@pytest.mark.parametrize("digest_field", ["request_fingerprint", "input_fingerprint", "reply_fingerprint"])
def test_all_response_digests_recomputed_even_for_unchecked_instances(coverage_pair, digest_field):
    _, response = coverage_pair
    broken = response.model_copy(update={digest_field: fingerprint({"other": digest_field})})
    with pytest.raises(ValueError, match="digest mismatch"):
        InterchangeResponse.model_validate(broken)
    with pytest.raises(ValueError):
        broken.to_json()


@pytest.mark.parametrize("field", ["attempt_id", "exchange_id", "task", "created_at", "input"])
def test_acceptance_compares_actual_retained_request_not_echoed_digests(coverage_pair, field):
    request, response = coverage_pair
    replacements = {
        "attempt_id": "retry-2",
        "exchange_id": "another-call",
        "task": Reference(id="another-task", version="v1"),
        "created_at": "2026-01-01T09:59:00Z",
        "input": request.input.model_copy(update={"limits": SearchLimits(max_work=1, deadline_seconds=None)}),
    }
    changed = InterchangeRequest.model_validate(request.model_copy(update={field: replacements[field]}))
    with pytest.raises(ValueError, match="another request"):
        accept_response(changed, response, received_at=datetime.fromisoformat("2026-01-01T10:02:00Z"))


@pytest.mark.parametrize("produced", ["2026-01-01T09:59:59Z", "2026-01-01T10:05:01Z", None])
def test_production_outside_request_bounds_or_unknown_reject(coverage_pair, produced):
    _, response = coverage_pair
    with pytest.raises(ValueError, match="production time"):
        InterchangeResponse.model_validate(response.model_copy(update={"produced_at": produced}))


@pytest.mark.parametrize("receipt", [None, "2026-01-01T10:00:59Z", "2026-01-01T10:05:01Z", "2026-01-01T10:02:00"])
def test_unknown_premature_stale_or_naive_receipt_reject(coverage_pair, receipt):
    request, response = coverage_pair
    with pytest.raises(ValueError):
        accept_response(request, response, received_at=receipt)


def test_inclusive_clock_boundaries_are_not_ambient_time(coverage_pair, respond):
    request, response = coverage_pair
    for instant in (request.created_at, request.expires_at):
        produced = respond(request, response.reply, produced_at=instant)
        assert accept_response(request, produced, received_at=instant) == produced
    assert accept_response(request, response, received_at=request.expires_at) == response
    with pytest.raises(TypeError):
        accept_response(request, cast(Any, None), received_at=request.expires_at)


@pytest.mark.parametrize("clock", ["created_at", "expires_at"])
def test_partial_clocks_reject(coverage_pair, clock):
    request, _ = coverage_pair
    with pytest.raises(ValueError, match="both"):
        InterchangeRequest.model_validate(request.model_copy(update={clock: None}))


def test_untimed_exchange_requires_explicit_null_receipt_and_production():
    response = decode_response((GOLDEN / "learning_only-response.json").read_bytes())
    assert accept_response(response.request, response, received_at=None) == response
    with pytest.raises(ValueError, match="null receipt"):
        accept_response(response.request, response, received_at=datetime.fromisoformat("2026-01-01T10:00:00Z"))
    with pytest.raises(ValueError, match="production clock"):
        InterchangeResponse.model_validate(response.model_copy(update={"produced_at": "2026-01-01T10:00:00Z"}))


@pytest.mark.parametrize("change", ["problem", "method", "limits"])
def test_actual_coverage_output_bound_to_problem_method_and_limits(coverage_pair, respond, change):
    request, response = coverage_pair
    call = request.input
    if change == "problem":
        other = call.problem.model_copy(update={"data_fingerprint": fingerprint({"another": "dataset"})})
        result = solve_coverage(other, method=call.method, limits=call.limits)
    elif change == "method":
        result = solve_coverage(call.problem, method="greedy_density_v1", limits=call.limits)
    else:
        result = solve_coverage(
            call.problem, method=call.method, limits=SearchLimits(max_work=1, deadline_seconds=None)
        )
    with pytest.raises(ValueError, match="problem/method/limits"):
        respond(request, CoverageReply(result=result), produced_at=response.produced_at)


def test_wrong_units_do_not_become_a_new_numeric_interpretation():
    payload = json.loads((GOLDEN / "coverage-request.json").read_text())
    payload["input"]["problem"]["constraints"]["budget"]["semantics"]["unit"] = "ms"
    with pytest.raises(ValueError):
        decode_request(json.dumps(payload))


@pytest.mark.parametrize("field", ["task", "implementation", "state"])
def test_inner_task_actual_implementation_and_state_codecs_are_not_inferred(field):
    response = decode_response((GOLDEN / "mab_only-response.json").read_bytes())
    assert response.state is not None
    if field == "task":
        broken = response.request.model_copy(update={"task": Reference(id="other", version="v1")})
        with pytest.raises(ValueError, match="task differs"):
            InterchangeRequest.model_validate(broken)
    elif field == "implementation":
        with pytest.raises(ValueError, match="implementation"):
            InterchangeResponse.model_validate(
                response.model_copy(update={"implementation": Reference(id="other", version="v1")})
            )
    else:
        state = StateBinding(codec=Reference(id="other-codec", version="v1"), digest=response.state.digest)
        with pytest.raises(ValueError, match="state codec"):
            InterchangeResponse.model_validate(response.model_copy(update={"state": state}))


def test_feedback_binding_does_not_validate_or_execute_a_consumer_policy():
    request = decode_request((GOLDEN / "feedback-request.json").read_bytes())
    assert isinstance(request.input, FeedbackCall)
    broken_batch = request.input.batch.model_copy(update={"decision_fingerprint": fingerprint({"wrong": "execution"})})
    with pytest.raises(ValueError, match="exact executed decision"):
        InterchangeRequest.model_validate(
            request.model_copy(update={"input": request.input.model_copy(update={"batch": broken_batch})})
        )


def test_ignored_and_unavailable_outputs_cannot_claim_changed_state(respond):
    request = decode_request((GOLDEN / "feedback-request.json").read_bytes())
    assert request.state is not None
    changed = StateBinding(codec=request.state.codec, digest=fingerprint({"changed": "state"}))
    for reply in (
        FeedbackReply(disposition="ignored"),
        UnavailableReply(reason="failed", detail="bounded-host-failure"),
    ):
        result = respond(request, reply, state=request.state)
        assert result.state == request.state
        with pytest.raises(ValueError, match="changed state"):
            respond(request, reply, state=changed)


def test_decision_reply_must_retain_actual_request_and_declared_implementation(respond):
    response = decode_response((GOLDEN / "mab_only-response.json").read_bytes())
    assert isinstance(response.reply, DecisionReply)
    decision = response.reply.decision
    for replacement in (
        {"request": decision.request.model_copy(update={"decision_id": "other"})},
        {"implementation": Reference(id="other", version="v1")},
    ):
        reply = response.reply.model_copy(update={"decision": decision.model_copy(update=replacement)})
        with pytest.raises(ValueError, match="mathematical request|implementation"):
            respond(response.request, reply, state=response.state)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e9999", "1e-9999", "true"])
def test_core_numeric_guards_survive_outer_codec(token):
    wire = (GOLDEN / "feedback-request.json").read_text()
    # The existing bounded loss declaration is binary64, unlike exact decimal solver quantities.
    assert '"minimum":0.0' in wire
    wire = wire.replace('"minimum":0.0', '"minimum":' + token, 1)
    with pytest.raises(ValueError):
        decode_request(wire)


def test_duplicate_keys_unknown_fields_and_non_executable_state():
    wire = (GOLDEN / "mab_only-request.json").read_text()
    with pytest.raises(ValueError):
        decode_request(wire.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1))
    payload = json.loads(wire)
    payload["state"]["pickle"] = "execute-me"
    with pytest.raises(ValueError):
        decode_request(json.dumps(payload))


def test_wire_bytes_and_typed_canonical_identity_are_distinct(coverage_pair):
    request, response = coverage_pair
    plain = response.to_json()
    spaced = json.dumps(json.loads(plain), indent=2) + "\n"
    assert wire_fingerprint(plain) != wire_fingerprint(spaced)
    assert decode_response(plain).fingerprint == decode_response(spaced).fingerprint
    assert decode_response(spaced, expected_wire_fingerprint=wire_fingerprint(spaced)) == response
    with pytest.raises(ValueError, match="wire digest"):
        decode_response(spaced, expected_wire_fingerprint=wire_fingerprint(plain))
    assert decode_request(request.to_json()).attempt_identity == request.attempt_identity


@pytest.mark.parametrize(
    "payload",
    [b"", b"\xff", b"[" * 65 + b"]" * 65, b"]", b" " * (MAXIMUM_WIRE_BYTES + 1)],
    ids=["empty", "invalid-utf8", "too-deep", "unopened-close", "oversized"],
)
def test_wire_acquisition_is_bounded_before_parsing(payload):
    with pytest.raises(ValueError):
        decode_request(payload)


def test_wire_str_prebound_and_quoted_braces_do_not_count_as_depth():
    with pytest.raises(ValueError, match="before UTF-8"):
        decode_request("x" * (MAXIMUM_WIRE_BYTES + 1))
    with pytest.raises(ValueError):
        decode_request("é" * (MAXIMUM_WIRE_BYTES // 2 + 1))
    for payload in ([], bytearray(b"{}"), object()):
        with pytest.raises(TypeError):
            decode_request(cast(Any, payload))
    assert wire_fingerprint('"' + "[" * 100 + '\\"' + "]" * 100 + '"').startswith("sha256:")


@pytest.mark.parametrize("exchange", ["a", "0", "build-1", "recipe_a", "b" * 32])
def test_content_and_attempt_keys_are_deterministic_but_separate(exchange):
    request = decode_request((GOLDEN / "coverage-request.json").read_bytes())
    changed = InterchangeRequest.model_validate(request.model_copy(update={"exchange_id": exchange}))
    reordered = dict(reversed(list(changed.model_dump(mode="json").items())))
    decoded = decode_request(json.dumps(reordered))
    assert decoded.fingerprint == changed.fingerprint
    assert decoded.attempt_identity == changed.attempt_identity
    timed = InterchangeRequest.model_validate(changed.model_copy(update={"created_at": "2026-01-01T09:00:00Z"}))
    assert timed.fingerprint != changed.fingerprint
    assert timed.attempt_identity == changed.attempt_identity
    retry = InterchangeRequest.model_validate(changed.model_copy(update={"attempt_id": "another-attempt"}))
    assert retry.logical_identity == changed.logical_identity
    assert retry.attempt_identity != changed.attempt_identity


def test_zero_negative_and_unavailable_inner_values_keep_explicit_semantics():
    payload = json.loads((GOLDEN / "feedback-request.json").read_text())
    value = payload["input"]["batch"]["feedback"]["row"]["values"][0]
    assert value["value"] == 0.0
    unavailable = copy.deepcopy(payload)
    target = unavailable["input"]["batch"]["feedback"]["row"]["values"][0]
    target.update(value=None, status="unavailable")
    absent = decode_request(json.dumps(unavailable))
    assert _chosen(absent).row.values[0].value is None
    assert _chosen(absent).row.values[0].status == "unavailable"
    negative = copy.deepcopy(payload)
    target = negative["input"]["batch"]["feedback"]["row"]["values"][0]
    target["spec"].update(minimum=-1.0, role="diagnostic")
    target.update(value=-0.5, transform=None)
    assert _chosen(decode_request(json.dumps(negative))).row.values[0].value == -0.5


def test_stateless_stateful_and_explicit_solver_identity_are_independent(coverage_pair, respond):
    request, response = coverage_pair
    opaque = StateBinding(
        codec=Reference(id="never-imported-host-codec", version="v9"), digest=fingerprint({"host": "state"})
    )
    with pytest.raises(ValueError, match="Stateless exchange"):
        respond(request, response.reply, produced_at=response.produced_at, state=opaque)
    for replacement in ({"state": opaque}, {"implementation": Reference(id="branch_bound_v1", version="coverage-v2")}):
        with pytest.raises(ValueError, match="exact stateless solver"):
            InterchangeRequest.model_validate(request.model_copy(update=replacement))
    stateful = decode_response((GOLDEN / "mab_only-response.json").read_bytes())
    with pytest.raises(ValueError, match="state codec"):
        respond(stateful.request, stateful.reply, state=None)
    with pytest.raises(ValueError, match="operation differs"):
        respond(stateful.request, FeedbackReply(disposition="applied"), state=stateful.state)
    # A declared new codec does not execute it, import it, or recover supplier state.
    changed = InterchangeRequest.model_validate(stateful.request.model_copy(update={"state": opaque}))
    assert changed.state is not None
    assert changed.state.codec.version == "v9"
    with pytest.raises(ValueError, match="expires before creation"):
        InterchangeRequest.model_validate(request.model_copy(update={"expires_at": "2026-01-01T09:59:00Z"}))


def test_canonical_signed_zero_search_input_difference_is_not_python_equality(coverage_pair, respond):
    request, response = coverage_pair
    positive = SearchLimits(deadline_seconds=0.0)
    negative = SearchLimits(deadline_seconds=-0.0)
    assert positive == negative
    changed = InterchangeRequest.model_validate(
        request.model_copy(
            update={"input": CoverageCall(problem=request.input.problem, method=request.input.method, limits=negative)}
        )
    )
    result = solve_coverage(request.input.problem, method=request.input.method, limits=positive)
    with pytest.raises(ValueError, match="problem/method/limits"):
        respond(changed, CoverageReply(result=result), produced_at=response.produced_at)


def test_wire_guard_does_not_erase_signed_zero_feedback_values():
    payload = json.loads((GOLDEN / "feedback-request.json").read_text())
    zero = decode_request(json.dumps(payload))
    payload["input"]["batch"]["feedback"]["row"]["values"][0]["value"] = -0.0
    signed = decode_request(json.dumps(payload))
    assert _chosen(zero).row.values[0].value == _chosen(signed).row.values[0].value
    assert zero.fingerprint != signed.fingerprint
    assert zero.attempt_identity == signed.attempt_identity
