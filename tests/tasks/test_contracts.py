"""Independent authored task wires and principal contract boundary regressions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from coleman.core.base import CoreContract
from coleman.core.identity import Reference
from coleman.results.domain import DomainRecord, Measurement, ResultProvenance
from coleman.tasks._base import MAXIMUM_BYTES
from coleman.tasks.contracts import (
    OwnedCommitProof,
    TaskJobRecord,
    TaskLimits,
    TaskPlan,
)
from coleman.tasks.identity import (
    CodecDeclaration,
    DistributionRoot,
    ExecutorImplementationIdentity,
    ExecutorRecipe,
    HostBootstrapProfile,
)
from coleman.tasks.protocols import ValueCodec, decode_value, encode_value

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks" / "v1"
READERS = (
    ("recipe.json", ExecutorRecipe),
    ("implementation.json", ExecutorImplementationIdentity),
    ("bootstrap.json", HostBootstrapProfile),
    ("plan.json", TaskPlan),
    ("job.json", TaskJobRecord),
    ("owned-proof.json", OwnedCommitProof),
)


def _raw(name="plan.json"):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _plan():
    return TaskPlan.from_json((FIXTURES / "plan.json").read_bytes())


def _changed(value, **updates):
    return type(value).model_validate(value.model_copy(update=updates))


def _reference(name):
    return Reference(id=name, version="1")


@pytest.mark.parametrize(("name", "reader"), READERS)
def test_independent_full_wire_golden_and_exact_fingerprint(name, reader):
    content = (FIXTURES / name).read_text(encoding="utf-8").removesuffix("\n")
    original = reader.from_json(content)
    hand = _raw("hand.json")
    expected = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert original.fingerprint == expected
    assert expected == hand["fingerprints"][name]
    assert original.to_json() == content
    assert reader.model_validate_json(content) == original
    assert reader.model_validate_json(bytearray(content.encode("utf-8"))) == original


@pytest.mark.parametrize(("name", "reader"), READERS)
@pytest.mark.parametrize("edit", ("future", "missing", "unknown", "foreign_namespace", "coerced_version"))
def test_new_namespace_readers_reject_future_missing_unknown_and_coerced_tags(name, reader, edit):
    value = _raw(name)
    if edit == "future":
        value["schema_version"] = 2
    elif edit == "missing":
        value.pop("schema_version")
    elif edit == "unknown":
        value["unrecognized"] = True
    elif edit == "foreign_namespace":
        value["contract"] = "coleman-workflows-v1"
    else:
        value["schema_version"] = True
    body = json.dumps(value)
    with pytest.raises(ValueError):
        reader.from_json(body)


def test_versioned_reader_rejects_duplicate_keys_and_validation_relaxation():
    body = (FIXTURES / "plan.json").read_text(encoding="utf-8")
    duplicate = body.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    with pytest.raises(ValueError, match="Duplicate"):
        TaskPlan.from_json(duplicate)
    with pytest.raises(ValueError, match="relax"):
        TaskPlan.model_validate_json(body, strict=False)
    with pytest.raises(ValueError, match="forbid"):
        TaskPlan.model_validate_json(body, extra="ignore")


@pytest.mark.parametrize("payload_kind", (str, bytes, bytearray))
def test_public_task_readers_bound_bytes_before_decoding(payload_kind):
    text = " " * (MAXIMUM_BYTES + 1)
    payload = text if payload_kind is str else payload_kind(text.encode("utf-8"))
    with pytest.raises(ValueError, match="byte|maximum_bytes"):
        TaskPlan.model_validate_json(payload)


def test_public_reader_and_unchecked_model_guard_depth_before_nested_legacy_validation():
    body = '{"unused":' + "[" * 65 + "0" + "]" * 65 + "}"
    with pytest.raises(ValueError, match="depth"):
        TaskPlan.from_json(body)
    plan = _plan()
    nested = "value"
    for _ in range(80):
        nested = [nested]
    runtime = plan.implementation.runtime.model_copy(update={"machine": nested})
    implementation = plan.implementation.model_copy(update={"runtime": runtime})
    unchecked = plan.model_copy(update={"implementation": implementation})
    with pytest.raises(ValueError, match="depth"):
        TaskPlan.model_validate(unchecked)


def test_hand_work_budget_seed_zero_null_and_negative_zero_are_distinct():
    plan = _plan()
    hand = _raw("hand.json")
    work = sum(item.target.position for item in plan.coordinates) * plan.limits.max_attempts
    assert work == 20
    assert work == hand["worst_case_work"]
    assert plan.coordinates[0].seed == 0
    assert plan.coordinates[1].seed is None
    assert plan.protocol is None
    assert plan.configuration.body == '{"eta":-0.0}'
    changed = _changed(plan.configuration, body='{"eta":0.0}')
    other = _changed(plan, configuration=changed)
    assert other.fingerprint != plan.fingerprint


@pytest.mark.parametrize("seed", (-1, 2**64, True, 1.0))
def test_seed_admits_uint64_only(seed):
    coordinate = _plan().coordinates[0]
    with pytest.raises(ValueError):
        _changed(coordinate, seed=seed)


def test_protocol_and_seed_require_explicit_not_applicable_instead_of_omission():
    raw = _raw()
    raw.pop("protocol")
    with pytest.raises(ValueError, match="protocol"):
        TaskPlan.model_validate(raw)
    coordinate = _raw()["coordinates"][1]
    coordinate.pop("seed")
    with pytest.raises(ValueError, match="seed"):
        TaskPlan.model_validate({**_raw(), "coordinates": [coordinate]})


def test_plan_rejects_budget_overflow_duplicate_ids_foreign_target_and_profile():
    plan = _plan()
    small_limits = _changed(plan.limits, max_work_units=19)
    with pytest.raises(ValueError, match="work"):
        _changed(plan, limits=small_limits)
    repeated = (plan.coordinates[0], plan.coordinates[0])
    with pytest.raises(ValueError, match="unique"):
        _changed(plan, coordinates=repeated)
    first = plan.coordinates[0]
    wrong_target = first.target.model_copy(update={"coordinate": _reference("foreign")})
    with pytest.raises(ValueError, match="same reference"):
        _changed(first, target=wrong_target)
    wrong_profile = first.target.profile.model_copy(update={"reference": _reference("different-units")})
    target = _changed(first.target, profile=wrong_profile)
    coordinate = _changed(first, target=target)
    with pytest.raises(ValueError, match="one exact declared frontier profile"):
        _changed(plan, coordinates=(coordinate, plan.coordinates[1]))


@pytest.mark.parametrize("role", ("configuration", "inputs", "coordinate"))
def test_plan_rejects_wrong_concrete_codec_role_before_runtime(role):
    plan = _plan()
    wrong = plan.implementation.recipe.codec_for("final_result")
    if role == "coordinate":
        body = _changed(plan.coordinates[0].body, codec=wrong)
        first = _changed(plan.coordinates[0], body=body)
        updates = {"coordinates": (first, plan.coordinates[1])}
    else:
        updates = {role: _changed(getattr(plan, role), codec=wrong)}
    with pytest.raises(ValueError, match="codec"):
        _changed(plan, **updates)


@pytest.mark.parametrize("field", ("max_coordinates", "max_work_units", "max_attempts", "max_invocations"))
@pytest.mark.parametrize("value", (0, True, 1.5))
def test_operational_integer_limits_never_coerce_or_accept_zero(field, value):
    with pytest.raises(ValueError):
        TaskLimits(**{field: value})


@pytest.mark.parametrize("seconds", (0.0, -1.0, 3600.0001, float("inf"), float("nan"), True))
def test_walltime_units_finite_positive_seconds(seconds):
    with pytest.raises(ValueError):
        TaskLimits(attempt_seconds=seconds)


def test_encoded_body_remains_exact_but_checks_utf8_depth_duplicates_and_finite_json():
    value = _plan().configuration
    assert _changed(value, body='{"optional":null,"x":-0.0}').body == '{"optional":null,"x":-0.0}'
    for invalid in ('{"x":1,"x":2}', '{"x":NaN}', "{", "[" * 65 + "0" + "]" * 65):
        with pytest.raises(ValueError):
            _changed(value, body=invalid)
    small = _changed(value.codec, maximum_bytes=4)
    with pytest.raises(ValueError, match="maximum_bytes|bytes"):
        _changed(value, codec=small)
    with pytest.raises(ValueError, match="unicode"):
        _changed(value, body='"\ud800"')


@pytest.mark.parametrize("path", ("", ".", "..", "/absolute", "a/../b", "a//b", "a/./b", "C:/x", "a\\b"))
def test_inventory_paths_are_bounded_portable_owned_relatives(path):
    item = _plan().implementation.files[0]
    with pytest.raises(ValueError):
        _changed(item, path=path)


def test_inventory_rejects_duplicate_unknown_owner_missing_root_and_changed_runtime_version():
    identity = _plan().implementation
    repeated = (*identity.files, identity.files[0])
    with pytest.raises(ValueError, match="unique"):
        _changed(identity, files=repeated)
    unknown = _changed(identity.files[0], distribution="foreign")
    with pytest.raises(ValueError, match="known"):
        _changed(identity, files=(unknown, identity.files[1]))
    with pytest.raises(ValueError, match="Every captured distribution"):
        _changed(identity, files=(identity.files[0],))
    runtime_item = identity.runtime.distributions[0].model_copy(update={"version": "future"})
    runtime = identity.runtime.model_copy(update={"distributions": (runtime_item,)})
    with pytest.raises(ValueError, match="runtime closure"):
        _changed(identity, runtime=runtime)


def test_inventory_requires_every_dependency_file_metadata_and_exact_selected_extras():
    identity = _plan().implementation
    dependency = _changed(identity.distributions[0], name="extra-dependency")
    with pytest.raises(ValueError, match="Every captured distribution"):
        _changed(identity, distributions=(*identity.distributions, dependency))
    coleman = next(item for item in identity.distributions if item.name == "coleman")
    mismatch = _changed(coleman, extras=("ci",))
    inventory = tuple(mismatch if item.name == "coleman" else item for item in identity.distributions)
    with pytest.raises(ValueError, match="selected runtime generation"):
        _changed(identity, distributions=inventory)


def test_declared_inventory_total_bytes_has_a_real_coupled_bound():
    identity = _plan().implementation
    first = identity.files[0]
    large = tuple(_changed(first, path=f"owned-{i}.bin", size_bytes=1024**3) for i in range(9))
    files = (*large, identity.files[1])
    with pytest.raises(ValueError, match="8 GiB"):
        _changed(identity, files=files)


def test_set_catalogues_canonicalize_but_trajectory_order_remains_semantic():
    plan = _plan()
    identity = plan.implementation
    reordered = _changed(
        identity, distributions=tuple(reversed(identity.distributions)), files=tuple(reversed(identity.files))
    )
    assert reordered.fingerprint == identity.fingerprint
    other = _changed(plan, coordinates=tuple(reversed(plan.coordinates)))
    assert other.fingerprint != plan.fingerprint
    root = DistributionRoot(name="provider", extras=("z", "a"))
    assert root.extras == ("a", "z")
    with pytest.raises(ValueError, match="unique"):
        DistributionRoot(name="provider", extras=("a", "a"))


def test_recipe_requires_complete_roles_roots_and_explicit_local_state_codecs():
    recipe = _plan().implementation.recipe
    incomplete = tuple(item for item in recipe.codecs if item.role != "progress_proof")
    with pytest.raises(ValueError, match="state and progress-proof"):
        _changed(recipe, codecs=incomplete)
    repeated = (*recipe.codecs[:6], recipe.codecs[0])
    with pytest.raises(ValueError, match="unique complete"):
        _changed(recipe, codecs=repeated)
    with pytest.raises(ValueError, match="COLEMAN and provider"):
        _changed(recipe, roots=(recipe.roots[0],))
    stateless_codecs = tuple(item for item in recipe.codecs if item.role not in {"state", "progress_proof"})
    stateless = _changed(recipe, effects="none", codecs=stateless_codecs)
    assert stateless.codec_for("state") is None
    assert stateless.codec_for("progress_proof") is None


def test_small_bootstrap_binds_capture_without_embedding_its_file_inventory():
    plan = _plan()
    profile = HostBootstrapProfile.from_json((FIXTURES / "bootstrap.json").read_bytes())
    plan.validate_bootstrap(profile)
    assert "files" not in profile.to_json()
    assert len(profile.to_json().encode("utf-8")) < 16 * 1024
    foreign = _changed(profile, reference=_reference("unapproved-profile"))
    with pytest.raises(ValueError, match="bootstrap"):
        plan.validate_bootstrap(foreign)
    approval = profile.executors[0]
    changed_binding = approval.implementation.model_copy(update={"definition_fingerprint": "sha256:" + "f" * 64})
    wrong = _changed(profile, executors=(_changed(approval, implementation=changed_binding),))
    rebound = _changed(plan, bootstrap=wrong.binding)
    with pytest.raises(ValueError, match="operator-approved"):
        rebound.validate_bootstrap(wrong)
    with pytest.raises(ValueError, match="unique recipe"):
        _changed(profile, executors=(approval, approval))


def test_small_bootstrap_rejects_argv_oversize_instead_of_claiming_all_count_maxima_fit():
    profile = _raw("bootstrap.json")
    recipe = profile["executors"][0]["recipe"]
    recipe["lineage"]["source"] = {"id": "x" * 17000, "version": "1"}
    body = json.dumps(profile)
    with pytest.raises(ValueError, match="maximum_bytes"):
        HostBootstrapProfile.from_json(body)


def test_job_plan_join_and_complete_last_are_not_inferred_from_a_directory():
    plan = _plan()
    job = TaskJobRecord.from_json((FIXTURES / "job.json").read_bytes())
    job.validate_for(plan)
    with pytest.raises(ValueError, match="every coordinate"):
        _changed(job, state="completed")
    with pytest.raises(ValueError, match="Only completed"):
        _changed(job, result_digest="sha256:" + "f" * 64)
    completed = _changed(job, state="completed", completed_coordinates=2, result_digest="sha256:" + "f" * 64)
    assert completed.completed_coordinates == completed.total_coordinates
    with pytest.raises(ValueError, match="nonnegative nondecreasing"):
        _changed(job, changed=-1.0)
    foreign = _changed(job, plan_id="f" * 64)
    with pytest.raises(ValueError, match="retained plan"):
        foreign.validate_for(plan)
    frontier = _changed(job.frontier, position=3)
    too_far = _changed(job, frontier=frontier)
    with pytest.raises(ValueError, match="frontier"):
        too_far.validate_for(plan)
    missing = _changed(job, current_coordinate="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="not in"):
        missing.validate_for(plan)


def test_job_rejects_foreign_counts_stages_and_operational_counters():
    plan = _plan()
    job = TaskJobRecord.from_json((FIXTURES / "job.json").read_bytes())
    total = _changed(job, total_coordinates=3)
    with pytest.raises(ValueError, match="catalogue"):
        total.validate_for(plan)
    stage = _changed(job, stage=_reference("wrong-stage"))
    with pytest.raises(ValueError, match="plan and stage"):
        stage.validate_for(plan)
    over_attempts = _changed(job, attempts=9)
    with pytest.raises(ValueError, match="operational caps"):
        over_attempts.validate_for(plan)
    with pytest.raises(ValueError, match="exact current coordinate"):
        _changed(job, current_coordinate=None)


def test_owned_proof_requires_actual_receipt_join_and_matching_current_ack():
    plan = _plan()
    proof = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    proof.validate_for(plan, plan.coordinates[0])
    raw = _raw("owned-proof.json")
    raw.pop("receipt")
    with pytest.raises(ValueError, match="receipt"):
        OwnedCommitProof.model_validate(raw)
    head = proof.observed_head.model_copy(update={"manifest_fingerprint": "sha256:" + "f" * 64})
    with pytest.raises(ValueError, match="observed current head"):
        _changed(proof, observed_head=head)
    intent = proof.request.intent.model_copy(update={"content_fingerprint": "sha256:" + "f" * 64})
    request = proof.request.model_copy(update={"intent": intent})
    with pytest.raises(ValueError, match="exact original intent"):
        _changed(proof, request=request)
    wrong = _changed(proof, plan_fingerprint="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="complete retained plan"):
        wrong.validate_for(plan, plan.coordinates[0])


def test_owned_proof_rejects_wrong_coordinate_frontier_runtime_and_owner_binding():
    plan = _plan()
    proof = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    second = _changed(proof, coordinate_fingerprint=plan.coordinates[1].fingerprint)
    with pytest.raises(ValueError, match="declared coordinate"):
        second.validate_for(plan, plan.coordinates[1])
    runtime = type(proof.compatibility.runtime).model_validate(
        {**proof.compatibility.runtime.model_dump(), "machine": "foreign-machine"}
    )
    compatibility = _changed(proof.compatibility, runtime=runtime)
    changed = _changed(proof, compatibility=compatibility)
    with pytest.raises(ValueError, match="runtime differs"):
        changed.validate_for(plan, plan.coordinates[0])
    owner = proof.compatibility.owner_implementations[0].model_copy(update={"reference": _reference("foreign-owner")})
    compatibility = _changed(proof.compatibility, owner_implementations=(owner,))
    changed = _changed(proof, compatibility=compatibility)
    with pytest.raises(ValueError, match="owner implementation"):
        changed.validate_for(plan, plan.coordinates[0])


def test_owned_proof_rejects_false_virgin_and_foreign_predecessor_joins():
    proof = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    virgin = _changed(proof.request, expected=None)
    with pytest.raises(ValueError, match="initialization generation zero"):
        _changed(proof, request=virgin)
    prior = proof.request.expected
    assert prior is not None
    foreign = prior.model_copy(update={"source_epoch": _reference("foreign-epoch")})
    request = _changed(proof.request, expected=foreign)
    with pytest.raises(ValueError, match="consecutive predecessor"):
        _changed(proof, request=request)


def test_owned_proof_explicit_zero_generation_initialization_is_admitted():
    proof = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    intent = _changed(proof.request.intent, generation=0, kind="initialize")
    head = _changed(proof.observed_head, generation=0, intent_fingerprint=intent.fingerprint)
    request = _changed(proof.request, expected=None, intent=intent)
    receipt = _changed(proof.receipt, accepted_head=head, current_head=head)
    frontier = _changed(proof.frontier, position=0)
    initialized = _changed(proof, request=request, receipt=receipt, observed_head=head, frontier=frontier)
    assert initialized.observed_head.generation == 0
    assert initialized.frontier.position == 0


def test_public_consistency_methods_revalidate_unchecked_receivers():
    plan = _plan()
    profile = HostBootstrapProfile.from_json((FIXTURES / "bootstrap.json").read_bytes())
    unchecked_plan = plan.model_copy(update={"coordinates": ()})
    with pytest.raises(ValueError, match="coordinates"):
        unchecked_plan.validate_bootstrap(profile)
    job = TaskJobRecord.from_json((FIXTURES / "job.json").read_bytes())
    unchecked_job = job.model_copy(update={"changed": -1.0})
    with pytest.raises(ValueError, match="nonnegative nondecreasing"):
        unchecked_job.validate_for(plan)
    proof = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    unchecked_proof = proof.model_copy(update={"receipt": None})
    with pytest.raises(ValueError, match="receipt"):
        unchecked_proof.validate_for(plan, plan.coordinates[0])


def test_task_namespace_cold_import_needs_no_native_application_storage_or_scientific_services():
    root = str(Path(__file__).resolve().parents[2])
    paths = [root, sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    program = """
import importlib.abc
import sys
sys.path[:0] = PATHS
blocked = ("coleman.application", "coleman.workflows", "coleman.evaluation", "coleman.registry",
           "coleman.runner", "coleman.cli", "coleman.mcp", "sqlite3", "numpy", "pandas", "polars", "pyarrow")
def forbidden(name):
    return any(name == item or name.startswith(item + ".") for item in blocked)
assert not any(forbidden(name) for name in sys.modules)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError("unexpected optional provider: " + fullname)
sys.meta_path.insert(0, Guard())
import coleman.tasks.contracts
import coleman.tasks.identity
import coleman.tasks.protocols
assert not any(forbidden(name) for name in sys.modules)
print("task cold boundary passed")
""".replace("PATHS", repr(paths))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "task cold boundary passed"


def test_storage_generation_is_not_logical_work_and_retained_current_head_is_preserved():
    proof = OwnedCommitProof.from_json((FIXTURES / "owned-proof.json").read_bytes())
    current = proof.observed_head.model_copy(
        update={
            "generation": 3,
            "manifest_fingerprint": "sha256:" + "b" * 64,
            "intent_fingerprint": "sha256:" + "c" * 64,
        }
    )
    receipt = proof.receipt.model_copy(update={"status": "retained", "current_head": current})
    frontier = _changed(proof.frontier, position=2)
    retained = _changed(proof, receipt=receipt, observed_head=current, frontier=frontier)
    assert proof.receipt.accepted_head.generation == 2
    assert proof.frontier.position == 1
    assert retained.receipt.accepted_head.generation == 2
    assert retained.observed_head.generation == 3
    assert retained.frontier.position == 2
    # Pure joins cannot derive this semantic work counter from a head digest.
    # The runtime ProgressVerifier must decode actual current owner state.
    unverified = _changed(retained, frontier=proof.frontier)
    assert unverified.frontier.position == 1
    foreign = current.model_copy(update={"definition_fingerprint": "sha256:" + "e" * 64})
    receipt = receipt.model_copy(update={"current_head": foreign})
    with pytest.raises(ValueError, match="changed its collection"):
        _changed(retained, receipt=receipt, observed_head=foreign)


@dataclass(frozen=True)
class _LegacyCodec:
    declaration: CodecDeclaration

    def validate(self, value: object) -> DomainRecord:
        return DomainRecord.model_validate(value)

    def encode(self, value: DomainRecord) -> str:
        return self.validate(value).to_json()

    def decode(self, body: str) -> DomainRecord:
        return DomainRecord.from_json(body)


def test_structural_value_codec_accepts_real_legacy_domain_record_without_core_inheritance():
    record = DomainRecord(
        schema_version=1,
        namespace="constructed.tasks.codec",
        run_id="run",
        execution_id="execution",
        decision_id="decision",
        data_fingerprint="sha256:" + "a" * 64,
        available_at=datetime(2026, 11, 8, tzinfo=UTC),
        provenance=ResultProvenance(
            producer="authored-codec-test",
            producer_version="1",
            spec_fingerprint="sha256:" + "b" * 64,
            evaluation_fingerprint="sha256:" + "c" * 64,
        ),
        measurements=(
            Measurement(
                name="loss", value=-0.0, unit="1", meaning="signed-zero-test", direction="minimize", status="observed"
            ),
        ),
    )
    assert not isinstance(record, CoreContract)
    codec: ValueCodec[DomainRecord] = _LegacyCodec(_plan().configuration.codec)
    content = codec.encode(record)
    recovered = codec.decode(content)
    assert recovered.to_json() == content
    assert '"value":-0.0' in content
    assert recovered.measurements == record.measurements
    encoded = encode_value(codec, record)
    assert encoded.body == content
    assert decode_value(codec, encoded).to_json() == content
    wrong = _changed(encoded, codec=_plan().inputs.codec)
    with pytest.raises(ValueError, match="selected concrete codec"):
        decode_value(codec, wrong)
    noncanonical = _changed(encoded, body=" " + content)
    with pytest.raises(ValueError, match="exact canonical"):
        decode_value(codec, noncanonical)
